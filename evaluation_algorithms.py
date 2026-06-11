import numpy as np
from vehicle import Vehicle

# configure logging
import logging
logger = logging.getLogger("rl.evaluation_algorithms")
# child logger of "rl"


class EvaluationAlgorithm():
    def __init__(self, environment):
        self.env = environment

    def predict(self, observation, deterministic):
        raise NotImplementedError

    def reset(self):
        """Called at the start of each episode. Override to reset per-episode state."""
        pass

class FixedActionAlgorithm(EvaluationAlgorithm):
    """
    Always returns the same fixed action.
    Use via algorithm names ACTION0, ACTION1, ACTION2, … in training_utils.
    """
    def __init__(self, environment, action: int):
        super().__init__(environment)
        self.action = action

    def predict(self, observation, deterministic):
        placeholder = "This is a placeholder, just to have the same Return signature as stable baseline's model.predict()"
        return self.action, placeholder
class RandomAlgorithm(EvaluationAlgorithm):
    """
    Select a random action.
    """
    def predict(self, observation, deterministic):
        action = self.env.action_space.sample() # select a random action
        placeholder = "This is a placeholder, just to have the same Return signature as stable baseline's model.predict()"
        return action, placeholder
    
class GreedyAlgorithm(EvaluationAlgorithm):
    """
    For the vehicle with an active charging request, if battery SOC is above 15kWh, do nothing. Otherwise, send the vehicle to the nearest charging station.
    """
    def predict(self, observation, deterministic):
        placeholder = "This is a placeholder, just to have the same Return signature as stable baseline's model.predict()"

        active_vehicle = observation["active_vehicle"]   # shape (12,)
        normalized_battery_soc = active_vehicle[0] # normalized by 100,000 to fit in [0, 1]
        distance_to_destination = active_vehicle[1]
        station_distances = active_vehicle[2:6]

        if normalized_battery_soc > 0.15:  # charges when remaining energy drops below 15,000 Wh (≈23% SoC with 64kWh battery, ≈62 km range at 240Wh/km)
            # — enough headroom to bridge any 25 km station gap on straight_120km
            logger.info(f"greedy → do nothing (normalized soc={normalized_battery_soc:.3f})")
            return 0, placeholder

        # Unreachable stations are encoded as -1; same-edge stations as 0 (vehicle already past them).
        # Stations beyond the destination are also useless. Treat all such cases as infinitely far
        # so argmin only considers stations genuinely reachable and ahead.
        distances = np.array(station_distances, dtype=float)
        distances[distances <= 0] = np.inf                       # unreachable or same-edge
        distances[distances > distance_to_destination] = np.inf  # station is past destination
        if np.all(np.isinf(distances)):
            logger.info(f"greedy → do nothing (normalized soc={normalized_battery_soc:.3f}, no reachable station before destination)")
            return 0, placeholder
        closest_station = int(np.argmin(distances))
        action = closest_station + 1
        logger.info(f"greedy → cs_{closest_station} (normalized soc={normalized_battery_soc:.3f})")
        return action, placeholder
    
class Perfect5VehAlgorithm(EvaluationAlgorithm):
    """
    Manually crafted heuristic that achieves perfect performance in the 5-vehicle same_route scenario.
    Only for evaluation, not a realistic baseline.
    """
    def __init__(self, environment):
        super().__init__(environment)
        self._sequence = [1, 2, 3, 4, 4, 0, 0, 4, 4]
        self.it = iter(self._sequence)

    def reset(self):
        self.it = iter(self._sequence)

    def predict(self, observation, deterministic):
        placeholder = "This is a placeholder, just to have the same Return signature as stable baseline's model.predict()"
        optimal_action = next(self.it, 0)  # default to do nothing after the sequence is exhausted
        logger.info(f"perfect → cs_{optimal_action}" if optimal_action > 0 else "perfect → do nothing")
        return optimal_action, placeholder
    
class Perfect20VehAlgorithm(EvaluationAlgorithm):
    """
    Manually crafted heuristic for the 20-vehicle same_route scenario.
    Distributes vehicles evenly across 4 stations (5 per station) using observation-based
    request classification:
      - Spawn request (last_action == 0): assign next station from round-robin sequence
      - Low battery request (last_action > 0, soc <= 0.2): re-recommend the same station
      - Post-charge request (last_action > 0, soc > 0.2): do nothing (action 0)
    Only for evaluation, not a realistic baseline.
    """
    LOW_BATTERY_THRESHOLD = 0.2

    def __init__(self, environment):
        super().__init__(environment)
        self._sequence = [1, 2, 3, 4] * 5  # 20 spawn assignments, 5 per station
        self.it = iter(self._sequence)

    def reset(self):
        self.it = iter(self._sequence)

    def predict(self, observation, deterministic):
        placeholder = "This is a placeholder, just to have the same Return signature as stable baseline's model.predict()"
        active_vehicle = observation["active_vehicle"]
        soc = active_vehicle[0]
        last_action = int(np.argmax(active_vehicle[6:11]))

        if last_action == 0:
            action = next(self.it)
            logger.info(f"perfect20 → spawn → cs_{action}")
        elif soc <= self.LOW_BATTERY_THRESHOLD:
            action = last_action
            logger.info(f"perfect20 → low battery → re-recommend cs_{action}")
        else:
            action = 0
            logger.info("perfect20 → post-charge → do nothing")

        return action, placeholder

class PerfectXVehAlgorithm(Perfect20VehAlgorithm):
    """
    Extension of Perfect20VehAlgorithm to support arbitrary vehicle counts in same_route scenario.
    Distributes vehicles evenly across 4 stations using the same observation-based request classification as Perfect20VehAlgorithm.
    Only for evaluation, not a realistic baseline.
    """
    def __init__(self, environment, n_vehicles):
        super().__init__(environment)
        assignments_per_station = (n_vehicles + 3) // 4  # round up to ensure enough assignments
        self._sequence = [1, 2, 3, 4] * assignments_per_station
        self.it = iter(self._sequence)


class BestGuessAlgorithm(EvaluationAlgorithm):
    """
    Routes to the least-congested reachable charging station when the remaining
    range is insufficient to reach the destination; otherwise does nothing.

    Decision logic per charging request:
      1. Query remaining range from the simulation (meters).
      2. If remaining_range >= dist_destination do nothing (action 0).
      3. Otherwise, build the reachable set: stations where dist_cs < remaining_range.
      4. From the reachable set, pick the station with fewest currently assigned live vehicles.
      5. If no reachable station exists do nothing (vehicle will likely go empty).

    Station distances are read from the observation (indices 2–5), which are sorted by cs_id,
    matching the sort order used in Vehicle.get_observation().
    Remaining range and station assignment counts are queried directly from the environment.
    """

    def predict(self, observation, deterministic):
        placeholder = "This is a placeholder, just to have the same Return signature as stable baseline's model.predict()"
        active_vehicle = observation["active_vehicle"]  # (12,)
        vehicle_id = self.env.active_charging_request_vehicle_id
        remaining_range = self.env.simulation.get_remaining_range(vehicle_id)
        if remaining_range is None:
            return 0, placeholder
        dist_norm = Vehicle.DISTANCE_NORMALIZATION_VALUE
        if dist_norm is None:
            return 0, placeholder
        
        # denormalize dist_destination
        dist_dest_norm = active_vehicle[1]
        dist_dest = dist_dest_norm * dist_norm if dist_dest_norm >= 0 else float('inf')

        if remaining_range >= dist_dest:
            logger.info(f"best_guess → do nothing (remaining_range={remaining_range:.0f}m >= dist_dest={dist_dest:.0f}m)")
            return 0, placeholder

        # Station distances in the observation are sorted by cs_id (see Vehicle.get_observation).
        cs_ids = sorted(self.env.simulation.get_all_charging_station_ids())
        cs_distances = [active_vehicle[2 + station_index] * dist_norm for station_index in range(len(cs_ids))]

        assignment_counts = {
            cs_id: sum(1 for v in self.env.vehicles.values() if v.target_cs_id == cs_id and v.is_online)
            for cs_id in cs_ids
        }

        reachable = []
        for cs_id, dist_cs in zip(cs_ids, cs_distances):
            if dist_cs < 0:  # observation encodes unreachable stations as -1
                continue
            if dist_cs < remaining_range:
                reachable.append(cs_id)

        if not reachable:
            logger.info(f"best_guess → do nothing (no reachable station, remaining_range={remaining_range:.0f}m)")
            return 0, placeholder

        best_cs_id = min(reachable, key=lambda cs_id: assignment_counts[cs_id])
        action = cs_ids.index(best_cs_id) + 1
        logger.info(f"best_guess → cs_{action} (remaining_range={remaining_range:.0f}m, assigned={assignment_counts[best_cs_id]})")
        return action, placeholder

