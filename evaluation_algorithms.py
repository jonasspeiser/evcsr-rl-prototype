import numpy as np

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
    For the vehicle with an active charging request, if battery SOC is above 0.2, do nothing. Otherwise, send the vehicle to the nearest charging station.
    """
    def predict(self, observation, deterministic):
        placeholder = "This is a placeholder, just to have the same Return signature as stable baseline's model.predict()"

        active = observation["active_vehicle"]   # shape (12,)
        normalized_battery_soc = active[0]
        distance_to_destination = active[1]
        station_distances = active[2:6]

        if normalized_battery_soc > 0.15:  # 0.15 × 64 kWh = 9,600 Wh ≈ 40 km at 240 Wh/km
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
        active = observation["active_vehicle"]
        soc = active[0]
        last_action = int(np.argmax(active[6:11]))

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

