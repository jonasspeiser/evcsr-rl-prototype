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
        
        for vehicle_id, vehicle_obs in observation.items():
            # find out which one is the active vehicle
            active_charging_request_flag = vehicle_obs[8]
            if active_charging_request_flag:
                active_vehicle = vehicle_id
                normalized_battery_soc = vehicle_obs[0]
                station_distances = vehicle_obs[2:6]
                break
        else: # for...else
            logger.warning("No vehicle with an active charging request found in the observation.")
            return 0, placeholder # "do nothing"
        
        if normalized_battery_soc is None:
            raise UnboundLocalError(f"battery_soc is None for the vehicle with active_charging_request {active_vehicle}. This shouldn't be possible.")
        
        if normalized_battery_soc > 0.15:  # 0.15 × 64 kWh = 9,600 Wh ≈ 40 km at 240 Wh/km
            # — enough headroom to bridge any 25 km station gap on straight_120km
            logger.info(f"greedy → do nothing for {active_vehicle} (normalized soc={normalized_battery_soc:.3f})")
            return 0, placeholder

        # Unreachable stations are encoded as -1; same-edge stations as 0 (vehicle already past them).
        # Stations beyond the destination are also useless. Treat all such cases as infinitely far
        # so argmin only considers stations genuinely reachable and ahead.
        distance_to_destination = vehicle_obs[1]
        distances = np.array(station_distances, dtype=float)
        distances[distances <= 0] = np.inf                       # unreachable or same-edge
        distances[distances > distance_to_destination] = np.inf  # station is past destination
        if np.all(np.isinf(distances)):
            logger.info(f"greedy → do nothing for {active_vehicle} (normalized soc={normalized_battery_soc:.3f}, no reachable station before destination)")
            return 0, placeholder  # no reachable station, do nothing
        closest_station = int(np.argmin(distances))
        action = closest_station + 1
        logger.info(f"greedy → cs_{closest_station} for {active_vehicle} (normalized soc={normalized_battery_soc:.3f})")
        return action, placeholder