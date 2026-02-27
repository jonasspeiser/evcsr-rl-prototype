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

class NeverChargeAlgorithm(EvaluationAlgorithm):
    """
    Always returns 'do nothing' action
    """
    def predict(self, observation, deterministic):
        action = 0
        placeholder = "This is a placeholder, just to have the same Return signature as stable baseline's model.predict()"
        return action, placeholder
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
                battery_soc = vehicle_obs[0]
                station_distances = vehicle_obs[2:6]
                break
        else: # for...else
            logger.warning("No vehicle with an active charging request found in the observation.")
            return 0, placeholder # "do nothing"
        
        if battery_soc is None:
            raise UnboundLocalError(f"battery_soc is None for the vehicle with active_charging_request {active_vehicle}. This shouldn't be possible.")
        
        if battery_soc > 0.07:  # 0.07 ≈ 7,000 Wh = 25 km range at 240 Wh/km + ~16% headroom
            return 0, placeholder # "do nothing"
        
        # Unreachable stations are encoded as -1; same-edge stations as 0 (vehicle already past them).
        # Treat both as infinitely far so argmin only considers stations genuinely ahead.
        distances = np.array(station_distances, dtype=float)
        distances[distances <= 0] = np.inf
        if np.all(np.isinf(distances)):
            return 0, placeholder  # no reachable station, do nothing
        closest_station = int(np.argmin(distances))
        action = closest_station + 1
        return action, placeholder