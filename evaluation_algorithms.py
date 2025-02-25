import numpy as np

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
        for vehicle_obs in observation.values():
            # find out which one is the active vehicle
            active_charging_request_flag = vehicle_obs[7]
            if active_charging_request_flag:
                battery_soc = vehicle_obs[0]
                station_distances = vehicle_obs[1:5]
                break
        if battery_soc > 0.2:
            return 0, placeholder # "do nothing"
        
        closest_station = np.argmin(station_distances) # the index of the lowest distance
        action = closest_station + 1
        return action, placeholder