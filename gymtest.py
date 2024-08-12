import gymnasium as gym
from gymnasium import spaces
import numpy as np
from circletest import Simulation
import logging

cs_id = "cs_0" # hardcoded for the toy use case. Will be part of the action in later versions.

class CircleEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



    def __init__(self, render_mode=None, log_level="info"):
        """
        Define self.observation_space and self.action_space
        """
        self.__set_logging_level(log_level)
        logging.debug("init")
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            gui = True
        else:
            gui = False
        self.simulation = Simulation(gui=gui)
        self.vehicles_to_spawn = 1
        self.simulation.add_vehicles(self.vehicles_to_spawn)
        self.simulation.step() # to spawn vehicles

        self.vehicle_ids = self.simulation.get_all_vehicle_ids()  
        logging.debug(f"vehicle_ids: {self.vehicle_ids}")

        # --- Define action space ---        
        # We have 2 actions for each vehicle: do nothing (0) and send charging (1)
        actions_per_vehicle = 2 
        # Dynamically create the action space for each vehicle
        # e.g. spaces.MultiDiscrete([2,2,2]) for 3 vehicles
        action_space_list = [actions_per_vehicle for vehicle in self.vehicle_ids]
        # MultiDiscrete action space because Stable Baselines doesn't support dict action spaces
        self.action_space = spaces.MultiDiscrete(action_space_list)


        # --- Define observation space ---
        # We have 2 types of observations: the current state of the battery and the current distance to the next charging station
        single_vehicle_observation_space = spaces.MultiDiscrete([100000, 2000]) # battery soc is in between 0 and 100000 Wh, distance to the next charging station is in between 0 and 2000 meters
        self.observation_space = spaces.Dict({
            str(vehicle_id): single_vehicle_observation_space for vehicle_id in self.vehicle_ids
        })

    def __get_observation(self):
        observation = dict()
        for vehicle_id in self.vehicle_ids:
            vehicle_state = self.state[vehicle_id]
            observation[vehicle_id] = np.array([vehicle_state["battery_soc"], vehicle_state["distance_to_next_cs"]], dtype=int)
        return observation
    
    def __get_info(self):
        return dict()

    def reset(self, seed=None, options=None): # Later: add possibility to set seed by passing it as an argument `env.reset(seed=<desired seed>)`
        """
        Returns: The observation of the initial state
        Reset the environment to initial state so that a new episode (independent of previous ones) may start
        """
        logging.debug("reset")
        super().reset(seed=seed) # needed for api compliance
        self.simulation.reset()
        self.simulation.add_vehicles(self.vehicles_to_spawn)
        self.simulation.step() # to spawn vehicles
        self.simulation.step()
        self.vehicle_ids = self.simulation.get_all_vehicle_ids()  
        logging.debug(f"vehicle_ids: {self.vehicle_ids}")
        self.state = self.simulation.get_state()
        observation = self.__get_observation()
        info = self.__get_info()
        logging.debug(f"reset observation: {observation}")
        return (observation, info)

    def __action_is_charge(self, vehicle_action):
        return vehicle_action == 1

    def __set_logging_level(self, log_level_str):
        # Map string to logging level
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        log_level = log_levels.get(log_level_str.lower(), logging.DEBUG)  # Default to DEBUG if not found
        logging.getLogger().setLevel(log_level)

    def __log_step_details(self, observation, reward, terminated, truncated):
        logging.debug(f"step observation: {observation}, step reward: {reward}")
        if terminated:
            logging.debug("step terminated")
            logging.debug(f"state: {self.state}")
            if destination_is_reached:
                logging.info("destination is reached")
            if battery_is_empty:
                logging.info("battery is empty")
        if truncated:
            logging.info("step truncated")
    
    def __process_vehicles(self, action, observation):
        reward = 0

        for index, vehicle_id in enumerate(observation):

            if self.__action_is_charge(action[index]):
                try:
                    self.simulation.reroute_for_charging(vehicle_id, cs_id)
                    logging.debug("REROUTED")
                except ValueError: # if the vehicle is past the charging station and rerouting doesn't work
                    logging.debug("Rerouting failed")
                    reward -= 1

            vehicle_state = self.state[vehicle_id]
            destination_is_reached = (vehicle_state["vehicle_destination"] == vehicle_state["vehicle_position"])
            battery_is_empty = (vehicle_state["battery_soc"] <= 0)
            reward_per_vehicle = -10 if battery_is_empty else 1 if destination_is_reached else 0
            reward += reward_per_vehicle

        return reward, destination_is_reached, battery_is_empty

    def step(self, action, log_level=None):
        """
        Returns: The next observation, the reward, done and optionally additional info
        """

        if log_level is not None:
            # Save the current logging level
            original_log_level = logging.getLogger().level
            # Set logging level dynamically
            self.__set_logging_level(log_level)

        self.state = self.simulation.get_state()
        observation = self.__get_observation()
        reward, destination_is_reached, battery_is_empty = self.__process_vehicles(action, observation)

        terminated = (destination_is_reached or battery_is_empty)
        truncated = not self.simulation.active_vehicles_exist()
        info = self.__get_info()

        self.simulation.step()
        self.__log_step_details(observation, reward, terminated, truncated)
        
        if log_level is not None:
            # Reset the logging level to its original state
            logging.getLogger().setLevel(original_log_level)
        
        return observation, reward, terminated, truncated, info

    
    def render(self, mode='human'):
        """
        Returns: None
        Show the current environment state e.g. the graphical window in 'CartPole-v1'
        This method must be implemented, but it is OK to have an empty implementation if rendering is not important
        """
        logging.debug("render")
        pass

    def close(self):
        """
        Returns: None
        This is optional. Used to cleanup all resources (threads, graphical windows, etc)
        """
        logging.debug("close")
        self.simulation.close()


if __name__ == "__main__":
    def test_env():
        from stable_baselines3.common.env_checker import check_env
        env = CircleEnv()
        check_env(env, skip_render_check=True)
        print("CHECKS PASSED")
        env.close()
    
    test_env()