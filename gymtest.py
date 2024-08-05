import gymnasium as gym
from gymnasium import spaces
import numpy as np
from circletest import Simulation

class CircleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        """
        Define self.observation_space and self.action_space
        """
        print("init")
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            print("gui")
            self.simulation = Simulation(gui=True)
        else:
            print("no gui")
            self.simulation = Simulation(5)
        self.simulation.add_vehicles()
        self.simulation.step() # to spawn vehicles

        self.vehicle_ids = self.simulation.get_all_vehicle_ids()  # This method needs to be implemented in the Simulation class
        print("vehicle_ids: ", self.vehicle_ids)
        # We have 2 actions for each vehicle: do nothing (0) and send charging (1)
        vehicle_action_space = spaces.Discrete(2)

        # Dynamically create the action space for each vehicle
        self.action_space = spaces.Dict({
            vehicle_id: vehicle_action_space for vehicle_id in self.vehicle_ids
        })

        # We have 2 types of observations: the current state of the battery and the current distance to the next charging station
        single_vehicle_observation_space = spaces.Dict({
            "battery": spaces.Discrete(5000), # battery soc is in between 0 and 5000 Wh
            "distance": spaces.Discrete(2000) # distance to the next charging station is in between 0 and 2000 meters
        })
        self.observation_space = spaces.Dict({
            str(vehicle_id): single_vehicle_observation_space for vehicle_id in self.vehicle_ids
        })

    def __get_observation(self):
        observation = dict()
        for vehicle_id in self.vehicle_ids:
            vehicle_state = self.state[vehicle_id]
            observation[vehicle_id] = {"battery": vehicle_state["battery_soc"], "distance": vehicle_state["distance_to_next_cs"]}
        return observation
    
    def __get_info(self):
        return dict()

    def reset(self, seed=None, options=None): # Later: add possibility to set seed by passing it as an argument `env.reset(seed=<desired seed>)`
        """
        Returns: The observation of the initial state
        Reset the environment to initial state so that a new episode (independent of previous ones) may start
        """
        print("reset")
        super().reset(seed=seed) # needed for api compliance
        self.simulation.reset()
        self.simulation.add_vehicles()
        self.simulation.step() # to spawn vehicles
        self.state = self.simulation.get_state()
        observation = self.__get_observation()
        info = self.__get_info()
        print("reset observation: ", observation)
        return (observation, info)

    def step(self, action):
        """
        Returns: The next observation, the reward, done and optionally additional info
        """
        print("step")
        cs_id = "cs_0"

        self.state = self.simulation.get_state()
        observation = self.__get_observation()
        reward = 0

        for vehicle_id in observation.keys():
            if action[vehicle_id] == 1:
                self.simulation.reroute_for_charging(vehicle_id, cs_id)
                print("REROUTED")
            vehicle_state = self.state[vehicle_id]
            destination_is_reached = (vehicle_state["vehicle_destination"] == vehicle_state["vehicle_position"])
            battery_is_empty = (vehicle_state["battery_soc"] <= 0)
            reward_per_vehicle = -10 if battery_is_empty else 1 if destination_is_reached else 0
            reward += reward_per_vehicle

        terminated = (destination_is_reached or battery_is_empty)
        truncated = not self.simulation.active_vehicles_exist()
        info = self.__get_info()

        self.simulation.step()
        print("step observation: ", observation)

        return observation, reward, terminated, truncated, info

    
    def render(self, mode='human'):
        """
        Returns: None
        Show the current environment state e.g. the graphical window in 'CartPole-v1'
        This method must be implemented, but it is OK to have an empty implementation if rendering is not important
        """
        print("render")
        pass

    def close(self):
        """
        Returns: None
        This is optional. Used to cleanup all resources (threads, graphical windows, etc)
        """
        print("close")
        self.simulation.close()


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    env = CircleEnv()
    check_env(env, skip_render_check=True)
    print("CHECKS PASSED")
    env.close()
