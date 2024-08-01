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
        # We have 2 types of observations: the current state of the battery and the current distance to the next charging station
        # battery soc is in between 0 and 5000 Wh
        battery_space = spaces.Discrete(5000)
        # distance to the next charging station is in between 0 and 2000 meters
        distance_space = spaces.Discrete(2000)
        self.observation_space = spaces.Dict({
            "battery": battery_space,
            "distance": distance_space
        })

        # We have 2 actions: do nothing (0) and send charging (1)
        self.action_space = spaces.Discrete(2)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.simulation = Simulation(gui=True)
        else:
            self.simulation = Simulation()
        self.simulation.add_vehicles()
        self.simulation.step() # to spawn vehicles
        

    def __get_observation(self):
        return {"battery": self.state["battery_soc"], "distance": self.state["distance_to_next_cs"]}

    def __get_info(self):
        return dict()

    def reset(self, seed=None, options=None): # Later: add possibility to set seed by passing it as an argument `env.reset(seed=<desired seed>)`
        """
        Returns: The observation of the initial state
        Reset the environment to initial state so that a new episode (independent of previous ones) may start
        """
        print("reset")
        vehicle_id = "myVehicle"
        super().reset(seed=seed) # needed for api compliance
        self.simulation.reset()
        self.simulation.add_vehicles()
        self.simulation.step() # to spawn vehicles
        self.state = self.simulation.get_state(vehicle_id)
        observation = self.__get_observation()
        info = self.__get_info()
        print("reset observation: ", observation)
        return (observation, info)

    def step(self, action):
        """
        Returns: The next observation, the reward, done and optionally additional info
        """
        print("step")
        vehicle_id = "myVehicle"
        cs_id = "cs_0"
        if action == 1:
            self.simulation.reroute_for_charging(vehicle_id, cs_id)
            print("REROUTED")
        self.state = self.simulation.get_state(vehicle_id)
        observation = self.__get_observation()
        destination_is_reached = (self.state["vehicle_destination"] == self.state["vehicle_position"])
        battery_is_empty = (self.state["battery_soc"] <= 0)
        reward = -1 if battery_is_empty else 1 if destination_is_reached else 0
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
