import gymnasium as gym
from gymnasium import spaces
import numpy as np
from circletest import Simulation

class CircleEnv(gym.Env):
    metadata = {'render_modes': ['human']}


    def __init__(self, render_mode=None):
        """
        Define self.observation_space and self.action_space
        """
        print("init")
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            gui = True
        else:
            gui = False
        self.simulation = Simulation(gui=gui)
        self.vehicles_to_spawn = 5
        self.simulation.add_vehicles(self.vehicles_to_spawn)
        self.simulation.step() # to spawn vehicles

        self.vehicle_ids = self.simulation.get_all_vehicle_ids()  
        print("vehicle_ids: ", self.vehicle_ids)

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
        single_vehicle_observation_space = spaces.MultiDiscrete([5000, 2000]) # battery soc is in between 0 and 5000 Wh, distance to the next charging station is in between 0 and 2000 meters
        self.observation_space = spaces.Dict({
            str(vehicle_id): single_vehicle_observation_space for vehicle_id in self.vehicle_ids
        })

    def __get_observation(self):
        observation = dict()
        for vehicle_id in self.vehicle_ids:
            vehicle_state = self.state[vehicle_id]
            print("vehicle_state: ", vehicle_state)
            # battery_soc = np.array([vehicle_state["battery_soc"]], dtype=int)
            # distance_to_next_cs = np.array([vehicle_state["distance_to_next_cs"]], dtype=int)
            # observation[vehicle_id] = [battery_soc, distance_to_next_cs]
            observation[vehicle_id] = np.array([vehicle_state["battery_soc"], vehicle_state["distance_to_next_cs"]], dtype=int)
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
        self.simulation.add_vehicles(self.vehicles_to_spawn)
        self.simulation.step() # to spawn vehicles
        self.simulation.step()
        self.vehicle_ids = self.simulation.get_all_vehicle_ids()  
        print("vehicle_ids: ", self.vehicle_ids)
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

        for index, vehicle_id in enumerate(observation):
            if action[index] == 1:
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
    def test_env():
        from stable_baselines3.common.env_checker import check_env
        env = CircleEnv()
        check_env(env, skip_render_check=True)
        print("CHECKS PASSED")
        env.close()
    
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env

    # Instantiate the env
    # vec_env = make_vec_env(CircleEnv, n_envs=1, env_kwargs=dict())
    # Train the agent
    env = CircleEnv("human")
    model = A2C("MultiInputPolicy", env, verbose=1).learn(5000)   
