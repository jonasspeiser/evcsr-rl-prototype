import gymnasium as gym
from gymnasium import spaces

class CircleEnv(gym.Env):
    def __init__(self):
        """
        Define self.observation_space and self.action_space
        """
        # We have 2 types of observations: the current state of the battery and the current distance to the next charging station
        # battery soc is in between 0 and 100 %
        battery_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        # distance to the next charging station is in between 0 and 1000 meters
        distance_space = spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.dict({
            "battery": battery_space,
            "distance": distance_space
        })

        # We have 2 actions: do nothing (0) and send charging (1)
        self.action_space = spaces.Discrete(2)

        pass

    def reset(self):
        """
        Returns: The observation of the initial state
        Reset the environment to initial state so that a new episode (independent of previous ones) may start
        """
        raise NotImplementedError

    def step(self, action):
        """
        Returns: The next observation, the reward, done and optionally additional info
        """
        raise NotImplementedError

    
    def render(self, mode='human'):
        """
        Returns: None
        Show the current environment state e.g. the graphical window in 'CartPole-v1'
        This method must be implemented, but it is OK to have an empty implementation if rendering is not important
        """
        pass

    def close(self):
        """
        Returns: None
        This is optional. Used to cleanup all resources (threads, graphical windows, etc)
        """
        pass

    def seed(self, seed=None):
        """
        Returns: List of seeds
        This is optional. Used to set seeds for the environment's random number generator for obtaining deterministic behaviour.
        """
        return