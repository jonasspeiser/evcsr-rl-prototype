import gymnasium as gym
from gymnasium import spaces
import numpy as np
from circletest import Simulation
import logging

class CircleEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



    def __init__(self, render_mode=None, log_level="info", vehicles_to_spawn=1):
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
        self.vehicles_to_spawn = vehicles_to_spawn
        self.simulation.add_vehicles(self.vehicles_to_spawn)

        self.vehicle_ids = self.simulation.get_all_vehicle_ids()
        logging.debug(f"vehicle_ids: {self.vehicle_ids}")
        self.arrived_vehicle_ids = []

        # --- Define action space ---        
        # We have 5 actions for each vehicle: do nothing (0), send charging to cs_0 (1), send charging to cs_1 (2), ...
        actions_per_vehicle = 5 
        # Dynamically create the action space for each vehicle
        # e.g. spaces.MultiDiscrete([2,2,2]) for 3 vehicles
        action_space_list = [actions_per_vehicle for vehicle in self.vehicle_ids]
        # MultiDiscrete action space because Stable Baselines doesn't support dict action spaces
        self.action_space = spaces.MultiDiscrete(action_space_list)


        # --- Define observation space ---
        # We have 2 types of observations: the current state of the battery and the current distance to the next charging station
        # battery soc is in between 0 and 100000 Wh, distance to the next charging station is in between 0 and 2000 meters
        # -1 is used to signal that the vehicle is not spawned yet ("padding")
        single_vehicle_observation_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1]), high=np.array([100000, 2000, 2000, 2000, 2000]), dtype=int) 
        self.observation_space = spaces.Dict({
            str(vehicle_id): single_vehicle_observation_space for vehicle_id in self.vehicle_ids
        })

    def __get_observation(self):
        self.state = self.simulation.get_state()
        observation = {}
        for vehicle_id in self.vehicle_ids:
            vehicle_state = self.state[vehicle_id]
            distance_dict = vehicle_state["distance_to_cs"]
            if distance_dict == None:
                vehicle_observation = [-1, -1, -1, -1, -1] # signal that vehicle is not spawned yet
            else:
                distance_to_cs = []
                for charging_station_id, distance in distance_dict.items():
                    distance_to_cs.append(distance)
                vehicle_observation = [vehicle_state["battery_soc"]] + distance_to_cs
            observation[vehicle_id] = np.array(vehicle_observation, dtype=int)
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
        self.added_vehicles = []
        self.simulation.reset()
        self.simulation.add_vehicles(self.vehicles_to_spawn)

        self.vehicle_ids = self.simulation.get_all_vehicle_ids()
        logging.debug(f"vehicle_ids: {self.vehicle_ids}")
        self.arrived_vehicle_ids = []

        self.simulation.step() # to spawn first vehicle
        self.simulation.step()        

        observation = self.__get_observation()
        info = self.__get_info()
        logging.debug(f"reset observation: {observation}")
        return (observation, info)

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

    def __log_step_details(self, observation, reward, terminated, truncated, all_vehicles_at_destination, battery_is_empty):
        logging.debug(f"state: {self.state}, step reward: {reward}")
        if terminated:
            logging.debug("episode terminated")
            if all_vehicles_at_destination:
                logging.info("all vehicles arrived at their destination")
            if battery_is_empty:
                logging.info("one vehicle's battery is empty")
        if truncated:
            logging.info("episode truncated")

    def __action_is_charge(self, vehicle_action):
        return vehicle_action in (1,2,3,4)
    
    def __action_is_do_nothing(self, vehicle_action):
        return vehicle_action == 0

    def __handle_vehicle_action(self, vehicle_id, vehicle_action):
        """
        Handles the action for the vehicle with the given vehicle_id.
        """
        logging.debug(f"action {vehicle_action} for {vehicle_id}")
        next_charging_stop = self.simulation.get_next_charging_stop_id(vehicle_id)
        charging_stop_is_planned = next_charging_stop is not None

        if self.__action_is_charge(vehicle_action):
            charging_stations = self.simulation.get_all_charging_station_ids()
            cs_id = charging_stations[vehicle_action-1] # action 1 means: go to cs_0 -> action-1 gives us the list index
            if cs_id == next_charging_stop:
                logging.debug(f"Charging stop at {cs_id} is already planned for vehicle {vehicle_id}")
                return
            try:
                self.simulation.reroute_for_charging(vehicle_id, cs_id)
                logging.debug(f"Vehicle {vehicle_id} rerouted for charging at {cs_id}")
            except ValueError: # if the vehicle is past the charging station and rerouting doesn't work
                raise ValueError(f"Rerouting failed for vehicle {vehicle_id}")
        elif self.__action_is_do_nothing(vehicle_action):
            if charging_stop_is_planned:
                self.simulation.remove_charging_stop(vehicle_id)
                logging.debug(f"Charging stop removed for vehicle {vehicle_id}")

    def __perform_actions(self, actionlist):
        """
        Perform the actions of all vehicles in the actionlist. Returns a penalty value (int) if illegal actions were used.
        """
        illegal_action_penalty = 0

        for index, vehicle_id in enumerate(self.vehicle_ids):
            vehicle_action = actionlist[index]
            try:
                self.__handle_vehicle_action(vehicle_id, vehicle_action)
            except ValueError as e:
                # penalize the agent for trying to take an illegal action (e.g. vehicle doesn't exist anymore or is past the charging station)
                logging.error(e)
                illegal_action_penalty -= 1

        return illegal_action_penalty

    def __get_reward_for_vehicle(self, vehicle_id, newly_arrived_ids):
        vehicle_state = self.state[vehicle_id]
        battery_soc = vehicle_state["battery_soc"]

        # destination_edge_is_reached = (vehicle_state["vehicle_position"] == vehicle_state["vehicle_destination"])
        destination_is_reached = True if vehicle_id in self.arrived_vehicle_ids else False
        logging.debug(f"checking if {vehicle_id} is in {self.arrived_vehicle_ids}")
        logging.debug(f"checking battery_is_empty for Vehicle {vehicle_id}. Battery_soc is {battery_soc} ")
        battery_is_empty = (battery_soc is not None) and (battery_soc == 0) # if battery_soc is None, the vehicle is not currently online
        vehicle_just_despawned = True if (newly_arrived_ids and vehicle_id in newly_arrived_ids) else False
        vehicle_just_reached_destination = destination_is_reached and vehicle_just_despawned
        
        logging.debug(f"newly_arrived_ids: {newly_arrived_ids} (checked with vehicle_id: {vehicle_id})")

        if vehicle_just_despawned:
            logging.info(f"Vehicle {vehicle_id} despawned")

        if vehicle_just_reached_destination:
            logging.info(f"Vehicle {vehicle_id} JUST reached destination")

        if destination_is_reached:
            logging.debug(f"Vehicle {vehicle_id} destination is reached")

        if battery_is_empty:
            logging.info(f"Vehicle {vehicle_id} is empty")

        reward_per_vehicle = -10 if battery_is_empty else 1 if vehicle_just_reached_destination else 0
        # TODO: battery_is_empty, destination_is_reached sollten im state gespeichert werden (z.B. in einem vehicle objekt). Dann ist die Funktion hier auch deutlich sauberer
        return reward_per_vehicle, battery_is_empty, destination_is_reached


    def __calculate_reward(self, observation, newly_arrived_ids):
        reward = 0
        one_vehicle_is_empty = False
        all_vehicles_at_destination = True

        for index, vehicle_id in enumerate(observation):
            reward_per_vehicle, vehicle_is_empty, vehicle_is_at_destination = self.__get_reward_for_vehicle(vehicle_id, newly_arrived_ids, )
            reward += reward_per_vehicle

            if vehicle_is_empty:
                one_vehicle_is_empty = True        

            if not vehicle_is_at_destination:
                all_vehicles_at_destination = False
        return reward, one_vehicle_is_empty, all_vehicles_at_destination

    def step(self, action, log_level=None):
        """
        Returns: The next observation, the reward, done and optionally additional info
        """
        # Dynamically set the logging level for this step
        if log_level is not None:
            # Save the current logging level to reset it after the step
            original_log_level = logging.getLogger().level
            self.__set_logging_level(log_level)
                
        charging_request = False
        accumulated_reward = 0

        # perform actions for all vehicles
        illegal_action_penalty = self.__perform_actions(action)
        accumulated_reward += illegal_action_penalty
        
        # loop through sumo-steps until the next vehicle goes online
        loop_counter = 0
        while not charging_request:
            
            logging.debug(f"current sumo time step: {self.simulation.get_current_time_step()}")
            observation = self.__get_observation()
            logging.debug(f"step while loop observation: {observation}")

            # Find out if there are new vehicle ids online or offline
            newly_spawned_ids = self.simulation.get_spawned_vehicle_ids()
            newly_arrived_ids = self.simulation.get_arrived_vehicle_ids()
            self.arrived_vehicle_ids.extend(newly_arrived_ids)

            temp_reward, one_vehicle_is_empty, all_vehicles_at_destination = self.__calculate_reward(observation, newly_arrived_ids)
            logging.debug(f"step while loop reward: {temp_reward}")
            accumulated_reward += temp_reward
            # Terminate only when either ONE vehicle is empty or ALL vehicles are at destination
            terminated = one_vehicle_is_empty or all_vehicles_at_destination
            # Truncate (abort) when it takes too long
            # truncated = not self.simulation.active_vehicles_exist()
            loop_counter += 1
            truncated = loop_counter > 300

            # end the step for the agent if there is a charging request or the episode is terminated or truncated  
            if newly_spawned_ids:
                charging_request = True
                logging.debug(f"charging request for {newly_spawned_ids}")

            if terminated or truncated:
                break

            self.simulation.step()
        
        reward = accumulated_reward
        info = self.__get_info()

        self.__log_step_details(observation, reward, terminated, truncated, all_vehicles_at_destination, one_vehicle_is_empty)
        # Reset the logging level to its original state
        if log_level is not None:
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

    def demo_env():
        env = CircleEnv(render_mode="human", log_level="info", vehicles_to_spawn=15)
        observation, info = env.reset()
        for _ in range(200):
            action = env.action_space.sample() # select a random action
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
        env.close()
    
    # test_env()
    demo_env()