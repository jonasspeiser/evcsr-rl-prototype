import gymnasium as gym
from gymnasium import spaces
import numpy as np
from circle_simulation import Simulation

# configure logging
import logging
logger = logging.getLogger("rl.environment") # child logger of "application", parent logger to "application.environment.simulation"

class CircleEnv(gym.Env):
    metadata = {'render_modes': ['human']}


    def __init__(self, render_mode=None, vehicles_to_spawn=1):
        """
        Define self.observation_space and self.action_space
        """
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

    def __log_step_details(self, observation, reward, terminated, truncated, all_vehicles_at_destination, battery_is_empty):
        logger.debug(f"state: {self.state}, step reward: {reward}")
        if terminated:
            logger.info("episode terminated")
            if all_vehicles_at_destination:
                logger.info("all vehicles arrived at their destination")
            if battery_is_empty:
                logger.info("one vehicle's battery is empty")
        if truncated:
            logger.info("episode truncated")        

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
        logger.debug("reset")
        super().reset(seed=seed) # needed for api compliance
        self.added_vehicles = []
        self.simulation.reset()
        self.simulation.add_vehicles(self.vehicles_to_spawn)

        self.vehicle_ids = self.simulation.get_all_vehicle_ids()
        logger.debug(f"vehicle_ids: {self.vehicle_ids}")
        self.arrived_vehicle_ids = []

        self.simulation.step() # to spawn first vehicle
        self.simulation.step()        

        observation = self.__get_observation()
        info = self.__get_info()
        logger.debug(f"reset observation: {observation}")
        return (observation, info)

    def __action_is_charge(self, vehicle_action):
        return vehicle_action in (1,2,3,4)
    
    def __action_is_do_nothing(self, vehicle_action):
        return vehicle_action == 0

    def __handle_vehicle_action(self, vehicle_id, vehicle_action):
        """
        Handles the action for the vehicle with the given vehicle_id.
        """
        logger.debug(f"action {vehicle_action} for {vehicle_id}")
        action_penalty = 0

        # for already arrived vehicles, do nothing and return. No penalty is given because the agent has to select an action for each vehicle in each step (due to the action space being static and not dynamic)
        # I could think about penalizing it if it tries to send it to a charging station instead of just taking action 0. But right now that doesn't seem necessary.
        if vehicle_id in self.arrived_vehicle_ids:
            return action_penalty

        next_charging_stop = self.simulation.get_next_charging_stop_id(vehicle_id)
        charging_stop_is_planned = next_charging_stop is not None
        

        if self.__action_is_charge(vehicle_action):
            charging_stations = self.simulation.get_all_charging_station_ids()
            cs_id = charging_stations[vehicle_action-1] # action 1 means: go to cs_0 -> action-1 gives us the list index
            if cs_id == next_charging_stop:
                logger.debug(f"Charging stop at {cs_id} is already planned for vehicle {vehicle_id}")
                return action_penalty
            try:
                self.simulation.reroute_for_charging(vehicle_id, cs_id)
                logger.debug(f"Vehicle {vehicle_id} rerouted for charging at {cs_id}")
                # penalize the agent if it sends a vehicle charging although its battery is full enough to reach the destination
                remaining_range_is_sufficient = self.simulation.remaining_range_is_sufficient(vehicle_id, buffer=0)
                if remaining_range_is_sufficient is None:
                    logger.debug(f"vehicle {vehicle_id} was asked for remaining range but doesn't seem to exist")
                if remaining_range_is_sufficient:
                    action_penalty = -1
            except ValueError as e: # if the vehicle is past the charging station and rerouting doesn't work
                logger.error(e)
                # penalize the agent for trying to take an illegal action (e.g. vehicle doesn't exist anymore or is past the charging station)
                action_penalty = -1
        elif self.__action_is_do_nothing(vehicle_action):
            if charging_stop_is_planned:
                self.simulation.remove_charging_stop(vehicle_id)
                logger.debug(f"Charging stop removed for vehicle {vehicle_id}")
        return action_penalty

    def __perform_actions(self, actionlist):
        """
        Perform the actions of all vehicles in the actionlist. Returns a penalty value (int) if illegal or unwanted actions were used.
        """
        action_penalty = 0
        for index, vehicle_id in enumerate(self.vehicle_ids):
            vehicle_action = actionlist[index]
            action_penalty += self.__handle_vehicle_action(vehicle_id, vehicle_action)
        return action_penalty

    def __get_reward_for_vehicle(self, vehicle_id, newly_arrived_ids):
        vehicle_state = self.state[vehicle_id]
        battery_soc = vehicle_state["battery_soc"]

        # destination_edge_is_reached = (vehicle_state["vehicle_position"] == vehicle_state["vehicle_destination"])
        destination_is_reached = True if vehicle_id in self.arrived_vehicle_ids else False
        battery_is_empty = (battery_soc is not None) and (battery_soc == 0) # if battery_soc is None, the vehicle is not currently online
        vehicle_just_despawned = True if (newly_arrived_ids and vehicle_id in newly_arrived_ids) else False
        vehicle_just_reached_destination = destination_is_reached and vehicle_just_despawned
        
        if vehicle_just_despawned:
            logger.info(f"Vehicle {vehicle_id} despawned")

        if vehicle_just_reached_destination:
            logger.info(f"Vehicle {vehicle_id} JUST reached destination")

        if destination_is_reached:
            logger.debug(f"Vehicle {vehicle_id} destination is reached")

        if battery_is_empty:
            logger.info(f"Vehicle {vehicle_id} is empty")

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

    def step(self, action):
        """
        Returns: The next observation, the reward, done and optionally additional info
        """
               
        charging_request = False
        accumulated_reward = 0

        # perform actions for all vehicles
        action_penalty = self.__perform_actions(action)
        accumulated_reward += action_penalty
        
        # loop through sumo-steps until the next vehicle goes online
        loop_counter = 0
        while not charging_request:
            
            logger.debug(f"current sumo time step: {self.simulation.get_current_time_step()}")
            observation = self.__get_observation()
            logger.debug(f"step while loop observation: {observation}")

            # Find out if there are new vehicle ids online or offline
            newly_spawned_ids = self.simulation.get_spawned_vehicle_ids()
            newly_arrived_ids = self.simulation.get_arrived_vehicle_ids()
            self.arrived_vehicle_ids.extend(newly_arrived_ids)
            logger.debug(f"newly_arrived_ids: {newly_arrived_ids}, arrived_vehicle_ids: {self.arrived_vehicle_ids}")
            # Find out if there are vehicles that just finished charging
            just_charged_ids = self.simulation.get_charging_stop_ending_vehicle_ids()

            temp_reward, one_vehicle_is_empty, all_vehicles_at_destination = self.__calculate_reward(observation, newly_arrived_ids)
            logger.debug(f"step while loop reward: {temp_reward}")
            accumulated_reward += temp_reward
            # Terminate only when either ONE vehicle is empty or ALL vehicles are at destination
            terminated = one_vehicle_is_empty or all_vehicles_at_destination
            # Truncate (abort) when it takes too long (i.e. more than 300 SUMO simulation steps WITHOUT a charging request being triggered)
            # truncated = not self.simulation.active_vehicles_exist()
            loop_counter += 1
            truncated = loop_counter > 300

            # end the step for the agent if there is a charging request or the episode is terminated or truncated  
            # a charging request is generated whenever a new vehicle enters the simulation or a vehicle just finished charging
            # i.e. a vehicle was not evaluated yet or needs re-evaluation
            if newly_spawned_ids or just_charged_ids:
                charging_request = True
                logger.debug(f"charging request for {newly_spawned_ids}, {just_charged_ids}")

            if terminated or truncated:
                break

            self.simulation.step()
        
        reward = accumulated_reward
        info = self.__get_info()

        self.__log_step_details(observation, reward, terminated, truncated, all_vehicles_at_destination, one_vehicle_is_empty)
        
        return observation, reward, terminated, truncated, info

    
    def render(self, mode='human'):
        """
        Returns: None
        Show the current environment state e.g. the graphical window in 'CartPole-v1'
        This method must be implemented, but it is OK to have an empty implementation if rendering is not important
        """
        logger.debug("render")
        pass

    def close(self):
        """
        Returns: None
        This is optional. Used to cleanup all resources (threads, graphical windows, etc)
        """
        logger.debug("close")
        self.simulation.close()


if __name__ == "__main__":

    def configure_logging():
        # set up logging to file
        logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='logs/myapp.log',
                        filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logging.getLogger("rl").addHandler(console_handler)

    # Some quick tests

    def test_env():
        from stable_baselines3.common.env_checker import check_env
        env = CircleEnv()
        check_env(env, skip_render_check=True)
        print("CHECKS PASSED")
        env.close()

    def demo_env():
        env = CircleEnv(render_mode="human", vehicles_to_spawn=15)
        observation, info = env.reset()
        for _ in range(20):
            action = env.action_space.sample() # select a random action
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
        env.close()
    
    configure_logging()
    test_env()
    # demo_env()