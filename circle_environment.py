import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque, Counter
from circle_simulation import Simulation
from data_processing import Obelis_Data_Provider

# configure logging
import logging
logger = logging.getLogger("rl.environment") # child logger of "application", parent logger to "application.environment.simulation"

""" Reward strategy could look like this:

class RewardStrategy:
    def calculate_step_reward(self, state, action, done):
        raise NotImplementedError
    
    def calculate_final_reward(self, state, action, done):
        raise NotImplementedError

class RewardBasic(RewardStrategy):
    def calculate_step_reward(self, state, action, done):
        return 0
    def calculate_final_reward(self, state, action, done):
        return ttt
class RewardOnlyPreventEmpty(RewardStrategy):
    def calculate_step_reward(self, state, action, done):
        return 1.0 if action == 0 else -1.0

class RewardShaping(RewardStrategy):
    def calculate_step_reward(self, state, action, done):
        return 1.0 if action == 0 else -1.0
"""

class CircleEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None, env_version="basic", vehicles_to_spawn=1,
                 observation_sampling_rate=30, truncate_after_n_steps=300, simulate_non_member_evs: bool = False):
        """
        Initialize the environment and simulation.
        Define self.observation_space and self.action_space.

        Parameters:
        - render_mode: str, the mode in which the environment should be rendered. If None, no rendering is done. "human" shows a graphical window with the simulation.
        - vehicles_to_spawn: int, the number of vehicles to spawn in the simulation
        - observation_sampling_rate: int, the rate at which the observation is sampled (i.e. every x simulation steps)
        - truncate_after_n_steps: int, the number of simulation steps after which the episode is truncated if no charging request is triggered
        - simulate_non_member_evs: bool, whether or not to spawn non-member EVs. Default: False
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_sampling_rate = observation_sampling_rate # only sample the observation every x simulation steps for performance reasons
        self.truncate_after_n_steps = truncate_after_n_steps # abort the episode if it takes too long without a charging request being triggered

        gui = (self.render_mode == "human")
        self.simulation = Simulation(gui=gui)
        self.vehicles_to_spawn = vehicles_to_spawn
        self.simulate_non_member_evs = simulate_non_member_evs
        if self.simulate_non_member_evs:
            self.data_provider = Obelis_Data_Provider()

        self.simulation.add_vehicles(self.vehicles_to_spawn)
        self.vehicle_ids = self.simulation.get_all_vehicle_ids()
        logger.debug(f"Initial vehicle_ids: {self.vehicle_ids}")

        # Create Vehicle instances for each vehicle id
        self.vehicles = {vid: Vehicle(vid) for vid in self.vehicle_ids}

        # --- Define action space ---        
        # We have 5 actions for each vehicle: do nothing (0), send charging to cs_0 (1), send charging to cs_1 (2), ...
        actions_per_vehicle = 5
        self.action_space = spaces.Discrete(actions_per_vehicle)


        # --- Define observation space ---
        # We have 4 types of observations per vehicle: the current state of the battery, the current distance to each charging station, the last selected action, arrived at destination, request pending
        # battery soc is in between 0 and 100000 Wh, distance to each of the charging stations is in between 0 and 2000 meters
        # These values are normalized.
        # The last selected action is 0 for "do_nothing" or 1-4 for the corresponding CS (cf. handle_action())
        # A binary value informs wether the vehicle has (1) or has not (0) reached its destination yet
        # A binary value signals the "active" vehicle which filed the current charging request (with a 1, otherwise 0)
        # -1 is used to signal that the vehicle is not spawned yet ("padding")
        single_vehicle_observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 4, 1, 1]),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            str(vehicle_id): single_vehicle_observation_space for vehicle_id in self.vehicle_ids
        })

    def __log_step_details(self, simulation_state, observation, reward, terminated, truncated, all_vehicles_at_destination, battery_is_empty):
        logger.debug(f"State: {simulation_state}, Step reward: {reward}")
        if terminated:
            logger.info("Episode terminated")
            if all_vehicles_at_destination:
                logger.info("All vehicles arrived at their destination")
            if battery_is_empty:
                logger.info("One or more vehicles ran out of battery")
        if truncated:
            logger.info("Episode truncated")

    def __set_charging_stops_per_episode_mean(self):
        """Used for tensorboard logging. Calculates the global average for number of charging stops per vehicle."""
        total = sum(self.charging_stops_per_episode_counter.values())
        count = len(self.charging_stops_per_episode_counter) if self.charging_stops_per_episode_counter else 1
        self.charging_stops_per_episode_mean = total / count

    def __update_accumulated_waiting_times(self):
        """Get the waiting times for each vehicle out of the simulation. This is only possible as long as a vehicle is still online."""
        for vehicle_id in self.simulation.get_online_vehicle_ids():
            self.accumulated_waiting_times[vehicle_id] = self.simulation.get_vehicle_waiting_time(vehicle_id)
    
    def __set_cumulated_waiting_time_per_episode(self):
        """Used for tensorboard logging. Sums up the individual waiting times to get one global value."""
        self.cumulated_waiting_time = sum(self.accumulated_waiting_times.values())

    def __set_cumulated_waiting_time_per_episode_terminated(self):
        """Used for tensorboard logging. Sums up the individual waiting times to get one global value. Only tracks terminated episodes (not truncated ones)"""
        self.cumulated_waiting_time_only_terminated = sum(self.accumulated_waiting_times.values())

    def __add_non_member_vehicles(self):
        self.simulation.add_non_member_routes()
        vehicle_data = self.data_provider.get_non_member_vehicle_data()
        for entry in vehicle_data:
            self.simulation.add_non_member_vehicle(cs_id=entry["cs_id"],
                                                   depart_time=entry["charge_begin_seconds"],
                                                   charge_duration=entry["charge_duration"])
    
    def __get_observation(self, simulation_state):
        """
        Build the observation dictionary by updating each vehicle from the simulation state.
        """
        observation = {}
        for vehicle_id in self.vehicle_ids:
            vehicle_state = simulation_state[vehicle_id]
            distance_dict = vehicle_state["distance_to_cs"]
            if distance_dict == None:
                vehicle_observation = [-1, -1, -1, -1, -1, -1, 0, 0] # signal that vehicle is not spawned yet
            else:
                distance_to_cs = []
                for charging_station_id, distance in distance_dict.items():
                    normalized_distance = distance / 2000 # 2000 km is considered as max. possible distance
                    distance_to_cs.append(normalized_distance)
                normalized_soc = vehicle_state["battery_soc"] / 100000 # 100000 Wh is considered as max. possible capacity
                last_selected_action = self.currently_selected_actions[vehicle_id]
                destination_reached = int(self.__destination_is_reached(vehicle_id))
                active_charging_request = int(vehicle_id == self.active_charging_request_for_vehicle_id)

                vehicle_observation = [normalized_soc] + distance_to_cs + [last_selected_action] + [destination_reached] + [active_charging_request]
            observation[vehicle_id] = np.array(vehicle_observation, dtype=np.float32)
        return observation
    
    def __get_info(self):
        return dict()

    def reset(self, seed=None, options=None): # Later: add possibility to set seed by passing it as an argument `env.reset(seed=<desired seed>)`
        """
        Reset the environment and simulation for a new episode.
        Returns: The observation of the initial state
        """
        logger.debug("Resetting environment")
        super().reset(seed=seed) # needed for api compliance

        # reset logging values (tensorboard logging)
        self.charging_stops_per_episode_counter = Counter(dict.fromkeys(self.vehicle_ids, 0))
        self.accumulated_waiting_times = {}
        #self.charging_stops_per_episode_mean = None
        #self.cumulated_waiting_time = None

        self.added_vehicles = []
        self.vehicle_times = {}
        self.simulation.reset()
        self.simulation.add_vehicles(self.vehicles_to_spawn)

        self.vehicle_ids = self.simulation.get_all_vehicle_ids()
        logger.debug(f"vehicle_ids: {self.vehicle_ids}")
        self.arrived_vehicle_ids = []
        self.empty_vehicle_ids = []
        self.low_battery_ids = []
        self.currently_selected_actions = dict.fromkeys(self.vehicle_ids, -1) # lists the last action the agent selected for each vehicle, puts -1 as default value (i.e. "no action selected yet")
        self.charging_request_queue = deque()
        self.active_charging_request_for_vehicle_id = None # states the vehicle_id for which the agent has to select an action in the current step

        if self.simulate_non_member_evs:
            self.__add_non_member_vehicles()

        # Wait until a charging request is generated.
        while self.active_charging_request_vehicle_id is None:
            newly_spawned_ids = self.simulation.get_spawned_vehicle_ids()
            self.__update_vehicle_times(newly_spawned_ids, newly_arrived_ids=None)
            self.__check_for_charging_request(newly_spawned_ids)
            self.simulation.step()

        simulation_state = self.simulation.get_state()
        observation = self.__get_observation(simulation_state)
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
        self.currently_selected_actions[vehicle_id] = vehicle_action
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
                    logger.debug(f"action penalty: vehicle {vehicle_id} was asked to charge but has enough range to reach destination (reward -1)")
            except ValueError as e: # if the vehicle is past the charging station and rerouting doesn't work
                logger.error(e)
                # penalize the agent for trying to take an illegal action (e.g. vehicle doesn't exist anymore or is past the charging station)
                action_penalty = -1
                logger.debug(f"action penalty: vehicle {vehicle_id} tried to charge but is past the charging station (reward -1)")
        elif self.__action_is_do_nothing(vehicle_action):
            if charging_stop_is_planned:
                self.simulation.remove_charging_stop(vehicle_id)
                logger.debug(f"Charging stop removed for vehicle {vehicle_id}")
        return action_penalty

    def __destination_is_reached(self, vehicle_id):
        return True if vehicle_id in self.arrived_vehicle_ids else False

    def __battery_is_empty(self, vehicle_id):
        battery_soc = self.simulation.get_battery_soc(vehicle_id)
        return (battery_soc is not None) and (battery_soc <= 0) # if battery_soc is None, the vehicle is not currently online
    
    def __battery_just_died(self, vehicle_id):
        if vehicle_id in self.empty_vehicle_ids: # i.e. vehicle was already empty before the current step
            return False
        if self.__battery_is_empty(vehicle_id): # i.e. vehicle is empty and was NOT empty before the current step
            self.empty_vehicle_ids.append(vehicle_id)
            return True
        return False # i.e. vehicle is NOT empty

    def __vehicle_has_just_despawned(self, vehicle_id, newly_arrived_ids):
        return True if (newly_arrived_ids and vehicle_id in newly_arrived_ids) else False

    def __get_reward_for_vehicle(self, vehicle_id, battery_just_died, vehicle_has_just_reached_destination):
        
        reward_per_vehicle = -100 if battery_just_died else 10 if vehicle_has_just_reached_destination else 0

        if vehicle_has_just_reached_destination:
            logger.info(f"Vehicle {vehicle_id} JUST reached destination (reward +10)")

        if battery_just_died:
            logger.info(f"Vehicle {vehicle_id} JUST died (reward -100)")

        return reward_per_vehicle

    def __update_vehicle_times(self, newly_spawned_ids, newly_arrived_ids):
        """Stores the actual departure and arrival times for all vehicles which arrived at destination during the current simulation step."""
        if not (newly_spawned_ids or newly_arrived_ids):
            return
        
        current_time = self.simulation.get_current_time_step()
    
        for vehicle_id in newly_spawned_ids or []:
            self.vehicle_times.setdefault(vehicle_id, {})['departure'] = current_time
            print(f"departure for {vehicle_id} at {current_time}")
    
        for vehicle_id in newly_arrived_ids or []:
            self.vehicle_times[vehicle_id]['arrival'] = current_time
            print(f"arrival for {vehicle_id} at {current_time}")

    def __calculate_final_reward(self):
        """Only given at the end of an episode"""
        global_ttt = 0
        for vehicle_id in self.vehicle_times:
            # Ensure 'arrival' exists; if not, set it to the current time
            if 'arrival' not in self.vehicle_times[vehicle_id]:
                self.vehicle_times[vehicle_id]['arrival'] = self.simulation.get_current_time_step()
            # Calculate TTT
            vehicle_ttt = self.vehicle_times[vehicle_id]['arrival'] - self.vehicle_times[vehicle_id]['arrival']
            global_ttt += vehicle_ttt
        return global_ttt

    def __calculate_step_reward(self, newly_arrived_ids, charging_ids):
        reward = 0
        one_vehicle_just_died = False
        all_vehicles_at_destination = True

        for index, vehicle_id in enumerate(self.vehicle_ids):
            vehicle_is_charging = vehicle_id in charging_ids
            vehicle_just_died = self.__battery_just_died(vehicle_id)
            vehicle_is_at_destination = self.__destination_is_reached(vehicle_id)
            vehicle_has_just_despawned = self.__vehicle_has_just_despawned(vehicle_id, newly_arrived_ids)
            if vehicle_has_just_despawned:
                logger.info(f"Vehicle {vehicle_id} despawned")

            vehicle_has_just_reached_destination = vehicle_is_at_destination and vehicle_has_just_despawned

            reward_per_vehicle = self.__get_reward_for_vehicle(vehicle_id, vehicle_just_died, vehicle_has_just_reached_destination)
            reward += reward_per_vehicle

            if vehicle_just_died:
                one_vehicle_just_died = True        

            if not vehicle_is_at_destination:
                all_vehicles_at_destination = False

            if vehicle_is_charging:
                # every sumo step (i.e. every second) a vehicle is charging and needs to do so to arrive at its destination, the agent gets +1 reward
                # TODO: This may be a bit much. Maybe reduce the reward to 0.1 or 0.01 as it is played out per second
                remaining_range_is_sufficient = self.simulation.remaining_range_is_sufficient(vehicle_id, buffer=0)
                if not remaining_range_is_sufficient:
                    logger.debug(f"vehicle {vehicle_id} is charging and otherwise doesn't have enough remaining range to arrive at its destination (+1 reward)")
                    reward += 1
                                
        return reward, one_vehicle_just_died, all_vehicles_at_destination

    def __update_low_battery_ids(self, just_charged_ids):
        """Deletes all vehicle_ids that just charged out of the low_battery_ids list"""
        self.low_battery_ids = [vehicle_id for vehicle_id in self.low_battery_ids if vehicle_id not in just_charged_ids]

    def __get_new_low_battery_ids(self):
        battery_threshold = 0.2 # the value under which the battery soc should be considered low
        state = self.simulation.get_state()
        new_low_battery_ids = []
        for vehicle_id, vehicle_stats in state.items():
            max_battery_capacity = vehicle_stats["max_battery_capacity"]
            battery_soc = vehicle_stats["battery_soc"]
            if battery_soc is None: # i.e. if the vehicle did not spawn in the simulation yet or despawned already
                continue
            relative_battery_soc = battery_soc / max_battery_capacity
            # only account for vehicles that just entered the state of low battery. Not the ones that where already low during the last step.
            if  relative_battery_soc < battery_threshold and vehicle_id not in self.low_battery_ids:
                new_low_battery_ids.append(vehicle_id)
                self.low_battery_ids.append(vehicle_id)
        return new_low_battery_ids
    
    def __check_for_charging_request(self, newly_spawned_ids):
        """
        A charging request is generated (added to the queue) whenever a new vehicle enters the simulation or a vehicle just finished charging (i.e. when a vehicle was not evaluated yet or needs re-evaluation).
        
        Pops the next charging request out of the queue (FIFO) and sets it as the active charging request.

        Returns: True if there is an active charging request, False otherwise
        """
        # Find out if there are vehicles that just finished charging
        just_charged_ids = self.simulation.get_charging_stop_ending_vehicle_ids()
        # Find out if there are vehicles that just entered low battery status
        self.__update_low_battery_ids(just_charged_ids)
        new_low_battery_ids = self.__get_new_low_battery_ids() 

        charging_requests = newly_spawned_ids + just_charged_ids + new_low_battery_ids

        # (this value is only used for logging) increase the counters for each vehicle that just stopped charging by one
        self.charging_stops_per_episode_counter.update(just_charged_ids)

        if charging_requests:
            logger.debug(f"newly_spawned_ids: {newly_spawned_ids}, just_charged_ids: {just_charged_ids}, new_low_battery_ids: {new_low_battery_ids}")
            self.charging_request_queue.extend(charging_requests)
        if self.charging_request_queue: # i.e. if the queue is not empty
            self.active_charging_request_for_vehicle_id = self.charging_request_queue.popleft()
            logger.debug(f"charging request for {self.active_charging_request_for_vehicle_id}")
            return True
        return False
    
    def step(self, action):
        """
        Executes simulation steps until the next charging request (or termination/truncation).
        Returns: The next observation, reward, done flags, and additional info.
        """
        charging_request = False
        accumulated_reward = 0
        loop_counter = 0

        # loop through sumo-steps until the next vehicle goes online or a vehicle just finished charging
        while not charging_request:
            logger.debug(f"Simulation time step: {self.simulation.get_current_time_step()}")
            newly_spawned_ids = self.simulation.get_spawned_vehicle_ids()
            newly_arrived_ids = self.simulation.get_arrived_vehicle_ids()
            charging_ids = self.simulation.get_charging_vehicle_ids()
            logger.debug(f"charging_ids: {charging_ids}")
            self.arrived_vehicle_ids.extend(newly_arrived_ids)
            logger.debug(f"newly_arrived_ids: {newly_arrived_ids}, arrived_vehicle_ids: {self.arrived_vehicle_ids}")

            if newly_spawned_ids or newly_arrived_ids:
                self.__update_vehicle_times(newly_spawned_ids, newly_arrived_ids)

            # only execute once per step
            loop_just_started = (loop_counter == 0)
            if loop_just_started:
                # perform action for the vehicle which filed the charging request
                action_penalty = self.__handle_vehicle_action(self.active_charging_request_for_vehicle_id, action)
                accumulated_reward += action_penalty

            # calculate reward
            temp_reward, one_vehicle_just_died, all_vehicles_at_destination = self.__calculate_step_reward(
                newly_arrived_ids, charging_ids
            )
            logger.debug(f"Step reward from simulation: {temp_reward}")
            accumulated_reward += temp_reward

            
            # Terminate only when ALL vehicles are at destination
            terminated = all_vehicles_at_destination
            # Truncate (abort) when it takes too long (i.e. more than x SUMO simulation steps WITHOUT a charging request being triggered)
            truncated = loop_counter > self.truncate_after_n_steps
            charging_request = self.__check_for_charging_request(newly_spawned_ids)

            important_event_happened = charging_request or terminated or truncated
            some_simulation_time_passed = (loop_counter % self.observation_sampling_rate == 0)

            self.__update_accumulated_waiting_times()

            # get observation (only every few simulation steps, for performance purposes)
            if loop_just_started or some_simulation_time_passed or important_event_happened:
                simulation_state = self.simulation.get_state()
                observation = self.__get_observation(simulation_state)
                logger.debug(f"Intermediate simulation state: {simulation_state}")

            # note: the step for the agent ends if there is a charging request (see while loop condition) or the episode is terminated or truncated  
            if terminated or truncated:
                final_reward = self.__calculate_final_reward()
                logger.info(f"Final reward: {final_reward}")
                if terminated:
                    self.__set_cumulated_waiting_time_per_episode_terminated()
                self.__set_charging_stops_per_episode_mean()
                self.__set_cumulated_waiting_time_per_episode()
                break

            loop_counter += 1
            self.simulation.step()

        reward = accumulated_reward
        info = self.__get_info()
        self.__log_step_details(simulation_state, observation, reward, terminated, truncated,
                                all_vehicles_at_destination, one_vehicle_just_died)
        return observation, reward, terminated, truncated, info

    
    def render(self, mode='human'):
        """
        Returns: None
        Show the current environment state e.g. the graphical window in 'CartPole-v1'
        This method must be implemented, but it is OK to have an empty implementation if rendering is not important
        """
        logger.debug("Render called")
        pass

    def close(self):
        """
        Returns: None
        This is optional. Used to cleanup all resources (threads, graphical windows, etc)
        """
        logger.debug("Closing environment")
        self.simulation.close()

# ======================================
# Main Testing Functions
# ======================================
if __name__ == "__main__":

    def configure_logging(log_file_path='logs/myapp.log'):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        # create file handler which logs even debug messages
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        # add the handlers to the (root) logger
        logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])

    def test_env():
        from stable_baselines3.common.env_checker import check_env
        env = CircleEnv()
        check_env(env, skip_render_check=True)
        print("CHECKS PASSED")
        env.close()

    def take_greedy_action(observation):
        """
        For the vehicle with an active charging request, if battery SOC is above 0.2, do nothing.
        Otherwise, send the vehicle to the nearest charging station.
        """
        for vehicle_obs in observation.values():
            # find out which one is the active vehicle
            active_charging_request_flag = vehicle_obs[7]
            if active_charging_request_flag:
                battery_soc = vehicle_obs[0]
                station_distances = vehicle_obs[1:5]
                break
        if battery_soc > 0.2:
            return 0 # "do nothing"
        
        closest_station = np.argmin(station_distances) # the index of the lowest distance
        return closest_station + 1

    def demo_env(random_seed=None):
        env = CircleEnv(render_mode="human", vehicles_to_spawn=15)
        import random
        random.seed(random_seed)
        observation, info = env.reset(seed=random_seed)
        env.action_space.seed(random_seed)
        # while env.simulation.active_vehicles_exist():

        for _ in range(20):
            #action = env.action_space.sample() # select a random action
            action = take_greedy_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
        env.close()

    configure_logging(log_file_path='logs/myapp.log')
    # Uncomment one of the following to test the environment:
    # test_env()
    demo_env(random_seed=1)