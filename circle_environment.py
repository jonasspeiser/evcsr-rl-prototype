import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque, Counter
from circle_simulation import Simulation
from vehicle import Vehicle
from rewards import BasicRewardStrategy, NoTimeComponentRewardStrategy, RewardShapingStrategy
from data_processing import Obelis_Data_Provider

# configure logging
import logging
logger = logging.getLogger("rl.environment") 
# child logger of "rl", parent logger to "rl.environment.simulation", "rl.environment.vehicle", "rl.environment.rewards"

class CircleEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None, env_version="basic", vehicles_to_spawn=1,
                 observation_sampling_rate=30, truncate_after_n_steps=3000, simulate_non_member_evs: bool = False, random_seed=None):
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
        self.simulation = Simulation(gui=gui, random_seed=random_seed)
        self.vehicles_to_spawn = vehicles_to_spawn
        self.simulate_non_member_evs = simulate_non_member_evs
        if self.simulate_non_member_evs:
            self.data_provider = Obelis_Data_Provider()

        # # Create Vehicle instances for each vehicle id
        # self.vehicles = {vid: Vehicle(vid, self.simulation) for vid in self.vehicle_ids}

        # --- Define action space ---        
        # We have 5 actions for each vehicle: do nothing (0), send charging to cs_0 (1), send charging to cs_1 (2), ...
        actions_per_vehicle = 5
        self.action_space = spaces.Discrete(actions_per_vehicle)


        # --- Define observation space ---
        # Types of observations per vehicle: [0] current state of the battery, [1] current distance to destination, [2:6] current distance to each charging station, [6] last selected action, [7] arrived at destination, [8] request pending
        # battery soc is in between 0 and 100000 Wh, distance to destination and distance to each of the charging stations is in between 0 and 2000 meters
        # These values are normalized.
        # The last selected action is 0 for "do_nothing" or 1-4 for the corresponding CS (cf. handle_action())
        # A binary value informs wether the vehicle has (1) or has not (0) reached its destination yet
        # A binary value signals the "active" vehicle which filed the current charging request (with a 1, otherwise 0)
        # -1 is used to signal that the vehicle is not spawned yet ("padding")
        single_vehicle_observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 4, 1, 1]),
            dtype=np.float32
        )
        self.simulation.add_vehicles(self.vehicles_to_spawn)
        self.vehicle_ids = self.simulation.get_all_vehicle_ids()
        logger.debug(f"Initial vehicle_ids: {self.vehicle_ids}")
        self.observation_space = spaces.Dict({
            str(vehicle_id): single_vehicle_observation_space for vehicle_id in self.vehicle_ids
        })

        # Instantiate the reward strategy based on env_version.
        if env_version == "basic":
            self.reward_strategy = BasicRewardStrategy()
        elif env_version == "noTime":
            self.reward_strategy = NoTimeComponentRewardStrategy()
        elif env_version == "shaping":
            self.reward_strategy = RewardShapingStrategy()
        else:
            raise ValueError(f"Unknown env_version: {env_version}")

    def _log_step_details(self, simulation_state, observation, reward, terminated, truncated, all_vehicles_at_destination):
        logger.debug(f"State: {simulation_state}, Step reward: {reward}")
        if terminated:
            logger.info("Episode terminated")
            if all_vehicles_at_destination:
                logger.info("All vehicles arrived at their destination")
            # if one_vehicle_is_empty:
            #     logger.info("One or more vehicles ran out of battery")
        if truncated:
            logger.info("Episode truncated")

    def _set_charging_stops_per_episode_mean(self):
        """Used for tensorboard logging. Calculates the global average for number of charging stops per vehicle."""
        total = sum(self.charging_stops_per_episode_counter.values())
        count = len(self.charging_stops_per_episode_counter) if self.charging_stops_per_episode_counter else 1
        self.charging_stops_per_episode_mean = total / count

    def _update_accumulated_waiting_times(self):
        """Get the waiting times for each vehicle out of the simulation. This is only possible as long as a vehicle is still online."""
        for vehicle_id in self.simulation.get_online_vehicle_ids():
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle:
                vehicle.waiting_time = self.simulation.get_vehicle_waiting_time(vehicle_id)
    
    def _set_cumulated_waiting_time_per_episode(self):
        """Used for tensorboard logging. Sums up the individual waiting times to get one global value."""
        self.cumulated_waiting_time = sum(v.waiting_time for v in self.vehicles.values())
    
    def _set_cumulated_waiting_time_per_episode_terminated(self):
        """Used for tensorboard logging. Sums up the individual waiting times to get one global value. Only tracks terminated episodes (not truncated ones)"""
        self.cumulated_waiting_time_only_terminated = sum(v.waiting_time for v in self.vehicles.values() if v.arrived)

    def _set_global_ttt(self, current_time):
        """Used for tensorboard logging. Sums up the total travel times of all vehicles."""
        global_ttt = 0
        for vehicle in self.vehicles.values():
            if vehicle.departure_time is None:
                continue
            if vehicle.arrival_time is None:
                vehicle.arrival_time = current_time # in case the episode was truncated, some vehicles never arrive. For these, we set the arrival time to the last step in the truncated episode, so that their travel time also counts into the global counter. These are often vehicles which stand are stuck and therefore have a long travel time already.
            global_ttt += vehicle.get_total_travel_time()
        self.global_ttt = global_ttt
        self.ttt_per_ev_mean = global_ttt / self.vehicles_to_spawn

    def _set_empty_vehicles_per_episode(self):
        self.empty_vehicles_per_episode = sum(
            1 for vehicle in self.vehicles.values()
            if vehicle.empty
        )
        logger.debug(f"empty_vehicles_per_episode: {self.empty_vehicles_per_episode}")


    def _add_non_member_vehicles(self):
        self.simulation.add_non_member_routes()
        vehicle_data = self.data_provider.get_non_member_vehicle_data()
        for entry in vehicle_data:
            self.simulation.add_non_member_vehicle(cs_id=entry["cs_id"],
                                                   depart_time=entry["charge_begin_seconds"],
                                                   charge_duration=entry["charge_duration"])
    
    def _update_and_get_observation(self):
        """
        Build the observation dictionary by updating and collecting each vehicle's observation.
        """
        observation = {}
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.arrived and not vehicle.empty:
                vehicle_state = self.simulation.get_vehicle_state(vehicle_id)
                vehicle.update_from_state(vehicle_state)
            # Mark the vehicle as active if it filed the current charging request.
            is_active = (vehicle_id == self.active_charging_request_vehicle_id)
            observation[vehicle_id] = vehicle.get_observation(is_active=is_active)
        return observation
    
    def _get_info(self):
        """
        Get additional info for logging, such as a human readable version of the simulation state.
        """
        info = {}
        for vehicle_id, vehicle in self.vehicles.items():
            info[vehicle_id] = vehicle.get_info()
        return info

    def reset(self, seed=None, options=None): # Later: add possibility to set seed by passing it as an argument `env.reset(seed=<desired seed>)`
        """
        Reset the environment and simulation for a new episode.
        Returns: The observation of the initial state
        """
        logger.debug("Resetting environment")
        super().reset(seed=seed) # needed for api compliance
        
        if seed:
            self.action_space.seed(seed) # for deterministic results when using env.actions_space.sample()
        self.simulation.reset()
        self.simulation.add_vehicles(amount=self.vehicles_to_spawn)
        self.vehicle_ids = self.simulation.get_all_vehicle_ids()
        logger.debug(f"Reset vehicle_ids: {self.vehicle_ids}")
        # Re-create the vehicles dictionary in case new vehicles were spawned.
        self.vehicles = {vid: Vehicle(vid, self.simulation) for vid in self.vehicle_ids}

        # Additional attributes for managing charging requests and logging
        self.charging_request_queue = deque()
        self.active_charging_request_vehicle_id = None #the vehicle_id for which the agent has to select an action in the current step
        self.charging_stops_per_episode_counter = Counter({vid: 0 for vid in self.vehicle_ids})
        self.low_battery_ids = []
        
        if self.simulate_non_member_evs:
            self._add_non_member_vehicles()

        # Wait until a charging request is generated.
        while self.active_charging_request_vehicle_id is None:
            newly_spawned_ids = self.simulation.get_spawned_vehicle_ids()
            self._update_vehicle_times(newly_spawned_ids, newly_arrived_ids=None)
            self._check_for_charging_request(newly_spawned_ids)
            self.simulation.step()
        
        observation = self._update_and_get_observation()
        info = self._get_info()
        logger.debug(f"Reset observation: {observation}")
        return observation, info

    def _update_vehicle_times(self, newly_spawned_ids, newly_arrived_ids):
        """Stores the actual departure and arrival times for all vehicles which arrived at destination during the current simulation step."""
        if not (newly_spawned_ids or newly_arrived_ids):
            return
        
        current_time = self.simulation.get_current_time_step()
        for vehicle_id in newly_spawned_ids or []:
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle and vehicle.departure_time is None:
                vehicle.departure_time = current_time
                logger.info(f"{vehicle_id} spawned at time {current_time}")
        for vehicle_id in newly_arrived_ids or []:
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle:
                vehicle.arrival_time = current_time
                vehicle.arrived = True
                logger.debug(f"{vehicle_id} arrived at time {current_time}")

    def _check_for_charging_request(self, newly_spawned_ids):
        """
        Checks for charging requests based on newly spawned, just-charged,
        and low-battery vehicles. If any exist, the first in the queue becomes active.
        Returns: True if there is an active charging request, False otherwise.
        """
        # Find out if there are vehicles that just finished charging
        just_charged_ids = self.simulation.get_charging_stop_ending_vehicle_ids()
        for vid in just_charged_ids:
            logger.info(f"{vid}: just finished charging")
        # Find out if there are vehicles that just entered low battery status
        new_low_battery_ids = self._get_new_low_battery_ids(just_charged_ids)
        charging_requests = (newly_spawned_ids or []) + just_charged_ids + new_low_battery_ids
        # (this value is only used for logging) increase the counters for each vehicle that just stopped charging by one
        self.charging_stops_per_episode_counter.update(just_charged_ids)
        if charging_requests:
            logger.debug(f"New charging requests: spawned={newly_spawned_ids}, just charged={just_charged_ids}, low battery={new_low_battery_ids}")
            self.charging_request_queue.extend(charging_requests)
        if self.charging_request_queue:
            self.active_charging_request_vehicle_id = self.charging_request_queue.popleft()
            logger.info(f"Active charging request for: {self.active_charging_request_vehicle_id}")
            return True
        return False

    def _get_new_low_battery_ids(self, just_charged_ids):
        battery_threshold = 0.2 # the value under which the battery soc should be considered low
        new_low_battery_ids = []
        for vehicle_id, vehicle in self.vehicles.items():
            relative_battery_soc = vehicle.relative_battery_soc
            if relative_battery_soc is None: # i.e. if the vehicle did not spawn in the simulation yet or despawned already
                continue
            if vehicle_id in self.low_battery_ids and vehicle_id in just_charged_ids:
                self.low_battery_ids.remove(vehicle_id)
            # only account for vehicles that just entered the state of low battery. Not the ones that where already low during the last step.
            if relative_battery_soc < battery_threshold and vehicle_id not in self.low_battery_ids:
                logger.info(f"{vehicle_id}: low battery")
                new_low_battery_ids.append(vehicle_id)
                self.low_battery_ids.append(vehicle_id)
        return new_low_battery_ids

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

            # Update vehicles' battery soc
            for vid, vehicle in self.vehicles.items():
                if not vehicle.arrived and not vehicle.empty:
                    vehicle.fetch_and_update_battery_values()

            # Update vehicles’ arrival status.
            for vid in newly_arrived_ids:
                if vid in self.vehicles:
                    self.vehicles[vid].arrived = True

            if newly_spawned_ids or newly_arrived_ids:
                self._update_vehicle_times(newly_spawned_ids, newly_arrived_ids)

            # only execute once per step
            loop_just_started = (loop_counter == 0)
            if loop_just_started:
                # Process the action for the vehicle that filed the charging request.
                if self.active_charging_request_vehicle_id in self.vehicles:
                    vehicle = self.vehicles[self.active_charging_request_vehicle_id]
                    # The vehicle handles its action, then the reward strategy computes a penalty based on the context.
                    action_penalty = vehicle.handle_action(action, self.reward_strategy)
                    accumulated_reward += action_penalty

            # Delegate reward calculation to the reward strategy.
            temp_reward = self.reward_strategy.calculate_step_reward(
                self.vehicles, newly_arrived_ids, charging_ids)
            logger.debug(f"Step reward from simulation: {temp_reward}")
            accumulated_reward += temp_reward

            
            # Terminate only when ALL vehicles are at destination
            all_vehicles_at_destination = all(vehicle.arrived for vehicle in self.vehicles.values())
            terminated = all_vehicles_at_destination
            # Truncate (abort) when it takes too long (i.e. more than x SUMO simulation steps WITHOUT a charging request being triggered)
            truncated = self.simulation.get_current_time_step() > self.truncate_after_n_steps
            charging_request = self._check_for_charging_request(newly_spawned_ids)

            important_event_happened = charging_request or terminated or truncated
            some_simulation_time_passed = (loop_counter % self.observation_sampling_rate == 0)

            self._update_accumulated_waiting_times()

            # get observation (only every few simulation steps, for performance purposes)
            if loop_just_started or some_simulation_time_passed or important_event_happened:
                observation = self._update_and_get_observation()
                info = self._get_info()
                logger.debug(f"Intermediate simulation state: {info}")

            # note: the step for the agent ends if there is a charging request (see while loop condition) or the episode is terminated or truncated  
            if terminated or truncated:
                if terminated:
                    self._set_cumulated_waiting_time_per_episode_terminated()
                self._set_charging_stops_per_episode_mean()
                self._set_empty_vehicles_per_episode()
                self._set_cumulated_waiting_time_per_episode()
                self._set_global_ttt(self.simulation.get_current_time_step())
                final_reward = self.reward_strategy.calculate_final_reward(self.ttt_per_ev_mean)
                accumulated_reward += final_reward
                logger.info(f"Final reward: {final_reward}")
                break

            loop_counter += 1
            self.simulation.step()

        reward = accumulated_reward
        info = self._get_info()
        self._log_step_details(info, observation, reward, terminated, truncated,
                                all_vehicles_at_destination)
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

    from evaluation_algorithms import RandomAlgorithm, GreedyAlgorithm, NeverChargeAlgorithm
    import random

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

    def demo_env(random_seed=None):
        env = CircleEnv(render_mode="human", vehicles_to_spawn=10)
        random.seed(random_seed)
        observation, info = env.reset(seed=random_seed)
        env.action_space.seed(random_seed)
        model = RandomAlgorithm(env)
        # model = GreedyAlgorithm(env)

        # while env.simulation.active_vehicles_exist():
        # for _ in range(50):
        done = False
        while not done:
            action = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                # observation, info = env.reset()
                done = True
        env.close()

    def demo_single_vehicle(random_seed=None):
        env = CircleEnv(render_mode="human", vehicles_to_spawn=1)
        random.seed(random_seed)
        observation, info = env.reset(seed=random_seed)
        env.action_space.seed(random_seed)
        model = NeverChargeAlgorithm(env)
        done = False
        while not done:
            action = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
        env.close()

    configure_logging(log_file_path='testlogs/myapp.log')
    # Uncomment one of the following to test the environment:
    # test_env()
    demo_env(random_seed=1)
    # demo_single_vehicle(random_seed=1)