import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque, Counter
from simulation import Simulation
from vehicle import Vehicle, get_padded_observation
from reward_strategies import BasicRewardStrategy, NoTimeComponentRewardStrategy, RewardShapingStrategy
from nmev_data_provider import Obelis_Data_Provider, Random_Data_Provider
import os

# configure logging
import logging
logger = logging.getLogger("rl.environment") 
# child logger of "rl", parent logger to "rl.environment.simulation", "rl.environment.vehicle", "rl.environment.rewards"

OBSERVATION_SPACE_SIZE = 5000 # The number of vehicles in the observation space. This is used to create a fixed-size observation space for all environments.
# This allows for a consistent observation space size across different environment instances, even if the number of vehicles varies.
class CustomEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, scenario_generator, render_mode=None, reward_strategy="basic", vehicles_to_spawn=1,
                 observation_sampling_rate=30, truncate_after_n_steps=3000, non_member_vehicles=None, random_seed=None):
        """
        Initialize the environment and simulation. Define self.observation_space and self.action_space.

        Args:
            scenario_generator (str or ScenarioGenerator): The scenario to use for the simulation. Can be a string like "bast", "all_random", or an instance of a ScenarioGenerator class.
            render_mode (str, optional): The mode in which the environment should be rendered. If None, no rendering is done. "human" shows a graphical window with the simulation.
            reward_strategy (str, optional): The version of the environment to use. Can be "basic", "noTime", or "shaping". "basic" uses the BasicRewardStrategy, which rewards the agent for reaching the destination and penalizes it for waiting. "noTime" uses the NoTimeComponentRewardStrategy, which does not consider the travel time in the reward calculation. "shaping" uses the RewardShapingStrategy, which rewards the agent for reaching the destination and penalizes it for waiting, but also considers the travel time in a more sophisticated way.
            vehicles_to_spawn (int, optional): The number of vehicles to spawn in the simulation. This is the number of member vehicles (MEVs) that the agent can observe and control. Specifies either the total amount per simulation run or the daily maximum, depending on the scenario_generator.
            observation_sampling_rate (int, optional): The rate at which the observation is sampled (i.e. every x simulation steps).
            truncate_after_n_steps (int, optional): The number of simulation steps after which the episode is truncated if no charging request is triggered.
            non_member_vehicles (int, optional): The number of non-member vehicles (i.e. not observable by the agent) to spawn in the simulation.
            random_seed (int, optional): The seed for the random number generator. Used for reproducibility of the environment.
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_sampling_rate = observation_sampling_rate # only sample the observation every x simulation steps for performance reasons
        if str(scenario_generator).lower() == "bast":
            self.truncate_after_n_simulation_steps = 24 * 3600 # set timeout to 24 h if BASt scenario is used
        else:
            self.truncate_after_n_simulation_steps = truncate_after_n_steps # abort the episode if it takes too long without a charging request being triggered
        
        gui = (self.render_mode == "human")
        self.simulation = Simulation(scenario_generator=scenario_generator, gui=gui, random_seed=random_seed)
        self.vehicles_to_spawn = vehicles_to_spawn

        self.non_member_vehicles = non_member_vehicles
        if non_member_vehicles:
            # self.nmev_data_provider = Obelis_Data_Provider()
            self.nmev_data_provider = Random_Data_Provider(n_nmevs=self.non_member_vehicles, n_cs=4, max_simulation_time=truncate_after_n_steps)

        # # Create Vehicle instances for each vehicle id
        # self.vehicles = {vid: Vehicle(vid, self.simulation) for vid in self.vehicle_ids}

        # --- Define action space ---        
        # We have 5 actions for each vehicle: do nothing (0), send charging to cs_0 (1), send charging to cs_1 (2), ...
        actions_per_vehicle = 5
        self.action_space = spaces.Discrete(actions_per_vehicle)


        # --- Define observation space ---
        # Types of observations per vehicle: [0] current state of the battery, [1] current distance to destination, [2:6] current distance to each charging station, [6] last selected action, [7] arrived at destination, [8] request pending
        # battery soc is in between 0 and 100000 Wh, distance to destination and distance to each of the charging stations in meters is between 0 and the maximum drivable distance in the simulated network.
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
        self.vehicle_ids = self.simulation.get_all_mev_ids()
        logger.debug(f"Initial vehicle_ids: {self.vehicle_ids}")

        self.observation_space_ids = [f"member_ev_{i}" for i in range(OBSERVATION_SPACE_SIZE)] # used to create a fixed-size observation space for all environments, even if the number of vehicles varies
        self.observation_space = spaces.Dict({
            vehicle_id: single_vehicle_observation_space for vehicle_id in self.observation_space_ids
        })

        # Instantiate the reward strategy based on reward_strategy.
        if reward_strategy == "basic":
            self.reward_strategy = BasicRewardStrategy()
        elif reward_strategy == "noTime":
            self.reward_strategy = NoTimeComponentRewardStrategy()
        elif reward_strategy == "shaping":
            self.reward_strategy = RewardShapingStrategy()
        else:
            raise ValueError(f"Unknown reward_strategy: {reward_strategy}")

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
        """
        Calculates the global average for number of charging stops per vehicle.

        Used for tensorboard logging.
        """
        total = sum(self.charging_stops_per_episode_counter.values())
        count = len(self.charging_stops_per_episode_counter) if self.charging_stops_per_episode_counter else 1
        self.charging_stops_per_episode_mean = total / count

    def _update_accumulated_waiting_times(self):
        """
        Get the waiting times for each vehicle out of the simulation.

        This is only possible as long as a vehicle is still online.
        """
        for vehicle_id in self.simulation.get_online_vehicle_ids():
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle:
                vehicle.waiting_time = self.simulation.get_vehicle_waiting_time(vehicle_id)
    
    def _set_cumulated_waiting_time_per_episode(self):
        """
        Sums up the individual waiting times to get one global value.

        Used for tensorboard logging.
        """
        self.cumulated_waiting_time = sum(v.waiting_time for v in self.vehicles.values())
    
    def _set_cumulated_waiting_time_per_episode_terminated(self):
        """
        Sums up the individual waiting times to get one global value.

        Only tracks terminated episodes (not truncated ones). Used for tensorboard logging.
        """
        self.cumulated_waiting_time_only_terminated = sum(v.waiting_time for v in self.vehicles.values() if v.arrived)

    def _set_final_simulation_time(self, current_time):
        """
        Logs the last simulation time before the episode ended.

        Used for tensorboard logging.
        Args:
            current_time (float): The last simulation time before the episode ended.
        """
        self.final_simulation_time = current_time

    def _set_global_ttt(self, current_time):
        """
        Sums up the total travel times of all vehicles.

        Used for tensorboard logging.
        Args:
            current_time (float): The current simulation time.
        """
        global_ttt = 0
        for vehicle in self.vehicles.values():
            if vehicle.departure_time is None:
                continue
            if vehicle.arrival_time is None:
                vehicle.arrival_time = current_time # in case the episode was truncated, some vehicles never arrive. For these, we set the arrival time to the last step in the truncated episode, so that their travel time also counts into the global counter. These are often vehicles which stand are stuck and therefore have a long travel time already.
            global_ttt += vehicle.get_total_travel_time()
        self.global_ttt = global_ttt
        if len(self.vehicle_ids) == 0: # if no vehicles where spawned, avoid division by zero
            self.ttt_per_ev_mean = 0
        else:
            self.ttt_per_ev_mean = global_ttt / len(self.vehicle_ids)

    def _set_global_ttt_only_terminated(self, current_time):
        """
        Sums up the total travel times of all vehicles. Only tracks terminated episodes (not truncated ones).

        Used for tensorboard logging.
        Args:
            current_time (float): The current simulation time.
        """
        global_ttt_only_terminated = 0
        for vehicle in self.vehicles.values():
            if vehicle.departure_time is None:
                continue
            if vehicle.arrival_time is None:
                vehicle.arrival_time = current_time # in case the episode was truncated, some vehicles never arrive. For these, we set the arrival time to the last step in the truncated episode, so that their travel time also counts into the global counter. These are often vehicles which stand are stuck and therefore have a long travel time already.
            global_ttt_only_terminated += vehicle.get_total_travel_time()
        self.global_ttt_only_terminated = global_ttt_only_terminated
        if len(self.vehicle_ids) == 0: # if no vehicles where spawned, avoid division by zero
            self.ttt_per_ev_mean_only_terminated = 0
        else:
            self.ttt_per_ev_mean_only_terminated = global_ttt_only_terminated / len(self.vehicle_ids)

    def _set_empty_vehicles_per_episode(self):
        """
        Calculates the number of empty vehicles per episode.
        """
        self.empty_vehicles_per_episode = sum(
            1 for vehicle in self.vehicles.values()
            if vehicle.empty
        )
        logger.debug(f"empty_vehicles_per_episode: {self.empty_vehicles_per_episode}")


    def _add_non_member_vehicles(self):
        """
        Adds non-member vehicles to the simulation using the data provider.
        """
        self.simulation.add_non_member_routes()
        vehicle_data = self.nmev_data_provider.get_non_member_vehicle_data()
        for entry in vehicle_data:
            self.simulation.add_non_member_vehicle(cs_id=entry["cs_id"],
                                                   depart_time=entry["charge_begin_seconds"],
                                                   charge_duration=entry["charge_duration"])

    def _update_and_get_observation(self):
        """
        Build the observation dictionary by updating and collecting each vehicle's observation.

        Returns:
            dict: The observation dictionary for all vehicles.
        """
        observation = {}
        # Actual vehicle state updates
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.arrived and not vehicle.empty:
                vehicle_state = self.simulation.get_vehicle_state(vehicle_id)
                vehicle.update_from_state(vehicle_state)
            # Mark the vehicle as active if it filed the current charging request.
            is_active = (vehicle_id == self.active_charging_request_vehicle_id)
            observation[vehicle_id] = vehicle.get_observation(is_active=is_active)
        # Pad the the remaining observation space to ensure it has a fixed size.
        for vehicle_id in self.observation_space_ids:
            if vehicle_id not in observation:
                observation[vehicle_id] = get_padded_observation()
        return observation
    
    def _get_info(self):
        """
        Get additional info for logging, such as a human readable version of the simulation state.

        Returns:
            dict: Additional info for each vehicle.
        """
        info = {}
        for vehicle_id, vehicle in self.vehicles.items():
            info[vehicle_id] = vehicle.get_info()
        return info

    def _reached_max_simulation_steps(self):
        return self.simulation.get_current_time_step() > self.truncate_after_n_simulation_steps

    def _update_vehicle_times(self, newly_spawned_ids, newly_arrived_ids):
        """
        Stores the actual departure and arrival times for all vehicles which arrived at destination during the current simulation step.

        Args:
            newly_spawned_ids (list): List of newly spawned vehicle IDs.
            newly_arrived_ids (list): List of newly arrived vehicle IDs.
        """
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

    def _check_for_charging_request(self, newly_spawned_ids, newly_despawned_ids=None):
        """
        Removes despawned vehicles from the charging request queue and checks for new charging requests.
        Checks for charging requests based on newly spawned, just-charged, and low-battery vehicles.
        If any exist, the first in the queue becomes active.

        Args:
            newly_spawned_ids (set): Set of newly spawned vehicle IDs.
            newly_despawned_ids (set): Set of vehicle IDs that have just despawned.

        Returns:
            bool: True if there is an active charging request, False otherwise.
        """
        # Find out if there are vehicles that just finished charging
        just_charged_ids = self.simulation.get_charging_stop_ending_vehicle_ids()
        for vid in just_charged_ids:
            logger.info(f"{vid}: just finished charging")
        # Find out if there are vehicles that just entered low battery status
        new_low_battery_ids = self._get_new_low_battery_ids(just_charged_ids)
        charging_requests = (newly_spawned_ids or set()) | just_charged_ids | new_low_battery_ids
        # (this value is only used for logging) increase the counters for each vehicle that just stopped charging by one
        self.charging_stops_per_episode_counter.update(just_charged_ids)
        
        # Remove despawned vehicles from the charging request queue and update the low battery ids
        if newly_despawned_ids:
            new_queue = deque()
            removed_ids = []
            for vid in self.charging_request_queue:
                if vid in newly_despawned_ids:
                    removed_ids.append(vid)
                else:
                    new_queue.append(vid)
            self.charging_request_queue = new_queue
            if removed_ids:
                logger.debug(f"Removing despawned vehicles from charging request queue: {removed_ids}")
                
            # If the active vehicle just despawned, reset the active vehicle id
            if self.active_charging_request_vehicle_id in newly_despawned_ids:
                self.active_charging_request_vehicle_id = None
            # Remove despawned vehicles from the low battery ids
            self.low_battery_ids.difference_update(newly_despawned_ids)

        # add charging requests to the queue (if any)
        if charging_requests:
            logger.debug(f"New charging requests: spawned={newly_spawned_ids}, just charged={just_charged_ids}, low battery={new_low_battery_ids}")
            self.charging_request_queue.extend(charging_requests)
        
        # set active charging request vehicle id (if any)
        if self.charging_request_queue:
            logger.debug(f"Charging request queue: {self.charging_request_queue}")
            self.active_charging_request_vehicle_id = self.charging_request_queue.popleft()
            logger.info(f"Active charging request for: {self.active_charging_request_vehicle_id}")
            return True
        
        return False

    def _get_new_low_battery_ids(self, just_charged_ids):
        """
        Get the IDs of vehicles that just entered low battery status.

        Args:
            just_charged_ids (set): Set of vehicle IDs that just finished charging.

        Returns:
            set: Set of new low battery vehicle IDs.
        """
        battery_threshold = 0.2 # the value under which the battery soc should be considered low
        new_low_battery_ids = set()
        for vehicle_id, vehicle in self.vehicles.items():
            # Ignore if the vehicle did not spawn in the simulation yet or despawned already
            if not vehicle.spawned or vehicle.arrived or vehicle.empty:
                continue
            if vehicle_id in self.low_battery_ids and vehicle_id in just_charged_ids:
                self.low_battery_ids.remove(vehicle_id)
            # only account for vehicles that just entered the state of low battery. Not the ones that where already low during the last step.
            if vehicle.relative_battery_soc < battery_threshold and vehicle_id not in self.low_battery_ids:
                logger.info(f"{vehicle_id}: low battery")
                new_low_battery_ids.add(vehicle_id)
                self.low_battery_ids.add(vehicle_id)
        return new_low_battery_ids
    
    def _collect_vehicle_status_updates(self):
        """Collect all vehicle status updates from simulation - POTENTIAL BOTTLENECK."""
        logger.debug(f"Simulation time step: {self.simulation.get_current_time_step()}")
        
        # Get vehicle status from simulation - these are likely expensive TraCI calls
        newly_spawned_ids = self.simulation.get_spawned_vehicle_ids()
        newly_arrived_ids = self.simulation.get_arrived_vehicle_ids()
        newly_removed_ids = self.simulation.get_removed_vehicle_ids()
        newly_despawned_ids = newly_arrived_ids | newly_removed_ids
        charging_ids = self.simulation.get_charging_vehicle_ids()
        
        if newly_despawned_ids:
            logger.debug(f"Despawining vehicles: arrived={newly_arrived_ids}, removed={newly_removed_ids}")

        # Update vehicles' arrival status
        for vid in newly_arrived_ids:
            if vid in self.vehicles:
                self.vehicles[vid].arrived = True

        # Update vehicles' battery soc
        for vid, vehicle in self.vehicles.items():
            if not vehicle.arrived and not vehicle.empty:
                vehicle.fetch_and_update_battery_values()

        # Update vehicle times if needed
        if newly_spawned_ids or newly_arrived_ids:
            self._update_vehicle_times(newly_spawned_ids, newly_arrived_ids)

        return {
            'newly_spawned_ids': newly_spawned_ids,
            'newly_arrived_ids': newly_arrived_ids,
            'newly_removed_ids': newly_removed_ids,
            'newly_despawned_ids': newly_despawned_ids,
            'charging_ids': charging_ids
        }

    def _process_initial_action(self, action):
        """Process the action for the active charging request vehicle."""
        if self.active_charging_request_vehicle_id in self.vehicles:
            vehicle = self.vehicles[self.active_charging_request_vehicle_id]
            action_penalty = vehicle.handle_action(action, self.reward_strategy)
            return action_penalty
        return 0

    def _calculate_step_reward(self, vehicle_status):
        """Calculate reward for the current simulation step."""
        temp_reward = self.reward_strategy.calculate_step_reward(
            self.vehicles, vehicle_status['newly_arrived_ids'], vehicle_status['charging_ids'])
        logger.debug(f"Step reward from simulation: {temp_reward}")
        return temp_reward

    def _check_termination_conditions(self):
        """Check if episode should terminate or truncate."""
        all_vehicles_at_destination = all(vehicle.arrived for vehicle in self.vehicles.values())
        terminated = all_vehicles_at_destination
        truncated = self._reached_max_simulation_steps()
        
        return {
            'terminated': terminated,
            'truncated': truncated,
            'all_vehicles_at_destination': all_vehicles_at_destination
        }

    def _should_update_observation(self, loop_counter, charging_request, termination_status):
        """Determine if observation should be updated this step."""
        loop_just_started = (loop_counter == 0)
        important_event_happened = charging_request or termination_status['terminated'] or termination_status['truncated']
        some_simulation_time_passed = (loop_counter % self.observation_sampling_rate == 0)
        
        return loop_just_started or some_simulation_time_passed or important_event_happened

    def _update_observation_and_info(self):
        """Update observation and info."""
        self._update_accumulated_waiting_times()
        observation = self._update_and_get_observation()
        info = self._get_info()
        logger.debug(f"Intermediate simulation state: {info}")
        return observation, info

    def _handle_episode_termination(self, termination_status):
        """Handle end-of-episode calculations and logging."""
        simulation_time = self.simulation.get_current_time_step()
        
        if termination_status['terminated']:
            self._set_cumulated_waiting_time_per_episode_terminated()
            self._set_global_ttt_only_terminated(simulation_time)
            
        self._set_charging_stops_per_episode_mean()
        self._set_empty_vehicles_per_episode()
        self._set_cumulated_waiting_time_per_episode()
        self._set_final_simulation_time(simulation_time)
        self._set_global_ttt(simulation_time)
        
        final_reward = self.reward_strategy.calculate_final_reward(self.ttt_per_ev_mean)
        logger.info(f"Final reward: {final_reward}")
        return final_reward
        
    def get_snapshot(self, *, max_vehicles: int = 200) -> dict:
        """
        Minimal, JSON-serializable snapshot of env state for crash debugging.
        Aim: SMALL and high-signal (NO huge routes, raw SUMO objects, etc.)
        """
        simulation_step = None
        try:
            simulation_step = self.simulation.get_current_time_step()
        except Exception:
            pass

        vehicles_summary = {}
        try:
            items = list(self.vehicles.items())[:max_vehicles]
            for vid, v in items:
                vehicles_summary[str(vid)] = {
                    "spawned": bool(getattr(v, "spawned", False)),
                    "arrived": bool(getattr(v, "arrived", False)),
                    "empty": bool(getattr(v, "empty", False)),
                    "battery_soc": getattr(v, "battery_soc", None),
                    "relative_battery_soc": getattr(v, "relative_battery_soc", None),
                    "waiting_time": getattr(v, "waiting_time", None),
                    "departure_time": getattr(v, "departure_time", None),
                    "arrival_time": getattr(v, "arrival_time", None),
                }
        except Exception:
            vehicles_summary = {"error": "failed to summarize vehicles"}

        open_charging_requests = []
        try:
            open_charging_requests = list(self.charging_request_queue)
        except Exception:
            pass

        snap = {
            "env": {
                "class": type(self).__name__,
                "render_mode": self.render_mode,
                "reward_strategy": type(self.reward_strategy).__name__,
                "vehicles_to_spawn": self.vehicles_to_spawn,
                "non_member_vehicles": self.non_member_vehicles,
                "observation_sampling_rate": self.observation_sampling_rate,
                "truncate_after_n_simulation_steps": self.truncate_after_n_simulation_steps,
            },
            "simulation": {
                "sim_step": simulation_step,
            },
            "charging": {
                "active_request_vehicle_id": self.active_charging_request_vehicle_id,
                "queue_len": len(open_charging_requests),
                "queue_head": open_charging_requests[0] if open_charging_requests else None,
                "queue_tail": open_charging_requests[-1] if open_charging_requests else None,
            },
            "vehicles": vehicles_summary,
            "counters": {
                "charging_stops_per_episode_counter": dict(getattr(self, "charging_stops_per_episode_counter", {})),
                "low_battery_ids": list(getattr(self, "low_battery_ids", [])),
            }
        }
        return snap

    def reset(self, seed=None, options=None): # Later: add possibility to set seed by passing it as an argument `env.reset(seed=<desired seed>)`
        """
        Reset the environment and simulation for a new episode.

        Args:
            seed (int, optional): The random seed to use for reproducibility.
            options (dict, optional): Additional options for reset.

        Returns:
            tuple: The observation of the initial state and additional info.      
        """
        logger.debug("Resetting environment")
        super().reset(seed=seed) # needed for api compliance
        
        if seed:
            self.action_space.seed(seed) # for deterministic results when using env.actions_space.sample()
        self.simulation.reset()
        self.simulation.add_vehicles(amount=self.vehicles_to_spawn)
        self.vehicle_ids = self.simulation.get_all_mev_ids()
        logger.debug(f"Reset vehicle_ids: {self.vehicle_ids}")
        # Set the maximum possible distance according to the currently loaded network (used for normalizing distances in the observation space).
        Vehicle.MAX_POSSIBLE_DISTANCE = self.simulation.get_max_possible_distance()
        # Re-create the vehicles dictionary in case new vehicles were spawned.
        self.vehicles = {vid: Vehicle(vid, self.simulation) for vid in self.vehicle_ids}

        # Additional attributes for managing charging requests and logging
        self.charging_request_queue = deque()
        self.active_charging_request_vehicle_id = None #the vehicle_id for which the agent has to select an action in the current step
        self.charging_stops_per_episode_counter = Counter({vid: 0 for vid in self.vehicle_ids})
        self.low_battery_ids = set()
        
        if self.non_member_vehicles:
            self._add_non_member_vehicles()

        # Wait until a charging request is generated.
        while self.active_charging_request_vehicle_id is None:
            newly_spawned_ids = self.simulation.get_spawned_vehicle_ids()
            self._update_vehicle_times(newly_spawned_ids, newly_arrived_ids=None)
            self._check_for_charging_request(newly_spawned_ids)
            self.simulation.step()
            if self._reached_max_simulation_steps():
                logger.warning("Reached maximum simulation steps without a charging request being generated. Truncating episode.")
                break
        
        observation = self._update_and_get_observation()
        info = self._get_info()
        logger.debug(f"Reset observation: {observation}")
        return observation, info

    def step(self, action):
        """
        Executes simulation steps until the next charging request (or termination/truncation).

        Args:
            action (int): The action to take for the active charging request vehicle.

        Returns:
            tuple: The next observation, reward, terminated flag, truncated flag, and additional info.
        """
        charging_request = False
        accumulated_reward = 0
        loop_counter = 0
        observation = None
        info = None
        termination_status = None

        # loop through sumo-steps until the next vehicle goes online or a vehicle just finished charging
        while not charging_request:
            # Collect vehicle status updates from simulation
            vehicle_status = self._collect_vehicle_status_updates()

            # Process initial action (only on first loop iteration)
            if loop_counter == 0:
                accumulated_reward += self._process_initial_action(action)

            # Calculate step reward from reward strategy
            accumulated_reward += self._calculate_step_reward(vehicle_status)

            # Check termination conditions
            termination_status = self._check_termination_conditions()

            # Check for new charging requests
            charging_request = self._check_for_charging_request(
                vehicle_status['newly_spawned_ids'], 
                vehicle_status['newly_despawned_ids']
            )

            # Update observations and info (only when needed for performance)
            if self._should_update_observation(loop_counter, charging_request, termination_status):
                observation, info = self._update_observation_and_info()

            # Handle episode termination
            if termination_status['terminated'] or termination_status['truncated']:
                accumulated_reward += self._handle_episode_termination(termination_status)
                break
            
            # Advance simulation and handle despawned active vehicles
            loop_counter += 1
            self.simulation.step()
            if self.active_charging_request_vehicle_id in vehicle_status['newly_despawned_ids']:
                charging_request = False
        
        # Final setup before returning
        reward = accumulated_reward
        if info is None:
            info = self._get_info()
        
        self._log_step_details(info, observation, reward, 
                              termination_status['terminated'], 
                              termination_status['truncated'],
                              termination_status['all_vehicles_at_destination'])
        return observation, reward, termination_status['terminated'], termination_status['truncated'], info

    def render(self, mode='human'):
        """
        Show the current environment state, e.g., the graphical window in 'CartPole-v1'.

        Args:
            mode (str): The render mode. Default is 'human'.

        Returns:
            None
        """
        logger.debug("Render called")
        pass

    def close(self):
        """
        Cleanup all resources (threads, graphical windows, etc).

        Returns:
            None
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
        """
        Configure logging for the application.

        Args:
            log_file_path (str): Path to the log file.
        """
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        # create file handler which logs even debug messages
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        # add the handlers to the (root) logger
        logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])

    def test_env():
        """
        Test the CustomEnv environment using the stable_baselines3 environment checker.
        """
        from stable_baselines3.common.env_checker import check_env
        env = CustomEnv(scenario_generator="all_random")
        check_env(env, skip_render_check=True)
        print("CHECKS PASSED")
        env.close()

    def demo_env(random_seed=None):
        """
        Run a demo of the environment with a random or greedy algorithm.

        Args:
            random_seed (int, optional): The random seed for reproducibility.
        """
        env = CustomEnv(scenario_generator="all_random", render_mode="human", vehicles_to_spawn=3, non_member_vehicles=5)
        # random.seed(random_seed)
        observation, info = env.reset(seed=random_seed)
        env.action_space.seed(random_seed)
        model = RandomAlgorithm(env)
        # model = GreedyAlgorithm(env)

        # while env.simulation.active_vehicles_exist():
        # for _ in range(50):
        done = False
        while not done:
            action = model.predict(observation, False)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                # observation, info = env.reset()
                done = True
        env.close()

    def demo_single_vehicle(random_seed=None):
        """
        Run a demo of the environment with a single vehicle and the NeverChargeAlgorithm.

        Args:
            random_seed (int, optional): The random seed for reproducibility.
        """
        env = CustomEnv(scenario_generator="all_random", render_mode="human", vehicles_to_spawn=1)
        # random.seed(random_seed)
        observation, info = env.reset(seed=random_seed)
        env.action_space.seed(random_seed)
        model = NeverChargeAlgorithm(env)
        done = False
        while not done:
            action = model.predict(observation, False)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
        env.close()

    configure_logging(log_file_path='testlogs/myapp.log')
    # Uncomment one of the following to test the environment:
    test_env()
    # demo_env(random_seed=1)
    # demo_single_vehicle(random_seed=1)