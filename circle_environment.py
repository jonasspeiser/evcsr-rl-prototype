import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque, Counter
from circle_simulation import Simulation
from data_processing import Obelis_Data_Provider

# configure logging
import logging
logger = logging.getLogger("rl.environment") # child logger of "application", parent logger to "application.environment.simulation"

# ======================================
# Reward Strategies
# ======================================
class RewardStrategy:
    def calculate_step_reward(self, vehicles, simulation, newly_arrived_ids, charging_ids):
        """
        Calculate the step reward.
        
        Parameters:
            vehicles: List of all vehicles.
            newly_arrived_ids: List of vehicle ids that just reached their destination.
            charging_ids: List of vehicle ids that are currently charging.
        
        Returns:
            A tuple (reward, one_vehicle_just_died, all_vehicles_at_destination)
        """
        raise NotImplementedError("calculate_step_reward must be implemented in subclasses.")

    def calculate_final_reward(self, vehicles, simulation):
        """
        Calculate the final reward at the end of an episode.
        
        Parameters:
            vehicles: List of all vehicles.
            
        Returns:
            A final reward value.
        """
        raise NotImplementedError("calculate_final_reward must be implemented in subclasses.")

    def calculate_action_penalty(self, vehicle, context):
        """
        Compute an action penalty based on the context.
        
        Parameters:
            vehicle: The vehicle that executed the action.
            simulation: The simulation instance.
            context: A dictionary with information about the action handling.
                     For example, it might contain:
                         - 'action': the action that was taken
                         - 'target_cs': the charging station the vehicle tried to go to
                         - 'next_charging_stop': the vehicle’s currently planned stop (if any)
                         - 'reroute_successful': whether rerouting succeeded (True/False)
                         - 'charging_stop_already_planned': whether a charging stop at the decided station was already planned in the last action
                         - 'sufficient_range': whether the vehicle has sufficient range
                         - 'rerouting_exception_occurred': whether an exception occurred during rerouting
        Returns:
            A numeric penalty (e.g., -1 for an undesired action, 0 for no penalty).
        """
        raise NotImplementedError

class BasicRewardStrategy(RewardStrategy):
    def calculate_step_reward(self, vehicles, simulation, newly_arrived_ids, charging_ids):
        """
        Calculate the step reward by iterating over all vehicles.
        A reward of -100 is given if a battery just died and +10 when a vehicle reaches its destination.
        Additionally, vehicles that are charging (and otherwise low on range) give an extra reward.
        """
        reward = 0
        one_vehicle_just_died = False
        all_vehicles_at_destination = True

        for vehicle in vehicles.values():
            vehicle_just_died = vehicle.battery_just_died()
            vehicle_is_at_destination = vehicle.arrived
            vehicle_has_just_reached_destination = vehicle_is_at_destination and (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids)
            
            if vehicle_just_died:
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST died (reward -100)")
                reward += -100
            elif vehicle_has_just_reached_destination:
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST reached destination (reward +10)")
                reward += 10

            if vehicle_just_died:
                one_vehicle_just_died = True
            if not vehicle_is_at_destination:
                all_vehicles_at_destination = False

            if vehicle.vehicle_id in charging_ids:
                # every sumo step (i.e. every second) a vehicle is charging and needs to do so to arrive at its destination, the agent gets +1 reward
                # TODO: This may be a bit much. Maybe reduce the reward to 0.1 or 0.01 as it is played out per second
                remaining_range_is_sufficient = simulation.remaining_range_is_sufficient(vehicle.vehicle_id, buffer=0)
                if not remaining_range_is_sufficient:
                    logger.debug(f"Vehicle {vehicle.vehicle_id} is charging with insufficient range (+1 reward)")
                    reward += 1

        return reward, one_vehicle_just_died, all_vehicles_at_destination

    def calculate_final_reward(self, vehicles, simulation):
        """
        Calculate the final reward (e.g. total travel time) by summing differences
        between each vehicle’s departure and arrival times.
        This reward is only given at the end of an episode.
        """
        global_ttt = 0
        for vehicle in vehicles.values():
            if vehicle.departure_time is None:
                continue
            if vehicle.arrival_time is None:
                vehicle.arrival_time = simulation.get_current_time_step()
            global_ttt += (vehicle.arrival_time - vehicle.departure_time)
        return global_ttt

    def calculate_action_penalty(self, vehicle, context):
        # Example logic: if the vehicle is asked to charge (action 1-4)
        # but has sufficient range (as determined by simulation.remaining_range_is_sufficient),
        # then apply a penalty of -1.
        action = context.get('action')
        if action in (1, 2, 3, 4):
            # If the vehicle already had this charging stop planned or rerouting failed,
            # don't apply a penalty.
            if context.get('charging_stop_already_planned', False):
                return 0
            if context.get('sufficient_range', False):
                logger.debug(f"Vehicle {vehicle.vehicle_id}: was asked to charge but has sufficient range (penalty -1)")
                return -1
            if context.get('rerouting_exception_ocurred', False):
                # penalize the agent for trying to take an illegal action (e.g. vehicle doesn't exist anymore or is past the charging station)
                logger.debug(f"Vehicle {vehicle.vehicle_id}: illegal charging action (penalty -1)")
                return -1
        return 0 # if action is "do nothing"

class RewardOnlyPreventEmpty(RewardStrategy):
    def calculate_step_reward(self, env, newly_arrived_ids, charging_ids):
        # Example: only penalize vehicles whose battery just died.
        reward = 0
        one_vehicle_just_died = False
        all_vehicles_at_destination = True

        for vehicle in env.vehicles.values():
            if vehicle.battery_just_died():
                logger.info(f"Vehicle {vehicle.vehicle_id} battery died (penalty -100)")
                reward += -100
                one_vehicle_just_died = True
            if not vehicle.arrived:
                all_vehicles_at_destination = False

        return reward, one_vehicle_just_died, all_vehicles_at_destination

    def calculate_final_reward(self, env):
        # For this strategy, you might choose a different final reward.
        return 0


class RewardShaping(RewardStrategy):
    def calculate_step_reward(self, env, newly_arrived_ids, charging_ids):
        # Example: shaped rewards that combine multiple signals.
        reward = 0
        one_vehicle_just_died = False
        all_vehicles_at_destination = True

        for vehicle in env.vehicles.values():
            if vehicle.battery_just_died():
                reward -= 50  # lesser penalty than BasicRewardStrategy
                one_vehicle_just_died = True
            if vehicle.arrived and (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids):
                reward += 5  # smaller bonus for reaching destination
            if vehicle.vehicle_id in charging_ids:
                # reward for charging if the vehicle is in danger of running empty
                remaining_range_is_sufficient = env.simulation.remaining_range_is_sufficient(vehicle.vehicle_id, buffer=0)
                if not remaining_range_is_sufficient:
                    reward += 2

            if not vehicle.arrived:
                all_vehicles_at_destination = False

        return reward, one_vehicle_just_died, all_vehicles_at_destination

    def calculate_final_reward(self, env):
        # A shaped final reward could also incorporate other metrics.
        return BasicRewardStrategy().calculate_final_reward(env)


# ======================================
# Vehicle Class
# ======================================
class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.last_action = -1
        self.arrived = False
        self.empty = False
        self.low_battery = False
        self.departure_time = None
        self.arrival_time = None
        self.waiting_time = 0
        self.battery_soc = None
        self.distance_to_cs = None  # Expected to be a dict {cs_id: distance}

    def update_from_state(self, state):
        """
        Update vehicle properties from the simulation-provided state.
        If the vehicle has not spawned yet (i.e. state is None or missing key),
        we reset to default values.
        """
        if state is None or state.get("distance_to_cs") is None:
            self.battery_soc = None
            self.distance_to_cs = None
        else:
            self.battery_soc = state.get("battery_soc")
            self.distance_to_cs = state.get("distance_to_cs")

    def get_observation(self, is_active=False):
        """
        Returns the observation for this vehicle as a NumPy array.
        Observation format:
          [normalized_soc] + 4 normalized distances + [last_action] + [destination_reached] + [active_charging_request]
        If the vehicle is not spawned yet, a padded observation is returned.
        """
        if self.distance_to_cs is None:
            # Use -1 as padding to indicate that the vehicle is not spawned.
            return np.array([-1, -1, -1, -1, -1, -1, 0, 0], dtype=np.float32)
        normalized_soc = self.battery_soc / 100000.0 if self.battery_soc is not None else -1 # 100000 Wh is considered as max. possible capacity
        # Ensure a fixed order by sorting charging station ids; pad if needed.
        distances = [self.distance_to_cs[k] / 2000.0 for k in sorted(self.distance_to_cs.keys())] # 2000 km is considered as max. possible distance
        while len(distances) < 4:
            distances.append(-1)
        # destination_reached flag (1 if arrived, 0 otherwise)
        destination_reached = int(self.arrived)
        # active charging request flag is provided via the is_active parameter
        obs = [normalized_soc] + distances[:4] + [self.last_action, destination_reached, int(is_active)]
        return np.array(obs, dtype=np.float32)

    def handle_action(self, simulation, action, reward_strategy):
        """
        Handle the action (e.g. rerouting, removing stops) and then delegate
        the penalty calculation to the reward strategy.
        Returns an action penalty (if any).
        """
        logger.debug(f"Vehicle {self.vehicle_id}: handling action {action}")
        self.last_action = action

        # Build a context dictionary with information useful for penalty calculation.
        context = {'action': action}

        if self.arrived:
            # If the vehicle already arrived, do nothing.
            return 0

        next_charging_stop = simulation.get_next_charging_stop_id(self.vehicle_id)
        charging_stop_is_planned = next_charging_stop is not None

        if action in (1, 2, 3, 4): # action is "charge"
            charging_stations = simulation.get_all_charging_station_ids()
            cs_id = charging_stations[action - 1] # action 1 means: go to cs_0 -> action-1 gives us the list index
            context['target_cs'] = cs_id
            context['next_charging_stop'] = next_charging_stop

            if cs_id == next_charging_stop:
                logger.debug(f"Vehicle {self.vehicle_id}: charging stop {cs_id} is already planned")
                context['reroute_successful'] = False
                context['charging_stop_already_planned']
                return 0
            try:
                simulation.reroute_for_charging(self.vehicle_id, cs_id)
                logger.debug(f"Vehicle {self.vehicle_id} rerouted for charging at {cs_id}")
                context['reroute_successful'] = True
                context['sufficient_range'] = simulation.remaining_range_is_sufficient(self.vehicle_id, buffer=0)    
            except ValueError as e: # if the vehicle is past the charging station and rerouting doesn't work
                logger.error(e)
                context['reroute_successful'] = False
                context['rerouting_exception_occurred'] = True


        elif action == 0: # action is "do nothing"
            if charging_stop_is_planned:
                simulation.remove_charging_stop(self.vehicle_id)
                logger.debug(f"Vehicle {self.vehicle_id}: removed planned charging stop")

        # Delegate penalty calculation to the reward strategy.
        action_penalty = reward_strategy.calculate_action_penalty(self, context)
        return action_penalty

    def is_battery_empty(self):
        return self.battery_soc is not None and self.battery_soc <= 0 # if battery_soc is None, the vehicle is not currently online

    def battery_just_died(self):
        """
        Returns True if the battery just died in this step.
        Once marked, it will not be flagged again.
        """
        if self.empty:
            return False
        if self.is_battery_empty():
            self.empty = True
            return True
        return False


# ======================================
# CircleEnv Class
# ======================================
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

        # Instantiate the reward strategy based on env_version.
        if env_version == "basic":
            self.reward_strategy = BasicRewardStrategy()
        elif env_version == "only_prevent_empty":
            self.reward_strategy = RewardOnlyPreventEmpty()
        elif env_version == "shaping":
            self.reward_strategy = RewardShaping()
        else:
            raise ValueError(f"Unknown env_version: {env_version}")

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
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle:
                vehicle.waiting_time = self.simulation.get_vehicle_waiting_time(vehicle_id)
    
    def __set_cumulated_waiting_time_per_episode(self):
        """Used for tensorboard logging. Sums up the individual waiting times to get one global value."""
        self.cumulated_waiting_time = sum(v.waiting_time for v in self.vehicles.values())
    
    def __set_cumulated_waiting_time_per_episode_terminated(self):
        """Used for tensorboard logging. Sums up the individual waiting times to get one global value. Only tracks terminated episodes (not truncated ones)"""
        self.cumulated_waiting_time_only_terminated = sum(v.waiting_time for v in self.vehicles.values() if v.arrived)

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
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle_state = simulation_state.get(vehicle_id)
            vehicle.update_from_state(vehicle_state)
            # Mark the vehicle as active if it filed the current charging request.
            is_active = (vehicle_id == self.active_charging_request_vehicle_id)
            observation[vehicle_id] = vehicle.get_observation(is_active=is_active)
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

        # Reset logging and per-episode counters
        self.charging_stops_per_episode_counter = Counter({vid: 0 for vid in self.vehicle_ids})
        for vehicle in self.vehicles.values():
            vehicle.waiting_time = 0
            vehicle.last_action = -1
            vehicle.arrived = False
            vehicle.empty = False
            vehicle.low_battery = False
            vehicle.departure_time = None
            vehicle.arrival_time = None

        self.simulation.reset()
        self.simulation.add_vehicles(self.vehicles_to_spawn)
        self.vehicle_ids = self.simulation.get_all_vehicle_ids()
        logger.debug(f"Reset vehicle_ids: {self.vehicle_ids}")
        # Re-create the vehicles dictionary in case new vehicles were spawned.
        self.vehicles = {vid: Vehicle(vid) for vid in self.vehicle_ids}
        self.active_charging_request_vehicle_id = None # states the vehicle_id for which the agent has to select an action in the current step

        # Additional attributes for managing charging requests and logging
        self.charging_request_queue = deque()
        self.active_charging_request_vehicle_id = None
        self.charging_stops_per_episode_counter = Counter()

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
        logger.debug(f"Reset observation: {observation}")
        return observation, info

    def __update_vehicle_times(self, newly_spawned_ids, newly_arrived_ids):
        """Stores the actual departure and arrival times for all vehicles which arrived at destination during the current simulation step."""
        if not (newly_spawned_ids or newly_arrived_ids):
            return
        
        current_time = self.simulation.get_current_time_step()
        for vehicle_id in newly_spawned_ids or []:
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle and vehicle.departure_time is None:
                vehicle.departure_time = current_time
                logger.debug(f"Vehicle {vehicle_id} departure time set to {current_time}")
        for vehicle_id in newly_arrived_ids or []:
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle:
                vehicle.arrival_time = current_time
                vehicle.arrived = True
                logger.debug(f"Vehicle {vehicle_id} arrival time set to {current_time}")

    def __check_for_charging_request(self, newly_spawned_ids):
        """
        Checks for charging requests based on newly spawned, just-charged,
        and low-battery vehicles. If any exist, the first in the queue becomes active.
        Returns: True if there is an active charging request, False otherwise.
        """
        # Find out if there are vehicles that just finished charging
        just_charged_ids = self.simulation.get_charging_stop_ending_vehicle_ids()
        # Find out if there are vehicles that just entered low battery status
        self.__update_low_battery_flags(just_charged_ids)
        new_low_battery_ids = self.__get_new_low_battery_ids()
        charging_requests = (newly_spawned_ids or []) + just_charged_ids + new_low_battery_ids
        # (this value is only used for logging) increase the counters for each vehicle that just stopped charging by one
        self.charging_stops_per_episode_counter.update(just_charged_ids)
        if charging_requests:
            logger.debug(f"New charging requests: spawned={newly_spawned_ids}, just charged={just_charged_ids}, low battery={new_low_battery_ids}")
            self.charging_request_queue.extend(charging_requests)
        if self.charging_request_queue:
            self.active_charging_request_vehicle_id = self.charging_request_queue.popleft()
            logger.debug(f"Active charging request: {self.active_charging_request_vehicle_id}")
            return True
        return False

    def __update_low_battery_flags(self, just_charged_ids):
        for vehicle in self.vehicles.values():
            if vehicle.vehicle_id in just_charged_ids:
                vehicle.low_battery = False

    def __get_new_low_battery_ids(self):
        battery_threshold = 0.2 # the value under which the battery soc should be considered low
        new_low_battery_ids = []
        state = self.simulation.get_state()
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle_state = state.get(vehicle_id)
            if vehicle_state is None:
                continue
            max_battery_capacity = vehicle_state.get("max_battery_capacity", 100000)
            battery_soc = vehicle_state.get("battery_soc")
            if battery_soc is None: # i.e. if the vehicle did not spawn in the simulation yet or despawned already
                continue
            relative_battery_soc = battery_soc / max_battery_capacity
            # only account for vehicles that just entered the state of low battery. Not the ones that where already low during the last step.
            if relative_battery_soc < battery_threshold and not vehicle.low_battery:
                new_low_battery_ids.append(vehicle_id)
                vehicle.low_battery = True
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

            # Update vehicles’ arrival status.
            for vid in newly_arrived_ids:
                if vid in self.vehicles:
                    self.vehicles[vid].arrived = True

            if newly_spawned_ids or newly_arrived_ids:
                self.__update_vehicle_times(newly_spawned_ids, newly_arrived_ids)

            # only execute once per step
            loop_just_started = (loop_counter == 0)
            if loop_just_started:
                # Process the action for the vehicle that filed the charging request.
                if self.active_charging_request_vehicle_id in self.vehicles:
                    vehicle = self.vehicles[self.active_charging_request_vehicle_id]
                    # The vehicle handles its action, then the reward strategy computes a penalty based on the context.
                    action_penalty = vehicle.handle_action(self.simulation, action, self.reward_strategy)
                    accumulated_reward += action_penalty

            # Delegate reward calculation to the reward strategy.
            temp_reward, one_vehicle_just_died, all_vehicles_at_destination = \
                self.reward_strategy.calculate_step_reward(self.vehicles, self.simulation, newly_arrived_ids, charging_ids)
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
                final_reward = self.reward_strategy.calculate_final_reward(self)
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