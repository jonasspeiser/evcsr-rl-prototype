import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque, Counter
from simulation import Simulation
from vehicle import Vehicle
from reward_strategies import BasicRewardStrategy, BasicWithCongestionPenaltyStrategy, BasicWithDestinationRewardStrategy, BasicWithChargingRewardStrategy, BasicWithShapingStrategy, NoTimeComponentRewardStrategy, RewardShapingStrategy, RelativeDestinationStrategy, BasicRelativeDestinationStrategy, RelativeDestinationWithChargingStrategy, RelativeDestinationWithIllegalPenaltyStrategy, RelativeDestinationWithCongestionStrategy, RelativeDestinationWithCongestionAndIllegalPenaltyStrategy, BasicWithShapingAndIllegalPenaltyStrategy, DestinationRewardStrategy, DestinationWithBatteryPenaltyStrategy, DestinationWithBatteryCongestionStrategy, arrive_concurrently
from noev_data_provider import Obelis_Data_Provider, Random_Data_Provider
import os

# configure logging
import logging
logger = logging.getLogger("rl.environment") 
# child logger of "rl", parent logger to "rl.environment.simulation", "rl.environment.vehicle", "rl.environment.rewards"

OBSERVATION_SPACE_SIZE = 5000 # Legacy constant — no longer used for observation space sizing. The observation space is now sized to vehicles_to_spawn (n_vehicles).
# Legacy comment: The number of vehicles in the observation space. This is used to create a fixed-size observation space for all environments. This allows for a consistent observation space size across different environment instances, even if the number of vehicles varies.
class CustomEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    OPTIONAL_OBS_FEATURES = frozenset({"simulation_time", "station_assignment_counts"})

    def __init__(self, scenario_generator, render_mode=None, reward_strategy="basic", vehicles_to_spawn=1,
                 max_vehicles=None, observation_sampling_rate=30, longest_route_duration=7200, non_observable_vehicles=None, noev_provider="obelis", random_seed=None, sumo_log_path=None, street_network="straight_100km", reward_kwargs=None, start_soc_bounds=None, obs_features=None):
        """
        Initialize the environment and simulation. Define self.observation_space and self.action_space.

        Args:
            scenario_generator (str or ScenarioGenerator): The scenario to use for the simulation. Can be a string like "bast", "all_random", or an instance of a ScenarioGenerator class.
            render_mode (str, optional): The mode in which the environment should be rendered. If None, no rendering is done. "human" shows a graphical window with the simulation.
            reward_strategy (str, optional): The version of the environment to use. Can be "basic", "noTime", or "shaping". "basic" uses the BasicRewardStrategy, which rewards the agent for reaching the destination and penalizes it for waiting. "noTime" uses the NoTimeComponentRewardStrategy, which does not consider the travel time in the reward calculation. "shaping" uses the RewardShapingStrategy, which rewards the agent for reaching the destination and penalizes it for waiting, but also considers the travel time in a more sophisticated way.
            vehicles_to_spawn (int, optional): The number of vehicles to spawn in the simulation. This is the number of observable vehicles (OEVs) that the agent can observe and control. Specifies either the total amount per simulation run or the daily maximum, depending on the scenario_generator.
            observation_sampling_rate (int, optional): The rate at which the observation is sampled (i.e. every x simulation steps).
            longest_route_duration (int, optional): The maximum duration of a route in seconds. Used for some reward strategies and episode truncation in scenarios with random data generation. If an episode in a random data scenario exceeds this duration, the episode is truncated.
            non_observable_vehicles (int, optional): The target number of non-observable vehicle charging sessions
                to inject per episode.
            noev_provider (str, optional): Which NOEV data provider to use. "obelis" uses the real OBELIS
                dataset pooled by weekday; any other value uses Random_Data_Provider. Default: "obelis".
            random_seed (int, optional): The seed for the random number generator. Used for reproducibility of the environment.
            reward_kwargs (dict, optional): Scalar parameters forwarded to the reward strategy and
                observation logic. Supported keys:
                    - "congestion_threshold_m" (int): Distance window in meters within which two vehicles
                      heading to the same station are considered to arrive concurrently. Used by
                      congestion-penalty strategies and station_assignment_counts. Default: 36000 m.
                    - "congestion_penalty" (float): Penalty budget per routing decision for congestion-
                      penalty strategies. Automatically divided by vehicles_to_spawn so the per-conflict
                      penalty scales down with fleet size. Default: 1.0.
                    - "battery_penalty_value" (float): Magnitude of the penalty when a vehicle's battery
                      runs empty. Used by RelativeDestination strategies. Default: 3.0.
                Defaults to None (all strategy parameters use their own defaults).
            obs_features (set, optional): Set of optional observation keys to include. Supported values:
                "simulation_time", "station_assignment_counts". Defaults to None (no optional features included).
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_sampling_rate = observation_sampling_rate # only sample the observation every x simulation steps for performance reasons
        if str(scenario_generator).lower() == "bast":
            self.truncate_after_n_simulation_steps = 24 * 3600 # set timeout to 24 h if BASt scenario is used
        else:
            self.truncate_after_n_simulation_steps = longest_route_duration # abort the episode if it takes too long without a charging request being triggered
        
        gui = (self.render_mode == "human")
        self.simulation = Simulation(scenario_generator=scenario_generator, gui=gui, random_seed=random_seed, sumo_log_path=sumo_log_path, street_network=street_network, start_soc_bounds=start_soc_bounds)
        self.vehicles_to_spawn = vehicles_to_spawn
        self.max_vehicles = max_vehicles if max_vehicles is not None else vehicles_to_spawn
        if self.vehicles_to_spawn > self.max_vehicles:
            raise ValueError(f"vehicles_to_spawn ({vehicles_to_spawn}) must not exceed max_vehicles ({self.max_vehicles})")

        self.non_observable_vehicles = non_observable_vehicles
        if non_observable_vehicles:
            if noev_provider == "obelis":
                self.noev_data_provider = Obelis_Data_Provider(
                    target_session_count=non_observable_vehicles,
                    seed=random_seed)
            else:
                self.noev_data_provider = Random_Data_Provider(
                    n_noevs=self.non_observable_vehicles,
                    n_cs=4,
                    max_simulation_time=self.truncate_after_n_simulation_steps,
                    seed=random_seed)

        # # Create Vehicle instances for each vehicle id
        # self.vehicles = {vid: Vehicle(vid, self.simulation) for vid in self.vehicle_ids}

        # --- Define action space ---        
        # We have 5 actions for each vehicle: do nothing (0), send charging to cs_1 (1), send charging to cs_2 (2), ...
        actions_per_vehicle = 5
        self.action_space = spaces.Discrete(actions_per_vehicle)


        # --- Define observation space ---
        # Each vehicle row has 12 features:
        #   [0]   battery_soc             normalized [0, 1]; -1 if unavailable
        #   [1]   distance_to_destination normalized [0, 1]; -1 if unavailable
        #   [2:6] distance_to_cs_1..4     normalized [0, 1]; -1 if unreachable
        #   [6:11] last_action one-hot    {0,1} x 5 (actions 0=do_nothing, 1-4=station)
        #   [11]  arrived_at_destination  {0, 1}
        # "active_vehicle": the vehicle filing the current charging request (shape (12,))
        # "other_vehicles": all other spawned vehicles, zero-padded to max_vehicles rows
        # "vehicle_mask": 1 for valid rows in other_vehicles, 0 for padding
        # "simulation_time": current simulation time normalized to [0, 1] by truncate_after_n_simulation_steps
        # "station_assignment_counts": per-station count of vehicles assigned to that station whose
        #   distance to it is within congestion_threshold_m of the active vehicle's distance (i.e.
        #   will arrive at roughly the same time). Falls back to raw assigned count if threshold is
        #   None or active vehicle has no distances yet. Normalized to [0, 1] by max_vehicles (shape (4,))
        self.simulation.add_vehicles(self.vehicles_to_spawn)
        self.vehicle_ids = self.simulation.get_all_oev_ids()
        logger.debug("Initial vehicle_ids: %s", self.vehicle_ids)

        self.obs_features = frozenset(obs_features) if obs_features is not None else frozenset()

        obs_space_dict = {
            "active_vehicle": spaces.Box(
                low=-1.0, high=1.0,
                shape=(12,), dtype=np.float32
            ),
            "other_vehicles": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.max_vehicles, 12), dtype=np.float32
            ),
            "vehicle_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_vehicles,), dtype=np.float32
            ),
        }
        if "simulation_time" in self.obs_features:
            obs_space_dict["simulation_time"] = spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32
            )
        if "station_assignment_counts" in self.obs_features:
            obs_space_dict["station_assignment_counts"] = spaces.Box(
                low=0.0, high=1.0,
                shape=(4,), dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_space_dict)

        self.episode_count = 0
        self.noev_sessions_injected = 0
        self.congestion_threshold_m = (reward_kwargs or {}).get('congestion_threshold_m')

        # Instantiate the reward strategy based on reward_strategy.
        kwargs = reward_kwargs or {}
        if reward_strategy == "basic":
            self.reward_strategy = BasicRewardStrategy()
        elif reward_strategy == "basicCongestion":
            # Normalize congestion_penalty by fleet size
            raw_penalty = kwargs.get("congestion_penalty", 1.0)
            normalized_penalty = raw_penalty / self.vehicles_to_spawn
            self.reward_strategy = BasicWithCongestionPenaltyStrategy(
                congestion_threshold_m=kwargs.get("congestion_threshold_m", 36000),
                congestion_penalty=normalized_penalty,
            )
        elif reward_strategy == "destination":
            self.reward_strategy = DestinationRewardStrategy(max_allowed_ttt=longest_route_duration)
        elif reward_strategy == "destinationBattery":
            self.reward_strategy = DestinationWithBatteryPenaltyStrategy(max_allowed_ttt=longest_route_duration)
        elif reward_strategy == "destinationBatteryCongestion":
            raw_penalty = kwargs.get("congestion_penalty", 1.0)
            self.reward_strategy = DestinationWithBatteryCongestionStrategy(
                max_allowed_ttt=longest_route_duration,
                congestion_threshold_m=kwargs.get("congestion_threshold_m", 36000),
                congestion_penalty_value=raw_penalty / self.vehicles_to_spawn,
            )
        elif reward_strategy == "basicDestination":
            self.reward_strategy = BasicWithDestinationRewardStrategy(max_allowed_ttt=longest_route_duration)
        elif reward_strategy == "basicCharging":
            self.reward_strategy = BasicWithChargingRewardStrategy()
        elif reward_strategy == "basicShaping":
            raw_penalty = kwargs.get("congestion_penalty", 1.0)
            self.reward_strategy = BasicWithShapingStrategy(
                max_allowed_ttt=longest_route_duration,
                congestion_threshold_m=kwargs.get("congestion_threshold_m", 36000),
                congestion_penalty_value=raw_penalty / self.vehicles_to_spawn,
            )
        elif reward_strategy == "basicShapingIllegal":
            raw_penalty = kwargs.get("congestion_penalty", 1.0)
            self.reward_strategy = BasicWithShapingAndIllegalPenaltyStrategy(
                max_allowed_ttt=longest_route_duration,
                congestion_threshold_m=kwargs.get("congestion_threshold_m", 36000),
                congestion_penalty_value=raw_penalty / self.vehicles_to_spawn,
            )
        elif reward_strategy == "noTime":
            self.reward_strategy = NoTimeComponentRewardStrategy()
        elif reward_strategy == "relativeDestination":
            self.reward_strategy = RelativeDestinationStrategy(
                battery_penalty_value=kwargs.get("battery_penalty_value", 3.0),
            )
        elif reward_strategy == "basicRelativeDestination":
            self.reward_strategy = BasicRelativeDestinationStrategy(
                battery_penalty_value=kwargs.get("battery_penalty_value", 3.0),
            )
        elif reward_strategy == "relativeDestinationCharging":
            self.reward_strategy = RelativeDestinationWithChargingStrategy(
                battery_penalty_value=kwargs.get("battery_penalty_value", 3.0),
            )
        elif reward_strategy == "relativeDestinationIllegal":
            self.reward_strategy = RelativeDestinationWithIllegalPenaltyStrategy(
                battery_penalty_value=kwargs.get("battery_penalty_value", 3.0),
            )
        elif reward_strategy == "relativeDestinationCongestion":
            raw_penalty = kwargs.get("congestion_penalty", 1.0)
            self.reward_strategy = RelativeDestinationWithCongestionStrategy(
                battery_penalty_value=kwargs.get("battery_penalty_value", 3.0),
                congestion_threshold_m=kwargs.get("congestion_threshold_m", 36000),
                congestion_penalty_value=raw_penalty / self.vehicles_to_spawn,
            )
        elif reward_strategy == "relativeDestinationCongestionIllegal":
            raw_penalty = kwargs.get("congestion_penalty", 1.0)
            self.reward_strategy = RelativeDestinationWithCongestionAndIllegalPenaltyStrategy(
                battery_penalty_value=kwargs.get("battery_penalty_value", 3.0),
                congestion_threshold_m=kwargs.get("congestion_threshold_m", 36000),
                congestion_penalty_value=raw_penalty / self.vehicles_to_spawn,
                illegal_penalty_value=kwargs.get("illegal_penalty_value", 0.1),
            )
        elif reward_strategy == "shaping":
            # longest_route_duration doubles as the reward upper bound: since episodes are truncated
            # at this limit, no vehicle can arrive with TTT > longest_route_duration, so the
            # arrival reward (longest_route_duration - TTT) is always non-negative.
            self.reward_strategy = RewardShapingStrategy(max_allowed_ttt=longest_route_duration)
        else:
            raise ValueError(f"Unknown reward_strategy: {reward_strategy}")


    def _set_charging_stops_per_episode_mean(self):
        """
        Calculates the global average for number of charging stops per vehicle.

        Used for tensorboard logging.
        """
        total = sum(self.charging_stops_per_episode_counter.values())
        count = len(self.charging_stops_per_episode_counter) if self.charging_stops_per_episode_counter else 1
        self.charging_stops_per_episode_mean = total / count

    def _set_realized_participation_rate(self):
        """
        Computes the realized participation rate for the episode:
        rho = OEV_sessions / (OEV_sessions + NOEV_sessions).
        OEV sessions = total charging stops across all vehicles this episode.
        NOEV sessions = injected NOEV session count (logged in _add_non_observable_vehicles).
        If both are zero (no charging occurred), sets rate to None so it is excluded from logging.
        """
        oev_sessions = sum(self.charging_stops_per_episode_counter.values())
        noev_sessions = self.noev_sessions_injected
        total = oev_sessions + noev_sessions
        if total == 0:
            self.realized_participation_rate = None
        else:
            self.realized_participation_rate = oev_sessions / total

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
        n_vehicles = len(self.vehicle_ids)
        self.cwt_per_ev_mean = self.cumulated_waiting_time / n_vehicles if n_vehicles > 0 else 0
    
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

        Used for reward calculation and tensorboard logging.
        Args:
            current_time (float): The current simulation time.
        """
        global_ttt = 0
        for vehicle in self.vehicles.values():
            if vehicle.departure_time is None:
                continue
            if vehicle.arrival_time is None:
                if vehicle.empty:
                    # Empty vehicles receive a fixed worst-case TTT equal to the longest possible travel time during an episode (departure_time + truncation_limit), independent of when the episode actually ended.
                    # This prevents a bad incentive: without this, an early-empty vehicle could get an artificially short TTT if the episode terminates before the truncation limit (because all other vehicles arrived), making "go empty" look like a good strategy.
                    vehicle.arrival_time = vehicle.departure_time + self.truncate_after_n_simulation_steps
                else:
                    # Non-empty vehicle still driving when truncation hit: use the actual episode-end time.
                    vehicle.arrival_time = current_time
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
                if vehicle.empty:
                    # Same worst-case penalty as in _set_global_ttt (see comment there).
                    vehicle.arrival_time = vehicle.departure_time + self.truncate_after_n_simulation_steps
                else:
                    # In a terminated episode all non-empty vehicles should have arrival_time set already, so this branch is only a safety fallback for unexpected states.
                    vehicle.arrival_time = current_time
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
        logger.debug("empty_vehicles_per_episode: %s", self.empty_vehicles_per_episode)

    def _set_arrival_soc_stats(self):
        """Mean SOC and remaining range (Wh / m) across vehicles that arrived at their destination."""
        arrived = [
            (self.simulation.soc_history[vid][-1], self.vehicle_range_cache.get(vid))
            for vid, v in self.vehicles.items()
            if v.arrived and self.simulation.soc_history.get(vid)
        ]
        if arrived:
            socs = [soc for soc, _ in arrived]
            ranges = [rng for _, rng in arrived if rng is not None]
            self.arrival_soc_wh_mean = sum(socs) / len(socs)
            self.arrival_range_m_mean = sum(ranges) / len(ranges) if ranges else None
        else:
            self.arrival_soc_wh_mean = None
            self.arrival_range_m_mean = None

    def _update_charging_stop_starts(self):
        """Record SOC and remaining range for vehicles beginning a charging stop this step."""
        for vid in self.simulation.get_stop_starting_vehicle_ids():
            soc = self.simulation.get_battery_soc(vid)
            remaining_range = self.simulation.get_remaining_range(vid)
            if soc is not None:
                self.charging_stop_start_records[vid] = (soc, remaining_range)

    def _set_charging_start_soc_stats(self):
        """Mean SOC and remaining range (Wh / m) across all charging stop starts in this episode."""
        all_records = list(self.charging_stop_start_records.values())
        if all_records:
            socs = [soc for soc, _ in all_records if soc is not None]
            ranges = [rng for _, rng in all_records if rng is not None]
            self.charging_start_soc_wh_mean = sum(socs) / len(socs) if socs else None
            self.charging_start_range_m_mean = sum(ranges) / len(ranges) if ranges else None
        else:
            self.charging_start_soc_wh_mean = None
            self.charging_start_range_m_mean = None


    def _get_episode_metrics(self):
        """Bundle per-episode summary stats into a plain dict for external consumers.

        Called once at episode end and stored in the step() info dict so that
        the SB3 training callback can read the values from infos[0] before the
        VecEnv auto-reset discards the episode state.
        """
        names = [
            "charging_stops_per_episode_mean",
            "noev_sessions_injected",
            "realized_participation_rate",
            "global_ttt", "global_ttt_only_terminated",
            "ttt_per_ev_mean", "ttt_per_ev_mean_only_terminated",
            "cumulated_waiting_time", "cwt_per_ev_mean", "cumulated_waiting_time_only_terminated",
            "empty_vehicles_per_episode", "final_simulation_time",
            "arrival_soc_wh_mean", "arrival_range_m_mean",
            "charging_start_soc_wh_mean", "charging_start_range_m_mean",
        ]
        return {n: getattr(self, n) for n in names if getattr(self, n, None) is not None}

    def _add_non_observable_vehicles(self):
        """
        Adds non-observable vehicles to the simulation using the data provider.
        """
        self.simulation.add_non_observable_routes()
        weekday = None
        if hasattr(self.simulation.scenario_generator, 'last_episode_date'):
            weekday = self.simulation.scenario_generator.last_episode_date.weekday()
        vehicle_data = self.noev_data_provider.get_non_observable_vehicle_data(weekday=weekday)
        for entry in vehicle_data:
            self.simulation.add_non_observable_vehicle(cs_id=entry["cs_id"],
                                                   depart_time=entry["charge_begin_seconds"],
                                                   charge_duration=entry["charge_duration"])
        self.noev_sessions_injected = len(vehicle_data)

    def _build_vehicle_observations(self):
        """Update vehicle states from simulation and build the per-vehicle observation arrays.

        Returns:
            tuple: (active_vehicle_obs, other_vehicles_obs, vehicle_mask)
        """
        active_vehicle_obs = np.zeros(12, dtype=np.float32)
        other_vehicles_obs = np.zeros((self.max_vehicles, 12), dtype=np.float32)
        vehicle_mask = np.zeros(self.max_vehicles, dtype=np.float32)
        other_idx = 0
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.arrived and not vehicle.empty:
                vehicle_state = self.simulation.get_vehicle_state(vehicle_id)
                vehicle.update_from_state(vehicle_state)
            obs = vehicle.get_observation()
            if vehicle_id == self.active_charging_request_vehicle_id:
                active_vehicle_obs = obs
            else:
                other_vehicles_obs[other_idx] = obs
                vehicle_mask[other_idx] = 1.0 if vehicle.is_online else 0.0
                other_idx += 1
        return active_vehicle_obs, other_vehicles_obs, vehicle_mask

    def _get_normalized_simulation_time(self):
        """Return the current simulation time normalized to [0, 1].

        Returns:
            np.ndarray: shape (1,)
        """
        return np.array(
            [min(1.0, self.simulation.get_current_time_step() / self.truncate_after_n_simulation_steps)],
            dtype=np.float32
        ) # Cap at 1.0 to avoid out-of-bounds values in case of simulation time exceeding truncation limit (e.g. due to observation_sampling_rate delaying episode termination).

    def _get_normalized_station_assignment_counts(self):
        """Return per-station count of vehicles that would arrive concurrently with the active vehicle.

        For each station, counts vehicles currently assigned to it whose distance to the station
        is within congestion_threshold_m of the active vehicle's distance (same arrival window).
        Falls back to raw assigned count if threshold is None or active vehicle has no distances yet.
        Normalized to [0, 1] by max_vehicles.

        Returns:
            np.ndarray: shape (4,)
        """
        cs_ids = self.simulation.get_all_charging_station_ids()
        threshold = self.congestion_threshold_m
        active_vehicle = self.vehicles.get(self.active_charging_request_vehicle_id)
        active_dists = active_vehicle.distance_to_cs_dict if (active_vehicle and active_vehicle.distance_to_cs_dict) else None
        station_counts = np.zeros(len(cs_ids), dtype=np.float32)
        for i, cs_id in enumerate(cs_ids):
            active_dist = active_dists.get(cs_id) if active_dists else None
            for vehicle in self.vehicles.values():
                if vehicle.vehicle_id == self.active_charging_request_vehicle_id:
                    continue
                if not vehicle.is_online:
                    continue
                if vehicle.target_cs_id != cs_id:
                    continue
                if threshold is not None and active_dist is not None:
                    dist_other = vehicle.distance_to_cs_dict.get(cs_id) if vehicle.distance_to_cs_dict else None
                    if dist_other is None or not arrive_concurrently(dist_other, active_dist, threshold):
                        continue
                station_counts[i] += 1
        return station_counts / self.max_vehicles

    def _update_and_get_observation(self):
        """
        Build the observation dict:
          "active_vehicle":           (12,) features of the vehicle filing the current charging request
          "other_vehicles":           (max_vehicles, 12) features of all other spawned vehicles, zero-padded
          "vehicle_mask":             (max_vehicles,) — 1 for valid rows in other_vehicles, 0 for padding
          "simulation_time":          (1,) current simulation time normalized to [0, 1]
          "station_assignment_counts": (4,) per-station concurrent-arrival count, normalized by max_vehicles

        Returns:
            dict: observation dictionary
        """
        active_vehicle_obs, other_vehicles_obs, vehicle_mask = self._build_vehicle_observations()
        obs = {
            "active_vehicle": active_vehicle_obs,
            "other_vehicles": other_vehicles_obs,
            "vehicle_mask": vehicle_mask,
        }
        if "simulation_time" in self.obs_features:
            obs["simulation_time"] = self._get_normalized_simulation_time()
        if "station_assignment_counts" in self.obs_features:
            obs["station_assignment_counts"] = self._get_normalized_station_assignment_counts()
        return obs
    
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
        just_spawned = []
        for vehicle_id in newly_spawned_ids or []:
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle and vehicle.departure_time is None:
                vehicle.departure_time = current_time
                just_spawned.append(vehicle_id)
        if just_spawned:
            logger.info("%s vehicle(s) spawned at time %s", len(just_spawned), current_time)
            logger.debug("Spawned vehicle IDs: %s", just_spawned)
        for vehicle_id in newly_arrived_ids or []:
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle:
                vehicle.arrival_time = current_time
                vehicle.arrived = True
                logger.info("%s arrived at time %s", vehicle_id, current_time)

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
            logger.info("%s: just finished charging", vid)
        # Find out if there are vehicles that just entered low battery status
        new_low_battery_ids = self._get_new_low_battery_ids(just_charged_ids)
        # Find vehicles that missed their recommended charging station (BadTimingRoutingError)
        missed_station_ids = {vid for vid, v in self.vehicles.items() if v.had_bad_timing_error}
        for vid in missed_station_ids:
            self.vehicles[vid].had_bad_timing_error = False
            logger.info("%s: missed charging station, re-queuing charging request", vid)
        charging_requests = (newly_spawned_ids or set()) | just_charged_ids | new_low_battery_ids | missed_station_ids
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
                logger.debug("Removing despawned vehicles from charging request queue: %s", removed_ids)
                
            # If the active vehicle just despawned, reset the active vehicle id
            if self.active_charging_request_vehicle_id in newly_despawned_ids:
                self.active_charging_request_vehicle_id = None
            # Remove despawned vehicles from the low battery ids
            self.low_battery_ids.difference_update(newly_despawned_ids)

        # add charging requests to the queue (if any)
        if charging_requests:
            logger.debug("New charging requests: spawned=%s, just charged=%s, low battery=%s", newly_spawned_ids, just_charged_ids, new_low_battery_ids)
            self.charging_request_queue.extend(sorted(charging_requests))
        
        # set active charging request vehicle id (if any)
        if self.charging_request_queue:
            logger.debug("Charging request queue: %s", self.charging_request_queue)
            self.active_charging_request_vehicle_id = self.charging_request_queue.popleft()
            logger.debug("Active charging request for: %s", self.active_charging_request_vehicle_id)
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
            if not vehicle.is_online:
                continue
            if vehicle_id in self.low_battery_ids and vehicle_id in just_charged_ids:
                self.low_battery_ids.remove(vehicle_id)
            # only account for vehicles that just entered the state of low battery. Not the ones that where already low during the last step.
            if vehicle.relative_battery_soc < battery_threshold and vehicle_id not in self.low_battery_ids:
                logger.info("%s: low battery", vehicle_id)
                new_low_battery_ids.add(vehicle_id)
                self.low_battery_ids.add(vehicle_id)
        return new_low_battery_ids
    
    def _collect_vehicle_status_updates(self):
        """Collect all vehicle status updates from simulation - POTENTIAL BOTTLENECK."""
        logger.debug("Simulation time step: %s", self.simulation.get_current_time_step())
        
        # Get vehicle status from simulation - these are likely expensive TraCI calls
        newly_spawned_ids = self.simulation.get_spawned_vehicle_ids()
        newly_arrived_ids = self.simulation.get_arrived_vehicle_ids()
        newly_removed_ids = self.simulation.get_removed_vehicle_ids()
        newly_despawned_ids = newly_arrived_ids | newly_removed_ids
        charging_ids = self.simulation.get_charging_vehicle_ids()
        
        if newly_despawned_ids:
            logger.debug("Despawining vehicles: arrived=%s, removed=%s", newly_arrived_ids, newly_removed_ids)

        # Update vehicles' battery soc and cache remaining range
        for vid, vehicle in self.vehicles.items():
            if vehicle.is_online:
                vehicle.fetch_and_update_battery_values()
                self.vehicle_range_cache[vid] = self.simulation.get_remaining_range(vid)

        # Mark vehicles whose battery just died — must run after SOC update and before reward/termination checks.
        # Centralised here so reward strategies don't need to call battery_just_died() for state management.
        newly_emptied_ids = set()
        for vid, vehicle in self.vehicles.items():
            if vehicle.battery_just_died():
                newly_emptied_ids.add(vid)
                logger.info("%s: battery empty", vid)

        # Update vehicles' arrival status
        for vid in newly_arrived_ids:
            if vid in self.vehicles:
                self.vehicles[vid].arrived = True

        # Update vehicle times if needed
        if newly_spawned_ids or newly_arrived_ids:
            self._update_vehicle_times(newly_spawned_ids, newly_arrived_ids)

        self._update_charging_stop_starts()

        return {
            'newly_spawned_ids': newly_spawned_ids,
            'newly_arrived_ids': newly_arrived_ids,
            'newly_removed_ids': newly_removed_ids,
            'newly_despawned_ids': newly_despawned_ids,
            'charging_ids': charging_ids,
            'newly_emptied_ids': newly_emptied_ids,
        }

    def _process_initial_action(self, action):
        """Process the action for the active charging request vehicle."""
        if self.active_charging_request_vehicle_id in self.vehicles:
            vehicle = self.vehicles[self.active_charging_request_vehicle_id]
            action_penalty = vehicle.handle_action(action, self.reward_strategy, extra_context={'all_vehicles': self.vehicles})
            return action_penalty
        return 0

    def _calculate_step_reward(self, vehicle_status):
        """Calculate reward for the current simulation step."""
        temp_reward = self.reward_strategy.calculate_step_reward(
            self.vehicles, vehicle_status['newly_arrived_ids'], vehicle_status['charging_ids'], vehicle_status['newly_emptied_ids'])
        logger.debug("Step reward from simulation: %s", temp_reward)
        return temp_reward

    def _check_termination_conditions(self):
        """Check if episode should terminate or truncate."""
        # Episode terminates when every vehicle has either arrived or gone empty.
        # Empty vehicles are considered done because they can no longer make progress;
        # their TTT penalty is handled separately in _set_global_ttt(set to departure_time + truncation_limit).
        all_vehicles_despawned = all(vehicle.arrived or vehicle.empty for vehicle in self.vehicles.values())
        terminated = all_vehicles_despawned
        truncated = self._reached_max_simulation_steps()

        return {
            'terminated': terminated,
            'truncated': truncated,
            'all_vehicles_despawned': all_vehicles_despawned
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
        logger.debug("Intermediate simulation state: %s", info)
        return observation, info

    def _handle_episode_termination(self, termination_status):
        """Handle end-of-episode calculations and logging."""
        simulation_time = self.simulation.get_current_time_step()
        
        if termination_status['terminated']:
            self._set_cumulated_waiting_time_per_episode_terminated()
            self._set_global_ttt_only_terminated(simulation_time)
            
        self._set_charging_stops_per_episode_mean()
        self._set_realized_participation_rate()
        self._set_empty_vehicles_per_episode()
        self._set_cumulated_waiting_time_per_episode()
        self._set_final_simulation_time(simulation_time)
        self._set_global_ttt(simulation_time)
        self._set_arrival_soc_stats()
        self._set_charging_start_soc_stats()
        
        final_reward = self.reward_strategy.calculate_final_reward(self.ttt_per_ev_mean)
        status_str = "terminated" if termination_status['terminated'] else "truncated"
        all_arrived = all(v.arrived for v in self.vehicles.values())
        arrived_str = ", all arrived" if all_arrived else ""
        logger.info("Episode %s ended (%s%s): reward = %.2f", self.episode_count, status_str, arrived_str, final_reward)
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
                "non_observable_vehicles": self.non_observable_vehicles,
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
        self.vehicle_ids = self.simulation.get_all_oev_ids()
        if len(self.vehicle_ids) > self.max_vehicles:
            raise ValueError(
                "Episode generated %d vehicles but max_vehicles=%d; the observation space "
                "cannot hold this many vehicles. Increase max_vehicles in your training config."
                % (len(self.vehicle_ids), self.max_vehicles)
            )
        self.episode_count += 1
        logger.info("--- Episode %s started (%s vehicles) ---", self.episode_count, len(self.vehicle_ids))
        # Set the maximum possible distance according to the currently loaded network (used for normalizing distances in the observation space).
        Vehicle.DISTANCE_NORMALIZATION_VALUE = self.simulation.get_max_possible_distance()
        # Re-create the vehicles dictionary in case new vehicles were spawned.
        self.vehicles = {vid: Vehicle(vid, self.simulation) for vid in self.vehicle_ids}

        # Additional attributes for managing charging requests and logging
        self.charging_request_queue = deque()
        self.active_charging_request_vehicle_id = None #the vehicle_id for which the agent has to select an action in the current step
        self.charging_stops_per_episode_counter = Counter({vid: 0 for vid in self.vehicle_ids})
        self.low_battery_ids = set()
        self.noev_sessions_injected = 0
        self.arrival_soc_wh_mean = None
        self.arrival_range_m_mean = None
        self.charging_start_soc_wh_mean = None
        self.charging_start_range_m_mean = None
        self.charging_stop_start_records: dict[str, tuple[float, float | None]] = {}
        self.vehicle_range_cache: dict[str, float | None] = {}  # last known remaining range (m) per vehicle
        
        if self.non_observable_vehicles:
            self._add_non_observable_vehicles()

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
        logger.debug("Reset observation: %s", observation)
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
                if info is None:
                    info = self._get_info()
                info["_episode_metrics"] = self._get_episode_metrics()
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
        
        logger.debug("State: %s, Step reward: %s", info, reward)
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
        env = CustomEnv(scenario_generator="all_random", render_mode="human", vehicles_to_spawn=3, non_observable_vehicles=5)
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