import json
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
from sumolib import checkBinary
from collections import Counter
from oev_scenario_generator import ScenarioGenerator, SameRouteScenario, CustomDistributionScenario, BAStDistributionScenario
import network_generator 
traci = None # Backend selection on first Simulation construction.
tc = None

def _select_backend(gui: bool):
    """Choose libsumo (fast, headless) or traci (supports sumo-gui). Process-wide."""
    global traci, tc
    if traci is not None:
        if gui and traci.__name__ == "libsumo":
            raise RuntimeError(
                "render_mode='human' requires TraCI, but libsumo is already active "
                "in this process. Run GUI evaluations in a separate process, "
                "or set USE_LIBSUMO=0."
            )
        return
    use_libsumo = (not gui) and os.environ.get("USE_LIBSUMO", "1") == "1"
    import importlib
    traci = importlib.import_module("libsumo" if use_libsumo else "traci")
    tc = traci.constants

# configure logging
import logging
logger = logging.getLogger("rl.environment.simulation") # child logger of "application.environment"

# define color values for vehicles in the GUI
BLUE = [153, 255, 255]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
RED = [255, 0, 0]

DEFAULT_STREET_NETWORK = "straight_100km"
CHARGING_DURATION = 1300
"""charging duration in seconds — charges from near-empty to ~80% of 64 kWh (51,200 Wh) in ~1293 s at 150 kW / 0.95 efficiency"""
EMPTY_SOC = 30 
"""value under which the battery should be considered empty by the simulation. This is set lower than the value for the environment because the simulation brings the vehicle to a standstill under this value, meaning that it will recuperate some energy (20-30 Wh) in the process."""

class RoutingError(Exception):
    """
    Base class for routing errors in the simulation.

    Args:
        message (str): The error message.
    """
    def __init__(self, message):
        super().__init__(message)

class PointlessRecommendationError(RoutingError):
    """
    Raised when a vehicle's destination is closer than the recommended charging station.

    Args:
        dist_dest (float): Distance to destination.
        dist_cs (float): Distance to charging station.
    """
    def __init__(self, message):
        super().__init__(message)

class ImpossibleRoutingError(RoutingError):
    """
    Raised when during route calculation, no route can be found for a vehicle to a charging station.

    Args:
        message (str): The error message.
    """
    def __init__(self, message):
        super().__init__(message)

class BadTimingRoutingError(RoutingError):
    """
    Raised when setting a valid charging stop but the vehicle is already on the cs edge but past the charging station, thus can't be rerouted.

    Args:
        message (str): The error message.
    """
    def __init__(self, message):
        super().__init__(message)

def construct_scenario_generator(scenario_generator, random_seed=None):
    """
    Constructs a scenario generator based on the provided scenario_generator argument.

    Args:
        scenario_generator (str or ScenarioGenerator): The type of scenario generator to create.
        random_seed (int, optional): The random seed to use for the scenario generator.

    Returns:
        ScenarioGenerator: An instance of the specified scenario generator.
    """
    if isinstance(scenario_generator, ScenarioGenerator):
        return scenario_generator
    match str(scenario_generator).lower():
        case "all_random":
            return ScenarioGenerator(seed=random_seed)
        case "same_route":
            return SameRouteScenario(seed=random_seed)
        case "custom_distribution":
            return CustomDistributionScenario(seed=random_seed)
        case "bast":
            return BAStDistributionScenario(seed=random_seed)
        case _:
            raise ValueError(f"Unknown scenario generator: {scenario_generator}")

class Simulation():

    def __init__(self, scenario_generator, gui:bool=False, random_seed = None, sumo_log_path = None, street_network = DEFAULT_STREET_NETWORK, start_soc_bounds = None):
        _select_backend(gui)
        self.sumo_config_stub = f"./street-networks/{street_network}/{street_network}"
        self.start_soc_bounds = start_soc_bounds

        with open(f"{self.sumo_config_stub}.all_distances.json", "r") as f:
            self.all_distances = json.load(f)

        if type(gui) is not bool:
            raise ValueError("gui must be a boolean")
        self.gui = gui
        if self.gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        config_file = f"{self.sumo_config_stub}.sumocfg"
        sumoCmd = [
            sumoBinary, 
            "-c", config_file, # start sumo with supplied config-file
            '--device.battery.probability', '1', # sets all vehicles to be EVs instead of combustion engine
            # '--device.stationfinder.probability', '1' # remove vehicle if it runs out of battery
            '--time-to-teleport', '-1', # disable teleporting of vehicles that are stuck in traffic, we want queues to form at charging stations
            ]
        if self.gui:
            sumoCmd += [
                '--delay', '100',   # slow down playback so the GUI is watchable
                '--start',          # open and immediately begin the simulation
            ]
        if random_seed is not None:
            sumoCmd += ['--seed', str(random_seed)]
        if sumo_log_path is not None:
            sumoCmd += [
                '--log', sumo_log_path,
                '--log.timestamps', 'true',
                '--aggregate-warnings', '100'
                ]

        # close any existing traci connection (e.g. from previous simulation runs) before starting a new one
        try:
            traci.close()
        except:
            pass
        traci.start(sumoCmd)
        self._state_file = f"initial_state_{os.getpid()}"
        traci.simulation.saveState(self._state_file) # needed for reset
        self.scenario_generator = construct_scenario_generator(scenario_generator, random_seed)
        self.charging_stations = self._fetch_charging_stations()
        self.reset()

    def reset(self):
        self.added_vehicles = set()
        self.charging_vehicle_ids = set()
        self.vehicle_destinations = {}
        self.max_capacities = {}
        self.departure_counts = Counter()
        self.just_removed_vehicle_ids = set()
        self.soc_history: dict[str, list[float]] = {}  # SOC (Wh) per vehicle per simulation step
        self.cumulative_waiting_times = {}  # true total waiting time per vehicle (seconds)
        self._prev_waiting_times = {}       # VAR_WAITING_TIME reading from the previous step
        self.driving_segment_baseline: dict[str, tuple[float, float]] = {}  # (soc_Wh, distance_m) at start of current driving segment; reset after charging stops
        traci.simulation.loadState(self._state_file)
        self._subscribe_to_simulation()
        self.simulation_data = traci.simulation.getSubscriptionResults()
        self.vehicle_data = None


    def _subscribe_to_simulation(self):
        """Subscribe to relevant simulation variables."""
        traci.simulation.subscribe([
            tc.VAR_MIN_EXPECTED_VEHICLES, 
            tc.VAR_TIME,
            tc.VAR_LOADED_VEHICLES_IDS,
            tc.VAR_DEPARTED_VEHICLES_IDS,
            tc.VAR_ARRIVED_VEHICLES_IDS,
            tc.VAR_STOP_ENDING_VEHICLES_IDS,
            tc.VAR_STOP_STARTING_VEHICLES_IDS,
            tc.VAR_TELEPORT_STARTING_VEHICLES_IDS,
            tc.VAR_TELEPORT_ENDING_VEHICLES_IDS,
        ])

    def add_non_observable_routes(self):
        """
        Add routes for charging station usage of non-observable EVs.
        """
        for cs_id, cs_edge in self.charging_stations.items():
            traci.route.add(f"{cs_id}_non_observable_route", [cs_edge])

    def add_non_observable_vehicle(self, cs_id, depart_time, charge_duration):
        """
        Add a vehicle with a specific departure time directly in front of specified charging station. The vehicle disappears shortly after charging has finished.

        Args:
            cs_id (str): The CS where the vehicle should charge.
            depart_time (int): The time step at which the vehicle should enter the simulation (in seconds).
            charge_duration (int): The charging duration in seconds.
        """
        count = self.departure_counts.get(depart_time, 0)
        suffix = f"_{count}" if count else ""
        vehicle_id = f"non_observable_ev_t{depart_time}{suffix}"
        self.departure_counts[depart_time] += 1

        route_id = f"{cs_id}_non_observable_route"
        traci.vehicle.add(vehicle_id, route_id, depart=depart_time)
        traci.vehicle.setChargingStationStop(vehicle_id, cs_id, duration=charge_duration)


    def add_vehicles(self, amount = 1, scenario_id = None):
        """
        Adds the specified amount of vehicles to the simulation.

        Args:
            amount (int): The number of vehicles to add.
            scenario_id (str, optional): The id of the scenario to use for generating the vehicles. If None, the scenario is selected randomly.
        """
        # edge_list = traci.edge.getIDList()
        all_routes = network_generator.get_all_routes(self.sumo_config_stub)
        start_soc_bounds = self.start_soc_bounds or network_generator.get_start_soc_bounds(self.sumo_config_stub)
        routes_dict = self.scenario_generator.generate_routes_from_routes_list(amount, all_routes)
        routes_id_list = list(routes_dict.keys())
        vehicles_dict = self.scenario_generator.generate_vehicles(amount, routes_id_list, start_soc_bounds, scenario_id)
        logger.debug(f"adding vehicles: {vehicles_dict}")

        for route_id, route in routes_dict.items():
            traci.route.add(route_id, route)

        for vehicle_id, vehicle in vehicles_dict.items():
            traci.vehicle.add(vehicle_id, vehicle["route"], typeID=vehicle["type"], depart=vehicle["depart_time"])
            traci.vehicle.setParameter(vehicle_id, "device.battery.maximumBatteryCapacity", str(vehicle["capacity"]))
            traci.vehicle.setParameter(vehicle_id, "device.battery.actualBatteryCapacity", str(vehicle["soc"]))
            logger.debug(f"Vehicle {vehicle_id} added with initial route: {vehicle['route']}")
            self.added_vehicles.add(vehicle_id)
            self._subscribe_to_vehicle(vehicle_id)
        logger.debug(f"Added {len(vehicles_dict)} vehicles: {list(vehicles_dict.keys())}")

    def _subscribe_to_vehicle(self, vehicle_id):
        """Helper function defining vehicle subscription to TraCI variables."""
        traci.vehicle.subscribe(vehicle_id, [
            tc.VAR_ROAD_ID, # Current edge ID
            # tc.VAR_POSITION, # Current position (x, y)
            tc.VAR_STOPSTATE, # Stop state (stopped, driving, etc.)
            tc.VAR_DISTANCE, # Distance travelled since departure
            tc.VAR_WAITING_TIME, # Current waiting time (resets when vehicle moves)
        ])
        traci.vehicle.subscribeParameterWithKey(vehicle_id, "device.battery.actualBatteryCapacity") # Current SOC

    def _fetch_charging_stations(self):
        charging_station_ids = traci.chargingstation.getIDList()
        charging_stations = {}
        for cs_id in charging_station_ids:
            cs_edge = self._get_cs_edge(cs_id)
            charging_stations[cs_id] = cs_edge
        return charging_stations

    def _get_cs_edge(self, cs_id):
        cs_lane = traci.chargingstation.getLaneID(cs_id)
        cs_edge = traci.lane.getEdgeID(cs_lane)
        return cs_edge

    def _get_vehicle_edge(self, vehicle_id):
        """
        Returns the edge of the vehicle with given id. Returns None if vehicle is not found in simulation or not on a lane.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            str or None: The edge ID or None if not found.
        """
        data = self.vehicle_data.get(vehicle_id)
        if data is None:
            logger.error(f"get_vehicle_edge: {vehicle_id} not found in simulation. It probably reached its destination already (or was removed).")
            return None
        vehicle_edge = data.get(tc.VAR_ROAD_ID)
        if not vehicle_edge: # None or ""
            logger.debug(f"get_vehicle_edge: {vehicle_id} is not on a road. It probably didn't spawn yet.")
            return None
        return vehicle_edge
        # try:
        #     vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
        # except traci.exceptions.TraCIException: 
        #     logger.error(f"get_vehicle_edge: {vehicle_id} not found in simulation. It probably reached its destination already (or was removed).")
        #     return None       
        # if str(vehicle_lane) == "":
        #     logger.debug(f"{vehicle_id} is not on a lane. It probably didn't spawn yet.")
        #     return None
        # vehicle_edge = traci.lane.getEdgeID(vehicle_lane)
        # return vehicle_edge


    def reroute_for_charging(self, vehicle_id, cs_id):
        """
        Reroutes the vehicle via a charging station.

        Args:
            vehicle_id (str): The ID of the vehicle to be rerouted.
            cs_id (str): The ID of the charging station.

        Returns:
            None
        """
        # get vehicle type, lane, edge, and destination
        #TODO: Use subscription (low prio)
        v_type = traci.vehicle.getTypeID(vehicle_id)
        current_vehicle_edge = self._get_vehicle_edge(vehicle_id)
        destination = self.get_vehicle_destination(vehicle_id)
        cs_edge = self.charging_stations[cs_id]
        dist_cs = self._calculate_distance(current_vehicle_edge, cs_edge)
        dist_dest = self._calculate_distance(current_vehicle_edge, destination)
        #TODO: Use subscription (low prio)
        logger.debug(f"Vehicle {vehicle_id} current route before reroute: {traci.vehicle.getRoute(vehicle_id)}, destination: {self.get_vehicle_destination(vehicle_id)}")

        if dist_cs is None or dist_dest is None:
            raise ImpossibleRoutingError(f"Cannot calculate route: vehicle_id={vehicle_id}, cs_id={cs_id}, dist_cs={dist_cs}, dist_dest={dist_dest}.")
        if dist_cs > dist_dest:
            raise PointlessRecommendationError(f"Recommendation denied: Recommended station is past destination. vehicle_id={vehicle_id}, cs_id={cs_id}, dist_cs={dist_cs}, dist_dest={dist_dest}")
        if current_vehicle_edge != cs_edge: # this check avoids that charging is abborted if this function gets called while a vehicle is charging
            # find route from vehicle position to charging station and from charging station to destination
            route_to_cs = traci.simulation.findRoute(current_vehicle_edge, cs_edge, v_type)
            if not route_to_cs or not route_to_cs.edges:
                raise ImpossibleRoutingError(f"No route found from position {current_vehicle_edge} to cs {cs_edge} for vehicle {vehicle_id}.")
            route_from_cs = traci.simulation.findRoute(cs_edge, destination, v_type)
            if not route_from_cs or not route_from_cs.edges:
                raise ImpossibleRoutingError(f"No route found from cs {cs_edge} to destination {destination} for vehicle {vehicle_id}.")
            new_route = route_to_cs.edges + route_from_cs.edges[1:]
            logger.debug(f"Rerouting {vehicle_id} to charging station {cs_id} via route: {new_route}")
            try:
                traci.vehicle.setRoute(vehicle_id, new_route)
            except traci.exceptions.TraCIException as e:
                raise Exception(f"TraCIException: {e} \n\
                                Route from position {current_vehicle_edge} to cs {cs_edge} to destination {destination}. \n\
                                route_to_cs: {route_to_cs} \n\
                                route_from_cs: {route_from_cs} \n\
                                new_route: {new_route}")
        try:
            traci.vehicle.setChargingStationStop(vehicle_id, cs_id, duration=CHARGING_DURATION)
        except traci.exceptions.TraCIException as e:
            raise BadTimingRoutingError(f"{vehicle_id} is past the charging station, rerouting not possible. TraCIException: {e}")

    def remove_charging_stop(self, vehicle_id):
        """
        Removes the next scheduled charging stop for the given vehicle.

        Args:
            vehicle_id (str): The ID of the vehicle.
        """
        try:
            traci.vehicle.replaceStop(vehicle_id, nextStopIndex=0, edgeID="")
        except traci.exceptions.TraCIException:
            logger.debug(f"{vehicle_id}: No charging stop to remove")

    def get_stops(self, vehicle_id):
        """
        Returns the list of stops for the given vehicle.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            list: List of stops.
        """
        try:
            #TODO: Use subscription (low prio)
            stops = traci.vehicle.getStops(vehicle_id)
            return stops
        except traci.exceptions.TraCIException:
            raise ValueError(f"Vehicle {vehicle_id} not found in simulation. It probably reached its destination already.")

    def vehicle_is_rerouted(self, vehicle_id) -> bool:
        """
        Checks if a vehicle is already rerouted to a charging station.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            bool: True if rerouted, False otherwise.
        """
        stops = self.get_stops(vehicle_id)
        if stops:
            if stops[0].stopFlags == 32:  # 32 is traci's code for a planned charging station stop, changes to 33 while charging
                return True
        return False

    def get_next_charging_stop_id(self, vehicle_id):
        """
        Returns the charging station id for the next planned stop for given vehicle_id.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            str or None: Charging station ID or None if no stop is planned.
        """
        try:
            stops = self.get_stops(vehicle_id)
        except ValueError as e:
            raise e
        if stops:
            return stops[0].stoppingPlaceID
        return None

    def get_remaining_range(self, vehicle_id):
        """
        Returns the estimated remaining range of given vehicle in meters.

        Computes consumption from actual battery SOC changes (device.battery.actualBatteryCapacity)
        over the current driving segment (since spawn or since the last charging stop ended).
        Falls back to CONSUMPTION_DEFAULT when the segment is too short for a reliable estimate.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            float or None: Remaining range in meters, or None if SOC is unavailable.
        """
        DISTANCE_THRESHOLD = 100 # the threshold under which the energy consumption calculation is deemed too unprecise
        CONSUMPTION_DEFAULT = 0.24 # Wh/m = 240 Wh/km, the default value for energy consumption, used until the live-calculation is deemed precise enough

        remaining_soc = self.get_battery_soc(vehicle_id)
        if remaining_soc is None:
            return None

        # Lazily initialise baseline from the first SOC reading of this episode.
        baseline = self.driving_segment_baseline.get(vehicle_id)
        if baseline is None:
            soc_hist = self.soc_history.get(vehicle_id, [])
            if soc_hist:
                baseline = (soc_hist[0], 0.0)
                self.driving_segment_baseline[vehicle_id] = baseline

        if baseline is not None:
            baseline_soc, baseline_distance = baseline
            current_distance = float(self.vehicle_data.get(vehicle_id, {}).get(tc.VAR_DISTANCE, 0))
            segment_distance = current_distance - baseline_distance
            segment_consumed = baseline_soc - remaining_soc  # positive = energy drawn

            if segment_distance >= DISTANCE_THRESHOLD and segment_consumed > 0:
                consumption = segment_consumed / segment_distance
                remaining_range_m = remaining_soc / consumption
                logger.debug(f"Remaining range of {vehicle_id}: {remaining_range_m:.0f} m "
                             f"(measured {consumption:.4f} Wh/m over {segment_distance:.0f} m)")
                return remaining_range_m

        # Fallback: not enough driving data yet, or SOC temporarily above baseline after charging
        remaining_range_m = remaining_soc / CONSUMPTION_DEFAULT
        logger.debug(f"Remaining range of {vehicle_id}: {remaining_range_m:.0f} m (default rate)")
        return remaining_range_m

    def get_max_battery_capacity(self, vehicle_id):
        """
        Returns the max. battery capacity of given vehicle. Caches the max. battery capacity for each vehicle.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            float: Maximum battery capacity.
        """
        max_capacity = self.max_capacities.get(vehicle_id)
        if max_capacity is None:
            max_capacity = float(traci.vehicle.getParameter(vehicle_id, "device.battery.maximumBatteryCapacity")) 
            self.max_capacities[vehicle_id] = max_capacity
        return max_capacity

    def get_max_possible_distance(self):
        """
        Returns the maximum possible distance between a vehicle's start and destination within the currently loaded network in meters.

        Returns:
            float: Maximum possible distance in meters.
        """
        return network_generator.get_max_possible_distance(self.sumo_config_stub)
    
    def get_vehicle_destination(self, vehicle_id):
        """
        Returns the destination of given vehicle. Caches the destination for each vehicle, so isn't aware if destination changes in SUMO.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            str: Destination edge ID.
        """
        destination = self.vehicle_destinations.get(vehicle_id)
        if destination is None:
            destination = traci.vehicle.getRoute(vehicle_id)[-1]
            self.vehicle_destinations[vehicle_id] = destination
        return destination

    def get_distance_to_destination(self, vehicle_id, vehicle_edge):
        """
        Returns the driving distance of given vehicle's current position to its destination.

        Args:
            vehicle_id (str): The ID of the vehicle.
            vehicle_edge (str): The current edge of the vehicle.

        Returns:
            float or None: Distance to destination, or None if unreachable.
        """
        if vehicle_edge is None:
            return None
        destination = self.get_vehicle_destination(vehicle_id)
        distance = self._calculate_distance(vehicle_edge, destination)
        return distance

    def get_ideal_travel_time(self, vehicle_id, vehicle_edge):
        """
        Return the estimated free-flow travel time from the vehicle's current position to its
        destination, using traci.simulation.findRoute.

        This mirrors what a real deployment would obtain from an external routing API at the
        start of a trip: a time estimate based on current network conditions and speed limits,
        with no charging detour. Because findRoute uses actual edge speeds rather than a hard-
        coded constant, it stays correct if speed limits or vehicle types change.

        Note: SUMO's findRoute reflects current traffic conditions on each edge. Calling it at
        spawn (before congestion builds) gives a close approximation to free-flow travel time.
        In production, the equivalent would be a routing API call made at trip start.

        Args:
            vehicle_id (str): The ID of the vehicle (used to determine vehicle type).
            vehicle_edge (str): The current edge of the vehicle (route start).

        Returns:
            float or None: Estimated travel time in seconds, or None if no route found.
        """
        if vehicle_edge is None:
            return None
        destination = self.get_vehicle_destination(vehicle_id)
        if destination is None:
            return None
        vtype = traci.vehicle.getTypeID(vehicle_id)
        try:
            stage = traci.simulation.findRoute(vehicle_edge, destination, vType=vtype)
            return stage.travelTime if stage.edges else None
        except traci.exceptions.TraCIException:
            return None

    def step(self):
        traci.simulationStep()
        self.just_removed_vehicle_ids = set()
        self.vehicle_data = traci.vehicle.getAllSubscriptionResults()
        self.simulation_data = traci.simulation.getSubscriptionResults()
        # VAR_WAITING_TIME is a running counter that SUMO resets to 0 when a vehicle moves.
        # We track the delta each step to get the true total waiting time across the entire episode.
        for vid, data in self.vehicle_data.items():
            current_wt = data.get(tc.VAR_WAITING_TIME) or 0
            prev_wt = self._prev_waiting_times.get(vid, 0)
            delta = current_wt - prev_wt if current_wt >= prev_wt else current_wt
            self.cumulative_waiting_times[vid] = self.cumulative_waiting_times.get(vid, 0) + delta
            self._prev_waiting_times[vid] = current_wt
        # When an observable EV ends a charging stop its SOC has been replenished, so reset the
        # driving segment baseline so get_remaining_range() uses only post-charge SOC changes.
        for vid in self._filter_set_for_observable_evs(
                set(self.simulation_data.get(tc.VAR_STOP_ENDING_VEHICLES_IDS, []))):
            data = self.vehicle_data.get(vid, {})
            soc_data = data.get(tc.VAR_PARAMETER_WITH_KEY)
            distance = float(data.get(tc.VAR_DISTANCE, 0))
            if soc_data and soc_data[0] == "device.battery.actualBatteryCapacity":
                self.driving_segment_baseline[vid] = (float(soc_data[1]), distance)
                logger.debug(f"Driving segment baseline reset for {vid}: soc={soc_data[1]} Wh, dist={distance:.0f} m")
        sim_time = self.simulation_data.get(tc.VAR_TIME)
        for vid in self.simulation_data.get(tc.VAR_TELEPORT_STARTING_VEHICLES_IDS, []):
            logger.warning(f"Teleport start: {vid} at t={sim_time}")
        for vid in self.simulation_data.get(tc.VAR_TELEPORT_ENDING_VEHICLES_IDS, []):
            logger.warning(f"Teleport end: {vid} at t={sim_time}")

    def active_vehicles_exist(self):
        expected_vehicles = self.simulation_data.get(tc.VAR_MIN_EXPECTED_VEHICLES, 0)
        # traci_value = traci.simulation.getMinExpectedNumber()
        return expected_vehicles > 0 

    def close(self):
        traci.close()
        try:
            os.remove(self._state_file)
        except FileNotFoundError:
            pass

    def get_all_charging_station_ids(self):
        return list(self.charging_stations.keys())
    
    def _filter_set_for_observable_evs(self, original_collection):
        return {item for item in original_collection if item.startswith("observable_ev")}

    def get_all_vehicle_ids(self):
        """
        Returns a set of all vehicle ids that have been added to the simulation.

        Returns:
            set: Set of vehicle IDs.
        """
        return self.added_vehicles
    
    def get_all_oev_ids(self):
        """
        Returns a set of all observable vehicle ids that have been added to the simulation.

        Returns:
            set: Set of observable vehicle IDs.
        """
        original_set = self.get_all_vehicle_ids()
        return self._filter_set_for_observable_evs(original_set)

    def get_online_vehicle_ids(self):
        """
        Returns a set of ids of all observable vehicles currently running within the scenario.

        Returns:
            set: Set of online observable vehicle IDs.
        """
        #TODO: Use subscription (low prio)
        original_set = set(traci.vehicle.getIDList())
        return self._filter_set_for_observable_evs(original_set)

    def get_loaded_vehicle_ids(self):
        """
        Returns a set of all loaded vehicle ids that have not yet arrived. This includes vehicles that are meant to depart in the future.

        Remark:
            Sumo does not load all vehicle definitions in advance but only when they are needed. If you give the vehicle definitions in an additional file instead, all will be parsed in advance but only if you define individual vehicles not with flows.

        Returns:
            set: Set of loaded observable vehicle IDs.
        """
        original_set = set(self.simulation_data.get(tc.VAR_LOADED_VEHICLES_IDS, []))
        return self._filter_set_for_observable_evs(original_set)


    def get_spawned_vehicle_ids(self):
        """
        Returns a set of ids of all observable vehicles that have spawned during the current time step.

        Returns:
            set: Set of spawned observable vehicle IDs.
        """
        original_list = set(self.simulation_data.get(tc.VAR_DEPARTED_VEHICLES_IDS, []))
        return self._filter_set_for_observable_evs(original_list)


    def get_arrived_vehicle_ids(self):
        """
        Returns a set of ids of all observable vehicles that have arrived at their destination during the current time step.

        Returns:
            set: Set of arrived observable vehicle IDs.
        """
        original_set = set(self.simulation_data.get(tc.VAR_ARRIVED_VEHICLES_IDS, []))
        return self._filter_set_for_observable_evs(original_set)

    def get_removed_vehicle_ids(self):
        """
        Returns a set of ids of all observable vehicles that have been manually removed from the simulation during the current time step.
        (E.g. because their battery is empty)

        Returns:
            set: Set of removed observable vehicle IDs.
        """
        return self.just_removed_vehicle_ids

    def get_charging_vehicle_ids(self):
        """
        Returns a set of ids of observable vehicles currently charging at any charging station.

        Returns:
            set: Set of charging observable vehicle IDs.
        """
        charging_vehicles = set()
        charging_stations_ids = self.get_all_charging_station_ids()

        for station in charging_stations_ids:
            # Get vehicles stopped at this charging station
            vehicles = traci.chargingstation.getVehicleIDs(station) # Note: Direct call, because it doesn't seem possible to get this value via traci subscription
            charging_vehicles.update(vehicles)

        original_set = charging_vehicles
        return self._filter_set_for_observable_evs(original_set)


    def get_charging_stop_ending_vehicle_ids(self):
        """
        Returns a set of ids of observable vehicles that begin to continue their journey, leaving a scheduled stop in this time step.

        Returns:
            set: Set of observable vehicle IDs leaving a charging stop.
        """
        original_set = set(self.simulation_data.get(tc.VAR_STOP_ENDING_VEHICLES_IDS, []))
        return self._filter_set_for_observable_evs(original_set)

    def get_vehicle_waiting_time(self, vehicle_id):
        """
        Return the total waiting time accumulated for the vehicle across all episode steps.
        SUMO's VAR_WAITING_TIME resets to 0 when a vehicle resumes movement, so we sum it step-by-step in self.cumulative_waiting_times to capture the true total.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            float: Total accumulated waiting time in seconds.
        """
        return self.cumulative_waiting_times.get(vehicle_id, 0)

    def get_soc_history(self) -> dict[str, list[float]]:
        """Returns SOC (Wh) recorded each simulation step per vehicle for the current episode."""
        return self.soc_history

    def get_stop_starting_vehicle_ids(self):
        """Returns a set of observable EV ids that begin a stop in this time step."""
        return self._filter_set_for_observable_evs(
            set(self.simulation_data.get(tc.VAR_STOP_STARTING_VEHICLES_IDS, []))
        )

    def _adapt_vehicle_color(self, vehicle_id, battery_soc):
        """
        Changes the vehicle's color in the GUI based on its battery SOC.
        Turns blue when a charging stop is planned.

        Args:
            vehicle_id (str): The ID of the vehicle.
            battery_soc (float): The state of charge (SOC) of the vehicle's battery.
        """
        battery_is_very_low = battery_soc < 50
        battery_is_quite_full = battery_soc > 200

        if battery_is_very_low:
            color = RED
        elif self.vehicle_is_rerouted(vehicle_id):
            color = BLUE
        elif battery_is_quite_full:
            color = GREEN
        else:
            color = YELLOW
        traci.vehicle.setColor(vehicle_id, color)  

    def _simulate_empty_battery(self, vehicle_id):
        """
        Removes the vehicle from simulation when its battery is empty.
        Idempotent: Simply returns if the vehicle was already removed.

        Args:
            vehicle_id (str): The ID of the vehicle.
        """
        if vehicle_id in self.just_removed_vehicle_ids:
            return
        logger.info(f"Battery empty, vehicle {vehicle_id} will be removed from simulation")
        traci.vehicle.unsubscribe(vehicle_id)
        traci.vehicle.remove(vehicle_id)
        self.just_removed_vehicle_ids.add(vehicle_id)

    def _calculate_distance(self, edgeID1, edgeID2):
        """
        Returns the distance between the given edgeIDs. Returns 0 if edgeID1 == edgeID2, Returns None if edgeID2 is not reachable from edgeID1.

        Args:
            edgeID1 (str): The starting edge ID.
            edgeID2 (str): The destination edge ID.

        Returns:
            float or None: Distance in meters, or None if unreachable.
        """
          
        # Old way: Request distance from TraCI:
        # distance = traci.simulation.getDistanceRoad(edgeID1=edgeID1, pos1=0, edgeID2=edgeID2, pos2=0, isDriving=True)
        # return None if distance < 0 else distance # sumo returns a negative distance if edgeID2 is not reachable from edgeID1. 
        
        if edgeID1 == edgeID2:
            return 0
        if edgeID1 in self.all_distances and edgeID2 in self.all_distances[edgeID1]:
            return self.all_distances[edgeID1][edgeID2]
        return None # if not found in precomputed distances, return None to indicate unreachable

    def _get_distances_to_all_cs(self, vehicle_position) -> dict:
        """
        Returns a dictionary of distances from the vehicle's position to all charging stations.

        Args:
            vehicle_position (str): The current edge of the vehicle.

        Returns:
            dict: Mapping from charging station ID to distance (float or None).
        """
        if vehicle_position is None:
            return None
        charging_stations = self.charging_stations
        distance_dict = {}
        for station_id in charging_stations.keys():
            charging_station_edge = charging_stations[station_id]
            distance = self._calculate_distance(vehicle_position, charging_station_edge)
            distance_dict[station_id] = float(distance) if distance is not None else None
        return distance_dict

    def get_battery_soc(self, vehicle_id) -> float | None:
        """
        Returns the current battery SOC (Wh) for the vehicle.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            float or None: Actual battery capacity in Wh, or None if not found.
        """
        data = self.vehicle_data.get(vehicle_id)
        if data is None:
            logger.debug(f"get_battery_soc(): {vehicle_id} not found in simulation. It probably reached its destination already (or was removed).")
            return None
        parameter_data = data.get(tc.VAR_PARAMETER_WITH_KEY, {})
        return float(parameter_data[1]) if parameter_data[0] == "device.battery.actualBatteryCapacity" else None

    def update_vehicle_soc(self, vehicle_id) -> float | None:
        """
        Reads the vehicle's battery SOC and runs all per-step side effects:
        records to soc_history, updates GUI colour, and removes the vehicle
        if the battery is empty. Should be called exactly once per vehicle per step.

        Returns:
            float or None: The SOC in Wh, or None if vehicle not found.
        """
        battery_soc = self.get_battery_soc(vehicle_id)
        if battery_soc is None:
            return None
        if self.gui:
            self._adapt_vehicle_color(vehicle_id, battery_soc)
        self.soc_history.setdefault(vehicle_id, []).append(battery_soc)
        if battery_soc <= EMPTY_SOC:
            self._simulate_empty_battery(vehicle_id)
        return battery_soc

    def get_vehicle_state(self, vehicle_id):
        """
        Retrieves the state of a vehicle.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            dict: A dictionary containing the following information:
                - battery_soc (float): The actual battery capacity of the vehicle.
                - max_battery_capacity (float): The maximum battery capacity of the vehicle (i.e. capacity at full charge).
                - distance_to_cs (dict): The distance to each charging station.
                - vehicle_position (str): The current position of the vehicle.
                - vehicle_destination (str): The destination of the vehicle.
                - distance_to_destination (float): The distance to the destination.
        """
        vehicle_edge = self._get_vehicle_edge(vehicle_id)

        vehicle_is_offline = vehicle_edge is None # if vehicle_edge is not returned by TraCI, signalling that the vehicle has not been spawned yet or has already been removed
        if vehicle_is_offline:
            battery_soc = None
            max_battery_capacity = None
            distance_to_cs = None
            vehicle_edge = None
            vehicle_destination = None
            distance_to_destination = None
            ideal_travel_time = None
        else:
            battery_soc = self.get_battery_soc(vehicle_id)
            distance_to_cs = self._get_distances_to_all_cs(vehicle_edge)
            vehicle_destination = self.get_vehicle_destination(vehicle_id)
            max_battery_capacity = self.get_max_battery_capacity(vehicle_id)
            distance_to_destination = self.get_distance_to_destination(vehicle_id, vehicle_edge)
            assert distance_to_destination is not None, f"destination not reachable for vehicle {vehicle_id}, position {vehicle_edge}, destination {vehicle_destination}"
            # For "production": Option to assert to handle this gracefully:
            # if distance_to_destination is None:
            #     logger.error(f"Destination not reachable for vehicle {vehicle_id}, position {vehicle_edge}, destination {vehicle_destination}. Removing vehicle from simulation.")
            #     # Remove the vehicle from simulation as its destination is unreachable
            #     try:
            #         traci.vehicle.remove(vehicle_id)
            #     except traci.exceptions.TraCIException:
            #         logger.warning(f"Could not remove vehicle {vehicle_id} - it may have already been removed.")
            #     # Return None state to indicate this vehicle should be ignored
            #     return {"battery_soc": None, "max_battery_capacity": None, "distance_to_cs": None, "vehicle_position": None, "vehicle_destination": None, "distance_to_destination": None}
            ideal_travel_time = self.get_ideal_travel_time(vehicle_id, vehicle_edge)

        state = {"battery_soc": battery_soc, "max_battery_capacity": max_battery_capacity, "distance_to_cs": distance_to_cs, "vehicle_position": vehicle_edge, "vehicle_destination": vehicle_destination, "distance_to_destination": distance_to_destination, "ideal_travel_time": ideal_travel_time}
        return state

    def get_departure_time_for_vehicle(self, vehicle_id):
        """
        Returns the actual departure time in seconds for the given vehicle.

        Args:
            vehicle_id (str): The ID of the vehicle.

        Returns:
            float: Departure time in seconds.
        """
        return traci.vehicle.getDeparture(vehicle_id)
    
    def get_state(self):
        """
        Get the current simulation state. Returns the state of all vehicles in the simulation.

        Returns:
            dict: Mapping from vehicle ID to vehicle state dict.
        """
        state = {}
        for vehicle_id in self.get_all_vehicle_ids():
            state[vehicle_id] = self.get_vehicle_state(vehicle_id)
        return state

    def get_current_time_step(self):
        """
        Returns the current time step of the SUMO simulation.

        Returns:
            float: Current simulation time step.
        """
        return self.simulation_data.get(tc.VAR_TIME, 0)

if __name__ == "__main__":

    simulation = Simulation(scenario_generator="BASt", gui=False)

    def _adapt_destination(vehicle_id, vehicle_state):
        """Routes the vehicles in circles"""
        start = "E0"
        end = "E10"
        position = vehicle_state["vehicle_position"]
        destination = vehicle_state["vehicle_destination"]
        if position == destination:
            traci.vehicle.changeTarget(vehicle_id, end if destination == start else start)  

    def _print_vehicle_charging(vehicle_ids):
        for vehicle_id in vehicle_ids:
            is_charging = traci.vehicle.getStopState(vehicle_id) & 2 ** 5 != 0

            if is_charging:
                charging_station = traci.vehicle.getStopState(vehicle_id) & 2 ** 13 != 0

                print(f"Vehicle {vehicle_id} is charging at a {'charging station' if charging_station else 'parking area'}")

                # Get additional charging information
                battery_capacity = traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")
                energy_charged = traci.vehicle.getParameter(vehicle_id, "device.battery.energyCharged")

                print(f"  Battery Capacity: {battery_capacity} Wh")
                print(f"  Energy Charged: {energy_charged} Wh")


    def driving_in_circles():
        cs_id = "cs_1"

        simulation.add_vehicles(50)

        while simulation.active_vehicles_exist():

            for vehicle_id in simulation.get_all_vehicle_ids():
                vehicle_state = simulation.get_vehicle_state(vehicle_id)
                print(vehicle_state)
                # send the vehicle driving in circles
                _adapt_destination(vehicle_id, vehicle_state)
                # reroute to charging station if battery is low
                if float(vehicle_state["battery_soc"]) < 100:
                    print("Battery low, rerouting to charge")
                    simulation.reroute_for_charging(vehicle_id, cs_id)

            simulation.step()

        simulation.close()

    def test_simulation_end():
        cs_id = "cs_1"
        simulation.add_vehicles(50)
        print("vehicles added")
        # while simulation_time < 24
        for i in range(200):
            simulation.step()
            print(f"simulation step {i}")
        simulation.close()


    def test_non_observable_vehicles():
        """Test whether non_observable_vehicles are spawning and despawning as expected - compare console output of this function to data source."""
        from noev_data_provider import Obelis_Data_Provider
        data_provider = Obelis_Data_Provider()
        print("Data provider added")
        simulation.add_non_observable_routes()
        print("routes added")
        vehicle_data = data_provider.get_non_observable_vehicle_data()
        print("got non observable vehicle data")
        for entry in vehicle_data:
            simulation.add_non_observable_vehicle(cs_id=entry["cs_id"], depart_time=entry["charge_begin_seconds"], charge_duration=entry["charge_duration"])

        # Set to keep track of vehicles in the simulation
        active_vehicles = set()
        # Subscribe to vehicle arrival and departure events
        traci.simulation.subscribe([tc.VAR_DEPARTED_VEHICLES_IDS, 
                                    tc.VAR_ARRIVED_VEHICLES_IDS])
        
        for i in range(40000):
            # Get the current simulation step
            current_step = traci.simulation.getTime()

            # Get subscription results
            result = traci.simulation.getSubscriptionResults()

            # Check for new vehicles (spawned)
            if result[tc.VAR_DEPARTED_VEHICLES_IDS]:
                for vehicle_id in result[tc.VAR_DEPARTED_VEHICLES_IDS]:
                    print(f"Step {current_step}: Vehicle {vehicle_id} spawned")
                    active_vehicles.add(vehicle_id)

            # Check for arrived vehicles (despawned)
            if result[tc.VAR_ARRIVED_VEHICLES_IDS]:
                for vehicle_id in result[tc.VAR_ARRIVED_VEHICLES_IDS]:
                    print(f"Step {current_step}: Vehicle {vehicle_id} despawned")
                    active_vehicles.remove(vehicle_id)
            
            _print_vehicle_charging(active_vehicles)
            
            simulation.step()
        
        simulation.close()
    
    # driving_in_circles()
    # test_non_observable_vehicles()
    test_simulation_end()

