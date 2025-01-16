import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import random
import traci
from sumolib import checkBinary

# configure logging
import logging
logger = logging.getLogger("rl.environment.simulation") # child logger of "application.environment"

# define color values for vehicles in the GUI
BLUE = [153, 255, 255]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
RED = [255, 0, 0]

SUMO_CONFIG_PATH = "circle.sumocfg"
CHARGING_DURATION = 5 # charging duration in seconds

class Simulation():

    def __init__(self, gui:bool=False):
        if type(gui) is not bool:
            raise ValueError("gui must be a boolean")
        self.gui = gui
        if self.gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        config_file = SUMO_CONFIG_PATH
        sumoCmd = [
            sumoBinary, 
            "-c", config_file, # start sumo with supplied config-file
            '--delay', '100', # delay between each sim step 100ms
            '--start', # start simulation immediately
            '--device.battery.probability', '1', # sets all vehicles to be EVs instead of combustion engine
            # '--device.stationfinder.probability', '1' # remove vehicle if it runs out of battery
            ]

        traci.start(sumoCmd)
        traci.simulation.saveState("initial_state") # needed for reset
        self.charging_stations = self.__fetch_charging_stations()
        self.added_vehicles = []
        self.charging_vehicle_ids = []
        self.vehicle_destinations = {}


    def __fetch_charging_stations(self):
        charging_station_ids = traci.chargingstation.getIDList()
        charging_stations = {}
        for cs_id in charging_station_ids:
            cs_edge = self.__get_cs_edge(cs_id)
            charging_stations[cs_id] = cs_edge
        return charging_stations

    def __get_cs_edge(self, cs_id):
        cs_lane = traci.chargingstation.getLaneID(cs_id)
        cs_edge = traci.lane.getEdgeID(cs_lane)
        return cs_edge
    
    def add_non_member_routes(self):
        """Add routes for charging station usage of non-member EVs"""
        for cs_id, cs_edge in self.charging_stations.items():
            traci.route.add(f"{cs_id}_non_member_route", [cs_edge])
        
    def add_non_member_vehicle(self, cs_id, depart_time, charge_duration):
        """
        Add a vehicle with a specific departure time directly in front of specified charging station. The vehicle disappears shortly after charging has finished.
        Params: 
            cs_id (str): The CS where the vehicle should charge
            depart_time (int): The time step at which the vehicle should enter the simulation (in seconds)
            charge_duration (int): The charging duraction in seconds
        """
        vehicle_id = f"non_member_ev_{depart_time}"
        route_id = f"{cs_id}_non_member_route"
        traci.vehicle.add(vehicle_id, route_id, depart=depart_time)
        traci.vehicle.setChargingStationStop(vehicle_id, cs_id, duration=charge_duration)


    def add_vehicles(self, amount = 1, random_seed = None):
        """ 
        Adds the specified amount of vehicles to the simulation. 
        Battery SOC is randomly chosen for each vehicle individually (between 50 and 500 Wh). 
        """
        traci.route.add("trip", ["E0", "E19"])
        for i in range(amount):
            vehID = "member_ev_" + str(i)
            traci.vehicle.add(vehID, "trip", typeID="DEFAULT_VEHTYPE")
            battery_min = 100
            battery_max = 500
            battery_soc = random.randint(battery_min, battery_max)
            traci.vehicle.setParameter(vehID, "device.battery.maximumBatteryCapacity", str(battery_max))
            traci.vehicle.setParameter(vehID, "device.battery.actualBatteryCapacity", str(battery_soc))
            self.added_vehicles.append(vehID)

    def __get_vehicle_edge(self, vehicle_id):
        vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
        if str(vehicle_lane) == "":
            raise ValueError("Vehicle is not on a lane")
        vehicle_edge = traci.lane.getEdgeID(vehicle_lane)
        return vehicle_edge

    def get_position_and_destination(self, vehicle_id):
        vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
        try:
            current_vehicle_edge = traci.lane.getEdgeID(vehicle_lane)
        except traci.exceptions.TraCIException:
            raise ValueError(f"Lane {vehicle_lane} not found, most likely vehicle {vehicle_id} does not exist in the simulation yet")
        destination = self.get_vehicle_destination(vehicle_id)
        return current_vehicle_edge, destination

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
        v_type = traci.vehicle.getTypeID(vehicle_id)
        current_vehicle_edge, destination = self.get_position_and_destination(vehicle_id)
        cs_edge = self.charging_stations[cs_id]
        if current_vehicle_edge != cs_edge: # this check avoids that charging is abborted if this function gets called while a vehicle is charging
            # find route to charging station and from charging station to destination
            route_to_cs = traci.simulation.findRoute(current_vehicle_edge, cs_edge, v_type)
            route_from_cs = traci.simulation.findRoute(cs_edge, destination, v_type)
            new_route = route_to_cs.edges + route_from_cs.edges[1:]
            traci.vehicle.setRoute(vehicle_id, new_route)
        try:
            traci.vehicle.setChargingStationStop(vehicle_id, cs_id, duration=CHARGING_DURATION)
        except traci.exceptions.TraCIException:
            raise ValueError("Vehicle is past the charging station, rerouting not possible")

    def remove_charging_stop(self, vehicle_id):
        try:
            traci.vehicle.replaceStop(vehicle_id, nextStopIndex=0, edgeID="")
        except traci.exceptions.TraCIException:
            logger.info("No charging stop to remove")

    def get_stops(self, vehicle_id):
        try:
            stops = traci.vehicle.getStops(vehicle_id)
            return stops
        except traci.exceptions.TraCIException:
            raise ValueError(f"Vehicle {vehicle_id} not found in simulation. It probably reached its destination already.")

    def vehicle_is_rerouted(self, vehicle_id) -> bool:
        """ Checks if a vehicle is already rerouted to a charging station. """
        stops = self.get_stops(vehicle_id)
        if stops:
            if stops[0].stopFlags == 32:  # 32 is traci's code for a planned charging station stop, changes to 33 while charging
                return True
        return False

    def get_next_charging_stop_id(self, vehicle_id):
        """ Returns the charging station id for the next planned stop for given vehicle_id. Returns None if no stop is planned. """
        try:
            stops = self.get_stops(vehicle_id)
        except ValueError as e:
            raise e
        if stops:
            return stops[0].stoppingPlaceID
        return None

    def get_remaining_range(self, vehicle_id):
        """
        Returns the estimated remaining range of given vehicle in km. 
        This is an approximation and may vary based on driving conditions.
        Returns None if remaining range can't be calculated.
        """
        remaining_capacity = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
        energy_consumed = float(traci.vehicle.getParameter(vehicle_id, "device.battery.totalEnergyConsumed"))
        distance_travelled = float(traci.vehicle.getDistance(vehicle_id))
        # Return None if remaining range can't be calculated (-> division by zero)
        if distance_travelled == 0:
            return None
        # Get the energy consumption in Wh/km
        try:
            energy_consumption = energy_consumed / distance_travelled
            remaining_range_km = remaining_capacity / energy_consumption
            logger.info(f"Remaining range of vehicle {vehicle_id}: {remaining_range_km} km")
            logger.debug(f"vehicle {vehicle_id}: Energy consumed: {energy_consumed}, distance travelled: {distance_travelled}, remaining capacity: {remaining_capacity}, energy consumption: {energy_consumption}")
        except ZeroDivisionError as e:
            raise ZeroDivisionError(f"Vehicle {vehicle_id} has not moved yet, can't calculate remaining range. energy_consumed: {energy_consumed}, distance_travelled: {distance_travelled}, remaining_capacity: {remaining_capacity}, energy_consumption: {energy_consumption}")
        return remaining_range_km

    def get_vehicle_destination(self, vehicle_id):
        """ Returns the destination of given vehicle. Caches the destination for each vehicle, so isn't aware if destination changes in SUMO. """
        destination = self.vehicle_destinations.get(vehicle_id)
        if destination is None:
            destination = traci.vehicle.getRoute(vehicle_id)[-1]
            self.vehicle_destinations[vehicle_id] = destination
        return destination

    def get_distance_to_destination(self, vehicle_id):
        """ Returns the driving distance of given vehicles current position to its destination. """
        current_vehicle_edge = self.__get_vehicle_edge(vehicle_id)
        destination = self.get_vehicle_destination(vehicle_id)
        distance = self.__calculate_distance(current_vehicle_edge, destination)
        return distance

    def remaining_range_is_sufficient(self, vehicle_id, buffer=0):
        """ 
        Returns whether a vehicle's battery soc is enough to reach its destination.
        If the optional "buffer" parameter is set, it must have a battery soc higher than "buffer" when arriving, 
        otherwise it just has to be not completely empty. 
        Returns None if remaining range is None.
        """
        remaining_range = self.get_remaining_range(vehicle_id)
        if remaining_range is None:
            return None
        distance_to_destination = self.get_distance_to_destination(vehicle_id)
        return remaining_range > (distance_to_destination + buffer)

    def step(self):
        traci.simulationStep()

    def active_vehicles_exist(self):
        return traci.simulation.getMinExpectedNumber() > 0

    def close(self):
        traci.close()

    def reset(self):
        self.added_vehicles = []
        traci.simulation.loadState("initial_state")

    def get_all_charging_station_ids(self):
        return list(self.charging_stations.keys())
    
    def __filter_list_for_member_evs(self, original_list):
        filtered_list = [item for item in original_list if item.startswith("member_ev")]
        return filtered_list

    def get_all_vehicle_ids(self):
        """ Returns a list of all vehicle ids that have been added to the simulation. """
        return self.added_vehicles

    def get_online_vehicle_ids(self):
        """Returns a list of ids of all member vehicles currently running within the scenario"""
        original_list= traci.vehicle.getIDList()
        return self.__filter_list_for_member_evs(original_list)

    def get_loaded_vehicle_ids(self):
        """
        Returns a list of all loaded vehicle ids that have not yet arrived. This includes vehicles that are meant to depart in the future.
        Remark: Sumo does not load all vehicle definitions in advance but only when they are needed. 
        If you give the vehicle definitions in an additional file instead, all will be parsed in advance but only if you define inidvidual vehicles not with flows. 
        """
        original_list = traci.simulation.getLoadedIDList()
        return self.__filter_list_for_member_evs(original_list)


    def get_spawned_vehicle_ids(self):
        """Returns a list of ids of all member vehicles that have spawned during the current time step"""
        original_list = traci.simulation.getDepartedIDList()
        return self.__filter_list_for_member_evs(original_list)


    def get_arrived_vehicle_ids(self):
        """Returns a list of ids of all member vehicles that have arrived at their destination during the current time step"""
        original_list = traci.simulation.getArrivedIDList()
        return self.__filter_list_for_member_evs(original_list)


    def get_charging_vehicle_ids(self):
        charging_vehicles = []
        charging_stations_ids = self.get_all_charging_station_ids()

        for station in charging_stations_ids:
            # Get vehicles stopped at this charging station
            vehicles = traci.chargingstation.getVehicleIDs(station)
            charging_vehicles.extend(vehicles)

        original_list = charging_vehicles
        return self.__filter_list_for_member_evs(original_list)


    def get_charging_stop_ending_vehicle_ids(self):
        """Returns a list of ids of member vehicles that begin to continue their journey, leaving a scheduled stop in this time step"""
        original_list = traci.simulation.getStopEndingVehiclesIDList()
        return self.__filter_list_for_member_evs(original_list)
    
    def get_vehicle_waiting_time(self, vehicle_id):
        """Return the accumulated waiting time for the vehicle. Due to traci limitations, this is only possible for online vehicles."""
        return traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
    

    def __adapt_vehicle_color(self, vehicle_id, battery_soc):
        """
        The vehicles color in the GUI is changed based on its battery SOC. 
        Furthermore, it turns blue when a charging stop is planned.

        Args:
            vehicle_id (str): The ID of the vehicle.
            battery_soc (float): The state of charge (SOC) of the vehicle's battery.

        Returns:
            None
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

    def __simulate_empty_battery(self, vehicle_id):
        logger.info(f"Battery empty, vehicle {vehicle_id} will stop and remain at its position")
        traci.vehicle.setSpeed(vehicle_id, 0)
        # traci.vehicle.remove(vehicle_id)

    def __calculate_distance(self, edgeID1, edgeID2):
        return traci.simulation.getDistanceRoad(edgeID1=edgeID1, pos1=0, edgeID2=edgeID2, pos2=0, isDriving=True)

    def __get_distance_to_cs(self, vehicle_position) -> dict:
        charging_stations = self.charging_stations
        distance_dict = {}
        for station_id in charging_stations.keys():
            charging_station_edge = charging_stations[station_id]
            distance = self.__calculate_distance(vehicle_position, charging_station_edge)
            distance_dict[station_id] = float(distance)
        return distance_dict

    def get_battery_soc(self, vehicle_id):
        """Returns the actual battery capacity of the vehicle."""
        try:
            battery_soc = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
            return battery_soc
        except traci.exceptions.TraCIException: 
            logger.error(f"Vehicle {vehicle_id} not found in simulation. It probably reached its destination already.")
            return None

    def get_vehicle_state(self, vehicle_id): 
        """
        Retrieves the state of a vehicle.

        Parameters:
        - vehicle_id (str): The ID of the vehicle.

        Returns:
        - dict: A dictionary containing the following information:
            - battery_soc (float): The actual battery capacity of the vehicle.
            - distance_to_cs (float): The distance to the next charging station.
            - vehicle_position (str): The current position of the vehicle.
            - vehicle_destination (str): The destination of the vehicle.
        """
        battery_soc = self.get_battery_soc(vehicle_id)
        if battery_soc is None: # if battery_soc is not returned by TraCI
            distance_to_cs = None
            vehicle_edge = None
            vehicle_destination = None
        else: # only execute if vehicle has a battery, otherwise skip
            if self.gui:
                self.__adapt_vehicle_color(vehicle_id, battery_soc)
            # stop vehicle if battery is empty
            if battery_soc <= 0:
                self.__simulate_empty_battery(vehicle_id)

            try:
                vehicle_edge = self.__get_vehicle_edge(vehicle_id) 
                distance_to_cs = self.__get_distance_to_cs(vehicle_edge)
            except ValueError: # if vehicle is not on a lane, i.e. it hasn't spawned yet or despawned after arriving at destination
                vehicle_edge = None
                distance_to_cs = None
            vehicle_destination = self.get_vehicle_destination(vehicle_id)

        state = {"battery_soc": battery_soc, "distance_to_cs": distance_to_cs, "vehicle_position": vehicle_edge, "vehicle_destination": vehicle_destination}
        return state

    def get_state(self):
        state = {}
        for vehicle_id in self.get_all_vehicle_ids():
            state[vehicle_id] = self.get_vehicle_state(vehicle_id)
        return state

    def get_current_time_step(self):
        """Returns the current time step of the SUMO simulation"""
        return traci.simulation.getTime()

if __name__ == "__main__":

    simulation = Simulation(gui=True)

    def __adapt_destination(vehicle_id, vehicle_state):
        """Routes the vehicles in circles"""
        start = "E0"
        end = "E10"
        position = vehicle_state["vehicle_position"]
        destination = vehicle_state["vehicle_destination"]
        if position == destination:
            traci.vehicle.changeTarget(vehicle_id, end if destination == start else start)  

    def __print_vehicle_charging(vehicle_ids):
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
        cs_id = "cs_0"

        simulation.add_vehicles(50)

        while simulation.active_vehicles_exist():

            for vehicle_id in simulation.get_all_vehicle_ids():
                vehicle_state = simulation.get_vehicle_state(vehicle_id)
                print(vehicle_state)
                # send the vehicle driving in circles
                __adapt_destination(vehicle_id, vehicle_state)
                # reroute to charging station if battery is low
                if float(vehicle_state["battery_soc"]) < 100:
                    print("Battery low, rerouting to charge")
                    simulation.reroute_for_charging(vehicle_id, cs_id)

            simulation.step()

        simulation.close()

    def test_non_member_vehicles():
        """Test whether non_member_vehicles are spawning and despawning as expected - compare console output of this function to data source."""
        from data_processing import Obelis_Data_Provider
        data_provider = Obelis_Data_Provider()
        print("Data provider added")
        simulation.add_non_member_routes()
        print("routes added")
        vehicle_data = data_provider.get_non_member_vehicle_data()
        print("got non member vehicle data")
        for entry in vehicle_data:
            simulation.add_non_member_vehicle(cs_id=entry["cs_id"], depart_time=entry["charge_begin_seconds"], charge_duration=entry["charge_duration"])

        # Set to keep track of vehicles in the simulation
        active_vehicles = set()
        # Subscribe to vehicle arrival and departure events
        traci.simulation.subscribe([traci.constants.VAR_DEPARTED_VEHICLES_IDS, 
                                    traci.constants.VAR_ARRIVED_VEHICLES_IDS])
        
        for i in range(40000):
            # Get the current simulation step
            current_step = traci.simulation.getTime()

            # Get subscription results
            result = traci.simulation.getSubscriptionResults()

            # Check for new vehicles (spawned)
            if result[traci.constants.VAR_DEPARTED_VEHICLES_IDS]:
                for vehicle_id in result[traci.constants.VAR_DEPARTED_VEHICLES_IDS]:
                    print(f"Step {current_step}: Vehicle {vehicle_id} spawned")
                    active_vehicles.add(vehicle_id)

            # Check for arrived vehicles (despawned)
            if result[traci.constants.VAR_ARRIVED_VEHICLES_IDS]:
                for vehicle_id in result[traci.constants.VAR_ARRIVED_VEHICLES_IDS]:
                    print(f"Step {current_step}: Vehicle {vehicle_id} despawned")
                    active_vehicles.remove(vehicle_id)
            
            __print_vehicle_charging(active_vehicles)
            
            simulation.step()
        
        simulation.close()
    
    # driving_in_circles()
    test_non_member_vehicles()

