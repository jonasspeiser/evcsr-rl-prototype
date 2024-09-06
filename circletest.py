import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import random
import traci
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from sumolib import checkBinary

BLUE = [153, 255, 255]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
RED = [255, 0, 0]

SUMO_CONFIG_PATH = r"C:\Users\SPJ1WI\projects\rl_toy_usecase\circle.sumocfg"
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
        self.charging_stations = traci.chargingstation.getIDList()
        self.added_vehicles = []


    def add_vehicles(self, amount = 1):
        """ 
        Adds the specified amount of vehicles to the simulation. 
        Battery SOC is randomly chosen for each vehicle individually (between 50 and 500 Wh). 
        """
        traci.route.add("trip", ["E0", "E19"])
        for i in range(amount):
            vehID = "myVehicle" + str(i)
            traci.vehicle.add(vehID, "trip", typeID="DEFAULT_VEHTYPE")
            battery_soc = random.randint(50, 500)
            traci.vehicle.setParameter(vehID, "device.battery.actualBatteryCapacity", str(battery_soc))
            self.added_vehicles.append(vehID)

    def __get_vehicle_edge(self, vehicle_id):
        vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
        if str(vehicle_lane) == "":
            raise ValueError("Vehicle is not on a lane")
        vehicle_edge = traci.lane.getEdgeID(vehicle_lane)
        return vehicle_edge

    def __get_cs_edge(self, cs_id):
        cs_lane = traci.chargingstation.getLaneID(cs_id)
        cs_edge = traci.lane.getEdgeID(cs_lane)
        return cs_edge

    def get_position_and_destination(self, vehicle_id):
        vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
        try:
            current_vehicle_edge = traci.lane.getEdgeID(vehicle_lane)
        except traci.exceptions.TraCIException:
            raise ValueError(f"Lane {vehicle_lane} not found, most likely vehicle {vehicle_id} does not exist in the simulation yet")
        destination = traci.vehicle.getRoute(vehicle_id)[-1]
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
        cs_edge = self.__get_cs_edge(cs_id)
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
            logging.info("No charging stop to remove")

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
        stops = self.get_stops(vehicle_id)
        if stops:
            return stops[0].stoppingPlaceID
        return None

    def step(self):
        traci.simulationStep()

    def active_vehicles_exist(self):
        return traci.simulation.getMinExpectedNumber() > 0

    def close(self):
        traci.close()

    def reset(self):
        self.added_vehicles = []
        traci.simulation.loadState("initial_state")

    def __calculate_distance(self, edgeID1, edgeID2):
        return traci.simulation.getDistanceRoad(edgeID1=edgeID1, pos1=0, edgeID2=edgeID2, pos2=0, isDriving=True)

    def __get_distance_to_cs(self, vehicle_position) -> dict:
        charging_stations = self.charging_stations
        distance_dict = {}
        for station_id in charging_stations:
            charging_station_position = self.__get_cs_edge(station_id)
            distance = self.__calculate_distance(vehicle_position, charging_station_position)
            distance_dict[station_id] = float(distance)
        return distance_dict


    def get_all_charging_station_ids(self):
        return self.charging_stations

    def get_all_vehicle_ids(self):
        """ Returns a list of all vehicle ids that have been added to the simulation. """
        return self.added_vehicles

    def get_online_vehicle_ids(self):
        """Returns a list of ids of all vehicles currently running within the scenario"""
        return traci.vehicle.getIDList()

    def get_loaded_vehicle_ids(self):
        """
        Returns a list of all loaded vehicle ids that have not yet arrived. This includes vehicles that are meant to depart in the future.
        Remark: Sumo does not load all vehicle definitions in advance but only when they are needed. 
        If you give the vehicle definitions in an additional file instead, all will be parsed in advance but only if you define inidvidual vehicles not with flows. 
        """
        return traci.simulation.getLoadedIDList()

    def get_arrived_vehicle_ids(self):
        """Returns a list of ids of all vehicles that have arrived at their destination during the current time step"""
        return traci.simulation.getArrivedIDList()

    def get_spawned_vehicle_ids(self):
        """Returns a list of ids of all vehicles that have spawned during the current time step"""
        return traci.simulation.getDepartedIDList()        

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
        logging.info("Battery empty, vehicle will dissapear shortly")
        traci.vehicle.setSpeed(vehicle_id, 0)
        # traci.vehicle.remove(vehicle_id)

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
        try:
            battery_soc = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
            battery_soc = int(round(battery_soc))
        except traci.exceptions.TraCIException: # if battery_soc is not returned by TraCI
            logging.error(f"Vehicle {vehicle_id} not found in simulation. It probably reached its destination already.")
            battery_soc = None
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
            except ValueError: # if vehicle is not on a lane, i.e. it hasn't spawned yet
                vehicle_edge = None
                distance_to_cs = None
            vehicle_destination = traci.vehicle.getRoute(vehicle_id)[-1]

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

    def adapt_destination(vehicle_id):
        """Routes the vehicles in circles"""
        start = "E0"
        end = "E10"
        position = state["vehicle_position"]
        destination = state["vehicle_destination"]
        if position == destination:
            traci.vehicle.changeTarget(vehicle_id, end if destination == start else start)  

    def driving_in_circles():
        cs_id = "cs_0"

        simulation = Simulation(gui=True)

        simulation.add_vehicles(50)

        while simulation.active_vehicles_exist():

            for vehicle_id in simulation.get_all_vehicle_ids():
                state = simulation.get_vehicle_state(vehicle_id)
                print(state)
                # send the vehicle driving in circles
                adapt_destination(vehicle_id)
                # reroute to charging station if battery is low
                if float(state["battery_soc"]) < 100:
                    print("Battery low, rerouting to charge")
                    simulation.reroute_for_charging(vehicle_id, cs_id)

            simulation.step()

        simulation.close()
    
    driving_in_circles()

