import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from sumolib import checkBinary

BLUE = [153, 255, 255]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
RED = [255, 0, 0]

class Simulation():

    def __init__(self, gui=False):
        self.gui = gui
        if self.gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        config_file = r"C:\Users\SPJ1WI\projects\rl_toy_usecase\circle.sumocfg"
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


    def add_vehicles(self, amount = 1):
        traci.route.add("trip", ["E0", "E10"])
        for i in range(amount):
            vehID = "myVehicle" + str(i)
            traci.vehicle.add(vehID, "trip", typeID="DEFAULT_VEHTYPE")
            traci.vehicle.setParameter(vehID, "device.battery.actualBatteryCapacity", "200")

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
        vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
        current_vehicle_edge = traci.lane.getEdgeID(vehicle_lane)
        destination = traci.vehicle.getRoute(vehicle_id)[-1]
        cs_edge = self.__get_cs_edge(cs_id)
        # find route to charging station and from charging station to destination
        route_to_cs = traci.simulation.findRoute(current_vehicle_edge, cs_edge, v_type)
        route_from_cs = traci.simulation.findRoute(cs_edge, destination, v_type)
        new_route = route_to_cs.edges + route_from_cs.edges[1:]
        traci.vehicle.setRoute(vehicle_id, new_route)
        traci.vehicle.setChargingStationStop(vehicle_id, cs_id, duration=100)

    def step(self):
        traci.simulationStep()

    def active_vehicles_exist(self):
        return traci.simulation.getMinExpectedNumber() > 0

    def close(self):
        traci.close()

    def reset(self):
        traci.simulation.loadState("initial_state")

    def __calculate_distance(self, edgeID1, edgeID2):
        return traci.simulation.getDistanceRoad(edgeID1=edgeID1, pos1=0, edgeID2=edgeID2, pos2=0, isDriving=True)

    def __get_distance_to_next_cs(self, vehicle_position):
        next_charging_station = "cs_0"
        next_charging_station_position = self.__get_cs_edge(next_charging_station)
        return self.__calculate_distance(vehicle_position, next_charging_station_position)

    def get_all_vehicles(self):
        return traci.vehicle.getIDList()

    def vehicle_is_rerouted(self, vehicle_id):
        stops = traci.vehicle.getNextStops(vehicle_id)
        if stops:
            if stops[0][3] == 64:  # 64 is traci's code for a planned charging station stop, changes to 65 while charging
                return True
        return False

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
        print("Battery empty, vehicle will dissapear shortly")
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
            - distance_to_next_cs (float): The distance to the next charging station.
            - vehicle_position (str): The current position of the vehicle.
            - vehicle_destination (str): The destination of the vehicle.
        """
        battery_soc = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
        vehicle_destination = traci.vehicle.getRoute(vehicle_id)[-1]
        if self.gui:
            self.__adapt_vehicle_color(vehicle_id, battery_soc)
        # stop vehicle if battery is empty
        if battery_soc <= 0:
            self.__simulate_empty_battery(vehicle_id)
        try:
            vehicle_edge = self.__get_vehicle_edge(vehicle_id) #traci.vehicle.getPosition(vehicle_id)
            distance_to_next_cs = float(self.__get_distance_to_next_cs(vehicle_edge))
            state = {"battery_soc": int(round(battery_soc)), "distance_to_next_cs": int(round(distance_to_next_cs)), "vehicle_position": vehicle_edge, "vehicle_destination": vehicle_destination}
        except ValueError:
            vehicle_edge = None
            distance_to_next_cs = None
            state = {"battery_soc": int(round(battery_soc)), "distance_to_next_cs": distance_to_next_cs, "vehicle_position": vehicle_edge, "vehicle_destination": vehicle_destination}
        return state

    def get_state():
        state = {}
        for vehicle_id in self.get_all_vehicles():
            state[vehicle_id] = simulation.get_vehicle_state(vehicle_id)
        return state

if __name__ == "__main__":

    def adapt_destination(vehicle_id):
        """Routes the vehicles in circles"""
        start = "E0"
        end = "E10"
        position = state["vehicle_position"]
        destination = state["vehicle_destination"]
        if position == destination:
            traci.vehicle.changeTarget(vehicle_id, end if destination == start else start)  

    cs_id = "cs_0"

    simulation = Simulation(gui=True)

    simulation.add_vehicles(50)

    while simulation.active_vehicles_exist():

        for vehicle_id in simulation.get_all_vehicles():
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