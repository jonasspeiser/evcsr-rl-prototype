import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from sumolib import checkBinary

class Simulation():

    def __init__(self):
        sumoBinary = checkBinary('sumo-gui')
        config_file = r"C:\Users\SPJ1WI\projects\rl_toy_usecase\circle.sumocfg"
        sumoCmd = [
            sumoBinary, 
            "-c", config_file, # start sumo with supplied config-file
            '--delay', '200', # delay between each sim step 200ms
            '--start', # start simulation immediately
            '--device.battery.probability', '1' # sets all vehicles to be EVs instead of combustion engine
            ]

        traci.start(sumoCmd)


    def add_vehicles(self):
        traci.route.add("trip", ["E0", "E10"])
        traci.vehicle.add("myVehicle", "trip", typeID="DEFAULT_VEHTYPE")

    def get_cs_position(self, cs_id):
        cs_lane = traci.chargingstation.getLaneID(cs_id)
        cs_edge = traci.lane.getEdgeID(cs_lane)
        return cs_edge

    def reroute_for_charging(self, vehicle_id, cs_id):
        """reroute vehicle via charging station"""
        # set vehicle color to red
        RED = [255, 0, 0]
        traci.vehicle.setColor(vehicle_id, RED)
        # get vehicle type, lane, edge, and destination
        v_type = traci.vehicle.getTypeID(vehicle_id)
        vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
        current_vehicle_edge = traci.lane.getEdgeID(vehicle_lane)
        destination = traci.vehicle.getRoute(vehicle_id)[-1]
        cs_edge = self.get_cs_position(cs_id)
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

    def calculate_distance(self, pos1, pos2):
        # TODO: implement this
        pass

    def get_distance_to_next_cs(self, vehicle_position):
        next_charging_station = "cs_0"
        next_charging_station_position = self.get_cs_position(next_charging_station)
        return self.calculate_distance(vehicle_position, next_charging_station_position)


    def get_state(self, vehicle_id): 
        battery_soc = traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")
        vehicle_position = traci.vehicle.getPosition(vehicle_id)
        vehicle_destination = traci.vehicle.getRoute(vehicle_id)[-1]
        distance_to_next_cs = self.get_distance_to_next_cs(vehicle_position)
        return {"battery_soc": battery_soc, "distance_to_next_cs": distance_to_next_cs, "vehicle_position": vehicle_position, "vehicle_destination": vehicle_destination}


if __name__ == "__main__":
    
    vehicle_id = "myVehicle"
    cs_id = "cs_0"

    simulation = Simulation()

    simulation.add_vehicles()

    while simulation.active_vehicles_exist():
        state = simulation.get_state(vehicle_id)
        print(state)
        battery_soc = state["battery_soc"]
        if float(battery_soc) < 17450:
            print("Battery low, rerouting to charge")
            simulation.reroute_for_charging(vehicle_id, cs_id)
        simulation.step()

    simulation.close()