import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from sumolib import checkBinary

def add_vehicles():
    traci.route.add("trip", ["E0", "E3"])
    traci.vehicle.add("myVehicle", "trip", typeID="DEFAULT_VEHTYPE")

def reroute_for_charging(vehicle_id, cs_id):
    """reroute vehicle via charging station"""
    v_type = traci.vehicle.getTypeID(vehicle_id)
    vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
    current_vehicle_edge = traci.lane.getEdgeID(vehicle_lane)
    destination = traci.vehicle.getRoute(vehicle_id)[-1]
    cs_lane = traci.chargingstation.getLaneID(cs_id)
    cs_edge = traci.lane.getEdgeID(cs_lane)

    route_to_cs = traci.simulation.findRoute(current_vehicle_edge, cs_edge, v_type)
    route_from_cs = traci.simulation.findRoute(cs_edge, destination, v_type)
    new_route = route_to_cs.edges + route_from_cs.edges[1:]
    traci.vehicle.setRoute(vehicle_id, new_route)
    traci.vehicle.setChargingStationStop(vehicle_id, cs_id, duration=100)


def run_simulation():
    """execute the TraCI control loop"""
    # main loop: while there are cars in the simulation
    while traci.simulation.getMinExpectedNumber() > 0: 
        try:
            vehicle_id = "myVehicle"
            battery_soc = traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")
            print("battery capacity: ", battery_soc)
            if float(battery_soc) < 17450:
                RED = [255, 0, 0]
                traci.vehicle.setColor(vehicle_id, RED)
                print("Battery low, rerouting to charge")
                cs_id = "cs_1"
                reroute_for_charging(vehicle_id, cs_id)
        except traci.TraCIException as e:
            print(e)
        traci.simulationStep()

    traci.close()


if __name__ == "__main__":
    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [
        sumoBinary, 
        "-c", r"C:\Users\SPJ1WI\projects\rl_toy_usecase\helloWorld\helloWorld.sumocfg", # start sumo with supplied config-file
        '--delay', '200', # delay between each sim step 200ms
        '--start', # start simulation immediately
        '--device.battery.probability', '1' # sets all vehicles to be EVs instead of combustion engine
        ]

    traci.start(sumoCmd)

    add_vehicles()

    run_simulation()