import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from sumolib import checkBinary

def add_vehicles():
    traci.route.add("trip", ["E0", "E3"])
    traci.vehicle.add("myVehicle", "trip", typeID="DEFAULT_VEHTYPE")

def run_simulation():
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0: # while there are cars in the simulation
        try:
            battery_soc = traci.vehicle.getParameter("myVehicle", "device.battery.actualBatteryCapacity")
            print("battery capacity: ", battery_soc)
        except traci.TraCIException as e:
            print(e)
        step += 1

    traci.close()


    """
    TODO: Charging Station available
    TODO: Tell car to reroute to a certain position
    TODO: Tell car to charge at said position
    TODO: Tell car to reroute to destination
    TODO: Use network file from eAlloc
    """

if __name__ == "__main__":
    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [
        sumoBinary, 
        "-c", r"C:\Users\SPJ1WI\projects\rl_toy_usecase\helloWorld.sumocfg", # start sumo with supplied config-file
        '--delay', '200', # delay between each sim step 200ms
        '--start', # start simulation immediately
        '--device.battery.probability', '1' # sets all vehicles to be EVs instead of combustion engine
        ]
    traci.start(sumoCmd)

    add_vehicles()

    run_simulation()