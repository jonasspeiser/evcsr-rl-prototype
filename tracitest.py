import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from sumolib import checkBinary
sumoBinary = checkBinary('sumo-gui')
# sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

# start sumo with config-file -c, delay between each sim step 200ms, start simulation immediately, all vehicles have a battery and are therefore EVs
sumoCmd = [
    sumoBinary, 
    "-c", r"C:\Users\SPJ1WI\projects\rl_toy_usecase\helloWorld.sumocfg", # start sumo with supplied config-file
    '--delay', '200', # delay between each sim step 200ms
    '--start', # start simulation immediately
    '--device.battery.probability', '1' # sets all vehicles to be EVs instead of combustion engine
    ]

traci.start(sumoCmd)

traci.route.add("trip", ["E0", "E3"])
traci.vehicle.add("newVeh", "trip", typeID="DEFAULT_VEHTYPE")


step = 0
while step < 100:
    traci.simulationStep()
    step += 1

"""
TODO: Car has battery which is drained
TODO: Charging Station available
TODO: If battery falls under certain value, Car stops at charging station and charges battery


"""

traci.close()
