import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from sumolib import checkBinary

sumoBinary = checkBinary('sumo-gui')
# sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

sumoCmd = [sumoBinary, "-c", r"C:\Users\SPJ1WI\projects\rl_toy_usecase\helloWorld.sumocfg", '--delay', '200', '--start']

traci.start(sumoCmd)

traci.route.add("trip", ["E0", "E3"])
traci.vehicle.add("newVeh", "trip", typeID="DEFAULT_VEHTYPE")


step = 0
while step < 100:
    traci.simulationStep()
    step += 1

traci.close()
