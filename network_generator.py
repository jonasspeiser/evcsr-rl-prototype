"""
This script generates a SUMO network for a highway with charging stations.
"""

import os
from datetime import datetime

# === GENERAL CONFIGURATION ===
km_total = 100
km_step = 1
lane_count = 3
lane_speed = 33.33  # m/s ~120 km/h
charging_speed = 8.33  # m/s ~30 km/h
charging_spots = [25, 50, 75, 100]  # in km
route_file = "autobahn.rou.xml"  # Optional

# === CHARGING STATION CONFIGURATION ===
charging_params = {
    "startPos": 10.0,
    "endPos": 40.0,
    "chargeDelay": 2,
    "chargeInTransit": 0,
    "power": 200_000,        # Watts
    "efficiency": 0.95
}


# === NODES ===
with open("autobahn.nod.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')

    # Main highway nodes
    for i in range(0, km_total + 1):
        f.write(f'  <node id="n{i}" x="{i * 1000}" y="0" type="priority"/>\n')

    # Charging station entry/exit nodes
    for cs in charging_spots:
        entry_x = cs * 1000
        f.write(f'  <node id="cs{cs}_entry" x="{entry_x}" y="-10" type="priority"/>\n')
        f.write(f'  <node id="cs{cs}_exit" x="{entry_x + 1000}" y="-10" type="priority"/>\n')

    f.write('</nodes>\n')

# === EDGES ===
with open("autobahn.edg.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')

    # Main highway edges
    for i in range(0, km_total):
        f.write(f'  <edge id="e{i}" from="n{i}" to="n{i+1}" type="highway"/>\n')

    # Charging station edges
    for cs in charging_spots:
        entry = f"cs{cs}_entry"
        exit = f"cs{cs}_exit"
        f.write(f'  <edge id="cs{cs}_entry_edge" from="n{cs}" to="{entry}" type="charging"/>\n')
        f.write(f'  <edge id="cs{cs}_main" from="{entry}" to="{exit}" type="charging"/>\n')
        if cs + 1 <= km_total:
            f.write(f'  <edge id="cs{cs}_exit_edge" from="{exit}" to="n{cs+1}" type="charging"/>\n')

    f.write('</edges>\n')

# === TYPES ===
with open("autobahn.typ.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<types>\n')
    f.write(f'  <type id="highway" numLanes="{lane_count}" speed="{lane_speed}"/>\n')
    f.write(f'  <type id="charging" numLanes="1" speed="{charging_speed}"/>\n')
    f.write('</types>\n')


# === ADDITIONAL: Charging Stations ===
with open("autobahn.add.xml", "w") as f:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
    f.write(f'<!-- generated on {timestamp} by Python script -->\n\n')
    f.write('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n')

    for i, cs in enumerate(charging_spots):
        lane_id = f"cs{cs}_main_0"
        f.write(f'  <chargingStation id="cs_{i}" lane="{lane_id}" '
                f'startPos="{charging_params["startPos"]}" '
                f'endPos="{charging_params["endPos"]}" '
                f'chargeDelay="{charging_params["chargeDelay"]}" '
                f'chargeInTransit="{charging_params["chargeInTransit"]}" '
                f'power="{charging_params["power"]}" '
                f'efficiency="{charging_params["efficiency"]}"/>\n')

    f.write('</additional>\n')

# === MERGING TO NETWORK FILE ===

# Generate net.xml via netconvert
print("XML data generated successfully:")
print("  - autobahn.nod.xml")
print("  - autobahn.edg.xml")
print("  - autobahn.typ.xml")
print("  - autobahn.add.xml")

print("Now creating SUMO net file with:")
print("\nnetconvert -n autobahn.nod.xml -e autobahn.edg.xml -t autobahn.typ.xml -o autobahn.net.xml\n")
# auto-run netconvert if installed
os.system("netconvert -n autobahn.nod.xml -e autobahn.edg.xml -t autobahn.typ.xml -o autobahn.net.xml")
print("Network file 'autobahn.net.xml' created successfully.")

# === SUMO CONFIGURATION ===
with open("autobahn.sumocfg", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
    f.write('<configuration>\n')
    f.write('    <input>\n')
    f.write('        <net-file value="autobahn.net.xml"/>\n')
    f.write(f'        <route-files value="{route_file}"/>\n')  # Optional file
    f.write('        <additional-files value="autobahn.add.xml"/>\n')
    f.write('    </input>\n\n')
    f.write('    <time>\n')
    f.write('        <begin value="0"/>\n')
    f.write('        <end value="3600"/>\n')  # 1 hour simulation
    f.write('    </time>\n\n')
    f.write('    <report>\n')
    f.write('        <verbose value="true"/>\n')
    f.write('        <no-step-log value="true"/>\n')
    f.write('    </report>\n')
    f.write('</configuration>\n')

print("SUMO configuration file 'autobahn.sumocfg' created.")
print("Network generation complete. You can now run SUMO with the generated files.")
print("\n🚦 Start the Simulation with:")
print("  sumo-gui autobahn.sumocfg")