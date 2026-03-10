"""
This script generates a SUMO network for a highway with charging stations.
"""

import os
from datetime import datetime
import subprocess
from sumolib.net import readNet
import json

# === CONFIGURATION ===
test_generated_files = True  # Set to False to skip launching SUMO

networks_dir = "street-networks"  # Directory to store generated files
network_name = "straight_120km"  # Base name for the network files


# === GENERAL CONFIGURATION ===
km_total = 120 # Total length of the generated highway straight in km
km_step = 1 # length of each individual edge in km
lane_count = 3
highway_lane_speed = 33.33  # m/s ~120 km/h
charging_lane_speed = 8.33  # m/s ~30 km/h
charging_spots = [25, 50, 75, 100]  # the number marks the distance from the first node in km
max_possible_distance = float((km_total * 1000) + (km_total * 0.01))  # Max possible distance between a vehicle's potential start and destination in meters, adding some margin as buffer 
max_battery_capacity = 100_000.0  # Used for SOC normalization in observations (not the physical battery capacity)
start_soc_bounds = (3880, 22_390)  # Start SOC bounds for vehicles in Wh (~6% to ~35% of the soulEV65's 64 kWh battery; estimated range at highway speed: ~18–102 km)

# === CHARGING STATION CONFIGURATION ===
charging_params = {
    "startPos": 500.0, # Distance from the entry node in meters
    "endPos": 510.0,
    "chargeDelay": 2,
    "chargeInTransit": 0,
    "power": 200_000,        # Watts
    "efficiency": 0.95
}

def write_all_routes_json(net_file_path: str, output_path: str):
    """
    Writes all possible edge-to-edge routes in the network to a JSON file.
    This includes all pairs of edges, excluding routes that start and end on the same edge.
    Only includes routes that are actually reachable.
    """

    print(f"Writing all edge-to-edge routes to JSON from {net_file_path}...")

    net = readNet(net_file_path)
    edges = net.getEdges()

    all_routes = []
    route_count = 0

    for from_edge in edges:
        for to_edge in edges:
            if from_edge.getID() == to_edge.getID():
                continue
            try:
                # Use getShortestPath to check if route is reachable
                path_result = net.getShortestPath(from_edge, to_edge)
                if not path_result or not path_result[0]:
                    continue
                    
                path_edges = path_result[0]
                route_ids = [e.getID() for e in path_edges]
                route_length = sum(e.getLength() for e in path_edges)
                
                # Additional validation: ensure the route is actually valid
                if route_length > 0:
                    all_routes.append({
                        "from": from_edge.getID(),
                        "to": to_edge.getID(),
                        "length": round(route_length, 2),
                        "route": route_ids
                    })
                    route_count += 1
            except Exception as e:
                # Skip unreachable pairs
                print(f"Skipping route from {from_edge.getID()} to {to_edge.getID()}: {e}")
                continue

    with open(output_path, "w") as f:
        json.dump(all_routes, f, indent=2)

    print(f"Wrote {route_count} valid routes to {output_path}")
    return all_routes

def get_all_routes(sumo_config_path_stub: str) -> dict:
    """
    Returns all possible edge-to-edge routes in the network as a dictionary.
    This includes all pairs of edges, excluding routes that start and end on the same edge.
    Args:
        sumo_config_path_stub (str): Path stub for the SUMO configuration files (without file extension). E.g. "street-networks/straight_100km/straight_100km"
    """
    routes_file_path = f"{sumo_config_path_stub}.all_routes.json"
    # if routes file doesn't exist, create it
    if not os.path.exists(routes_file_path):
        all_routes = write_all_routes_json(f"{sumo_config_path_stub}.net.xml", routes_file_path)
    # read edge pairings from routes file
    else:
        with open(routes_file_path, "r") as f:
            all_routes = json.load(f)
    return all_routes

def write_all_distances_json(net_file_path: str, output_path: str):
    """
    Writes all possible edge-to-edge distances in the network to a JSON file.
    This includes all pairs of edges, excluding distances that start and end on the same edge.
    Only includes distances that are actually reachable.
    """

    print(f"Writing all edge-to-edge distances to JSON from {net_file_path}...")

    net = readNet(net_file_path, withInternal=True)
    edges = net.getEdges(withInternal=True)

    all_distances = {}
    distance_count = 0

    for from_edge in edges:
        for to_edge in edges:
            if from_edge.getID() == to_edge.getID():
                continue
            # Use getShortestPath to check if distance is reachable
            path_result = net.getShortestPath(from_edge, to_edge)
            if not path_result or not path_result[0]:
                continue
            path_edges = path_result[0]
            route_length = sum(e.getLength() for e in path_edges)
            # Additional validation: ensure the distance is actually valid
            if route_length > 0:
                if from_edge.getID() not in all_distances: # Initialize sub-dict if not present
                    all_distances[from_edge.getID()] = {}
                all_distances[from_edge.getID()][to_edge.getID()] = round(route_length, 2)
                distance_count += 1
            
    with open(output_path, "w") as f:
        json.dump(all_distances, f, indent=2)

    print(f"Wrote {distance_count} valid distances to {output_path}")
    return all_distances

def get_max_possible_distance(SUMO_CONFIG_STUB) -> float:
    """
    Returns the maximum possible distance between a vehicle's potential start and destination in km.
    Args:
        SUMO_CONFIG_STUB (str): Path stub for the SUMO configuration files (without file extension). E.g. "street-networks/straight_100km/straight_100km"
    """
    config_file_path = f"{SUMO_CONFIG_STUB}.config.json"
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} does not exist.")
    
    with open(config_file_path, "r") as f:
        config = json.load(f)
    
    return config["max_possible_distance"]


def compute_auto_truncation_limit(street_network: str, avg_speed_kmh: float = 100, safety_factor: float = 1.2) -> int:
    """
    Computes a truncation limit (in simulation seconds) based on the network's maximum possible
    distance.
    Assumes all vehicles depart at t=0 (true for all non-BASt scenarios).
    The BASt 24 h override in the environment is unaffected by this value.

    Args:
        street_network (str): Street network directory name, e.g. "straight_120km".
        avg_speed_kmh (float): Assumed average travel speed in km/h. Defaults to 100.
        safety_factor (float): Multiplier applied to the raw trip duration estimate. Defaults to 1.2.
    """
    sumo_config_stub = f"./street-networks/{street_network}/{street_network}"
    max_dist = get_max_possible_distance(sumo_config_stub)
    avg_speed_ms = avg_speed_kmh / 3.6
    return int(max_dist / avg_speed_ms * safety_factor)


def get_start_soc_bounds(sumo_config_path_stub: str) -> tuple:
    """
    Returns the start SOC bounds for vehicles in the network.
    Args:
        sumo_config_path_stub (str): Path stub for the SUMO configuration files (without file extension). E.g. "street-networks/straight_100km/straight_100km"
    """
    config_file_path = f"{sumo_config_path_stub}.config.json"
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} does not exist.")
    
    with open(config_file_path, "r") as f:
        config = json.load(f)
    
    return tuple(config["start_soc_bounds"])


def generate_test_routes(network_path_stub: str):
    """
    Generates a simple set of test routes for the highway network.
    This creates 5 vehicles with a detour for one vehicle at a charging station.
    """
    print("Generating test routes with sumolib...")

    # Load the generated network
    net = readNet(f"{network_path_stub}.net.xml")

    # Get all highway edges
    edges = [e for e in net.getEdges() if e.getID().startswith("e")]
    edge_dict = {e.getID(): e for e in edges}

    with open(f"{network_path_stub}.rou.xml", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write('  <vType id="car" accel="1.5" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="33.33"/>\n')

        for i in range(5):
            depart_time = i * 5
            from_edge = edge_dict["e0"]
            to_edge = edge_dict[f"e{km_total - 1}"]  # e99

            # Get shortest path (list of edge objects)
            path_edges = net.getShortestPath(from_edge, to_edge)[0]
            edge_ids = [e.getID() for e in path_edges]

            if i == 1:
                # Insert charging station detour between e25 and e26
                try:
                    # find the position of the segment you want to replace
                    idx = edge_ids.index("e25")
                    # build a new list: everything *before* e25, then the 3 detour edges,
                    # then everything *after* e25
                    edge_ids = (
                        edge_ids[:idx]
                        + ["cs25_entry_edge", "cs25_main", "cs25_exit_edge"]
                        + edge_ids[idx + 1:]
                    )
                except ValueError:
                    print("⚠️ Warning: e25 not in shortest path — skipping detour for veh1")

            # Write vehicle with optional stop
            f.write(f'  <vehicle id="veh{i}" type="car" depart="{depart_time}">\n')
            f.write(f'    <route edges="{" ".join(edge_ids)}"/>\n')

            if i == 1:
                f.write('    <stop lane="cs25_main_0" startPos="20.0" duration="30"/>\n')

            f.write('  </vehicle>\n')

        f.write('</routes>\n')

    print("Route file written.")

    with open(f"{network_path_stub}.test.sumocfg", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
        f.write('<configuration>\n')
        f.write('    <input>\n')
        f.write(f'        <net-file value="{network_name}.net.xml"/>\n')
        f.write(f'        <route-files value="{network_name}.rou.xml"/>\n')  # Optional file
        f.write(f'        <additional-files value="{network_name}.add.xml"/>\n')
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
    print("Test routes generated and written to .rou.xml and .sumocfg files.")


def start_sumo_gui(network_path_stub: str):
    """
    Launches the SUMO GUI with the specified network configuration.
    """
    print("🚦 Launching SUMO GUI...")
    subprocess.run(["sumo-gui", f"{network_path_stub}.test.sumocfg"])


if __name__ == "__main__":
    # === FILE PATH SETUP ===
    # Ensure the street-networks directory exists
    network_working_dir = os.path.join(".", networks_dir, network_name) # Directory for the current network
    if not os.path.exists(network_working_dir):
        os.makedirs(network_working_dir)
    network_path_stub = os.path.join(network_working_dir, network_name) # network path without extension

    route_file = f"{network_path_stub}.rou.xml"  # Optional

    # === NODES ===
    with open(f"{network_path_stub}.nod.xml", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')

        # Main highway nodes
        for i in range(0, km_total + 1):
            f.write(f'  <node id="n{i}" x="{i * 1000}" y="0" type="priority"/>\n')

        # Charging station entry/exit nodes
        for cs in charging_spots:
            entry_x = cs * 1000
            f.write(f'  <node id="cs{cs}_entry" x="{entry_x}" y="0" type="priority"/>\n')
            f.write(f'  <node id="cs{cs}_exit" x="{entry_x + 1000}" y="0" type="priority"/>\n')

        f.write('</nodes>\n')

    # === EDGES ===
    with open(f"{network_path_stub}.edg.xml", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')

        # Main highway edges
        for i in range(0, km_total):
            f.write(f'  <edge id="e{i}" from="n{i}" to="n{i+1}" type="highway"/>\n')

        # Charging station edges
        for cs in charging_spots:
            entry = f"cs{cs}_entry"
            exit = f"cs{cs}_exit"
            entry_x = cs * 1000
            exit_x  = entry_x + 1000
            # branch down: highway y=0 -> station y=-10
            f.write(
                f'  <edge id="cs{cs}_entry_edge" from="n{cs}" to="{entry}" type="charging" '
                f'shape="{entry_x},0 {entry_x},-10"/>\n'
            )
            # main station lane at y=-10
            f.write(
                f'  <edge id="cs{cs}_main" from="{entry}" to="{exit}" type="charging" '
                f'shape="{entry_x},-10 {exit_x},-10"/>\n'
            )
            # branch up: station y=-10 -> highway y=0
            if cs + 1 <= km_total:
                f.write(
                    f'  <edge id="cs{cs}_exit_edge" from="{exit}" to="n{cs+1}" type="charging" '
                    f'shape="{exit_x},-10 {exit_x},0"/>\n'
                )

        f.write('</edges>\n')

    # === TYPES ===
    with open(f"{network_path_stub}.typ.xml", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<types>\n')
        f.write(f'  <type id="highway" numLanes="{lane_count}" speed="{highway_lane_speed}"/>\n')
        f.write(f'  <type id="charging" numLanes="1" speed="{charging_lane_speed}"/>\n')
        f.write('</types>\n')


    # === ADDITIONAL: Charging Stations ===
    with open(f"{network_path_stub}.add.xml", "w") as f:
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

    # === GENERATE NETWORK ===
    print("✔ Generating SUMO network...")
    os.system(
        f'netconvert -n {network_path_stub}.nod.xml '
        f'-e {network_path_stub}.edg.xml '
        f'-t {network_path_stub}.typ.xml '
        f"--junctions.join true "  
        f'-o {network_path_stub}.net.xml'
    )


    # === GENERATE SUMOCFG ===
    with open(f"{network_path_stub}.sumocfg", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
        f.write('<configuration>\n')
        f.write('    <input>\n')
        f.write(f'        <net-file value="{network_name}.net.xml"/>\n')
        f.write(f'        <additional-files value="{network_name}.add.xml, ../../vehicle-models/soulEV65.add.xml"/>\n')
        f.write('    </input>\n\n')
        f.write('    <report>\n')
        f.write('        <verbose value="true"/>\n')
        f.write('        <no-step-log value="true"/>\n')
        f.write('    </report>\n')
        f.write('</configuration>\n')


    # === GENERATE CONFIG FILE ===
    with open(f"{network_path_stub}.config.json", "w") as f:
        config = {
            "network_name": network_name,
            "km_total": km_total,
            "max_possible_distance": max_possible_distance,
            "max_battery_capacity": max_battery_capacity,
            "start_soc_bounds": start_soc_bounds,
        }
        json.dump(config, f, indent=2)

    # === GENERATE ROUTE MATRIX FOR ANALYSIS ===
    write_all_routes_json(
        net_file_path=f"{network_path_stub}.net.xml",
        output_path=f"{network_path_stub}.all_routes.json"
        )

    write_all_distances_json(
        net_file_path=f"{network_path_stub}.net.xml",
        output_path=f"{network_path_stub}.all_distances.json"
    )

    if test_generated_files:
        generate_test_routes(network_path_stub)
        start_sumo_gui(network_path_stub)

    print("Network generation complete. Check the 'street-networks' directory for files.")
