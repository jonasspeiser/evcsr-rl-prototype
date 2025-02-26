import random

BATTERY_MIN = 200
BATTERY_MAX = 500

class ScenarioGenerator():

    def __init__(self, seed):
        if seed is not None:
            print("seed set ")
            random.seed(seed)

    def generate_routes(self, n_routes, edge_list):
        """
        Params:
            n_routes (int): The amount of routes that should be generated
            edge_list (List): A list including all of the maps edges
        """
        routes_dict = {}
        for i in range(n_routes):
            route_id = f"trip{i}"
            start, end = random.sample(edge_list, 2) # Picks two UNIQUE elements
            routes_dict[route_id] = [start, end]
        return routes_dict

    def generate_vehicles(self, n_vehicles, routes_list):
        """
        Params:
            n_vehicles (int): The amount of vehicles that should be generated
            routes_list (List): A list including all available route ids
        """
        vehicles_dict = {}
        for i in range(n_vehicles):
            vehicle_id = f"member_ev_{i}"
            vehicle_type = "DEFAULT_VEHTYPE"
            battery_capacity = BATTERY_MAX
            start_soc = random.randint(BATTERY_MIN, BATTERY_MAX)
            route_id = random.choice(routes_list)
            depart_time = None # Making use of depart time is not yet implemented
            
            vehicles_dict[vehicle_id] = {
                "type": vehicle_type,
                "capacity": battery_capacity,
                "soc": start_soc,
                "route": route_id,
                "depart_time": depart_time
            }

        return vehicles_dict

class BasicScenario(ScenarioGenerator):
    """
    Vehicles all start at E0 and end at E19, starting directly one after the other. Start battery values are variable.
    """