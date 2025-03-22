import random

BATTERY_MIN = 200
BATTERY_MAX = 500

class ScenarioGenerator():
    """
    Vehicles have different routes, where start and endpoint are random. Vehicles are starting directly one after the other. Start battery values are variable.
    """

    def __init__(self, seed):
        if seed is not None:
            print("seed set ")
            random.seed(seed)

    def _select_route(self, edge_list):
        """
        Selects two UNIQUE edges randomly.
        """
        if len(edge_list) < 2:
            raise ValueError("edge_list must contain at least two elements.")
        return list(random.sample(edge_list, 2))
    
    def generate_routes(self, n_routes, edge_list):
        """
        Params:
            n_routes (int): The amount of routes that should be generated
            edge_list (List): A list including all of the maps edges
        """
        routes_dict = {}
        for i in range(n_routes):
            route_id = f"trip{i}"
            start, end = self._select_route(edge_list)
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
            route_id = random.choice(routes_list)
            battery_capacity = BATTERY_MAX
            start_soc = random.randint(BATTERY_MIN, BATTERY_MAX)
            depart_time = None # Making use of depart time is not yet implemented
            
            vehicles_dict[vehicle_id] = {
                "type": vehicle_type,
                "route": route_id,
                "capacity": battery_capacity,
                "soc": start_soc,
                "depart_time": depart_time
            }

        return vehicles_dict

class SameRouteScenario(ScenarioGenerator):
    """
    All Vehicles have the same route. Vehicles all start at the first edge (E0) and end at the last edge (E19) of the network. Vehicles are starting directly one after the other. Start battery values are variable.
    """

    def _select_route(self, edge_list):
        """ Vehicles all start at the first edge and end at the last edge in edge_list. """
        if not edge_list:
            raise ValueError("edge_list must not be empty.")
        return [edge_list[0], edge_list[-1]]
    