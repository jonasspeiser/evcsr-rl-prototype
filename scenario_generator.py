import random

DEFAULT_BATTERY_MIN = 200
DEFAULT_BATTERY_MAX = 500

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
    
    def _select_soc(self, start_soc_bounds):
        soc = random.randint(start_soc_bounds[0], start_soc_bounds[1])
        return soc

    def generate_routes_from_routes_list(self, amount, routes_list):
        """
        Generate exactly `amount` valid unique routes on the current SUMO network.
        Returns a dict mapping route_id (str) -> list_of_edge_ids (list of str).        
        Params:
            amount (int): The amount of routes that should be generated
            routes_list (list): A list of dicts containing possible routes in the network. Format:
                [{
                    "from": edge ID,
                    "to": edge ID,
                    "length": float,
                    "route": a list of edge IDs
                }, ...]
        """
        if len(routes_list) < amount:
            raise ValueError("Not enough routes available in routes_list.")
        
        selected_routes = random.sample(routes_list, amount)
        routes_dict = {f"trip{i}": [item["from"], item["to"]] for i, item in enumerate(selected_routes)}
        return routes_dict

    def generate_routes_from_edge_list(self, amount, edge_list):
        """
        Generate exactly `amount` valid routes on the current SUMO network.
        Returns a dict mapping route_id (str) -> list_of_edge_ids (list of str).        
        Params:
            amount (int): The amount of routes that should be generated
            edge_list (List): A list including all of the maps edges
            veh_type (str): Optional vehicle type ID to pass to findRoute (can be None).
        """
        routes_dict = {}
        for i in range(amount):
            route_id = f"trip{i}"
            start, end = self._select_route(edge_list)
            routes_dict[route_id] = [start, end]
        return routes_dict

    def generate_vehicles(self, n_vehicles, routes_list, start_soc_bounds=(DEFAULT_BATTERY_MIN, DEFAULT_BATTERY_MAX)):
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
            battery_capacity = start_soc_bounds[1]
            start_soc = self._select_soc(start_soc_bounds)
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
    def __init__(self, seed=None):
        super().__init__(seed)
        self._cached_route = None

    def _select_route(self, edge_list):
        """ Vehicles all start at the first edge and end at the last edge in edge_list. """
        if self._cached_route is None:
            if not edge_list:
                raise ValueError("edge_list must not be empty.")
            # Default: first to last
            self._cached_route = (edge_list[0], edge_list[-1])
        return self._cached_route
    
class SameSOCSameRouteScenario(SameRouteScenario):
    """
    All Vehicles have the same route. Vehicles all start at the first edge (E0) and end at the last edge (E19) of the network. Vehicles are starting directly one after the other. Start battery values are equal for all vehicles (50 % of BATTERY_MAX).
    """

    def _select_soc(self, start_soc_bounds):
        battery_max = start_soc_bounds[1]
        soc = battery_max / 2
        return soc
