import random

class ScenarioGenerator():
    #TODO: Weiß nicht, ob ich wirklich eine Klasse brauche...
    def __init__(self, seed):
        self.seed = seed

    def generate_routes(self, n_vehicles, edge_list):
        pass

    def generate_vehicles(self, n_vehicles, routes_list):
        pass

    vehicles = {
        vehicle_id: {
            "vehicle_type": vehicle_type,
            "battery_capacity": battery_capacity,
            "start_soc": start_soc,
            "route": route_id,
            "depart_time": depart_time
        }
    }
