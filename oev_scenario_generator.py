import random
import datetime
import json
import logging

# configure logging
logger = logging.getLogger("rl.environment.scenario_generator")


DEFAULT_BATTERY_MIN = 200
DEFAULT_BATTERY_MAX = 500


BAST_DISTRIBUTION_PATH = "datasets/bast_data/relatives_verkehrsaufkommen_2022.json"


def json_keys_to_int(x):
    if isinstance(x, dict):
        new_dict = {}
        for key, value in x.items():
            try:
                new_key = int(key)
            except (ValueError, TypeError):
                new_key = key
            new_dict[new_key] = value
        return new_dict
    return x

class ScenarioGenerator():
    """
    Vehicles have different routes, where start and endpoint are random. Vehicles are starting directly one after the other on simulation start. Start battery values are random within start_soc_bounds.
    """

    def __init__(self, seed):

        self.rng = random.Random(seed)  # random.Random(None) works as well if no seed is provided.

    
    def _select_soc(self, start_soc_bounds):
        soc = self.rng.randint(start_soc_bounds[0], start_soc_bounds[1])
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
        # Filter routes to only include those starting from main highway edges (not charging station edges)
        # This ensures vehicles start on the main highway and travel meaningful distances
        main_highway_routes = [route for route in routes_list 
                              if route["from"].startswith("e") and route["from"] != route["to"]
                              and route.get("length", 0) > 10000]  # Filter for routes longer than 10km
        
        if len(main_highway_routes) < amount:
            logger.warning(f"Not enough long highway routes available. Requested {amount}, but only {len(main_highway_routes)} highway routes available. Using all available routes as fallback.")
            # Fallback to all valid routes if not enough highway routes
            valid_routes = [route for route in routes_list if route.get("length", 0) > 0]
            filtered_routes = valid_routes
        else:
            filtered_routes = main_highway_routes
        
        if len(filtered_routes) < amount:
            # If still not enough, take what we have and log a warning
            logger.warning(f"Using {len(filtered_routes)} routes instead of requested {amount}")
            amount = len(filtered_routes)
            
        selected_routes = self.rng.sample(filtered_routes, amount)
        routes_dict = {f"trip{i}": [item["from"], item["to"]] for i, item in enumerate(selected_routes)}
        return routes_dict


    def _get_depart_time_list(self, n_vehicles, scenario_id=None):
        """The default implementation always returns an empty list."""
        return []
    
    def _select_depart_time(self, depart_time_iter):
        """The default implementation returns always 0."""
        return 0

    def generate_vehicles(self, n_vehicles, routes_list, start_soc_bounds=(DEFAULT_BATTERY_MIN, DEFAULT_BATTERY_MAX), scenario_id=None):
        """
        Params:
            n_vehicles (int): The amount of vehicles that should be generated
            routes_list (List): A list including all available route ids
            start_soc_bounds (Tuple): Optional. Outer bounds for the start soc value in the shape (MIN_VALUE, MAX_VALUE)
            scenario_id (str): Optional. Scenario ID to reproduce a certain scenario.
        Returns:
            vehicles_dict (dict): A dictionary with all generated vehicles and their attributes.
        """
        depart_time_iter = iter(self._get_depart_time_list(n_vehicles, scenario_id)) # initialize an iterator for depart times

        vehicles_dict = {}

        for i in range(n_vehicles):
            depart_time = self._select_depart_time( depart_time_iter) # TODO: Is depart_time already implemented in simulation? i.e. is the value we are passing here used?
            
            if depart_time is None:
                print(f"Warning: No departure time available for vehicle {i}. Stopping vehicle generation.")
                break
            vehicle_id = f"observable_ev_{i}"
            vehicle_type = "soulEV65"
            route_id = self.rng.choice(routes_list)
            battery_capacity = start_soc_bounds[1]
            start_soc = self._select_soc(start_soc_bounds)
            
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
    All Vehicles have the same route. Vehicles all start at the first edge and end at the last edge of the network. Vehicles are starting directly one after the other on simulation start. Start battery values are random within start_soc_bounds.
    """
    def __init__(self, seed=None):
        super().__init__(seed)
        self._cached_route = None


    def generate_routes_from_routes_list(self, amount, routes_list):
        """All vehicles share the same route: the longest available route in the network."""
        if self._cached_route is None:
            valid_routes = [r for r in routes_list if r.get("length", 0) > 0]
            if not valid_routes:
                raise ValueError("No valid routes available in routes_list.")
            longest = max(valid_routes, key=lambda r: r.get("length", 0))
            self._cached_route = (longest["from"], longest["to"])
        return {f"trip{i}": list(self._cached_route) for i in range(amount)}
    

class CustomDistributionScenario(ScenarioGenerator):
    """
    Vehicles have different routes, where start and endpoint are random. Vehicles are starting according to a dataset which needs to be specified by subclassing and overriding '_get_depart_time_list'. Start battery values are variable.
    """

    def _get_depart_time_list(self, n_vehicles, scenario_id=None):
        """
        Constructs a list of departure times based on a custom distribution.
        
        Parameters:
            n_vehicles (int): The number of vehicles to generate.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _construct_depart_time_list_from_distribution(self, daily_rel_depart_time_distribution_dict, n_vehicles):
        """ 
        Constructs a list of departure times from the given dictionary of vehicle distributions over one day. 

        Parameters:
            daily_rel_depart_time_distribution_dict (dict): A dictionary mapping the relative amount of vehicles departing to each hour in a day (relative to the max. expected number of vehicles per day). The keys should be in the range [0, 24).
        """

        if not daily_rel_depart_time_distribution_dict:
            raise ValueError("When using a custom distribution scenario, depart_time_distribution_dict must be set before generating vehicles.")

        depart_time_list = []
        # Add vehicles to the departure time list based on the relative distribution
        for hour, rel_amount in daily_rel_depart_time_distribution_dict.items():

            if rel_amount > 1 or rel_amount < 0:
                raise ValueError(f"Relative amount of vehicles for hour {hour} must be in the range [0, 1]. Found: {rel_amount}")

            amount = round(rel_amount * n_vehicles)
            print(f"Hour {hour}: {amount} vehicles (relative amount: {rel_amount})")

            # Set a departure time for each vehicle in this hour
            for _ in range(int(amount)):
                hour_in_seconds = (hour - 1) * 3600
                minutes_in_seconds = self.rng.randint(0, 59) * 60
                depart_time_list.append(hour_in_seconds + minutes_in_seconds)
        return depart_time_list

class BAStDistributionScenario(CustomDistributionScenario):
    """
    Vehicles have different routes, where start and endpoint are random. Vehicles are starting according to BASt data. Start battery values are variable.
    """
    
    def __init__(self, seed=None):
        super().__init__(seed)
        with open(BAST_DISTRIBUTION_PATH, 'r') as file:
            self.depart_time_distribution_dict = json.load(file, object_hook=json_keys_to_int)

    def _random_date(self, start_year=2022, end_year=2022):
        """
        Generate a random date between Jan 1 of start_year and Dec 31 of end_year.
        """
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)
        delta_days = (end_date - start_date).days
        random_days = self.rng.randint(0, delta_days)
        return start_date + datetime.timedelta(days=random_days)

    def _get_depart_time_list(self, n_vehicles, scenario_id=None):
        """ Constructs a list of departure times based on the BASt distribution data. If no scenario_id is provided, a random date in 2022 is used to select the distribution. """
        if scenario_id is None:
            date = self._random_date(2022, 2022)
            year = (date.year)
            month = (date.month)
            day = (date.day)
        daily_rel_depart_time_distribution_dict = self.depart_time_distribution_dict[year][month][day]
        return self._construct_depart_time_list_from_distribution(daily_rel_depart_time_distribution_dict, n_vehicles)