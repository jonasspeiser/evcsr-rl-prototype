import random
import datetime
import json


DEFAULT_BATTERY_MIN = 200
DEFAULT_BATTERY_MAX = 500


BAST_DISTRIBUTION_PATH = "datasets/bast_data/relatives_verkehrsaufkommen_2022.json"

def random_date(start_year=2022, end_year=2022):
    """
    Generate a random date between Jan 1 of start_year and Dec 31 of end_year.
    """
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    delta_days = (end_date - start_date).days
    random_days = random.randint(0, delta_days)
    return start_date + datetime.timedelta(days=random_days)

def day_of_year_to_month_day(day_of_year, year):
    """
    Convert a day-of-year number to (month, day).
    
    Parameters:
        day_of_year (int): Day of the year (1-366)
        year (int): Year to account for leap years (default: current year)
        
    Returns:
        (int, int): (month, day)
    """
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
    return date.month, date.day

def month_day_to_day_of_year(month, day, year):
    """
    Convert (month, day) to day of the year.

    Args:
        month (int): Month (1-12)
        day (int): Day of the month (1-31 depending on month)
        year (int): Year to account for leap years (default: current year)

    Returns:
        int: Day of the year (1-366)
    """
    date = datetime.datetime(year, month, day)
    return date.timetuple().tm_yday

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

    def _get_depart_time_list(self, n_vehicles, scenario_id=None):
        """The default implementation always returns None."""
        return None
    
    def _select_depart_time(self, depart_time_iter):
        """The default implementation returns always 0."""
        return 0

    def generate_vehicles(self, n_vehicles, routes_list, start_soc_bounds=(DEFAULT_BATTERY_MIN, DEFAULT_BATTERY_MAX), scenario_id=None):
        """
        Params:
            n_vehicles (int): The amount of vehicles that should be generated
            routes_list (List): A list including all available route ids
        """
        depart_time_iter = iter(self._get_depart_time_list(n_vehicles, scenario_id)) # initialize an iterator for depart times

        vehicles_dict = {}

        for i in range(n_vehicles):
            depart_time = self._select_depart_time( depart_time_iter) # TODO: Is depart_time already implemented in simulation? i.e. is the value we are passing here used?
            
            if depart_time is None:
                print(f"Warning: No departure time available for vehicle {i}. Stopping vehicle generation.")
                break
            vehicle_id = f"member_ev_{i}"
            vehicle_type = "soulEV65"
            route_id = random.choice(routes_list)
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
                minutes_in_seconds = random.randint(0, 59) * 60
                depart_time_list.append(hour_in_seconds + minutes_in_seconds)
        return depart_time_list

    def _select_depart_time(self, depart_time_iter):
        """ Selects the next departure time from a predefined list of departure times. Returns None if the list is exhausted. """
        return next(depart_time_iter, None)

class BAStDistributionScenario(CustomDistributionScenario):
    """
    Vehicles have different routes, where start and endpoint are random. Vehicles are starting according to BASt data. Start battery values are variable.
    """
    
    def __init__(self, seed=None):
        super().__init__(seed)
        with open(BAST_DISTRIBUTION_PATH, 'r') as file:
            self.depart_time_distribution_dict = json.load(file, object_hook=json_keys_to_int)


    def _get_depart_time_list(self, n_vehicles, scenario_id=None):
        """ Constructs a list of departure times based on the BASt distribution data. If no scenario_id is provided, a random date in 2022 is used to select the distribution. """
        if scenario_id is None:
            date = random_date(2022, 2022)
            year = (date.year)
            month = (date.month)
            day = (date.day)
        daily_rel_depart_time_distribution_dict = self.depart_time_distribution_dict[year][month][day]
        return self._construct_depart_time_list_from_distribution(daily_rel_depart_time_distribution_dict, n_vehicles)