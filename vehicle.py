import numpy as np
from circle_simulation import Simulation, PointlessRecommendationError, BadTimingRoutingError, ImpossibleRoutingError

# configure logging
import logging
logger = logging.getLogger("rl.environment.vehicle")

def get_padded_observation():
    """
    Returns a padded observation for vehicles that are not present in the current observation.
    """
    return np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0], dtype=np.float32)

class Vehicle:

    EMPTY_SOC = 30 # value under which the battery should be considered empty by the environment (used for monitoring the number of empty vehicles) 
    MAX_POSSIBLE_CAPACITY = 100000.0 # 100000 Wh is considered as max. possible battery capacity, regardless of vehicle type (used for normalization)
    MAX_POSSIBLE_DISTANCE = None 
    """ MAX_POSSIBLE_DISTANCE is set once during runtime upon simulation start. Max. possible distance between a vehicles start and destination in meters (used for normalization)."""

    def __init__(self, vehicle_id, simulation:Simulation):
        self.vehicle_id = vehicle_id
        self.simulation = simulation
        self.last_action = -1
        self.spawned = False
        self.arrived = False
        self.empty = False
        self.departure_time = None
        self.arrival_time = None
        self.waiting_time = 0
        self.max_battery_capacity = None
        self.battery_soc = None
        self.relative_battery_soc = None
        self.distance_to_cs_dict = None  # Expected to be a dict {cs_id: distance}
        self.distance_to_destination = None

    def fetch_and_update_battery_values(self):
        self.battery_soc = self.simulation.get_battery_soc(self.vehicle_id)
        if self.battery_soc is None: # i.e. if the vehicle did not spawn in the simulation yet or despawned already
            return
        self.max_battery_capacity = self.simulation.get_max_battery_capacity(self.vehicle_id)
        self.relative_battery_soc = self.battery_soc / self.max_battery_capacity
        # only account for vehicles that just entered the state of low battery. Not the ones that where already low during the last step.
    
    def update_from_state(self, state):
        """
        Update vehicle properties from the simulation-provided state.
        If the vehicle has not spawned yet (i.e. state is None or missing key),
        we reset to default values.
        """
        if state is None or state.get("distance_to_cs") is None:
            self.battery_soc = None
            self.distance_to_cs_dict = None
            self.distance_to_destination = None
        else:
            self.spawned = True
            self.battery_soc = state.get("battery_soc")
            self.distance_to_cs_dict = state.get("distance_to_cs")
            self.distance_to_destination = state.get("distance_to_destination")

    def get_observation(self, is_active=False):
        """
        Returns the observation for this vehicle as a NumPy array.
        Observation format:
          [normalized_soc] + 4 normalized distances + [last_action] + [destination_reached] + [active_charging_request]
        If the vehicle is not spawned yet, a padded observation is returned.
        """
        vehicle_is_offline = self.distance_to_cs_dict is None
        if vehicle_is_offline:
            # Use -1 as padding to indicate that the vehicle is not spawned.
            return get_padded_observation()
        normalized_soc = self.battery_soc / self.MAX_POSSIBLE_CAPACITY if self.battery_soc is not None else -1 
        dist_destination = self.distance_to_destination / self.MAX_POSSIBLE_DISTANCE if self.distance_to_destination is not None else -1
        # Ensure a fixed order by sorting charging station ids; pad if needed.
        distances = [
            -1 if self.distance_to_cs_dict[k] is None # if distance is None, the target is not reachable. Thus return -1
            else self.distance_to_cs_dict[k] / self.MAX_POSSIBLE_DISTANCE # if reachable, normalise distance
            for k in sorted(self.distance_to_cs_dict.keys())
            ]
        while len(distances) < 4:
            distances.append(-1)
        # destination_reached flag (1 if arrived, 0 otherwise)
        destination_reached = int(self.arrived)
        # active charging request flag is provided via the is_active parameter
        obs = [normalized_soc] + [dist_destination] + distances[:4] + [self.last_action, destination_reached, int(is_active)]
        return np.array(obs, dtype=np.float32)

    def get_info(self):
        """
        Get additional info for logging, such as a human readable version of the vehicle state.
        """
        state = {
            "battery_soc": self.battery_soc, 
            "max_battery_capacity": self.max_battery_capacity, 
            "distance_to_cs": self.distance_to_cs_dict, 
            "arrival_time": self.arrival_time, 
            "empty": self.empty
            }
        # "vehicle_position": vehicle_edge, "vehicle_destination": vehicle_destination
        return state
    
    def handle_action(self, action, reward_strategy):
        """
        Handle the action (e.g. rerouting, removing stops) and then delegate
        the penalty calculation to the reward strategy.
        Returns an action penalty (if any).
        """
        if isinstance(action, tuple):
            # My evaluation algos return tuples. Only the first value is of interest.
            action = action[0]
        
        logger.info(f"{self.vehicle_id}: handling action {action}")
        self.last_action = action

        # Build a context dictionary with information useful for penalty calculation.
        context = {'action': action}

        if self.arrived:
            # If the vehicle already arrived, do nothing.
            return 0

        try:
            next_charging_stop = self.simulation.get_next_charging_stop_id(self.vehicle_id)
        except ValueError as e:
            logger.error(f"Handling action {action} for {self.vehicle_id} failed: {e}")
            return 0
        
        charging_stop_is_planned = next_charging_stop is not None

        if action in (1, 2, 3, 4): # action is "charge"
            charging_stations = self.simulation.get_all_charging_station_ids()
            cs_id = charging_stations[action - 1] # action 1 means: go to cs_0 -> action-1 gives us the list index
            context['target_cs'] = cs_id
            context['next_charging_stop'] = next_charging_stop

            if cs_id == next_charging_stop:
                logger.debug(f"Vehicle {self.vehicle_id}: charging stop {cs_id} is already planned")
                context['reroute_successful'] = False
                context['charging_stop_already_planned'] = True
                return 0
            try:
                self.simulation.reroute_for_charging(self.vehicle_id, cs_id)
                logger.debug(f"Vehicle {self.vehicle_id} rerouted for charging at {cs_id}")
                context.update({
                    'reroute_successful': True,
                    'sufficient_range': self.is_remaining_range_sufficient(buffer=0),
                }) 
            except (ImpossibleRoutingError, BadTimingRoutingError) as e:
                logger.error(e)
                context['reroute_successful'] = False
                context['rerouting_exception_occurred'] = True
            except PointlessRecommendationError as e:
                logger.error(e)
                context['reroute_successful'] = False
                context['recommendation_past_destination'] = True


        elif action == 0: # action is "do nothing"
            if charging_stop_is_planned:
                self.simulation.remove_charging_stop(self.vehicle_id)
                logger.debug(f"Vehicle {self.vehicle_id}: removed planned charging stop")

        # Delegate penalty calculation to the reward strategy.
        action_penalty = reward_strategy.calculate_action_penalty(self, context)
        return action_penalty

    def is_battery_empty(self):
        return self.battery_soc is not None and self.battery_soc <= self.EMPTY_SOC # if battery_soc is None, the vehicle is not currently online

    def battery_just_died(self):
        """
        Returns True if the battery just died in this step.
        Once marked, it will not be flagged again.
        """
        if self.empty:
            return False
        if self.is_battery_empty():
            logger.info(f"{self.vehicle_id}: battery empty")
            self.empty = True
            return True
        return False
    
    def is_remaining_range_sufficient(self, buffer=0):
        """ 
        Returns whether a vehicle's battery soc is enough to reach its destination.
        If the optional "buffer" parameter is set, it must have a battery soc higher than "buffer" when arriving, 
        otherwise it just has to be not completely empty. 
        Returns None if remaining range is None.
        """
        remaining_range = self.simulation.get_remaining_range(self.vehicle_id)
        if remaining_range is None:
            return None
        vehicle_edge = self.simulation._get_vehicle_edge(self.vehicle_id)
        distance_to_destination = self.simulation.get_distance_to_destination(self.vehicle_id, vehicle_edge)
        return remaining_range > (distance_to_destination + buffer)
    
    def get_total_travel_time(self):
        if not self.arrival_time:
            raise AttributeError(f"Arrival time not yet set for vehicle{self.vehicle_id}.")  
        return self.arrival_time - self.departure_time