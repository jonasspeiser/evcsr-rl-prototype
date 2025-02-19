import numpy as np

# configure logging
import logging
logger = logging.getLogger("rl.environment.vehicle")

MAX_POSSIBLE_CAPACITY = 100000.0 # 100000 Wh is considered as max. possible battery capacity (used for normalization)
MAX_POSSIBLE_DISTANCE = 2000.0 # 2000 km is considered as max. possible distance between a vehicles start and destination (used for normalization)
EMPTY_SOC = 100 # value under which the battery should be considered empty
class Vehicle:
    def __init__(self, vehicle_id, simulation):
        self.vehicle_id = vehicle_id
        self.simulation = simulation
        self.last_action = -1
        self.arrived = False
        self.empty = False
        self.low_battery = False
        self.departure_time = None
        self.arrival_time = None
        self.waiting_time = 0
        self.battery_soc = None
        self.distance_to_cs = None  # Expected to be a dict {cs_id: distance}

    def update_from_state(self, state):
        """
        Update vehicle properties from the simulation-provided state.
        If the vehicle has not spawned yet (i.e. state is None or missing key),
        we reset to default values.
        """
        if state is None or state.get("distance_to_cs") is None:
            self.battery_soc = None
            self.distance_to_cs = None
        else:
            self.battery_soc = state.get("battery_soc")
            self.distance_to_cs = state.get("distance_to_cs")

    def get_observation(self, is_active=False):
        """
        Returns the observation for this vehicle as a NumPy array.
        Observation format:
          [normalized_soc] + 4 normalized distances + [last_action] + [destination_reached] + [active_charging_request]
        If the vehicle is not spawned yet, a padded observation is returned.
        """
        if self.distance_to_cs is None:
            # Use -1 as padding to indicate that the vehicle is not spawned.
            return np.array([-1, -1, -1, -1, -1, -1, 0, 0], dtype=np.float32)
        normalized_soc = self.battery_soc / MAX_POSSIBLE_CAPACITY if self.battery_soc is not None else -1 
        # Ensure a fixed order by sorting charging station ids; pad if needed.
        distances = [self.distance_to_cs[k] / MAX_POSSIBLE_DISTANCE for k in sorted(self.distance_to_cs.keys())]
        while len(distances) < 4:
            distances.append(-1)
        # destination_reached flag (1 if arrived, 0 otherwise)
        destination_reached = int(self.arrived)
        # active charging request flag is provided via the is_active parameter
        obs = [normalized_soc] + distances[:4] + [self.last_action, destination_reached, int(is_active)]
        return np.array(obs, dtype=np.float32)

    def handle_action(self, action, reward_strategy):
        """
        Handle the action (e.g. rerouting, removing stops) and then delegate
        the penalty calculation to the reward strategy.
        Returns an action penalty (if any).
        """
        logger.debug(f"Vehicle {self.vehicle_id}: handling action {action}")
        self.last_action = action

        # Build a context dictionary with information useful for penalty calculation.
        context = {'action': action}

        if self.arrived:
            # If the vehicle already arrived, do nothing.
            return 0

        next_charging_stop = self.simulation.get_next_charging_stop_id(self.vehicle_id)
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
                context['reroute_successful'] = True
                context['sufficient_range'] = self.is_remaining_range_sufficient(buffer=0)    
            except ValueError as e: # if the vehicle is past the charging station and rerouting doesn't work
                logger.error(e)
                context['reroute_successful'] = False
                context['rerouting_exception_occurred'] = True


        elif action == 0: # action is "do nothing"
            if charging_stop_is_planned:
                self.simulation.remove_charging_stop(self.vehicle_id)
                logger.debug(f"Vehicle {self.vehicle_id}: removed planned charging stop")

        # Delegate penalty calculation to the reward strategy.
        action_penalty = reward_strategy.calculate_action_penalty(self, context)
        return action_penalty

    def is_battery_empty(self):
        return self.battery_soc is not None and self.battery_soc <= EMPTY_SOC # if battery_soc is None, the vehicle is not currently online

    def battery_just_died(self):
        """
        Returns True if the battery just died in this step.
        Once marked, it will not be flagged again.
        """
        if self.empty:
            return False
        if self.is_battery_empty():
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
        distance_to_destination = self.simulation.get_distance_to_destination(self.vehicle_id)
        return remaining_range > (distance_to_destination + buffer)
    
    def get_total_travel_time(self):
        if not self.arrival_time:
            raise AttributeError(f"Arrival time not yet set for vehicle{self.vehicle_id}.")  
        return self.arrival_time - self.departure_time