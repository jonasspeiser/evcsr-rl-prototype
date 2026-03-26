import numpy as np
from simulation import Simulation, PointlessRecommendationError, BadTimingRoutingError, ImpossibleRoutingError

# configure logging
import logging
logger = logging.getLogger("rl.environment.vehicle")


class Vehicle:

    EMPTY_SOC = 30 
    """ Value in Wh under which the battery should be considered empty by the environment (used for monitoring the number of empty vehicles)"""
    DISTANCE_NORMALIZATION_VALUE = None 
    """ DISTANCE_NORMALIZATION_VALUE is set once during runtime upon simulation start. Max. possible distance between a vehicles start and destination in meters (used for normalization)."""
    CAPACITY_NORMALIZATION_VALUE = 100000.0 
    """100000 Wh is considered as max. possible battery capacity, regardless of vehicle type (used for normalization)"""

    def __init__(self, vehicle_id, simulation:Simulation):
        self.vehicle_id = vehicle_id
        self.simulation = simulation
        self.last_action = -1
        self.target_cs_id = None  # CS the vehicle is currently routed to (None if no charging stop planned)
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
        self.had_bad_timing_error = False

    def fetch_and_update_battery_values(self):
        # Ignore if the vehicle did not spawn in the simulation yet or despawned already
        if not self.spawned or self.arrived or self.empty:
            self.battery_soc = None
            return
        self.battery_soc = self.simulation.update_vehicle_soc(self.vehicle_id)
        if self.battery_soc is None:
            logger.warning(f"{self.vehicle_id}: Unwanted behaviour: Simulation responded with battery_soc=None although python flags are spawned=True, empty=False, arrived=False.")
            return
        self.max_battery_capacity = self.simulation.get_max_battery_capacity(self.vehicle_id)
        self.relative_battery_soc = self.battery_soc / self.max_battery_capacity
    
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
            self.spawned = True

    def get_observation(self):
        """
        Returns the observation for this vehicle as a NumPy array of shape (12,).
        Observation format (12 features):
          [soc, dist_destination, dist_cs_1..4, last_action_one_hot_0..4, arrived]
        If the vehicle is not spawned yet, returns zeros (the environment sets vehicle_mask=0).
        """
        if self.distance_to_cs_dict is None:
            return np.zeros(12, dtype=np.float32)
        normalized_soc = self.battery_soc / self.CAPACITY_NORMALIZATION_VALUE if self.battery_soc is not None else -1
        dist_destination = self.distance_to_destination / self.DISTANCE_NORMALIZATION_VALUE if self.distance_to_destination is not None else -1
        # Ensure a fixed order by sorting charging station ids; pad with -1 if needed.
        distances = [
            -1 if self.distance_to_cs_dict[k] is None
            else self.distance_to_cs_dict[k] / self.DISTANCE_NORMALIZATION_VALUE
            for k in sorted(self.distance_to_cs_dict.keys())
        ]
        while len(distances) < 4:
            distances.append(-1)
        # One-hot encode last_action (0=do_nothing, 1-4=charging station); -1 (no action yet) → all zeros
        last_action_oh = np.zeros(5, dtype=np.float32)
        if 0 <= self.last_action <= 4:
            last_action_oh[self.last_action] = 1.0
        return np.concatenate([
            [normalized_soc, dist_destination],
            distances[:4],
            last_action_oh,
            [float(self.arrived)],
        ]).astype(np.float32)

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
    
    def handle_action(self, action, reward_strategy, extra_context=None):
        """
        Handle the action (e.g. rerouting, removing stops) and then delegate
        the penalty calculation to the reward strategy.
        Returns an action penalty (if any).
        """
        if isinstance(action, tuple):
            # My evaluation algos return tuples. Only the first value is of interest.
            action = action[0]
        
        logger.debug(f"{self.vehicle_id}: handling action {action}")
        self.last_action = action

        # Build a context dictionary with information useful for penalty calculation.
        context = {'action': action}

        if self.arrived:
            # If the vehicle already arrived, do nothing.
            return 0

        try:
            next_charging_stop = self.simulation.get_next_charging_stop_id(self.vehicle_id)
        except ValueError as e:
            logger.error(f"{type(e).__name__}: Handling action {action} for {self.vehicle_id} failed: {e}")
            return 0
        
        charging_stop_is_planned = next_charging_stop is not None

        if action in (1, 2, 3, 4): # action is "charge"
            charging_stations = self.simulation.get_all_charging_station_ids()
            cs_id = charging_stations[action - 1] # action 1 means: go to cs_1 -> action-1 gives us the list index
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
                self.target_cs_id = cs_id
                context.update({
                    'reroute_successful': True,
                    'sufficient_range': self.is_remaining_range_sufficient(buffer=0),
                })
            except (ImpossibleRoutingError, BadTimingRoutingError) as e:
                logger.error(f"{type(e).__name__}: {e}")
                context['reroute_successful'] = False
                context['rerouting_exception_occurred'] = True
                if isinstance(e, BadTimingRoutingError):
                    self.had_bad_timing_error = True
            except PointlessRecommendationError as e:
                logger.error(f"{type(e).__name__}: {e}")
                context['reroute_successful'] = False
                context['recommendation_past_destination'] = True


        elif action == 0: # action is "do nothing"
            self.target_cs_id = None
            if charging_stop_is_planned:
                self.simulation.remove_charging_stop(self.vehicle_id)
                logger.debug(f"Vehicle {self.vehicle_id}: removed planned charging stop")

        # Delegate penalty calculation to the reward strategy.
        if extra_context:
            context.update(extra_context)
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