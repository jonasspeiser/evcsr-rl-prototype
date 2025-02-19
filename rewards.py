# configure logging
import logging
logger = logging.getLogger("rl.environment.rewards")

class RewardStrategy:
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        """
        Calculate the step reward.
        
        Parameters:
            vehicles: List of all vehicles.
            newly_arrived_ids: List of vehicle ids that just reached their destination.
            charging_ids: List of vehicle ids that are currently charging.
        
        Returns:
            A tuple (reward, one_vehicle_just_died, all_vehicles_at_destination)
        """
        raise NotImplementedError("calculate_step_reward must be implemented in subclasses.")

    def calculate_final_reward(self, vehicles, current_time):
        """
        Calculate the final reward at the end of an episode.
        
        Parameters:
            vehicles: List of all vehicles.
            
        Returns:
            A final reward value.
        """
        raise NotImplementedError("calculate_final_reward must be implemented in subclasses.")

    def calculate_action_penalty(self, vehicle, context):
        """
        Compute an action penalty based on the context.
        
        Parameters:
            vehicle: The vehicle that executed the action.
            simulation: The simulation instance.
            context: A dictionary with information about the action handling.
                     For example, it might contain:
                         - 'action': the action that was taken
                         - 'target_cs': the charging station the vehicle tried to go to
                         - 'next_charging_stop': the vehicle’s currently planned stop (if any)
                         - 'reroute_successful': whether rerouting succeeded (True/False)
                         - 'charging_stop_already_planned': whether a charging stop at the decided station was already planned in the last action
                         - 'sufficient_range': whether the vehicle has sufficient range
                         - 'rerouting_exception_occurred': whether an exception occurred during rerouting
        Returns:
            A numeric penalty (e.g., -1 for an undesired action, 0 for no penalty).
        """
        raise NotImplementedError

class BasicRewardStrategy(RewardStrategy):
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        """
        Calculate the step reward by iterating over all vehicles.
        A reward of -100 is given if a battery just died and +10 when a vehicle reaches its destination.
        Additionally, vehicles that are charging (and otherwise low on range) give an extra reward.
        """
        reward = 0
        one_vehicle_just_died = False
        all_vehicles_at_destination = True

        for vehicle in vehicles.values():
            vehicle_just_died = vehicle.battery_just_died()
            vehicle_is_at_destination = vehicle.arrived
            vehicle_has_just_reached_destination = vehicle_is_at_destination and (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids)
            
            if vehicle_just_died:
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST died (reward -100)")
                reward += -100
            elif vehicle_has_just_reached_destination:
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST reached destination (reward +10)")
                reward += 10

            if vehicle_just_died:
                one_vehicle_just_died = True
            if not vehicle_is_at_destination:
                all_vehicles_at_destination = False

            if vehicle.vehicle_id in charging_ids:
                # every sumo step (i.e. every second) a vehicle is charging and needs to do so to arrive at its destination, the agent gets +1 reward
                # TODO: This may be a bit much. Maybe reduce the reward to 0.1 or 0.01 as it is played out per second
                remaining_range_is_sufficient = vehicle.remaining_range_is_sufficient(buffer=0)
                if not remaining_range_is_sufficient:
                    logger.debug(f"Vehicle {vehicle.vehicle_id} is charging with insufficient range (+1 reward)")
                    reward += 1

        return reward, one_vehicle_just_died, all_vehicles_at_destination

    def calculate_final_reward(self, vehicles, current_time):
        """
        Calculate the final reward (e.g. total travel time) by summing differences
        between each vehicle’s departure and arrival times.
        This reward is only given at the end of an episode.
        """
        global_ttt = 0
        for vehicle in vehicles.values():
            if vehicle.departure_time is None:
                continue
            if vehicle.arrival_time is None:
                vehicle.arrival_time = current_time
            global_ttt += (vehicle.arrival_time - vehicle.departure_time)
        return global_ttt

    def calculate_action_penalty(self, vehicle, context):
        action = context.get('action')
        if action in (1, 2, 3, 4):
            # If the vehicle already had this charging stop planned or rerouting failed,
            # don't apply a penalty.
            if context.get('charging_stop_already_planned', False):
                return 0
            if context.get('sufficient_range', False):
                logger.debug(f"Vehicle {vehicle.vehicle_id}: was asked to charge but has sufficient range (penalty -1)")
                return -1
            if context.get('rerouting_exception_ocurred', False):
                # penalize the agent for trying to take an illegal action (e.g. vehicle doesn't exist anymore or is past the charging station)
                logger.debug(f"Vehicle {vehicle.vehicle_id}: illegal charging action (penalty -1)")
                return -1
        return 0 # if action is "do nothing"

class RewardOnlyPreventEmpty(RewardStrategy):
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        # Example: only penalize vehicles whose battery just died.
        reward = 0
        one_vehicle_just_died = False
        all_vehicles_at_destination = True

        for vehicle in vehicles.values():
            if vehicle.battery_just_died():
                logger.info(f"Vehicle {vehicle.vehicle_id} battery died (penalty -100)")
                reward += -100
                one_vehicle_just_died = True
            if not vehicle.arrived:
                all_vehicles_at_destination = False

        return reward, one_vehicle_just_died, all_vehicles_at_destination

    def calculate_final_reward(self, vehicles, current_time):
        # For this strategy, you might choose a different final reward.
        return 0


class RewardShaping(RewardStrategy):
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        # Example: shaped rewards that combine multiple signals.
        reward = 0
        one_vehicle_just_died = False
        all_vehicles_at_destination = True

        for vehicle in vehicles.values():
            if vehicle.battery_just_died():
                reward -= 50  # lesser penalty than BasicRewardStrategy
                one_vehicle_just_died = True
            if vehicle.arrived and (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids):
                reward += 5  # smaller bonus for reaching destination
            if vehicle.vehicle_id in charging_ids:
                # reward for charging if the vehicle is in danger of running empty
                remaining_range_is_sufficient = vehicle.remaining_range_is_sufficient(buffer=0)
                if not remaining_range_is_sufficient:
                    reward += 2

            if not vehicle.arrived:
                all_vehicles_at_destination = False

        return reward, one_vehicle_just_died, all_vehicles_at_destination

    def calculate_final_reward(self, vehicles, current_time):
        # A shaped final reward could also incorporate other metrics.
        return BasicRewardStrategy().calculate_final_reward(vehicles, current_time)
