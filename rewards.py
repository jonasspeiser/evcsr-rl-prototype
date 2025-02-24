# configure logging
import logging
logger = logging.getLogger("rl.environment.rewards")

MAX_ALLOWED_TTT = 100
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

    def calculate_final_reward(self, global_ttt):
        """
        Calculate the final reward at the end of an episode.
        
        Parameters:
            vehicles: List of all vehicles.
            current_time: The current simulation timestep.
            
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

class NoTimeComponentRewardStrategy(RewardStrategy):
    """
    This Reward Strategy only prevents vehicles from going empty but doesn't reward for shorter waiting- or travel times. It uses reward shaping to 
    - make convergence quicker
    - prevent the agent simply always recommending to charge, even if it's not necessary
    - prevent illegal actions (recommending charging station which is not reachable anymore)

    Step reward:
    A reward of -100 is given if a battery just died and +10 when a vehicle reaches its destination.
    Additionally, vehicles that are charging (and otherwise low on range) get an extra reward.

    Final reward:
    Calculate the final reward (total travel time) by summing differences
    between each vehicle’s departure and arrival times.

    Action penalties:
    - Illegal charging actions (vehicle doesn't exist anymore or is past the charging station)
    - Unnecessary charging actions
    """
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        reward = 0

        for vehicle in vehicles.values():
            vehicle_is_at_destination = vehicle.arrived
            vehicle_has_just_reached_destination = vehicle_is_at_destination and (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids)
            
            if vehicle.battery_just_died():
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST died (reward -100)")
                reward += -100
            elif vehicle_has_just_reached_destination:
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST reached destination (reward +10)")
                reward += 10

            if vehicle.vehicle_id in charging_ids:
                # every sumo step (i.e. every second) a vehicle is charging and needs to do so to arrive at its destination, the agent gets +1 reward
                # TODO: This may be a bit much. Maybe reduce the reward to 0.1 or 0.01 as it is played out per second
                remaining_range_is_sufficient = vehicle.is_remaining_range_sufficient(buffer=0)
                if not remaining_range_is_sufficient:
                    logger.debug(f"Vehicle {vehicle.vehicle_id} is charging with insufficient range (+1 reward)")
                    reward += 1

        return reward

    def calculate_final_reward(self, global_ttt):
        return 0

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
            if context.get('rerouting_exception_occurred', False):
                # penalize the agent for trying to take an illegal action (e.g. vehicle doesn't exist anymore or is past the charging station)
                logger.debug(f"Vehicle {vehicle.vehicle_id}: illegal charging action (penalty -1)")
                return -1
        return 0 # if action is "do nothing"

class BasicRewardStrategy(RewardStrategy):
    """
    This Reward Strategy only gives a reward at the end of an episode, containing the negative value for the objective which we want to minimize (total travel time).

    Step reward: 0

    Final reward:
    The negative total travel time.

    Action penalties: 0
    """
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        # This is just here to trigger the vehicle.battery_jus_died() function and therefore get the info if a vehicle died during the current step
        for vehicle in vehicles.values():
            if vehicle.battery_just_died():
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST died")
        return 0

    def calculate_final_reward(self, global_ttt):
        return -global_ttt
    
    def calculate_action_penalty(self, vehicle, context):
        return 0

class RewardShapingStrategy(RewardStrategy):
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        reward = 0

        for vehicle in vehicles.values():
            vehicle_is_at_destination = vehicle.arrived
            vehicle_has_just_reached_destination = vehicle_is_at_destination and (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids)
            
            if vehicle.battery_just_died():
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST died (reward -100)")
                reward += -100
            elif vehicle_has_just_reached_destination:
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST reached destination (reward k-TTT)")
                reward += MAX_ALLOWED_TTT - vehicle.get_total_travel_time()

            if vehicle.vehicle_id in charging_ids:
                # every sumo step (i.e. every second) a vehicle is charging and needs to do so to arrive at its destination, the agent gets +1 reward
                # TODO: This may be a bit much. Maybe reduce the reward to 0.1 or 0.01 as it is played out per second
                remaining_range_is_sufficient = vehicle.is_remaining_range_sufficient(buffer=0)
                if not remaining_range_is_sufficient:
                    logger.debug(f"Vehicle {vehicle.vehicle_id} is charging with insufficient range (+1 reward)")
                    reward += 1

        return reward

    def calculate_final_reward(self, global_ttt):
        return BasicRewardStrategy().calculate_final_reward(global_ttt)

    def calculate_action_penalty(self, vehicle, context):
        return NoTimeComponentRewardStrategy().calculate_action_penalty(vehicle, context)