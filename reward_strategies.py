# configure logging
import logging
logger = logging.getLogger("rl.environment.rewards")


def arrive_concurrently(dist_a_m, dist_b_m, threshold_m):
    """Return True if two vehicles are close enough in distance to a station that they will
    arrive within the same congestion window (approximated by distance at constant highway speed)."""
    return abs(dist_a_m - dist_b_m) < threshold_m


class RewardStrategy:
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        """
        Calculate the step reward.
        
        Parameters:
            vehicles: List of all vehicles.
            newly_arrived_ids: List of vehicle ids that just reached their destination.
            charging_ids: List of vehicle ids that are currently charging.
        
        Returns:
            A step reward value.
        """
        raise NotImplementedError("calculate_step_reward must be implemented in subclasses.")

    def calculate_final_reward(self, ttt_per_ev_mean):
        """
        Calculate the final reward at the end of an episode.
        
        Parameters:
            ttt_per_ev_mean: Mean travel time per vehicle (seconds).

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
                         - 'recommendation_past_destination': whether the recommended cs is further away than the vehicles destination
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

    def calculate_final_reward(self, ttt_per_ev_mean):
        return 0

    def calculate_action_penalty(self, vehicle, context):
        action = context.get('action')
        if action in (1, 2, 3, 4):
            # If the vehicle already had this charging stop planned or rerouting failed,
            # don't apply a penalty.
            if context.get('charging_stop_already_planned', False):
                return 0
            if context.get('sufficient_range', False):
                logger.info(f"Vehicle {vehicle.vehicle_id}: was asked to charge but has sufficient range (penalty -100)")
                return -100
            if context.get('rerouting_exception_occurred', False):
                # penalize the agent for trying to take an illegal action (e.g. vehicle doesn't exist anymore or is past the charging station)
                logger.info(f"Vehicle {vehicle.vehicle_id}: illegal charging action (penalty -100)")
                return -100
        return 0 # if action is "do nothing"

class BasicRewardStrategy(RewardStrategy):
    """
    This Reward Strategy only gives a reward at the end of an episode, containing the negative value for the objective which we want to minimize (total travel time).

    Step reward: 0

    Final reward:
    The negative mean travel time per vehicle.

    Action penalties: 0
    """
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        # This is just here to trigger the vehicle.battery_just_died() function and therefore get the info if a vehicle died during the current step
        for vehicle in vehicles.values():
            if vehicle.battery_just_died():
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST died")
        return 0

    def calculate_final_reward(self, ttt_per_ev_mean):
        return -(ttt_per_ev_mean / 1000)  # scaling reward down makes it easier for the agent to learn
    
    def calculate_action_penalty(self, vehicle, context):
        return 0


class BasicWithCongestionPenaltyStrategy(BasicRewardStrategy):
    """
    Extends BasicRewardStrategy with an action-time congestion penalty.

    When a vehicle is successfully routed to a CS, a penalty is applied for each other vehicle
    that is already heading to the same CS and will arrive within a similar time window
    (approximated by distance at constant highway speed).

    Step reward: 0 (inherited)

    Final reward: negative mean travel time (inherited)

    Action penalties:
    - congestion_penalty per conflicting vehicle whose distance to the target CS differs by less
      than congestion_threshold_m (default 33600 m ≈ 100 km/h × 20 min charging stop).

    Args:
            congestion_threshold_m: Distance window (meters) within which two vehicles heading to the
                same CS are considered to conflict. Default 33600 m ≈ 100 km/h × 20 min charging stop.
            congestion_penalty: Penalty applied per conflicting vehicle.
    """
    def __init__(self, congestion_threshold_m=33600, congestion_penalty=1.0):
        self.congestion_threshold_m = congestion_threshold_m
        self.congestion_penalty = congestion_penalty

    def calculate_action_penalty(self, vehicle, context):
        if not context.get('reroute_successful'):
            return 0
        target_cs = context.get('target_cs')
        dist_self = vehicle.distance_to_cs_dict.get(target_cs) if vehicle.distance_to_cs_dict else None
        if dist_self is None:
            return 0
        all_vehicles = context.get('all_vehicles', {})
        penalty = 0
        for other in all_vehicles.values():
            if other.vehicle_id == vehicle.vehicle_id:
                continue
            if other.arrived or other.empty or not other.spawned:
                continue
            if other.target_cs_id != target_cs:
                continue
            dist_other = other.distance_to_cs_dict.get(target_cs) if other.distance_to_cs_dict else None
            if dist_other is None:
                continue
            if arrive_concurrently(dist_self, dist_other, self.congestion_threshold_m):
                logger.info(f"Vehicle {vehicle.vehicle_id}: congestion penalty for routing to {target_cs} (conflict with {other.vehicle_id})")
                penalty -= self.congestion_penalty
        return penalty

class RewardShapingStrategy(RewardStrategy):
    def __init__(self, max_allowed_ttt):
        self.max_allowed_ttt = max_allowed_ttt

    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids):
        reward = 0

        for vehicle in vehicles.values():
            vehicle_is_at_destination = vehicle.arrived
            vehicle_has_just_reached_destination = vehicle_is_at_destination and (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids)
            
            # if vehicle.battery_just_died():
            #     logger.info(f"Vehicle {vehicle.vehicle_id} JUST died (reward -100)")
            #     reward += -100
            if vehicle_has_just_reached_destination:
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST reached destination (reward k-TTT)")
                reward += self.max_allowed_ttt - vehicle.get_total_travel_time()

            if vehicle.vehicle_id in charging_ids:
                # every sumo step (i.e. every second) a vehicle is charging and needs to do so to arrive at its destination, the agent gets +1 reward
                # TODO: This may be a bit much. Maybe reduce the reward to 0.1 or 0.01 as it is played out per second
                remaining_range_is_sufficient = vehicle.is_remaining_range_sufficient(buffer=0)
                if not remaining_range_is_sufficient:
                    logger.info(f"Vehicle {vehicle.vehicle_id} is charging with insufficient range (+100 reward)")
                    reward += 1

        return reward

    def calculate_final_reward(self, ttt_per_ev_mean):
        return BasicRewardStrategy().calculate_final_reward(ttt_per_ev_mean)

    def calculate_action_penalty(self, vehicle, context):
        penalty = NoTimeComponentRewardStrategy().calculate_action_penalty(vehicle, context)
        penalty += BasicWithCongestionPenaltyStrategy(congestion_threshold_m=33600, congestion_penalty=500).calculate_action_penalty(vehicle, context)
        return penalty