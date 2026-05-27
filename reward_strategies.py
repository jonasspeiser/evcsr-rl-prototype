# configure logging
import logging
logger = logging.getLogger("rl.environment.rewards")


def arrive_concurrently(dist_a_m, dist_b_m, threshold_m):
    """Return True if two vehicles are close enough in distance to a station that they will
    arrive within the same congestion window (approximated by distance at constant highway speed)."""
    return abs(dist_a_m - dist_b_m) < threshold_m


def congestion_penalty(vehicle, context, congestion_threshold_m, congestion_penalty):
    """Return the total congestion penalty for routing a vehicle to a charging station.

    Sums a penalty for each other live vehicle that is already heading to the same station and
    will arrive within the same congestion window (approximated by distance threshold).

    Returns 0 if rerouting was not successful or distance information is unavailable.
    """
    if not context.get('reroute_successful'):
        return 0
    target_cs = context.get('target_cs')
    dist_self = vehicle.distance_to_cs_dict.get(target_cs) if vehicle.distance_to_cs_dict else None
    if dist_self is None:
        return 0
    penalty = 0
    for other in context['all_vehicles'].values():
        if other.vehicle_id == vehicle.vehicle_id or not other.is_online:
            continue
        if other.target_cs_id != target_cs:
            continue
        dist_other = other.distance_to_cs_dict.get(target_cs) if other.distance_to_cs_dict else None
        if dist_other is None:
            continue
        if arrive_concurrently(dist_self, dist_other, congestion_threshold_m):
            logger.info(f"Vehicle {vehicle.vehicle_id}: congestion penalty for routing to {target_cs} (conflict with {other.vehicle_id})")
            penalty -= congestion_penalty
    return penalty

def destination_reward(vehicle, newly_arrived_ids, max_allowed_ttt):
    """Return the reward for a vehicle that just reached its destination, 0 otherwise."""
    if not (vehicle.arrived and newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids):
        return 0
    reward = max_allowed_ttt - vehicle.get_total_travel_time()
    logger.info(f"Vehicle {vehicle.vehicle_id} JUST reached destination (reward k-TTT)")
    # Normalize with the same value as the end of episode reward to keep the same scale and make it easier for the agent to learn
    reward /= 3000
    # Make smaller than end of episode reward
    reward /= 10
    return reward


def relative_destination_reward(vehicle, newly_arrived_ids, scale_s):
    """Return the overhead penalty for a vehicle that just arrived, 0 otherwise.

    Reward = -(actual_ttt - ideal_ttt) / scale_s, where ideal_ttt is the free-flow travel time
    estimate obtained from the routing API at spawn. Returns 0 if the vehicle has not just arrived
    or if ideal_ttt is unavailable.
    """
    if not (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids):
        return 0
    ideal = vehicle.get_ideal_travel_time()
    if ideal is None:
        return 0
    overhead = vehicle.get_total_travel_time() - ideal
    logger.info(f"Vehicle {vehicle.vehicle_id} arrived: overhead={overhead:.0f}s, reward={-overhead/scale_s:.4f}")
    return -overhead / scale_s


def battery_penalty(vehicle, newly_emptied_ids, penalty=1.0):
    """Return a penalty if the vehicle's battery just died this step, 0 otherwise.

    Args:
        penalty: Magnitude of the penalty (returned as negative). Default 1.0.
                 Increase relative to the expected charging overhead to widen the
                 incentive gap; a 5-7× ratio over typical overhead is a good starting point.
    """
    if vehicle.vehicle_id not in newly_emptied_ids:
        return 0
    logger.info(f"Vehicle {vehicle.vehicle_id} JUST died (penalty -{penalty})")
    return -penalty


def illegal_action_penalty(vehicle, context):
    """Return a penalty if the agent recommended an unreachable charging station, 0 otherwise."""
    action = context.get('action')
    if action not in (1, 2, 3, 4):
        return 0
    if context.get('charging_stop_already_planned', False):
        return 0
    if context.get('rerouting_exception_occurred', False):
        logger.info(f"Vehicle {vehicle.vehicle_id}: illegal charging action (penalty -0.01)")
        return -0.01
    return 0


def charging_reward(vehicle, charging_ids):
    """Return the per-step reward for a vehicle that is charging with insufficient range, 0 otherwise."""
    if vehicle.vehicle_id not in charging_ids:
        return 0
    if vehicle.is_remaining_range_sufficient(buffer=0):
        return 0
    logger.info(f"Vehicle {vehicle.vehicle_id} is charging with insufficient range (+1 reward)")
    return 0.01

class RewardStrategy:
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids, newly_emptied_ids):
        """
        Calculate the step reward.

        Parameters:
            vehicles: List of all vehicles.
            newly_arrived_ids: List of vehicle ids that just reached their destination.
            charging_ids: List of vehicle ids that are currently charging.
            newly_emptied_ids: Set of vehicle ids whose battery just ran empty this step.

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
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids, newly_emptied_ids):
        reward = 0

        for vehicle in vehicles.values():
            vehicle_is_at_destination = vehicle.arrived
            vehicle_has_just_reached_destination = vehicle_is_at_destination and (newly_arrived_ids and vehicle.vehicle_id in newly_arrived_ids)

            if vehicle.vehicle_id in newly_emptied_ids:
                logger.info(f"Vehicle {vehicle.vehicle_id} JUST died (reward -1)")
                reward += -1
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
                logger.info(f"Vehicle {vehicle.vehicle_id}: was asked to charge but has sufficient range (penalty -0.01)")
                return -0.01
            if context.get('rerouting_exception_occurred', False):
                # penalize the agent for trying to take an illegal action (e.g. vehicle doesn't exist anymore or is past the charging station)
                logger.info(f"Vehicle {vehicle.vehicle_id}: illegal charging action (penalty -0.01)")
                return -0.01
        return 0 # if action is "do nothing"

class BasicRewardStrategy(RewardStrategy):
    """
    This Reward Strategy only gives a reward at the end of an episode, containing the negative value for the objective which we want to minimize (total travel time).

    Step reward: 0

    Final reward:
    The negative mean travel time per vehicle.

    Action penalties: 0
    """
    def calculate_step_reward(self, _vehicles, _newly_arrived_ids, _charging_ids, __newly_emptied_ids):
        return 0

    def calculate_final_reward(self, ttt_per_ev_mean):
        return -(ttt_per_ev_mean / 3000)  # scaling reward down makes it easier for the agent to learn
    
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
      than congestion_threshold_m (default 36000 m ≈ 100 km/h × 22 min charging stop (150 kW / 0.95 efficiency)).

    Args:
            congestion_threshold_m: Distance window (meters) within which two vehicles heading to the
                same CS are considered to conflict. Default 36000 m ≈ 100 km/h × 22 min charging stop (150 kW / 0.95 efficiency).
            congestion_penalty: Penalty applied per conflicting vehicle.
    """
    def __init__(self, congestion_threshold_m=36000, congestion_penalty=1.0):
        self.congestion_threshold_m = congestion_threshold_m
        self.congestion_penalty = congestion_penalty

    def calculate_action_penalty(self, vehicle, context):
        return congestion_penalty(vehicle, context, self.congestion_threshold_m, self.congestion_penalty)

class BasicWithDestinationRewardStrategy(BasicRewardStrategy):
    """
    Extends BasicRewardStrategy with a per-arrival destination reward.

    Step reward: destination_reward per vehicle that just arrived.
    Final reward: negative mean travel time (inherited).
    Action penalties: 0 (inherited).

    Args:
        max_allowed_ttt: Upper bound on travel time (seconds). A vehicle arriving in less time
            yields a positive reward; one exceeding it yields a negative reward.
    """
    def __init__(self, max_allowed_ttt):
        self.max_allowed_ttt = max_allowed_ttt

    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids, newly_emptied_ids):
        reward = super().calculate_step_reward(vehicles, newly_arrived_ids, charging_ids, newly_emptied_ids)
        for vehicle in vehicles.values():
            reward += destination_reward(vehicle, newly_arrived_ids, self.max_allowed_ttt)
        return reward


class BasicWithChargingRewardStrategy(BasicRewardStrategy):
    """
    Extends BasicRewardStrategy with a per-step reward for necessary charging.

    Step reward: charging_reward per vehicle that is charging with insufficient range.
    Final reward: negative mean travel time (inherited).
    Action penalties: 0 (inherited).
    """
    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids, newly_emptied_ids):
        reward = super().calculate_step_reward(vehicles, newly_arrived_ids, charging_ids, newly_emptied_ids)
        for vehicle in vehicles.values():
            reward += charging_reward(vehicle, charging_ids)
        return reward


class BasicWithShapingStrategy(BasicRewardStrategy):
    """
    Extends BasicRewardStrategy with destination reward, charging reward, and congestion penalty.

    Step reward: destination_reward + charging_reward per vehicle.
    Final reward: negative mean travel time (inherited).
    Action penalties: congestion_penalty per conflicting vehicle heading to the same station.

    Args:
        max_allowed_ttt: Upper bound on travel time used by destination_reward.
        congestion_threshold_m: Distance window within which two vehicles are considered to conflict.
        congestion_penalty_value: Penalty applied per conflicting vehicle.
    """
    def __init__(self, max_allowed_ttt, congestion_threshold_m=36000, congestion_penalty_value=1.0):
        self.max_allowed_ttt = max_allowed_ttt
        self.congestion_threshold_m = congestion_threshold_m
        self.congestion_penalty_value = congestion_penalty_value

    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids, newly_emptied_ids):
        reward = super().calculate_step_reward(vehicles, newly_arrived_ids, charging_ids, newly_emptied_ids)
        for vehicle in vehicles.values():
            reward += destination_reward(vehicle, newly_arrived_ids, self.max_allowed_ttt)
            reward += charging_reward(vehicle, charging_ids)
        return reward

    def calculate_action_penalty(self, vehicle, context):
        return congestion_penalty(vehicle, context, self.congestion_threshold_m, self.congestion_penalty_value)


class RewardShapingStrategy(RewardStrategy):
    def __init__(self, max_allowed_ttt):
        self.max_allowed_ttt = max_allowed_ttt

    def calculate_step_reward(self, vehicles, newly_arrived_ids, charging_ids, _newly_emptied_ids):
        reward = 0

        for vehicle in vehicles.values():
            reward += destination_reward(vehicle, newly_arrived_ids, self.max_allowed_ttt)
            reward += charging_reward(vehicle, charging_ids)

        return reward

    def calculate_final_reward(self, ttt_per_ev_mean):
        return BasicRewardStrategy().calculate_final_reward(ttt_per_ev_mean)

    def calculate_action_penalty(self, vehicle, context):
        penalty = NoTimeComponentRewardStrategy().calculate_action_penalty(vehicle, context)
        penalty += congestion_penalty(vehicle, context, congestion_threshold_m=36000, congestion_penalty=500)
        return penalty


class RelativeDestinationStrategy(RewardStrategy):
    """
    Per-vehicle destination reward based on charging overhead relative to the ideal (no-stop)
    travel time obtained from the routing API at spawn.

    Reward per arriving vehicle = -(actual_ttt - ideal_ttt) / scale_s:
      - Vehicle arrives with no charging detour: overhead ≈ 0, reward ≈ 0.
      - Vehicle charged once (~1300 s stop + detour): reward ≈ -0.5 at default scale.
      - Vehicle's battery went empty: no arrival, battery_penalty = -1.

    The incentive ordering is correct: charging when needed (reward ≈ -0.5) is always preferred
    over going empty (reward = -1), while unnecessary charging still incurs a cost.

    The ideal_ttt comes from traci.simulation.findRoute at the moment of spawn, which uses actual
    edge speeds and vehicle type — the same information a real routing API would return at trip
    start. This is more robust than dividing by a hardcoded speed constant, and naturally adapts
    to different networks, speed limits, and vehicle types.

    Step reward: -(overhead / scale_s) per arrived vehicle; battery_penalty per dead vehicle.
    Final reward: 0.
    Action penalties: 0.

    Args:
        scale_s: Divisor in seconds to normalize the overhead. Default 3000 s (~50 min charging
                 stop = reward -1 at default battery_penalty_value).
        battery_penalty_value: Magnitude of the penalty when a vehicle's battery runs empty.
                               Should be significantly larger than the typical charging overhead
                               reward to give a clear incentive. Default 3.0 (~7× a typical
                               1300 s charging stop overhead at scale_s=3000).
    """
    def __init__(self, scale_s=3000, battery_penalty_value=3.0):
        self.scale_s = scale_s
        self.battery_penalty_value = battery_penalty_value

    def calculate_step_reward(self, vehicles, newly_arrived_ids, _charging_ids, newly_emptied_ids):
        reward = 0
        for vehicle in vehicles.values():
            reward += relative_destination_reward(vehicle, newly_arrived_ids, self.scale_s)
            reward += battery_penalty(vehicle, newly_emptied_ids, self.battery_penalty_value)
        return reward

    def calculate_final_reward(self, _ttt_per_ev_mean):
        return 0

    def calculate_action_penalty(self, _vehicle, _context):
        return 0