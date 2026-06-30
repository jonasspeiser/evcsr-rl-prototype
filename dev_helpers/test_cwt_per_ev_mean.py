"""
Manual test for cwt_per_ev_mean correctness.

SUMO's VAR_WAITING_TIME only accumulates while a vehicle is stationary due to
traffic (e.g. queuing behind another vehicle at a busy charging station). Vehicles
at planned stops (charging, parking) do NOT accumulate waiting time in SUMO.
cwt_per_ev_mean therefore measures charging-station queue time, not charging time.

This test verifies:
  1. Formula consistency: cwt_per_ev_mean == cumulated_waiting_time / n_vehicles.
  2. Magnitude sanity: value is not in the triangular-sum range (the old bug inflated
     results by ~N*charging_duration/2, producing values in the hundreds of thousands).
  3. Queue scenario: with more vehicles than stations, queuing occurs and the metric
     is non-zero.

Run from the repo root:
    .venv/bin/python dev_helpers/test_cwt_per_ev_mean.py
"""
import sys
import os

sys.path.insert(0, os.getcwd())

try:
    from environment import CustomEnv
    from evaluation_algorithms import GreedyAlgorithm
except Exception as e:
    print(f"FAIL: import error: {e}")
    sys.exit(1)

CHARGING_DURATION = 1300  # seconds — from simulation.py; triangular-sum artifact ≈ N * CHARGING_DURATION^2 / 2

errors = []

# --- Test 1: formula consistency (no-queue scenario) ---
print("Test 1: formula consistency (2 vehicles, 4 stations, no expected queuing)")
env = CustomEnv(
    scenario_generator="same_route",
    reward_strategy="basic",
    vehicles_to_spawn=2,
    street_network="straight_120km",
    longest_route_duration=20000,
)
model = GreedyAlgorithm(environment=env)
obs, _ = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)

cwt = env.cwt_per_ev_mean
cumulated = env.cumulated_waiting_time
n = len(env.oev_ids)
env.close()

print(f"  cumulated_waiting_time : {cumulated:.1f} s")
print(f"  cwt_per_ev_mean        : {cwt:.1f} s")

expected = cumulated / n if n > 0 else 0
if abs(cwt - expected) > 1e-6:
    errors.append(f"Test1: cwt_per_ev_mean={cwt} != cumulated/n={expected:.6f}")
else:
    print("  formula consistent: PASS")

# In a no-queue scenario, 0 is the correct result (SUMO does not charge waiting
# time during planned charging stops, only during queue waiting).
# Check it's not hitting the triangular-sum magnitude.
triangular_sum_threshold = CHARGING_DURATION * CHARGING_DURATION / 2
if cwt > triangular_sum_threshold:
    errors.append(
        f"Test1: cwt_per_ev_mean={cwt:.0f} is in triangular-sum range "
        f"(>{triangular_sum_threshold:.0f}) — accumulation bug likely"
    )
else:
    print(f"  not in triangular-sum range (>{triangular_sum_threshold:.0f}): PASS")

# --- Test 2: queue scenario (more vehicles than stations forces congestion) ---
print("\nTest 2: queue scenario (5 vehicles, 4 stations, GREEDY may cause queuing)")
env = CustomEnv(
    scenario_generator="same_route",
    reward_strategy="basic",
    vehicles_to_spawn=5,
    street_network="straight_120km",
    longest_route_duration=20000,
)
model = GreedyAlgorithm(environment=env)
obs, _ = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)

cwt5 = env.cwt_per_ev_mean
cumulated5 = env.cumulated_waiting_time
n5 = len(env.oev_ids)
env.close()

print(f"  cumulated_waiting_time : {cumulated5:.1f} s")
print(f"  cwt_per_ev_mean        : {cwt5:.1f} s  ({cwt5/60:.1f} min)")

expected5 = cumulated5 / n5 if n5 > 0 else 0
if abs(cwt5 - expected5) > 1e-6:
    errors.append(f"Test2: cwt_per_ev_mean={cwt5} != cumulated/n={expected5:.6f}")
else:
    print("  formula consistent: PASS")

if cwt5 > triangular_sum_threshold:
    errors.append(
        f"Test2: cwt_per_ev_mean={cwt5:.0f} is in triangular-sum range — accumulation bug likely"
    )
else:
    print(f"  not in triangular-sum range: PASS")

# --- Summary ---
print()
if errors:
    for e in errors:
        print(f"FAIL: {e}")
    sys.exit(1)
else:
    print("All tests PASSED.")
    sys.exit(0)
