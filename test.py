"""
Tests to find faulty commits with git bisect
"""

import os
import sys
import traceback

# Make sure imports work from the repo root that git bisect checks out
REPO_ROOT = os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def is_signature_typeerror(e: TypeError) -> bool:
    msg = str(e)
    # Common signature-mismatch messages across Python versions
    return (
        "got an unexpected keyword argument" in msg
        or "missing" in msg and "required positional argument" in msg
        or "positional argument" in msg
        or "takes" in msg and "positional" in msg
    )

def test_remaining_range_calculation() -> int:
    """Verify that get_remaining_range() produces physically plausible values.

    After vehicles have driven past DISTANCE_THRESHOLD (100 m) the implied
    consumption rate should be ~0.24 Wh/m (highway propulsion physics).
    A broken implementation (e.g. using VAR_ELECTRICITYCONSUMPTION) gives
    ~4 Wh/m due to the inflated constantPowerIntake in the emission model.

    Returns 0 (good), 1 (bad), or 125 (skip / untestable).
    """
    try:
        from environment import CustomEnv
    except Exception:
        print("Import of CustomEnv failed; skipping")
        traceback.print_exc()
        return 125

    env = None
    try:
        env = CustomEnv(
            scenario_generator="same_soc_same_route",
            vehicles_to_spawn=2,
            street_network="straight_120km",
            random_seed=42,
            truncate_after_n_steps=500,
        )
        env.reset()
        sim = env.simulation

        # Advance the simulation directly (bypassing env.step overhead) until
        # all observable EVs have driven past DISTANCE_THRESHOLD (100 m).
        # At max speed ~29 m/s this takes ~4 sim steps; 200 is a hard safety cap.
        # update_vehicle_soc() is called each step to keep soc_history populated,
        # which is required for the SOC-delta consumption estimate in get_remaining_range().
        DISTANCE_THRESHOLD = 100  # metres, same constant as in simulation.py
        steps_taken = 0
        for _ in range(200):
            sim.step()
            steps_taken += 1
            for vid in list(sim.vehicle_data):
                if vid.startswith("observable_ev"):
                    sim.update_vehicle_soc(vid)
            active_evs = [v for v in sim.vehicle_data if v.startswith("observable_ev")]
            if not active_evs:
                continue
            if all(float(sim.vehicle_data[v].get(0x84, 0)) >= DISTANCE_THRESHOLD
                   for v in active_evs):
                break
        print(f"  Ran {steps_taken} sim step(s); "
              f"{len(active_evs)} observable EV(s) active")

        bad_rates = []
        checked = 0
        for vid in list(sim.vehicle_data):
            if not vid.startswith("observable_ev"):
                continue
            remaining_range = sim.get_remaining_range(vid)
            soc = sim.get_battery_soc(vid)
            if remaining_range is None or soc is None or remaining_range <= 0:
                continue

            current_distance = float(sim.vehicle_data[vid].get(0x84, 0))
            baseline = sim.driving_segment_baseline.get(vid)
            if baseline is not None:
                baseline_soc, baseline_dist = baseline
                segment_dist = current_distance - baseline_dist
                segment_consumed = baseline_soc - soc
                print(f"  {vid}: segment={segment_dist:.0f} m, "
                      f"consumed={segment_consumed:.1f} Wh, "
                      f"SOC={soc:.0f} Wh, "
                      f"range={remaining_range/1000:.1f} km, "
                      f"rate={soc/remaining_range:.3f} Wh/m")
            else:
                print(f"  {vid}: no baseline — using fallback rate; "
                      f"SOC={soc:.0f} Wh, range={remaining_range/1000:.1f} km")

            implied_rate = soc / remaining_range  # Wh/m
            checked += 1

            # Good commits: ~0.24 Wh/m at steady state; up to ~0.60 Wh/m when vehicles
            # are still accelerating from rest (kinetic energy gain inflates the rate).
            # Bad commits: ~3.78-5.33 Wh/m (inflated by constantPowerIntake in emission model).
            if not (0.10 <= implied_rate <= 0.80):
                bad_rates.append(f"{vid}: {implied_rate:.3f} Wh/m")

        if checked == 0:
            print("No active vehicles to check — skipping")
            return 125

        if bad_rates:
            print(f"FAIL — implausible consumption rate(s): {bad_rates}")
            return 1

        print(f"PASS — remaining range plausible for {checked} vehicle(s)")
        return 0

    except Exception:
        print("Error during remaining-range test:")
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> int:
    print("--- test: remaining range calculation ---")
    result = test_remaining_range_calculation()
    if result != 0:
        return result

    print("--- test: training ---")
    try:
        from training_utils import train_model
    except Exception:
        print("Import failed; skipping commit (module moved / deps missing?)")
        traceback.print_exc()
        return 125  # skip untestable commits

    try:
        try:  # test for the new signature of train_model
            train_model(
                scenario="all_random",
                algorithm="PPO",
                policy="MultiInputPolicy",
                version_tag="testv0.9",
                reward_strategy="basic",
                street_network="straight_100km",
                n_vehicles=2,
                n_training_units=1,
                execution_context="local",
                random_seed=None,
                use_wandb=False
            )
        except TypeError as e:
            # if not is_signature_typeerror(e):
            #     # A real TypeError inside training -> treat as failure
            #     raise

            try:
                # test for the old signature
                train_model(
                    algorithm="PPO",
                    policy="MultiInputPolicy",
                    version="testv0.7.9",
                    env_version="basic",
                    street_network="straight_100km",
                    n_vehicles=2,
                    n_steps=1,
                    execution_context="local",
                    random_seed=None,
                )
            except TypeError as e2:
                if "CustomEnv.__init__() missing" in str(e2):
                    return 125  # skip untestable commits
                raise

        print("Training completed successfully")
        return 0  # good commit

    except Exception:
        print("Error during training:")
        traceback.print_exc()
        return 1  # bad commit

if __name__ == "__main__":
    return_code = main()
    print(f"Exiting with code {return_code}")
    sys.exit(return_code)
