"""
Define and launch parallel training runs.

Edit the TRAININGS and EVALS lists below to configure which experiments to run.

Run all experiments in parallel:
    python train.py

Detached execution:
    nohup python train.py > train_output.log 2>&1 & 
    tail -f train_output.log

Each process gets its own PID -> own SUMO instance -> own state file. No port management needed.
"""

from functools import partial

from training_utils import get_git_version, train_model, evaluate_model, evaluate_model_with_config, further_train_model, get_latest_n_models, get_models_from_folder, training_units_to_steps, run_parallel

_N_VEHICLES = 600

BASE_EVAL = partial(evaluate_model,
        scenario="bast",
        # start_soc_bounds=(22_000, 22_000),
        version_tag=get_git_version(),
        reward_strategy="relativeDestination",
        street_network="straight_120km",
        n_vehicles=_N_VEHICLES,
        n_noevs=0,
        n_episodes=10,
        # longest_route_duration=28_000,
        model_load_path=None,
        # render_mode="human",
        random_seed=54321,
)

# --- Experiment 4: NOEV partial-observability evaluation sweep ---
# Evaluation-first design: Exp 3 trained agents evaluated across NOEV session counts.
# NOEV counts chosen from realized-participation probes (GREEDY, 3 eps each):
#   40 -> ~0.75 | 160 -> ~0.43 | 320 -> ~0.20  (0 = full-observability)

# Trimmed to the scientifically load-bearing set: the new congestion variant, the dense variant WITHOUT a congestion penalty
_EXP4_MODELS = {
    "relativeDestinationCongestion": "runs/2026-06-13_14-46-59_pid942615_v1.4.1-28-gf1bf83d_relativeDestinationCongestion_bast_straight_120km_PPO/2026-06-13_14-46-59_pid942615_v1.4.1-28-gf1bf83d_relativeDestinationCongestion_bast_straight_120km_PPO.zip",
    "relativeDestination": "runs/_runs_20260612_1004 (bast 600)/2026-06-12_00-10-04_pid2420667_v1.4.1-18-g88057f4_relativeDestination_bast_straight_120km_PPO/2026-06-12_00-10-04_pid2420667_v1.4.1-18-g88057f4_relativeDestination_bast_straight_120km_PPO.zip",
}

_EXP4_NOEV_COUNTS = [0, 40, 160, 320]
_EXP4_N_EPISODES = 50

EVALS_EXP4 = [
    partial(BASE_EVAL, algorithm="PPO", model_load_path=path, reward_strategy=strategy,
            n_noevs=n_noevs, noev_provider="obelis", n_episodes=_EXP4_N_EPISODES)
    for strategy, path in _EXP4_MODELS.items()
    for n_noevs in _EXP4_NOEV_COUNTS
] + [
    partial(BASE_EVAL, algorithm=baseline,
            n_noevs=n_noevs, noev_provider="obelis", n_episodes=_EXP4_N_EPISODES)
    for baseline in ("GREEDY", "BEST_GUESS", "RANDOM")
    for n_noevs in _EXP4_NOEV_COUNTS
]


if __name__ == "__main__":
    import time
    start_time = time.perf_counter()

    # Experiment 4: NOEV partial-observability sweep (20 conditions: {relativeDestination, relativeDestinationCongestion} x {0,40,160,320} NOEV + GREEDY/BEST_GUESS/RANDOM x 4).
    run_parallel(EVALS_EXP4, max_workers=3, stagger_s=20)
    
    print("All runs done")

    elapsed_time = time.perf_counter() - start_time
    print(f"Execution took {elapsed_time / 60:.2f} minutes")