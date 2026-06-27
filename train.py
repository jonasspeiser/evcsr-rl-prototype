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
from multiprocessing import Process

from training_utils import get_git_version, train_model, evaluate_model, evaluate_model_with_config, further_train_model, get_latest_n_models, get_models_from_folder, training_units_to_steps

_N_VEHICLES = 600
_MAX_TRAINING_HOURS = 16 # Training duration by wall clock time. Set to None to disable time-based stopping.
_CHECKPOINT_FREQ = 20_000 # how often the model should be saved during training (in training steps)
_N_TRAINING_UNITS = 10_000_000 # Training duration by training steps. if you use MAX_TRAINING_HOURS, set this value close to infinite (e.g. 10_000_000)
_N_CHECKPOINTS = 2 # how often the model should be saved during training (does NOT work in combination with MAX_TRAINING_HOURS)

BASE_TRAINING = partial(train_model,
    algorithm="PPO",
    reward_strategy="relativeDestination",
    policy="MultiInputPolicy",
    version_tag=get_git_version(),
    scenario="bast",
    # start_soc_bounds=(22_000, 22_000),
    street_network="straight_120km",
    obs_features={"simulation_time"},
    reward_kwargs={"congestion_threshold_m": 36000, "congestion_penalty": 0.1, "battery_penalty_value": 10},
    use_custom_extractor=True,
    # longest_route_duration=28_000,
    n_vehicles=_N_VEHICLES,
    max_vehicles=600, # obs padded beyond spawn count to allow for all vehicles to be present at once
    n_noevs=0,
    max_training_hours=_MAX_TRAINING_HOURS,
    n_training_units=_N_TRAINING_UNITS,
    ent_coef=0.1,
    use_wandb=False,
    # checkpoint_freq=training_units_to_steps(_N_TRAINING_UNITS // _N_CHECKPOINTS, _N_VEHICLES),
    checkpoint_freq=_CHECKPOINT_FREQ,
)

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

# --- Define your TRAINING RUNS here ---
# (given parameters override the ones in the base training configuration)

TRAININGS = [
    # partial(BASE_TRAINING, use_custom_extractor=False),
    # partial(BASE_TRAINING, use_custom_extractor=True),
    partial(BASE_TRAINING, reward_strategy="basic"),
    partial(BASE_TRAINING, reward_strategy="basicCongestion", obs_features={"simulation_time", "station_assignment_counts"}),
    partial(BASE_TRAINING, reward_strategy="basicRelativeDestination"),
    partial(BASE_TRAINING, reward_strategy="relativeDestination"),
]

# Exp 3b: dense per-arrival reward + a congestion penalty that actually bites at scale.
# Diagnosis (Exp 3 evals at 600 veh): relativeDestination has NO congestion penalty and
# queues ~10x more than GREEDY (CWT 3245 vs 318); basicCongestion's penalty is normalized
# to ~0.00017/conflict (congestion_penalty / vehicles_to_spawn = 0.1/600), three orders of
# magnitude below a single charge event (~0.5), so it is effectively absent.
# Here congestion_penalty=60 -> 60/600 = 0.1 per conflicting vehicle post-normalization,
# comparable to the delayed queue-wait cost it proxies. battery_penalty=10 keeps the
# empty(-10) << congested-charge << normal-charge(-0.5) ordering, so a biting penalty
# cannot push the agent into stranding vehicles to dodge it.
TRAININGS_EXP3B = [
    partial(BASE_TRAINING,
            reward_strategy="relativeDestinationCongestion",
            reward_kwargs={"congestion_threshold_m": 36000, "congestion_penalty": 60, "battery_penalty_value": 10},
            obs_features={"simulation_time", "station_assignment_counts"}),
]

# Exp 3c: same as 3b (dense reward + calibrated congestion) PLUS a sized illegal-action
# penalty. Diagnosis of the 3b model: under congestion-only reward, recommending an
# already-passed station is unpenalized, so the agent does it ~5400x/episode (BadTimingRouting
# -> re-queue), wasting steps and stranding ~0.5 extra vehicles vs greedy. The legacy illegal
# penalty was 0.01 (negligible); here illegal_penalty_value=0.2 makes a passed-station pick
# clearly worse than do-nothing (0). Everything else identical to 3b to isolate the effect
# and serve as a clean reward-formulation ablation rung (RQ2.1).
TRAININGS_EXP3C = [
    partial(BASE_TRAINING,
            reward_strategy="relativeDestinationCongestionIllegal",
            reward_kwargs={"congestion_threshold_m": 36000, "congestion_penalty": 60, "battery_penalty_value": 10, "illegal_penalty_value": 0.2},
            obs_features={"simulation_time", "station_assignment_counts"}),
]

# --- Define your FURTHER TRAINING RUNS here ---
# get_latest_n_models(4) returns the 4 most recently created model paths

FURTHER_TRAININGS = [
    partial(further_train_model, model_load_path=path, 
            max_training_hours=_MAX_TRAINING_HOURS,
            n_training_units=_N_TRAINING_UNITS,
            # checkpoint_freq=training_units_to_steps(_N_TRAINING_UNITS // _N_CHECKPOINTS, _N_VEHICLES), 
            checkpoint_freq=_CHECKPOINT_FREQ,
            )
    for path in get_latest_n_models(4)
]

# FURTHER_TRAININGS_FOLDER = [
#     partial(further_train_model, model_load_path=path,
#             max_training_hours=_MAX_TRAINING_HOURS,
#             n_training_units=_N_TRAINING_UNITS,
#             checkpoint_freq=_CHECKPOINT_FREQ,
#             )
#     for path in get_models_from_folder("runs/_runs_20260610_1759")
# ]

# --- Define your EVALUATION RUNS here ---
# (given parameters override the ones in the base evaluation configuration)

EVALS = [
    partial(BASE_EVAL, algorithm="GREEDY"),
    partial(BASE_EVAL, algorithm="BEST_GUESS"),
    partial(BASE_EVAL, algorithm="RANDOM"),
]

# --- Experiment 4: NOEV partial-observability evaluation sweep ---
# Evaluation-first design: Exp 3 trained agents evaluated across NOEV session counts.
# NOEV counts chosen from realized-participation probes (GREEDY, 3 eps each):
#   40 -> rho ~0.75 | 160 -> rho ~0.43 | 320 -> rho ~0.20  (0 = full-observability anchor)
# 480 was probed and rejected: realized rho collapses to ~0.10 because station
# congestion suppresses OEV charging stops (rho numerator), overshooting the
# intended ~0.25 low-participation band.
# Paired comparison: identical random_seed across all algorithms at each NOEV level.

# Trimmed to the scientifically load-bearing set (Day 4): the new congestion variant
# (centerpiece), the dense variant WITHOUT a congestion penalty (the RQ2.1 contrast that
# isolates the penalty's effect under partial observability), and the three baselines.
# The basic/basicCongestion/basicRelativeDestination full-observability failures are kept
# out of the sweep; their 10-episode numbers already feed the Exp 3 ablation table.
_EXP4_MODELS = {
    "relativeDestinationCongestion": "runs/2026-06-13_14-46-59_pid942615_v1.4.1-28-gf1bf83d_relativeDestinationCongestion_bast_straight_120km_PPO/2026-06-13_14-46-59_pid942615_v1.4.1-28-gf1bf83d_relativeDestinationCongestion_bast_straight_120km_PPO.zip",
    "relativeDestination": "runs/2026-06-12_00-10-04_pid2420667_v1.4.1-18-g88057f4_relativeDestination_bast_straight_120km_PPO/2026-06-12_00-10-04_pid2420667_v1.4.1-18-g88057f4_relativeDestination_bast_straight_120km_PPO.zip",
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

    def run_parallel(run_list, max_workers=None, stagger_s=0):
        """Run all jobs with at most max_workers running concurrently.

        max_workers=None means unbounded (legacy behaviour). On this machine the Exp 4
        sweep MUST cap concurrency: 28 unbounded processes each running a 600-vehicle SUMO
        + loading the OBELIS feather saturated 12 cores / 30 GB and completed 0 conditions.
        stagger_s spaces out the start of each new worker to smooth the OBELIS load spike.
        """
        run_queue = list(run_list)
        cap = max_workers or len(run_queue)
        running = []
        while run_queue or running:
            while run_queue and len(running) < cap:
                p = Process(target=run_queue.pop(0))
                p.start()
                running.append(p)
                if stagger_s:
                    time.sleep(stagger_s)
            for p in running[:]:
                p.join(timeout=1)          # reap finished workers, free their slot
                if not p.is_alive():
                    running.remove(p)

    def suspend_system():
        import subprocess
        subprocess.run(["systemctl", "suspend"])

    # run_parallel(TRAININGS)

    # run_parallel(EVALS)

    # run_parallel(FURTHER_TRAININGS)

    # MODEL_EVALS = [
    #     partial(evaluate_model_with_config, model_load_path=path, n_episodes=10, n_vehicles=_N_VEHICLES, random_seed=54321)
    #     for path in get_latest_n_models(4)
    # ]
    # run_parallel(MODEL_EVALS)

    # Exp 3b: single training run, calibrated congestion penalty (see TRAININGS_EXP3B). DONE.
    # run_parallel(TRAININGS_EXP3B)

    # Exp 3c: dense + calibrated congestion + sized illegal penalty (see TRAININGS_EXP3C).
    run_parallel(TRAININGS_EXP3C)

    # Experiment 4: NOEV partial-observability sweep (28 conditions, ~50 eps each).
    # CAP at 3 workers: 28 unbounded 600-vehicle SUMO + OBELIS processes saturated the
    # machine and completed 0/28 last time. ~10 h wall at cap=3; relaunch once Exp 3b lands.
    # run_parallel(EVALS_EXP4, max_workers=3, stagger_s=20)

    # run_parallel(FURTHER_TRAININGS_FOLDER)

    # # get_models_from_folder returns the newest .zip per run dir, which is now the further-trained model
    # FOLDER_MODEL_EVALS = [
    #     partial(evaluate_model_with_config, model_load_path=path, n_episodes=10, random_seed=54321)
    #     for path in get_models_from_folder("runs/_runs_20260610_1759")
    # ]

    # run_parallel(FOLDER_MODEL_EVALS)
    
    print("All runs done")

    elapsed_time = time.perf_counter() - start_time
    print(f"Execution took {elapsed_time / 60:.2f} minutes")

    # suspend_system()
