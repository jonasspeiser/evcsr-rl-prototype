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

from training_utils import get_git_version, train_model, evaluate_model, evaluate_model_with_config, further_train_model, get_latest_n_models, training_units_to_steps

_N_VEHICLES = 20
_MAX_TRAINING_HOURS = 8.5 # Training duration by wall clock time. Set to None to disable time-based stopping.
_CHECKPOINT_FREQ = 20_000 # how often the model should be saved during training (in training steps)
_N_TRAINING_UNITS = 10_000_000 # Training duration by training steps. if you use MAX_TRAINING_HOURS, set this value close to infinite (e.g. 10_000_000)
_N_CHECKPOINTS = 2 # how often the model should be saved during training (does NOT work in combination with MAX_TRAINING_HOURS)

BASE_TRAINING = partial(train_model,
    algorithm="PPO",
    reward_strategy="basic",
    policy="MultiInputPolicy",
    version_tag=get_git_version(),
    scenario="all_random",
    # start_soc_bounds=(22_000, 22_000),
    street_network="straight_120km",
    obs_features={"simulation_time"},
    reward_kwargs={"congestion_threshold_m": 36000, "congestion_penalty": 0.1},
    use_custom_extractor=False,
    # longest_route_duration=28_000,
    n_vehicles=_N_VEHICLES,
    n_noevs=0,
    max_training_hours=_MAX_TRAINING_HOURS,
    n_training_units=_N_TRAINING_UNITS,
    ent_coef=0.1,
    use_wandb=False,
    # checkpoint_freq=training_units_to_steps(_N_TRAINING_UNITS // _N_CHECKPOINTS, _N_VEHICLES),
    checkpoint_freq=_CHECKPOINT_FREQ,
)

BASE_EVAL = partial(evaluate_model,
        scenario="all_random",
        # start_soc_bounds=(22_000, 22_000),
        version_tag=get_git_version(),
        reward_strategy="basic",
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
    partial(BASE_TRAINING, reward_strategy="basic"),
    partial(BASE_TRAINING, reward_strategy="basicDestination"),
    partial(BASE_TRAINING, reward_strategy="basicCharging"),
    partial(BASE_TRAINING, reward_strategy="basicShaping", obs_features={"simulation_time", "station_assignment_counts"}),
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

# --- Define your EVALUATION RUNS here ---
# (given parameters override the ones in the base evaluation configuration)

EVALS = [
    partial(BASE_EVAL, algorithm="GREEDY"),
    partial(BASE_EVAL, algorithm="BEST_GUESS"),
    partial(BASE_EVAL, algorithm="RANDOM"),
]



if __name__ == "__main__":
    import time
    start_time = time.perf_counter()

    def run_parallel(run_list):
        processes = [Process(target=run) for run in run_list]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def suspend_system():
        import subprocess
        subprocess.run(["systemctl", "suspend"])

    run_parallel(TRAININGS)
    
    run_parallel(EVALS)

    # run_parallel(FURTHER_TRAININGS)

    # Build model evals lazily after further training completes (new model paths now exist)
    MODEL_EVALS = [
        partial(evaluate_model_with_config, model_load_path=path, n_episodes=10, random_seed=54321)
        for path in get_latest_n_models(4)
    ]

    run_parallel(MODEL_EVALS)
    
    print("All runs done")

    elapsed_time = time.perf_counter() - start_time
    print(f"Execution took {elapsed_time / 60:.2f} minutes")

    suspend_system()
