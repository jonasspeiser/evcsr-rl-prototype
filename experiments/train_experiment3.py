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

# Exp 3b: dense per-arrival reward + congestion penalty (scaled up).
TRAININGS_EXP3B = [
    partial(BASE_TRAINING,
            reward_strategy="relativeDestinationCongestion",
            reward_kwargs={"congestion_threshold_m": 36000, "congestion_penalty": 60, "battery_penalty_value": 10},
            obs_features={"simulation_time", "station_assignment_counts"}),
]

# Exp 3c: same as 3b (dense reward + calibrated congestion) PLUS a sized illegal-action penalty.
TRAININGS_EXP3C = [
    partial(BASE_TRAINING,
            reward_strategy="relativeDestinationCongestionIllegal",
            reward_kwargs={"congestion_threshold_m": 36000, "congestion_penalty": 60, "battery_penalty_value": 10, "illegal_penalty_value": 0.2},
            obs_features={"simulation_time", "station_assignment_counts"}),
]

if __name__ == "__main__":
    import time
    start_time = time.perf_counter()


    # Exp 3b: single training run, calibrated congestion penalty (see TRAININGS_EXP3B).
    run_parallel(TRAININGS_EXP3B)

    # Exp 3c: dense + calibrated congestion + sized illegal penalty (see TRAININGS_EXP3C).
    run_parallel(TRAININGS_EXP3C)