"""
Define and launch parallel training runs.

Edit the RUNS list below to configure which experiments to run.
Each partial() call is a direct call to train_model() with full IDE support
(autocomplete, type hints, docstring on hover).

Run all experiments in parallel:
    python train.py

Each process gets its own PID → own SUMO instance → own state file. No port management needed.
"""

from functools import partial
from multiprocessing import Process

from training_utils import get_git_version, train_model

BASE = partial(train_model,
    algorithm="PPO",
    policy="MultiInputPolicy",
    version_tag=get_git_version(),
    scenario="same_route",
    start_soc_bounds=(22_000, 22_000),
    street_network="straight_120km",
    congestion_kwargs={"congestion_threshold_m": 33600, "congestion_penalty": 500.0},
    n_vehicles=5,
    n_noevs=0,
    n_training_units=300,
    ent_coef=0.0,
    use_wandb=False,
)

# --- Define your experiments here ---
RUNS = [
    partial(BASE,
        reward_strategy="basic",
        obs_features={"simulation_time"},         
    ),
    partial(BASE,
        reward_strategy="basic",
        obs_features={"simulation_time", "station_assignment_counts"},         
    ),
    partial(BASE,
        reward_strategy="basicCongestion",
        obs_features={"simulation_time"},  
    ),
    partial(BASE,
        reward_strategy="basicCongestion",
        obs_features={"simulation_time", "station_assignment_counts"},  
    ),
]


if __name__ == "__main__":
    processes = [Process(target=run) for run in RUNS]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("All runs done")
