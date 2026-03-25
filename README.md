
# RL environment for EV Charging Station Recommendation (EVCSR)

A reinforcement learning environment for training agents to recommend charging stations to electric vehicles (EVs) in a SUMO traffic simulation. The agent decides whether and where to route an EV for charging at each decision point (vehicle spawn, low battery, post-charge). The objective is to minimize total travel time across all vehicles.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Key Concepts](#key-concepts)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward Strategies](#reward-strategies)
  - [Scenarios](#scenarios)
  - [Street Networks](#street-networks)
- [Training & Evaluation](#training--evaluation)
- [Baseline Algorithms](#baseline-algorithms)
- [Run Outputs](#run-outputs)
- [Logging & Metrics](#logging--metrics)
- [Testing & Debugging](#testing--debugging)
- [How To](#how-to)

---

## Overview

Each episode, one or more observable EVs (OEVs) drive through a SUMO road network. Whenever a vehicle needs a charging decision (at spawn, low battery, or after finishing a charge), the RL agent receives an observation and picks an action:

- **Action 0**: Do nothing, continue on current route
- **Actions 1–4**: Reroute the vehicle to charging station cs_1 through cs_4

The simulation uses [SUMO](https://sumo.dlr.de/) (Simulation of Urban MObility) as the underlying traffic simulator, interfaced via [TraCI](https://sumo.dlr.de/docs/TraCI.html). The RL layer is built on [Gymnasium](https://gymnasium.farama.org/) and trained with [stable-baselines3](https://stable-baselines3.readthedocs.io/).

---

## Installation

### 1. Install SUMO (External Dependency)

This project requires **SUMO ≥ 1.26**.

Official installation instructions: https://sumo.dlr.de/docs/Installing/index.html

Linux users may install via their package manager (see https://software.opensuse.org//download.html?project=science%3Adlr&package=sumo)

Example (Fedora 43):
```bash
dnf config-manager addrepo --from-repofile=https://download.opensuse.org/repositories/science:dlr/Fedora_43/science:dlr.repo
dnf install sumo
```

Verify installation:
```bash
sumo --version
```

> If that doesn't work: depending on your system, `SUMO_HOME` may need to be set. Flatpak installation is possible but discouraged due to extra configuration steps (PATH, `SUMO_HOME`, manual TCP port exposure for TraCI).

### 2. Python Setup

This project uses **uv** for a fully reproducible Python environment (Python 3.11 is managed automatically).

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into a virtual environment
uv sync

# Optional: include dev tools (Jupyter, W&B, seaborn, snakeviz)
uv sync --extra dev

# Activate the venv manually if needed
source .venv/bin/activate
```

All dependencies are locked via `uv.lock` for reproducibility.

---

## Project Structure

```
evcsr-rl-prototype/
├── environment.py          # Gymnasium env (CustomEnv) — main RL loop
├── simulation.py           # SUMO/TraCI wrapper — vehicle & battery management
├── vehicle.py              # Per-vehicle state, observation generation
├── reward_strategies.py    # Pluggable reward functions
├── oev_scenario_generator.py  # Vehicle spawning scenarios
├── evaluation_algorithms.py   # Baseline algorithms (Greedy, Random, Fixed)
├── feature_extractor.py    # Deep Sets feature extractor for the policy
├── training_utils.py       # High-level train/evaluate API
├── network_generator.py    # SUMO network generation & route extraction
├── logging_utils.py        # Custom logging, run management, crash bundles
├── plotting_utils.py       # Violin plots and evaluation visualization
├── noev_data_provider.py   # Background traffic (NOEV) data providers
├── test.py                 # git bisect tests (returns 0/1/125)
├── training.ipynb          # Interactive training notebook
├── visualization.ipynb     # Results visualization notebook
├── street-networks/        # SUMO network files (see Street Networks)
│   ├── circle/
│   ├── straight_100km/
│   └── straight_120km/
├── vehicle-models/
│   └── soulEV65.add.xml    # Kia Soul EV 65 kWh vehicle model
├── datasets/               # NOEV/traffic datasets (OBELIS, BASt)
└── runs/                   # Training & evaluation outputs (gitignored)
```

### Architecture diagram

![Overview](./docs/figures/architecture.excalidraw.png)

---

## Quick Start

The primary entry point for experiments is `training_utils.py`. Use `training.ipynb` for interactive runs.

```python
from training_utils import train_and_evaluate, get_git_version

# Train a PPO model and evaluate it against baselines in one call
model_path, eval_paths = train_and_evaluate(
    scenario="same_route",
    algorithm="PPO",
    policy="MultiInputPolicy",
    version_tag=get_git_version(),
    reward_strategy="basic",
    street_network="straight_120km",
    n_vehicles=1,
    n_noevs=0,
    n_training_units=2000,    # ≈ episodes; converted to env steps internally
    ent_coef=0.01,
    eval_episodes=20,
    random_seed_eval=54321,
    execution_context="local",
)
```

This trains a model, then runs evaluation against ACTION0 (do-nothing), RANDOM, and GREEDY baselines, saving all results under `runs/`.

---

## Key Concepts

### Observation Space

At each decision point one vehicle is the "active" vehicle requesting a charging recommendation. The observation is a dict:

```python
{
    "active_vehicle":  Box(12,),               # vehicle requesting a decision
    "other_vehicles":  Box(max_vehicles, 12),  # all other spawned OEVs (zero-padded)
    "vehicle_mask":    Box(max_vehicles,),     # 1.0 = valid, 0.0 = padding
}
```

Each vehicle is represented as a 12-dimensional vector:

| Index | Feature | Notes |
|-------|---------|-------|
| 0 | `battery_soc` | Normalized to [0, 1]; −1 if unavailable |
| 1 | `distance_to_destination` | Normalized to [0, 1]; −1 if unavailable |
| 2–5 | `distance_to_cs_1..4` | Normalized to [0, 1]; −1 if unreachable or already past station |
| 6–10 | `last_action_one_hot` | 5-element encoding of previous action (0=do_nothing, 1–4=cs) |
| 11 | `arrived_at_destination` | Binary {0, 1} |

The policy uses a **Deep Sets** feature extractor ([`feature_extractor.py`](feature_extractor.py)): a shared encoder φ is applied to each vehicle independently, then mean-pooled for context. This makes the policy permutation-invariant with respect to vehicle order and generalizes to different numbers of vehicles.

### Action Space

`Discrete(5)`:
- `0` — Do nothing (vehicle continues on current route)
- `1` — Route vehicle to charging station cs_1
- `2` — Route vehicle to charging station cs_2
- `3` — Route vehicle to charging station cs_3
- `4` — Route vehicle to charging station cs_4

Charging stations are numbered `cs_1` through `cs_4` (1-indexed). Routing to a station the vehicle has already passed raises a `PointlessRecommendationError` and is treated as do-nothing.

### Reward Strategies

Configured via the `reward_strategy` parameter. Three strategies are available:

| Strategy | Step Reward | Terminal Reward | Notes |
|----------|-------------|-----------------|-------|
| `"basic"` | 0 | −mean_travel_time | Pure TTT minimization |
| `"noTime"` | −100 on death, +10 on arrival | 0 | No time component |
| `"shaping"` | −100 on death, +(MAX_TTT − travel_time) on arrival | −mean_travel_time | Hybrid |

All strategies apply a penalty for unnecessary or illegal charging actions (routing to an already-passed station, or routing when the vehicle has sufficient range to reach the destination).

**Key constants** (defined in `reward_strategies.py`):
- Death penalty: `−100` per vehicle
- Arrival bonus (shaping): `MAX_ALLOWED_TTT − travel_time` (MAX_ALLOWED_TTT = 100 s)
- Action penalty: `−1` for pointless/illegal actions

### Scenarios

Configured via the `scenario` parameter of `CustomEnv` / `train_model`:

| Scenario | Description |
|----------|-------------|
| `"all_random"` | Random routes, random start SOCs (200–500 Wh) |
| `"same_route"` | All vehicles use the network's longest route; random SOCs |
| `"same_soc_same_route"` | Same route, fixed SOC at 50% of battery upper bound (~11,195 Wh) |
| `"custom_distribution"` | Custom departure-time distribution (subclass to implement) |
| `"bast"` | Real German BASt traffic data (2022 dates); hourly traffic patterns |

For training and benchmarking, `"same_route"` and `"same_soc_same_route"` are the most commonly used as they reduce variance.

### Street Networks

Located in `street-networks/`. Each network directory contains SUMO `.net.xml`, `.add.xml`, `.rou.xml`, and pre-computed `.all_routes.json` / `.all_distances.json` files.

| Network | Length | Charging Stations | Notes |
|---------|--------|-------------------|-------|
| `straight_100km` | 100 km | 4 (at 25, 50, 75, 100 km) | Standard training network |
| `straight_120km` | 120 km | 4 (at 25, 50, 75, 100 km) | Longer variant |
| `circle` | ~8 km | — | Small test network |

Charging stations provide 200 kW at 0.95 efficiency; charging to 80% takes ~969 seconds.

**Vehicle model**: Kia Soul EV 65 kWh (`vehicle-models/soulEV65.add.xml`). Battery capacity in simulation: 64 kWh. Empirical consumption: ~240 Wh/km (0.24 Wh/m) at highway speeds.

To change or add a network, see [How To: Add a new street network](#add-a-new-street-network).

---

## Training & Evaluation

### Core API (`training_utils.py`)

```python
# Train a new model
model_path = train_model(
    scenario="same_route",          # vehicle spawning scenario
    algorithm="PPO",                # "PPO", "A2C", or "DQN"
    policy="MultiInputPolicy",
    version_tag=get_git_version(),
    reward_strategy="basic",        # "basic", "noTime", or "shaping"
    street_network="straight_120km",
    n_vehicles=1,                   # number of observable EVs
    n_training_units=2000,          # ≈ episodes (converted: units × n_vehicles × 3)
    ent_coef=0.01,                  # entropy coeff for PPO/A2C
    execution_context="local",
)

# Continue training from a checkpoint (saves into the same run directory)
further_path = further_train_model(
    model_load_path=model_path,
    n_training_units=1000,
)

# Evaluate a model or baseline
evaluate_model(
    scenario="same_route",
    algorithm="PPO",                # or "GREEDY", "RANDOM", "ACTION0"
    version_tag=get_git_version(),
    reward_strategy="basic",
    street_network="straight_120km",
    n_vehicles=1,
    n_episodes=20,
    model_load_path=further_path,   # None for baselines
    random_seed=54321,
)

# Evaluate a trained model using its saved config
evaluate_model_with_config(model_load_path=further_path, n_episodes=20)
```

> **Important**: always use the explicit return value of `further_train_model` as `model_load_path` for evaluation. `get_latest_model()` returns the alphabetically-last `.zip` in the most recently named run directory — this is fragile if other runs exist or are running concurrently.

### n_training_units → env steps

`training_units_to_steps(n, n_oevs) = n × n_oevs × 3`

One training unit ≈ one episode. Each OEV generates ~3 charging requests per episode (spawn + low battery or post-charge), so steps per episode ≈ `n_oevs × 3`. Note that `env.step()` advances the simulation until the *next charging request*, not a fixed number of simulation steps.

### Experiment tracking (W&B)

W&B is optional. When disabled, all results are written to local files.

```python
train_model(..., use_wandb=True, wandb_entity="evcs-rl")
```

### SUMO GUI visualization

To visually inspect a run, pass `render_mode="human"` to `evaluate_model`. After the GUI opens: `Edit → Edit Visualisation (F9) → Vehicles → Draw with constant size when zoomed out`.

---

## Baseline Algorithms

Defined in `evaluation_algorithms.py`:

| Algorithm | `algorithm` param | Description |
|-----------|------------------|-------------|
| Do-nothing | `"ACTION0"` | Always action 0 (never charges) |
| Random | `"RANDOM"` | Uniformly random action each step |
| Greedy | `"GREEDY"` | If normalized SOC > 0.15 (~40 km range): do nothing. Otherwise route to nearest reachable station ahead of destination. |
| Fixed action N | `"ACTION1"` … `"ACTION4"` | Always chooses station N |

The greedy threshold of 0.15 normalized SOC corresponds to ~9,600 Wh ≈ 40 km of range (well above the 25 km maximum station gap in the straight networks).

---

## Run Outputs

All runs are saved under `runs/` (gitignored):

```
runs/
└── <timestamp>_<version>_<reward>_<scenario>_<network>_<algo>/
    ├── <timestamp>_<version>_...zip      # trained model weights
    ├── run_config.json                   # full training config
    ├── training/
    │   ├── tensorboard/                  # TensorBoard logs
    │   └── <timestamp>_..._training.jsonl  # step-by-step episode log
    └── evaluation/
        ├── run_config_<timestamp>.json   # eval config
        ├── metrics<timestamp>.json       # per-episode metrics
        └── <timestamp>_..._evaluation.jsonl  # step-by-step eval log
```

When `further_train_model` is called, the new `.zip` is saved in the **same** run directory as the base model, with a new timestamp in the filename.

---

## Logging & Metrics

### TensorBoard

```bash
tensorboard --logdir runs/
```

Key metrics:

| Metric | Description |
|--------|-------------|
| `env/global_ttt` | Cumulated total travel time (seconds) of all OEVs |
| `env/global_ttt_only_terminated` | Same, episodes that terminated (not truncated) |
| `env/ttt_per_ev_mean` | Mean TTT per vehicle |
| `env/ttt_per_ev_mean_only_terminated` | Mean TTT per vehicle (terminated episodes only) |
| `env/empty_vehicles_per_episode` | Vehicles that ran out of battery |
| `env/charging_stops_per_episode_mean` | Average charging stops per episode |
| `env/arrival_soc_wh_mean` | Mean battery SOC (Wh) at destination arrival |
| `env/arrival_range_m_mean` | Mean remaining range (m) at destination arrival |
| `env/charging_start_soc_wh_mean` | Mean SOC when charging begins |
| `env/charging_start_range_m_mean` | Mean remaining range when charging begins |
| `env/final_simulation_time` | SUMO time (s) when episode ended |

### Evaluation JSON

`metrics<timestamp>.json` contains a list of per-episode dicts with all metrics above plus `episode`, `episode_length`, `reward`, `action_counts`, `was_truncated`.

### Visualization

`visualization.ipynb` loads metrics files and generates comparison plots (violin plots by algorithm). `plotting_utils.py` provides `make_violinplot()` and helpers for loading the last N runs.

---

## Testing & Debugging

### git bisect tests (`test.py`)

`test.py` is designed for `git bisect`. It returns:
- `0` — good commit
- `1` — bad commit
- `125` — skip (untestable commit)

```bash
git bisect start
git bisect bad HEAD
git bisect good <known-good-sha>
git bisect run uv run python test.py
```

**test_remaining_range_calculation**: Verifies that `simulation.get_remaining_range()` produces a physically plausible implied consumption rate of 0.10–0.80 Wh/m. Bad commits using `VAR_ELECTRICITYCONSUMPTION` produce ~3.78–5.33 Wh/m (inflated by 15×).

### Range calculation notes

`get_remaining_range()` uses a **SOC-delta approach**: consumption rate is estimated as `(baseline_soc − current_soc) / segment_distance`, where the baseline resets after each charging stop. Falls back to 0.24 Wh/m when less than 100 m of segment data is available.

> **Why not `VAR_ELECTRICITYCONSUMPTION`?** SUMO's energy emission model applies `constantPowerIntake = 100` as ~100 Wh/timestep (not 100 W × 1 s = 0.028 Wh/step), inflating measured consumption by ~16×. The battery device model (`device.battery.actualBatteryCapacity`) applies physics correctly and is used for SOC tracking.

---

## How To

### Add a new street network

1. Generate SUMO network files (`.net.xml`, `.nod.xml`, `.edg.xml`, `.add.xml`, `.sumocfg`) and place them in `street-networks/<name>/`
2. Run `network_generator.py` to pre-compute routes and distances:
   ```python
   from network_generator import write_all_routes_json
   write_all_routes_json("straight_120km")
   ```
3. Pass `street_network="<name>"` to `train_model` / `evaluate_model`

### Add a new vehicle model

1. Create an `.add.xml` file in `vehicle-models/` following the SUMO additional file format with a `vType` element and `device.battery` parameters
2. Reference it in the network's `.sumocfg` additional files section

### Add a new evaluation baseline algorithm

1. Subclass `EvaluationAlgorithm` in `evaluation_algorithms.py` and implement `predict(observation, deterministic) → (action, _)`
2. Register it in `EVAL_ALGOS` in `training_utils.py`
3. Pass its key as `algorithm=` to `evaluate_model`

### Add a new reward strategy

1. Subclass `RewardStrategy` in `reward_strategies.py` and implement `calculate_step_reward(...)` and `calculate_terminal_reward(...)`
2. Register the strategy name in `CustomEnv.__init__` in `environment.py`
3. Pass the name as `reward_strategy=` to `train_model`

### Add a new scenario

1. Subclass `ScenarioGenerator` in `oev_scenario_generator.py` and implement `generate_vehicles(...)`
2. Register the scenario key in `CustomEnv.__init__` in `environment.py`
3. Pass the key as `scenario=` to `train_model`
