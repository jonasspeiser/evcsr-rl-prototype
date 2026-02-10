
# RL environment for SUMO

---

## 🧾 Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Modeling](#modeling)
- [Results](#results)

---

## Installation
### SUMO setup
 - Ubuntu: install from apt
 -  Fedora 43:
    ```bash
    dnf config-manager addrepo --from-repofile=https://download.opensuse.org/repositories/science:dlr/Fedora_43/science:dlr.repo
    dnf install sumo
    ```
 - Other Linux distros: https://software.opensuse.org//download.html?project=science%3Adlr&package=sumo
 - else refer to https://sumo.dlr.de/docs/Installing/index.html
    - or https://sumo.dlr.de/docs/Downloads.php
 
Hint: SUMO is also available as a flatpak (flathub) for many linux distros. The use of flatpak is possible but disencouraged as it involves several manual steps for a working setup (configuring SUMO_HOME and PATH in bashrc, manually starting SUMO on a fixed exposed TCP port, configuring TraCi to connect to said port).


<!-- 
SUMO is also available as a flatpak (flathub) for many linux distros. When installing as a flatpack, make sure the Path-Variable for SUMO_HOME is set. Otherwise you will get a `FileNotFound Exception` when trying to start the simulation.
  1. Find out your flatpak installation path  
      * typically `~/.local/share/flatpak/app/org.eclipse.sumo/...` (user install) or `/var/lib/flatpak/app/org.eclipse.sumo/...` (system-wide)
  1. Find exact path: `flatpak info org.eclipse.sumo`
      * typically `.../org.eclipse.sumo/x86_64/stable/active/files` (adjust based on your install; verify it contains `bin/`)
  1. Add to `~/.bashrc` file: `export SUMO_HOME=/path/to/flatpak/sumo` and `export PATH="$SUMO_HOME/bin:$PATH"`
  1. Reload your session via `source ~/.bashrc`

Furthermore, when using flatpak, you will have to explicitely start SUMO on a certain TCP port and tell TraCi to connect to it -> not recommended for comfort. 
-->
 
 <!-- TODO: supply Docker container for easy Testing for Prof. -->

### Python setup


Create a virtual environment and install dependencies:
  ```bash
  python -m venv venv
  source venv/bin/activate        # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```




### Editing or changing the simulation map or scenario
The underlying map and the simulated scenarios can be easily customized and switched out. This can be done by replacing the correspondant SUMO files (circle.net.xml, circle.add.xml, circle.sumocfg).
More information can be found here: [Link to SUMO documentation]
In this way, the map with all its attributes as well as the number, position and specifica of charging stations can be changed.

## Overview


## Project structure

![Overview](./docs/figures/architecture.excalidraw.png)

### Sumo

### Simulation Interface
simulation.py

### Gymnasium Environment
environment.py

### Map/ Network Generator
network_generator.py

### Scenario Generator
scenarion_generator.py
* Default scenario
* Same route scenario
* Same SOC same route scenario

### Reward Strategy
rewards.py
* Basic reward strategy
* No time component reward strategy
* Reward shaping strategy

### NMEV Dataset Strategy
data_processing.py
* Random data provider
* OBELIS data provider

### Training and Evaluation
training.ipynb

#### Evaluation Algorithms
evaluation_algorithms.py
* Random
* Greedy
* Never charge

## Usage

#### Experiment Tracking
Experiments can optionally be tracked using Weights & Biases.
This is not required to run the code or reproduce results.
When disabled, all results are written to local files (JSON, TensorBoard logs, model checkpoints).
To enable W&B tracking, call `train_model(...)` and `evaluate_model(...)` with the parameter `use_wandb=True`.

### Loggers

#### Interpreting tensorboard values
* `env/charging_stops_per_episode_mean`:
* `env/global_ttt`: The cumulated total travel time of all member vehicles, in seconds.
* `env/global_ttt_only_terminated`: global_ttt but only for episodes which terminated (= were not truncated)
* `env/ttt_per_ev_mean`: global_ttt divided by the number of member EVs
* `env/ttt_per_ev_mean_only_terminated`: ttt_per_ev_mean but only for episodes which terminated (= were not truncated)
* `env/cumulated_waiting_time`:
* `env/cumulated_waiting_time_only_terminated`:
* `env/empty_vehicles_per_episode`: How many vehicles went completely empty
* `env/final_simulation_time`: The value of the final SUMO-timestep when the episode ended

## Data

The datasets used in this project and their documentation can be found in the "datasets" directory.

- [`./datasets/README.md`](./datasets/README.md) provides an overview of all datasets.
- Each subdirectory includes a `data_exploration.ipynb` describing the data format and usage.

## Modeling

## Results