
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
 - else: refer to https://sumo.dlr.de/docs/Downloads.php

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

## Usage

## Data

The datasets used in this project and their documentation can be found in the "datasets" directory.

- [`./datasets/README.md`](./datasets/README.md) provides an overview of all datasets.
- Each subdirectory includes a `data_exploration.ipynb` describing the data format and usage.

## Modeling

## Results