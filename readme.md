## Python setup
- Stable Baselines 3 via `conda install -c conda-forge stable-baselines3`
- TensorBoard via `conda install -c conda-forge tensorboard`
- Sumo TraCI via `pip install traci`

## Editing or changing the simulation map or scenario
The underlying map and the simulated scenarios can be easily customized and switched out. This can be done by replacing the correspondant SUMO files (circle.net.xml, circle.add.xml, circle.sumocfg).
More information can be found here: [Link to SUMO documentation]
In this way, the map with all its attributes as well as the number, position and specifica of charging stations can be changed.