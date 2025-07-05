## Install Eclipse SUMO

## Python setup

#### Create new venv (optional but recommended)
1. Run the venv creation command:

   - On **Windows**:
     ```bash
     python -m venv evcsr-rl-prototype
     ```
   - On **macOS/Linux**:
     ```bash
     python3 -m venv evcsr-rl-prototype
     ```

1. Activate the virtual environment:

   - On **Windows** (cmd):
     ```cmd
     evcsr-rl-prototype\Scripts\activate
     ```
   - On **Windows** (PowerShell):
     ```powershell
     .\evcsr-rl-prototype\Scripts\Activate.ps1
     ```
   - On **macOS/Linux**:
     ```bash
     source evcsr-rl-prototype/bin/activate
     ```
   After activation, your command prompt will show the environment name, indicating it's active.  


#### Install dependencies
```bash
pip install -r requirements.txt
```




## Editing or changing the simulation map or scenario
The underlying map and the simulated scenarios can be easily customized and switched out. This can be done by replacing the correspondant SUMO files (circle.net.xml, circle.add.xml, circle.sumocfg).
More information can be found here: [Link to SUMO documentation]
In this way, the map with all its attributes as well as the number, position and specifica of charging stations can be changed.