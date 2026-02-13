"""
Tests to find faulty commits with git bisect
"""

import sys
from training_utils import train_model

try:
    try: # test for the new signature of train_model
        train_model(
            scenario="all_random",
            algorithm="PPO", 
            policy="MultiInputPolicy", 
            version_tag="v0.7.9", 
            reward_strategy="basic", 
            map="straight100Test", 
            n_vehicles=2, n_steps=1, execution_context="local", random_seed=None,
        )
    except TypeError: # test for the old signature
        train_model(
            algorithm="PPO", 
            policy="MultiInputPolicy", 
            version="v0.7.9", 
            env_version="basic", 
            map="straight100Test", 
            n_vehicles=2, n_steps=1, execution_context="local", random_seed=None,
        )
except Exception as e:
    print("Error during training:", e)
    sys.exit(1)  # bad commit

sys.exit(0)  # good commit