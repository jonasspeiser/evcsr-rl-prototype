"""
Tests to find faulty commits with git bisect
"""

import os
import sys
import traceback

# Make sure imports work from the repo root that git bisect checks out
REPO_ROOT = os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def is_signature_typeerror(e: TypeError) -> bool:
    msg = str(e)
    # Common signature-mismatch messages across Python versions
    return (
        "got an unexpected keyword argument" in msg
        or "missing" in msg and "required positional argument" in msg
        or "positional argument" in msg
        or "takes" in msg and "positional" in msg
    )

def main() -> int:
    try:
        from training_utils import train_model
    except Exception:
        print("Import failed; skipping commit (module moved / deps missing?)")
        traceback.print_exc()
        return 125  # skip untestable commits

    try:
        try:  # test for the new signature of train_model
            train_model(
                scenario="all_random",
                algorithm="PPO",
                policy="MultiInputPolicy",
                version_tag="v0.7.9",
                reward_strategy="basic",
                street_network="straight100Test",
                n_vehicles=2,
                n_steps=1,
                execution_context="local",
                random_seed=None,
                use_wandb=True, wandb_entity="evcs-rl", wandb_project="v0.7.9_straight100Test_PPO"
            )
        except TypeError as e:
            # if not is_signature_typeerror(e):
            #     # A real TypeError inside training -> treat as failure
            #     raise

            try:
                # test for the old signature
                train_model(
                    algorithm="PPO",
                    policy="MultiInputPolicy",
                    version="v0.7.9",
                    env_version="basic",
                    street_network="straight100Test",
                    n_vehicles=2,
                    n_steps=1,
                    execution_context="local",
                    random_seed=None,
                )
            except TypeError as e2:
                if "CircleEnv.__init__() missing" in str(e2):
                    return 125  # skip untestable commits
                raise

        print("Training completed successfully")
        return 0  # good commit

    except Exception:
        print("Error during training:")
        traceback.print_exc()
        return 1  # bad commit

if __name__ == "__main__":
    return_code = main()
    print(f"Exiting with code {return_code}")
    sys.exit(return_code)
