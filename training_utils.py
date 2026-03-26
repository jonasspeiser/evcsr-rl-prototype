"""Script containing utility functions for training and evaluating RL models."""

from environment import CustomEnv
from network_generator import get_longest_route_duration
from evaluation_algorithms import RandomAlgorithm, GreedyAlgorithm, FixedActionAlgorithm
from collections import Counter
from stable_baselines3 import PPO, A2C, DQN
from datetime import datetime, timezone
import json
import os
from logging_utils import RunLogging, setup_run_logging
import subprocess

SB3_ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}

EVAL_ALGOS = {
    "RANDOM": RandomAlgorithm,
    "GREEDY": GreedyAlgorithm,
}


def _load_model(model_path, algorithm, env):
    if algorithm in SB3_ALGOS:
        try:
            return SB3_ALGOS[algorithm].load(model_path, env=env)
        except ValueError as e:
            # Compatibility handling for old models with renamed observation space keys (member_vehicle -> observable_vehicle)
            # TODO: Remove this try/catch block once all models are based on v0.9.2 or newer
            if "Observation spaces do not match" not in str(e):
                raise
            print(f"Warning: Observation space key mismatch (old model with renamed keys). Retrying with space override.\n  {e}")
            return SB3_ALGOS[algorithm].load(model_path, env=env, custom_objects={"observation_space": env.observation_space})
    if algorithm in EVAL_ALGOS:
        return EVAL_ALGOS[algorithm](environment=env)
    if algorithm.startswith("ACTION") and algorithm[6:].isdigit():
        return FixedActionAlgorithm(environment=env, action=int(algorithm[6:]))
    raise ValueError(f"Invalid model type: {algorithm}")


def _dump_to_file(content, filepath):
    with open(filepath, 'w') as f:
        json.dump(content, f, indent=2)

def _save_run_config(directory, config):
    """Save run configuration to run_config.json in the given directory."""
    _dump_to_file(config, os.path.join(directory, "run_config.json"))

def _load_run_config(model_load_path):
    """Load the run config for the given model.

    Check for config files saved through further_train_model first (model_load_path with _config.json suffix), then fall back to the original run_config.json in the same directory as the model .zip file.
    """
    model_config_path = model_load_path.replace(".zip", "_config.json")
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            return json.load(f)
    run_dir = os.path.dirname(os.path.abspath(model_load_path))
    config_path = os.path.join(run_dir, "run_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def _run_training(*, env, log: RunLogging, model, n_steps, reset_num_timesteps=True):
    try:
        model.learn(n_steps, tb_log_name="tensorboard", callback=log.callback, reset_num_timesteps=reset_num_timesteps)
        model.save(log.model_save_path)
        return log.model_save_path
    except Exception as e:
        try:
            # Ensure that the model is saved even if an error occurs during training
            model.save(log.model_save_path)
        except Exception as e2:
            # If saving fails, skip it to avoid masking the original exception
            print(f"Error saving model after exception: {e2}")
            
        # dump crash bundle for debugging (ring buffer log + environment snapshot)
        log.dump_and_log_crash_bundle(env_snapshot=env.get_snapshot(),exc=e,context={"n_steps": n_steps})

        log.mark_failed(e)
        raise
    finally:
        env.close()
        log.close()

def training_units_to_steps(n_training_units, n_oevs):
    """Convert training units to environment steps.

    One training unit roughly corresponds to one episode.
    Each OEV generates ~3 charging requests per episode (spawn + low battery/post-charge),
    so steps per episode ≈ n_oevs * 3.
    """
    return n_training_units * n_oevs * 3


def get_latest_model(runs_dir="runs"):
    """Return the path to the most recently trained model .zip file."""
    run_dirs = sorted(
        e for e in (os.path.join(runs_dir, d) for d in os.listdir(runs_dir))
        if os.path.isdir(e)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No runs found in {runs_dir!r}")
    for run_dir in reversed(run_dirs):
        models = sorted(
            e for e in (os.path.join(run_dir, f) for f in os.listdir(run_dir))
            if e.endswith(".zip")
        )
        if models:
            return models[-1]
    raise FileNotFoundError(f"No model .zip found in any run directory under {runs_dir!r}")

def get_git_version():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def evaluate_policy(model, env, n_eval_episodes, callback, metadata, random_seed):
    metrics_list = []
    total_step = 0

    identifier = f"{metadata.get('algorithm','?')}-{metadata.get('version_tag','?')}"

    for episode in range(n_eval_episodes):
        # Note: We add the episode number to the seed to ensure different trajectories across episodes while maintaining reproducibility.
        seed = None if random_seed is None else random_seed + episode
        # if you instead want to use the same seed for all episodes, uncomment the following line and comment out the one above:
        # seed = random_seed
        observation, info = env.reset(seed=seed)
        episode_reward = 0.0
        episode_length = 0
        terminated = truncated = False

        action_counts = Counter()
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            action_counts[int(action)] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            total_step += 1
            episode_reward += reward
            episode_length += 1

        # Retrieve metrics from the environment after an episode ends.
        episode_metrics = callback.log_evaluation(env, total_step)
        episode_metrics.update({
            "episode": episode,
            "episode_length": episode_length,
            "was_truncated": truncated,
            "reward": episode_reward,
            "action_counts": action_counts,
            **metadata
            })
        metrics_list.append(episode_metrics)
        
        print(f"[{identifier}] Episode {episode + 1}: reward = {episode_reward:.2f}, length = {episode_length}")
    
    return metrics_list

def train_model(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles, n_training_units, n_noevs=None, max_vehicles=None, execution_context="local", random_seed=None, ent_coef=0.0, longest_route_duration=None, use_wandb=False, wandb_entity=None):
    """Trains a reinforcement learning model with the specified configuration and logs the training process.

    Args:
        scenario (str): The scenario configuration for the environment.
        algorithm (str): The RL algorithm to use for training.
        policy (str): The policy configuration to use.
        version_tag (str): The current environment version (current git version tag).
        reward_strategy (str): The reward strategy to use in the environment.
        street_network (str): The street network directory name to use (e.g., "straight_100km", "straight_120km").
            Must match a subdirectory under street-networks/ containing the corresponding SUMO files.
        n_vehicles (int): The number of OEVs (observable electric vehicles) in the environment.
        n_training_units (int): The number of training units (roughly episodes) to train for.
        n_noevs (int, optional): The number of NOEVs (non observable electric vehicles). Defaults to None.
        execution_context (str, optional): The execution context (e.g., "local", "remote"). Defaults to "local".
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
        ent_coef (float, optional): Entropy regularization coefficient for PPO/A2C. Higher values encourage
            more exploration by penalizing overconfident policies. 0.01 is a typical starting point.
            Defaults to 0.0 (SB3 default, no entropy bonus). Not used for DQN.
        longest_route_duration (int, optional): The maximum route duration in seconds for the given street network. Used for episode truncation and as the upper bound in the shaping reward. Defaults to the value returned by get_longest_route_duration(street_network).
        use_wandb (bool, optional): Whether to log training with Weights & Biases. Defaults to False.
        wandb_entity (str, optional): The Weights & Biases entity (project/team) to log under, if use_wandb is True. Defaults to None.

    Returns:
        The file path of the saved model (.zip).
    """
    n_steps = training_units_to_steps(n_training_units, n_vehicles)

    # setup logging
    log = setup_run_logging(
        algorithm=algorithm,
        version_tag=version_tag,
        reward_strategy=reward_strategy,
        scenario=scenario,
        street_network=street_network,
        n_vehicles=n_vehicles,
        n_noevs=n_noevs,
        n_steps=n_steps,
        mode="training",
        execution_context=execution_context,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
    )

    _save_run_config(log.run_dir, {
        "version_tag": version_tag,
        "mode": "training",
        "algorithm": algorithm,
        "policy": policy,
        "reward_strategy": reward_strategy,
        "scenario": scenario,
        "street_network": street_network,
        "n_vehicles": n_vehicles,
        "n_noevs": n_noevs,
        "n_training_units": n_training_units,
        "n_steps": n_steps,
        "random_seed": random_seed,
        "ent_coef": ent_coef,
        "max_vehicles": max_vehicles,
        "longest_route_duration": longest_route_duration,
        "execution_context": execution_context,
    })

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=None, reward_strategy=reward_strategy, vehicles_to_spawn=n_vehicles, max_vehicles=max_vehicles, random_seed=None, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'), street_network=street_network, longest_route_duration=longest_route_duration or get_longest_route_duration(street_network))

    # Train the agent
    algorithm_class = SB3_ALGOS.get(algorithm)
    if algorithm_class is None:
        raise ValueError(f"Invalid model type for training: {algorithm}")
    algo_kwargs = {"ent_coef": ent_coef} if algorithm in ("PPO", "A2C") else {}
    model = algorithm_class(policy, env, seed=random_seed, verbose=1, tensorboard_log=log.run_dir, **algo_kwargs)
    return _run_training(env=env, log=log, model=model, n_steps=n_steps)

def further_train_model(model_load_path, n_training_units, execution_context="local", use_wandb=False, wandb_entity=None):
    """Continue training an existing model, loading scenario/algorithm/etc. from its run_config.json.

    Args:
        model_load_path (str): The file path to the existing model .zip file to load and continue training.
        n_training_units (int): The number of additional training units (roughly episodes) to train for.
        execution_context (str, optional): The execution context (e.g., "local", "remote"). Defaults to "local".
        use_wandb (bool, optional): Whether to log training with Weights & Biases. Defaults to False.
        wandb_entity (str, optional): The Weights & Biases entity (project/team) to log under, if use_wandb is True. Defaults to None.

    Returns:
        The file path of the newly saved model (.zip) after further training.
    """
    config = _load_run_config(model_load_path)
    scenario = config["scenario"]
    algorithm = config["algorithm"]
    reward_strategy = config["reward_strategy"]
    street_network = config["street_network"]
    n_vehicles = config["n_vehicles"]
    n_noevs = config.get("n_noevs")
    ent_coef = config.get("ent_coef", 0.0)
    max_vehicles = config.get("max_vehicles")
    longest_route_duration = config.get("longest_route_duration")
    version_tag = get_git_version()
    n_steps = training_units_to_steps(n_training_units, n_vehicles)

    # setup logging
    log = setup_run_logging(
        algorithm=algorithm,
        version_tag=version_tag,
        reward_strategy=reward_strategy,
        scenario=scenario,
        street_network=street_network,
        n_vehicles=n_vehicles,
        n_noevs=n_noevs,
        n_steps=n_steps,
        mode="training",
        model_load_path=model_load_path,
        execution_context=execution_context,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
    )

    # Save alongside the new model file (not in run_dir root, to avoid overwriting the original run_config.json).
    # base_config nests the previous run's config, so the full training history is preserved for chains of
    # train_model -> further_train_model -> further_train_model -> ...
    _dump_to_file({
        "version_tag": version_tag,
        "mode": "training_continued",
        "algorithm": algorithm,
        "policy": config.get("policy"),
        "reward_strategy": reward_strategy,
        "scenario": scenario,
        "street_network": street_network,
        "n_vehicles": n_vehicles,
        "n_noevs": n_noevs,
        "n_training_units": n_training_units,
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "max_vehicles": max_vehicles,
        "longest_route_duration": longest_route_duration,
        "execution_context": execution_context,
        "continued_from": model_load_path,
        "base_config": config,
    }, log.model_save_path.replace(".zip", "_config.json"))

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=None, reward_strategy=reward_strategy, vehicles_to_spawn=n_vehicles, max_vehicles=max_vehicles, random_seed=None, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'), street_network=street_network, longest_route_duration=longest_route_duration or get_longest_route_duration(street_network))

    # Train the agent
    model = _load_model(model_load_path, algorithm, env)
    return _run_training(env=env, log=log, model=model, n_steps=n_steps, reset_num_timesteps=False)

def evaluate_model(scenario, algorithm, version_tag, reward_strategy, street_network, n_vehicles, n_episodes, model_load_path=None, n_noevs=None, execution_context="local", render_mode=None, random_seed=None, longest_route_duration=None, use_wandb=False, wandb_entity=None):
    """
    Evaluates a trained model or baseline algorithm in the specified environment configuration and logs the evaluation metrics.

    Args:
        scenario (str): The scenario configuration for the environment.
        algorithm (str): The RL algorithm or baseline algorithm to evaluate.
        version_tag (str): The current environment version (current git version tag).
        reward_strategy (str): The reward strategy to use in the environment.
        street_network (str): The street network directory name to use (e.g., "straight_100km", "straight_120km").
            Must match a subdirectory under street-networks/ containing the corresponding SUMO files.
        n_vehicles (int): The number of OEVs (observable electric vehicles) in the environment.
        n_episodes (int): The number of episodes to run for evaluation.
        model_load_path (str, optional): The file path to the trained model .zip file   to load for evaluation. If None, the function will evaluate a baseline algorithm specified by the `algorithm` argument. Defaults to None.
        n_noevs (int, optional): The number of NOEVs (non observable electric vehicles). Defaults to None.
        execution_context (str, optional): The execution context (e.g., "local", "remote"). Defaults to "local".
        render_mode (str, optional): The render mode to use for the environment during evaluation (e.g., "human", "rgb_array"). Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
        longest_route_duration (int, optional): The maximum route duration in seconds for the given street network. Used for episode truncation and as the upper bound in the shaping reward. Defaults to the value from the model's run config, or get_longest_route_duration(street_network) if not set.
        use_wandb (bool, optional): Whether to log evaluation with Weights & Biases. Defaults to False.
        wandb_entity (str, optional): The Weights & Biases entity (project/team) to log under, if use_wandb is True. Defaults to None.

    Returns:
        The file path of the evaluation metrics file (str).
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metadata = {
        "timestamp": current_time,
        "algorithm": algorithm,
        "version_tag": version_tag,
        "reward_strategy": reward_strategy,
        "street_network": street_network,
        "n_vehicles": n_vehicles,
        "n_episodes": n_episodes,
        "random_seed": random_seed
    }

    # setup logging
    log = setup_run_logging(
        algorithm=algorithm,
        version_tag=version_tag,
        reward_strategy=reward_strategy,
        scenario=scenario,
        street_network=street_network,
        n_vehicles=n_vehicles,
        n_noevs=n_noevs,
        n_steps=None,
        mode="evaluation",
        model_load_path=model_load_path,
        execution_context=execution_context,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
    )

    # Load training hyperparameters from the model's run config for traceability in the evaluation config.
    training_config = _load_run_config(model_load_path) if model_load_path else {}
    max_vehicles = training_config.get("max_vehicles")
    longest_route_duration = longest_route_duration or training_config.get("longest_route_duration")

    # Use the same timestamp as the metrics file so configs and metrics are paired by name
    _dump_to_file({
        "version_tag": version_tag,
        "mode": "evaluation",
        "algorithm": algorithm,
        "reward_strategy": reward_strategy,
        "scenario": scenario,
        "street_network": street_network,
        "n_vehicles": n_vehicles,
        "n_noevs": n_noevs,
        "n_episodes": n_episodes,
        "random_seed": random_seed,
        "ent_coef": training_config.get("ent_coef"),
        "n_training_units": training_config.get("n_training_units"),
        "n_steps": training_config.get("n_steps"),
        "model_load_path": model_load_path,
        "longest_route_duration": longest_route_duration,
        "execution_context": execution_context,
        "training_config": training_config or None,
    }, f"{log.run_dir}/evaluation/run_config_{current_time}.json")

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=render_mode, reward_strategy=reward_strategy, vehicles_to_spawn=n_vehicles, max_vehicles=max_vehicles, random_seed=random_seed, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'), street_network=street_network, longest_route_duration=longest_route_duration or get_longest_route_duration(street_network))

    # Load saved model or evaluation algorithm
    model = _load_model(model_load_path, algorithm, env)

    # Evaluate the agent
    try:
        metrics_dict = evaluate_policy(model, env, n_eval_episodes=n_episodes, callback=log.callback, metadata=metadata, random_seed=random_seed)
        # print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")
        evaluation_path = f"{log.run_dir}/evaluation/metrics{current_time}.json"
        _dump_to_file(metrics_dict, evaluation_path)
        return evaluation_path

    except KeyboardInterrupt as e:
        log.mark_interrupted(e)
        raise

    finally:
        env.close()
        log.close()

def evaluate_model_with_config(model_load_path, n_episodes, random_seed=None, render_mode=None, execution_context="local", use_wandb=False, wandb_entity=None):
    """Evaluate a trained model, loading scenario/algorithm/etc. from its run_config.json.

    Returns:
        The file path of the evaluation metrics file (str).
    """
    config = _load_run_config(model_load_path)
    return evaluate_model(
        scenario=config["scenario"],
        algorithm=config["algorithm"],
        version_tag=get_git_version(),
        reward_strategy=config["reward_strategy"],
        street_network=config["street_network"],
        n_vehicles=config["n_vehicles"],
        n_noevs=config.get("n_noevs"),
        n_episodes=n_episodes,
        model_load_path=model_load_path,
        render_mode=render_mode,
        random_seed=random_seed,
        execution_context=execution_context,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
    )

def train_and_evaluate(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles, n_training_units, n_noevs=None, max_vehicles=None, execution_context="local", random_seed_training=None, random_seed_eval=123, eval_episodes=50, ent_coef=0.0, longest_route_duration=None):
    """Trains a reinforcement learning model and evaluates it against baseline algorithms.

    This function trains a new model using the specified algorithm and policy,
    then evaluates the trained model's performance alongside RANDOM and GREEDY
    baseline algorithms.

    Args:
        scenario (str): The scenario configuration for the environment.
        algorithm (str): The RL algorithm to use for training.
        policy (str): The policy configuration to use.
        version_tag (str): The current environment version (current git version tag).
        reward_strategy (str): The reward strategy to use in the environment.
        street_network (str): The street network directory name to use (e.g., "straight_100km", "straight_120km").
            Must match a subdirectory under street-networks/ containing the corresponding SUMO files.
        n_vehicles (int): The number of OEVs (observable electric vehicles) in the environment.
        n_training_units (int): The number of training units (roughly episodes) to train for.
        n_noevs (int, optional): The number of NOEVs (non observable electric vehicles).
            Defaults to None.
        execution_context (str, optional): The execution context (e.g., "local", "cloud").
            Defaults to "local".
        random_seed_training (int, optional): Random seed for training reproducibility.
            Defaults to None.
        random_seed_eval (int, optional): Random seed for evaluation reproducibility.
            Defaults to 123.
        eval_episodes (int, optional): The number of episodes to run during evaluation.
            Defaults to 50.
        ent_coef (float, optional): Entropy regularization coefficient for PPO/A2C. Defaults to 0.0.
        longest_route_duration (int, optional): The maximum route duration in seconds for the given street network. Passed to both train_model and evaluate_model. Defaults to get_longest_route_duration(street_network).

        tuple: A tuple containing:
            - model_path (str): The file path of the trained model.
            - eval_metrics_filepath_list (List[str]): A list of the filepaths for the evaluation metrics files of the trained model, the RANDOM and the GREEDY baseline algorithm.
    """

    # train new model
    model_path = train_model(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles=n_vehicles, n_training_units=n_training_units, n_noevs=n_noevs, max_vehicles=max_vehicles, execution_context=execution_context, random_seed=random_seed_training, ent_coef=ent_coef, longest_route_duration=longest_route_duration)
    # evaluate with random and greedy
    model_evaluation_path = evaluate_model(scenario, algorithm, version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=model_path, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval, longest_route_duration=longest_route_duration)
    nocharge_evaluation_path = evaluate_model(scenario, "ACTION0", version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=None, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval, longest_route_duration=longest_route_duration)
    random_evaluation_path = evaluate_model(scenario, "RANDOM", version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=None, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval, longest_route_duration=longest_route_duration)
    greedy_evaluation_path = evaluate_model(scenario, "GREEDY", version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=None, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval, longest_route_duration=longest_route_duration)
    eval_metrics_filepath_list = [nocharge_evaluation_path, random_evaluation_path, greedy_evaluation_path, model_evaluation_path]
    return model_path, eval_metrics_filepath_list

if __name__ == "__main__":
    
    train_and_evaluate(
        scenario="same_route",
        algorithm="PPO",
        policy="MultiInputPolicy",
        version_tag=get_git_version(),
        reward_strategy="basicCongestion",
        street_network="straight_120km",
        n_vehicles=20,
        n_noevs=0,
        n_training_units=600,
        # longest_route_duration=7_200,
        ent_coef=0.01,
        eval_episodes=10,
        random_seed_eval=54321,
        execution_context="local",
    )

    # train_model(
    #     scenario="same_route",
    #     algorithm="PPO",
    #     policy="MultiInputPolicy",
    #     version_tag=get_git_version(),
    #     reward_strategy="basic",
    #     street_network="straight_100km",
    #     n_vehicles=20,
    #     n_noevs=0,
    #     n_training_units=100,
    #     use_wandb=False,
    #     wandb_entity="evcs-rl"
    # )

    # further_train_model(
    #     model_load_path=get_latest_model(),
    #     n_training_units=600,
    #     execution_context="local"
    # )

    # evaluate_model_with_config(
    #     # model_load_path=get_latest_model(),
    #     model_load_path="runs/2026-03-05_22-28-05_v0.9.5-5-g631ed03_basic_same_route_straight100km_PPO/2026-03-05_22-28-05_v0.9.5-5-g631ed03_basic_same_route_straight100km_PPO.zip",
    #     n_episodes=5,
    #     random_seed=123,
    # )

    # evaluate_model(
    #     scenario="same_route",
    #     algorithm="GREEDY",
    #     version_tag=get_git_version(),
    #     reward_strategy="basic",
    #     street_network="straight_120km",
    #     n_vehicles=20,
    #     n_noevs=0,
    #     n_episodes=20,
    #     longest_route_duration=6_000,
    #     model_load_path=None,
    #     execution_context="local",
    #     # render_mode="human",
    #     random_seed=54321,
    # )
