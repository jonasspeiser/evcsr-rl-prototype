"""Script containing utility functions for training and evaluating RL models."""

from environment import CustomEnv
from network_generator import get_longest_route_duration
from evaluation_algorithms import RandomAlgorithm, GreedyAlgorithm, FixedActionAlgorithm, Perfect5VehAlgorithm, Perfect20VehAlgorithm, PerfectXVehAlgorithm, BestGuessAlgorithm
from feature_extractor import VehicleSetExtractor
from collections import Counter
from stable_baselines3 import PPO, A2C, DQN
from datetime import datetime, timezone
import json
import os
import time
from multiprocessing import Process
from logging_utils import RunLogging, setup_run_logging, _find_tb_callback
import subprocess

_STATION_CAPACITY = 2
"""Number of vehicles that can charge simultaneously at one station.
Derived from the station lane length (10 m) and the default SUMO vehicle length (~5 m)."""

_CHARGING_DURATION = 1300
"""Charging duration per vehicle in seconds. Must match CHARGING_DURATION in simulation.py."""


SB3_ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}

EVAL_ALGOS = {
    "RANDOM": RandomAlgorithm,
    "GREEDY": GreedyAlgorithm,
    "BEST_GUESS": BestGuessAlgorithm,
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
    if algorithm.startswith("PERFECT") and algorithm[7:].isdigit():
        n = int(algorithm[7:])
        if n == 5:
            return Perfect5VehAlgorithm(environment=env)
        elif n == 20:
            return Perfect20VehAlgorithm(environment=env)
        else:
            return PerfectXVehAlgorithm(environment=env, n_vehicles=n)
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
    # run_config.json lives in the run dir. The model may sit in the run dir itself
    # (final model) or in a checkpoints/ subdir (SB3 CheckpointCallback), so walk up.
    model_dir = os.path.dirname(os.path.abspath(model_load_path))
    for run_dir in (model_dir, os.path.dirname(model_dir)):
        config_path = os.path.join(run_dir, "run_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(f"run_config.json not found near {model_load_path}")

def _run_training(*, env, log: RunLogging, model, n_steps, reset_num_timesteps=True):
    try:
        t_start = time.monotonic()
        model.learn(n_steps, tb_log_name="tensorboard", callback=log.callback, reset_num_timesteps=reset_num_timesteps)
        duration_s = time.monotonic() - t_start
        model.save(log.model_save_path)
        return log.model_save_path, duration_s
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
    results = get_latest_n_models(1, runs_dir=runs_dir)
    return results[0]

def get_latest_n_models(n: int, runs_dir="runs") -> list[str]:
    """Return paths to the .zip files from the n most recently created run directories.

    Directories are sorted alphabetically (timestamp prefix ensures chronological order).
    Prints a warning and returns a shorter list if fewer than n models are found.
    """
    run_dirs = sorted(
        e for e in (os.path.join(runs_dir, d) for d in os.listdir(runs_dir))
        if os.path.isdir(e)
    )
    results = []
    for run_dir in reversed(run_dirs):
        models = sorted(
            e for e in (os.path.join(run_dir, f) for f in os.listdir(run_dir))
            if e.endswith(".zip")
        )
        if models:
            results.append(models[-1])
        if len(results) == n:
            return results
    if not results:
        print(f"Warning: no models found in {runs_dir!r}")
    elif len(results) < n:
        print(f"Warning: requested {n} models but only {len(results)} found in {runs_dir!r}")
    return results

def get_git_version():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"
    
def get_episode_truncation_limit(street_network: str, n_vehicles: int, n_stations: int = 4, charging_duration: int = _CHARGING_DURATION) -> int:
    """
    Computes the episode truncation limit in simulation seconds.

    Accounts for both driving time and worst-case queuing time at charging stations.
    Worst case: all vehicles queue at a single station -> (n_vehicles / STATION_CAPACITY) * CHARGING_DURATION.

    Args:
        street_network (str): Street network directory name, e.g. "straight_120km".
        n_vehicles (int): Number of vehicles in the episode.
        n_stations (int): Number of charging stations. Defaults to 4.
        charging_duration (int): Charging duration per vehicle in seconds.
            Must match CHARGING_DURATION in simulation.py. Defaults to _CHARGING_DURATION.
    """
    longest_route = get_longest_route_duration(street_network)
    worst_case_charging = (n_vehicles / _STATION_CAPACITY) * charging_duration
    return longest_route + worst_case_charging


def evaluate_policy(model, env, n_eval_episodes, callback, metadata, random_seed, deterministic=True):
    metrics_list = []
    total_step = 0

    identifier = f"{metadata.get('algorithm','?')}-{metadata.get('version_tag','?')}"

    for episode in range(n_eval_episodes):
        # Note: We add the episode number to the seed to ensure different trajectories across episodes while maintaining reproducibility.
        seed = None if random_seed is None else random_seed + episode
        # if you instead want to use the same seed for all episodes, uncomment the following line and comment out the one above:
        # seed = random_seed
        observation, info = env.reset(seed=seed)
        if hasattr(model, 'reset'): # guard for stable-baselines3 models which don't have a reset method
            model.reset()
        episode_reward = 0.0
        episode_length = 0
        terminated = truncated = False

        action_counts = Counter()
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            action_counts[int(action)] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            total_step += 1
            episode_reward += reward
            episode_length += 1

        # Retrieve metrics from the environment after an episode ends.
        episode_metrics = callback.log_evaluation(env, total_step, episode_length=episode_length, episode_reward=episode_reward)
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

def train_model(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles, n_training_units, n_noevs=None, noev_provider="obelis", max_vehicles=None, execution_context="local", random_seed=None, ent_coef=0.0, longest_route_duration=None, use_wandb=False, wandb_entity=None, reward_kwargs=None, start_soc_bounds=None, obs_features=None, use_custom_extractor=False, checkpoint_freq=None, max_training_hours=None):
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
        max_vehicles (int, optional): Size of the observation space vehicle dimension. Must be >= n_vehicles.
            When None, defaults to n_vehicles. Set to a larger value to allow evaluating on more vehicles
            than were trained with (cross-N generalization), as long as max_vehicles stays constant.
        ent_coef (float, optional): Entropy regularization coefficient for PPO/A2C. Higher values encourage
            more exploration by penalizing overconfident policies. 0.01 is a typical starting point.
            Defaults to 0.0 (SB3 default, no entropy bonus). Not used for DQN.
        longest_route_duration (int, optional): The maximum route duration in seconds for the given street network. Used for episode truncation and as the upper bound in the shaping reward. Defaults to the value returned by get_longest_route_duration(street_network).
        use_wandb (bool, optional): Whether to log training with Weights & Biases. Defaults to False.
        wandb_entity (str, optional): The Weights & Biases entity (project/team) to log under, if use_wandb is True. Defaults to None.
        reward_kwargs (dict, optional): Scalar parameters forwarded to the reward strategy and observation
            logic. Supported keys: "congestion_threshold_m", "congestion_penalty", "battery_penalty_value".
            Defaults to None (all strategy parameters use their own defaults).
        start_soc_bounds (tuple, optional): (min_soc_wh, max_soc_wh) override for vehicle starting SOC. Defaults to None (uses value from the street network config).
        obs_features (set, optional): Set of optional observation keys to include. Defaults to None (no optional features included).
        use_custom_extractor (bool, optional): Whether to use the DeepSets-based EVChargingFeatureExtractor.
            Provides permutation-invariant vehicle aggregation. Requires obs_features to be empty (None)
            or the extractor will handle extra keys by concatenating them after the pooled embeddings.
            Defaults to False (SB3 default MLP policy).

    Returns:
        The file path of the saved model (.zip).
    """
    n_steps = training_units_to_steps(n_training_units, n_vehicles)
    longest_route_duration = longest_route_duration or get_episode_truncation_limit(street_network, n_vehicles)

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
        checkpoint_freq=checkpoint_freq,
        max_training_hours=max_training_hours,
    )

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=None, reward_strategy=reward_strategy, vehicles_to_spawn=n_vehicles, max_vehicles=max_vehicles, non_observable_vehicles=n_noevs, noev_provider=noev_provider, random_seed=random_seed, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'), street_network=street_network, longest_route_duration=longest_route_duration, reward_kwargs=reward_kwargs, start_soc_bounds=start_soc_bounds, obs_features=obs_features)

    # Train the agent
    algorithm_class = SB3_ALGOS.get(algorithm)
    if algorithm_class is None:
        raise ValueError(f"Invalid model type for training: {algorithm}")
    algo_kwargs = {}
    if algorithm in ("PPO", "A2C"):
        algo_kwargs["ent_coef"] = ent_coef
    if use_custom_extractor:
        algo_kwargs["policy_kwargs"] = {"features_extractor_class": VehicleSetExtractor}
    model = algorithm_class(policy, env, seed=random_seed, verbose=1, tensorboard_log=log.run_dir, **algo_kwargs)
    model_path, duration_s = _run_training(env=env, log=log, model=model, n_steps=n_steps)

    tb_cb = _find_tb_callback(log.callback)
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
        "noev_provider": noev_provider,
        "n_training_units": n_training_units,
        "n_steps": n_steps,
        "random_seed": random_seed,
        "ent_coef": ent_coef,
        "max_vehicles": max_vehicles,
        "longest_route_duration": longest_route_duration,
        "execution_context": execution_context,
        "reward_kwargs": reward_kwargs,
        "start_soc_bounds": start_soc_bounds,
        "obs_features": list(obs_features) if obs_features is not None else None,
        "use_custom_extractor": use_custom_extractor,
        "max_training_hours": max_training_hours,
        "training_duration_s": round(duration_s),
        "actual_steps": model.num_timesteps,
        "total_episodes": tb_cb._episode_count if tb_cb is not None else None,
        "training_end_timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return model_path

def further_train_model(model_load_path, n_training_units, execution_context="local", use_wandb=False, wandb_entity=None, checkpoint_freq=None, max_training_hours=None):
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
    noev_provider = config.get("noev_provider", "obelis")
    ent_coef = config.get("ent_coef", 0.0)
    max_vehicles = config.get("max_vehicles")
    longest_route_duration = config.get("longest_route_duration") or get_episode_truncation_limit(street_network, n_vehicles)
    reward_kwargs = config.get("reward_kwargs")
    if reward_kwargs is None and config.get("congestion_kwargs") is not None:
        raise KeyError("run_config.json uses the old key 'congestion_kwargs' — rename it to 'reward_kwargs'.")
    start_soc_bounds = config.get("start_soc_bounds")
    obs_features = config.get("obs_features")
    use_custom_extractor = config.get("use_custom_extractor", False)
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
        checkpoint_freq=checkpoint_freq,
        max_training_hours=max_training_hours,
    )

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=None, reward_strategy=reward_strategy, vehicles_to_spawn=n_vehicles, max_vehicles=max_vehicles, non_observable_vehicles=n_noevs, noev_provider=noev_provider, random_seed=None, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'), street_network=street_network, longest_route_duration=longest_route_duration, reward_kwargs=reward_kwargs, start_soc_bounds=start_soc_bounds, obs_features=obs_features)

    # Train the agent
    model = _load_model(model_load_path, algorithm, env)
    initial_steps = model.num_timesteps
    model_path, duration_s = _run_training(env=env, log=log, model=model, n_steps=n_steps, reset_num_timesteps=False)

    # Save alongside the new model file (not in run_dir root, to avoid overwriting the original run_config.json).
    # base_config nests the previous run's config, so the full training history is preserved for chains of
    # train_model -> further_train_model -> further_train_model -> ...
    tb_cb = _find_tb_callback(log.callback)
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
        "noev_provider": noev_provider,
        "n_training_units": n_training_units,
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "max_vehicles": max_vehicles,
        "longest_route_duration": longest_route_duration,
        "execution_context": execution_context,
        "reward_kwargs": reward_kwargs,
        "start_soc_bounds": start_soc_bounds,
        "obs_features": list(obs_features) if obs_features is not None else None,
        "use_custom_extractor": use_custom_extractor,
        "max_training_hours": max_training_hours,
        "training_duration_s": round(duration_s),
        "actual_steps": model.num_timesteps - initial_steps,
        "total_episodes": tb_cb._episode_count if tb_cb is not None else None,
        "training_end_timestamp": datetime.now(timezone.utc).isoformat(),
        "continued_from": model_load_path,
        "base_config": config,
    }, log.model_save_path.replace(".zip", "_config.json"))

    return model_path

def evaluate_model(scenario, algorithm, version_tag, reward_strategy, street_network, n_vehicles, n_episodes, model_load_path=None, n_noevs=None, noev_provider="obelis", execution_context="local", render_mode=None, random_seed=None, longest_route_duration=None, use_wandb=False, wandb_entity=None, start_soc_bounds=None, deterministic=True):
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
        start_soc_bounds (tuple, optional): (min_soc_wh, max_soc_wh) override for vehicle starting SOC. Takes precedence over the value from the model's run config. Defaults to None.

    Returns:
        The file path of the evaluation metrics file (str).
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    longest_route_duration = longest_route_duration or training_config.get("longest_route_duration") or get_episode_truncation_limit(street_network, n_vehicles)
    reward_kwargs = training_config.get("reward_kwargs")
    if reward_kwargs is None and training_config.get("congestion_kwargs") is not None:
        raise KeyError("run_config.json uses the old key 'congestion_kwargs' — rename it to 'reward_kwargs'.")
    start_soc_bounds = start_soc_bounds or training_config.get("start_soc_bounds")
    obs_features = training_config.get("obs_features")
    use_custom_extractor = training_config.get("use_custom_extractor", False)
    metadata = {
        "timestamp": current_time,
        "algorithm": algorithm,
        "version_tag": version_tag,
        "reward_strategy": reward_strategy,
        "obs_features": sorted(obs_features) if obs_features else [],
        "use_custom_extractor": use_custom_extractor,
        "street_network": street_network,
        "n_vehicles": n_vehicles,
        "n_episodes": n_episodes,
        "random_seed": random_seed
    }

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
        "noev_provider": noev_provider,
        "n_episodes": n_episodes,
        "random_seed": random_seed,
        "model_load_path": model_load_path,
        "longest_route_duration": longest_route_duration,
        "deterministic": deterministic,
        "execution_context": execution_context,
        "training_config": training_config or None,
    }, f"{log.run_dir}/evaluation/run_config_{current_time}.json")

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=render_mode, reward_strategy=reward_strategy, vehicles_to_spawn=n_vehicles, max_vehicles=max_vehicles, non_observable_vehicles=n_noevs, noev_provider=noev_provider, random_seed=random_seed, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'), street_network=street_network, longest_route_duration=longest_route_duration, reward_kwargs=reward_kwargs, start_soc_bounds=start_soc_bounds, obs_features=obs_features)

    # Load saved model or evaluation algorithm
    model = _load_model(model_load_path, algorithm, env)

    # Evaluate the agent
    try:
        metrics_dict = evaluate_policy(model, env, n_eval_episodes=n_episodes, callback=log.callback, metadata=metadata, random_seed=random_seed, deterministic=deterministic)
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

def evaluate_model_with_config(model_load_path, n_episodes, n_vehicles=None, n_noevs=None, random_seed=None, render_mode=None, execution_context="local", use_wandb=False, wandb_entity=None, deterministic=True):
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
        n_vehicles=n_vehicles if n_vehicles is not None else config["n_vehicles"],
        n_noevs=n_noevs if n_noevs is not None else config.get("n_noevs"),
        noev_provider=config.get("noev_provider", "obelis"),
        n_episodes=n_episodes,
        model_load_path=model_load_path,
        render_mode=render_mode,
        random_seed=random_seed,
        execution_context=execution_context,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        deterministic=deterministic,
    )

def get_models_from_folder(folder_path) -> list[str]:
    """Return paths to the .zip model files from every run directory inside folder_path.

    Args:
        folder_path (str): Path to a folder containing run directories (e.g. "runs/_runs_20260610_1759").

    Returns:
        List of .zip model paths, sorted alphabetically by run directory name.
    """
    folder = os.path.abspath(folder_path)
    results = []
    for run_dir in sorted(os.listdir(folder)):
        run_path = os.path.join(folder, run_dir)
        if not os.path.isdir(run_path):
            continue
        zips = sorted(f for f in os.listdir(run_path) if f.endswith(".zip"))
        if zips:
            results.append(os.path.join(run_path, zips[-1]))
    return results


def train_and_evaluate(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles, n_training_units, n_noevs=None, max_vehicles=None, execution_context="local", random_seed_training=None, random_seed_eval=123, eval_episodes=50, ent_coef=0.0, longest_route_duration=None, reward_kwargs=None, start_soc_bounds=None, obs_features=None, use_custom_extractor=False):
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
        longest_route_duration (int, optional): The maximum route duration in seconds for the given street network. Passed to both train_model and evaluate_model. Used for episode truncation and as the upper bound in the shaping reward. Defaults to get_longest_route_duration(street_network).
        reward_kwargs (dict, optional): Scalar parameters forwarded to the reward strategy and observation
            logic. Supported keys: "congestion_threshold_m", "congestion_penalty", "battery_penalty_value".
            Defaults to None (all strategy parameters use their own defaults).
        start_soc_bounds (tuple, optional): (min_soc_wh, max_soc_wh) override for vehicle starting SOC. Defaults to None (uses value from the street network config).
        obs_features (set, optional): Set of optional observation keys to include. Supports any combination of {"simulation_time", "station_assignment_counts"}. Defaults to None (no optional features included).
        use_custom_extractor (bool, optional): Whether to use the DeepSets-based EVChargingFeatureExtractor. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - model_path (str): The file path of the trained model.
            - eval_metrics_filepath_list (List[str]): A list of the filepaths for the evaluation metrics files of the trained model, the RANDOM and the GREEDY baseline algorithm.
    """

    # train new model
    model_path = train_model(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles=n_vehicles, n_training_units=n_training_units, n_noevs=n_noevs, max_vehicles=max_vehicles, execution_context=execution_context, random_seed=random_seed_training, ent_coef=ent_coef, longest_route_duration=longest_route_duration, reward_kwargs=reward_kwargs, start_soc_bounds=start_soc_bounds, obs_features=obs_features, use_custom_extractor=use_custom_extractor)
    # evaluate with random and greedy
    model_evaluation_path = evaluate_model(scenario, algorithm, version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=model_path, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval, longest_route_duration=longest_route_duration)
    nocharge_evaluation_path = evaluate_model(scenario, "ACTION0", version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=None, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval, longest_route_duration=longest_route_duration)
    random_evaluation_path = evaluate_model(scenario, "RANDOM", version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=None, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval, longest_route_duration=longest_route_duration)
    greedy_evaluation_path = evaluate_model(scenario, "GREEDY", version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=None, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval, longest_route_duration=longest_route_duration)
    eval_metrics_filepath_list = [nocharge_evaluation_path, random_evaluation_path, greedy_evaluation_path, model_evaluation_path]
    return model_path, eval_metrics_filepath_list


def run_parallel(run_list, max_workers=None, stagger_s=0):
    """Run all jobs with at most max_workers running concurrently.

    max_workers=None means unbounded (legacy behaviour). On this machine the Exp 4
    sweep MUST cap concurrency: 28 unbounded processes each running a 600-vehicle SUMO
    + loading the OBELIS feather saturated 12 cores / 30 GB and completed 0 conditions.
    stagger_s spaces out the start of each new worker to smooth the OBELIS load spike.
    """
    run_queue = list(run_list)
    cap = max_workers or len(run_queue)
    running = []
    while run_queue or running:
        while run_queue and len(running) < cap:
            p = Process(target=run_queue.pop(0))
            p.start()
            running.append(p)
            if stagger_s:
                time.sleep(stagger_s)
        for p in running[:]:
            p.join(timeout=1)          # reap finished workers, free their slot
            if not p.is_alive():
                running.remove(p)


if __name__ == "__main__":
    
    # train_and_evaluate(
    #     scenario="same_route",
    #     start_soc_bounds=(22_000, 22_000),
    #     algorithm="PPO",
    #     policy="MultiInputPolicy",
    #     version_tag=get_git_version(),
    #     reward_strategy="basic",
    #     obs_features={"simulation_time", "station_assignment_counts"}, 
    #     reward_kwargs={"congestion_threshold_m": 33600, "congestion_penalty": 500.0},
    #     street_network="straight_120km",
    #     n_vehicles=5,
    #     n_noevs=0,
    #     n_training_units=1_200,
    #     # longest_route_duration=7_200,
    #     ent_coef=0.0,
    #     eval_episodes=10,
    #     random_seed_eval=54321,
    #     execution_context="local",
    # )

    # train_model(
    #     algorithm="PPO",
    #     reward_strategy="relativeDestination",
    #     policy="MultiInputPolicy",
    #     version_tag=get_git_version(),
    #     scenario="bast",
    #     street_network="straight_120km",
    #     obs_features={"simulation_time", "station_assignment_counts"},
    #     reward_kwargs={"congestion_threshold_m": 36000, "congestion_penalty": 0.1, "battery_penalty_value": 10},
    #     use_custom_extractor=True,
    #     n_vehicles=600,
    #     max_vehicles=600,
    #     n_noevs=0,
    #     max_training_hours=0.05,
    #     n_training_units=10_000_000,
    #     ent_coef=0.1,
    #     use_wandb=False,
    #     wandb_entity="evcs-rl"
    # )

    # further_train_model(
    #     model_load_path=get_latest_model(),
    #     # model_load_path="runs/2026-05-19_14-32-08_pid1183697_v1.3.2-1-gdfd46de_basic_all_random_straight_120km_PPO/2026-05-21_12-34-34_pid2005555_v1.3.2-4-g31239cf_basic_all_random_straight_120km_PPO.zip",
    #     max_training_hours=0.5,
    #     n_training_units=10_000_000,  # effectively unlimited further training, the run will be stopped by max_training_hours or manual interruption
    #     # checkpoint_freq=15_000,
    #     execution_context="local"
    # )

    evaluate_model_with_config(
        # model_load_path=get_latest_model(),
        model_load_path="runs/2026-06-12_00-10-03_pid2420665_v1.4.1-18-g88057f4_basicCongestion_bast_straight_120km_PPO/2026-06-12_00-10-03_pid2420665_v1.4.1-18-g88057f4_basicCongestion_bast_straight_120km_PPO.zip",
        n_episodes=10,
        # longest_route_duration=6_000,
        execution_context="local",
        # render_mode="human",
        random_seed=54321,
        deterministic=True,
    )
    
    # evaluate_model(
    #     scenario="bast",
    #     # start_soc_bounds=(22_000, 22_000),
    #     algorithm="GREEDY",
    #     version_tag=get_git_version(),
    #     reward_strategy="basic",
    #     street_network="straight_120km",
    #     n_vehicles=200,
    #     n_noevs=0,
    #     n_episodes=1,
    #     # longest_route_duration=30_000,
    #     model_load_path=None,
    #     execution_context="local",
    #     render_mode="human",
    #     random_seed=54321,
    # )

    # EVALUATE FROM FOLDER:
    # for path in get_models_from_folder("runs/_runs_20260610_1759"):
    #     evaluate_model_with_config(model_load_path=path, n_episodes=10, random_seed=54321)
