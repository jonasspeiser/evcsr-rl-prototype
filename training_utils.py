"""Script containing utility functions for training and evaluating RL models."""

from environment import CustomEnv
from evaluation_algorithms import RandomAlgorithm, GreedyAlgorithm, NeverChargeAlgorithm
from stable_baselines3 import PPO, A2C, DQN
from datetime import datetime
import json
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
    "NOCHARGE": NeverChargeAlgorithm,
}


def _load_model(model_path, algorithm, env):
    if algorithm in SB3_ALGOS:
        return SB3_ALGOS[algorithm].load(model_path, env=env)
    if algorithm in EVAL_ALGOS:
        return EVAL_ALGOS[algorithm](environment=env)
    raise ValueError(f"Invalid model type: {algorithm}")


def _dump_to_file(content, filepath):
    with open(filepath, 'w') as f:
        json.dump(content, f, indent=2)

def _run_training(*, env, log: RunLogging, model, n_steps):
    try:
        model.learn(n_steps, tb_log_name="tensorboard", callback=log.callback)
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

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
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
            **metadata
            })
        metrics_list.append(episode_metrics)
        
        print(f"[{identifier}] Episode {episode + 1}: reward = {episode_reward:.2f}, length = {episode_length}")
    
    return metrics_list

def train_model(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles, n_steps, n_noevs=None, execution_context="local", random_seed=None, use_wandb=False, wandb_entity=None):

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

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=None, reward_strategy= reward_strategy, vehicles_to_spawn=n_vehicles, random_seed=None, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'))


    # Train the agent
    algorithm_class = SB3_ALGOS.get(algorithm)
    if algorithm_class is None:
        raise ValueError(f"Invalid model type for training: {algorithm}")
    model = algorithm_class(policy, env, seed=random_seed, verbose=1, tensorboard_log=log.run_dir)
    return _run_training(env=env, log=log, model=model, n_steps=n_steps)

def further_train_model(scenario, algorithm, version_tag, reward_strategy, street_network, n_vehicles, n_steps, model_load_path, n_noevs=None, execution_context="local", use_wandb=False, wandb_entity=None):

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

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=None, reward_strategy= reward_strategy, vehicles_to_spawn=n_vehicles, random_seed=None, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'))

    # Train the agent
    model = _load_model(model_load_path, algorithm, env)
    return _run_training(env=env, log=log, model=model, n_steps=n_steps)

def evaluate_model(scenario, algorithm, version_tag, reward_strategy, street_network, n_vehicles, n_episodes, model_load_path=None, n_noevs=None,execution_context="local", render_mode="human", random_seed=None, use_wandb=False, wandb_entity=None):
    """
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

    # initiate environment
    env = CustomEnv(scenario_generator=scenario, render_mode=render_mode, reward_strategy= reward_strategy, vehicles_to_spawn=n_vehicles, random_seed=random_seed, sumo_log_path=log.py_log_path.replace('.jsonl', '.sumo.log'))

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

def train_and_evaluate(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles, n_steps, n_noevs=None, execution_context="local", random_seed_training=None, random_seed_eval=123, eval_episodes=50):
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
        street_network (str): The street network to use.
        n_vehicles (int): The number of OEVs (observable electric vehicles) in the environment.
        n_steps (int): The number of training steps.
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

        tuple: A tuple containing:
            - model_path (str): The file path of the trained model.
            - eval_metrics_filepath_list (List[str]): A list of the filepaths for the evaluation metrics files of the trained model, the RANDOM and the GREEDY baseline algorithm.
    """

    # train new model
    model_path = train_model(scenario, algorithm, policy, version_tag, reward_strategy, street_network, n_vehicles=n_vehicles, n_steps=n_steps, n_noevs=n_noevs, execution_context=execution_context, random_seed=random_seed_training)
    # evaluate with random and greedy
    model_evaluation_path = evaluate_model(scenario, algorithm, version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=model_path, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval)
    random_evaluation_path = evaluate_model(scenario, "RANDOM", version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=None, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval)
    greedy_evaluation_path = evaluate_model(scenario, "GREEDY", version_tag, reward_strategy, street_network, n_vehicles, n_noevs=n_noevs, n_episodes=eval_episodes, model_load_path=None, execution_context=execution_context, render_mode=None, random_seed=random_seed_eval)
    eval_metrics_filepath_list = [random_evaluation_path, greedy_evaluation_path, model_evaluation_path]
    return model_path, eval_metrics_filepath_list

if __name__ == "__main__":
    
    train_and_evaluate(
        scenario="BASt",
        algorithm="PPO",
        policy="MultiInputPolicy",
        version_tag=get_git_version(),
        reward_strategy="basic",
        street_network="straight100km",
        n_vehicles=500,
        n_noevs=0,
        n_steps=10000,
        eval_episodes=5,
        random_seed_eval=123,
        execution_context="local"
    )