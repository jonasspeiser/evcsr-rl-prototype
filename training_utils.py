"""Script containing utility functions for training and evaluating RL models."""

from circle_environment import CircleEnv
from evaluation_algorithms import RandomAlgorithm, GreedyAlgorithm, NeverChargeAlgorithm
from stable_baselines3 import PPO, A2C, DQN
from datetime import datetime
import json
from logging_utils import setup_run_logging

def load_model(model_path, algorithm, env):
    match algorithm:
        case "PPO":
            model = PPO.load(model_path, env=env)
        case "A2C":
            model = A2C.load(model_path, env=env)
        case "DQN":
            model = DQN.load(model_path, env=env)
        case "RANDOM":
            model = RandomAlgorithm(environment=env)
        case "GREEDY":
            model = GreedyAlgorithm(environment=env)
        case "NOCHARGE":
            model = NeverChargeAlgorithm(environment=env)
        case _:
            raise ValueError("Invalid model type")
    return model

def dump_to_file(content, filepath):
    with open(filepath, 'w') as f:
        json.dump(content, f, indent=2)

def evaluate_policy(model, env, n_eval_episodes, callback, metadata, random_seed):
    metrics_list = []
    total_step = 0

    identifier = None #TODO: generate identifier from metadata

    for episode in range(n_eval_episodes):
        observation, info = env.reset(seed=random_seed)
        done = False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_step += 1
            episode_reward += reward
            episode_length += 1
            if terminated or truncated:
                done = True
        # Retrieve metrics from the environment after an episode ends.
        episode_metrics_dict = callback.log_evaluation(env, total_step)
        episode_metrics_dict.update({
            "episode": episode,
            "episode_length": episode_length,
            "was_truncated": truncated,
            "reward": episode_reward,
            })
        episode_metrics_dict.update(metadata)
        metrics_list.append(episode_metrics_dict)
        
        print(f"[{identifier}] Episode {episode + 1}: reward = {episode_reward:.2f}, length = {episode_length}")
    
    return metrics_list

def train_model(scenario, algorithm, policy, version_tag, reward_strategy, map, n_vehicles, n_steps, n_nmevs=None, execution_context="local", random_seed=None, use_wandb=False, wandb_entity=None, wandb_project=None):
    # initiate environment
    env = CircleEnv(scenario_generator=scenario, render_mode=None, reward_strategy= reward_strategy, vehicles_to_spawn=n_vehicles, random_seed=None)

    # setup logging
    log = setup_run_logging(
        algorithm=algorithm,
        version_tag=version_tag,
        reward_strategy=reward_strategy,
        map=map,
        n_vehicles=n_vehicles,
        n_steps=n_steps,
        mode="training",
        execution_context=execution_context,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )
    
    # Train the agent
    match algorithm:
        case "PPO":
            model = PPO(policy, env, seed=random_seed, verbose=1, tensorboard_log=log.model_dir)
        case "A2C":
            model = A2C(policy, env, seed=random_seed, verbose=1, tensorboard_log=log.model_dir)
        case "DQN":
            model = DQN(policy, env, seed=random_seed, verbose=1, tensorboard_log=log.model_dir)
        case _:
            raise ValueError("Invalid model type")
    tb_log_name = "tensorboard"
    try:
        model.learn(n_steps, tb_log_name=tb_log_name, callback=log.callback)
        model.save(log.model_save_path)
        return log.model_save_path
    except Exception as e:
        # Ensure that the model is saved even if an error occurs during training
        try:
            model.save(log.model_save_path)
        except Exception:
            # If saving fails, skip it to avoid masking the original exception
            pass
        log.mark_failed(e)
        raise
    finally:
        # Clean up resources
        env.close()
        log.close()

def further_train_model(scenario, algorithm, version_tag, reward_strategy, map, n_vehicles, n_steps, model_load_path, n_nmevs=None, execution_context="local", use_wandb=False, wandb_entity=None, wandb_project=None):
    # initiate environment
    env = CircleEnv(scenario_generator=scenario, render_mode=None, reward_strategy= reward_strategy, vehicles_to_spawn=n_vehicles, random_seed=None)

    # setup logging
    log = setup_run_logging(
        algorithm=algorithm,
        version_tag=version_tag,
        reward_strategy=reward_strategy,
        map=map,
        n_vehicles=n_vehicles,
        n_steps=n_steps,
        mode="training",
        execution_context=execution_context,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )

    # Train the agent
    model = load_model(model_load_path, algorithm, env)
    tb_log_name = "tensorboard"
    try:
        model.learn(n_steps, tb_log_name=tb_log_name, callback=log.callback)
        model.save(log.model_save_path)
        return log.model_save_path
    except Exception as e:
        # Ensure that the model is saved even if an error occurs during training
        try:
            model.save(log.model_save_path)
        except Exception:
            # If saving fails, skip it to avoid masking the original exception
            pass
        log.mark_failed(e)
        raise
    finally:
        # Clean up resources
        env.close()
        log.close()

def evaluate_model(scenario, algorithm, version_tag, reward_strategy, map, n_vehicles, n_episodes, model_load_path=None, n_nmevs=None,execution_context="local", render_mode="human", random_seed=None, use_wandb=False, wandb_entity=None, wandb_project=None):
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
        "map": map,
        "n_vehicles": n_vehicles,
        "n_episodes": n_episodes,
        "random_seed": random_seed
    }
    # initiate environment
    env = CircleEnv(scenario_generator=scenario, render_mode=render_mode, reward_strategy= reward_strategy, vehicles_to_spawn=n_vehicles, random_seed=random_seed)
    # setup logging
    log = setup_run_logging(
        algorithm=algorithm,
        version_tag=version_tag,
        reward_strategy=reward_strategy,
        map=map,
        n_vehicles=n_vehicles,
        n_steps=None,
        mode="evaluation",
        model_load_path=model_load_path,
        execution_context=execution_context,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )

    # Load saved model or evaluation algorithm
    model = load_model(model_load_path, algorithm, env)

    # Evaluate the agent
    try:
        metrics_dict = evaluate_policy(model, env, n_eval_episodes=n_episodes, callback=log.callback, metadata=metadata, random_seed=random_seed)
        # print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")
        evaluation_path = f"{log.model_dir}/evaluation/metrics{current_time}.json"
        dump_to_file(metrics_dict, evaluation_path)
        return evaluation_path

    except KeyboardInterrupt as e:
        log.mark_interrupted()
        raise

    finally:
        env.close()
        log.close()