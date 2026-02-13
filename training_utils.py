"""Script containing utility functions for training and evaluating RL models, including logging configuration."""

from circle_environment import CircleEnv
from evaluation_algorithms import RandomAlgorithm, GreedyAlgorithm, NeverChargeAlgorithm
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
import os
import logging
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json

class JsonlFileHandler(logging.Handler):
    """
    Logging handler that writes one JSON object per line (JSONL format).
    """
    def __init__(self, filepath, mode="a"):
        super().__init__()
        self.file = open(filepath, mode, encoding="utf-8")

    def emit(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        self.file.write(json.dumps(log_entry) + "\n")
        self.file.flush()

    def formatTime(self, record):
        return datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

    def close(self):
        self.file.close()
        super().close()

        
def configure_logging(log_file_path, console_log_level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handlers = []
    if console_log_level is not None:
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    if log_file_path is not None:
        # create file handler which logs DEBUG messages to a log file
        if log_file_path.endswith(".log"):
            # Replace .log with .jsonl automatically (for compatibility with existing code)
            log_file_path = log_file_path.replace(".log", ".jsonl")
    
        json_handler = JsonlFileHandler(log_file_path, mode="a")
        json_handler.setLevel(logging.DEBUG)
        handlers.append(json_handler)
    
    # add the handlers to the (root) logger
    logging.basicConfig(level=logging.DEBUG, 
                    handlers=handlers,
                    force=True) # force=True overwrites the logging configuration so that we can change the logfile name

def get_log_level(training_or_evaluation):
    match training_or_evaluation:
        case "training":
            return None
        case "evaluation":
            return logging.INFO
        case _:
            return None

def setup_logging(algorithm, version_tag, reward_strategy, map, n_vehicles, training_or_evaluation, model_load_path=None, execution_context="local"):
    # Create a unique identifier for this training run
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_id = f"{current_time}_{version_tag}_{reward_strategy}_{map}_{algorithm}_{n_vehicles}vehicles_{training_or_evaluation}"    
    new_model_id = f"{current_time}_{version_tag}_{reward_strategy}_{map}_{algorithm}"
    # if a model path is given, i.e. an existing model is evaluated or trained further
    if model_load_path is not None:
        current_model_dir = model_load_path.split("/")[-2] #.rsplit(".", 1)[0] # Extract directory from model_load_path
    else:
        current_model_dir = new_model_id
    
    # Set up non existing directories
    match execution_context:
        case "local":
            root = "."
        case "colab":
            root = "/content/drive/MyDrive/Colab Notebooks/rl-charging-allocation"
        case _:
            raise ValueError(f"Invalid execution context {execution_context}. Must be 'local' or 'colab'.")

    model_dir = f"{root}/models/{current_model_dir}"
    log_dir = f"{model_dir}/{training_or_evaluation}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    model_save_path = f"{model_dir}/{new_model_id}.zip"
    py_log_path = f"{log_dir}/{log_id}.jsonl"
    console_log_level = get_log_level(training_or_evaluation)
    configure_logging(log_file_path=py_log_path, console_log_level=console_log_level)
    return model_dir, model_save_path

def setup_wandb(wandb_entity, wandb_project, algorithm, version_tag, reward_strategy, map, n_vehicles, n_steps, training_or_evaluation, model_load_path=None):
    import wandb

    config = {
        "algorithm": algorithm,
        "version_tag": version_tag,
        "reward_strategy": reward_strategy,
        "map": map,
        "n_vehicles": n_vehicles,
        "n_steps": n_steps,
        "training_or_evaluation": training_or_evaluation,
        "model_load_path": model_load_path
    }
    run = wandb.init(
        # Set the wandb entity where the project will be logged (e.g. team name).
        entity=wandb_entity,
        # Set the wandb project where this run will be logged.
        project=wandb_project,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    return run

def train_model(scenario, algorithm, policy, version_tag, reward_strategy, map, n_vehicles, n_steps, n_nmevs=None, execution_context="local", random_seed=None, use_wandb=False, wandb_entity=None, wandb_project=None):
    # initiate environment
    env = CircleEnv(scenario_generator=scenario, render_mode=None, reward_strategy= reward_strategy, vehicles_to_spawn=n_vehicles, random_seed=None)
    # setup logging
    model_dir, model_save_path = setup_logging(algorithm, version_tag, reward_strategy, map, n_vehicles, "training", execution_context=execution_context)
    callback = CustomTensorboardCallback()
    # Setup WandB tracking (optional)
    if use_wandb:
        if wandb_entity is None or wandb_project is None:
            raise ValueError("WandB entity and project must be provided if use_wandb is True.")
        # from stable_baselines3.common.callbacks import CallbackList
        # from wandb.integration.sb3 import WandbCallback
        run = setup_wandb(wandb_entity, wandb_project, algorithm, version_tag, reward_strategy, map, n_vehicles, n_steps, "training")
        callback = CustomTensorboardCallback(wandb_run=run, wandb_prefix="train/")
        # callback = CallbackList([
        #     callback, 
        #     WandbCallback(
        #         model_save_path=f"{model_dir}/wandb_models",
        #         verbose=2,
        #     )
        # ])

    
    # Train the agent
    match algorithm:
        case "PPO":
            model = PPO(policy, env, seed=random_seed, verbose=1, tensorboard_log=model_dir)
        case "A2C":
            model = A2C(policy, env, seed=random_seed, verbose=1, tensorboard_log=model_dir)
        case "DQN":
            model = DQN(policy, env, seed=random_seed, verbose=1, tensorboard_log=model_dir)
        case _:
            raise ValueError("Invalid model type")
    tb_log_name = "tensorboard"
    try:
        model.learn(n_steps, tb_log_name=tb_log_name, callback=callback)
        model.save(model_save_path)
        return model_save_path
    except Exception as e:
        # Ensure that the model is saved even if an error occurs during training
        try:
            model.save(model_save_path)
        except Exception:
            # If saving fails, skip it to avoid masking the original exception
            pass
        if use_wandb:
            # Log the exception to WandB if it's being used
            run.summary["status"] = "failed"
            run.summary["exception"] = str(e)
        raise e
    finally:
        # Clean up resources
        env.close()
        if use_wandb:
            run.finish()
    
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

def further_train_model(scenario, algorithm, version_tag, reward_strategy, map, n_vehicles, n_steps, model_load_path, n_nmevs=None, execution_context="local", use_wandb=False, wandb_entity=None, wandb_project=None):
    # initiate environment
    env = CircleEnv(scenario_generator=scenario, render_mode=None, reward_strategy= reward_strategy, vehicles_to_spawn=n_vehicles, random_seed=None)
    # setup logging
    model_dir, model_save_path = setup_logging(algorithm, version_tag, reward_strategy, map, n_vehicles, "training", model_load_path, execution_context=execution_context)
    callback = CustomTensorboardCallback()
    # Setup WandB tracking (optional)
    if use_wandb:
        if wandb_entity is None or wandb_project is None:
            raise ValueError("WandB entity and project must be provided if use_wandb is True.")
        from stable_baselines3.common.callbacks import CallbackList
        from wandb.integration.sb3 import WandbCallback
        run = setup_wandb(wandb_entity, wandb_project, algorithm, version_tag, reward_strategy, map, n_vehicles, "training", model_load_path)
        callback = CallbackList([
            callback, 
            WandbCallback(
                model_save_path=f"{model_dir}/wandb_models",
                verbose=2,
            )
        ])

    # Train the agent
    model = load_model(model_load_path, algorithm, env)
    tb_log_name = "tensorboard"
    try:
        model.learn(n_steps, tb_log_name=tb_log_name, callback=callback)
        model.save(model_save_path)
        return model_save_path
    except Exception as e:
        # Ensure that the model is saved even if an error occurs during training
        try:
            model.save(model_save_path)
        except Exception:
            # If saving fails, skip it to avoid masking the original exception
            pass
        if use_wandb:
            # Log the exception to WandB if it's being used
            run.summary["status"] = "failed"
            run.summary["exception"] = str(e)
        raise e    
    finally:
        # Clean up resources
        env.close()
        if use_wandb:
            run.finish()

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
    model_dir, model_save_path = setup_logging(algorithm, version_tag, reward_strategy, map, n_vehicles, "evaluation", model_load_path, execution_context=execution_context)

    # Load saved model or evaluation algorithm
    model = load_model(model_load_path, algorithm, env)
    # Set up TensorBoard writer and custom callback
    writer = SummaryWriter(log_dir=f"{model_dir}/evaluation")
    callback = CustomTensorboardCallback(writer)
    # Set up WandB tracking (optional)
    if use_wandb:
        if wandb_entity is None or wandb_project is None:
            raise ValueError("WandB entity and project must be provided if use_wandb is True.")
        from stable_baselines3.common.callbacks import CallbackList
        from wandb.integration.sb3 import WandbCallback
        run = setup_wandb(wandb_entity, wandb_project, algorithm, version_tag, reward_strategy, map, n_vehicles, "evaluation", model_load_path)
        callback = CallbackList([
            callback, 
            WandbCallback(
                model_save_path=f"{model_dir}/wandb_models",
                verbose=2,            
            )
        ])

    # Evaluate the agent
    try:
        metrics_dict = evaluate_policy(model, env, n_eval_episodes=n_episodes, callback=callback, metadata=metadata, random_seed=random_seed)
        # print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")
        evaluation_path = f"{model_dir}/evaluation/metrics{current_time}.json"
        dump_to_file(metrics_dict, evaluation_path)
        return evaluation_path

    except KeyboardInterrupt as e:
        if use_wandb:
            # Log the exception to WandB if it's being used
            run.summary["status"] = "interrupted"
            run.summary["exception"] = str(e)
    finally:
        env.close()
        writer.close()
        if use_wandb:
            run.finish()

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


class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for logging rolling mean of environment metrics to Tensorboard.
    This version can be used both during training and for independent evaluation.
    
    Attributes:
        rtw_size (int): The size of the rolling time window.
        buffers (Dict[str, Deque[float]]): Buffers to store recent metric values.
        logger_rl (logging.Logger): Logger for debugging RL steps.
    """

    METRIC_NAMES = [
        "charging_stops_per_episode_mean",
        "global_ttt",
        "global_ttt_only_terminated",
        "ttt_per_ev_mean",
        "ttt_per_ev_mean_only_terminated",
        "cumulated_waiting_time",
        "cumulated_waiting_time_only_terminated",
        "empty_vehicles_per_episode",
        "final_simulation_time",
    ]

    def __init__(self, writer=None, verbose=0, rtw_size=100, wandb_run=None, wandb_prefix=""):
        # If used in training with SB3, the BaseCallback machinery will set up the logger.
        super(CustomTensorboardCallback, self).__init__(verbose)
        self.rtw_size = rtw_size
        self.buffers = {
            f"env/{name}": deque(maxlen=rtw_size)
            for name in self.METRIC_NAMES
        }
        self.logger_rl = logging.getLogger("rl")
        self.writer = writer
        self.wandb_run = wandb_run  
        self.wandb_prefix = wandb_prefix

    def get_metrics(self, env):
        metrics = {}
        for name in self.METRIC_NAMES:
            key = f"env/{name}"
            # Use getattr to dynamically fetch the attribute from env.
            value = getattr(env, name, None)
            # Only add the metric if it exists (important e.g. for cumulated_waiting_time_only_terminated)
            if value is not None:
                metrics[key] = value
        return metrics
    
    def _on_step(self):
        """Called at every step during training to update Tensorboard metrics."""
        self.logger_rl.debug("Agent Step {}".format(self.num_timesteps))

        # Retrieve the first environment (suitable for vectorized envs)
        env = self.training_env.envs[0]

        if self.locals['dones'][0]:  # If an episode ended ("done")
            # Gather metrics from the environment
            metrics = self.get_metrics(env)
            wandb_payload = {}

            # Update each metric's buffer, calculate and record each metric's mean over a rolling time window (rtw) 
            for metric_name, value in metrics.items():
                self.buffers[metric_name].append(value)
                rolling_mean = sum(self.buffers[metric_name]) / len(self.buffers[metric_name])

                # SB3 logger (TensorBoard)
                self.logger.record(metric_name, rolling_mean)

                # W&B logger (optional)
                if self.wandb_run is not None:
                    key = f"{self.wandb_prefix}{metric_name}"
                    wandb_payload[key] = rolling_mean

            # on step end, dump all recorded metrics to Tensorboard and log to W&B if applicable
            self.logger.dump(step=self.num_timesteps)
            if self.wandb_run is not None and wandb_payload:
                self.wandb_run.log(wandb_payload, step=self.num_timesteps)

        return True

    def log_evaluation(self, env, step):
        """Log evaluation metrics when running independently (e.g., for a random algorithm)."""
        metrics = self.get_metrics(env)

        # logging to TensorBoard
        if self.writer is not None:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(metric_name, value, step)
            self.writer.flush()

        # logging to W&B (optional)
        if self.wandb_run is not None:
            self.wandb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

        self.logger_rl.debug(f"Logged evaluation metrics at step {step}")
        # For further use of logged values
        return metrics