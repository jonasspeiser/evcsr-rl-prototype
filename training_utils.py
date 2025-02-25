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
        # create file handler which logs even debug messages
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
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

def setup_logging(algorithm, version, env_version, map, n_vehicles, training_or_evaluation, model_load_path=None, execution_context="local"):
    # Create a unique identifier for this training run
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_id = f"{current_time}_{version}_{env_version}_{map}_{algorithm}_{n_vehicles}vehicles_{training_or_evaluation}"    
    new_model_id = f"{current_time}_{version}_{env_version}_{map}_{algorithm}"
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
    py_log_path = f"{log_dir}/{log_id}.log"
    console_log_level = get_log_level(training_or_evaluation)
    configure_logging(log_file_path=py_log_path, console_log_level=console_log_level)
    return model_dir, model_save_path

def train_model(algorithm, policy, version, env_version, map, n_vehicles, n_steps, execution_context="local"):
    # initiate environment
    env = CircleEnv(render_mode=None, env_version= env_version, vehicles_to_spawn=n_vehicles)
    # setup logging
    model_dir, model_save_path = setup_logging(algorithm, version, env_version, map, n_vehicles, "training", execution_context=execution_context)
    # Train the agent
    match algorithm:
        case "PPO":
            model = PPO(policy, env, verbose=1, tensorboard_log=model_dir)
        case "A2C":
            model = A2C(policy, env, verbose=1, tensorboard_log=model_dir)
        case "DQN":
            model = DQN(policy, env, verbose=1, tensorboard_log=model_dir)
        case _:
            raise ValueError("Invalid model type")
    tb_log_name = "tensorboard"
    try:
        model.learn(n_steps, tb_log_name=tb_log_name, callback=CustomTensorboardCallback())
        model.save(model_save_path)
        env.close()
        return model_save_path
    except Exception as e:
        model.save(model_save_path)
        env.close()
        raise e
    
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

def further_train_model(algorithm, version, env_version, map, n_vehicles, n_steps, model_load_path, execution_context="local"):
    # initiate environment
    env = CircleEnv(render_mode=None, env_version= env_version, vehicles_to_spawn=n_vehicles)
    # setup logging
    model_dir, model_save_path = setup_logging(algorithm, version, env_version, map, n_vehicles, "training", model_load_path, execution_context=execution_context)
    # Train the agent
    model = load_model(model_load_path, algorithm, env)
    tb_log_name = "tensorboard"
    try:
        model.learn(n_steps, tb_log_name=tb_log_name, callback=CustomTensorboardCallback())
        model.save(model_save_path)
        env.close()
        return model_save_path
    except Exception as e:
        model.save(model_save_path)
        env.close()
        raise e    

def evaluate_model(algorithm, version, env_version, map, n_vehicles, n_episodes, model_load_path=None, execution_context="local", render_mode="human"):
    # initiate environment
    env = CircleEnv(render_mode=render_mode, env_version= env_version, vehicles_to_spawn=n_vehicles)
    # setup logging
    model_dir, model_save_path = setup_logging(algorithm, version, env_version, map, n_vehicles, "evaluation", model_load_path, execution_context=execution_context)
    # Load saved model or evaluation algorithm
    model = load_model(model_load_path, algorithm, env)
    # Set up TensorBoard writer and custom callback
    writer = SummaryWriter(log_dir=model_dir)
    custom_callback = CustomTensorboardCallback(writer)
    # Evaluate the agent
    try:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes, callback=custom_callback, identifier=None)
        print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")
        env.close()
        writer.close()
    except KeyboardInterrupt:
        env.close()
        writer.close()

def evaluate_policy(model, env, n_eval_episodes, callback, identifier):
    ep_rewards = []
    ep_lengths = []
    total_step = 0

    for episode in range(n_eval_episodes):
        observation, info = env.reset()
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
        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_length)
        # Retrieve metrics from the environment after an episode ends.
        callback.log_evaluation(env, total_step)
        
        print(f"[{identifier}] Episode {episode + 1}: reward = {episode_reward:.2f}, length = {episode_length}")
    
    avg_reward = np.mean(ep_rewards)
    avg_length = np.mean(ep_lengths)
    return avg_reward, avg_length


class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for logging rolling mean of environment metrics to Tensorboard.
    This version can be used both during training and for independent evaluation.
    
    Attributes:
        rtw_size (int): The size of the rolling time window.
        buffers (Dict[str, Deque[float]]): Buffers to store recent metric values.
        logger_rl (logging.Logger): Logger for debugging RL steps.
    """
    def __init__(self, writer=None, verbose=0, rtw_size=100):
        # If used in training with SB3, the BaseCallback machinery will set up the logger.
        super(CustomTensorboardCallback, self).__init__(verbose)
        self.rtw_size = rtw_size
        self.buffers = {
            "env/charging_stops_per_episode_mean": deque(maxlen=rtw_size),
            "env/global_ttt": deque(maxlen=rtw_size),
            "env/cumulated_waiting_time": deque(maxlen=rtw_size),
            "env/cumulated_waiting_time_only_terminated": deque(maxlen=rtw_size),
            "env/empty_vehicles_per_episode": deque(maxlen=rtw_size),
        }
        self.logger_rl = logging.getLogger("rl")
        # If no writer is passed (e.g., when running independently), create one.
        if writer is None:
            self.writer = SummaryWriter(log_dir="./tensorboard_logs/random_policy")
        else:
            self.writer = writer

    def get_metrics(self, env):
        metrics = {
            "env/charging_stops_per_episode_mean": env.charging_stops_per_episode_mean,
            "env/global_ttt": env.global_ttt,
            "env/cumulated_waiting_time": env.cumulated_waiting_time,
            "env/empty_vehicles_per_episode": env.empty_vehicles_per_episode,
        }
        if hasattr(env, "cumulated_waiting_time_only_terminated"):
            metrics["env/cumulated_waiting_time_only_terminated"] = env.cumulated_waiting_time_only_terminated
        return metrics
    
    def _on_step(self):
        """Called at every step during training to update Tensorboard metrics."""
        self.logger_rl.debug("Agent Step {}".format(self.num_timesteps))

        # Retrieve the first environment (suitable for vectorized envs)
        env = self.training_env.envs[0]

        if self.locals['dones'][0]:  # If an episode ended ("done")
            # Gather metrics from the environment
            metrics = self.get_metrics(env)

            # Update each metric's buffer, calculate and record each metric's mean over a rolling time window (rtw) 
            for metric_name, value in metrics.items():
                self.buffers[metric_name].append(value)
                rolling_mean = sum(self.buffers[metric_name]) / len(self.buffers[metric_name])
                self.logger.record(metric_name, rolling_mean)

            self.logger.dump(step=self.num_timesteps)

        return True

    def log_evaluation(self, env, step):
        """Log evaluation metrics when running independently (e.g., for a random algorithm)."""
        metrics = self.get_metrics(env)
        for metric_name, value in metrics.items():
            self.buffers[metric_name].append(value)
            rolling_mean = sum(self.buffers[metric_name]) / len(self.buffers[metric_name])
            self.writer.add_scalar(metric_name, rolling_mean, step)
        self.writer.flush()
        self.logger_rl.debug(f"Logged evaluation metrics at step {step}")