"""Script containing logging utilities."""

from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import os
import logging
from collections import deque
import json

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

