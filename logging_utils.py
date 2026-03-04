"""Script containing logging utilities."""

import os
import logging
import json
import gzip
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Literal
from dataclasses import dataclass
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


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
 
class RingBufferHandler(logging.Handler):
    """Logging handler that keeps only the last N LogRecords in memory (ring buffer)."""
    def __init__(self, capacity: int = 2000, level: int = logging.DEBUG):
        super().__init__(level)
        self.buffer: Deque[logging.LogRecord] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(record)

class LoggerPrefixFilter(logging.Filter):
    """
    Allow only records whose logger name starts with any of the given prefixes.
    Example: prefixes=("rl",) will allow:
        rl
        rl.environment
        rl.environment.simulation
        etc.
    """
    def __init__(self, prefixes):
        super().__init__()
        self.prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name or ""
        return any(name == p or name.startswith(p + ".") for p in self.prefixes)

class CustomTensorboardCallback(BaseCallback):
    """
    Custom sb3 callback for logging rolling mean of environment metrics to Tensorboard.
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

@dataclass
class RunLogging:
    """Orchestrator and Data class to hold logging-related objects for a training or evaluation run."""
    run_dir: str
    model_save_path: str
    callback: BaseCallback | CallbackList  # SB3 callback (BaseCallback or CallbackList)
    writer: Optional[object] = None  # SummaryWriter when evaluating
    wandb_run: Optional[object] = None
    ring: Optional[RingBufferHandler] = None
    py_log_path: Optional[str] = None  # where JsonlFileHandler writes (optional)
    mode: Optional[str] = None         # "training"/"evaluation" (optional)

    def _dump_crash_bundle(self, env_snapshot: dict, exc: BaseException, context: dict | None = None) -> Optional[dict]:
        if self.ring is None:
            return None
        out_dir = os.path.join(self.run_dir, "debug")
        return _dump_crash_bundle_from_ring(
            ring=self.ring,
            out_dir=out_dir,
            env_snapshot=env_snapshot,
            exc=exc,
            context=context,
            compress=False,
        )

    def mark_failed(self, exc: Exception):
        if self.wandb_run is not None:
            self.wandb_run.summary["status"] = "failed"
            self.wandb_run.summary["exception"] = str(exc)

    def mark_interrupted(self, exc: Exception):
        if self.wandb_run is not None:
            self.wandb_run.summary["status"] = "interrupted"
            self.wandb_run.summary["exception"] = str(exc)
    
    def dump_and_log_crash_bundle(self, env_snapshot: dict, exc: BaseException, context: dict | None = None) -> Optional[dict]:
        """
        Dumps crash bundle (ring buffer + snapshot + error) and, if wandb is enabled, logs it as an artifact. Returns bundle paths dict or None.
        """
        bundle = self._dump_crash_bundle(env_snapshot=env_snapshot, exc=exc, context=context)
        if not bundle:
            return None

        if self.wandb_run is not None:
            try:
                # add crash bundle as artifact to wandb
                import wandb
                art = wandb.Artifact(name=f"crash_bundle_{self.wandb_run.id}", type="debug")
                art.add_file(bundle["logs"])
                art.add_file(bundle["snapshot"])
                art.add_file(bundle["error"])
                self.wandb_run.log_artifact(art)

                # additionally add last 50 entries of ring buffer as table to wandb for easy preview (without downloading the artifact)
                rows = [_record_to_dict(rec) for rec in list(self.ring.buffer)[-50:]]
                if rows:
                    table = wandb.Table(
                        data=[list(r.values()) for r in rows],
                        columns=list(rows[0].keys())
                    )
                    self.wandb_run.log({"crash/last_logs_preview": table})

                # additionally add some crash info to wandb summary for easy filtering and overview
                self.wandb_run.summary["crash/exception_type"] = type(exc).__name__
                self.wandb_run.summary["crash/sim_step"] = env_snapshot.get("simulation", {}).get("sim_step")
                self.wandb_run.summary["crash/queue_len"] = env_snapshot.get("charging", {}).get("queue_len")

            except Exception as e:
                # If saving fails, skip it to avoid masking the original exception
                print(f"Error logging crash bundle to WandB: {e}")
            

        return bundle


    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

def _get_log_level(training_or_evaluation):
    match training_or_evaluation:
        case "training":
            return None
        case "evaluation":
            return logging.INFO
        case _:
            return None

def _configure_logging(log_file_path, console_log_level=logging.INFO, ring_capacity: int = 0, file_log_level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handlers = []
    ring = None

    if console_log_level is not None:
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_file_path is not None:
        if log_file_path.endswith(".log"):
            # Replace .log with .jsonl automatically (for compatibility with existing code)
            log_file_path = log_file_path.replace(".log", ".jsonl")
        json_handler = JsonlFileHandler(log_file_path, mode="a")
        json_handler.setLevel(file_log_level)
        # Filter out logs from other modules to reduce noise and make comparisons between runs easier
        # prefixes=("rl",) will allow all loggers starting with "rl", e.g. "rl.environment.simulation"
        json_handler.addFilter(LoggerPrefixFilter(prefixes=("rl",)))
        handlers.append(json_handler)
    
    if ring_capacity and ring_capacity > 0:
        ring = RingBufferHandler(capacity=ring_capacity, level=logging.DEBUG)
        handlers.append(ring)

    # add the handlers to the (root) logger
    logging.basicConfig(level=logging.DEBUG, 
                    handlers=handlers,
                    force=True) # force=True overwrites the logging configuration so that we can change the logfile name
    return ring, log_file_path

def _setup_logging(algorithm, version_tag, reward_strategy, scenario, street_network, n_vehicles, n_noevs, training_or_evaluation, model_load_path=None, execution_context="local"):
    # Create a unique identifier for this training run
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_id = "_".join([current_time, version_tag, reward_strategy, scenario, street_network, algorithm, f"{n_vehicles}OEV", f"{n_noevs}NOEV", training_or_evaluation])
    new_model_id = "_".join([current_time, version_tag, reward_strategy, scenario, street_network, algorithm])
    # if a model path is given, i.e. an existing model is evaluated or trained further
    if model_load_path is not None:
        current_run_dir = model_load_path.split("/")[-2] #.rsplit(".", 1)[0] # Extract directory from model_load_path
    else:
        current_run_dir = new_model_id
    
    # Set up non existing directories
    match execution_context:
        case "local":
            root = "."
        case "colab":
            root = "/content/drive/MyDrive/Colab Notebooks/rl-charging-allocation"
        case _:
            raise ValueError(f"Invalid execution context {execution_context}. Must be 'local' or 'colab'.")

    run_dir = f"{root}/runs/{current_run_dir}"
    log_dir = f"{run_dir}/{training_or_evaluation}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    model_save_path = f"{run_dir}/{new_model_id}.zip"
    py_log_path = f"{log_dir}/{log_id}.jsonl"
    console_log_level = _get_log_level(training_or_evaluation)

    ring, resolved_log_path = _configure_logging(
        log_file_path=py_log_path,
        console_log_level=console_log_level,
        ring_capacity=2000,
        file_log_level=logging.INFO,
    )
    return run_dir, model_save_path, resolved_log_path, ring

def _setup_wandb(wandb_entity, algorithm, version_tag, reward_strategy, street_network, n_vehicles, n_steps, training_or_evaluation, model_load_path=None):
    import wandb

    config = {
        "algorithm": algorithm,
        "version_tag": version_tag,
        "reward_strategy": reward_strategy,
        "street_network": street_network,
        "n_vehicles": n_vehicles,
        "n_steps": n_steps,
        "training_or_evaluation": training_or_evaluation,
        "model_load_path": model_load_path
    }
    run = wandb.init(
        # Set the wandb entity where the project will be logged (e.g. team name).
        entity=wandb_entity,
        # Set the wandb project where this run will be logged.
        project="_".join([version_tag, street_network, algorithm]),
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    return run

def _dump_crash_bundle_from_ring(
    *,
    ring: RingBufferHandler,
    out_dir: str,
    env_snapshot: Dict[str, Any],
    exc: BaseException,
    context: Dict[str, Any] | None = None,
    compress: bool = True,
) -> Dict[str, str]:
    """
    Writes:
      - last_logs.jsonl(.gz)
      - env_snapshot.json
      - error.json
    Returns paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    bundle_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bundle_dir = os.path.join(out_dir, f"crash_{bundle_id}")
    os.makedirs(bundle_dir, exist_ok=True)

    logs_path = os.path.join(bundle_dir, "last_logs.jsonl" + (".gz" if compress else ""))
    if compress:
        with gzip.open(logs_path, "wt", encoding="utf-8") as f:
            for rec in ring.buffer:
                f.write(json.dumps(_record_to_dict(rec)) + "\n")
    else:
        with open(logs_path, "w", encoding="utf-8") as f:
            for rec in ring.buffer:
                f.write(json.dumps(_record_to_dict(rec)) + "\n")

    snapshot_path = os.path.join(bundle_dir, "env_snapshot.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(env_snapshot, f, indent=2, default=str)

    error_path = os.path.join(bundle_dir, "error.json")
    err = {
        "timestamp": bundle_id,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "context": context or {},
    }
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(err, f, indent=2, default=str)

    return {"bundle_dir": bundle_dir, "logs": logs_path, "snapshot": snapshot_path, "error": error_path}

def _record_to_dict(record: logging.LogRecord) -> Dict[str, Any]:
    """Convert LogRecord (including extra fields) into JSON-serializable dict."""
    d: Dict[str, Any] = {
        "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
        "logger": record.name,
        "level": record.levelname,
        "message": record.getMessage(),
        "module": record.module,
        "function": record.funcName,
        "line": record.lineno,
    }

    reserved = {
        "name","msg","args","levelname","levelno","pathname","filename","module","exc_info","exc_text",
        "stack_info","lineno","funcName","created","msecs","relativeCreated","thread","threadName",
        "processName","process"
    }
    for k, v in record.__dict__.items():
        if k in reserved or k in d:
            continue
        # best effort JSON
        try:
            json.dumps(v)
            d[k] = v
        except Exception:
            d[k] = repr(v)

    return d

def setup_run_logging(
    *,
    algorithm: str,
    version_tag: str,
    reward_strategy: str,
    scenario: str,
    street_network: str,
    n_vehicles: int,
    n_noevs: int,
    mode: Literal["training", "evaluation"],
    execution_context: str = "local",
    model_load_path: str | None = None,
    n_steps: int | None = None,
    use_wandb: bool = False,
    wandb_entity: str | None = None,
):
    # path + python logging setup
    run_dir, model_save_path, py_log_path, ring = _setup_logging(
        algorithm=algorithm, version_tag=version_tag, reward_strategy=reward_strategy, scenario=scenario, street_network=street_network, n_vehicles=n_vehicles, n_noevs=n_noevs, training_or_evaluation=mode, model_load_path=model_load_path, execution_context=execution_context
    )

    # sb3 callback (tensorboard + optional wandb logging inside it)
    run = None
    callback = CustomTensorboardCallback()

    # evaluation-specific setup 
    writer = None
    if mode == "evaluation":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{run_dir}/evaluation")
        callback = CustomTensorboardCallback(writer=writer)

    # wandb setup (optional)
    if use_wandb:
        if wandb_entity is None:
            raise ValueError("WandB entity must be provided if use_wandb is True.")

        run = _setup_wandb(
            wandb_entity,
            algorithm, version_tag, reward_strategy, street_network, n_vehicles,
            n_steps, mode, model_load_path
        )

        # internal wandb logging in CustomTensorboardCallback:
        # - training: prefix "train/"
        # - eval: prefix "" (log_evaluation already prefixes eval/)
        if mode == "training":
            callback = CustomTensorboardCallback(wandb_run=run, wandb_prefix="train/")

        # ALTERNATIVE WandbCallback instead of CustomTensorboardCallback's internal logging (commented out since it doesn't support my custom metrics and would require more refactoring):
        # from stable_baselines3.common.callbacks import CallbackList
        # from wandb.integration.sb3 import WandbCallback
        # callback = CallbackList([callback, WandbCallback(model_save_path=f"{run_dir}/wandb_models", verbose=2)])

    return RunLogging(
        run_dir=run_dir,
        model_save_path=model_save_path,
        callback=callback,
        writer=writer,
        wandb_run=run,
        ring=ring,
        py_log_path=py_log_path,
        mode=mode,
    )
