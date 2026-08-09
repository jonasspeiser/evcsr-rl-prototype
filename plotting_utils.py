import ast
import re
import statistics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from datetime import datetime
import os
from pathlib import Path
from pprint import pprint


# ---------------------------------------------------------------------------
# Publication styling: display names, axis labels, tick handling
# ---------------------------------------------------------------------------
# Internal run labels (built by _build_display_label / _build_document_label) carry
# obs-feature tags, extractor flags and PIDs so parallel runs stay distinguishable.
# None of that belongs in a thesis figure, so every rendered label goes through
# to_display_name() first. "3b"/"3c" are the short names the document defines for
# relativeDestinationCongestion / relativeDestinationCongestionIllegal from Exp 3 on;
# the formulations without a short name keep their full name in parentheses.

REWARD_DISPLAY_NAMES = {
    "basic": "basic",
    "basicCongestion": "basicCongestion",
    "relativeDestination": "relativeDestination",
    "relativeDestinationCongestion": "3b",
    "relativeDestinationCongestionIllegal": "3c",
}

BASELINE_DISPLAY_NAMES = {
    "GREEDY": "Greedy",
    "BEST_GUESS": "Best Guess",
    "RANDOM": "Random",
    "PERFECT": "Perfect",
}

# Y-axis labels: human-readable metric name plus unit, keyed by the metric name with
# any "env/" prefix stripped (the form _run_plots passes to the plot helpers).
METRIC_AXIS_LABELS = {
    "ttt_per_ev_mean": "Mean travel time per vehicle (s)",
    "ttt_per_ev_mean_only_terminated": "Mean travel time per vehicle, completed trips (s)",
    "global_ttt": "Total travel time (s)",
    "global_ttt_only_terminated": "Total travel time, completed trips (s)",
    "cwt_per_ev_mean": "Mean waiting time per vehicle (s)",
    "cumulated_waiting_time": "Cumulated waiting time (s)",
    "cumulated_waiting_time_only_terminated": "Cumulated waiting time, completed trips (s)",
    "charging_stops_per_episode_mean": "Charging stops per vehicle",
    "empty_vehicles_per_episode": "Stranded vehicles per episode",
    "final_simulation_time": "Episode end time (s)",
    "episode_length": "Episode length (agent steps)",
    "reward": "Episode return",
    "noev_sessions_injected": "Injected NOEV charging sessions",
    "realized_participation_rate": "Realised participation rate",
}

# Figures render at roughly full text width in the thesis (10 in wide artwork scaled to
# ~6.3 in, i.e. ~0.63x), so on-canvas sizes must be ~1.6x the desired print size.
# 15 pt / 16 pt here land at ~9.5 pt / ~10 pt on the printed page.
TICK_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 16

_METHOD_TOKEN_RE = re.compile(r"^(PPO|A2C|DQN)_([A-Za-z0-9]+)")
_PID_RE = re.compile(r"_pid\d+")


def to_display_name(internal_name: str) -> str:
    """Map an internal run label to the thesis display name.

    Examples:
        PPO_basic_pid927830                                        -> "PPO (basic)"
        PPO_basicCongestion-simulation_time-..._pid1035816         -> "PPO (basicCongestion)"
        PPO_relativeDestination-simulation_time_pid1760            -> "PPO (relativeDestination)"
        PPO_relativeDestinationCongestion                          -> "PPO (3b)"
        PPO_relativeDestinationCongestionIllegal-..._cext_pid1681  -> "PPO (3c)"
        GREEDY / BEST_GUESS / PERFECT20 / RANDOM                   -> "Greedy" / "Best Guess" / "Perfect" / "Random"

    A PID never survives this function: unrecognised labels still get their
    ``_pid<digits>`` suffix stripped rather than leaking a process id into a figure.
    """
    name = str(internal_name)
    if name in BASELINE_DISPLAY_NAMES:
        return BASELINE_DISPLAY_NAMES[name]
    # PERFECT5 / PERFECT20 / PERFECT<N> are all "Perfect" in the document
    if name.startswith("PERFECT"):
        return "Perfect"
    match = _METHOD_TOKEN_RE.match(name)
    if match:
        algo, reward = match.group(1), match.group(2)
        return f"{algo} ({REWARD_DISPLAY_NAMES.get(reward, reward)})"
    return _PID_RE.sub("", name)


def _ordered_methods(series):
    """Unique method labels in order of first appearance (the plot's left-to-right order)."""
    return list(dict.fromkeys(series))


def _method_palette(order):
    """Position-keyed 'muted' colours, so a method keeps its colour across figures."""
    colors = sns.color_palette("muted", n_colors=len(order))
    return {label: colors[i] for i, label in enumerate(order)}


def _apply_method_xaxis(ax, order, fig=None):
    """Put display names under each category, rotating them only if they would collide."""
    labels = [to_display_name(label) for label in order]
    ax.set_xticks(range(len(labels)))

    fig = fig or ax.get_figure()
    fig_width_pt = fig.get_size_inches()[0] * 72
    slot_pt = fig_width_pt / max(len(labels), 1)
    # ~0.58 em average glyph advance for DejaVu Sans
    widest_pt = max((len(l) for l in labels), default=0) * TICK_FONTSIZE * 0.58

    if widest_pt > slot_pt * 0.95:
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=TICK_FONTSIZE)
    else:
        ax.set_xticklabels(labels, fontsize=TICK_FONTSIZE)
    ax.set_xlabel("")  # method names are self-explanatory; no "Algorithm" label needed
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def get_last_x_metrics_files(n_files,runs_dir="runs"):
    runs_path = Path(runs_dir)

    # Get only directories
    run_dirs = [p for p in runs_path.iterdir() if p.is_dir()]
    if len(run_dirs) < n_files:
        raise ValueError(f"Less than {n_files} runs available")

    # Sort by folder name (works because timestamp is at beginning)
    run_dirs.sort()

    # Take last n_files
    last_n = run_dirs[-n_files:]

    metrics_files = []
    for run in last_n:
        eval_dir = run / "evaluation"
        metrics = list(eval_dir.glob("metrics*.json"))
        if metrics:
            metrics_files.append(sorted(metrics)[-1])  # take newest (timestamp in filename)

    return metrics_files

def get_metrics_files_from_folder(folder_path):
    """Return all metrics files from every run directory inside folder_path.

    Args:
        folder_path (str or Path): Path to a folder containing run directories
            (e.g. "runs/_runs_20260610_1759").

    Returns:
        List of Path objects, one per run that has an evaluation/metrics*.json file.
    """
    folder = Path(folder_path)
    metrics_files = []
    for run_dir in sorted(folder.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_dir = run_dir / "evaluation"
        metrics = list(eval_dir.glob("metrics*.json")) if eval_dir.exists() else []
        if metrics:
            metrics_files.append(sorted(metrics)[-1])
    return metrics_files


def get_all_metrics_files_from_folder(folder_path):
    """Return *every* metrics file under folder_path (recursively), not just the latest per run.

    Companion to :func:`get_metrics_files_from_folder`. Required for NOEV participation
    sweeps, where a single run directory stores one metrics file per participation level
    (so keeping only the latest would silently drop the other levels).

    Args:
        folder_path (str or Path): folder containing run directories.

    Returns:
        Sorted list of Path objects for all ``metrics*.json`` files found below it.
    """
    return sorted(Path(folder_path).glob("**/metrics*.json"))


def make_violinplot(data_df, metric_name, save_path=None, y_label=None, title=None):
    """Violin plot of one metric per method, styled for the thesis.

    Method display names sit directly under each violin (no numeric codes, no legend)
    and the y-axis carries the metric name plus unit. Colours are assigned by position
    from the 'muted' palette, so a given figure keeps the colours it had before.

    Args:
        data_df: long-format frame with an "algorithm" column and a "Value" column.
        metric_name: metric key (e.g. "ttt_per_ev_mean") used to look up the y-label
            in METRIC_AXIS_LABELS, or an already human-readable label.
        y_label: explicit y-axis label; overrides the METRIC_AXIS_LABELS lookup.
        title: optional title. Left off by default — the thesis supplies captions.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    order = _ordered_methods(data_df["algorithm"])
    palette = _method_palette(order)

    sns.violinplot(x="algorithm", y="Value", data=data_df, order=order,
                   hue="algorithm", hue_order=order, palette=palette, legend=False,
                   cut=0, ax=ax)
    # add in for smaller datasets:
    # sns.stripplot(df_filtered, x="algorithm", y="Value", color=".3", ax=ax)

    _apply_method_xaxis(ax, order, fig)
    ax.set_ylabel(y_label or METRIC_AXIS_LABELS.get(metric_name, metric_name),
                  fontsize=AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE + 1)
    ax.grid(axis="y", linestyle="-", alpha=0.7)
    plt.tight_layout()

    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_action_distribution(df, save_path=None):
    """Plot the mean action distribution per algorithm as a grouped bar chart.

    Normalises each episode's action_counts to fractions first so that
    episodes of different lengths are weighted equally.
    """
    if "action_counts" not in df.columns:
        return

    rows = []
    for _, row in df.iterrows():
        counts = row.get("action_counts")
        if not isinstance(counts, dict) or not counts:
            continue
        total = sum(counts.values())
        for action, count in counts.items():
            rows.append({
                "algorithm": row["algorithm"],
                "action": int(action),
                "fraction": count / total if total > 0 else 0,
            })

    if not rows:
        return

    df_actions = pd.DataFrame(rows)
    df_agg = df_actions.groupby(["algorithm", "action"])["fraction"].mean().reset_index()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    order = _ordered_methods(df["algorithm"])

    sns.barplot(data=df_agg, x="algorithm", y="fraction", hue="action", palette="Set2",
                order=order, ax=ax)

    # Seaborn auto-creates a legend for hue="action"; that is the only legend needed —
    # method names now sit under the bars instead of in a number-mapping legend.
    action_legend = ax.get_legend()
    action_legend.set_title("Action")
    action_legend.set_bbox_to_anchor((1.0, 1.0))

    _apply_method_xaxis(ax, order, fig)
    ax.set_ylabel("Mean fraction of steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_action_counts_absolute(df, save_path=None):
    """Plot the total absolute action counts per algorithm as a grouped bar chart."""
    if "action_counts" not in df.columns:
        return

    rows = []
    for _, row in df.iterrows():
        counts = row.get("action_counts")
        if not isinstance(counts, dict) or not counts:
            continue
        for action, count in counts.items():
            rows.append({
                "algorithm": row["algorithm"],
                "action": int(action),
                "count": count,
            })

    if not rows:
        return

    df_actions = pd.DataFrame(rows)
    df_agg = df_actions.groupby(["algorithm", "action"])["count"].mean().reset_index()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    order = _ordered_methods(df["algorithm"])

    sns.barplot(data=df_agg, x="algorithm", y="count", hue="action", palette="Set2",
                order=order, ax=ax)

    action_legend = ax.get_legend()
    action_legend.set_title("Action")
    action_legend.set_bbox_to_anchor((1.0, 1.0))

    # Cap y-axis to suppress extreme outliers; mark clipped bars with ▲
    y_max = df_agg["count"].quantile(0.95) * 1.1
    ax.set_ylim(0, y_max)
    for patch in ax.patches:
        if patch.get_height() > y_max:
            ax.annotate("▲", xy=(patch.get_x() + patch.get_width() / 2, y_max),
                        ha="center", va="bottom", fontsize=9, color="black", clip_on=False)

    _apply_method_xaxis(ax, order, fig)
    ax.set_ylabel("Mean action count per episode", fontsize=AXIS_LABEL_FONTSIZE)
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_termination_status(df, save_path=None):
    """Plot the proportion of truncated vs terminated episodes per algorithm as a stacked bar chart."""
    if "was_truncated" not in df.columns:
        return

    df_valid = df[df["was_truncated"].notna()].copy()
    if df_valid.empty:
        return

    agg = (
        df_valid.groupby("algorithm")["was_truncated"]
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
    )
    agg["status"] = agg["was_truncated"].map({True: "truncated", False: "terminated"})

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    order = _ordered_methods(df_valid["algorithm"])
    x = range(len(order))
    truncated_fracs = [
        agg.loc[(agg["algorithm"] == alg) & (agg["status"] == "truncated"), "fraction"].sum()
        for alg in order
    ]
    terminated_fracs = [1.0 - f for f in truncated_fracs]

    ax.bar(x, terminated_fracs, label="terminated", color=sns.color_palette("muted")[0])
    ax.bar(x, truncated_fracs, bottom=terminated_fracs, label="truncated", color=sns.color_palette("muted")[1])

    ax.set_ylim(0, 1)
    _apply_method_xaxis(ax, order, fig)
    ax.set_ylabel("Fraction of episodes", fontsize=AXIS_LABEL_FONTSIZE)

    # Only the terminated/truncated legend is needed; method names are on the x-axis.
    ax.legend(loc="upper right")

    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def _build_display_label(row) -> str:
    """Derive a human-readable plot label from a metrics row.

    For baseline algorithms (GREEDY, RANDOM, etc.) the algorithm name alone is
    sufficient. For trained models (PPO/A2C/DQN) the reward strategy, obs features,
    and PID (extracted from the run directory name) are appended so parallel runs
    with identical configs are distinguishable.
    """
    algo = row.get("algorithm", "?")
    trained_algos = {"PPO", "A2C", "DQN"}
    if algo not in trained_algos:
        return algo
    reward = row.get("reward_strategy", "")
    obs = row.get("obs_features")
    obs_tag = "-" + "-".join(sorted(obs)) if obs else ""
    extractor_tag = "_cext" if row.get("use_custom_extractor") else ""
    # Extract pid from run directory name (e.g. "2026-03-31_16-13-08_pid927830_...")
    run_dir = row.get("_run_dir", "")
    pid_match = re.search(r"pid(\d+)", run_dir)
    pid_tag = f"_pid{pid_match.group(1)}" if pid_match else ""
    return f"{algo}_{reward}{obs_tag}{extractor_tag}{pid_tag}"


def _build_document_label(row) -> str:
    """Derive a short, thesis-ready plot label from a metrics row.

    Unlike :func:`_build_display_label` (which appends obs features, extractor flag
    and PID to disambiguate parallel single-run evaluations), this keeps only the
    information that is meaningful in the written document: the reward strategy for
    trained models, and the bare name for rule-based baselines.

    Examples:
        PPO          + relativeDestinationCongestionIllegal -> "PPO_relativeDestinationCongestionIllegal"
        GREEDY / BEST_GUESS / RANDOM / PERFECT              -> unchanged
    """
    algo = row.get("algorithm", "?")
    trained_algos = {"PPO", "A2C", "DQN"}
    if algo not in trained_algos:
        return algo
    reward = row.get("reward_strategy", "")
    return f"{algo}_{reward}" if reward else algo


def get_algorithm_labels(filepath_list):
    """Return the display labels that plot_results would assign to each run.

    Call this first to find the exact label strings to pass to algorithms_to_include.

    Example output:
        ['GREEDY', 'PERFECT', 'PPO_shaping-soc_pid927830', 'PPO_shaping-soc_pid931042']
    """
    data_list = []
    for filepath in filepath_list:
        with open(filepath) as file:
            rows = json.load(file)
        run_dir = Path(filepath).parent.parent.name
        for row in rows:
            row["_run_dir"] = run_dir
        data_list.extend(rows)

    df = pd.DataFrame(data_list)
    df["algorithm"] = df.apply(_build_display_label, axis=1)
    return sorted(df["algorithm"].unique().tolist())


def plot_results(filepath_list, metrics_to_plot="all", save_figure=False, algorithms_to_include=None):
    """
    Plot metrics from evaluation runs.
    
    Args:
        filepath_list (list): list of paths to metrics JSON files from evaluation runs 
            (e.g.["runs/2026-03-05_22-28-05_v0.9.5_basic_same_route_straight100km_PPO/evaluation/metrics2026-03-06_12-18-04.json", ...] )
        metrics_to_plot (list or "all"): list of metric names to plot (or "all" for all available metrics). Available metrics:
            [
                "action_distribution",
                "action_counts_absolute",
                "termination_status",
                "reward",
                "episode_length",
                "env/charging_stops_per_episode_mean",
                "env/global_ttt",
                "env/global_ttt_only_terminated",
                "env/ttt_per_ev_mean",
                "env/ttt_per_ev_mean_only_terminated",
                "env/cumulated_waiting_time",
                "env/cumulated_waiting_time_only_terminated",
                "env/cwt_per_ev_mean",
                "env/empty_vehicles_per_episode",
                "env/final_simulation_time",
                "arrival_stats",
                "charging_start_stats",
            ]

        save_figure (bool): whether to save the generated figures (in ./visuals with timestamp). Defaults to False.
    """
    data_list = []
    for filepath in filepath_list:
        with open(filepath) as file:
            rows = json.load(file)
        run_dir = Path(filepath).parent.parent.name  # e.g. "2026-03-31_16-13-08_pid927830_..._PPO"
        for row in rows:
            row["_run_dir"] = run_dir
        data_list.extend(rows)

    df = pd.DataFrame(data_list)
    df["algorithm"] = df.apply(_build_display_label, axis=1)

    if algorithms_to_include is not None:
        df = df[df["algorithm"].isin(algorithms_to_include)]

    # Select relevant numerical metrics for plotting. If "all" was specified instead of a list, overwrite it with all available metrics.
    if metrics_to_plot == "all":
        metrics_to_plot = [
            "action_distribution",
            "action_counts_absolute",
            "termination_status",
            "reward",
            "episode_length",
            "env/charging_stops_per_episode_mean",
            "env/global_ttt",
            "env/global_ttt_only_terminated",
            "env/ttt_per_ev_mean",
            "env/ttt_per_ev_mean_only_terminated",
            "env/cumulated_waiting_time",
            "env/cumulated_waiting_time_only_terminated",
            "env/cwt_per_ev_mean",
            "env/empty_vehicles_per_episode",
            "env/final_simulation_time",
            "arrival_stats",
            "charging_start_stats",
        ]

    if save_figure:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = f"./visuals/{timestamp}"
        os.makedirs(save_directory, exist_ok=True)

        # Save the list of source files for reproducibility
        with open(f"{save_directory}/source_files.json", "w") as f:
            json.dump({"source_files": [str(fp) for fp in filepath_list]}, f, indent=2)

        # Copy run configs from each evaluation folder for reproducibility
        run_configs = []
        for fp in filepath_list:
            config_path = Path(fp).parent / Path(fp).name.replace("metrics", "run_config_")
            if config_path.exists():
                with open(config_path) as f:
                    run_configs.append(json.load(f))
        if run_configs:
            with open(f"{save_directory}/run_configs.json", "w") as f:
                json.dump(run_configs, f, indent=2)
    else:
        save_directory = None
        run_configs = []

    _run_plots(df, metrics_to_plot, save_directory)

    if run_configs:
        print("\nRun configurations for plotted evaluations:")
        pprint(run_configs)


def plot_for_document(filepath_list, metrics_to_plot, save_directory=None,
                      algorithms_to_include=None, order=None, noev_filter=None,
                      filename_prefix=""):
    """Render thesis-ready figures with short, document-friendly algorithm labels.

    This is the document-facing counterpart to :func:`plot_results`. It is identical
    in rendering but uses :func:`_build_document_label` (e.g.
    ``PPO_relativeDestinationCongestionIllegal`` / ``GREEDY``) instead of the verbose
    single-run labels, and adds two conveniences needed for the experiment write-ups:
    a participation/NOEV filter and explicit method ordering.

    Args:
        filepath_list (list): paths to evaluation ``metrics*.json`` files.
        metrics_to_plot (list): metric names, same vocabulary as :func:`plot_results`
            (e.g. ``["env/ttt_per_ev_mean", "action_distribution"]``).
        save_directory (str or None): directory to write PNGs into (created if missing).
            Unlike :func:`plot_results`, no timestamped subfolder is added, so figures
            land exactly where requested (e.g. ``thesis/figures``). If None, figures are
            only shown, not saved.
        algorithms_to_include (list or None): keep only these document labels.
        order (list or None): document labels in the desired left-to-right plot order.
            Acts as an inclusion filter as well, so passing ``order`` alone is enough.
        noev_filter (int or None): if set, keep only episodes whose
            ``env/noev_sessions_injected`` equals this value. Use ``0`` to extract the
            full-observability arm from a mixed-level NOEV sweep file.
        filename_prefix (str): prefix for saved PNG filenames (e.g. ``"exp3_"``).
    """
    data_list = []
    for filepath in filepath_list:
        with open(filepath) as file:
            rows = json.load(file)
        run_dir = Path(filepath).parent.parent.name
        for row in rows:
            row["_run_dir"] = run_dir
        data_list.extend(rows)

    df = pd.DataFrame(data_list)

    if noev_filter is not None:
        if "env/noev_sessions_injected" not in df.columns:
            raise ValueError("noev_filter requested but 'env/noev_sessions_injected' is "
                             "absent from the metrics — these runs predate NOEV logging.")
        df = df[df["env/noev_sessions_injected"] == noev_filter]

    df["algorithm"] = df.apply(_build_document_label, axis=1)

    if algorithms_to_include is not None:
        df = df[df["algorithm"].isin(algorithms_to_include)]

    if order is not None:
        rank = {name: i for i, name in enumerate(order)}
        df = df[df["algorithm"].isin(rank)]
        df = df.sort_values(by="algorithm", key=lambda s: s.map(rank), kind="stable")

    if df.empty:
        raise ValueError("No episodes left to plot after filtering. Check "
                         "algorithms_to_include / order / noev_filter against the inputs.")

    if save_directory:
        os.makedirs(save_directory, exist_ok=True)
        with open(f"{save_directory}/{filename_prefix}source_files.json", "w") as f:
            json.dump({"source_files": [str(fp) for fp in filepath_list]}, f, indent=2)

    _run_plots(df, metrics_to_plot, save_directory, filename_prefix=filename_prefix)


def plot_results_from_csv(csv_path, metrics_to_plot="all", save_figure=False, algorithms_to_include=None):
    """Re-generate plots from a previously saved data_raw.csv without re-running evaluation.

    Args:
        csv_path (str): Path to a data_raw.csv saved by plot_results.
        metrics_to_plot (list or "all"): same as plot_results.
        save_figure (bool): whether to save figures alongside the source CSV.
        algorithms_to_include (list or None): if given, only plot these algorithm display labels.
            Call get_algorithm_labels() first to see what's available.
    """
    df = pd.read_csv(csv_path)

    # action_counts is stored as a string in CSV — parse it back to a dict
    if "action_counts" in df.columns:
        df["action_counts"] = df["action_counts"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    if algorithms_to_include is not None:
        df = df[df["algorithm"].isin(algorithms_to_include)]

    if metrics_to_plot == "all":
        metrics_to_plot = [
            "action_distribution",
            "action_counts_absolute",
            "termination_status",
            "reward",
            "episode_length",
            "env/charging_stops_per_episode_mean",
            "env/global_ttt",
            "env/global_ttt_only_terminated",
            "env/ttt_per_ev_mean",
            "env/ttt_per_ev_mean_only_terminated",
            "env/cumulated_waiting_time",
            "env/cumulated_waiting_time_only_terminated",
            "env/empty_vehicles_per_episode",
            "env/final_simulation_time",
            "arrival_stats",
            "charging_start_stats",
        ]

    save_directory = None
    if save_figure:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = str(Path(csv_path).parent / timestamp)
        os.makedirs(save_directory, exist_ok=True)

    _run_plots(df, metrics_to_plot, save_directory)


def _run_plots(df, metrics_to_plot, save_directory, filename_prefix=""):
    """Build df_melted and render all requested plots. Used by both plot_results and plot_results_from_csv.

    Args:
        filename_prefix (str): optional prefix prepended to every saved PNG filename
            (e.g. "exp3_") so figures from different experiments can share a directory.
    """
    # Melt the DataFrame to long format.
    # Special-case metrics are not plain scalar columns; map them to their backing columns so
    # they still end up in data_melted.csv and can be reproduced without re-running evaluation.
    # action_distribution uses nested dicts (action_counts) — not meltable, stays in data_raw.csv only.
    SPECIAL_METRIC_COLUMNS = {
        "termination_status": ["was_truncated"],
        "arrival_stats": ["env/arrival_soc_wh_mean", "env/arrival_range_m_mean"],
        "charging_start_stats": ["env/charging_start_soc_wh_mean", "env/charging_start_range_m_mean"],
    }
    meltable_metrics = []
    for m in metrics_to_plot:
        if m in SPECIAL_METRIC_COLUMNS:
            meltable_metrics.extend(SPECIAL_METRIC_COLUMNS[m])
        elif m not in ("action_distribution", "action_counts_absolute"):
            meltable_metrics.append(m)
    meltable_metrics = [m for m in meltable_metrics if m in df.columns]
    df_melted = df.melt(id_vars=["algorithm"],
                        value_vars=meltable_metrics,
                        var_name="Metric",
                        value_name="Value")

    if save_directory:
        df.to_csv(f"{save_directory}/{filename_prefix}data_raw.csv", index=False)
        df_melted.to_csv(f"{save_directory}/{filename_prefix}data_melted.csv", index=False)

    for metric in metrics_to_plot:
        if metric == "action_distribution":
            save_path = f"{save_directory}/{filename_prefix}action_distribution.png" if save_directory else None
            plot_action_distribution(df, save_path=save_path)
        elif metric == "action_counts_absolute":
            save_path = f"{save_directory}/{filename_prefix}action_counts_absolute.png" if save_directory else None
            plot_action_counts_absolute(df, save_path=save_path)
        elif metric == "termination_status":
            save_path = f"{save_directory}/{filename_prefix}termination_status.png" if save_directory else None
            plot_termination_status(df, save_path=save_path)
        elif metric == "arrival_stats":
            save_path = f"{save_directory}/{filename_prefix}arrival_stats.png" if save_directory else None
            plot_soc_at_arrival(df, save_path=save_path)
        elif metric == "charging_start_stats":
            save_path = f"{save_directory}/{filename_prefix}charging_start_stats.png" if save_directory else None
            plot_soc_at_charging_start(df, save_path=save_path)
        else:
            df_filtered = df_melted[df_melted["Metric"] == metric]
            metric_display_name = metric[4:] if metric.startswith("env/") else metric
            figure_save_path = f"{save_directory}/{filename_prefix}{metric_display_name}.png" if save_directory else None
            make_violinplot(data_df=df_filtered, metric_name=metric_display_name, save_path=figure_save_path)


def generate_latex_tables(csv_path, metrics=None, algorithms_to_include=None, save_path=None):
    """Generate LaTeX summary tables (mean ± std per algorithm) from a saved data_raw.csv.

    Args:
        csv_path (str): Path to a data_raw.csv saved by plot_results.
        metrics (list or None): scalar metric column names to include. Defaults to all numeric
            columns except internal ones. Example: ["reward", "env/global_ttt", "env/ttt_per_ev_mean"]
        algorithms_to_include (list or None): filter to these display labels (same as plot_results).
        save_path (str or None): if given, write the .tex file here; otherwise print to stdout.

    Returns:
        str: the LaTeX table source.
    """
    df = pd.read_csv(csv_path)

    if algorithms_to_include is not None:
        df = df[df["algorithm"].isin(algorithms_to_include)]

    # Default: all numeric columns that are actual metrics
    exclude = {"_run_dir", "was_truncated"}
    if metrics is None:
        metrics = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    else:
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            print(f"Note: skipping metrics not found as columns in CSV: {missing}")
        metrics = [m for m in metrics if m in df.columns]

    # Compute mean and std per algorithm for each metric
    agg = df.groupby("algorithm")[metrics].agg(["mean", "std"])

    # Flatten multi-level columns and format as "mean ± std"
    rows = {}
    for metric in metrics:
        col_mean = agg[(metric, "mean")]
        col_std = agg[(metric, "std")]
        rows[metric] = col_mean.map("{:.2f}".format) + " $\\pm$ " + col_std.map("{:.2f}".format)

    result_df = pd.DataFrame(rows)
    result_df.index.name = "Algorithm"

    # Shorten metric names for column headers (strip "env/" prefix)
    result_df.columns = [c[4:] if c.startswith("env/") else c for c in result_df.columns]

    result_df = result_df.T  # metrics as rows, algorithms as columns
    result_df.index.name = "Metric"

    # Escape underscores in index and column names so LaTeX doesn't treat them as subscripts
    result_df.index = result_df.index.str.replace("_", r"\_", regex=False)
    result_df.columns = result_df.columns.str.replace("_", r"\_", regex=False)

    n_algorithms = len(result_df.columns)
    latex = result_df.to_latex(
        caption="Evaluation results (mean $\\pm$ std across episodes).",
        label="tab:eval_results",
        escape=False,
        column_format="l" + "r" * n_algorithms,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex)
    else:
        print(latex)

    return latex


def plot_soc_at_arrival(df, save_path=None):
    """Violin plots of mean SOC (Wh) and remaining range (km) at vehicle arrival."""
    base = Path(save_path) if save_path else None

    col_wh = "env/arrival_soc_wh_mean"
    if col_wh in df.columns and not df[col_wh].dropna().empty:
        plot_df = df[["algorithm", col_wh]].dropna().rename(columns={col_wh: "Value"})
        sp = str(base.with_name(base.stem + "_soc_wh" + base.suffix)) if base else None
        make_violinplot(plot_df, metric_name="SOC at arrival (Wh)", save_path=sp)

    col_m = "env/arrival_range_m_mean"
    if col_m in df.columns and not df[col_m].dropna().empty:
        plot_df = df[["algorithm", col_m]].dropna().copy()
        plot_df[col_m] = plot_df[col_m] / 1000
        plot_df = plot_df.rename(columns={col_m: "Value"})
        sp = str(base.with_name(base.stem + "_range_km" + base.suffix)) if base else None
        make_violinplot(plot_df, metric_name="Remaining range at arrival (km)", save_path=sp)


def plot_soc_at_charging_start(df, save_path=None):
    """Violin plots of mean SOC (Wh) and remaining range (km) at charging stop begin."""
    base = Path(save_path) if save_path else None

    col_wh = "env/charging_start_soc_wh_mean"
    if col_wh in df.columns and not df[col_wh].dropna().empty:
        plot_df = df[["algorithm", col_wh]].dropna().rename(columns={col_wh: "Value"})
        sp = str(base.with_name(base.stem + "_soc_wh" + base.suffix)) if base else None
        make_violinplot(plot_df, metric_name="SOC at charging start (Wh)", save_path=sp)

    col_m = "env/charging_start_range_m_mean"
    if col_m in df.columns and not df[col_m].dropna().empty:
        plot_df = df[["algorithm", col_m]].dropna().copy()
        plot_df[col_m] = plot_df[col_m] / 1000
        plot_df = plot_df.rename(columns={col_m: "Value"})
        sp = str(base.with_name(base.stem + "_range_km" + base.suffix)) if base else None
        make_violinplot(plot_df, metric_name="Remaining range at charging start (km)", save_path=sp)


def plot_soc_history(soc_history: dict[str, list[float]], max_capacity_wh: float = 22_390, title: str = "SOC per vehicle over episode"):
    """
    Plot the battery SOC (Wh) for each vehicle across all simulation steps of one episode.

    Args:
        soc_history: dict returned by simulation.get_soc_history()
        max_capacity_wh: effective maximum battery capacity in Wh (used as reference line)
        title: plot title
    """
    _, ax = plt.subplots(figsize=(14, 4))
    for vid, history in soc_history.items():
        ax.plot(history, label=vid, alpha=0.7)
    ax.axhline(max_capacity_wh, color="black", linestyle="--", linewidth=1, label=f"max capacity ({max_capacity_wh:,.0f} Wh)")
    ax.axhline(30, color="red", linestyle="--", linewidth=1, label="empty threshold (30 Wh)")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("SOC (Wh)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# NOEV participation-sweep figures (degradation curves + reliability scatter)
# ---------------------------------------------------------------------------
# These aggregate evaluation episodes by (method, NOEV level) and render the two
# partial-observability figures used in the Experiment 3/4 write-ups. They are kept
# here so every plotting helper lives in one place; the thin drivers in dev_helpers/
# only supply run-directory paths and output locations.

DEFAULT_NOEV_LEVELS = (0, 40, 160, 320)

# Default left-to-right / legend order for the sweep figures. Labels follow the
# document convention produced by _build_document_label (PPO_<reward> / GREEDY / ...),
# matching the violin, action-distribution and table naming used elsewhere in the chapter.
SWEEP_METHOD_ORDER = [
    "PPO_relativeDestinationCongestionIllegal",
    "PPO_relativeDestinationCongestion",
    "PPO_relativeDestination",
    "GREEDY",
    "BEST_GUESS",
    "RANDOM",
]


def _nearest_level(value, levels):
    return min(levels, key=lambda n: abs(n - value))


def _cell_mean(eps, key):
    vals = [e[key] for e in eps if key in e and e[key] is not None]
    return statistics.mean(vals) if vals else None


def _cell_std(eps, key):
    vals = [e[key] for e in eps if key in e and e[key] is not None]
    return statistics.pstdev(vals) if vals else None


def build_sweep_cells(filepath_list, levels=DEFAULT_NOEV_LEVELS, label_fn=_build_document_label):
    """Aggregate sweep evaluation files into ``{(label, noev_level): [episodes]}``.

    Each metrics file is assigned to the nearest NOEV level by the mean of its
    per-episode ``env/noev_sessions_injected`` values, so single-level and mixed-level
    files are handled alike. Files that map to the same (label, level) are concatenated.

    Args:
        filepath_list: metrics files, e.g. from :func:`get_all_metrics_files_from_folder`.
        levels: candidate NOEV levels episodes are snapped to.
        label_fn: maps a metrics row to its label (default: short document label).
    """
    cells = {}
    for mf in filepath_list:
        with open(mf) as f:
            eps = json.load(f)
        if not isinstance(eps, list) or not eps:
            continue
        label = label_fn(eps[0])
        injected = [e["env/noev_sessions_injected"] for e in eps if "env/noev_sessions_injected" in e]
        level = _nearest_level(statistics.mean(injected), levels) if injected else levels[0]
        cells.setdefault((label, level), []).extend(eps)
    return cells


def plot_sweep_degradation(cells, levels=DEFAULT_NOEV_LEVELS, metric="env/ttt_per_ev_mean",
                           order=None, display_labels=None, highlight=None, save_path=None,
                           title="Travel-time degradation under partial observability",
                           xlabel="Injected NOEV charging sessions  (lower participation →)",
                           ylabel="Mean travel time per vehicle (s)"):
    """Line plot of ``metric`` vs NOEV level, one line (mean ± std) per method.

    Args:
        cells: output of :func:`build_sweep_cells`.
        levels: NOEV levels on the x-axis.
        metric: per-episode metric key aggregated as mean ± std across episodes.
        order: method labels to draw, in order (defaults to all present, sorted).
        display_labels: optional ``{label: pretty_name}`` for the legend.
        highlight: labels drawn with a thicker line.
        save_path: if given, the figure is saved here at 150 dpi.
    """
    order = order or sorted({lab for (lab, _) in cells})
    display_labels = display_labels or {}
    highlight = set(highlight or [])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in order:
        xs, ys, es = [], [], []
        for n in levels:
            eps = cells.get((label, n))
            if not eps:
                continue
            m = _cell_mean(eps, metric)
            if m is None:
                continue
            xs.append(n); ys.append(m); es.append(_cell_std(eps, metric) or 0.0)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                    label=display_labels.get(label) or to_display_name(label),
                    linewidth=2 if label in highlight else 1.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(levels))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_reliability_vs_congestion(cells, level, order=None, display_labels=None,
                                   x_metric="env/empty_vehicles_per_episode",
                                   y_metric="env/cwt_per_ev_mean", save_path=None, title=None,
                                   xlabel="Stranded vehicles / episode  (reliability → better left)",
                                   # Wrapped onto two lines: on one line this rotated label needs
                                   # ~84% of the canvas height and gets clipped at the top.
                                   ylabel="Charging wait time / vehicle (s)\n(congestion → better down)",
                                   xlim=None, ylim=None):
    """Scatter of reliability (x: stranded/ep) vs congestion (y: wait/ev) at one NOEV level.

    Lower-left is better on both axes. Each method is a coloured point identified via a
    legend, so labels follow the document naming convention used elsewhere in the chapter
    rather than being annotated inline (which overlaps for clustered methods).

    If ``xlim`` and/or ``ylim`` are given, the axes zoom to that window and any method
    falling outside it is drawn as a triangle clamped to the corresponding edge and
    annotated with its true (x, y) value — so off-scale outliers stay visible with their
    numbers without compressing the remaining methods.

    Args:
        cells: output of :func:`build_sweep_cells`.
        level: NOEV level to plot.
        order: method labels to include, in order (defaults to all present at ``level``).
        display_labels: optional ``{label: pretty_name}`` for the legend (default: identity).
        x_metric / y_metric: per-episode metrics averaged for each method.
        save_path: if given, the figure is saved here at 150 dpi (clip-safe).
        title: defaults to ``"Reliability vs congestion (NOEV=<level>)"``.
        xlim / ylim: optional ``(min, max)`` zoom windows enabling outlier clamping.
    """
    order = order or sorted({lab for (lab, n) in cells if n == level})
    display_labels = display_labels or {}
    if title is None:
        title = f"Reliability vs congestion (NOEV={level})"

    points = []
    for label in order:
        eps = cells.get((label, level))
        if not eps:
            continue
        x = _cell_mean(eps, x_metric)
        y = _cell_mean(eps, y_metric)
        if x is not None and y is not None:
            points.append((label, x, y))

    palette = sns.color_palette("muted", n_colors=len(points))
    colors = {lab: palette[i] for i, (lab, _, _) in enumerate(points)}

    xmin, xmax = xlim if xlim else (None, None)
    ymin, ymax = ylim if ylim else (None, None)

    mid_x = (xmin + xmax) / 2 if (xmin is not None and xmax is not None) else None

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, x, y in points:
        c = colors[label]
        name = display_labels.get(label) or to_display_name(label)
        above = ymax is not None and y > ymax
        below = ymin is not None and y < ymin
        right = xmax is not None and x > xmax
        left = xmin is not None and x < xmin
        if not (above or below or right or left):
            # In-window point: label sits next to it, on whichever side has more room.
            ax.scatter(x, y, s=70, color=c, zorder=3)
            if mid_x is not None and x > mid_x:
                ax.annotate(name, (x, y), fontsize=8, xytext=(-7, 0),
                            textcoords="offset points", ha="right", va="center", zorder=4)
            else:
                ax.annotate(name, (x, y), fontsize=8, xytext=(7, 0),
                            textcoords="offset points", ha="left", va="center", zorder=4)
            continue
        # Off-scale: clamp to the edge, point a triangle outward, label with the true value.
        cx = min(max(x, xmin if xmin is not None else x), xmax if xmax is not None else x)
        cy = min(max(y, ymin if ymin is not None else y), ymax if ymax is not None else y)
        text = f"{name}\n({x:.0f}, {y:.0f})"
        right_half = mid_x is not None and cx > mid_x
        if above:
            marker = "^"
            off, ha, va = ((-4, -13), "right", "top") if right_half else ((4, -13), "left", "top")
        elif below:
            marker = "v"
            off, ha, va = ((-4, 13), "right", "bottom") if right_half else ((4, 13), "left", "bottom")
        elif right:
            marker, off, ha, va = ">", (-11, 0), "right", "center"
        else:
            marker, off, ha, va = "<", (11, 0), "left", "center"
        ax.scatter(cx, cy, s=95, color=c, marker=marker, zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(text, (cx, cy), fontsize=8, xytext=off,
                    textcoords="offset points", ha=ha, va=va, zorder=4)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.show()
    if save_path:
        # No bbox_inches="tight" here: the figure already uses constrained_layout, and
        # the two layout passes disagree — the tight crop cut into the (long) y-axis
        # label. constrained_layout alone fits every artist inside the canvas.
        fig.savefig(save_path, dpi=150)
    return save_path