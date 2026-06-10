import ast
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from datetime import datetime
import os
from pathlib import Path
from pprint import pprint


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


def make_violinplot(data_df, metric_name, save_path=None):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    algorithms = data_df["algorithm"].unique()
    palette = sns.color_palette("muted", n_colors=len(algorithms))

    plot = sns.violinplot(x="algorithm", y="Value", data=data_df, palette="muted", cut=0, ax=ax)
    # add in for smaller datasets:
    # sns.stripplot(df_filtered, x="algorithm", y="Value", color=".3", ax=ax)

    # Replace long x-tick labels with numbers; put full names in legend
    ax.set_xticklabels(range(1, len(algorithms) + 1))
    legend_patches = [mpatches.Patch(color=palette[i], label=f"{i + 1}: {alg}") for i, alg in enumerate(algorithms)]
    ax.legend(handles=legend_patches)

    ax.set_title(f"Comparison of {metric_name}", fontsize=14)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Value")
    ax.grid(axis="y", linestyle="-", alpha=0.7)
    plt.tight_layout()

    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)


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

    algorithms = df["algorithm"].unique()

    # Map algorithm names to numbers so the x-axis stays readable
    algo_to_num = {alg: i + 1 for i, alg in enumerate(algorithms)}
    df_agg["algorithm_num"] = df_agg["algorithm"].map(algo_to_num).astype(str)

    order = [str(i + 1) for i in range(len(algorithms))]
    sns.barplot(data=df_agg, x="algorithm_num", y="fraction", hue="action", palette="Set2", order=order, ax=ax)

    # Seaborn auto-creates a legend for hue="action"; keep it, place it upper right
    action_legend = ax.get_legend()
    action_legend.set_title("Action")
    action_legend.set_bbox_to_anchor((1.0, 1.0))

    # Add a second legend mapping numbers to full algorithm names
    algo_patches = [mpatches.Patch(color="lightgray", label=f"{i + 1}: {alg}") for i, alg in enumerate(algorithms)]
    algo_legend = ax.legend(handles=algo_patches, title="Algorithm", loc="upper left")
    ax.add_artist(action_legend)  # restore action legend after it was replaced

    ax.set_title("Action Distribution per Algorithm", fontsize=14)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean fraction of steps")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)


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

    algorithms = df["algorithm"].unique()

    algo_to_num = {alg: i + 1 for i, alg in enumerate(algorithms)}
    df_agg["algorithm_num"] = df_agg["algorithm"].map(algo_to_num).astype(str)

    order = [str(i + 1) for i in range(len(algorithms))]
    sns.barplot(data=df_agg, x="algorithm_num", y="count", hue="action", palette="Set2", order=order, ax=ax)

    action_legend = ax.get_legend()
    action_legend.set_title("Action")
    action_legend.set_bbox_to_anchor((1.0, 1.0))

    algo_patches = [mpatches.Patch(color="lightgray", label=f"{i + 1}: {alg}") for i, alg in enumerate(algorithms)]
    ax.legend(handles=algo_patches, title="Algorithm", loc="upper left")
    ax.add_artist(action_legend)

    # Cap y-axis to suppress extreme outliers; mark clipped bars with ▲
    y_max = df_agg["count"].quantile(0.95) * 1.1
    ax.set_ylim(0, y_max)
    for patch in ax.patches:
        if patch.get_height() > y_max:
            ax.annotate("▲", xy=(patch.get_x() + patch.get_width() / 2, y_max),
                        ha="center", va="bottom", fontsize=9, color="black", clip_on=False)

    ax.set_title("Mean Action Counts per Episode per Algorithm", fontsize=14)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean action count per episode")
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)


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

    algorithms = df_valid["algorithm"].unique()
    x = range(len(algorithms))
    truncated_fracs = [
        agg.loc[(agg["algorithm"] == alg) & (agg["status"] == "truncated"), "fraction"].sum()
        for alg in algorithms
    ]
    terminated_fracs = [1.0 - f for f in truncated_fracs]

    ax.bar(x, terminated_fracs, label="terminated", color=sns.color_palette("muted")[0])
    ax.bar(x, truncated_fracs, bottom=terminated_fracs, label="truncated", color=sns.color_palette("muted")[1])

    ax.set_xticks(list(x))
    ax.set_xticklabels(range(1, len(algorithms) + 1))
    ax.set_ylim(0, 1)
    ax.set_title("Episode Termination Status per Algorithm", fontsize=14)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Fraction of episodes")

    # Status legend (terminated/truncated) + algorithm number mapping
    status_legend = ax.legend(loc="upper right")
    algo_patches = [mpatches.Patch(color="lightgray", label=f"{i + 1}: {alg}") for i, alg in enumerate(algorithms)]
    ax.legend(handles=algo_patches, title="Algorithm", loc="upper left")
    ax.add_artist(status_legend)

    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)


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


def _run_plots(df, metrics_to_plot, save_directory):
    """Build df_melted and render all requested plots. Used by both plot_results and plot_results_from_csv."""
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
        df.to_csv(f"{save_directory}/data_raw.csv", index=False)
        df_melted.to_csv(f"{save_directory}/data_melted.csv", index=False)

    for metric in metrics_to_plot:
        if metric == "action_distribution":
            save_path = f"{save_directory}/action_distribution.png" if save_directory else None
            plot_action_distribution(df, save_path=save_path)
        elif metric == "action_counts_absolute":
            save_path = f"{save_directory}/action_counts_absolute.png" if save_directory else None
            plot_action_counts_absolute(df, save_path=save_path)
        elif metric == "termination_status":
            save_path = f"{save_directory}/termination_status.png" if save_directory else None
            plot_termination_status(df, save_path=save_path)
        elif metric == "arrival_stats":
            save_path = f"{save_directory}/arrival_stats.png" if save_directory else None
            plot_soc_at_arrival(df, save_path=save_path)
        elif metric == "charging_start_stats":
            save_path = f"{save_directory}/charging_start_stats.png" if save_directory else None
            plot_soc_at_charging_start(df, save_path=save_path)
        else:
            df_filtered = df_melted[df_melted["Metric"] == metric]
            metric_display_name = metric[4:] if metric.startswith("env/") else metric
            figure_save_path = f"{save_directory}/{metric_display_name}.png" if save_directory else None
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