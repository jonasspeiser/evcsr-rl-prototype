import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
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

def make_violinplot(data_df, metric_name, save_path=None):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))    # Create plots
    sns.set_theme(style="whitegrid")

    plot = sns.violinplot(x="algorithm", y="Value", data=data_df, palette="muted", cut = 0)
    # add in for smaller datasets:
    # sns.stripplot(df_filtered, x="algorithm", y="Value", color=".3")

    plt.title(f"Comparison of {metric_name}", fontsize=14)
    plt.xlabel("Algorithm")
    plt.ylabel("Value")
    plt.grid(axis="y", linestyle="-", alpha=0.7) 

    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig = plot.get_figure()
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
    plt.figure(figsize=(10, 6))
    plot = sns.barplot(data=df_agg, x="action", y="fraction", hue="algorithm", palette="muted")
    plt.title("Action Distribution per Algorithm", fontsize=14)
    plt.xlabel("Action")
    plt.ylabel("Mean fraction of steps")
    plt.ylim(0, 1)
    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig = plot.get_figure()
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

    algorithms = agg["algorithm"].unique()
    x = range(len(algorithms))
    truncated_fracs = [
        agg.loc[(agg["algorithm"] == alg) & (agg["status"] == "truncated"), "fraction"].sum()
        for alg in algorithms
    ]
    terminated_fracs = [1.0 - f for f in truncated_fracs]

    ax.bar(x, terminated_fracs, label="terminated", color=sns.color_palette("muted")[0])
    ax.bar(x, truncated_fracs, bottom=terminated_fracs, label="truncated", color=sns.color_palette("muted")[1])

    ax.set_xticks(list(x))
    ax.set_xticklabels(algorithms)
    ax.set_ylim(0, 1)
    ax.set_title("Episode Termination Status per Algorithm", fontsize=14)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Fraction of episodes")
    ax.legend()
    sns.despine(left=True, bottom=True)
    plt.show()

    if save_path:
        fig.savefig(save_path)


def plot_results(filepath_list, metrics_to_plot="all", save_figure=False):
    data_list = []
    for filepath in filepath_list:
        with open(filepath) as file:
            data_list.extend(json.load(file))

    data = data_list
    df = pd.DataFrame(data)

    # Select relevant numerical metrics for plotting. If "all" was specified instead of a list, overwrite it with all available metrics.
    if metrics_to_plot == "all":
        metrics_to_plot = [
            "action_distribution",
            "termination_status",
            "reward",
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

    # Melt the DataFrame to long format — special-case metrics are not plain columns, filter them out
    SPECIAL_METRICS = {"action_distribution", "termination_status", "arrival_stats", "charging_start_stats"}
    meltable_metrics = [m for m in metrics_to_plot if m not in SPECIAL_METRICS]
    df_melted = df.melt(id_vars=["algorithm"],
                        value_vars=meltable_metrics,
                        var_name="Metric",
                        value_name="Value")

    if save_figure:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = f"./visuals/{timestamp}"
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the list of source files for reproducibility
        with open(f"{save_directory}/source_files.json", "w") as f:
            # Convert Path objects to strings for JSON serialization
            json.dump({"source_files": [str(fp) for fp in filepath_list]}, f, indent=2)

        # Copy run configs from each evaluation folder for reproducibility
        run_configs = []
        for fp in filepath_list:
            # Config filename mirrors the metrics 
            config_path = Path(fp).parent / Path(fp).name.replace("metrics", "run_config_")
            if config_path.exists():
                with open(config_path) as f:
                    run_configs.append(json.load(f))
        if run_configs:
            with open(f"{save_directory}/run_configs.json", "w") as f:
                json.dump(run_configs, f, indent=2)

        # Save dataframe for easier later modifications of visuals
        df.to_csv(f"{save_directory}/data_raw.csv", index=False)
        df_melted.to_csv(f"{save_directory}/data_melted.csv", index=False)


    # Create plots
    for metric in metrics_to_plot:
        if metric == "action_distribution":
            save_path = f"{save_directory}/action_distribution.png" if save_figure else None
            plot_action_distribution(df, save_path=save_path)
        elif metric == "termination_status":
            save_path = f"{save_directory}/termination_status.png" if save_figure else None
            plot_termination_status(df, save_path=save_path)
        elif metric == "arrival_stats":
            save_path = f"{save_directory}/arrival_stats.png" if save_figure else None
            plot_soc_at_arrival(df, save_path=save_path)
        elif metric == "charging_start_stats":
            save_path = f"{save_directory}/charging_start_stats.png" if save_figure else None
            plot_soc_at_charging_start(df, save_path=save_path)
        else:
            # Filter data for the current metric
            df_filtered = df_melted[df_melted["Metric"] == metric]
            metric_display_name = metric[4:] if metric.startswith("env/") else metric
            figure_save_path = f"{save_directory}/{metric_display_name}.png" if save_figure else None
            make_violinplot(data_df=df_filtered, metric_name=metric_display_name, save_path=figure_save_path)
    
    if run_configs:
        print("\nRun configurations for plotted evaluations:")
        pprint(run_configs)


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