import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from pathlib import Path

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
            metrics_files.append(metrics[0])  # assuming one metrics file per run

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
            "env/charging_stops_per_episode_mean",
            "env/global_ttt",
            "env/global_ttt_only_terminated",
            "env/ttt_per_ev_mean",
            "env/ttt_per_ev_mean_only_terminated",
            "env/cumulated_waiting_time",
            "env/cumulated_waiting_time_only_terminated",
            "env/empty_vehicles_per_episode",
            "env/final_simulation_time",
        ]

    # Melt the DataFrame to long format
    df_melted = df.melt(id_vars=["algorithm"], 
                        value_vars=metrics_to_plot, 
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

        # Save dataframe for easier later modifications of visuals
        df.to_csv(f"{save_directory}/data_raw.csv", index=False)
        df_melted.to_csv(f"{save_directory}/data_melted.csv", index=False)


    # Create plots
    for metric in metrics_to_plot:
        # Filter data for the current metric
        df_filtered = df_melted[df_melted["Metric"] == metric]
        # Cut out the leading "/env" in the variable name
        metric_display_name = metric[4:]
        figure_save_path = f"{save_directory}/{metric_display_name}.png" if save_figure else None
        make_violinplot(data_df=df_filtered, metric_name=metric_display_name, save_path=figure_save_path)