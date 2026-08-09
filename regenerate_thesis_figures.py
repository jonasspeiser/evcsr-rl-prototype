"""Regenerate the thesis result figures with publication-quality labels.

Re-renders the violin and diagnostic plots for Experiments 0-3 from the *stored*
evaluation data (each visuals folder's data_raw.csv, and the Exp 3 sweep metrics).
No evaluation is re-run, and no existing figure is overwritten: every experiment
writes into a new dated sibling folder, keeping the original filenames so the
thesis's \\includegraphics paths only need the folder swapped.

Styling comes from plotting_utils (to_display_name + METRIC_AXIS_LABELS):
method names under each violin, no numeric codes, no PIDs, y-axis with units.

Run from the repo root:
    .venv/bin/python regenerate_thesis_figures.py
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save PNGs, never block on plt.show()

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plotting_utils import _run_plots, to_display_name  # noqa: E402

STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Metrics rendered for the CSV-backed experiments. Mirrors the metric set the original
# folders contain, minus the two that need the raw nested action_counts column when it
# is unavailable (handled automatically: the helpers no-op if the column is missing).
CSV_METRICS = [
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
    "env/cwt_per_ev_mean",
    "env/cumulated_waiting_time",
    "env/cumulated_waiting_time_only_terminated",
    "env/empty_vehicles_per_episode",
    "env/final_simulation_time",
    "arrival_stats",
    "charging_start_stats",
]

# Experiment -> the visuals folder the thesis currently includes figures from.
CSV_SOURCES = {
    "Experiment 0 (5 EV, SameSocSameRoute)":
        "visuals/2026-04-01_12-03-02 5EV PPO funktioniert!!!/2026-04-16_15-26-00",
    "Experiment 1 (20 EV, calibration)":
        "visuals/2026-04-12_08-42-32 basicCongestion 20EV calibration (600k steps)/2026-04-27_12-39-31",
    "Experiment 2 (20 EV, AllRandom)":
        "visuals/2026-06-07_10-24-40 20EV DeepSets test (260k steps)/2026-06-10_12-46-13",
}


def regenerate_from_csv(label, old_folder):
    """Re-render one visuals folder into a dated sibling folder."""
    old = Path(old_folder)
    raw_csv = old / "data_raw.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"{raw_csv} not found — cannot regenerate without stored data")

    df = pd.read_csv(raw_csv)
    if "action_counts" in df.columns:
        import ast
        df["action_counts"] = df["action_counts"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    new_folder = old.parent / STAMP
    os.makedirs(new_folder, exist_ok=True)

    methods = list(dict.fromkeys(df["algorithm"]))
    print(f"\n{label}")
    print(f"  source : {raw_csv}")
    print(f"  output : {new_folder}")
    for m in methods:
        print(f"    {m}  ->  {to_display_name(m)}")

    _run_plots(df, CSV_METRICS, str(new_folder))
    n = len(list(new_folder.glob('*.png')))
    print(f"  wrote {n} figures")
    return str(new_folder)


def regenerate_exp3():
    """Re-render the Experiment 3 document figures into a dated folder under thesis/figures."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "plot_exp3_document", "experiments/experiment_helpers/plot_exp3_document.py")
    exp3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp3)

    out = Path("thesis/figures") / STAMP
    os.makedirs(out, exist_ok=True)

    print("\nExperiment 3 (600 EV, bast)")
    print(f"  output : {out}")
    for m in exp3.ORDER:
        print(f"    {m}  ->  {to_display_name(m)}")

    original_out = exp3.OUT
    try:
        exp3.OUT = str(out)  # redirect so the originals in thesis/figures survive
        exp3.main()
    finally:
        exp3.OUT = original_out
    return str(out)


def regenerate_exp4(out_dir):
    """Re-render the Experiment 4 sweep figures with the harmonised method names.

    Same data, order, colours, geometry and titles as before; only the legend/point
    labels change, from the ad-hoc "PPO (congestion+illegal)" / "GREEDY" naming to the
    document convention ("PPO (3c)" / "Greedy") used by every other figure. The labels
    now come from plotting_utils.to_display_name, so they cannot drift again.
    """
    from plotting_utils import (get_all_metrics_files_from_folder, build_sweep_cells,
                                plot_sweep_degradation, plot_reliability_vs_congestion,
                                SWEEP_METHOD_ORDER)

    sweep_dir = "runs/_exp4_sweep_20260615"
    os.makedirs(out_dir, exist_ok=True)

    print("\nExperiment 4 (NOEV sweep)")
    print(f"  source : {sweep_dir}")
    print(f"  output : {out_dir}")
    for m in SWEEP_METHOD_ORDER:
        print(f"    {m}  ->  {to_display_name(m)}")

    cells = build_sweep_cells(get_all_metrics_files_from_folder(sweep_dir))
    f1 = plot_sweep_degradation(
        cells,
        order=SWEEP_METHOD_ORDER,
        highlight=["PPO_relativeDestinationCongestionIllegal"],
        save_path=os.path.join(out_dir, "exp4_ttt_degradation.png"),
    )
    f2 = plot_reliability_vs_congestion(
        cells,
        level=320,
        order=SWEEP_METHOD_ORDER,
        save_path=os.path.join(out_dir, "exp4_reliability_vs_congestion.png"),
        title="Reliability vs congestion at heavy partial obs (NOEV=320)",
    )
    print(f"  wrote {os.path.basename(f1)}, {os.path.basename(f2)}")
    return out_dir


if __name__ == "__main__":
    # `python regenerate_thesis_figures.py <exp3|exp4|thesis> <existing-stamp>` re-renders
    # only those figures into an existing output folder, so all thesis/figures artwork
    # from one regeneration pass stays together under a single dated folder.
    if len(sys.argv) > 1 and sys.argv[1] in ("exp3", "exp4", "thesis"):
        which = sys.argv[1]
        stamp = sys.argv[2] if len(sys.argv) > 2 else STAMP
        target = os.path.join("thesis/figures", stamp)
        if which in ("exp3", "thesis"):
            STAMP = stamp  # regenerate_exp3 derives its output folder from STAMP
            regenerate_exp3()
        if which in ("exp4", "thesis"):
            regenerate_exp4(target)
        raise SystemExit(0)

    written = {}
    for label, folder in CSV_SOURCES.items():
        written[label] = regenerate_from_csv(label, folder)
    written["Experiment 3 (600 EV, bast)"] = regenerate_exp3()
    written["Experiment 4 (NOEV sweep)"] = regenerate_exp4(
        os.path.join("thesis/figures", STAMP))

    print("\n" + "=" * 72)
    print("New figure folders (original filenames preserved):")
    for label, path in written.items():
        print(f"  {label}\n      {path}")
