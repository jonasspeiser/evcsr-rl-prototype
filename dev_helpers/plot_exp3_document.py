"""
Experiment 3 (Real World Data / BASt, full observability) thesis figures.

Generates the two document figures the Section 5.6 draft still flags as missing:
  - exp3_ttt_per_ev_mean.png  : violin of travel time per vehicle (Travel Times of Exp 3)
  - exp3_action_distribution.png : mean action distribution per method

Uses plotting_utils.plot_for_document so the styling matches the rest of the thesis
figures, with short labels (PPO_<reward> / GREEDY / BEST_GUESS).

Data note: the three dense PPO policies live in the Exp 4 sweep folder, whose metrics
files mix NOEV levels {0,40,160,320} in one file, so we filter to the 0-NOEV arm
(noev_filter=0) to get the full-observability (Experiment 3) episodes. `basic` has no
sweep arm; its canonical 50-episode 0-NOEV eval is taken from the bast-600 batch.
Read-only w.r.t. all runs.
"""
import os
import sys
import glob

import json

import matplotlib
matplotlib.use("Agg")  # headless: save PNGs, never block on plt.show()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from plotting_utils import (  # noqa: E402
    plot_for_document,
    _build_document_label,
    get_all_metrics_files_from_folder,
    build_sweep_cells,
    plot_reliability_vs_congestion,
    DOCUMENT_DISPLAY_LABELS,
    SWEEP_METHOD_ORDER,
)

SWEEP_DIR = "runs/_exp4_sweep_20260615"
BAST600_DIR = "runs/_runs_20260612_1004 (bast 600)"
OUT = "thesis/figures"

# Left-to-right plot order = Table 5.9 ordering restricted to the 6-method subset
# (descending travel time): basic, relativeDestination, BEST_GUESS,
# relativeDestinationCongestion (3b), GREEDY, relativeDestinationCongestionIllegal (3c).
ORDER = [
    "PPO_basic",
    "PPO_relativeDestination",
    "BEST_GUESS",
    "PPO_relativeDestinationCongestion",
    "GREEDY",
    "PPO_relativeDestinationCongestionIllegal",
]


def _is_zero_noev_file(mf):
    """True if every episode in the metrics file is the 0-NOEV (full-obs) arm."""
    eps = json.load(open(mf))
    levels = {e.get("env/noev_sessions_injected") for e in eps}
    return levels == {0}


def _collect_zero_noev_files(sweep_dir):
    """One 0-NOEV metrics file per method from the sweep dir.

    The sweep stores one file per NOEV level inside each run dir (and 3c additionally
    has a duplicate level-0 re-run), so we cannot use get_metrics_files_from_folder
    (which keeps only the latest file per dir = the 320 level). Instead, scan for the
    pure level-0 files and keep the first per document label.
    """
    by_label = {}
    for mf in sorted(glob.glob(os.path.join(sweep_dir, "*", "evaluation", "metrics*.json"))):
        if not _is_zero_noev_file(mf):
            continue
        label = _build_document_label(json.load(open(mf))[0])
        by_label.setdefault(label, mf)  # first (earliest timestamp) wins
    return list(by_label.values())


def main():
    # `basic` only exists in the bast-600 batch; pick its canonical 50-episode eval
    # (the 2026-06-14 file; the two earlier files are 10-episode warm-up evals).
    basic_files = glob.glob(
        os.path.join(BAST600_DIR, "*_basic_*", "evaluation", "metrics2026-06-14*.json")
    )
    if not basic_files:
        raise FileNotFoundError("Could not find the 50-episode `basic` metrics file in "
                                f"{BAST600_DIR}")

    # Dense PPO policies + baselines: exactly one 0-NOEV (full-observability) file each.
    sweep_files = _collect_zero_noev_files(SWEEP_DIR)

    filepath_list = basic_files + sweep_files

    plot_for_document(
        filepath_list,
        metrics_to_plot=["env/ttt_per_ev_mean", "action_distribution"],
        save_directory=OUT,
        order=ORDER,
        noev_filter=0,
        filename_prefix="exp3_",
    )
    # Reliability vs congestion scatter at full observability (NOEV=0), the companion
    # to the Exp 4 NOEV=320 figure. Reuses the shared plotting_utils helper for identical
    # styling; the 0-NOEV arm of the same sweep provides all six methods.
    cells = build_sweep_cells(get_all_metrics_files_from_folder(SWEEP_DIR))
    rel_path = plot_reliability_vs_congestion(
        cells,
        level=0,
        order=SWEEP_METHOD_ORDER,
        display_labels=DOCUMENT_DISPLAY_LABELS,
        save_path=os.path.join(OUT, "exp3_reliability_vs_congestion.png"),
        title="Reliability vs congestion at full observability (NOEV=0)",
    )

    print("saved Exp 3 figures to", OUT)
    print("  ", os.path.join(OUT, "exp3_ttt_per_ev_mean.png"))
    print("  ", os.path.join(OUT, "exp3_action_distribution.png"))
    print("  ", rel_path)


if __name__ == "__main__":
    main()
