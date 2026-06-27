"""
Exp 4 headline figures: TTT degradation curve + reliability-vs-congestion scatter.

Thin driver around the reusable plotting helpers in plotting_utils.py. It only supplies
the sweep run directory and output location; the actual aggregation/rendering lives in
plotting_utils (build_sweep_cells, plot_sweep_degradation, plot_reliability_vs_congestion).

Saves PNGs to visuals/. Read-only w.r.t. experiments; matplotlib only.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plotting_utils import (  # noqa: E402
    get_all_metrics_files_from_folder,
    build_sweep_cells,
    plot_sweep_degradation,
    plot_reliability_vs_congestion,
    SWEEP_METHOD_ORDER,
)

SWEEP_DIR = "runs/_exp4_sweep_20260615"
OUT = "visuals"


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = build_sweep_cells(get_all_metrics_files_from_folder(SWEEP_DIR))

    f1 = plot_sweep_degradation(
        cells,
        order=SWEEP_METHOD_ORDER,
        highlight=["PPO_relativeDestinationCongestionIllegal"],
        save_path=os.path.join(OUT, "exp4_ttt_degradation.png"),
    )
    f2 = plot_reliability_vs_congestion(
        cells,
        level=320,
        order=SWEEP_METHOD_ORDER,
        save_path=os.path.join(OUT, "exp4_reliability_vs_congestion.png"),
        title="Reliability vs congestion at heavy partial obs (NOEV=320)",
    )
    print("saved:", f1, "|", f2)


if __name__ == "__main__":
    main()
