"""
Workstream A: Paired statistical tests for thesis empirical claims.

Comparisons:
  Exp 3 (full observability / NOEV=0):
    - 3c vs GREEDY (50 paired episodes)
  Exp 4 (partial observability, all NOEV levels):
    - 3c vs GREEDY at NOEV = 0, 40, 160, 320
    - 3c vs BEST_GUESS at NOEV = 0, 40, 160, 320
      (the NOEV=320 / lowest-rho cell is expected within noise; included to confirm)

Pairing key: episode index 0-49. All methods use seed = 54321 + episode, so
episode i across methods sees the same sampled day and NOEV draw.

Metric: env/ttt_per_ev_mean (seconds per EV). Lower is better.
Sign convention: d_i = 3c_TTT - baseline_TTT. Negative => 3c is better.

Run from repo root:
    python analysis_paired_tests.py

Outputs:
    analysis_results/paired_tests.csv
    analysis_results/paired_tests_latex.tex
"""

import json
import os
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# File paths (all relative to repo root)
# ---------------------------------------------------------------------------

SWEEP = "runs/_exp4_sweep_20260615"

PATHS = {
    "3c": {
        0:   f"{SWEEP}/2026-06-14_16-58-22_pid1992290_v1.4.1-29-g5afbde5_relativeDestinationCongestionIllegal_bast_straight_120km_PPO__sweep_evals/evaluation/metrics2026-06-15_11-21-50.json",
        40:  f"{SWEEP}/2026-06-14_16-58-22_pid1992290_v1.4.1-29-g5afbde5_relativeDestinationCongestionIllegal_bast_straight_120km_PPO__sweep_evals/evaluation/metrics2026-06-15_11-22-10.json",
        160: f"{SWEEP}/2026-06-14_16-58-22_pid1992290_v1.4.1-29-g5afbde5_relativeDestinationCongestionIllegal_bast_straight_120km_PPO__sweep_evals/evaluation/metrics2026-06-15_11-22-30.json",
        320: f"{SWEEP}/2026-06-14_16-58-22_pid1992290_v1.4.1-29-g5afbde5_relativeDestinationCongestionIllegal_bast_straight_120km_PPO__sweep_evals/evaluation/metrics2026-06-15_11-22-50.json",
    },
    "GREEDY": {
        0:   f"{SWEEP}/2026-06-15_02-58-43_pid2513162_v1.4.1-34-g1f65747_relativeDestination_bast_straight_120km_GREEDY/evaluation/metrics2026-06-15_02-58-43.json",
        40:  f"{SWEEP}/2026-06-15_04-08-50_pid2566153_v1.4.1-34-g1f65747_relativeDestination_bast_straight_120km_GREEDY/evaluation/metrics2026-06-15_04-08-50.json",
        160: f"{SWEEP}/2026-06-15_04-59-43_pid2604561_v1.4.1-34-g1f65747_relativeDestination_bast_straight_120km_GREEDY/evaluation/metrics2026-06-15_04-59-43.json",
        320: f"{SWEEP}/2026-06-15_05-12-53_pid2614750_v1.4.1-34-g1f65747_relativeDestination_bast_straight_120km_GREEDY/evaluation/metrics2026-06-15_05-12-53.json",
    },
    "BEST_GUESS": {
        0:   f"{SWEEP}/2026-06-15_05-21-46_pid2621691_v1.4.1-34-g1f65747_relativeDestination_bast_straight_120km_BEST_GUESS/evaluation/metrics2026-06-15_05-21-46.json",
        40:  f"{SWEEP}/2026-06-15_06-23-52_pid2669699_v1.4.1-34-g1f65747_relativeDestination_bast_straight_120km_BEST_GUESS/evaluation/metrics2026-06-15_06-23-52.json",
        160: f"{SWEEP}/2026-06-15_06-25-47_pid2671204_v1.4.1-34-g1f65747_relativeDestination_bast_straight_120km_BEST_GUESS/evaluation/metrics2026-06-15_06-25-47.json",
        320: f"{SWEEP}/2026-06-15_06-54-35_pid2693629_v1.4.1-34-g1f65747_relativeDestination_bast_straight_120km_BEST_GUESS/evaluation/metrics2026-06-15_06-54-35.json",
    },
}

METRIC = "env/ttt_per_ev_mean"
N_BOOT = 10_000
RNG_SEED = 42  # only controls bootstrap sampling, not the eval data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path):
    with open(path) as f:
        data = json.load(f)
    data.sort(key=lambda x: x["episode"])
    return data


def as_series(episodes, key):
    return {ep["episode"]: ep[key] for ep in episodes}


def bootstrap_ci(diffs, n_boot=N_BOOT, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(diffs)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def paired_test(eps_a, eps_b, label_a, label_b, noev, experiment):
    """Run paired analysis; d_i = TTT_a - TTT_b (negative = a is better)."""
    ttt_a = as_series(eps_a, METRIC)
    ttt_b = as_series(eps_b, METRIC)

    common = sorted(set(ttt_a) & set(ttt_b))
    n = len(common)

    # --- sanity checks ---
    assert set(range(50)) == set(common), (
        f"[{label_a} vs {label_b}, NOEV={noev}] Episode indices are not exactly 0-49. "
        f"Got: {sorted(common)}"
    )
    assert n == 50, f"Expected 50 paired episodes, got {n}"

    vals_a = np.array([ttt_a[i] for i in common])
    vals_b = np.array([ttt_b[i] for i in common])
    diffs = vals_a - vals_b

    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = bootstrap_ci(diffs)
    _, p_val = stats.wilcoxon(diffs, alternative="two-sided")

    rho_vals = [ep.get("env/realized_participation_rate", float("nan")) for ep in eps_a]
    rho = float(np.nanmean(rho_vals))

    return {
        "experiment": experiment,
        "comparison": f"{label_a} vs {label_b}",
        "noev": noev,
        "rho": round(rho, 2),
        "n": n,
        "mean_3c_ttt": round(float(vals_a.mean()), 1),
        "mean_baseline_ttt": round(float(vals_b.mean()), 1),
        "mean_diff_s": round(mean_diff, 1),
        "ci_lo": round(ci_lo, 1),
        "ci_hi": round(ci_hi, 1),
        "p_value": float(p_val),
    }


def fmt_p(p):
    if p < 0.001:
        return "$<$0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def fmt_ci(lo, hi):
    return f"[{lo:.1f},\\; {hi:.1f}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    # Load all data upfront
    data = {
        method: {noev: load(path) for noev, path in noev_map.items()}
        for method, noev_map in PATHS.items()
    }

    # --- sanity: episode counts ---
    print("=== Episode count sanity check ===")
    for method, noev_map in data.items():
        for noev, eps in noev_map.items():
            n = len(eps)
            ok = "OK" if n == 50 else "FAIL"
            print(f"  {method:12s} NOEV={noev:3d}: {n} episodes [{ok}]")
    print()

    # --- sanity: verify NOEV field matches filename ---
    print("=== NOEV injection sanity check ===")
    for method, noev_map in data.items():
        for expected_noev, eps in noev_map.items():
            observed = {ep.get("env/noev_sessions_injected") for ep in eps}
            ok = "OK" if observed == {expected_noev} else f"MISMATCH: got {observed}"
            print(f"  {method:12s} NOEV={expected_noev:3d}: [{ok}]")
    print()

    results = []

    # --- Experiment 3: 3c vs GREEDY at full observability (NOEV=0) ---
    r = paired_test(data["3c"][0], data["GREEDY"][0],
                    "3c", "GREEDY", noev=0, experiment="Exp3")
    results.append(r)

    # --- Experiment 4: 3c vs GREEDY at all NOEV levels ---
    for noev in [0, 40, 160, 320]:
        r = paired_test(data["3c"][noev], data["GREEDY"][noev],
                        "3c", "GREEDY", noev=noev, experiment="Exp4")
        results.append(r)

    # --- Experiment 4: 3c vs BEST_GUESS at all NOEV levels ---
    for noev in [0, 40, 160, 320]:
        r = paired_test(data["3c"][noev], data["BEST_GUESS"][noev],
                        "3c", "BEST_GUESS", noev=noev, experiment="Exp4")
        results.append(r)

    # --- print results table ---
    print("=== Paired test results ===")
    header = (
        f"{'Exp':5s} {'Comparison':22s} {'NOEV':>5} {'rho':>5} "
        f"{'n':>3} {'mean_3c':>8} {'mean_bl':>8} {'diff':>8} "
        f"{'CI_lo':>8} {'CI_hi':>8} {'p':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['experiment']:5s} {r['comparison']:22s} {r['noev']:5d} {r['rho']:5.2f} "
            f"{r['n']:3d} {r['mean_3c_ttt']:8.1f} {r['mean_baseline_ttt']:8.1f} "
            f"{r['mean_diff_s']:8.1f} {r['ci_lo']:8.1f} {r['ci_hi']:8.1f} "
            f"{r['p_value']:10.4f}"
        )
    print()

    # --- sign check ---
    print("=== Sign check (flag if diff sign conflicts with aggregate direction) ===")
    for r in results:
        sign = "3c BETTER (lower TTT)" if r["mean_diff_s"] < 0 else "3c WORSE (higher TTT)"
        flag = " *** UNEXPECTED — CHECK THESIS TABLE ***" if r["mean_diff_s"] > 0 else ""
        print(f"  {r['experiment']} {r['comparison']:22s} NOEV={r['noev']:3d}: {sign}{flag}")
    print()

    # --- write CSV ---
    os.makedirs("analysis_results", exist_ok=True)
    csv_path = "analysis_results/paired_tests.csv"
    csv_fields = ["experiment", "comparison", "noev", "rho", "n",
                  "mean_3c_ttt", "mean_baseline_ttt", "mean_diff_s",
                  "ci_lo", "ci_hi", "p_value"]
    with open(csv_path, "w") as f:
        f.write(",".join(csv_fields) + "\n")
        for r in results:
            f.write(",".join(str(r[k]) for k in csv_fields) + "\n")
    print(f"CSV written: {csv_path}")

    # --- write LaTeX ---
    tex_path = "analysis_results/paired_tests_latex.tex"
    with open(tex_path, "w") as f:
        f.write("% Paired statistical test results — drop into thesis\n")
        f.write("% d_i = TTT_{3c} - TTT_{baseline}  (negative => 3c is better)\n")
        f.write("% Bootstrap 95\\% CI (10\\,000 resamples); paired Wilcoxon signed-rank $p$\n")
        f.write("%\n")
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Exp. & Comparison & $\\rho$ & $n$ & "
            "\\multicolumn{2}{c}{Mean TTT/EV (s)} & "
            "$\\Delta$ (s) [95\\% CI] & $p$ \\\\\n"
        )
        f.write(
            "& & & & 3c & Baseline & & \\\\\n"
        )
        f.write("\\midrule\n")
        prev_exp = None
        for r in results:
            exp_label = r["experiment"] if r["experiment"] != prev_exp else ""
            prev_exp = r["experiment"]
            comp = r["comparison"].replace("vs", "vs.")
            ci_str = fmt_ci(r["ci_lo"], r["ci_hi"])
            p_str = fmt_p(r["p_value"])
            f.write(
                f"{exp_label} & {comp} & {r['rho']:.2f} & {r['n']} & "
                f"{r['mean_3c_ttt']:.0f} & {r['mean_baseline_ttt']:.0f} & "
                f"{r['mean_diff_s']:.0f} {ci_str} & {p_str} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"LaTeX written: {tex_path}")

    # --- alignment spot-check: show first 3 paired rows for 3c vs GREEDY NOEV=0 ---
    print("\n=== Spot-check: first 3 paired episodes (3c vs GREEDY, NOEV=0) ===")
    eps_3c = as_series(data["3c"][0], METRIC)
    eps_gr = as_series(data["GREEDY"][0], METRIC)
    print(f"  {'ep':>3}  {'3c_TTT':>10}  {'GREEDY_TTT':>10}  {'diff':>10}")
    for i in range(3):
        print(f"  {i:3d}  {eps_3c[i]:10.1f}  {eps_gr[i]:10.1f}  {eps_3c[i]-eps_gr[i]:10.1f}")


if __name__ == "__main__":
    main()
