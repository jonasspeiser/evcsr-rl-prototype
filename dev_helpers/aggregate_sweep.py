"""
Aggregate the Exp 4 NOEV sweep into the degradation tables (RQ2 / RQ2.1 / RQ2.2).

Reads every eval metrics file produced since the sweep marker, groups by
(algorithm / reward_strategy) x (nominal NOEV count), and reports mean TTT, empty/ep,
CWT, and realized participation rate. Read-only; safe to run anytime.

Usage: python dev_helpers/aggregate_sweep.py [marker_path]   (default /tmp/exp4sweep_marker)
"""
import os, sys, glob, json, statistics as st

SWEEP_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/_exp4_sweep_20260615"
NOEV_LEVELS = [0, 40, 160, 320]

def _nearest_level(x):
    return min(NOEV_LEVELS, key=lambda n: abs(n - x))

def collect():
    """Read metadata directly from each metrics file's CONTENT (robust to run-dir/path
    matching quirks). Group by (label, NOEV level), NOEV inferred from injected sessions.
    Reads the consolidated sweep folder only, so the PPO metrics (copied there from the
    model training dirs) are not double-counted with their originals elsewhere under runs/."""
    rows = {}  # (label, n_noevs) -> metrics list
    seen = set()
    for mf in glob.glob(os.path.join(SWEEP_DIR, "**", "metrics*.json"), recursive=True):
        if mf in seen:
            continue
        seen.add(mf)
        try: eps = json.load(open(mf))
        except Exception: continue
        if not isinstance(eps, list) or not eps: continue
        meta = eps[0]
        algo = meta.get("algorithm")
        if algo is None: continue
        label = meta.get("reward_strategy") if algo == "PPO" else algo
        inj = [e["env/noev_sessions_injected"] for e in eps if "env/noev_sessions_injected" in e]
        n_noevs = _nearest_level(st.mean(inj)) if inj else 0
        rows.setdefault((label, n_noevs), []).extend(eps)
    return rows

def agg(eps, key):
    v = [e[key] for e in eps if key in e and e[key] is not None]
    return st.mean(v) if v else float("nan")

def main():
    rows = collect()
    labels = sorted({lab for lab, _ in rows})
    for metric, key, fmt in [
        ("TTT/ev (s)", "env/ttt_per_ev_mean", "{:7.0f}"),
        ("empty/ep",   "env/empty_vehicles_per_episode", "{:7.1f}"),
        ("CWT/ev (s)", "env/cwt_per_ev_mean", "{:7.0f}"),
        ("realized rho","env/realized_participation_rate", "{:7.3f}"),
    ]:
        print(f"\n=== {metric} — rows: algorithm/reward, cols: NOEV sessions ===")
        print(f"{'':32s}" + "".join(f"{n:>9d}" for n in NOEV_LEVELS))
        for lab in labels:
            cells = []
            for n in NOEV_LEVELS:
                eps = rows.get((lab, n))
                cells.append(fmt.format(agg(eps, key)) if eps else "    -   ")
            neps = sum(len(rows.get((lab, n), [])) for n in NOEV_LEVELS)
            print(f"{lab:32s}" + "".join(f"{c:>9s}" for c in cells))
    # RQ2.2 helper: at each NOEV level, is the best PPO better than GREEDY on TTT?
    print("\n=== RQ2.2: PPO best-vs-GREEDY TTT margin per NOEV level (negative = PPO better) ===")
    for n in NOEV_LEVELS:
        g = rows.get(("GREEDY", n));
        ppo = [(lab, agg(rows[(lab, n)], "env/ttt_per_ev_mean")) for lab in labels
               if lab not in ("GREEDY","BEST_GUESS","RANDOM") and (lab, n) in rows]
        if not g or not ppo: print(f"  NOEV={n}: insufficient data"); continue
        gt = agg(g, "env/ttt_per_ev_mean")
        best = min(ppo, key=lambda x: x[1])
        rho = agg(rows[(best[0], n)], "env/realized_participation_rate")
        print(f"  NOEV={n:3d} (rho~{rho:.2f}): best PPO {best[0]} TTT={best[1]:.0f} vs GREEDY {gt:.0f}  -> margin {best[1]-gt:+.0f}")
    total = sum(len(v) for v in rows.values())
    print(f"\n{len(rows)} (label,NOEV) cells, {total} episodes total.")

if __name__ == "__main__":
    main()
