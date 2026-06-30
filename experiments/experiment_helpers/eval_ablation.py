"""
Full-observability reward-formulation ablation (Section 5.8 table).

Evaluates each already-trained reward formulation at 600 vehicles, 0 NOEV, deterministic,
seed 54321, 50 episodes for stable mean ± std. Distinct formulations -> distinct run dirs,
so no metrics-file collision (unlike same-family checkpoint evals).

Capped at 3 concurrent workers (machine comfortably handles 4 trainings; 1 training + 3 evals
leaves headroom). Run alongside the Exp 3c training. Prints an ablation table sorted by TTT.
"""
import os, sys, glob, json, time, statistics as st
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_utils import evaluate_model_with_config

N_EPISODES = 50
SEED = 54321
MAX_WORKERS = 3

_BASE = "runs/_runs_20260612_1004 (bast 600)"
# (label, run dir) — the original v1.4.1-18 set (consistent with the prior decomposition);
# the newer ~9h-longer copies in the same folder are "basically the same" per the user.
FORMULATIONS = [
    ("basic",                        f"{_BASE}/2026-06-12_00-10-03_pid2420664_v1.4.1-18-g88057f4_basic_bast_straight_120km_PPO"),
    ("basicCongestion",              f"{_BASE}/2026-06-12_00-10-03_pid2420665_v1.4.1-18-g88057f4_basicCongestion_bast_straight_120km_PPO"),
    ("basicRelativeDestination",     f"{_BASE}/2026-06-12_00-10-04_pid2420666_v1.4.1-18-g88057f4_basicRelativeDestination_bast_straight_120km_PPO"),
    ("relativeDestination",          f"{_BASE}/2026-06-12_00-10-04_pid2420667_v1.4.1-18-g88057f4_relativeDestination_bast_straight_120km_PPO"),
    ("relativeDestinationCongestion","runs/2026-06-13_14-46-59_pid942615_v1.4.1-28-gf1bf83d_relativeDestinationCongestion_bast_straight_120km_PPO"),
]

def model_for(run_dir):
    if not os.path.isdir(run_dir):
        return None
    canonical = os.path.join(run_dir, os.path.basename(run_dir) + ".zip")
    if os.path.exists(canonical):
        return canonical
    zips = sorted(glob.glob(os.path.join(run_dir, "*.zip")))  # top-level only, not checkpoints/
    return zips[0] if zips else None

def run_one(model_path):
    evaluate_model_with_config(model_load_path=model_path, n_episodes=N_EPISODES,
                               n_vehicles=600, random_seed=SEED, deterministic=True)

def run_capped(jobs, cap, stagger_s=20):
    """At most `cap` workers concurrently; `stagger_s` between new starts so the
    simultaneous 852MB OBELIS feather loads don't spike RAM all at once."""
    q=list(jobs); running=[]
    while q or running:
        while q and len(running) < cap:
            p=Process(target=run_one, args=(q.pop(0),)); p.start(); running.append(p)
            if stagger_s: time.sleep(stagger_s)
        for p in running[:]:
            p.join(timeout=1)
            if not p.is_alive(): running.remove(p)

def summarize(tagged):
    print(f"\n{'formulation':30s} {'eps':>3s} {'empty/ep':>9s} {'TTT/ev':>9s} {'CWT/ev':>9s} {'stops':>6s}")
    rows=[]
    for tag, mp in tagged:
        best=None
        for rc in sorted(glob.glob("runs/*/evaluation/run_config_*.json"), key=os.path.getmtime, reverse=True):
            try: c=json.load(open(rc))
            except: continue
            if c.get("model_load_path")==mp and c.get("random_seed")==SEED and c.get("n_episodes")==N_EPISODES:
                mf=os.path.join(os.path.dirname(rc), "metrics"+os.path.basename(rc)[len("run_config_"):])
                if os.path.exists(mf): best=mf; break
        if not best: print(f"{tag:30s}  (no metrics)"); continue
        eps=json.load(open(best))
        def m(k):
            v=[e[k] for e in eps if k in e]; return st.mean(v) if v else float('nan')
        def sd(k):
            v=[e[k] for e in eps if k in e]; return st.stdev(v) if len(v)>1 else 0.0
        rows.append((tag, len(eps), m("env/empty_vehicles_per_episode"), m("env/ttt_per_ev_mean"),
                     sd("env/ttt_per_ev_mean"), m("env/cwt_per_ev_mean"), m("env/charging_stops_per_episode_mean")))
    for r in sorted(rows, key=lambda x:x[3]):
        print(f"{r[0]:30s} {r[1]:3d} {r[2]:9.2f} {r[3]:6.0f}±{r[4]:<4.0f} {r[5]:9.0f} {r[6]:6.2f}")
    print("\n[GREEDY] empty=1.6 TTT=3288 CWT=318   [BEST_GUESS] empty=6.5 TTT=4269 CWT=65")

if __name__ == "__main__":
    tagged=[]
    for lab, pat in FORMULATIONS:
        mp=model_for(pat)
        if mp: tagged.append((lab, mp)); print("queued", lab, "->", os.path.basename(mp))
        else: print("MISSING model for", lab, pat)
    run_capped([mp for _,mp in tagged], MAX_WORKERS)
    summarize(tagged)
    print("DONE")
