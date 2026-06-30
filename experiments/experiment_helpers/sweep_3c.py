"""
Exp 4 addendum: run the 4 NOEV conditions for the 3c model
(relativeDestinationCongestionIllegal), which beats 3b at full observability and so
should be the RL agent represented in the partial-observability analysis.

4 conditions (NOEV {0,40,160,320}) x 50 eps, seed 54321, paired with the existing sweep.
Both big jobs are finished, so cap=4 (== machine's safe SUMO max) runs all four at once.
"""
import os, sys, time
from functools import partial
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_utils import evaluate_model, _load_run_config

MODEL = "runs/2026-06-14_16-58-22_pid1992290_v1.4.1-29-g5afbde5_relativeDestinationCongestionIllegal_bast_straight_120km_PPO/2026-06-14_16-58-22_pid1992290_v1.4.1-29-g5afbde5_relativeDestinationCongestionIllegal_bast_straight_120km_PPO.zip"
NOEV_LEVELS = [0, 40, 160, 320]
N_EPISODES = 50
SEED = 54321

def run_one(n_noevs):
    cfg = _load_run_config(MODEL)
    evaluate_model(
        scenario=cfg["scenario"], algorithm="PPO", version_tag=cfg.get("version_tag", "exp4c"),
        reward_strategy=cfg["reward_strategy"], street_network=cfg["street_network"],
        n_vehicles=cfg["n_vehicles"], n_episodes=N_EPISODES, model_load_path=MODEL,
        n_noevs=n_noevs, noev_provider="obelis", random_seed=SEED, deterministic=True,
    )

def run_capped(jobs, cap=4, stagger_s=20):
    q = list(jobs); running = []
    while q or running:
        while q and len(running) < cap:
            p = Process(target=run_one, args=(q.pop(0),)); p.start(); running.append(p)
            if stagger_s: time.sleep(stagger_s)
        for p in running[:]:
            p.join(timeout=1)
            if not p.is_alive(): running.remove(p)

if __name__ == "__main__":
    assert os.path.exists(MODEL), MODEL
    print("Exp4c (3c) NOEV sweep:", NOEV_LEVELS, "x", N_EPISODES, "eps, seed", SEED)
    run_capped(NOEV_LEVELS, cap=4, stagger_s=20)
    print("All runs done")
