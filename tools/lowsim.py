"""Low-sim scaling probe for the LIVE-HINT latency/quality tradeoff.

The offline solver is already ~100% via budget escalation; the open product
question is how STRONG az.pt is at LOW sims/move (= low hint latency). This
maps az.pt win rate over held-out seeds at a range of small sim budgets, plus
per-move latency, so we know the cheapest sims that still plays well.

Run: uv run --extra ai python -m tools.lowsim
"""
from __future__ import annotations
import multiprocessing as mp, os, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

NET = "src/ai/az.pt"
SEED_OFFSET = 16          # held out from the training gate (EVAL_SEEDS[:16])
NSEEDS = 16
SIMS = [16, 32, 64, 128, 300]
WORKERS = 10

_CACHE: dict = {}


def _play(task):
    import torch
    torch.set_num_threads(1)
    from src.ai.alphazero import az_action_fn
    from src.ai.policy import load_policy, run_episode_decomposed
    path, seed, sims = task
    if path not in _CACHE:
        _CACHE[path] = load_policy(path)
    act = az_action_fn(_CACHE[path], simulations=sims)
    total, won, steps, parts = run_episode_decomposed(act, seed)
    return sims, seed, won, total, steps, parts["clear"]


if __name__ == "__main__":
    from src.ai.policy import eval_seed
    seeds = [eval_seed(i) for i in range(SEED_OFFSET, SEED_OFFSET + NSEEDS)]
    tasks = [(NET, s, sims) for sims in SIMS for s in seeds]
    print(f"=== low-sim scaling | az.pt | {NSEEDS} held-out seeds "
          f"| sims={SIMS} | {len(tasks)} games ===", flush=True)
    t0 = time.time()
    res: dict[int, list] = {s: [] for s in SIMS}
    with mp.Pool(WORKERS) as pool:
        done = 0
        for sims, seed, won, total, steps, clear in pool.imap_unordered(_play, tasks):
            res[sims].append((seed, won, total, steps, clear))
            done += 1
            if done % 16 == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:5.0f}s", flush=True)
    print("\n" + "=" * 60)
    print(f"{'sims':>6}  {'win%':>6}  {'wins':>6}  {'return':>8}  {'clear':>7}  {'steps':>6}")
    for sims in SIMS:
        rows = res[sims]
        n = len(rows)
        wins = sum(1 for _, w, _, _, _ in rows if w)
        print(f"{sims:>6}  {100*wins/n:>6.1f}  {wins:>3}/{n:<2}  "
              f"{sum(r[2] for r in rows)/n:>8.2f}  "
              f"{sum(r[4] for r in rows)/n:>7.1f}  "
              f"{sum(r[3] for r in rows)/n:>6.1f}", flush=True)
    print(f"\ntotal wall {time.time()-t0:.0f}s")
