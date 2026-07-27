"""Head-to-head: Gumbel-root vs plain PUCT on az.pt at low sims (16 held-out)."""
from __future__ import annotations
import multiprocessing as mp, os, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

NET = "src/ai/az.pt"
SEED_OFFSET, NSEEDS = 32, 32   # fresh window, disjoint from the 16-31 probe
# (label, sims, m)  m=0 => plain PUCT az_action_fn
CONFIGS = [
    ("plain",  64, 0), ("plain", 128, 0), ("plain", 300, 0),
    ("gumbel", 32, 16), ("gumbel", 64, 16), ("gumbel", 128, 16),
]
WORKERS = 10
_CACHE: dict = {}


def _play(task):
    import torch
    torch.set_num_threads(1)
    from src.ai.policy import load_policy, run_episode_decomposed
    label, sims, m, seed = task
    if NET not in _CACHE:
        _CACHE[NET] = load_policy(NET)
    net = _CACHE[NET]
    if m == 0:
        from src.ai.alphazero import az_action_fn
        act = az_action_fn(net, simulations=sims)
    else:
        from tools.gumbel import gumbel_action_fn
        act = gumbel_action_fn(net, simulations=sims, m=m)
    total, won, steps, parts = run_episode_decomposed(act, seed)
    return (label, sims, m), seed, won, total, steps, parts["clear"]


if __name__ == "__main__":
    from src.ai.policy import eval_seed
    seeds = [eval_seed(i) for i in range(SEED_OFFSET, SEED_OFFSET + NSEEDS)]
    tasks = [(lbl, s, m, sd) for (lbl, s, m) in CONFIGS for sd in seeds]
    print(f"=== gumbel vs plain | az.pt | {NSEEDS} held-out seeds "
          f"| {len(tasks)} games ===", flush=True)
    t0 = time.time()
    agg: dict = {c: [] for c in CONFIGS}
    with mp.Pool(WORKERS) as pool:
        done = 0
        for key, seed, won, total, steps, clear in pool.imap_unordered(_play, tasks):
            agg[key].append((won, total, steps, clear))
            done += 1
            if done % 16 == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:5.0f}s", flush=True)
    print("\n" + "=" * 64)
    print(f"{'config':>16}  {'win%':>6}  {'wins':>6}  {'return':>8}  {'clear':>7}  {'steps':>6}")
    for c in CONFIGS:
        lbl, sims, m = c
        rows = agg[c]
        n = len(rows)
        wins = sum(1 for r in rows if r[0])
        tag = f"{lbl}@{sims}" + (f" m{m}" if m else "")
        print(f"{tag:>16}  {100*wins/n:>6.1f}  {wins:>3}/{n:<2}  "
              f"{sum(r[1] for r in rows)/n:>8.2f}  "
              f"{sum(r[3] for r in rows)/n:>7.1f}  "
              f"{sum(r[2] for r in rows)/n:>6.1f}", flush=True)
    print(f"\ntotal wall {time.time()-t0:.0f}s")
