"""Parallel benchmark: depth-limited negamax vs az.pt@1000 (held-out)."""
from __future__ import annotations
import multiprocessing as mp, os, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

NET = "src/ai/az.pt"
# (depth, k)
CONFIGS = [(6, 4), (5, 5), (7, 3)]
_C: dict = {}


def _play(task):
    import torch
    torch.set_num_threads(1)
    from src.ai.policy import load_policy, run_episode_decomposed
    import tools.negamax as nm
    d, k, seed = task
    if NET not in _C:
        _C[NET] = load_policy(NET)
    nm._EVALS = 0
    t = time.time()
    total, won, steps, parts = run_episode_decomposed(
        nm.negamax_action_fn(_C[NET], depth=d, k=k), seed)
    epm = nm._EVALS // max(1, steps)
    return (d, k), won, parts["clear"], epm, time.time() - t


if __name__ == "__main__":
    import argparse
    from src.ai.policy import eval_seed
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--offset", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    seeds = [eval_seed(i) for i in range(a.offset, a.offset + a.seeds)]
    tasks = [(d, k, s) for (d, k) in CONFIGS for s in seeds]
    print(f"=== negamax bench | {a.seeds} held-out seeds (offset {a.offset}) "
          f"| {len(tasks)} runs ===", flush=True)
    t0 = time.time()
    agg: dict = {c: [] for c in CONFIGS}
    with mp.Pool(a.workers) as pool:
        for key, won, clear, epm, dt in pool.imap_unordered(_play, tasks):
            agg[key].append((won, clear, epm, dt))
            print(f"  d{key[0]}k{key[1]} -> won={won} clear={clear:.0f} "
                  f"evals/move~{epm} ({dt:.0f}s)", flush=True)
    print("\n" + "=" * 60)
    print(f"{'config':>10}  {'win%':>6}  {'wins':>6}  {'clear':>7}  {'ev/mv':>7}  {'s/seed':>7}")
    for c in CONFIGS:
        rows = agg[c]
        n = len(rows)
        wins = sum(1 for r in rows if r[0])
        print(f"{'d'+str(c[0])+'k'+str(c[1]):>10}  {100*wins/n:>6.1f}  {wins:>3}/{n:<2}  "
              f"{sum(r[1] for r in rows)/n:>7.0f}  {sum(r[2] for r in rows)/n:>7.0f}  "
              f"{sum(r[3] for r in rows)/n:>7.0f}", flush=True)
    print(f"\nref az.pt@1000 on offset-16 seeds: ~5-6/8 (1000 ev/move)")
    print(f"total wall {time.time()-t0:.0f}s")
