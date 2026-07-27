"""Held-out verdict: az_nega (negamax + PUCT) vs az.pt (PUCT@1000)."""
from __future__ import annotations
import multiprocessing as mp, os, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# (label, net, mode, arg)  mode: "nm" -> arg=(depth,k) ; "puct" -> arg=sims
CONFIGS = [
    ("az_nega negamax d5k5", "src/ai/az_nega.pt", "nm", (5, 5)),
    ("az_nega puct@1000",    "src/ai/az_nega.pt", "puct", 1000),
    ("az.pt puct@1000 (ref)", "src/ai/az.pt",     "puct", 1000),
]
_C: dict = {}


def _play(task):
    import torch
    torch.set_num_threads(1)
    from src.ai.policy import load_policy, run_episode_decomposed
    label, net_path, mode, arg, seed = task
    if net_path not in _C:
        _C[net_path] = load_policy(net_path)
    net = _C[net_path]
    if mode == "nm":
        from src.ai.alphazero import nm_action_fn
        act = nm_action_fn(net, depth=arg[0], k=arg[1])
    else:
        from src.ai.alphazero import az_action_fn
        act = az_action_fn(net, simulations=arg)
    total, won, steps, parts = run_episode_decomposed(act, seed)
    return label, won, total, parts["clear"], steps


if __name__ == "__main__":
    import argparse
    from src.ai.policy import eval_seed
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--offset", type=int, default=16)
    p.add_argument("--workers", type=int, default=11)
    a = p.parse_args()
    seeds = [eval_seed(i) for i in range(a.offset, a.offset + a.seeds)]
    tasks = [(lbl, net, mode, arg, s)
             for (lbl, net, mode, arg) in CONFIGS for s in seeds]
    print(f"=== nega verdict | {a.seeds} held-out seeds (offset {a.offset}) "
          f"| {len(tasks)} games ===", flush=True)
    t0 = time.time()
    agg: dict = {c[0]: [] for c in CONFIGS}
    with mp.Pool(a.workers) as pool:
        done = 0
        for label, won, total, clear, steps in pool.imap_unordered(_play, tasks):
            agg[label].append((won, total, clear, steps))
            done += 1
            if done % 16 == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:5.0f}s", flush=True)
    print("\n" + "=" * 68)
    print(f"{'config':>24}  {'win%':>6}  {'wins':>6}  {'return':>8}  {'clear':>7}  {'steps':>6}")
    for lbl, *_ in CONFIGS:
        rows = agg[lbl]
        n = len(rows)
        wins = sum(1 for r in rows if r[0])
        print(f"{lbl:>24}  {100*wins/n:>6.1f}  {wins:>3}/{n:<2}  "
              f"{sum(r[1] for r in rows)/n:>8.2f}  {sum(r[2] for r in rows)/n:>7.1f}  "
              f"{sum(r[3] for r in rows)/n:>6.1f}", flush=True)
    print(f"\nref: az.pt@1000 held-out = 23/32 (71.9%) documented")
    print(f"total wall {time.time()-t0:.0f}s")
