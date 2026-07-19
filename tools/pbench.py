"""Parallel benchmark: one (net, seed) game per worker process.

The serial `baselines` run uses one core of twelve — the bottleneck is the
Python game loop, not torch (a 128-wide MLP gets nothing from torch's 6 default
threads, and 12 processes x 6 threads would thrash, hence set_num_threads(1)).

Parallelising is safe here because every game is deterministic given
(net, seed): AZ's eval search takes the most-visited root action with Dirichlet
noise disabled, so the numbers are identical to the serial run — only the wall
clock changes.

Prints the same per-term table as `baselines`, plus the per-seed win/loss list
(a 32-seed win rate is the number that matters; the individual seeds let you
compare *which* games a net wins against another net's set).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_NET_CACHE: dict = {}


def _play(task: tuple[str, int, int]) -> tuple[str, int, bool, float, int, dict]:
    import torch
    torch.set_num_threads(1)
    from src.ai.alphazero import az_action_fn
    from src.ai.policy import load_policy, run_episode_decomposed

    net_path, seed, sims = task
    if net_path not in _NET_CACHE:
        _NET_CACHE[net_path] = load_policy(net_path)
    act = az_action_fn(_NET_CACHE[net_path], simulations=sims)
    total, won, steps, parts = run_episode_decomposed(act, seed)
    return net_path, seed, won, total, steps, parts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nets", nargs="+", required=True)
    p.add_argument("--sims", type=int, default=300)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed-offset", type=int, default=16)
    p.add_argument("--workers", type=int, default=11)
    a = p.parse_args()

    from src.ai.policy import eval_seed
    from src.game import REWARD_PARTS

    seeds = [eval_seed(i) for i in range(a.seed_offset, a.seed_offset + a.seeds)]
    tasks = [(net, s, a.sims) for net in a.nets for s in seeds]
    print(f"=== PARALLEL BENCH | {len(a.nets)} nets x {len(seeds)} seeds "
          f"= {len(tasks)} games @ {a.sims} sims | {a.workers} workers ===",
          flush=True)
    print(f"seeds[{a.seed_offset}:{a.seed_offset + a.seeds}] (held out from the "
          f"training gate, which selects on EVAL_SEEDS[:16])\n", flush=True)

    agg: dict[str, list] = {net: [] for net in a.nets}
    with mp.Pool(a.workers) as pool:
        done = 0
        for net, seed, won, total, steps, parts in pool.imap_unordered(_play, tasks):
            agg[net].append((seed, won, total, steps, parts))
            done += 1
            print(f"  [{done:3d}/{len(tasks)}] {os.path.basename(net):32s} "
                  f"seed {seed:5d}: {'WIN ' if won else 'loss'} "
                  f"| return {total:7.2f} | clear {parts['clear']:6.1f}",
                  flush=True)

    print("\n" + "=" * 78)
    cols = ["win%", "wins", "return", "steps", *REWARD_PARTS]
    name_w = max(len("net"), *(len(os.path.basename(n)) for n in a.nets))
    print("net".ljust(name_w) + "  " + "  ".join(c.rjust(8) for c in cols))
    print("-" * (name_w + 2 + 10 * len(cols)))
    for net in a.nets:
        rows = agg[net]
        n = len(rows)
        wins = sum(1 for _, won, _, _, _ in rows if won)
        row = {"win%": 100.0 * wins / n, "wins": wins,
               "return": sum(r[2] for r in rows) / n,
               "steps": sum(r[3] for r in rows) / n}
        row.update({part: sum(r[4][part] for r in rows) / n
                    for part in REWARD_PARTS})
        print(os.path.basename(net).ljust(name_w) + "  " +
              "  ".join(f"{row[c]:8.2f}" for c in cols))
        won_seeds = sorted(s for s, won, _, _, _ in rows if won)
        print(f"{'':>{name_w}}   wins: {won_seeds}")

    print(f"\nreference on these SAME seeds: az.pt@300 16/32 (50.0%) | "
          f"mcts@300 14/32 (43.75%)")
    print(f"bar to beat az.pt beyond noise: ~22/32 (binomial sd ~2.8 wins)")
