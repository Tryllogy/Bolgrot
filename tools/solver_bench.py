"""Quick parallel benchmark of the beam solver vs az.pt@1000 (held-out)."""
from __future__ import annotations
import multiprocessing as mp, os, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

NET = "src/ai/az.pt"
# (scoring, beam, max_evals)
CONFIGS = [("prior", 800, 60_000), ("value", 800, 60_000), ("mix", 800, 60_000)]
_C: dict = {}


def _play(task):
    import torch
    torch.set_num_threads(1)
    from src.ai.policy import load_policy
    from src.game import Game
    from src.entity.player import Player
    from src import constant
    from tools.solver import solve
    scoring, beam, mx, seed = task
    if NET not in _C:
        _C[NET] = load_policy(NET)
    g = Game(Player(*constant.BASE_PLAYER_POS), seed=seed)
    t = time.time()
    won, _line, ev = solve(g, _C[NET], max_evals=mx, beam=beam, scoring=scoring)
    return (scoring, beam, mx), won, ev, time.time() - t


if __name__ == "__main__":
    import argparse
    from src.ai.policy import eval_seed
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--offset", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    seeds = [eval_seed(i) for i in range(a.offset, a.offset + a.seeds)]
    tasks = [(sc, b, m, s) for (sc, b, m) in CONFIGS for s in seeds]
    print(f"=== beam solver bench | {a.seeds} held-out seeds "
          f"(offset {a.offset}) | {len(tasks)} runs ===", flush=True)
    t0 = time.time()
    agg: dict = {c: [] for c in CONFIGS}
    with mp.Pool(a.workers) as pool:
        for key, won, ev, dt in pool.imap_unordered(_play, tasks):
            agg[key].append((won, ev, dt))
            print(f"  {key} -> won={won} evals={ev} ({dt:.0f}s)", flush=True)
    print("\n" + "=" * 56)
    print(f"{'config':>22}  {'win%':>6}  {'wins':>6}  {'evals':>8}  {'s/seed':>7}")
    for c in CONFIGS:
        rows = agg[c]
        n = len(rows)
        wins = sum(1 for r in rows if r[0])
        print(f"{str(c):>22}  {100*wins/n:>6.1f}  {wins:>3}/{n:<2}  "
              f"{sum(r[1] for r in rows)/n:>8.0f}  {sum(r[2] for r in rows)/n:>7.1f}",
              flush=True)
    print(f"\nref az.pt@1000 on offset-16 seeds: ~5-6/8 held-out")
    print(f"total wall {time.time()-t0:.0f}s")
