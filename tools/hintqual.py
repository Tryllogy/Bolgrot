"""Quality-safety test for wiring Gumbel into /hint.

The hint currently runs plain PUCT @1000. A real latency win needs a *cheaper*
Gumbel search to match plain@1000 quality. This measures whether gumbel@{128,256}
>= plain@1000 on held-out seeds (full-game autoplay = following the hint).
Only if a cheaper Gumbel ties plain@1000 do we drop the hint budget.
"""
from __future__ import annotations
import multiprocessing as mp, os, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

NET = "src/ai/az.pt"
SEED_OFFSET, NSEEDS = 16, 16
# (label, sims, m)   m=0 => plain PUCT (current hint), m>0 => gumbel
CONFIGS = [("plain", 1000, 0), ("gumbel", 128, 16), ("gumbel", 256, 16)]
WORKERS = 4          # modest: coexist with the background training (11 workers)
_C: dict = {}


def _play(task):
    import torch
    torch.set_num_threads(1)
    from src.ai.policy import load_policy, run_episode_decomposed
    lbl, sims, m, seed = task
    if NET not in _C:
        _C[NET] = load_policy(NET)
    net = _C[NET]
    if m == 0:
        from src.ai.alphazero import az_action_fn
        act = az_action_fn(net, simulations=sims)
    else:
        from tools.gumbel import gumbel_action_fn
        act = gumbel_action_fn(net, simulations=sims, m=m)
    total, won, steps, parts = run_episode_decomposed(act, seed)
    return (lbl, sims, m), won, total, parts["clear"]


if __name__ == "__main__":
    from src.ai.policy import eval_seed
    seeds = [eval_seed(i) for i in range(SEED_OFFSET, SEED_OFFSET + NSEEDS)]
    tasks = [(lbl, s, m, sd) for (lbl, s, m) in CONFIGS for sd in seeds]
    print(f"=== hint quality-safety | az.pt | {NSEEDS} held-out seeds ===", flush=True)
    t0 = time.time()
    agg: dict = {c: [] for c in CONFIGS}
    with mp.Pool(WORKERS) as pool:
        done = 0
        for key, won, total, clear in pool.imap_unordered(_play, tasks):
            agg[key].append((won, total, clear))
            done += 1
            if done % 8 == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:5.0f}s", flush=True)
    print("\n" + "=" * 56)
    print(f"{'config':>16}  {'win%':>6}  {'wins':>6}  {'return':>8}  {'clear':>7}")
    for c in CONFIGS:
        lbl, sims, m = c
        rows = agg[c]
        n = len(rows)
        wins = sum(1 for r in rows if r[0])
        print(f"{lbl+'@'+str(sims):>16}  {100*wins/n:>6.1f}  {wins:>3}/{n:<2}  "
              f"{sum(r[1] for r in rows)/n:>8.2f}  {sum(r[2] for r in rows)/n:>7.1f}",
              flush=True)
    print(f"\ntotal wall {time.time()-t0:.0f}s")
