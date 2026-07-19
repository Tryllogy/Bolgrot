"""az.pt @ N sims on an explicit seed list — probe whether more budget closes
the seeds a lower budget leaves. One deterministic game per worker."""
import argparse, multiprocessing as mp, os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
_NET={}
def _play(t):
    net,seed,sims=t
    import torch; torch.set_num_threads(1)
    from src.ai.alphazero import az_action_fn
    from src.ai.policy import load_policy, run_episode_decomposed
    if net not in _NET: _NET[net]=load_policy(net)
    total,won,steps,parts=run_episode_decomposed(az_action_fn(_NET[net],simulations=sims),seed)
    return seed,won,total,steps,parts["clear"]
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--net",default="src/ai/az.pt")
    p.add_argument("--sims",type=int,default=16000)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--workers",type=int,default=11)
    a=p.parse_args()
    tasks=[(a.net,s,a.sims) for s in a.seeds]
    print(f"=== {os.path.basename(a.net)} @ {a.sims} sims on {len(a.seeds)} seeds ===",flush=True)
    res={}
    with mp.Pool(min(a.workers,len(tasks))) as pool:
        for seed,won,total,steps,clear in pool.imap_unordered(_play,tasks):
            res[seed]=won
            print(f"  seed {seed}: {'WIN ' if won else 'loss'} | return {total:7.2f} | clear {clear:6.1f} | steps {steps}",flush=True)
    w=sum(res.values())
    print(f"\n=== {w}/{len(a.seeds)} won: {sorted(s for s in res if res[s])} | still lost: {sorted(s for s in res if not res[s])} ===")
