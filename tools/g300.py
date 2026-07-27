import multiprocessing as mp, os, time
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
_C={}
def _play(sd):
    import torch; torch.set_num_threads(1)
    from src.ai.policy import load_policy, run_episode_decomposed
    from tools.gumbel import gumbel_action_fn
    if "n" not in _C: _C["n"]=load_policy("src/ai/az.pt")
    act=gumbel_action_fn(_C["n"],simulations=300,m=16)
    total,won,steps,parts=run_episode_decomposed(act,sd)
    return won,total,steps,parts["clear"]
if __name__=="__main__":
    from src.ai.policy import eval_seed
    seeds=[eval_seed(i) for i in range(32,64)]
    t0=time.time()
    with mp.Pool(10) as p: R=p.map(_play,seeds)
    n=len(R); wins=sum(1 for r in R if r[0])
    print(f"gumbel@300 m16: {100*wins/n:.1f}%  {wins}/{n}  return {sum(r[1] for r in R)/n:.1f}  clear {sum(r[3] for r in R)/n:.1f}  steps {sum(r[2] for r in R)/n:.1f}  ({time.time()-t0:.0f}s)")
    print("ref plain@300 same seeds: 43.8% 14/32  return 363.6  clear 321.2")
