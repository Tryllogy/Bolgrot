import multiprocessing as mp, os, time
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
_C={}
def _play(a):
    sd,m=a
    import torch; torch.set_num_threads(1)
    from src.ai.policy import load_policy, run_episode_decomposed
    if "n" not in _C: _C["n"]=load_policy("src/ai/az.pt")
    if m==0:
        from src.ai.alphazero import az_action_fn; act=az_action_fn(_C["n"],simulations=300)
    else:
        from tools.gumbel import gumbel_action_fn; act=gumbel_action_fn(_C["n"],simulations=300,m=m)
    total,won,steps,parts=run_episode_decomposed(act,sd)
    return m,won,total,parts["clear"]
if __name__=="__main__":
    from src.ai.policy import eval_seed
    seeds=[eval_seed(i) for i in range(16,48)]   # canonical held-out window
    tasks=[(s,m) for m in (0,16) for s in seeds]
    t0=time.time()
    with mp.Pool(10) as p: R=p.map(_play,tasks)
    for m in (0,16):
        rows=[r for r in R if r[0]==m]; n=len(rows); wins=sum(1 for r in rows if r[1])
        tag="plain@300" if m==0 else "gumbel@300 m16"
        print(f"{tag:16s}: {100*wins/n:5.1f}%  {wins}/{n}  return {sum(r[2] for r in rows)/n:6.1f}  clear {sum(r[3] for r in rows)/n:6.1f}")
    print(f"({time.time()-t0:.0f}s) | canonical ref: az.pt@300 = 16/32 (50%)")
