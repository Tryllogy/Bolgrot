import os,time
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import torch; torch.set_num_threads(1)
from src.game import Game
from src.entity.player import Player
from src import constant
from src.ai.policy import load_policy, eval_seed
from src.ai.alphazero import az_action_fn
from tools.gumbel import gumbel_action_fn
net=load_policy("src/ai/az.pt")
g=Game(Player(*constant.BASE_PLAYER_POS), seed=eval_seed(20))
warm=az_action_fn(net,128,seed=0)
for _ in range(10):
    if g.done: break
    g.step(warm(g))
print(f"warmed to a mid-board position (step {10}, flames on board)")
def bench(name,fn):
    gg=g.clone(); ts=[]
    for _ in range(5):
        if gg.done: break
        t=time.time(); a=fn(gg); ts.append(time.time()-t); gg.step(a)
    print(f"  {name:20s} {1000*sum(ts)/len(ts):7.1f} ms/move")
for s in (64,128,300):
    bench(f"plain@{s}", az_action_fn(net,s,seed=0))
for s in (64,128):
    bench(f"gumbel@{s} m16", gumbel_action_fn(net,s,m=16,seed=0))
