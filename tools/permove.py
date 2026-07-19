import os, time
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import torch; torch.set_num_threads(1)
from src.game import Game
from src.entity.player import Player
from src import constant
from src.ai.policy import load_policy, eval_seed
from src.ai.alphazero import az_action_fn

net=load_policy("src/ai/az.pt")
def new_game(seed):
    return Game(Player(*constant.BASE_PLAYER_POS), seed=seed)

# warm the game to a representative mid-board position with a cheap policy
warm=az_action_fn(net,300,seed=0)
g=new_game(eval_seed(2))
for _ in range(8):
    if g.done: break
    g.step(warm(g))

for sims in (1000, 8000, 16000):
    fn=az_action_fn(net,sims,seed=0)
    ts=[]
    gg=g.clone()
    for _ in range(3):
        if gg.done: break
        t=time.time(); a=fn(gg); ts.append(time.time()-t); gg.step(a)
    avg=sum(ts)/len(ts)
    print(f"{sims:6d} sims/move: {avg:6.2f} s/move  ({avg/sims*1e6:5.1f} us/sim)  [n={len(ts)}]")
