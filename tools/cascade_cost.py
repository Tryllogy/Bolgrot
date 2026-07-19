"""Cost comparison of the two guaranteed-100% cascade routes on window 16-47.

Currency = wall-clock seconds per seed (the deployable cost). Per-move time is
measured (tmp/timing.py); AZ per-move scales linearly with sims (per-sim ~const),
MCTS@300 measured directly. plies treated flat (means are 59-61 across configs;
loss games ~52 -- a <15% effect dwarfed by the 1000->8000 sims spread).

Route A -- budget escalation on ONE net (az.pt): replay 1000->2000->4000->8000,
          stop at first win.
Route B -- portfolio at FIXED budget: az.pt@1000 -> az_deep@1000 -> MCTS@300 RNGs,
          stop at first win.
Both reach 32/32 on 16-47.
"""
import json, sys

# ---- measured per-move seconds (filled from tmp/timing.py) ----
AZ_MOVE_S_AT_1000 = float(sys.argv[1])   # seconds/move for az@1000
MCTS_MOVE_S_AT_300 = float(sys.argv[2])  # seconds/move for mcts@300
PLIES = 58

def az_game_s(sims):   return AZ_MOVE_S_AT_1000 * (sims/1000) * PLIES
def mcts_game_s():     return MCTS_MOVE_S_AT_300 * PLIES

# ---- win-sets, window 16-47 ----
J = json.load(open("tools/floor_winsets.json"))
seeds = J["seeds"]
az1000 = set(J["az"]["az.pt@1000"])
azdeep = set(J["az"]["az_deep@1000"])
mcts = {int(k): set(v) for k,v in J["mcts"].items()}

# az.pt budget ladder (from budget logs)
w2000 = {1623,1825,1926,2027,2128,2229,2431,2532,2633,2835,2936,3037,3239,3340,
         3441,3542,3643,3744,3845,3946,4047,4249,4350,4653,4754}
w4000 = {1623,1724,1825,2027,2128,2229,2431,2532,2633,2734,2835,2936,3037,3138,
         3340,3441,3542,3643,3744,3845,3946,4047,4148,4350,4451,4552,4653,4754}
w8000 = set(seeds)  # all 32
ladder = [(1000, az1000), (2000, w2000), (4000, w4000), (8000, w8000)]

# ---------- Route A: budget escalation ----------
A_cost, A_wins, A_budget_hist = [], 0, {}
for s in seeds:
    tried_s = 0.0; won=False
    for b, wset in ladder:
        tried_s += az_game_s(b)
        if s in wset:
            won=True; A_budget_hist[b]=A_budget_hist.get(b,0)+1; break
    A_cost.append(tried_s); A_wins += won
meanA = sum(A_cost)/len(A_cost)

# ---------- Route B: portfolio cascade ----------
# order best-first: az.pt@1000(23), az_deep@1000(21), then MCTS RNGs by count
rng_order = sorted(mcts, key=lambda k: -len(mcts[k]))
portfolio = [("az.pt@1000", az1000, az_game_s(1000)),
             ("az_deep@1000", azdeep, az_game_s(1000))]
for k in rng_order:
    portfolio.append((f"mcts@300/rng{k}", mcts[k], mcts_game_s()))

def route_B(agents):
    cost, wins, depth_hist = [], 0, {}
    for s in seeds:
        tried=0.0; won=False
        for i,(name,wset,gs) in enumerate(agents,1):
            tried += gs
            if s in wset:
                won=True; depth_hist[i]=depth_hist.get(i,0)+1; break
        cost.append(tried); wins += won
    return sum(cost)/len(cost), wins, depth_hist

# minimal portfolio: 2 AZ + enough RNGs to cover orphans
orphans = set(seeds) - az1000 - azdeep
print(f"seeds={len(seeds)}  2-AZ union={len(az1000|azdeep)}/32  orphans={sorted(orphans)}")
# find min RNGs (index order) to cover orphans
covered=set(); needed=[]
for k in rng_order:
    if orphans - covered:
        needed.append(k); covered |= (mcts[k] & orphans)
    if not (orphans - covered): break
print(f"orphan-covering RNGs (best-first): {needed}")

for label, agents in [("Route B (2 AZ + covering RNGs)",
                       portfolio[:2] + [(f"mcts@300/rng{k}", mcts[k], mcts_game_s()) for k in needed]),
                      ("Route B (2 AZ + all 5 RNGs)", portfolio)]:
    m,w,dh = route_B(agents)
    print(f"\n{label}: win {w}/32  mean {m:6.1f}s/seed")
    print(f"   cascade depth histogram (agent# : #seeds): {dict(sorted(dh.items()))}")

print(f"\nRoute A (budget escalation 1000->8000): win {A_wins}/32  mean {meanA:6.1f}s/seed")
print(f"   first-win budget histogram: {dict(sorted(A_budget_hist.items()))}")
print(f"\nper-game seconds: az@1000={az_game_s(1000):.1f} az@2000={az_game_s(2000):.1f} "
      f"az@4000={az_game_s(4000):.1f} az@8000={az_game_s(8000):.1f} mcts@300={mcts_game_s():.1f}")
