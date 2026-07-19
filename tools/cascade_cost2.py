import json, sys
AZ=float(sys.argv[1]); MC=float(sys.argv[2]); PLIES=58
azg=lambda s: AZ*(s/1000)*PLIES; mcg=lambda: MC*PLIES
J=json.load(open("tools/floor2_winsets.json")); seeds=J["seeds"]
az1000=set(J["az"]["az.pt@1000"]); azdeep=set(J["az"]["az_deep2_final_iter300.pt@1000"])
mcts={int(k):set(v) for k,v in J["mcts"].items()}
w2000={4855,4956,5057,5158,5259,5360,5461,5562,5764,5966,6067,6168,6370,6471,6572,6673,6774,6875,7077,7178,7279,7481,7582,7683,7784,7885,7986}
w4000={4855,4956,5057,5158,5259,5360,5461,5562,5663,5764,5865,6067,6168,6269,6370,6471,6572,6673,6774,6875,6976,7077,7178,7279,7380,7481,7582,7683,7885}
w8000={4855,4956,5057,5158,5259,5360,5562,5663,5764,5865,6067,6168,6269,6370,6572,6673,6774,6875,7077,7178,7380,7481,7582,7683,7784,7885,7986}
w16000=set(seeds)
ladder=[(1000,az1000),(2000,w2000),(4000,w4000),(8000,w8000),(16000,w16000)]
# Route A
A=[]; hist={}
for s in seeds:
    t=0
    for b,ws in ladder:
        t+=azg(b)
        if s in ws: hist[b]=hist.get(b,0)+1; break
    A.append(t)
print(f"Route A (escalation 1000->16000): mean {sum(A)/len(A):6.1f}s/seed  first-win {dict(sorted(hist.items()))}")
# Route B
rng=sorted(mcts,key=lambda k:-len(mcts[k]))
orph=set(seeds)-az1000-azdeep
cov=set(); need=[]
for k in rng:
    if orph-cov: need.append(k); cov|=mcts[k]&orph
    if not orph-cov: break
Bagents=[("az.pt@1000",az1000,azg(1000)),("az_deep2@1000",azdeep,azg(1000))]+[(f"rng{k}",mcts[k],mcg()) for k in need]
B=[]; dh={}
for s in seeds:
    t=0; won=False
    for i,(n,ws,g) in enumerate(Bagents,1):
        t+=g
        if s in ws: dh[i]=dh.get(i,0)+1; won=True; break
    if not won: dh['LOSS']=dh.get('LOSS',0)+1
    B.append(t)
print(f"Route B (2 AZ + RNGs {need}): 2-AZ union={len(az1000|azdeep)}/32 orphans={sorted(orph)}")
print(f"   mean {sum(B)/len(B):6.1f}s/seed  depth {dict(sorted(dh.items(),key=lambda x:str(x[0])))}")
print(f"per-game s: az@1000={azg(1000):.0f} az@2000={azg(2000):.0f} az@4000={azg(4000):.0f} az@8000={azg(8000):.0f} az@16000={azg(16000):.0f} mcts@300={mcg():.0f}")
