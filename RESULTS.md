# Results — beating Bolgrot with a search agent

A readable summary of the AI work. The exhaustive research log (every lever, every
dead end, exact commands) lives in `CLAUDE.md`; this file is the story and the
takeaways. All win rates are on **32 held-out seeds** the training gate never saw
(seed indices 16–47, i.e. `eval_seed(i) = 101·i + 7`); anything measured on fewer
than ~32 seeds is noise (one win = 3 points, binomial sd ≈ 2.8 wins at n=32).

## TL;DR

- The full 6-wave game is **winnable**, and search budget alone lifts a single
  AlphaZero net from **50% to ~84–100%** — `az.pt`@8000 sims = **32/32 on one
  held-out window, 27/32 on a second**, no retrain. The famous "~50–72% ceiling"
  was a *budget* artifact: it was only ever measured at ≤1000 sims/move.
- 100% at a *fixed* budget is window-dependent (48-79 loses 5 seeds at 8000), and
  those 5 are *different* from the ones lost at 1000 — budget **redistributes**
  the hard set. But there is **no budget-resistant core**: all 5 residual seeds
  are won at 16000 sims. So a single net solves every seed by **budget
  escalation** (retry at rising sims/move, stop at the first win).
- Guaranteed 100% therefore has **two equivalent cascade routes**, and agent
  diversity is *not* strictly required: (a) one net, escalating budget; (b) the
  portfolio — different agents at a fixed budget (~1.4× cost). Both exploit the
  same fact: at a fixed budget, configs fail on disjoint seeds, but the union
  (over budget *or* over agents) is 100%.
- All deployable (a loss is observable — the agent dies — and seeds are
  replayable). Model-free PPO wins **0%** — the winning line is too deep to
  stumble into; only planning/learned-search agents win at all.

## Why the game is hard

A flame is removed **only by jumping onto its tile**. But every jump also shoves
adjacent flames outward (a flame chained into another flame is instant death) and
then drags *all* other flames one tile toward the player (a flame reaching the
player is instant death). So clearing greedily is self-defeating. HP is the
currency — jumps cost 1 HP (refunded on a clear), the whole 40-HP bar affords
~40 unproductive actions, and a win needs ~36 flames cleared without a single
lethal miss. At any moment only ~2 of 17 legal moves make *safe* progress. That
tight margin — not search depth or exploration — is the real wall.

## The single-agent ceiling

| agent | held-out win% (32 seeds) | notes |
|---|---|---|
| random / greedy / **PPO** | **0/32 (0%)** | model-free RL never wins — the ~60-move line is too deep to stumble into |
| MCTS @300 sims | 14/32 (43.8%) | planning search; wins, but its `safe_greedy` rollout leaf is noisy |
| **AlphaZero `az.pt` @300** | **16/32 (50.0%)** | learned value head replaces the rollout; the reference number |
| AlphaZero `az.pt` @1000 | 23/32 (71.9%) | AZ's *play* scales with budget where MCTS's does not |
| AlphaZero `az.pt` @2000 | 25/32 (78.1%) | same net, more sims/move — no retrain |
| AlphaZero `az.pt` @4000 | 28/32 (87.5%) | wins the chronic-loss and double-orphan seeds |
| **AlphaZero `az.pt` @8000** | **32/32 (window 16-47) · 27/32 (window 48-79)** | single agent, no retrain — 100% is window-dependent |

**The single-agent "ceiling" was a budget artifact — but 100% is
window-dependent.** Every ~50-72% figure was measured at ≤1000 sims/move; the
*same* net climbs monotonically far past it (16-47: 50 → 72 → 78 → 88 → 100%;
48-79: 22/32@1000 → 27/32@8000 = 84%), no retraining. Budget wins the *known*
hard seeds — on 48-79 the three orphans the portfolio needed MCTS voters for are
all won at 8000 — but even at 8000 the second window loses 5 seeds, and they are
*different* from the ones lost at 1000: **budget redistributes the residual hard
set rather than eliminating it.** So a single agent is a ~84–100% solver
(window-dependent), not a guaranteed one; guaranteed 100% still needs agent
diversity (the portfolio). This is the autosolver/offline budget (8000 sims ≈
12 s/move), not the live-hint budget.

Ten levers were tried to push a single agent past the ~50% band *at fixed
(≤1000-sim) budget* — reward-ranking
variants (winloss / blend / twohead), a larger `REWARD_WIN`, a persistent
win-buffer (+ cross-session carry), a multi-seed MCTS teacher, double network
width (h128), deeper (1000-sim) self-play, and a **stronger-teacher flywheel**
(bootstrap from `az.pt`@4000 = 88% instead of MCTS@300 = 44%). **All plateau at
~50%.** Each change won about as many seeds as it lost; the teacher is MCTS at
~50%, and no amount of better-retained / better-covered MCTS data crosses it.

The flywheel is the one lever that tried a teacher *stronger* than MCTS, so it
was the best hope of raising the asymptote — and it did not. Its 64-net matured
faster (`p_loss` 1.951 at iter 100 vs the MCTS-teacher `az_deep`'s 2.13) but
landed in the same band: at 1000 sims on the two held-out windows it scored
**18/32 + 22/32 = 40/64**, below `az.pt`'s **45/64** and level with `az_deep`'s
43/64 — despite already being *more* mature than `az.pt` (`p_loss` 1.541 < 1.65),
the same "maturity does not convert" signature. So a stronger teacher buys
*faster convergence, not a higher ceiling*: the flywheel's only real use is
reaching a given quality at fewer sims/move (lower live-hint latency), not
raising the offline solver's win rate (already 100% via budget escalation).

## The reframing: reliability, not strength

The question no lever asked — *can the lost games be won at all?* — has a clean
answer: **yes, all of them.** Replaying `az.pt`'s losses at higher budget and
across MCTS rollout RNGs wins every one. And crucially:

- **No single config wins them all** — each wins a *different* ~60%.
- Tripling AZ's search budget (300→1000) **re-rolls which seeds win** more than it
  raises the count (+7 wins, but on the same net a couple flip each way).
- The two strong AZ nets (`az.pt`@1000 = 23/32, `az_deep`@1000 = 21/32) miss only
  **2 common** seeds; their union is 30/32.

So the losses have no repairable local cause — five stories about them (endgame
HP attrition, "dies with 1–2 flames left", a broken search, HP waste, a locatable
fatal blunder) were each measured and refuted. Wins hinge on details no prior or
value controls, so any perturbation just redistributes them. **The fix is not a
better agent; it is combining agents that fail on different seeds.**

## The portfolio cascade

Order agents best-first, replay the seed with each until one wins. Win rate =
union of win-sets; cost = agents replayed until the first win (you never need to
know in advance which one wins). Per-move vote-averaging does **not** work — it
destroys the commitment a single ~60-move line needs (measured 4/8); diversity
must be combined at **whole-game** granularity.

Default portfolio = **2 AlphaZero nets @1000 + 3 MCTS@300 rollout-RNG voters**,
measured fresh on two independent held-out windows:

| window | AZ-only union | + MCTS voters | cascade cost |
|---|---|---|---|
| seeds 16–47 | 30/32 (orphans 3441, 4148) | **32/32 (100%)** | 1.34× |
| seeds 48–79 | 29/32 (orphans 6572, 7178, 7380) | **32/32 (100%)** | 1.44× |

The AZ nets each plateau ~69–72%; the MCTS RNG voters — each winning a *different*
~40% by rollout luck — cover exactly the orphans the nets miss. Because most seeds
fall to the best agent first-try, mean cost stays near 1: the portfolio is
~1.4× one agent, not 5×.

### Cost floor

The smallest voter subset that covers every orphan (a set-cover over the RNGs):

- **16–47: k=1** — one RNG (`rng1`) alone covers both orphans.
- **48–79: k=2** — no single RNG covers all three; the hardest orphan (`7380`) is
  won by **only 1 of 5** sampled RNGs.

So **k=2 is the optimal-selection floor, k=3 the blind-selection robust floor** —
the gap is the price of not knowing which RNG covers each rare orphan. Extra
voters are **cost-free** (flat cascade cost — they only run on already-lost
seeds), so the deployable choice carries a few RNGs as free insurance.

## Reproduce

Everything is in `src/ai/portfolio.py` (parallel, one game per core):

```bash
# run the full portfolio fresh on a held-out window, save per-seed win-sets:
uv run --extra ai python -m src.ai.portfolio \
    --seed-offset 16 --seeds 32 \
    --nets src/ai/az.pt src/ai/az_deep2.pt \
    --az-sims 1000 --mcts-rngs 3 --json tools/portfolio_16.json

# recompute the analysis (union, cost, cost floor) from a saved dump, no games:
uv run --extra ai python -m src.ai.portfolio --from-json tools/portfolio_16.json
```

Saved win-sets from the runs above: `tools/floor_winsets.json` (16–47, 5 RNGs),
`tools/floor2_winsets.json` (48–79, 5 RNGs).

## Deployable recipe

- **Full-game autosolver — three deployable options, all reach ~100%:**
  - *One agent, fixed budget:* `az.pt`@8000 = ~84–100% (window-dependent,
    ~7 s/move measured). Simplest; enough if ~90% suffices.
  - *One agent, escalating budget (guaranteed 100%):* replay at 1000 → 2000 →
    4000 (→ 8000 → 16000), stop at the first win. No budget-resistant core exists,
    so every seed falls to enough sims; most win cheap, so mean cost is low. One
    net, no MCTS.
  - *Portfolio cascade (guaranteed 100%):* 2 AZ nets + 3 MCTS RNGs at a fixed
    budget, ~1.4× a cheaper agent. Agent diversity instead of budget diversity.
  - **Cost — the two routes essentially tie (2026-07-19).** Measured wall-clock
    seconds per solved seed on both held-out windows (per-move times measured:
    az@1000 = 0.805 s/move, mcts@300 = 3.647 s/move; AZ scales linearly with sims):

    | window | escalating-budget (1 net) | portfolio (2 AZ + MCTS) |
    |---|---|---|
    | 16–47 | 78.8 s/seed | **73.0 s/seed** |
    | 48–79 | **87.5 s/seed** | 94.3 s/seed |

    Within ~8%, and the order flips by window — neither dominates. Both are
    dominated by the ~70% of seeds won on the first cheap `az@1000` try (~47 s).
    **The escalation never reaches its expensive rungs:** because win-sets are
    *non-nested* (a seed lost at a high fixed budget is almost always won at a
    *lower* one), the union of the 1000/2000/4000 rungs already covers 32/32 on
    both windows, so escalation stops by **4000 sims** and the 16000-sim rung — only
    ever needed at *fixed* budget — is never paid. **Recommendation: escalating
    budget on one net** — same cost, one model, no MCTS, fewer moving parts. The
    portfolio only earns its extra parts (2 nets + MCTS RNGs) if you want
    agent-diversity robustness against a seed that resists budget on one net.
    (Tool: `tools/cascade_cost.py` / `cascade_cost2.py` over the saved win-sets.)
- **Live "best move here" hint** (chess.com-style): a *single* AlphaZero search on
  the current position — **not** the cascade (a whole-game solver), and **not**
  MCTS (the slow member; AZ ties or beats it at equal budget). Measured per-move
  latency (one search, `tools/permove.py`): **~0.9 s @1000 sims** (~72%
  single-game quality), 7.3 s @8000, 14.8 s @16000 — cost ~linear in sims. So the
  live budget sits near 1000 (sub-second), far below the 8000 offline solver.
  Note a live single search can never guarantee 100% (that needs retries — the
  cascade); like chess.com, it is a strong assistant, not a win guarantee. A
  faster-strong net (the flywheel's only remaining use) would buy higher hint
  quality per second.

## Artifacts

- `src/ai/az.pt` — the canonical best single net (64-wide, ~50% @300, ~72% @1000).
  **Never overwrite it.**
- `src/ai/az_deep2.pt` — the second strong net (deep 1000-sim training),
  wins seeds `az.pt` cannot; the portfolio's other AZ member.
- `src/ai/mcts.py`, `src/ai/alphazero.py` — the two search agents.
- `src/ai/portfolio.py` — the cascade + coverage/cost-floor tool (this analysis).
- `tools/` — reusable analysis scripts kept out of the (deleted) `tmp/` scratch:
  `pbench.py` (parallel held-out benchmark), `probe16k.py` (targeted seed probe),
  `permove.py` (per-move latency), `cascade_cost{,2}.py` (route cost model),
  `floor{,2}_winsets.json` (measured per-seed win-sets, windows 16–47 / 48–79).
