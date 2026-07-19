# Handoff — current state (updated 2026-07-19)

Pick-up note. Exhaustive research log = `CLAUDE.md`; readable results = `RESULTS.md`.
**`src/ai/az.pt` (md5 `6b9ad21c50d175ebe266493696d0b121`) must NEVER be
overwritten** — every training run uses its own `--out`/`--checkpoint`.

## The research thread is CLOSED

Ten single-agent levers (3 reward-ranking, larger `REWARD_WIN`, win-buffer,
buffer-carry, multi-seed teacher, h128 width, 1000-sim training, AZ-teacher
flywheel) all plateau at the MCTS-teacher's ~50% **at fixed ≤1000-sim budget**.
The decisive finding: **that "ceiling" was a budget artifact.** Sweeping the same
`az.pt` up in sims/move (no retrain) climbs monotonically, and there is **no
budget-resistant core** — every held-out seed is won at some budget ≤16000:

| `az.pt` @ sims | 16-47 | 48-79 | measured latency |
|---|---|---|---|
| 300 | 16/32 (50%) | — | ~0.24 s/move |
| 1000 | 23/32 (72%) | 22/32 | ~0.9 s/move |
| 4000 | 28/32 (88%) | — | ~3.4 s/move |
| 8000 | 32/32 (100%) | 27/32 (84%) | ~7.3 s/move |
| 16000 | — | (residual 5 → 5/5) | ~14.8 s/move |

Reframing (see RESULTS.md): the wall was never *strength*, it was *reliability /
coverage*. Guaranteed 100% is deployable two equivalent ways — **escalate budget
on one net** (1000→2000→4000…, stop at first win) or the **portfolio** (2 AZ nets
+ MCTS RNGs at fixed budget). **Cost: they tie (~75-95 s/seed, ≤8% apart).
Recommendation: escalate one net** (same cost, one model, no MCTS).

The flywheel (10th lever, 2026-07-19): distilling a *stronger* teacher
(`az.pt`@4000=88%) gave **faster convergence, not a higher ceiling** — the net
benched 40/64 vs `az.pt`'s 45/64 across both windows despite lower `p_loss`. Same
"maturity doesn't convert" wall. Closed. (Full box in CLAUDE.md.)

## Deployment artifacts (now committed — were all git-ignored)

- `src/ai/az.pt` — **primary site net.** 50%@300, 72%@1000, ~100% via budget
  escalation. Never overwrite.
- `src/ai/az_deep2.pt` — 2nd strong net (wins seeds `az.pt` cannot, e.g. 1623);
  the portfolio's other AZ member. Kept for the agent-diversity route.
- `src/ai/portfolio.py` — cascade + coverage + cost-floor tool (`--from-json`).
- `src/ai/{mcts,alphazero}.py` — the two search agents.
- `tools/` — reusable scripts (moved out of the deleted `tmp/`): `pbench.py`
  (parallel benchmark), `probe16k.py` (seed probe), `permove.py` (latency),
  `cascade_cost{,2}.py` (route cost model), `floor{,2}_winsets.json` (win-sets).

Housekeeping done 2026-07-19: committed the 2 site nets + `tools/`, **deleted all
closed-lever snapshots and the 107 MB `tmp/`** (findings live in the docs).

## Product direction (the live discussion, 2026-07-19)

The game space is huge — 5 openings × P(114,6) ≈ **10^13 distinct games** — so you
**cannot precompute/cache all seeds**. Consequences for a product:

- **100% is a property of the cascade (retries), not of any single live search.**
  No fixed live budget wins 100% in one attempt (even 16000 in a single pass isn't
  proven 100% — budget *redistributes* the hard set). So a live single search is,
  like chess.com/Stockfish, a **strong assistant, not a win guarantee**.
- **Three product shapes** (they don't mix — chess.com forces the same choice):
  1. *Live "best move" hint* (chess.com-style): one AZ search on the current
     position, **~0.9 s @1000 sims / ~72% single-game quality**, drop MCTS. Ship
     with an honest disclaimer ("strong bot, not guaranteed"). Scalable, cheap.
  2. *Autoplay + "retry harder" that ESCALATES budget*: deterministic + no
     budget-resistant core ⇒ converges to a win on every level. Effectively 100%,
     no precompute; the player waits on hard levels. (Trap: a retry at the *same*
     budget on the *same* seed replays the identical loss — retry must change the
     budget or the seed.)
  3. *Guaranteed 100% at scale* = the "tablebase" model: restrict to a **bounded,
     curated level set**, pre-solve them offline (escalation, ~1-6 min CPU/level),
     serve stored replays (instant, 100%). Only works if levels are bounded, not
     fresh-random-per-player.

The only remaining use of a *faster-strong* net (e.g. the flywheel idea) is lower
live-hint latency — reaching a given quality at fewer sims/move. Not needed for
the offline 100%, which budget escalation already solves.

## If you pick the work back up

- **Benchmark the site nets**: `uv run --extra ai python tools/pbench.py --nets
  src/ai/az.pt --sims 1000 --seeds 32 --seed-offset 16 --workers 11` (or 48 for
  window 2). References: `az.pt`@1000 = 23/32 (16-47), 22/32 (48-79).
- **Recompute cost model**: `python tools/cascade_cost.py 0.805 3.647`
  (per-move seconds for az@1000 / mcts@300; re-measure with `tools/permove.py` if
  hardware changed).
- Every win rate must be re-measured after any flame-mechanic change, and on
  ≥32 seeds off the gate window (`--seed-offset 16`) — see CLAUDE.md's
  selection-effect and reference-number boxes.
