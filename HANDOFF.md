# Handoff — 2026-07-18 → tomorrow

Working note for picking up tomorrow. The exhaustive log is `CLAUDE.md`; the
readable summary is `RESULTS.md`. This file = what happened today + what to do
next. **`src/ai/az.pt` (md5 `6b9ad21c50d175ebe266493696d0b121`) must NEVER be
overwritten** — every run uses its own `--out`/`--checkpoint`.

## What happened today (in order)

1. **Portfolio 100% replicated on a 2nd held-out window.** The agent-cascade
   (2 AZ nets @1000 + 3 MCTS RNGs) reaches union **32/32 on both** seeds 16-47
   and 48-79, ~1.4× the cost of one agent. Not this-set luck. (`src/ai/portfolio.py`)

2. **Cost floor measured.** Minimal MCTS-RNG voter subset covering the AZ
   orphans: **k=1 on 16-47** (one RNG covers both orphans), **k=2 on 48-79**
   (no single RNG covers all 3; the hardest orphan has a *unique* coverer). So
   the robust cross-window floor is **k=3 RNGs**; extra voters are cost-free
   (flat cascade cost — only invoked on already-lost seeds).

3. **Consolidated tooling**: `src/ai/portfolio.py` (versioned cascade + coverage
   + cost-floor, with `--from-json` to recompute from saved win-sets) and
   `RESULTS.md` (readable results doc). Replaced the throwaway `tmp/` scripts.

4. **THE BIG ONE — the single-agent "~50-72% ceiling" was a BUDGET artifact.**
   Sweeping the *same* `az.pt` up in sims/move (no retrain):
   | az.pt @ sims | 16-47 | 48-79 |
   |---|---|---|
   | 300 | 16/32 (50%) | — |
   | 1000 | 23/32 (72%) | 22/32 |
   | 2000 | 25/32 (78%) | — |
   | 4000 | 28/32 (88%) | — |
   | 8000 | **32/32 (100%)** | 27/32 (84%) |
   - 100% is **window-dependent** (16-47 hits it at 8000; 48-79 loses 5 at 8000).
   - **No budget-resistant core**: those 5 residual seeds ALL win at **16000**
     sims (`tmp/probe16k.py` → 5/5). Budget *redistributes* the hard set but
     nothing resists escalation.
   - So there are **two equivalent cascade routes to guaranteed 100%**: one net
     with escalating budget, OR the portfolio (agent diversity at fixed budget).
     Agent diversity is not strictly required.
   - Nine training levers chased a ceiling that only existed at ≤1000 sims.

5. **Built the flywheel lever (AZ-teacher bootstrap) and launched a run.** See
   below — this is what's still running.

## The flywheel run — STILL RUNNING (check it first tomorrow)

**Question it answers:** distilling a *stronger* teacher (`az.pt`@4000 = 88%,
vs MCTS@300 = 44%) — is the resulting 1000-sim net **better**, or **just faster
to train**? (The `az.pt`@1000 ≫ MCTS@300 gap on 32 seeds is what re-opened this;
every prior lever bootstrapped from a ~50% teacher and plateaued there.)

**Code shipped (commit `9dc33ee`):** `az_bootstrap_game` + `_az_bootstrap_task`
+ `_worker_load_teacher` in `alphazero.py`; CLI `--bootstrap-teacher {mcts,az}`
+ `--bootstrap-teacher-path` in `az_train.py`. The `mcts` path is byte-for-byte
unchanged. Smoke-tested (parallel + serial + guard).

**The run:** fresh 64-net, `--curriculum --value-target shaped --simulations
1000 --win-buffer-frac 0.25 --bootstrap-games 8 --bootstrap-win-retries 4
--bootstrap-teacher az --bootstrap-teacher-path src/ai/az.pt
--bootstrap-simulations 4000`, out → `src/ai/az_flywheel.pt` / `_ckpt.pt`.
Log: `tmp/flywheel.log`.

**Orchestrator `tmp/orch_flywheel2.sh`** (PID at write time 155701) matures it to
a **p_loss PLATEAU** (a 100-iter session dropping p_loss by < 0.02), then
benchmarks the final + gate nets at **1000 sims on 32 held-out seeds**. Result
appended to `tmp/orch_flywheel2.log` AND to the "RESULT" section at the bottom of
this file by the cleanup waiter. Why plateau not "≤1.65": p_loss is a *flawed*
maturity axis (it decoupled from win rate before — az_deep hit 1.505 and benched
*below* az.pt), so we compare **converged** nets to isolate "better" from
"faster".

**State at handoff (~iter 167):** climbed the full curriculum to stage 8; p_loss
**~1.59 and still falling** (already below az.pt's 1.65 — hence plateau, not a
fixed cutoff); **self-play wins the full game natively at 37-75%** — the 64-net
with an MCTS teacher was chronically 0% here, so this is a real new signal (but
h128b had it too and didn't convert — suggestive, not conclusive). Gate
`eval_win@1000` bounces 44-75% (noisy, ignore — it's on the 16 selection seeds).
`p_loss` at iter 100 was **1.951 vs az_deep's 2.13** at the same point → the AZ
teacher converges faster.

## TODO tomorrow

1. **Read the benchmark result** (bottom of this file / `tmp/orch_flywheel2.log`).
   Compare flywheel@1000 held-out to the references **on the same seeds**:
   - `az.pt`@1000 = **23/32**, `az_deep`@1000 = **21/32**.
   - **≥ ~29/32 → genuinely BETTER** (the strong teacher raised the asymptote;
     the flywheel works — big deal, and it directly serves the live-hint product).
   - **~21-23/32 → JUST FASTER** (same asymptote as MCTS-teacher nets; p_loss
     converged sooner but the ceiling is unchanged).

2. **Beware the gate-selection effect.** The orchestrator benches BOTH the
   `final` net (honest, unselected) and `src/ai/az_flywheel.pt` (gate pick, may
   be inflated on its own selection seeds). Trust the **final** net's number.
   Then **verify on the 2nd window** (`--seed-offset 48`) before believing any
   gain — the selection-effect lesson (a gate-picked 83% collapsed to 28%
   held-out once).

3. **Decide next step by the verdict:**
   - *Better:* run the flywheel *iteration* (bootstrap the next net from
     flywheel@higher-budget — a >88% teacher → the self-improving loop), and wire
     it toward the live-hint product (a faster strong net). Consider committing
     `az_flywheel.pt` + updating `RESULTS.md`/`CLAUDE.md`.
   - *Just faster:* still a real win for **live hints** (reaches a given quality
     at fewer sims/move = lower latency). Document it; the offline 100% is
     already solved by budget escalation, so "better single net" only mattered
     for live latency anyway.

4. **Housekeeping:** `git status` — commit `RESULTS.md`/`CLAUDE.md`/`HANDOFF.md`
   updates. Confirm `az.pt` md5 still `6b9ad21c50d175ebe266493696d0b121`.

## Key artifacts / paths
- `src/ai/portfolio.py` — cascade + cost-floor tool (`--from-json` recompute).
- `src/ai/az.pt` — canonical best single net (NEVER overwrite). 50%@300, 100%@8000 on 16-47.
- `tmp/az_deep2_final_iter300.pt` — 2nd strong net (portfolio's other AZ member), 21/32@1000.
- `src/ai/az_flywheel.pt` / `_ckpt.pt` (+`.winbuf`) — the flywheel net (in progress).
- `tmp/floor_winsets.json`, `tmp/floor2_winsets.json` — per-seed win-sets (16-47, 48-79, 5 RNGs each).
- `tmp/pbench.py` — parallel held-out benchmark. `tmp/probe16k.py` — targeted seed-list probe.

## Session commits (newest first)
`9dc33ee` AZ-teacher bootstrap · `ce1c199` no budget-resistant core ·
`b6bb3b8` 48-79 replication (window-dependent 100%) · `0dae594` az.pt@8000=100% ·
`e1abeb6` budget sweep 87.5% · `f3c04b0` consolidate portfolio.py+RESULTS.md ·
`4bfc565` minimal cover k=1/k=2 · `9fb5ffa` floor k=3 · `eef3e04` cost floor ·
`dc47854`+`0d3bbfc` portfolio 100%.

---
## RESULT — flywheel benchmarked, verdict "just faster, not better" (2026-07-19)

The iter-200 net (cut before plateau) was extracted and benchmarked at **1000
sims on BOTH held-out windows** — the cheap two-window robustness check instead of
the 2-4h plateau run (user's call).

| net | 16-47 | 48-79 | total /64 | p_loss |
|---|---|---|---|---|
| **az.pt** (MCTS teacher) | 23/32 | 22/32 | **45/64** | 1.65 |
| az_deep (MCTS teacher) | 21/32 | 22/32 | 43/64 | 1.505 |
| **flywheel iter-200** (az.pt@4000 teacher) | **18/32** | **22/32** | **40/64** | **1.541** |

**Verdict: NOT better.** On 48-79 it ties the MCTS-teacher nets exactly
(22=22=22); on 16-47 it is 5 below; aggregate 40/64 vs az.pt's 45/64 — same
~50-72% band, côté bas. And it got there *already more mature than az.pt*
(p_loss 1.541 < 1.65) — the identical "maturity does not convert" signature as
az_deep. The strong teacher bought **faster convergence** (p_loss 1.951 @iter100
vs az_deep's 2.13), **not a higher ceiling**. 10th lever, same wall.

**Decision: closed the lever, did NOT run the 2-4h plateau** — two-window
robustness + the maturity precedent make crossing 45/64 very unlikely. Documented
in CLAUDE.md (new box after "No budget-resistant core") + RESULTS.md (ten-levers
para). Committed. `az.pt` intact (`6b9ad21c50d175ebe266493696d0b121`).

Artifacts kept: `tmp/az_flywheel_iter200.pt` (the 40/64 net),
`src/ai/az_flywheel_ckpt.pt(.winbuf)` (iter-200 ckpt, resumable if ever
revisited), `src/ai/az_flywheel.pt` (gate pick — do not trust over the final).
Bench logs: `tmp/flywheel_bench_iter200.log`, `tmp/flywheel_bench_iter200_w2.log`.

## Where the project stands / next directions

The core research thread is **closed**: ten single-agent levers all plateau at
the MCTS-teacher's ~50% *at fixed ≤1000-sim budget*, and the "ceiling" was shown
to be a **budget artifact** — a single `az.pt` reaches ~84-100% by search-budget
escalation, and guaranteed 100% is deployable via either cascade route
(budget-escalation on one net, or the agent-diversity portfolio). Nothing about
the game is unsolved.

Open threads, in rough priority (none urgent):
1. ~~**Cost question:** which guaranteed-100% cascade is compute-cheaper.~~
   **RESOLVED (2026-07-19): they tie (~75-95 s/seed, ≤8% apart, order flips by
   window).** Both dominated by the ~70% of seeds won at the first cheap az@1000
   try; the escalation never reaches its 8000/16000 rungs (non-nested win-sets ⇒
   union of 1000/2000/4000 = 32/32, stops by 4000). **Recommendation: escalate
   budget on one net** — same cost, one model, no MCTS. Portfolio only for
   agent-diversity robustness. See RESULTS.md "Deployable recipe" + benched budget
   ladders `tmp/budget{2000,4000}_4879.log`, tool `tmp/cascade_cost{,2}.py`.
2. **Live-hint product:** a single AZ search on the current position (~0.2-2 s/
   move, drop MCTS). This is the *only* place a faster-strong net (the flywheel's
   real use) would matter — reaching a given hint quality at fewer sims/move.
3. Housekeeping: `tmp/` has many one-off scripts + snapshots; could prune to the
   ones RESULTS.md/CLAUDE.md reference.
