"""Stockfish-style goal-directed solver for the (deterministic, 1-player) game.

PUCT/MCTS is built for stochastic/adversarial play and *averages* visits; this
game is deterministic and single-player, so a search that commits to a line and
verifies it to the win predicate is a better structural match. This is a
net-guided **beam search** (bounded-width best-first) over az.pt:

  - leaf eval  = az.pt value head   (the "NNUE" — how close a state is to a win)
  - move order = az.pt policy prior  (the trained move preferences)
  - death pruning = a lost child is dropped; a *zugzwang* (all of a node's
    children die) leaves no survivors and drops out, so the beam BACKTRACKS past
    the trap reactive PUCT walks into ~10 plies early
  - stop at the FIRST child that reaches the win predicate => a complete winning
    line (deterministic, so replaying it wins the seed for sure)

Scoring modes (how the beam ranks survivors each ply):
  "value"  — by the value head only (collapses onto a non-winning basin; weak)
  "prior"  — by cumulative log policy-prior (follow the trained policy's line)
  "mix"    — value*vw + cumulative log-prior

Budget = net value-evaluations (the fair unit vs a PUCT "sim" = one leaf eval),
reported with win% to read efficiency vs az@1000. No retraining.
"""
from __future__ import annotations
import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import torch  # noqa: E402

from src.ai.network import MASK_FILL  # noqa: E402
from src.game import Game  # noqa: E402
from src.entity.player import Player  # noqa: E402
from src import constant  # noqa: E402


def _board_key(state: dict) -> tuple:
    return (state["px"], state["py"], state["hp"], state["pa"],
            state["waves_spawned"], state["turn"],
            frozenset(map(tuple, state["flames"])))


def _prior(logits_row: torch.Tensor, mask: list[bool]) -> dict[int, float]:
    m = torch.as_tensor(mask, dtype=torch.bool)
    p = torch.softmax(logits_row.masked_fill(~m, MASK_FILL), dim=-1)
    return {i: float(p[i]) for i, ok in enumerate(mask) if ok}


@torch.no_grad()
def solve(game: Game, net, max_evals: int = 50_000, beam: int = 400,
          scoring: str = "prior", vw: float = 0.02,
          max_depth: int = 220) -> tuple[bool, list[int] | None, int]:
    """Beam-search for a winning line. Returns (won, line, evals_used)."""
    work = game.clone()
    root_state = work.get_state()
    work.set_state(root_state)
    root_logits = net(torch.as_tensor([work.get_observation()],
                                      dtype=torch.float32))[0][0]
    root_prior = _prior(root_logits, work.legal_actions())
    # frontier entry: (state, path, cum_logprob, prior_over_own_legal)
    frontier = [(root_state, [], 0.0, root_prior)]
    evals = 1

    for _ in range(max_depth):
        if not frontier:
            return False, None, evals
        seen: set = set()
        cand_obs: list[list[float]] = []
        meta: list[tuple] = []          # (child_state, path, cum_lp, child_mask)

        for state, path, cum_lp, prior in frontier:
            work.set_state(state)
            legal = [i for i, ok in enumerate(work.legal_actions()) if ok]
            for a in legal:
                work.set_state(state)
                work.step(a)
                if work.done:
                    if work.won:
                        return True, path + [a], evals
                    continue
                child = work.get_state()
                key = _board_key(child)
                if key in seen:
                    continue
                seen.add(key)
                lp = cum_lp + math.log(prior.get(a, 1e-9) + 1e-12)
                cand_obs.append(work.get_observation())
                meta.append((child, path + [a], lp, work.legal_actions()))

        if not meta:
            return False, None, evals
        logits, values = net(torch.as_tensor(cand_obs, dtype=torch.float32))
        evals += len(meta)

        scored = []
        for i, (child, cpath, lp, cmask) in enumerate(meta):
            if scoring == "prior":
                s = lp
            elif scoring == "value":
                s = float(values[i])
            else:  # mix
                s = vw * float(values[i]) + lp
            scored.append((s, i))
        scored.sort(reverse=True)
        frontier = []
        for _s, i in scored[:beam]:
            child, cpath, lp, cmask = meta[i]
            frontier.append((child, cpath, lp, _prior(logits[i], cmask)))
        if evals >= max_evals:
            return False, None, evals

    return False, None, evals


def solver_win(seed: int, max_evals: int, beam: int, scoring: str,
               net_path: str) -> tuple[bool, int]:
    from src.ai.policy import load_policy
    net = load_policy(net_path)
    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed)
    won, _line, evals = solve(game, net, max_evals=max_evals, beam=beam,
                              scoring=scoring)
    return won, evals


if __name__ == "__main__":
    import time
    from src.ai.policy import load_policy, eval_seed
    net = load_policy("src/ai/az.pt")
    # 1825 (idx18) az@300 wins; 1623 (idx16) only mcts wins (hard)
    for idx in (18, 19, 16):
        for scoring in ("prior", "mix"):
            g = Game(Player(*constant.BASE_PLAYER_POS), seed=eval_seed(idx))
            t = time.time()
            won, line, ev = solve(g, net, max_evals=30_000, beam=300,
                                  scoring=scoring)
            print(f"seed {eval_seed(idx)} {scoring:5s} beam300: won={won} "
                  f"len={len(line) if line else '-'} evals={ev} "
                  f"({time.time()-t:.1f}s)", flush=True)
