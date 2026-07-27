"""Depth-limited max-backup ("negamax", 1-player) search over az.pt — the
faithful Stockfish test the beam solver was missing: it BACKS UP the value.

Per move (like PUCT is per move):
  value(s, d) = net_value(s)*VALUE_SCALE                       if d == 0
              = max over top-k-by-prior a of [ r(s,a) + gamma*value(s', d-1) ]
  terminal    = r + (+WIN if won else -LOSS)   (dominant, so a forced win within
                d plies is chosen and a forced death within d plies is avoided)

Selective search: only the top-k children by az.pt's policy prior are expanded
(move ordering + hard pruning to k — a chess-engine-style selective search), so
cost is ~k^d net evals/move. Leaf eval and ordering both come from az.pt; no
retraining. Plugs in as an ActionFn (full-game play + win detection reused).
"""
from __future__ import annotations
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import torch  # noqa: E402

from src.ai.network import MASK_FILL  # noqa: E402
from src.ai.alphazero import VALUE_SCALE  # noqa: E402
from src.game import Game  # noqa: E402
from src.ai.policy import ActionFn  # noqa: E402

WIN = 1e5
LOSS = 1e5

# per-call net-eval counter (rough budget read-out)
_EVALS = 0


@torch.no_grad()
def _value(work: Game, state: dict, depth: int, k: int, net, gamma: float
           ) -> float:
    global _EVALS
    work.set_state(state)
    obs = torch.as_tensor(work.get_observation(), dtype=torch.float32)
    logits, value = net(obs)
    _EVALS += 1
    if depth == 0:
        return float(value) * VALUE_SCALE
    mask = work.legal_actions()
    m = torch.as_tensor(mask, dtype=torch.bool)
    probs = torch.softmax(logits.masked_fill(~m, MASK_FILL), dim=-1)
    legal = [i for i, ok in enumerate(mask) if ok]
    topk = sorted(legal, key=lambda a: float(probs[a]), reverse=True)[:k]
    best = -1e18
    for a in topk:
        work.set_state(state)
        r, _ = work.step(a)
        if work.done:
            v = r + (WIN if work.won else -LOSS)
        else:
            v = r + gamma * _value(work, work.get_state(), depth - 1, k, net,
                                   gamma)
        if v > best:
            best = v
    return best


def negamax_action_fn(net, depth: int = 5, k: int = 4, gamma: float = 0.99
                      ) -> ActionFn:
    @torch.no_grad()
    def act(game: Game) -> int:
        if game.done:
            return 0
        work = game.clone()
        root = work.get_state()
        obs = torch.as_tensor(work.get_observation(), dtype=torch.float32)
        logits, _ = net(obs)
        mask = work.legal_actions()
        m = torch.as_tensor(mask, dtype=torch.bool)
        probs = torch.softmax(logits.masked_fill(~m, MASK_FILL), dim=-1)
        legal = [i for i, ok in enumerate(mask) if ok]
        topk = sorted(legal, key=lambda a: float(probs[a]), reverse=True)[:k]
        best_a, best_v = topk[0], -1e18
        for a in topk:
            work.set_state(root)
            r, _ = work.step(a)
            if work.done:
                v = r + (WIN if work.won else -LOSS)
            else:
                v = r + gamma * _value(work, work.get_state(), depth - 1, k,
                                       net, gamma)
            if v > best_v:
                best_v, best_a = v, a
        return best_a
    return act


if __name__ == "__main__":
    import time
    from src.ai.policy import load_policy, eval_seed, run_episode_decomposed
    net = load_policy("src/ai/az.pt")
    # trivial correctness + one real seed at a shallow depth
    for idx in (18, 19):
        for (d, k) in ((4, 4), (6, 3)):
            globals()["_EVALS"] = 0
            act = negamax_action_fn(net, depth=d, k=k)
            t = time.time()
            total, won, steps, parts = run_episode_decomposed(act, eval_seed(idx))
            print(f"seed {eval_seed(idx)} d{d}k{k}: won={won} clear={parts['clear']:.0f} "
                  f"steps={steps} evals/move~{_EVALS//max(1,steps)} ({time.time()-t:.0f}s)",
                  flush=True)
