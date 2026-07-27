"""Gumbel-AlphaZero root action selection on the EXISTING az.pt (no retrain).

Danihelka et al. 2022, "Policy improvement by planning with Gumbel": at low
simulation budgets plain PUCT can fail to improve over the policy prior (visits
just track the prior — exactly what the autopsy box in CLAUDE.md observed at the
fatal states). Gumbel replaces the root rule with:
  1. Gumbel-Top-k: sample g(a)~Gumbel, take the top-m of (logit(a)+g(a)) as
     candidate actions (sampling without replacement, weighted by the prior).
  2. Sequential halving: split the sim budget into ceil(log2 m) phases; each
     phase gives every surviving candidate an equal share of visits, then keeps
     the top half by  g(a)+logit(a)+sigma(q_hat(a)).
  3. Return the last survivor (deterministic; no visit-count sampling).
Below the root we keep ordinary PUCT (the cheap, well-known "Gumbel root"
simplification; the paper's non-root rule is a further refinement).

This is a PURE ACT-TIME change — it reuses az.pt's trained policy+value heads,
so it measures whether better *search* (not a better net) lifts the low-sim
curve, which is the live-hint latency question.
"""
from __future__ import annotations
import math
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import torch  # noqa: E402

from src.ai.alphazero import (_Node, _evaluate, _puct_select,  # noqa: E402
                              _MinMaxStats)
from src.game import Game  # noqa: E402
from src.ai.policy import ActionFn  # noqa: E402


def _gumbel(rng: random.Random) -> float:
    u = rng.random()
    return -math.log(-math.log(u + 1e-12) + 1e-12)


def _simulate_forced(root: _Node, work: Game, net, forced_a: int,
                     c_puct: float, gamma: float, minmax: _MinMaxStats,
                     value_mode: str = "shaped") -> None:
    """One PUCT simulation whose FIRST action at the root is ``forced_a``."""
    node = root
    path: list[tuple[_Node, int]] = []
    leaf_value = 0.0
    first = True
    while node.expanded and not node.done:
        a = forced_a if first else _puct_select(node, c_puct, minmax)
        first = False
        path.append((node, a))
        child = node.children.get(a)
        if child is not None:
            node = child
            continue
        work.set_state(node.state)
        reward, _ = work.step(a)
        node.edge_reward[a] = reward
        child = _Node(work)
        node.children[a] = child
        leaf_value = 0.0 if child.done else _evaluate(child, work, net, value_mode)
        break
    g = leaf_value
    for parent, a in reversed(path):
        g = parent.edge_reward[a] + gamma * g
        parent.N[a] += 1
        parent.W[a] += g
        minmax.update(parent.W[a] / parent.N[a])


def gumbel_root_action(game: Game, net, sims: int, m: int = 8,
                       c_visit: float = 50.0, c_scale: float = 1.0,
                       c_puct: float = 1.5, gamma: float = 0.99,
                       rng: random.Random | None = None,
                       value_mode: str = "shaped") -> int:
    rng = rng or random
    work = game.clone()
    root = _Node(work)
    _evaluate(root, work, net, value_mode)
    legal = root.legal
    if len(legal) <= 1:
        return legal[0] if legal else 0

    obs = torch.as_tensor(work.get_observation(), dtype=torch.float32)
    with torch.no_grad():
        logits_full, _ = net(obs)
    logit = {a: float(logits_full[a]) for a in legal}
    gv = {a: _gumbel(rng) for a in legal}

    m = min(m, len(legal))
    cand = sorted(legal, key=lambda a: logit[a] + gv[a], reverse=True)[:m]
    minmax = _MinMaxStats()

    def sigma_score(a: int, max_n: int) -> float:
        n = root.N[a]
        q_hat = minmax.normalize(root.W[a] / n) if n else 0.0
        return gv[a] + logit[a] + (c_visit + max_n) * c_scale * q_hat

    phases = max(1, math.ceil(math.log2(m)))
    remaining = cand
    for _ in range(phases):
        per = max(1, sims // (phases * len(remaining)))
        for a in remaining:
            for _ in range(per):
                _simulate_forced(root, work, net, a, c_puct, gamma, minmax,
                                 value_mode)
        if len(remaining) == 1:
            break
        max_n = max((root.N[a] for a in cand if root.N[a]), default=1)
        remaining = sorted(remaining, key=lambda a: sigma_score(a, max_n),
                           reverse=True)[:max(1, len(remaining) // 2)]

    max_n = max((root.N[a] for a in cand if root.N[a]), default=1)
    return max(remaining, key=lambda a: sigma_score(a, max_n))


def gumbel_action_fn(net, simulations: int = 100, m: int = 8,
                     c_puct: float = 1.5, gamma: float = 0.99, seed: int = 0,
                     value_mode: str = "shaped") -> ActionFn:
    rng = random.Random(seed)

    def act(game: Game) -> int:
        if game.done:
            return 0
        return gumbel_root_action(game, net, simulations, m=m, c_puct=c_puct,
                                  gamma=gamma, rng=rng, value_mode=value_mode)
    return act
