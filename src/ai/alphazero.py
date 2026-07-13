"""AlphaZero-style PUCT search + self-play trainer (Phase 2 of the planner).

The pure-planning MCTS in :mod:`~src.ai.mcts` wins the full game but is slow:
it rebuilds a fresh tree with random/greedy *rollouts* every move. AlphaZero
replaces the two costly, hand-tuned parts of that search with a learned
:class:`~src.ai.network.ActorCritic`:

* the **rollout** becomes a single value-head evaluation of the leaf (``V(s)``,
  the return-to-go estimate), so no simulation to the end is needed;
* the **UCB1 exploration** becomes **PUCT**, whose exploration term is weighted
  by the policy head's prior ``P(a|s)`` — the net biases the search toward the
  moves it already likes, so far fewer simulations reach the same strength.

The net is trained by **self-play**: play games choosing moves in proportion to
the search's visit counts, then regress the policy head onto those visit
distributions (the search is a stronger policy than the raw net — this is the
policy-improvement step) and the value head onto the discounted return-to-go.

Two carry-overs from :mod:`~src.ai.mcts` that this game forces:

* **Shaped value, not win/loss.** Wins are far too deep for a fresh value head
  to label leaves usefully, so the value target is the *discounted shaped
  return-to-go* (same quantity PPO's value head already represents), which
  exposes ``clear`` (+10/flame) and the terminal signals. Because that return
  is unbounded and large (a full win is ~+470), PUCT normalises ``Q`` with a
  running :class:`_MinMaxStats` (MuZero's trick) so ``c_puct`` stays scaleless.
* **A curriculum.** An untrained value head is noise, so on the full game early
  self-play has no more signal than PPO did. Self-play therefore runs on the
  same difficulty ramp (:data:`~src.ai.train.DEFAULT_CURRICULUM`), promoting a
  stage once the search reliably wins it.

Everything is driven through the env API and the existing
:mod:`~src.ai.policy` helpers, so :func:`az_action_fn` is an
:data:`~src.ai.policy.ActionFn` that drops straight into ``baselines`` /
``play`` / ``evaluate`` like every other policy here.
"""
from __future__ import annotations

import math
import os
import random
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn, optim

from ..entity import Player
from ..game import Game
from .. import constant
from .network import ActorCritic, MASK_FILL
from .policy import (
    ActionFn,
    EVAL_SEEDS,
    format_reward_parts,
    save_policy,
)


# Value-target scale. The value head is trained on shaped returns, which run to
# O(100) (REWARD_WIN alone is +100); squared, that MSE is O(10^4), whose
# gradient — after the global grad-norm clip — swamps the O(1) policy
# cross-entropy and leaves the policy head at its uniform init (measured: prior
# entropy within 1% of uniform, p_loss flat, so PUCT loses its prior and
# degenerates to unguided search). We divide the value *target* by VALUE_SCALE
# so both losses are O(1) and share the trunk's gradient fairly; the head's
# output is then in units of VALUE_SCALE, so the search multiplies it back up
# to raw reward units before mixing it with per-edge rewards in the backup.
VALUE_SCALE = 100.0


# Value-target modes (the `--value-target` A/B). "shaped" is the historical
# recipe: the value head regresses the discounted *shaped* return-to-go and the
# search backs up per-step shaped rewards. Both 64- and 128-wide nets plateau at
# it (= MCTS ~50%) because that target is really a *flame-count* predictor —
# clears (+10 each, up to +360) swamp REWARD_WIN (+100), so a leaf that "clears
# 30 then dies" scores nearly as high as one that wins, and PUCT can't rank the
# winning branch above "die late clearing more".
#
# "winloss" replaced the target and the search backup with the discounted *game
# outcome* (+1 win / -1 loss, terminal only). Measured (2026-07-13): it climbs
# the curriculum but *collapses on the full game* — self-play never wins there,
# so every full-game target is -1, the value head confidently predicts "all
# lost", and with the dense shaped guidance gone the search has no gradient to
# find flames and degenerates to passing turns (stall -480, 0% for 26 straight
# iters). Sparse outcome removes the exploration signal the game needs.
#
# "blend" keeps the dense shaped reward (so the search still explores toward
# flames) and *adds* a large terminal outcome bonus ±BLEND_TERMINAL_BONUS to
# both the search edges and the value target — enough to lift a win clear of the
# best near-win, which is the separation shaped alone lacks. Dense guidance of
# shaped + the win/near-win ranking winloss was after, without winloss's sparsity.
# Measured (2026-07-13): blend avoids winloss's passive collapse (stall stayed
# modest, clear ~100, self-play won the full game ~12%) but at equal budget
# scores 0/6 — *worse* than shaped's 3/6. The terminal ±300 spike folded into
# the *single* shaped value target poisons the value estimate near a terminal
# (a state one avoidable-death away is dragged toward -300·gamma), so the net
# learns to avoid death without learning to finish; it clears fewer flames than
# plain shaped.
#
# "twohead" is the fix for that: keep the shaped value target *clean* (no
# terminal spike) on one value head, and learn P(win) (the ±1 outcome) on a
# *second* head. The search combines them at a leaf as
# ``V_shaped·VALUE_SCALE + TWOHEAD_WIN_WEIGHT·P(win)`` and adds the same
# ±TWOHEAD_WIN_WEIGHT to a terminal edge, so a win still outranks the best
# near-win (the winloss/blend goal) — but the ranking pressure lives in the
# bounded P(win) head and never distorts the dense shaped estimate that drives
# exploration. Needs a fresh run: the value head has 2 outputs, so the width is
# baked into the checkpoint (value_dim), like `hidden`.
VALUE_MODES = ("shaped", "winloss", "blend", "twohead")

# Raw-reward bonus added to the terminal step in "blend" mode (+win / -loss).
# Sized above the clear-count spread (~±360) so the outcome dominates the value
# head's clear-count noise — the whole point is that a win outranks "clear 32
# then die". The value scale is raised in step so the (larger) target stays O(1).
BLEND_TERMINAL_BONUS = 300.0
BLEND_VALUE_SCALE = 300.0

# "twohead" mixing weight λ: raw-reward weight on the P(win) head in the leaf
# value and on the terminal edge. Same magnitude as BLEND_TERMINAL_BONUS (above
# the clear spread) so a near-certain win lifts a leaf clear of the best
# near-win. The shaped head keeps VALUE_SCALE; the P(win) head is O(1) already.
TWOHEAD_WIN_WEIGHT = 300.0


def _value_scale(value_mode: str) -> float:
    """Target/leaf scale for a value mode.

    Shaped returns are O(100) so they are divided by :data:`VALUE_SCALE` to keep
    the value loss O(1); outcome targets are already O(1), so no scaling; blend
    returns reach ~±770 (shaped + the terminal bonus) so they use a larger scale.
    """
    if value_mode == "winloss":
        return 1.0
    if value_mode == "blend":
        return BLEND_VALUE_SCALE
    return VALUE_SCALE


# --- Q-value normalisation (MuZero-style) -----------------------------------

class _MinMaxStats:
    """Running min/max of backed-up Q-values to normalise PUCT exploitation.

    Shaped returns here span roughly ``[-40, +470]``; mapping the observed
    range onto ``[0, 1]`` keeps the PUCT exploration constant ``c_puct``
    independent of the reward scale.
    """

    __slots__ = ("lo", "hi")

    def __init__(self) -> None:
        self.lo = math.inf
        self.hi = -math.inf

    def update(self, value: float) -> None:
        self.lo = min(self.lo, value)
        self.hi = max(self.hi, value)

    def normalize(self, value: float) -> float:
        if self.hi > self.lo:
            return (value - self.lo) / (self.hi - self.lo)
        return value


# --- Search tree ------------------------------------------------------------

class _Node:
    """A game state in the PUCT tree (a snapshot plus per-edge statistics).

    Edge stats are kept as dicts keyed by action index over the legal actions
    at this state. A node is *expanded* the first time it is evaluated by the
    net (priors filled in); terminal nodes are never expanded.
    """

    __slots__ = ("state", "done", "won", "expanded", "legal",
                 "P", "N", "W", "edge_reward", "children")

    def __init__(self, game: Game) -> None:
        self.state = game.get_state()
        self.done = game.done
        self.won = game.won
        self.expanded = False
        self.legal: list[int] = []
        self.P: dict[int, float] = {}
        self.N: dict[int, int] = {}
        self.W: dict[int, float] = {}
        self.edge_reward: dict[int, float] = {}
        self.children: dict[int, "_Node"] = {}


def _leaf_value(value: torch.Tensor, value_mode: str) -> float:
    """Raw-reward leaf value from the net's value output for ``value_mode``.

    Single-head modes ("shaped"/"winloss"/"blend") scale the scalar head back up
    to raw reward units. "twohead" combines the two heads:
    ``V_shaped·VALUE_SCALE + TWOHEAD_WIN_WEIGHT·P(win)`` — the dense shaped
    estimate plus a bounded win-ranking term.
    """
    if value_mode == "twohead":
        return (float(value[0]) * VALUE_SCALE
                + TWOHEAD_WIN_WEIGHT * float(value[1]))
    return float(value) * _value_scale(value_mode)


@torch.no_grad()
def _evaluate(node: _Node, work: Game, net: ActorCritic,
              value_mode: str = "shaped") -> float:
    """Expand ``node`` with the net's masked priors; return its value estimate.

    ``work`` must already be at ``node``'s state. Fills ``node.P/N/W`` over
    the legal actions and marks it expanded; returns ``V(s)`` (the
    return-to-go / outcome estimate used as the leaf value in backup).
    """
    obs = torch.as_tensor(work.get_observation(), dtype=torch.float32)
    mask = torch.as_tensor(work.legal_actions(), dtype=torch.bool)
    logits, value = net(obs)
    probs = torch.softmax(logits.masked_fill(~mask, MASK_FILL), dim=-1)
    node.legal = [i for i, ok in enumerate(mask.tolist()) if ok]
    for a in node.legal:
        node.P[a] = float(probs[a])
        node.N[a] = 0
        node.W[a] = 0.0
    node.expanded = True
    # Head(s) trained in scaled units; the backup mixes this leaf value with
    # per-edge rewards in raw units, so convert back to raw units here.
    return _leaf_value(value, value_mode)


def _puct_select(node: _Node, c_puct: float, minmax: _MinMaxStats) -> int:
    """Pick the action maximising the PUCT score at ``node``.

    ``Q(a) + c_puct·P(a)·sqrt(1+ΣN)/(1+N(a))`` with ``Q`` min-max normalised.
    Unvisited edges take ``Q=0`` and rely on the prior-weighted exploration
    term, so the first simulations follow the policy head.
    """
    total = sum(node.N.values())
    sqrt_total = math.sqrt(1 + total)
    best_a = node.legal[0]
    best_score = -math.inf
    for a in node.legal:
        n = node.N[a]
        q = node.W[a] / n if n else 0.0
        u = c_puct * node.P[a] * sqrt_total / (1 + n)
        score = minmax.normalize(q) + u
        if score > best_score:
            best_score, best_a = score, a
    return best_a


def _simulate(root: _Node, work: Game, net: ActorCritic,
              c_puct: float, gamma: float, minmax: _MinMaxStats,
              value_mode: str = "shaped") -> None:
    """One PUCT iteration: select to a leaf, expand+evaluate, back up return.

    Selection uses only cached edge stats (no stepping) until it chooses an
    action with no child yet; that action is expanded by stepping ``work`` once
    and evaluating the new leaf with the net. The discounted return
    (edge rewards + leaf value) then updates every edge on the path.

    In ``"winloss"`` mode the per-edge reward is the discounted *outcome*
    (``+1``/``-1`` on the terminal step, ``0`` otherwise) rather than the shaped
    step reward, and the leaf value is the net's P(win) at unit scale — so the
    whole backup measures "does this line win", matching the value target.
    In ``"blend"`` mode edges stay shaped (dense guidance) but a terminal step
    gets an extra ``±BLEND_TERMINAL_BONUS`` so wins outrank near-wins.
    ``"twohead"`` likewise keeps shaped edges and adds ``±TWOHEAD_WIN_WEIGHT``
    on a terminal step, matching the P(win) leaf term's scale.
    """
    node = root
    path: list[tuple[_Node, int]] = []
    leaf_value = 0.0

    while node.expanded and not node.done:
        a = _puct_select(node, c_puct, minmax)
        path.append((node, a))
        child = node.children.get(a)
        if child is not None:
            node = child
            continue
        # Expand this edge: step the working game once and evaluate the leaf.
        work.set_state(node.state)
        reward, _ = work.step(a)
        if value_mode == "winloss":
            reward = (1.0 if work.won else -1.0) if work.done else 0.0
        elif value_mode == "blend" and work.done:
            reward += BLEND_TERMINAL_BONUS if work.won else -BLEND_TERMINAL_BONUS
        elif value_mode == "twohead" and work.done:
            reward += TWOHEAD_WIN_WEIGHT if work.won else -TWOHEAD_WIN_WEIGHT
        node.edge_reward[a] = reward
        child = _Node(work)
        node.children[a] = child
        leaf_value = (0.0 if child.done
                      else _evaluate(child, work, net, value_mode))
        break
    # Loop fell through without expanding (terminal reached): leaf stays 0.

    # Backup: fold edge rewards into the leaf value from the bottom up.
    g = leaf_value
    for parent, a in reversed(path):
        g = parent.edge_reward[a] + gamma * g
        parent.N[a] += 1
        parent.W[a] += g
        minmax.update(parent.W[a] / parent.N[a])


def _visit_counts(root: _Node) -> dict[int, int]:
    return {a: root.N[a] for a in root.legal}


# --- ActionFn (evaluation / play) -------------------------------------------

def az_action_fn(net: ActorCritic, simulations: int = 100, c_puct: float = 1.5,
                 gamma: float = 0.99, seed: int = 0,
                 value_mode: str = "shaped") -> ActionFn:
    """Wrap ``net`` as an :data:`ActionFn` that picks moves by PUCT search.

    Deterministic (most-visited root action), no root noise — the evaluation /
    viewer form. Reuses the same search as self-play with exploration disabled.
    ``value_mode`` must match the net's training mode (a P(win) head searched
    with shaped backup, or vice-versa, is incoherent).
    """
    rng = random.Random(seed)

    def act(game: Game) -> int:
        if game.done:
            return 0
        root, _ = _search(game, net, simulations, c_puct, gamma, rng,
                          value_mode=value_mode)
        counts = _visit_counts(root)
        if not counts:
            legal = game.legal_actions()
            return next((i for i, ok in enumerate(legal) if ok), 0)
        return max(counts, key=lambda a: counts[a])

    return act


def _search(game: Game, net: ActorCritic, simulations: int, c_puct: float,
            gamma: float, rng: random.Random,
            dirichlet_alpha: float = 0.0,
            value_mode: str = "shaped") -> tuple[_Node, _MinMaxStats]:
    """Run ``simulations`` PUCT iterations rooted at ``game``; return root."""
    work = game.clone()
    root = _Node(work)
    _evaluate(root, work, net, value_mode)
    if dirichlet_alpha > 0.0 and len(root.legal) > 1:
        samples = [rng.gammavariate(dirichlet_alpha, 1.0) for _ in root.legal]
        s = sum(samples) or 1.0
        for a, g in zip(root.legal, samples):
            root.P[a] = 0.75 * root.P[a] + 0.25 * (g / s)
    minmax = _MinMaxStats()
    for _ in range(simulations):
        _simulate(root, work, net, c_puct, gamma, minmax, value_mode)
    return root, minmax


# --- Self-play --------------------------------------------------------------

@dataclass
class _Sample:
    """One training example: state, legal mask, search policy, value target(s).

    ``value`` is the value-head target (return-to-go); ``value_win`` is the
    second head's P(win) target, used only by ``value_mode="twohead"`` (left at
    ``0.0`` and ignored by the single-head modes).
    """

    obs: list[float]
    mask: list[bool]
    pi: list[float]           # visit distribution over ALL actions
    value: float = 0.0        # filled with return-to-go after the game
    value_win: float = 0.0    # twohead only: discounted ±1 outcome target


def _discounted_returns(stream: list[float], gamma: float) -> list[float]:
    """Return-to-go of ``stream`` under ``gamma`` (same length as ``stream``)."""
    g = 0.0
    returns: list[float] = [0.0] * len(stream)
    for t in reversed(range(len(stream))):
        g = stream[t] + gamma * g
        returns[t] = g
    return returns


def _fill_value_targets(samples: list["_Sample"], rewards: list[float],
                        won: bool, gamma: float, value_mode: str) -> None:
    """Fill each sample's value target(s) with the discounted return-to-go.

    ``"shaped"`` discounts the per-step shaped ``rewards``; ``"winloss"``
    discounts a stream that is ``0`` everywhere except the terminal step, which
    carries the game outcome (``+1`` won / ``-1`` otherwise) — so a sample's
    target is ``±gamma**(plies-to-end)``, a pure P(win) signal. ``"blend"``
    keeps the shaped stream but adds ``±BLEND_TERMINAL_BONUS`` to the terminal
    step, so the target is the dense shaped return lifted by a strong outcome.
    ``"twohead"`` fills *both* targets: ``value`` = the *clean* shaped return
    (no terminal spike) and ``value_win`` = the discounted ±1 outcome, so the
    two heads learn the dense estimate and P(win) independently.
    """
    outcome = 1.0 if won else -1.0
    if value_mode == "twohead":
        shaped_returns = _discounted_returns(rewards, gamma)
        win_stream = [0.0] * len(rewards)
        if win_stream:
            win_stream[-1] = outcome
        win_returns = _discounted_returns(win_stream, gamma)
        for i, sample in enumerate(samples):
            sample.value = shaped_returns[i] if i < len(shaped_returns) else 0.0
            sample.value_win = win_returns[i] if i < len(win_returns) else 0.0
        return

    if value_mode == "winloss":
        stream = [0.0] * len(rewards)
        if stream:
            stream[-1] = outcome
    elif value_mode == "blend":
        stream = list(rewards)
        if stream:
            stream[-1] += BLEND_TERMINAL_BONUS if won else -BLEND_TERMINAL_BONUS
    else:
        stream = rewards
    returns = _discounted_returns(stream, gamma)
    for i, sample in enumerate(samples):
        sample.value = returns[i] if i < len(returns) else 0.0


def self_play_game(net: ActorCritic, seed: int, simulations: int,
                   c_puct: float,
                   gamma: float, rng: random.Random, num_actions: int,
                   num_waves: int | None = None,
                   flames_per_wave: int | None = None,
                   spawn_radius: int | None = None,
                   spawn_formation: str | None = None,
                   dirichlet_alpha: float = 0.3,
                   temp_moves: int = 12,
                   value_mode: str = "shaped",
                   max_steps: int = 500) -> tuple[list[_Sample], bool]:
    """Play one self-play game; return ``(samples, won)``.

    Each move runs a PUCT search with root Dirichlet noise. The move is sampled
    from the visit counts (temperature 1 for the first ``temp_moves`` plies,
    greedy after) so early exploration is broad but the endgame is sharp. The
    per-move visit distribution is the policy target; value targets are the
    discounted return-to-go (shaped) or game outcome (``value_mode="winloss"``),
    filled in once the game ends.
    """
    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed,
                num_waves=num_waves, flames_per_wave=flames_per_wave,
                spawn_radius=spawn_radius, spawn_formation=spawn_formation)
    samples: list[_Sample] = []
    rewards: list[float] = []

    for ply in range(max_steps):
        if game.done:
            break
        root, _ = _search(game, net, simulations, c_puct, gamma, rng,
                          dirichlet_alpha=dirichlet_alpha,
                          value_mode=value_mode)
        counts = _visit_counts(root)
        if not counts:
            legal = game.legal_actions()
            action = next((i for i, ok in enumerate(legal) if ok), 0)
            reward, _ = game.step(action)
            rewards.append(reward)
            continue
        pi = [0.0] * num_actions
        total = sum(counts.values())
        for a, n in counts.items():
            pi[a] = n / total
        samples.append(_Sample(obs=game.get_observation(),
                               mask=game.legal_actions(), pi=pi))
        if ply < temp_moves:
            actions = list(counts)
            weights = [counts[a] for a in actions]
            action = rng.choices(actions, weights=weights, k=1)[0]
        else:
            action = max(counts, key=lambda a: counts[a])
        reward, _ = game.step(action)
        rewards.append(reward)

    # Value target for every recorded state (samples are recorded before their
    # move, so sample i owns rewards[i:]).
    _fill_value_targets(samples, rewards, game.won, gamma, value_mode)
    return samples, game.won


def mcts_bootstrap_game(seed: int, simulations: int, gamma: float,
                        rng: random.Random, num_actions: int,
                        num_waves: int | None = None,
                        flames_per_wave: int | None = None,
                        spawn_radius: int | None = None,
                        spawn_formation: str | None = None,
                        rollout_depth: int = 16,
                        value_mode: str = "shaped",
                        max_steps: int = 500) -> tuple[list[_Sample], bool]:
    """Play one game with **MCTS** and record it as AlphaZero training samples.

    Self-play bootstraps the net off its own search, which only works while that
    search sometimes wins: on the full game ``selfplay_win`` sits at 0%, so the
    policy target is the visit distribution of a player who always dies and the
    value head only ever fits losing shaped returns. MCTS wins ~50% of full
    games at 300 sims, so seeding the buffer with *its* trajectories gives the
    net winning lines to imitate — the missing ingredient, not more iterations.

    Same ``_Sample`` layout as :func:`self_play_game` (visit distribution as the
    policy target, discounted return-to-go as the value target); no network is
    involved, and moves are greedy on visits (no Dirichlet noise — these games
    are meant to be strong, not exploratory).
    """
    from .mcts import mcts_root_visits

    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed,
                num_waves=num_waves, flames_per_wave=flames_per_wave,
                spawn_radius=spawn_radius, spawn_formation=spawn_formation)
    samples: list[_Sample] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        if game.done:
            break
        counts = mcts_root_visits(game, simulations, rng,
                                  rollout_depth=rollout_depth, gamma=gamma)
        if not counts:
            break
        pi = [0.0] * num_actions
        total = sum(counts.values())
        for a, n in counts.items():
            pi[a] = n / total
        samples.append(_Sample(obs=game.get_observation(),
                               mask=game.legal_actions(), pi=pi))
        action = max(counts, key=lambda a: counts[a])
        reward, _ = game.step(action)
        rewards.append(reward)

    _fill_value_targets(samples, rewards, game.won, gamma, value_mode)
    return samples, game.won


# --- Trainer ----------------------------------------------------------------

@dataclass
class AZConfig:
    """Hyperparameters for an AlphaZero self-play run."""

    iterations: int = 100           # self-play + update cycles this session
    games_per_iter: int = 8         # self-play games collected per iteration
    simulations: int = 80           # PUCT sims per move during self-play
    bootstrap_games: int = 0        # MCTS games seeded into the buffer, once
    bootstrap_simulations: int = 300  # MCTS sims/move for those games
    eval_simulations: int = 120     # PUCT sims per move at evaluation
    c_puct: float = 1.5
    gamma: float = 0.99
    dirichlet_alpha: float = 0.3    # root exploration noise
    temp_moves: int = 12            # plies sampled from visits before greedy
    buffer_size: int = 20_000       # replay buffer capacity (samples)
    # Persistent win buffer (the "buffer lever"). The main buffer is a rolling
    # window: wins are ~0.3% of samples and evict within ~40 iters, so their
    # signal is washed out (measured: reward/ranking variants clear ~24 flames
    # but win 0/12 — the net learns to play, never to finish). A separate
    # win-only buffer that never evicts, from which a fixed fraction of every
    # minibatch is drawn, keeps winning trajectories (self-play *and* the MCTS
    # bootstrap's) permanently represented. 0.0 = off (byte-for-byte the old
    # behaviour: no win buffer maintained, minibatch is the plain rolling draw).
    win_buffer_fraction: float = 0.0  # share of each batch from the win buffer
    win_buffer_size: int = 40_000     # win-buffer capacity (>> a run's wins)
    batch_size: int = 128           # SGD minibatch (skips update until filled)
    update_steps: int = 64          # SGD minibatches per iteration
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_coef: float = 1.0         # value-loss weight relative to policy loss
    value_mode: str = "shaped"      # "shaped"|"winloss"|"blend"|"twohead"
    hidden: int = 64
    seed: int = 1
    eval_every: int = 5
    eval_seeds: list[int] = field(default_factory=lambda: EVAL_SEEDS[:16])
    device: str = "cpu"
    out_path: str = ""
    checkpoint_path: str = ""
    resume_from: str = ""
    curriculum: tuple[tuple[int | None, int | None, int | None], ...] = ()
    promote_win_rate: float = 0.6


def _save_az_checkpoint(path: str, net: ActorCritic,
                        optimizer: optim.Optimizer,
                        iteration: int, best_score: float,
                        stage_idx: int) -> None:
    """Resumable AZ training state (weights + optimizer + progress)."""
    torch.save(
        {
            "state_dict": net.state_dict(),
            "obs_size": net.obs_size,
            "num_actions": net.num_actions,
            "hidden": net.hidden,
            "value_dim": net.value_dim,
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "best_score": best_score,
            "stage_idx": stage_idx,
        },
        path,
    )


def _update(net: ActorCritic, optimizer: optim.Optimizer,
            buffer: deque, cfg: AZConfig, rng: random.Random,
            win_buffer: deque | None = None) -> tuple:
    """Run ``cfg.update_steps`` SGD steps; return mean (p_loss, v_loss).

    Policy loss is the cross-entropy of the net's masked policy against the
    search visit distribution; value loss is MSE against the return-to-go. In
    ``"twohead"`` mode the value loss sums the two heads' MSEs (shaped return on
    head 0, P(win) on head 1); the reported ``v_loss`` is that sum.

    When ``win_buffer`` is non-empty and ``cfg.win_buffer_fraction > 0``, that
    fraction of each minibatch is drawn (with replacement) from the persistent
    win buffer and the rest from the rolling ``buffer``; with fraction 0 or an
    empty win buffer the draw is the plain rolling sample (unchanged).
    """
    if len(buffer) < cfg.batch_size:
        return float("nan"), float("nan")
    net.train()
    twohead = cfg.value_mode == "twohead"
    data = list(buffer)
    win_data = list(win_buffer) if win_buffer else []
    n_win = round(cfg.batch_size * cfg.win_buffer_fraction) if win_data else 0
    n_main = cfg.batch_size - n_win
    p_tot = v_tot = 0.0
    for _ in range(cfg.update_steps):
        batch = rng.sample(data, n_main)
        if n_win:
            batch = batch + rng.choices(win_data, k=n_win)
        dev = cfg.device
        obs = torch.tensor([s.obs for s in batch], dtype=torch.float32,
                           device=dev)
        mask = torch.tensor([s.mask for s in batch], dtype=torch.bool,
                            device=dev)
        pi = torch.tensor([s.pi for s in batch], dtype=torch.float32,
                          device=dev)
        logits, value = net(obs)
        logp = torch.log_softmax(logits.masked_fill(~mask, MASK_FILL), dim=-1)
        policy_loss = -(pi * logp).sum(dim=-1).mean()
        if twohead:
            # Head 0 regresses the clean shaped return (in VALUE_SCALE units);
            # head 1 regresses the discounted ±1 outcome (already O(1)).
            z_shaped = torch.tensor([s.value / VALUE_SCALE for s in batch],
                                    dtype=torch.float32, device=dev)
            z_win = torch.tensor([s.value_win for s in batch],
                                 dtype=torch.float32, device=dev)
            value_loss = (F.mse_loss(value[:, 0], z_shaped)
                          + F.mse_loss(value[:, 1], z_win))
        else:
            z = torch.tensor([s.value / _value_scale(cfg.value_mode)
                              for s in batch], dtype=torch.float32, device=dev)
            value_loss = F.mse_loss(value, z)
        loss = policy_loss + cfg.value_coef * value_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        p_tot += policy_loss.item()
        v_tot += value_loss.item()
    return p_tot / cfg.update_steps, v_tot / cfg.update_steps


def train(cfg: AZConfig) -> ActorCritic:
    """Run AlphaZero self-play for ``cfg.iterations`` and return the network.

    Saves the best-by-eval policy to ``cfg.out_path`` (as :func:`save_policy`,
    so ``play``/``baselines`` load it like any policy) and a resumable AZ
    checkpoint to ``cfg.checkpoint_path``.
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    obs_size = Game.observation_size()
    num_actions = Game.num_actions()
    # "twohead" needs a second value output (shaped + P(win)); baked into the
    # checkpoint like `hidden`, so it cannot resume a single-head checkpoint.
    value_dim = 2 if cfg.value_mode == "twohead" else 1
    net = ActorCritic(obs_size, num_actions, cfg.hidden,
                      value_dim=value_dim).to(cfg.device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)

    iter_offset = 0
    best_score = float("-inf")
    stage_idx = 0
    if cfg.resume_from:
        if not os.path.exists(cfg.resume_from):
            raise SystemExit(f"No checkpoint to resume at {cfg.resume_from}.")
        ckpt = torch.load(cfg.resume_from, map_location=cfg.device,
                          weights_only=True)
        if (ckpt["obs_size"] != obs_size
                or ckpt["num_actions"] != num_actions
                or ckpt.get("hidden") != cfg.hidden
                or ckpt.get("value_dim", 1) != value_dim):
            raise SystemExit("Checkpoint shape mismatch — train fresh.")
        net.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        iter_offset = ckpt.get("iteration", 0)
        best_score = ckpt.get("best_score", float("-inf"))
        stage_idx = ckpt.get("stage_idx", 0)
        print(f"Resumed from {cfg.resume_from}: iter {iter_offset}, "
              f"best {best_score:.2f}, stage {stage_idx}", flush=True)

    stages: list[tuple[int | None, int | None, int | None]] = (
        list(cfg.curriculum) if cfg.curriculum else [(None, None, None)])
    stage_idx = min(stage_idx, len(stages) - 1)
    buffer: deque = deque(maxlen=cfg.buffer_size)
    # Persistent win-only buffer (the buffer lever); empty/unused when the
    # fraction is 0. Fed by winning self-play AND bootstrap games below.
    win_buffer: deque = deque(maxlen=cfg.win_buffer_size)
    use_win_buffer = cfg.win_buffer_fraction > 0.0

    def bootstrap(idx: int) -> None:
        """Seed the buffer with MCTS games at stage ``idx``.

        Run at start *and on every promotion*: a stage change swaps the game
        under the net, and the buffer is a rolling window, so the winning lines
        seeded for one stage are gone (and wrong) by the time the next is
        reached. The stage that actually needs this is the last one, where
        self-play never wins on its own.
        """
        if not cfg.bootstrap_games:
            return
        nw, fpw, sr, fm = (tuple(stages[idx]) + (None,))[:4]
        wins = 0
        for _ in range(cfg.bootstrap_games):
            samples, won = mcts_bootstrap_game(
                seed=rng.randrange(1 << 30),
                simulations=cfg.bootstrap_simulations, gamma=cfg.gamma,
                rng=rng, num_actions=num_actions,
                num_waves=nw, flames_per_wave=fpw, spawn_radius=sr,
                spawn_formation=fm, value_mode=cfg.value_mode)
            buffer.extend(samples)
            if use_win_buffer and won:
                win_buffer.extend(samples)
            wins += int(won)
        print(f"Bootstrapped {cfg.bootstrap_games} MCTS games at "
              f"{cfg.bootstrap_simulations} sims (stage {idx}): "
              f"{wins} won, buffer now {len(buffer)} samples", flush=True)

    bootstrap(stage_idx)

    for iteration in range(1, cfg.iterations + 1):
        total_iter = iter_offset + iteration
        # Stages are (waves, flames, radius) or (waves, flames, radius, shape).
        nw, fpw, sr, fm = (tuple(stages[stage_idx]) + (None,))[:4]

        # --- Self-play -------------------------------------------------
        wins = 0
        for g in range(cfg.games_per_iter):
            samples, won = self_play_game(
                net, seed=rng.randrange(1 << 30),
                simulations=cfg.simulations, c_puct=cfg.c_puct,
                gamma=cfg.gamma, rng=rng, num_actions=num_actions,
                num_waves=nw, flames_per_wave=fpw, spawn_radius=sr,
                spawn_formation=fm,
                dirichlet_alpha=cfg.dirichlet_alpha, temp_moves=cfg.temp_moves,
                value_mode=cfg.value_mode)
            buffer.extend(samples)
            if use_win_buffer and won:
                win_buffer.extend(samples)
            wins += int(won)

        # --- Update ----------------------------------------------------
        p_loss, v_loss = _update(net, optimizer, buffer, cfg, rng,
                                 win_buffer if use_win_buffer else None)

        # --- Eval / checkpoint -----------------------------------------
        sp_win = wins / cfg.games_per_iter
        wbuf_tag = f" | wbuf {len(win_buffer):5d}" if use_win_buffer else ""
        if iteration % cfg.eval_every == 0 or iteration == cfg.iterations:
            net.eval()
            act = az_action_fn(net, simulations=cfg.eval_simulations,
                               c_puct=cfg.c_puct, gamma=cfg.gamma,
                               value_mode=cfg.value_mode)
            mean_reward, win_rate, parts = _eval_action_fn(
                act, cfg.eval_seeds, nw, fpw, sr, fm)
            stage_tag = (f" | stage {stage_idx}/{len(stages) - 1} "
                         f"(w={nw},f={fpw},r={sr}"
                         f"{',' + fm if fm else ''})" if len(stages) > 1 else "")
            print(
                f"iter {total_iter:4d}{stage_tag} | buffer {len(buffer):6d}"
                f"{wbuf_tag} | "
                f"selfplay_win {sp_win:5.1%} | p_loss {p_loss:6.3f} | "
                f"v_loss {v_loss:8.2f} | eval_win {win_rate:5.1%} | "
                f"eval_reward {mean_reward:7.2f}\n"
                f"           reward parts: {format_reward_parts(parts)}",
                flush=True)
            # Lexicographic: stage, then win rate, then shaped reward. The
            # win-rate band must exceed the reward's whole range or a policy
            # that never wins can outscore one that does — with `win_rate*1000`
            # a single win in 16 (+62.5) lost to an +80 swing in shaped reward,
            # and a 0%-win policy overwrote the only full-game winner we had.
            score = (stage_idx * 10_000_000
                     + win_rate * 100_000
                     + mean_reward)
            if cfg.out_path and score > best_score:
                best_score = score
                save_policy(net, cfg.out_path)
            if (win_rate >= cfg.promote_win_rate
                    and stage_idx < len(stages) - 1):
                stage_idx += 1
                print(f"           >> promoted to stage {stage_idx} "
                      f"{stages[stage_idx]}", flush=True)
                bootstrap(stage_idx)
            if cfg.checkpoint_path:
                _save_az_checkpoint(cfg.checkpoint_path, net, optimizer,
                                    total_iter, best_score, stage_idx)
        else:
            print(f"iter {total_iter:4d} | buffer {len(buffer):6d}"
                  f"{wbuf_tag} | "
                  f"selfplay_win {sp_win:5.1%} | p_loss {p_loss:6.3f} | "
                  f"v_loss {v_loss:8.2f}", flush=True)

    if cfg.out_path and best_score == float("-inf"):
        save_policy(net, cfg.out_path)
    if cfg.checkpoint_path:
        _save_az_checkpoint(cfg.checkpoint_path, net, optimizer,
                            iter_offset + cfg.iterations, best_score,
                            stage_idx)
    return net


def _eval_action_fn(act: ActionFn, seeds: list[int],
                    num_waves: int | None, flames_per_wave: int | None,
                    spawn_radius: int | None,
                    spawn_formation: str | None = None) -> tuple:
    """Mean reward, win rate and parts for an ActionFn over ``seeds``."""
    from .policy import run_episode_decomposed
    from ..game import REWARD_PARTS
    results = [run_episode_decomposed(act, s, num_waves=num_waves,
                                      flames_per_wave=flames_per_wave,
                                      spawn_radius=spawn_radius,
                                      spawn_formation=spawn_formation)
               for s in seeds]
    n = len(seeds)
    mean_reward = sum(r[0] for r in results) / n
    win_rate = sum(1 for r in results if r[1]) / n
    parts = {p: sum(r[3][p] for r in results) / n for p in REWARD_PARTS}
    return mean_reward, win_rate, parts
