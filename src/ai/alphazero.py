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


@torch.no_grad()
def _evaluate(node: _Node, work: Game, net: ActorCritic) -> float:
    """Expand ``node`` with the net's masked priors; return its value estimate.

    ``work`` must already be at ``node``'s state. Fills ``node.P/N/W`` over
    the legal actions and marks it expanded; returns ``V(s)`` (the
    return-to-go estimate used as the leaf value in backup).
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
    # Head is trained in VALUE_SCALE units; the backup mixes this leaf value
    # with per-edge rewards in raw units, so scale it back up here.
    return float(value) * VALUE_SCALE


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
              c_puct: float, gamma: float, minmax: _MinMaxStats) -> None:
    """One PUCT iteration: select to a leaf, expand+evaluate, back up return.

    Selection uses only cached edge stats (no stepping) until it chooses an
    action with no child yet; that action is expanded by stepping ``work`` once
    and evaluating the new leaf with the net. The discounted return
    (edge rewards + leaf value) then updates every edge on the path.
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
        node.edge_reward[a] = reward
        child = _Node(work)
        node.children[a] = child
        leaf_value = 0.0 if child.done else _evaluate(child, work, net)
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
                 gamma: float = 0.99, seed: int = 0) -> ActionFn:
    """Wrap ``net`` as an :data:`ActionFn` that picks moves by PUCT search.

    Deterministic (most-visited root action), no root noise — the evaluation /
    viewer form. Reuses the same search as self-play with exploration disabled.
    """
    rng = random.Random(seed)

    def act(game: Game) -> int:
        if game.done:
            return 0
        root, _ = _search(game, net, simulations, c_puct, gamma, rng)
        counts = _visit_counts(root)
        if not counts:
            legal = game.legal_actions()
            return next((i for i, ok in enumerate(legal) if ok), 0)
        return max(counts, key=lambda a: counts[a])

    return act


def _search(game: Game, net: ActorCritic, simulations: int, c_puct: float,
            gamma: float, rng: random.Random,
            dirichlet_alpha: float = 0.0) -> tuple[_Node, _MinMaxStats]:
    """Run ``simulations`` PUCT iterations rooted at ``game``; return root."""
    work = game.clone()
    root = _Node(work)
    _evaluate(root, work, net)
    if dirichlet_alpha > 0.0 and len(root.legal) > 1:
        samples = [rng.gammavariate(dirichlet_alpha, 1.0) for _ in root.legal]
        s = sum(samples) or 1.0
        for a, g in zip(root.legal, samples):
            root.P[a] = 0.75 * root.P[a] + 0.25 * (g / s)
    minmax = _MinMaxStats()
    for _ in range(simulations):
        _simulate(root, work, net, c_puct, gamma, minmax)
    return root, minmax


# --- Self-play --------------------------------------------------------------

@dataclass
class _Sample:
    """One training example: state, legal mask, search policy, value target."""

    obs: list[float]
    mask: list[bool]
    pi: list[float]           # visit distribution over ALL actions
    value: float = 0.0        # filled with return-to-go after the game


def self_play_game(net: ActorCritic, seed: int, simulations: int,
                   c_puct: float,
                   gamma: float, rng: random.Random, num_actions: int,
                   num_waves: int | None = None,
                   flames_per_wave: int | None = None,
                   spawn_radius: int | None = None,
                   spawn_formation: str | None = None,
                   dirichlet_alpha: float = 0.3,
                   temp_moves: int = 12,
                   max_steps: int = 500) -> tuple[list[_Sample], bool]:
    """Play one self-play game; return ``(samples, won)``.

    Each move runs a PUCT search with root Dirichlet noise. The move is sampled
    from the visit counts (temperature 1 for the first ``temp_moves`` plies,
    greedy after) so early exploration is broad but the endgame is sharp. The
    per-move visit distribution is the policy target; value targets are the
    discounted return-to-go, filled in once the game ends.
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
                          dirichlet_alpha=dirichlet_alpha)
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

    # Discounted return-to-go as the value target for every recorded state.
    # Samples are recorded before their move, so sample i owns rewards[i:].
    g = 0.0
    returns: list[float] = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    for i, sample in enumerate(samples):
        sample.value = returns[i] if i < len(returns) else 0.0
    return samples, game.won


def mcts_bootstrap_game(seed: int, simulations: int, gamma: float,
                        rng: random.Random, num_actions: int,
                        num_waves: int | None = None,
                        flames_per_wave: int | None = None,
                        spawn_radius: int | None = None,
                        spawn_formation: str | None = None,
                        rollout_depth: int = 16,
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

    g = 0.0
    returns: list[float] = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    for i, sample in enumerate(samples):
        sample.value = returns[i] if i < len(returns) else 0.0
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
    batch_size: int = 128           # SGD minibatch (skips update until filled)
    update_steps: int = 64          # SGD minibatches per iteration
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_coef: float = 1.0         # value-loss weight relative to policy loss
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
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "best_score": best_score,
            "stage_idx": stage_idx,
        },
        path,
    )


def _update(net: ActorCritic, optimizer: optim.Optimizer,
            buffer: deque, cfg: AZConfig, rng: random.Random) -> tuple:
    """Run ``cfg.update_steps`` SGD steps; return mean (p_loss, v_loss).

    Policy loss is the cross-entropy of the net's masked policy against the
    search visit distribution; value loss is MSE against the return-to-go.
    """
    if len(buffer) < cfg.batch_size:
        return float("nan"), float("nan")
    net.train()
    data = list(buffer)
    p_tot = v_tot = 0.0
    for _ in range(cfg.update_steps):
        batch = rng.sample(data, cfg.batch_size)
        dev = cfg.device
        obs = torch.tensor([s.obs for s in batch], dtype=torch.float32,
                           device=dev)
        mask = torch.tensor([s.mask for s in batch], dtype=torch.bool,
                            device=dev)
        pi = torch.tensor([s.pi for s in batch], dtype=torch.float32,
                          device=dev)
        z = torch.tensor([s.value / VALUE_SCALE for s in batch],
                         dtype=torch.float32, device=dev)
        logits, value = net(obs)
        logp = torch.log_softmax(logits.masked_fill(~mask, MASK_FILL), dim=-1)
        policy_loss = -(pi * logp).sum(dim=-1).mean()
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
    net = ActorCritic(obs_size, num_actions, cfg.hidden).to(cfg.device)
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
                or ckpt.get("hidden") != cfg.hidden):
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
                spawn_formation=fm)
            buffer.extend(samples)
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
                dirichlet_alpha=cfg.dirichlet_alpha, temp_moves=cfg.temp_moves)
            buffer.extend(samples)
            wins += int(won)

        # --- Update ----------------------------------------------------
        p_loss, v_loss = _update(net, optimizer, buffer, cfg, rng)

        # --- Eval / checkpoint -----------------------------------------
        sp_win = wins / cfg.games_per_iter
        if iteration % cfg.eval_every == 0 or iteration == cfg.iterations:
            net.eval()
            act = az_action_fn(net, simulations=cfg.eval_simulations,
                               c_puct=cfg.c_puct, gamma=cfg.gamma)
            mean_reward, win_rate, parts = _eval_action_fn(
                act, cfg.eval_seeds, nw, fpw, sr, fm)
            stage_tag = (f" | stage {stage_idx}/{len(stages) - 1} "
                         f"(w={nw},f={fpw},r={sr}"
                         f"{',' + fm if fm else ''})" if len(stages) > 1 else "")
            print(
                f"iter {total_iter:4d}{stage_tag} | buffer {len(buffer):6d} | "
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
            print(f"iter {total_iter:4d} | buffer {len(buffer):6d} | "
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
