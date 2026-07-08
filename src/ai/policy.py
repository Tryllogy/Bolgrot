"""Glue between a trained :class:`ActorCritic` and the :class:`Game` env.

Both the trainer (``ppo.py``) and the viewer (``play.py``) drive the game
through these helpers so action selection and (de)serialisation live in exactly
one place. Nothing here contains game logic — it only reads the environment API
(``get_observation`` / ``legal_actions`` / ``step``).
"""
from __future__ import annotations

import torch

from ..game import Game
from ..entity import Player
from .. import constant
from .network import ActorCritic

# Fixed seeds used to score a policy so results are comparable across training
# runs while still covering several flame layouts.
EVAL_SEEDS: list[int] = [101, 202, 303, 404, 505, 606, 707, 808]

# Safety cap so a policy that never makes progress can't loop forever. HP/PA
# are finite so episodes terminate on their own; this only guards regressions.
MAX_STEPS = 500


def obs_and_mask(game: Game) -> tuple[torch.Tensor, torch.Tensor]:
    """Current observation and legal-action mask as batch-less tensors."""
    obs = torch.as_tensor(game.get_observation(), dtype=torch.float32)
    mask = torch.as_tensor(game.legal_actions(), dtype=torch.bool)
    return obs, mask


@torch.no_grad()
def choose_action(net: ActorCritic, game: Game,
                  deterministic: bool = True) -> int:
    """Pick a legal action for the current state.

    ``deterministic`` takes the highest-probability legal action (greedy, used
    for watching/evaluating); otherwise samples from the masked policy.
    """
    obs, mask = obs_and_mask(game)
    logits, _ = net(obs)
    logits = logits.masked_fill(~mask, float("-inf"))
    if deterministic:
        return int(torch.argmax(logits).item())
    return int(torch.distributions.Categorical(logits=logits).sample().item())


def run_episode(net: ActorCritic, seed: int,
                deterministic: bool = True,
                max_steps: int = MAX_STEPS) -> tuple[float, bool]:
    """Play one full game and return ``(total_reward, won)``."""
    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed)
    total = 0.0
    for _ in range(max_steps):
        if game.done:
            break
        reward, done = game.step(choose_action(net, game, deterministic))
        total += reward
        if done:
            break
    return total, game.won


def evaluate(net: ActorCritic, seeds: list[int] = EVAL_SEEDS,
             deterministic: bool = True) -> tuple[float, float]:
    """Mean total reward and win rate of ``net`` over ``seeds``."""
    net.eval()
    results = [run_episode(net, s, deterministic) for s in seeds]
    net.train()
    mean_reward = sum(r for r, _ in results) / len(results)
    win_rate = sum(1 for _, won in results if won) / len(results)
    return mean_reward, win_rate


def save_policy(net: ActorCritic, path: str) -> None:
    """Serialise ``net`` together with the shapes needed to rebuild it."""
    torch.save(
        {
            "state_dict": net.state_dict(),
            "obs_size": net.obs_size,
            "num_actions": net.num_actions,
            "hidden": net.hidden,
        },
        path,
    )


def load_policy(path: str, device: str = "cpu") -> ActorCritic:
    """Rebuild an :class:`ActorCritic` saved by :func:`save_policy`."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = ActorCritic(ckpt["obs_size"], ckpt["num_actions"], ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()
    return net
