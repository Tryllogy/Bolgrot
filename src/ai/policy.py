"""Shared glue between a NEAT network and the :class:`Game` environment.

Both the trainer (``train.py``) and the viewer (``play.py``) drive the game
through these helpers so the action-selection logic lives in exactly one place.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from ..game import Game
from ..entity import Player
from .. import constant

if TYPE_CHECKING:
    import neat

# Safety cap so a network that never makes progress can't loop forever. HP/PA
# are finite so episodes terminate on their own, but this guards regressions.
MAX_STEPS = 2000


def choose_action(net: "neat.nn.FeedForwardNetwork", game: Game) -> int:
    """Return the highest-scoring *legal* action for the current state.

    Activates the network on ``game.get_observation()`` and takes the argmax
    over outputs, masking illegal actions so the network can only pick moves
    the game will actually accept.
    """
    outputs = net.activate(game.get_observation())
    legal = game.legal_actions()
    best_i = -1
    best_v = float("-inf")
    for i, ok in enumerate(legal):
        if ok and outputs[i] > best_v:
            best_v, best_i = outputs[i], i
    # legal always contains at least "end turn" while the game is live.
    return best_i


def run_episode(
    net: "neat.nn.FeedForwardNetwork",
    seed: int,
    max_steps: int = MAX_STEPS,
) -> float:
    """Play one full game with ``net`` and return the total reward."""
    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed)
    total = 0.0
    steps = 0
    while not game.done and steps < max_steps:
        reward, _ = game.step(choose_action(net, game))
        total += reward
        steps += 1
    return total


def evaluate(
    net: "neat.nn.FeedForwardNetwork",
    seeds: list[int],
    max_steps: int = MAX_STEPS,
) -> float:
    """Mean total reward of ``net`` over the given episode ``seeds``.

    Using a fixed seed set keeps fitness comparable across the population and
    across generations while still testing several flame layouts.
    """
    return sum(run_episode(net, s, max_steps) for s in seeds) / len(seeds)
