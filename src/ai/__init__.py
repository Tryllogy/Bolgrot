"""NEAT neuroevolution layer that learns to play the game headlessly.

- ``policy``  — glue between a NEAT network and :class:`src.game.Game`.
- ``train``   — evolve a population (``python -m src.ai.train``).
- ``play``    — watch the best genome with the renderer
  (``python -m src.ai.play``).
"""
from .policy import choose_action, run_episode, evaluate

__all__ = ["choose_action", "run_episode", "evaluate"]
