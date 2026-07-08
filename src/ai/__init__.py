"""PPO agent that learns to play the game headlessly through the
:class:`src.game.Game` environment API (``get_observation``/``legal_actions``/
``step``).

- ``network`` — the masked actor-critic (:class:`ActorCritic`).
- ``policy``  — action selection, evaluation and (de)serialisation glue.
- ``ppo``     — the PPO training loop (rollouts, GAE, clipped update).
- ``train``   — CLI entry point (``python -m src.ai.train``).
- ``play``    — watch a trained policy with the renderer
  (``python -m src.ai.play``).

Nothing here contains game logic; it only drives ``Game`` through its
environment API. Requires the optional ``ai`` dependency group (``torch``).
"""
