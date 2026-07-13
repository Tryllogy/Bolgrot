"""Actor-critic network for the PPO agent.

A small shared MLP trunk feeds two heads: a policy head producing one logit per
discrete action in :data:`Game.ACTIONS`, and a value head estimating the state
value. Illegal actions are masked out (their logits driven to ``MASK_FILL``)
before forming the categorical distribution, so the policy can only ever assign
probability to moves the game will accept.
"""
from __future__ import annotations
import math

import torch
from torch import nn
from torch.distributions import Categorical

# Large negative logit for illegal actions: e^MASK_FILL ≈ 0 probability without
# the NaNs that -inf would produce if a whole row were masked.
MASK_FILL = -1e8


def _layer_init(layer: nn.Linear, std: float = math.sqrt(2),
                bias: float = 0.0) -> nn.Linear:
    """Orthogonal weight init — the standard PPO initialisation."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Shared-trunk policy/value network with action masking."""

    def __init__(self, obs_size: int, num_actions: int,
                 hidden: int = 64, value_dim: int = 1) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden = hidden
        # value_dim=1 is the usual single state-value head. value_dim=2 is the
        # AlphaZero "twohead" value target (src/ai/alphazero.py): head 0 is the
        # dense shaped return, head 1 is P(win) — kept separate so the win
        # signal never spikes the shaped estimate near a terminal (the failure
        # of the single-head "blend" mode).
        self.value_dim = value_dim
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_size, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        # Small policy-head gain keeps the initial policy near-uniform; unit
        # gain on the value head is the usual choice.
        self.policy_head = _layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.value_head = _layer_init(nn.Linear(hidden, value_dim), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)`` for a batch of observations.

        ``value`` is squeezed to a scalar per state when ``value_dim == 1``
        (the single-head case, unchanged); for ``value_dim > 1`` it keeps its
        trailing head axis (``[..., value_dim]``).
        """
        h = self.trunk(obs)
        value = self.value_head(h)
        if self.value_dim == 1:
            value = value.squeeze(-1)
        return self.policy_head(h), value

    def _dist(self, obs: torch.Tensor, mask: torch.Tensor
              ) -> tuple[Categorical, torch.Tensor]:
        """Masked action distribution and value for ``obs``.

        ``mask`` is a boolean tensor (``True`` = legal). Illegal logits are set
        to :data:`MASK_FILL` before building the categorical.
        """
        logits, value = self(obs)
        logits = logits.masked_fill(~mask, MASK_FILL)
        return Categorical(logits=logits), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """State-value estimate(s) for ``obs`` (no policy computation)."""
        return self(obs)[1]

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or score) an action and return
        ``(action, log_prob, entropy, value)``.

        When ``action`` is given its log-prob is evaluated instead of sampling
        — used during the PPO update to score the stored actions under the
        current policy.
        """
        dist, value = self._dist(obs, mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
