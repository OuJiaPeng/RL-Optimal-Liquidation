"""Beta-distribution policy for PPO on bounded [0, 1] action spaces.

Why: SB3's default `DiagGaussianDistribution` on `Box(0, 1)` samples from
`N(μ, σ²)`, clips the sample to the box for env consumption, but computes
`log_prob` on the *unclipped* sample. The mismatch creates a systematic
boundary bias — actions whose true mean should be at 1.0 require μ → ∞ to
balance the truncated probability mass, which the network can't represent
stably. Almgren-Chriss's optimum has explicit boundary behavior (`f_k → 1`
at the terminal step), so the Gaussian bias compounds.

Beta(α, β) lives on (0, 1) natively. No clipping; log_prob is well-defined
everywhere on the support; the distribution can collapse to near-boundary
modes by sending α/(α+β) → 0 or 1 with α+β → ∞.

Parameterization: the network outputs `2*action_dim` raw values per state.
softplus + 1 maps each half to α, β ≥ 1, so the Beta is unimodal with mode
`(α-1)/(α+β-2)` everywhere in [0, 1].
"""
from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BetaDistribution(Distribution):
    """Per-dimension independent Beta(α, β) with α, β ≥ 1 (unimodal)."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution: Optional[Beta] = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Linear:
        # First action_dim outputs -> raw α, second action_dim -> raw β.
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "BetaDistribution":
        # Defensive clamp: under large reward-scale regimes (Phase 2: 24x bigger
        # rewards than Phase 1), gradient spikes during VecNormalize warmup can
        # push raw_params to large/non-finite values and propagate NaN into
        # Beta() construction. ±100 bounds softplus output to ~101, allowing
        # very concentrated policies (Beta(101,101) std ≈ 0.035) without
        # restricting normal training.
        raw_params = raw_params.clamp(-100.0, 100.0)
        alpha = F.softplus(raw_params[..., : self.action_dim]) + 1.0
        beta = F.softplus(raw_params[..., self.action_dim :]) + 1.0
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        eps = 1e-6
        a = actions.clamp(eps, 1.0 - eps)
        log_prob = self.distribution.log_prob(a)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        ent = self.distribution.entropy()
        if ent.ndim > 1:
            ent = ent.sum(dim=-1)
        return ent

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        return (alpha - 1.0) / (alpha + beta - 2.0)

    def actions_from_params(
        self, raw_params: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        self.proba_distribution(raw_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, raw_params: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class BetaActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with a Beta action distribution.

    Drop-in for SB3's default `MlpPolicy` on Box(0, 1) action spaces. The
    only overrides are `_build` (to construct a BetaDistribution and its
    head) and `_get_action_dist_from_latent` (to feed raw params instead
    of (mean, log_std)).
    """

    def _build(self, lr_schedule) -> None:
        # Replace the parent's DiagGaussianDistribution with Beta.
        self.action_dist = BetaDistribution(int(np.prod(self.action_space.shape)))

        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1.0,
            }
            if not self.share_features_extractor:
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
                del module_gains[self.features_extractor]
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        raw_params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(raw_params)
