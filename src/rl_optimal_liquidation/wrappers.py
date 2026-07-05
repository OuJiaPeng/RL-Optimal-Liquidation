"""Reward wrappers for training-time variance reduction."""
from __future__ import annotations

import gymnasium as gym

from .baselines.exact_lq import (
    exact_optimal_trades,
    per_step_costs_of_trades,
    sigma_profile_path,
    smart_static_trades,
)


class ControlVariateReward(gym.Wrapper):
    """Subtract a computable classical schedule's per-step cost from the reward.

    The reference schedule depends only on the episode's scenario (its sigma
    path), never on the agent's actions, so the optimal policy is unchanged —
    but returns become regret-vs-classical, collapsing the cross-scenario
    return variance that dominates PPO's gradient noise in this env.

    mode='ce'           re-solve the exact schedule for the episode's realized
                        sigma path (tightest reference; return ~ regret vs the
                        causal ceiling).
    mode='smart_static' price one fixed distribution-optimal schedule.
    """

    def __init__(self, env: gym.Env, mode: str = "ce"):
        super().__init__(env)
        if mode not in ("ce", "smart_static"):
            raise ValueError(f"control variate mode must be 'ce' or 'smart_static', got {mode!r}")
        self.mode = mode
        p = env.unwrapped.p
        self._profile = sigma_profile_path(p)
        self._static = smart_static_trades(p, p.sigma_noise_std) if mode == "smart_static" else None
        self._ref_costs = None
        self._k = 0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        p = self.env.unwrapped.p
        path = self.env.unwrapped._sigma_scale * self._profile
        trades = self._static if self.mode == "smart_static" else exact_optimal_trades(p, path)
        self._ref_costs = per_step_costs_of_trades(trades, p, path)
        self._k = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # env reward is -cost; adding the reference cost makes it -(cost - ref).
        reward = float(reward) + float(self._ref_costs[self._k])
        self._k += 1
        return obs, reward, terminated, truncated, info
