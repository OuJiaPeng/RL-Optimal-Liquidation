"""Rollout helpers and an AC-as-policy adapter for evaluation."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .baselines.almgren_chriss import ac_schedule
from .envs.liquidation_env import LiquidationEnv

PolicyFn = Callable[[np.ndarray, bool], np.ndarray]


def rollout(
    env: LiquidationEnv, policy_fn: PolicyFn, deterministic: bool = True
) -> dict[str, np.ndarray]:
    """Run one episode and return inventory, actions, rewards, prices.

    Robust to gym wrappers (Monitor etc.) — uses `env.unwrapped` to read
    initial inventory and price before any wrapper post-processing.
    """
    obs, _ = env.reset()
    base = env.unwrapped
    inventory = [float(base.q)]
    prices = [float(base.S)]
    actions: list[float] = []
    rewards: list[float] = []
    done = False
    while not done:
        a = policy_fn(obs, deterministic)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        actions.append(info["executed"])
        inventory.append(info["inventory"])
        prices.append(info["price"])
        rewards.append(r)
    return {
        "inventory": np.asarray(inventory),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "prices": np.asarray(prices),
    }


def ac_policy_fn(env: LiquidationEnv) -> PolicyFn:
    """Open-loop AC schedule expressed as a fraction-of-inventory policy.

    Since AC is deterministic and the env follows the same dynamics, executing
    schedule[k] shares each step drains inventory to ~0 by step N.
    """
    p = env.unwrapped.p
    schedule = ac_schedule(p.Q, p.T, p.N, p.lam, p.sigma, p.eta)

    def fn(obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # obs[0] = k/N is the step about to execute; obs[1] = q/Q.
        k = int(round(float(obs[0]) * p.N))
        q = float(obs[1]) * p.Q
        if k >= len(schedule) or q <= 0.0:
            return np.array([1.0], dtype=np.float32)
        frac = schedule[k] / q if q > 0.0 else 0.0
        return np.array([float(np.clip(frac, 0.0, 1.0))], dtype=np.float32)

    return fn


def episode_cost(rewards: np.ndarray) -> float:
    """Total execution cost = -sum(rewards)."""
    return float(-rewards.sum())


def episode_costs_over_seeds(
    env: LiquidationEnv, policy_fn: PolicyFn, episodes: int, seed: int = 0
) -> np.ndarray:
    """Per-episode costs over `episodes` seeded rollouts (seed+i for episode i).

    Two policies evaluated with the same (env, episodes, seed) see identical
    noise realizations, so their cost arrays are paired — difference them for
    matched-pair statistics instead of comparing marginal means.
    """
    costs = np.empty(episodes)
    for i in range(episodes):
        env.reset(seed=seed + i)
        out = rollout(env, policy_fn, deterministic=True)
        costs[i] = episode_cost(out["rewards"])
    return costs


def mean_cost(
    env: LiquidationEnv, policy_fn: PolicyFn, episodes: int, seed: int = 0
) -> tuple[float, float]:
    """Mean and std of episode cost over `episodes` Monte Carlo rollouts."""
    costs = episode_costs_over_seeds(env, policy_fn, episodes, seed=seed)
    return float(costs.mean()), float(costs.std())
