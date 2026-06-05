"""Reference execution-cost spec (quadratic / linear-impact regime).

A standalone, readable definition of the per-step + terminal execution cost that
`LiquidationEnv.step` mirrors exactly (the env inlines the same formula for vectorized
speed). Baselines and value-of-conditioning oracles price candidate schedules through
these functions, so the cost cannot drift between the thing being measured and the
things it is measured against:

    per-step = eta_k * a^2 / dt  +  gamma * a^2  +  lam * sigma_k^2 * q^2 * dt
    terminal = M * q_N^2
"""
from __future__ import annotations

import numpy as np


def per_step_cost(a, q, eta_k, sigma_k, p) -> np.ndarray:
    """Cost charged at one step for selling `a` shares out of inventory `q`.

    Vectorized: a, q, eta_k may be arrays (broadcast). Matches the env's reward sign-flip.
    """
    dt = p.T / p.N
    a = np.asarray(a, dtype=float)
    q = np.asarray(q, dtype=float)
    temp = eta_k * a * a / dt
    perm = p.gamma * a * a            # quadratic permanent-impact cost
    inv = p.lam * sigma_k * sigma_k * q * q * dt
    return temp + perm + inv


def terminal_cost(q_final, p) -> np.ndarray:
    return p.terminal_penalty * np.asarray(q_final, dtype=float) ** 2


def schedule_cost(trades, p, eta_path, sigma_path) -> float:
    """Total realized cost of an explicit trade schedule on a fixed (eta, sigma) path.

    `trades` is length-N shares sold per step (a_k, not fractions). Used to score any
    candidate schedule (oracle / MPC / AC / DP-greedy) on a fixed realized scenario.
    """
    trades = np.asarray(trades, dtype=float)
    eta_path = np.asarray(eta_path, dtype=float)
    sigma_path = np.asarray(sigma_path, dtype=float)
    q = float(p.Q)
    total = 0.0
    for k in range(p.N):
        a = trades[k]
        total += float(per_step_cost(a, q, eta_path[k], sigma_path[k], p))
        q -= a
    total += float(terminal_cost(q, p))
    return total
