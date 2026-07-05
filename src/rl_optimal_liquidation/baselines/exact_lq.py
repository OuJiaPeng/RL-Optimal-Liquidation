"""Exact LQ schedule machinery: closed-form optima for deterministic sigma paths.

For any deterministic per-step sigma path, the env's episode cost is a convex
quadratic in the inventory path (q_1..q_N) — including the soft terminal
penalty M*q_N^2 — so the optimal open-loop schedule is the solution of a
tridiagonal linear system: exact, instant, no iterative optimizer.

This is the engine behind the classical baseline ladder
(scripts/eval_phase2_baselines.py):
  naive AC       -> baselines/almgren_chriss.py (hard-liquidation closed form)
  smart-static   -> exact schedule at effective risk lam * E[scale^2] * profile^2
                    (episode cost is linear in scale^2 given a fixed schedule,
                    so the expected-cost optimum needs only E[scale^2])
  CE-AC / oracle -> exact schedule at the realized sigma path
and the control-variate reward wrapper (wrappers.py).
"""
from __future__ import annotations

import numpy as np

from ..envs.liquidation_env import LiquidationParams


def sigma_profile_path(p: LiquidationParams) -> np.ndarray:
    """sigma(t_k) at unit scale for k=0..N-1 (the deterministic profile)."""
    t = np.arange(p.N) * p.dt()
    return np.array([p.sigma_at(tk, scale=1.0) for tk in t])


def exact_optimal_trades(p: LiquidationParams, sigma_path: np.ndarray) -> np.ndarray:
    """Exact argmin of the env cost for a deterministic sigma path.

    min over q_1..q_N of  sum_k [ c_a (q_k - q_{k+1})^2 + r_k q_k^2 ] + M q_N^2,
    q_0 = Q, c_a = eta/dt + gamma, r_k = lam sigma_k^2 dt (risk charged on
    pre-trade inventory, matching the env). First-order conditions are a
    tridiagonal linear system.
    """
    N = p.N
    dt = p.dt()
    c_a = p.eta / dt + p.gamma
    r = p.lam * np.asarray(sigma_path) ** 2 * dt  # r_0..r_{N-1}

    A = np.zeros((N, N))
    b = np.zeros(N)
    for j in range(1, N):        # unknowns q_1..q_{N-1} -> rows 0..N-2
        i = j - 1
        A[i, i] = 2.0 * c_a + r[j]
        if i > 0:
            A[i, i - 1] = -c_a
        A[i, i + 1] = -c_a
    b[0] = c_a * p.Q
    A[N - 1, N - 2] = -c_a       # q_N row: soft terminal penalty, no risk term
    A[N - 1, N - 1] = c_a + p.terminal_penalty

    q = np.concatenate([[p.Q], np.linalg.solve(A, b)])
    trades = q[:-1] - q[1:]
    assert (trades >= -1e-9 * p.Q).all(), "exact solve produced a negative trade"
    return trades


def per_step_costs_of_trades(
    trades: np.ndarray, p: LiquidationParams, sigma_path: np.ndarray
) -> np.ndarray:
    """Length-N per-step costs of a fixed schedule (terminal penalty folded into
    the last step), matching the env's reward decomposition exactly."""
    trades = np.asarray(trades, dtype=float)
    dt = p.dt()
    c_a = p.eta / dt + p.gamma
    q = p.Q - np.concatenate([[0.0], np.cumsum(trades)[:-1]])  # pre-trade inventory
    costs = c_a * trades**2 + p.lam * np.asarray(sigma_path) ** 2 * q**2 * dt
    q_final = p.Q - trades.sum()
    costs[-1] += p.terminal_penalty * q_final**2
    return costs


def cost_of_trades(trades: np.ndarray, p: LiquidationParams, sigma_path: np.ndarray) -> float:
    return float(per_step_costs_of_trades(trades, p, sigma_path).sum())


def expected_scale_sq(noise_std: float, n: int = 1_000_000, seed: int = 12345) -> float:
    """E[scale^2] for scale = max(1 + noise_std*Z, 0.1) — the env's clamped draw."""
    if noise_std <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    s = np.maximum(1.0 + rng.normal(0.0, noise_std, n), 0.1)
    return float((s**2).mean())


def smart_static_trades(p: LiquidationParams, noise_std: float) -> np.ndarray:
    """Best single schedule for a desk knowing the profile and the scale
    distribution but never the realized scale."""
    return exact_optimal_trades(p, np.sqrt(expected_scale_sq(noise_std)) * sigma_profile_path(p))
