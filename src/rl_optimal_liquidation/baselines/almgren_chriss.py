"""Analytical Almgren-Chriss (2001) baseline for the linear-impact regime.

Closed form (continuous time):
    v*_t = (kappa / sinh(kappa T)) * cosh(kappa (T - t)) * Q,
    kappa = sqrt(lambda * sigma^2 / eta).

Discrete inventory path: q*_k = Q * sinh(kappa (T - t_k)) / sinh(kappa T),
with t_k = k * (T/N).  Trades a*_k = q*_k - q*_{k+1}.
"""
from __future__ import annotations

import numpy as np


def kappa(lam: float, sigma: float, eta: float) -> float:
    return float(np.sqrt(lam * sigma * sigma / eta))


def ac_inventory_path(
    Q: float, T: float, N: int, lam: float, sigma: float, eta: float
) -> np.ndarray:
    """Discrete inventory q*_k for k=0..N. Length N+1; q*_0 = Q, q*_N = 0."""
    k = kappa(lam, sigma, eta)
    t = np.linspace(0.0, T, N + 1)
    if k * T < 1e-9:
        # Risk vanishes vs cost: AC degenerates to TWAP.
        return Q * (1.0 - t / T)
    return Q * np.sinh(k * (T - t)) / np.sinh(k * T)


def ac_schedule(
    Q: float, T: float, N: int, lam: float, sigma: float, eta: float
) -> np.ndarray:
    """Shares to sell each step: a*_k = q*_k - q*_{k+1}. Length N."""
    q = ac_inventory_path(Q, T, N, lam, sigma, eta)
    return q[:-1] - q[1:]


def ac_expected_cost(
    Q: float, T: float, N: int,
    lam: float, sigma: float, eta: float, gamma: float,
) -> float:
    """Deterministic-equivalent execution cost under the AC discrete schedule.

    Sums the same per-step cost expression the env reports as negative reward:
        eta a^2 / dt  +  gamma a^2  +  lam sigma^2 q^2 dt
    Drift, noise, and terminal penalty are zero by construction (q*_N = 0).
    """
    dt = T / N
    a = ac_schedule(Q, T, N, lam, sigma, eta)
    q = ac_inventory_path(Q, T, N, lam, sigma, eta)[:-1]
    cost = (eta * a * a) / dt + gamma * a * a + lam * sigma * sigma * q * q * dt
    return float(cost.sum())
