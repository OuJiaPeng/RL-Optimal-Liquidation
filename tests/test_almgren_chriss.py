"""Sanity checks on the analytical AC baseline and the env-AC consistency."""
from __future__ import annotations

import numpy as np
import pytest

from rl_optimal_liquidation.baselines.almgren_chriss import (
    ac_expected_cost,
    ac_inventory_path,
    ac_schedule,
    kappa,
)
from rl_optimal_liquidation.diagnostics import ac_policy_fn, episode_cost, rollout
from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams


def test_inventory_path_endpoints():
    q = ac_inventory_path(Q=1e6, T=1.0, N=50, lam=1e-6, sigma=0.3, eta=1e-6)
    assert q.shape == (51,)
    assert q[0] == pytest.approx(1e6)
    assert q[-1] == pytest.approx(0.0, abs=1e-6)


def test_schedule_sums_to_Q():
    a = ac_schedule(Q=1e6, T=1.0, N=50, lam=1e-6, sigma=0.3, eta=1e-6)
    assert a.shape == (50,)
    assert a.sum() == pytest.approx(1e6)
    assert (a >= -1e-9).all()  # AC trades are non-negative for selling


def test_kappa_zero_risk_is_twap():
    # lam = 0 => kappa = 0 => schedule degenerates to TWAP (uniform).
    a = ac_schedule(Q=1e6, T=1.0, N=10, lam=0.0, sigma=0.3, eta=1e-6)
    assert np.allclose(a, 1e5)


def test_kappa_formula():
    assert kappa(lam=4.0, sigma=1.0, eta=1.0) == pytest.approx(2.0)


def test_env_with_ac_policy_finishes_at_zero_inventory():
    # Deterministic env (sigma = 0) so the AC open-loop policy is exact.
    params = LiquidationParams(
        Q=1e6, T=1.0, N=50, S0=100.0, mu=0.0, sigma=0.0,
        eta=1e-6, gamma=1e-7, lam=1e-6, terminal_penalty=0.0,
    )
    env = LiquidationEnv(params, seed=0)
    out = rollout(env, ac_policy_fn(env))
    assert out["inventory"][-1] == pytest.approx(0.0, abs=1.0)


def test_env_ac_cost_matches_analytical_when_sigma_zero():
    # No price noise, no terminal penalty => env cost == analytical AC cost.
    # gamma=0 so we don't have to worry about whether perm impact would shift
    # one side vs the other; with the textbook-AC formulation both env and
    # analytic drop perm entirely from the cost regardless of gamma's value,
    # but pinning gamma=0 keeps this test independent of that convention.
    params = LiquidationParams(
        Q=1e6, T=1.0, N=50, S0=100.0, mu=0.0, sigma=0.0,
        eta=1e-6, gamma=0.0, lam=1e-6, terminal_penalty=0.0,
    )
    env = LiquidationEnv(params, seed=0)
    out = rollout(env, ac_policy_fn(env))
    env_cost = episode_cost(out["rewards"])

    # With sigma=0 the inventory-penalty term in the cost is zero too, so the
    # analytical-cost call uses the env's stated sigma. They should agree.
    analytic = ac_expected_cost(
        params.Q, params.T, params.N,
        params.lam, params.sigma, params.eta,
    )
    assert env_cost == pytest.approx(analytic, rel=1e-6)
