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
from rl_optimal_liquidation.baselines.exact_lq import (
    cost_of_trades,
    exact_optimal_trades,
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


def test_kt3_is_discriminating_regime():
    # Pins the "why phase1_kt3" claim in the docs: at kappa*T=0.3 the optimal
    # schedule is nearly flat, so naive TWAP (uniform shares) is within ~0.02%
    # of AC and the acceptance band cannot tell "recovered AC" from "learned
    # anything reasonable". At kappa*T=3 the optimum front-loads, so TWAP is
    # ~32% worse and landing near AC is a meaningful test.
    for lam, expected_gap_pct in [(1e-6, 0.017), (1e-4, 31.6)]:
        p = LiquidationParams(lam=lam)  # constant sigma
        path = np.full(p.N, p.sigma)
        ac = ac_expected_cost(p.Q, p.T, p.N, p.lam, p.sigma, p.eta, p.gamma)
        twap = cost_of_trades(np.full(p.N, p.Q / p.N), p, path)  # uniform shares
        gap_pct = (twap - ac) / ac * 100.0
        assert gap_pct == pytest.approx(expected_gap_pct, rel=0.05)


def test_exact_lq_soft_terminal_residual():
    # The exact tridiagonal optimum must sit -0.094% below the hard-liquidation
    # AC closed form at Phase 1 params: the documented direct-opt residual is
    # the soft terminal penalty (optimal leftover q_N = q_{N-1}*c_a/(c_a+M)),
    # not optimizer noise.
    p = LiquidationParams()
    path = np.full(p.N, p.sigma)
    exact = cost_of_trades(exact_optimal_trades(p, path), p, path)
    ac = ac_expected_cost(p.Q, p.T, p.N, p.lam, p.sigma, p.eta, p.gamma)
    assert (exact - ac) / ac * 100.0 == pytest.approx(-0.0943, abs=0.002)


def test_env_with_ac_policy_finishes_at_zero_inventory():
    # Deterministic env (sigma = 0) so the AC open-loop policy is exact.
    params = LiquidationParams(
        Q=1e6, T=1.0, N=50, S0=100.0, mu=0.0, sigma=0.0,
        eta=1e-6, gamma=1e-7, lam=1e-6, terminal_penalty=0.0,
    )
    env = LiquidationEnv(params, seed=0)
    out = rollout(env, ac_policy_fn(env))
    assert out["inventory"][-1] == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("sigma", [0.0, 0.3])
def test_env_ac_cost_matches_analytical(sigma):
    # Episode cost is deterministic given the schedule — price noise never
    # enters the reward — so the sigma=0.3 case pins the full per-step formula
    # including the lam*sigma^2*q^2*dt risk term, not just the impact terms.
    params = LiquidationParams(
        Q=1e6, T=1.0, N=50, S0=100.0, mu=0.0, sigma=sigma,
        eta=1e-6, gamma=1e-7, lam=1e-6, terminal_penalty=0.0,
    )
    env = LiquidationEnv(params, seed=0)
    out = rollout(env, ac_policy_fn(env))
    env_cost = episode_cost(out["rewards"])

    analytic = ac_expected_cost(
        params.Q, params.T, params.N,
        params.lam, params.sigma, params.eta, params.gamma,
    )
    assert env_cost == pytest.approx(analytic, rel=1e-6)
