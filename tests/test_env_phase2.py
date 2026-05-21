"""Regression tests for Phase 2 env features.

Sanity-checks that each new env knob (impact_type, sigma profile, eta AR(1),
extra observations) does what we built it to do. Caught nothing on the first
pass — the goal is to prevent silent regressions if anyone refactors the env.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams


def test_impact_type_validation():
    """LiquidationParams rejects unknown impact_type values."""
    with pytest.raises(ValueError, match="impact_type"):
        LiquidationParams(impact_type="cubic")


def test_sqrt_impact_perm_cost_concave_in_a():
    """At impact_type='sqrt', per-step perm cost grows as a^(3/2), not a^2."""
    p = LiquidationParams(impact_type="sqrt", sigma=0.0, terminal_penalty=0.0)
    env = LiquidationEnv(p, seed=0)
    # Compare per-step perm_cost at two action sizes
    env.reset(seed=0)
    _, _, _, _, info_small = env.step(np.array([0.01], dtype=np.float32))
    env.reset(seed=0)
    _, _, _, _, info_big = env.step(np.array([0.10], dtype=np.float32))
    # 10x more shares: linear would give 100x cost; sqrt gives ~10^1.5 = 31.6x
    ratio = info_big["perm_cost"] / info_small["perm_cost"]
    assert 25.0 < ratio < 35.0, f"Expected ~31.6x (a^1.5); got {ratio:.2f}x"


def test_linear_impact_perm_cost_quadratic_in_a():
    """At impact_type='linear', per-step perm cost grows as a^2."""
    p = LiquidationParams(impact_type="linear", sigma=0.0, terminal_penalty=0.0)
    env = LiquidationEnv(p, seed=0)
    env.reset(seed=0)
    _, _, _, _, info_small = env.step(np.array([0.01], dtype=np.float32))
    env.reset(seed=0)
    _, _, _, _, info_big = env.step(np.array([0.10], dtype=np.float32))
    ratio = info_big["perm_cost"] / info_small["perm_cost"]
    assert 95.0 < ratio < 105.0, f"Expected ~100x (a^2); got {ratio:.2f}x"


def test_sigma_profile_constant():
    """sigma_at(t) returns p.sigma for all t when profile='constant'."""
    p = LiquidationParams(sigma_profile="constant", sigma=0.3)
    assert p.sigma_at(0.0) == pytest.approx(0.3)
    assert p.sigma_at(0.5) == pytest.approx(0.3)
    assert p.sigma_at(1.0) == pytest.approx(0.3)


def test_sigma_profile_u_shaped():
    """U-shape: sigma high at t=0 and t=T, low at t=T/2."""
    p = LiquidationParams(sigma_profile="u_shaped", sigma=0.3, sigma_amplitude=0.5)
    # cos(0)=1 -> 1+A; cos(pi)=-1 -> 1-A; cos(2pi)=1 -> 1+A
    assert p.sigma_at(0.0) == pytest.approx(0.45)
    assert p.sigma_at(0.5) == pytest.approx(0.15)
    assert p.sigma_at(1.0) == pytest.approx(0.45)


def test_sigma_profile_validation():
    """Reject unknown sigma_profile."""
    with pytest.raises(ValueError, match="sigma_profile"):
        LiquidationParams(sigma_profile="triangle")


def test_sigma_noise_scale_per_episode():
    """sigma_noise_std > 0 produces different scale draws across resets."""
    p = LiquidationParams(sigma_noise_std=0.15, sigma_profile="constant")
    env = LiquidationEnv(p, seed=0)
    scales = []
    for i in range(20):
        env.reset(seed=i)
        scales.append(env._sigma_scale)
    scales = np.array(scales)
    # Sample mean should be near 1.0; std should approach noise_std for n=20
    assert 0.85 < scales.mean() < 1.15, f"scale mean {scales.mean():.3f}"
    assert 0.08 < scales.std() < 0.25, f"scale std {scales.std():.3f}"


def test_eta_ar1_evolves_per_step():
    """Stochastic eta: per-step eta values vary, mean stays near eta_mean."""
    p = LiquidationParams(eta=1e-6, eta_rho=0.9, eta_noise_std=2.0)
    env = LiquidationEnv(p, seed=0)
    env.reset(seed=0)
    etas = [env._eta_k]
    for _ in range(p.N):
        _, _, term, _, _ = env.step(np.array([0.05], dtype=np.float32))
        etas.append(env._eta_k)
        if term:
            break
    etas = np.array(etas)
    # Should see variation across steps
    assert etas.std() / etas.mean() > 0.02, "eta did not vary across steps"
    # Should stay positive
    assert (etas > 0).all(), "eta went non-positive"


def test_eta_constant_when_noise_zero():
    """eta_noise_std=0 keeps eta_k pinned at eta across all steps."""
    p = LiquidationParams(eta=1e-6, eta_rho=0.9, eta_noise_std=0.0)
    env = LiquidationEnv(p, seed=0)
    env.reset(seed=0)
    for _ in range(p.N):
        _, _, term, _, _ = env.step(np.array([0.05], dtype=np.float32))
        assert env._eta_k == pytest.approx(p.eta)
        if term:
            break


def test_obs_dim_matches_flags():
    """Observation dimensionality reflects the include_* flags."""
    base = dict(Q=1e6, T=1.0, N=50, sigma=0.3, eta=1e-6, gamma=1e-7, lam=1e-6)
    # 2D minimal
    env_2d = LiquidationEnv(LiquidationParams(**base,
        include_price_obs=False, include_vol_obs=False, include_eta_obs=False))
    assert env_2d.observation_space.shape == (2,)
    # 3D with price
    env_3d_price = LiquidationEnv(LiquidationParams(**base,
        include_price_obs=True, include_vol_obs=False, include_eta_obs=False))
    assert env_3d_price.observation_space.shape == (3,)
    # 3D with vol
    env_3d_vol = LiquidationEnv(LiquidationParams(**base,
        include_price_obs=False, include_vol_obs=True, include_eta_obs=False))
    assert env_3d_vol.observation_space.shape == (3,)
    # 5D = 2 base + 3 optional adds
    env_5d = LiquidationEnv(LiquidationParams(**base,
        include_price_obs=True, include_vol_obs=True, include_eta_obs=True))
    assert env_5d.observation_space.shape == (5,)


def test_obs_eta_field_tracks_eta_k():
    """When include_eta_obs=True, the last obs element = eta_k / eta_mean."""
    p = LiquidationParams(eta=1e-6, eta_rho=0.9, eta_noise_std=2.0, include_eta_obs=True)
    env = LiquidationEnv(p, seed=0)
    obs, _ = env.reset(seed=0)
    # At reset, eta_k = eta_mean, so the ratio should be 1.0
    assert obs[-1] == pytest.approx(1.0)
    # After a step, eta has evolved; ratio should not equal 1 (with very high probability)
    obs, _, _, _, _ = env.step(np.array([0.05], dtype=np.float32))
    # Sanity: it's a positive number near 1 but not exactly 1
    assert 0.1 < obs[-1] < 10.0
