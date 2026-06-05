"""Regression tests for Phase 2 env features.

Sanity-checks that each env knob (sigma profile, per-episode sigma noise, extra
observations, linear permanent-impact cost) does what we built it to do. The goal
is to prevent silent regressions if anyone refactors the env.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams


def test_linear_impact_perm_cost_quadratic_in_a():
    """Per-step permanent-impact cost grows as a^2."""
    p = LiquidationParams(sigma=0.0, terminal_penalty=0.0)
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


def test_obs_dim_matches_flags():
    """Observation dimensionality reflects the include_* flags."""
    base = dict(Q=1e6, T=1.0, N=50, sigma=0.3, eta=1e-6, gamma=1e-7, lam=1e-6)
    # 2D minimal
    env_2d = LiquidationEnv(LiquidationParams(**base,
        include_price_obs=False, include_vol_obs=False))
    assert env_2d.observation_space.shape == (2,)
    # 3D with price
    env_3d_price = LiquidationEnv(LiquidationParams(**base,
        include_price_obs=True, include_vol_obs=False))
    assert env_3d_price.observation_space.shape == (3,)
    # 3D with vol
    env_3d_vol = LiquidationEnv(LiquidationParams(**base,
        include_price_obs=False, include_vol_obs=True))
    assert env_3d_vol.observation_space.shape == (3,)
    # 4D = 2 base + price + vol
    env_4d = LiquidationEnv(LiquidationParams(**base,
        include_price_obs=True, include_vol_obs=True))
    assert env_4d.observation_space.shape == (4,)
