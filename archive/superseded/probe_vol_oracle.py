"""Vol-conditioning oracle diagnostic (matched-pair).

Closes the methodology gap on the Phase 2 vol result. The matched-pair oracle
gap at noise=0.15 is 1.98% (too small for PPO to capture — which informed the
bump to 0.3); this script recomputes the oracle at any noise level. It does
matched-pair MC: same per-episode sigma_scale draws, compute (1) oracle =
direct-opt knowing the scale, (2) AC schedule cost.

Gap (AC - oracle) / AC = maximum value of vol-conditioning at that noise level.
NOTE: the "AC" baseline here is the NAIVE constant-sigma AC schedule — see
scripts/eval_phase2_baselines.py for the smart-static and CE-AC baselines that
decompose this gap into profile knowledge vs genuine per-episode conditioning.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from rl_optimal_liquidation.baselines.almgren_chriss import ac_inventory_path
from rl_optimal_liquidation.envs import LiquidationParams


def opt_cost_at_scale(scale: float, p: LiquidationParams, n_steps: int = 3000) -> float:
    """Best 50-param schedule given a known per-episode sigma_scale."""
    dt = p.T / p.N
    theta = torch.zeros(p.N, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    for _ in range(n_steps):
        opt.zero_grad()
        f = torch.sigmoid(theta)
        q = torch.tensor(p.Q, dtype=f.dtype)
        total = torch.zeros((), dtype=f.dtype)
        for k in range(p.N):
            a = f[k] * q
            sigma_k = p.sigma * scale * (1.0 + p.sigma_amplitude * float(np.cos(2 * np.pi * k * dt / p.T)))
            total = total + p.eta * a * a / dt + p.gamma * a * a + p.lam * sigma_k * sigma_k * q * q * dt
            q = q - a
        total = total + p.terminal_penalty * q * q
        total.backward()
        opt.step()
    return float(total.item())


def cost_of_schedule_at_scale(f_array: np.ndarray, scale: float, p: LiquidationParams) -> float:
    """Cost of a fixed schedule (e.g. AC's) under a given sigma_scale."""
    dt = p.T / p.N
    q = p.Q
    total = 0.0
    for k in range(p.N):
        a = f_array[k] * q
        sigma_k = p.sigma * scale * (1.0 + p.sigma_amplitude * float(np.cos(2 * np.pi * k * dt / p.T)))
        total += p.eta * a * a / dt + p.gamma * a * a + p.lam * sigma_k * sigma_k * q * q * dt
        q = q - a
    total += p.terminal_penalty * q * q
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2_vol.yaml",
                    help="env params loaded from this config; oracle drift-proofs against config changes")
    ap.add_argument("--noise-std", type=float, default=0.3,
                    help="sigma_noise_std for the per-episode MC; overrides config")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)

    # Load env params from config so this oracle stays in lock-step with what was trained.
    # Override sigma_noise_std to 0 — the MC sweeps scale externally and we need
    # deterministic-per-realization rollouts to get a clean oracle.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_kwargs = dict(cfg["env"])
    env_kwargs["sigma_noise_std"] = 0.0
    p = LiquidationParams(**env_kwargs)

    q_ac = ac_inventory_path(p.Q, p.T, p.N, p.lam, p.sigma, p.eta)
    a_ac = q_ac[:-1] - q_ac[1:]
    f_ac = a_ac / np.maximum(q_ac[:-1], 1e-9)

    rng = np.random.default_rng(args.seed)
    scales = np.clip(1.0 + rng.normal(0.0, args.noise_std, args.episodes), 0.1, None)

    print(f"Computing oracle + AC matched-pair on {args.episodes} sigma_scale samples"
          f" (noise_std={args.noise_std}, range_3sig=[{1-3*args.noise_std:.2f}, {1+3*args.noise_std:.2f}])...")
    oracle_costs = []
    ac_costs = []
    for i, s in enumerate(scales):
        oracle_costs.append(opt_cost_at_scale(float(s), p))
        ac_costs.append(cost_of_schedule_at_scale(f_ac, float(s), p))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.episodes}: oracle={np.mean(oracle_costs):.4e}  ac={np.mean(ac_costs):.4e}")

    oracle_mean = float(np.mean(oracle_costs))
    ac_mean = float(np.mean(ac_costs))
    gap = (ac_mean - oracle_mean) / ac_mean * 100.0

    print()
    print(f"On the same {args.episodes} sigma_scale samples:")
    print(f"  Perfect-foresight oracle:        {oracle_mean:.4e}")
    print(f"  AC schedule (blind to scale):    {ac_mean:.4e}")
    print(f"  Avg sigma_scale: {np.mean(scales):.3f} +/- {np.std(scales):.3f}")
    print()
    print(f"Value of sigma-conditioning (AC vs oracle): {gap:+.2f}%")


if __name__ == "__main__":
    main()
