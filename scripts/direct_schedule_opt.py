"""Direct optimization of the 50-step trade schedule via Adam.

Diagnostic that the env / AC math are correct: under linear impact the 50-parameter
optimum recovers AC's cost to numerical precision, so the residual PPO gap is policy
parameterization, not env bugs. Also serves as the non-closed-form reference under
time-varying volatility.

Parameterization: f_k = sigmoid(theta_k), each fraction in (0,1). Cost is
deterministic given a schedule (no price-dependent term), so a single forward
pass gives the true expected cost; Adam handles the 50-d optimization.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from rl_optimal_liquidation.baselines.almgren_chriss import (
    ac_expected_cost,
    ac_inventory_path,
)
from rl_optimal_liquidation.envs import LiquidationParams


def schedule_cost(f: torch.Tensor, p: LiquidationParams, dt: float) -> torch.Tensor:
    """Deterministic execution cost for schedule f ∈ (0,1)^N (incl. terminal penalty).
    Uses `p.sigma_at(t_k)` so it honors time-varying volatility. Mirrors the env's
    per-step cost exactly."""
    q = torch.tensor(p.Q, dtype=f.dtype)
    total = torch.zeros((), dtype=f.dtype)
    for k in range(p.N):
        a = f[k] * q
        sigma_k = p.sigma_at(k * dt)  # python float; multiplies tensors fine
        temp = p.eta * a * a / dt
        perm = p.gamma * a * a
        inv = p.lam * sigma_k * sigma_k * q * q * dt
        total = total + temp + perm + inv
        q = q - a
    total = total + p.terminal_penalty * q * q
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None,
                    help="Override lambda (risk aversion). Default uses LiquidationParams (1e-6).")
    ap.add_argument("--sigma-profile", choices=["constant", "u_shaped"], default="constant")
    ap.add_argument("--sigma-amplitude", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    kwargs = {
        "sigma_profile": args.sigma_profile,
        "sigma_amplitude": args.sigma_amplitude,
    }
    if args.gamma is not None:
        kwargs["gamma"] = args.gamma
    if args.lam is not None:
        kwargs["lam"] = args.lam
    p = LiquidationParams(**kwargs)
    dt = p.T / p.N

    theta = torch.zeros(p.N, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)

    print(f"Optimizing 50-parameter schedule via Adam (linear impact, gamma={p.gamma}) ...")
    for step in range(args.steps + 1):
        opt.zero_grad()
        f = torch.sigmoid(theta)
        c = schedule_cost(f, p, dt)
        c.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:>5}: cost = {c.item():.6e}")

    direct_cost = c.item()
    f_direct = torch.sigmoid(theta).detach().numpy()

    print()
    print(f"  Direct-opt cost (linear): {direct_cost:.6e}")

    ac_analytic = ac_expected_cost(p.Q, p.T, p.N, p.lam, p.sigma, p.eta, p.gamma)
    q_ac = ac_inventory_path(p.Q, p.T, p.N, p.lam, p.sigma, p.eta)
    a_ac = q_ac[:-1] - q_ac[1:]
    f_ac = torch.tensor(a_ac / np.maximum(q_ac[:-1], 1e-9), dtype=torch.float64)
    ac_via_env = schedule_cost(f_ac, p, dt).item()
    print(f"  AC analytic cost          : {ac_analytic:.6e}")
    print(f"  AC schedule via env       : {ac_via_env:.6e}")
    print(f"  Gap direct vs AC analytic : {(direct_cost - ac_analytic) / ac_analytic * 100:+.4f}%")
    print(f"  Gap direct vs AC-via-env  : {(direct_cost - ac_via_env) / ac_via_env * 100:+.4f}%  <- learnable signal")
    print()
    print(f"  Direct f_k first 5: {np.array2string(f_direct[:5], precision=4)}")
    print(f"  AC     f_k first 5: {np.array2string(f_ac.numpy()[:5], precision=4)}")
    print(f"  Direct f_k last 5:  {np.array2string(f_direct[-5:], precision=4)}")
    print(f"  AC     f_k last 5:  {np.array2string(f_ac.numpy()[-5:], precision=4)}")


if __name__ == "__main__":
    main()
