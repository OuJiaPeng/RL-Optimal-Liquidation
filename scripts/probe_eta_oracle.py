"""Spread-conditioning oracle diagnostic.

Computes the perfect-foresight cost: given the full eta_k trajectory of an
episode, what's the optimal 50-step trading schedule? This is an UPPER BOUND
on what any closed-loop policy could achieve — a true online policy can only
condition on past+present eta, not future, so its expected cost is >= the
perfect-foresight cost.

If perfect-foresight ~ AC: spread-conditioning has essentially zero value at
this noise level. The agent's negative result is "nothing to capture, agent
just trips over its own weak conditioning."

If perfect-foresight << AC: conditioning has real value. The agent's failure
to capture it is a PPO/architecture limitation, not a problem-structure one.

The diagnostic we should have run before the spread training. Closes the
methodology loop opened in the concave-impact and vol experiments.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from rl_optimal_liquidation.baselines.almgren_chriss import ac_inventory_path
from rl_optimal_liquidation.envs import LiquidationParams


def cost_of_schedule(f_array: np.ndarray, eta_traj: np.ndarray, p: LiquidationParams) -> float:
    """Per-step cost of a given schedule (fractions) under a given eta trajectory.
    Textbook AC linear branch: no per-step perm cost."""
    dt = p.T / p.N
    q = p.Q
    total = 0.0
    for k in range(p.N):
        a = f_array[k] * q
        total += eta_traj[k] * a * a / dt + p.lam * p.sigma * p.sigma * q * q * dt
        q = q - a
    total += p.terminal_penalty * q * q
    return float(total)


def simulate_eta_trajectory(p: LiquidationParams, rng: np.random.Generator) -> np.ndarray:
    """Sample one episode's per-step eta path under the env's log-AR(1) process."""
    dt = p.T / p.N
    log_mean = float(np.log(p.eta))
    log_eta = log_mean
    eta_traj = np.empty(p.N)
    # Match env semantics: eta_0 = eta_mean (set in reset()), then evolve.
    eta_traj[0] = p.eta
    for k in range(1, p.N):
        eps = float(rng.normal())
        log_eta = (
            (1.0 - p.eta_rho) * log_mean
            + p.eta_rho * log_eta
            + p.eta_noise_std * np.sqrt(dt) * eps
        )
        eta_traj[k] = float(np.exp(log_eta))
    return eta_traj


def perfect_foresight_cost(eta_traj: np.ndarray, p: LiquidationParams, n_iter: int = 2500) -> float:
    """Cost of the best 50-step schedule that knows the full eta trajectory."""
    dt = p.T / p.N
    eta_t = torch.tensor(eta_traj, dtype=torch.float64)
    theta = torch.zeros(p.N, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    for _ in range(n_iter):
        opt.zero_grad()
        f = torch.sigmoid(theta)
        q = torch.tensor(p.Q, dtype=f.dtype)
        total = torch.zeros((), dtype=f.dtype)
        for k in range(p.N):
            a = f[k] * q
            sigma_k = p.sigma  # constant in this experiment
            # Textbook AC linear branch: no per-step perm cost (drops as constant).
            total = total + eta_t[k] * a * a / dt + p.lam * sigma_k * sigma_k * q * q * dt
            q = q - a
        total = total + p.terminal_penalty * q * q
        total.backward()
        opt.step()
    return float(total.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2_spread.yaml",
                    help="env params loaded from this config; keeps oracle in lock-step with training")
    ap.add_argument("--episodes", type=int, default=40,
                    help="MC samples of eta trajectories")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eta-noise-std", type=float, default=None,
                    help="Override eta_noise_std from config. 0.3 = narrow (range ~[0.84, 1.16]); "
                         "2.0 = wide (range ~[0.5, 2.0]). If None, uses the config value.")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)

    # Load env params from config so this oracle stays consistent with training.
    # If --eta-noise-std is given, override; else use whatever's in the config.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_kwargs = dict(cfg["env"])
    if args.eta_noise_std is not None:
        env_kwargs["eta_noise_std"] = args.eta_noise_std
    p = LiquidationParams(**env_kwargs)
    print(f"Config: eta_noise_std={p.eta_noise_std}, rho={p.eta_rho}, lam={p.lam}")

    # AC schedule (fixed, based on constant-eta formula at eta_mean) — apply this
    # to each MC eta sample to get a matched-pair AC cost. This is what a "blind"
    # AC policy would achieve in the stochastic-eta env, computed on the same
    # realizations as the oracle for direct comparison.
    q_ac = ac_inventory_path(p.Q, p.T, p.N, p.lam, p.sigma, p.eta)
    a_ac = q_ac[:-1] - q_ac[1:]
    f_ac = a_ac / np.maximum(q_ac[:-1], 1e-9)

    rng = np.random.default_rng(args.seed)
    print(f"Computing oracle + AC-matched costs on {args.episodes} eta realizations...")
    oracle_costs = []
    ac_costs = []
    eta_ranges = []
    for i in range(args.episodes):
        eta_traj = simulate_eta_trajectory(p, rng)
        oracle_costs.append(perfect_foresight_cost(eta_traj, p))
        ac_costs.append(cost_of_schedule(f_ac, eta_traj, p))
        eta_ranges.append((eta_traj.min() / p.eta, eta_traj.max() / p.eta))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.episodes}: oracle={np.mean(oracle_costs):.4e}  ac={np.mean(ac_costs):.4e}")

    oracle_mean = float(np.mean(oracle_costs))
    oracle_std = float(np.std(oracle_costs))
    ac_mean = float(np.mean(ac_costs))
    ac_std = float(np.std(ac_costs))
    eta_lo = np.mean([r[0] for r in eta_ranges])
    eta_hi = np.mean([r[1] for r in eta_ranges])

    print()
    print(f"On the same {args.episodes} MC eta realizations:")
    print(f"  Perfect-foresight oracle:             {oracle_mean:.4e}  (std {oracle_std:.2e})")
    print(f"  AC schedule (blind to eta path):      {ac_mean:.4e}  (std {ac_std:.2e})")
    print(f"  Average eta range per episode:        [{eta_lo:.3f}, {eta_hi:.3f}] x eta_mean")
    print()
    gap_oracle_vs_ac = (ac_mean - oracle_mean) / ac_mean * 100.0
    print(f"Value of perfect eta-conditioning (AC vs oracle):  {gap_oracle_vs_ac:+.2f}%")
    print()
    if gap_oracle_vs_ac < 1.0:
        print("Interpretation: <1% gap => no value to capture at this eta variation.")
    elif gap_oracle_vs_ac < 3.0:
        print("Interpretation: modest value (1-3%). Worth training, may be marginal.")
    else:
        print("Interpretation: significant value (>3%). Training is warranted.")


if __name__ == "__main__":
    main()
