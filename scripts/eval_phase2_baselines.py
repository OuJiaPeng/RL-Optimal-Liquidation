"""Phase 2 classical-baseline decomposition (matched-pair, exact schedules).

The historical oracle probe (archive/superseded/probe_vol_oracle.py) measured
"value of conditioning" as (naive AC - perfect-foresight oracle). That gap
conflates two different kinds of knowledge, and this script separates them
with two baselines the repo previously lacked:

  1. naive AC       — constant-sigma closed form, as deployed everywhere else.
  2. smart-static   — the best SINGLE schedule for a desk that knows the sigma
                      profile and the scale distribution but never observes the
                      realized scale. Because episode cost is linear in scale^2
                      given a fixed trade schedule, this is exactly the optimal
                      schedule at effective risk weights lam * E[scale^2] *
                      sigma_profile(t)^2 — no conditioning involved.
  3. CE-AC = oracle — per-episode re-plan at the observed sigma_hat. In this
                      env the first sigma_hat observation reveals the episode
                      scale exactly (deterministic profile x scale), so
                      certainty-equivalent re-planning coincides with the
                      perfect-foresight oracle: it is the ceiling for ANY
                      causal policy, including the RL agent.
  4. RL agent(s)    — trained checkpoints rolled out on the same scale draws.

All arms are priced on the SAME per-episode scale draws (matched pairs), so
gaps carry paired 95% CIs. Schedules are exact minimizers of the env's own
cost (including the soft terminal penalty M*q_N^2): with pre-trade inventory
risk the objective is an unconstrained convex quadratic in (q_1..q_N), solved
via its tridiagonal normal equations — no iterative optimizer, no tolerance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from rl_optimal_liquidation.baselines.almgren_chriss import ac_expected_cost, ac_inventory_path
from rl_optimal_liquidation.baselines.exact_lq import (
    cost_of_trades,
    exact_optimal_trades,
    expected_scale_sq,
    sigma_profile_path,
)
from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams
# Imported so pickle can resolve BetaActorCriticPolicy when loading a Beta-trained model.
from rl_optimal_liquidation.policies import BetaActorCriticPolicy  # noqa: F401


def rl_costs(model_path: Path, p: LiquidationParams, scales: np.ndarray, seed: int) -> np.ndarray:
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    env = LiquidationEnv(p, seed=seed)
    costs = np.empty(len(scales))
    for i, s in enumerate(scales):
        env.reset(seed=seed + i)
        # Force the matched per-episode scale (params have noise 0, so reset
        # left the scale at 1.0) and rebuild the initial observation with it.
        env._sigma_scale = float(s)
        obs = env._obs()
        total, done = 0.0, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32).reshape(-1))
            total += rew
            done = term or trunc
        costs[i] = -total
    return costs


def paired_gap(costs: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(gap %, 95% halfwidth %) of costs vs ref on matched episodes."""
    d = costs - ref
    ref_mean = ref.mean()
    gap = float(d.mean()) / ref_mean * 100.0
    half = 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d)) / ref_mean * 100.0
    return gap, half


def validate_exact_solver() -> None:
    """At Phase 1 params the exact soft-terminal optimum must sit -0.09% below
    the hard-liquidation AC closed form (the documented direct-opt residual)."""
    p = LiquidationParams()  # Phase 1 defaults: lam=1e-6, sigma=0.3, M=1e-3
    path = np.full(p.N, p.sigma)
    exact = cost_of_trades(exact_optimal_trades(p, path), p, path)
    ac = ac_expected_cost(p.Q, p.T, p.N, p.lam, p.sigma, p.eta, p.gamma)
    gap = (exact - ac) / ac * 100.0
    print(f"[validate] exact optimum vs AC closed form at Phase 1 params: {gap:+.4f}% "
          f"(expected ~-0.094%: the soft terminal penalty M=1e-3 makes leaving "
          f"~4.8% of the last step's inventory optimal)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2_vol.yaml")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=777,
                    help="Seed for the scale draws and RL rollouts; distinct from "
                         "the training-callback (1e6+) and evaluate.py (1e4+) seeds.")
    ap.add_argument("--models", default="runs/phase2_n30_s*/best_model.zip",
                    help="Glob of trained checkpoints to include (empty matches skip RL).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_kwargs = dict(cfg["env"])
    noise_std = float(env_kwargs.get("sigma_noise_std", 0.0))
    env_kwargs["sigma_noise_std"] = 0.0  # scales are drawn/forced externally
    p = LiquidationParams(**env_kwargs)

    validate_exact_solver()

    rng = np.random.default_rng(args.seed)
    scales = np.maximum(1.0 + rng.normal(0.0, noise_std, args.episodes), 0.1)
    profile = sigma_profile_path(p)
    e_s2 = expected_scale_sq(noise_std)
    print(f"\nConfig {args.config}: profile={p.sigma_profile}, noise_std={noise_std}, "
          f"E[scale^2]={e_s2:.4f}, episodes={args.episodes}")

    # --- classical arms (all priced on the same realized sigma paths) ---
    q_ac = ac_inventory_path(p.Q, p.T, p.N, p.lam, p.sigma, p.eta)
    trades_naive = q_ac[:-1] - q_ac[1:]
    trades_static = exact_optimal_trades(p, np.sqrt(e_s2) * profile)

    naive = np.empty(args.episodes)
    static = np.empty(args.episodes)
    oracle = np.empty(args.episodes)
    for i, s in enumerate(scales):
        path = s * profile
        naive[i] = cost_of_trades(trades_naive, p, path)
        static[i] = cost_of_trades(trades_static, p, path)
        oracle[i] = cost_of_trades(exact_optimal_trades(p, path), p, path)

    # --- RL arms ---
    model_paths = sorted(Path(".").glob(args.models))
    rl_by_model: dict[str, np.ndarray] = {}
    for mp in model_paths:
        name = mp.parent.name
        print(f"rolling out {mp} ...")
        rl_by_model[name] = rl_costs(mp, p, scales, seed=args.seed)

    # --- report ---
    def row(label: str, costs: np.ndarray | None) -> None:
        if costs is None:
            print(f"  {label:<28} {'':>12}  --")
            return
        gap, half = paired_gap(costs, naive)
        print(f"  {label:<28} {costs.mean():>12.4e}  {gap:+6.2f}%  "
              f"[{gap - half:+6.2f}%, {gap + half:+6.2f}%]")

    print(f"\n=== Matched-pair costs vs naive AC (paired 95% CI, n={args.episodes}) ===")
    print(f"  {'arm':<28} {'mean cost':>12}  gap vs naive AC")
    row("naive AC (deployed)", None)
    row("smart-static (no cond.)", static)
    row("CE-AC = oracle (ceiling)", oracle)
    for name, costs in rl_by_model.items():
        row(f"RL {name}", costs)
    if rl_by_model:
        rl_mean = np.mean(list(rl_by_model.values()), axis=0)
        row("RL mean of seeds", rl_mean)

    static_gap, _ = paired_gap(static, naive)
    oracle_gap, _ = paired_gap(oracle, naive)
    print("\n=== Decomposition of the CE-AC edge over naive AC ===")
    print(f"  profile + noise-distribution knowledge (static, no conditioning): "
          f"{-static_gap:.2f}pp")
    print(f"  genuine per-episode conditioning (CE-AC minus smart-static):      "
          f"{-(oracle_gap - static_gap):.2f}pp")
    if rl_by_model:
        rl_gap, rl_half = paired_gap(rl_mean, naive)
        vs_static_gap, vs_static_half = paired_gap(rl_mean, static)
        print(f"\n  RL mean gap vs naive AC:     {rl_gap:+.2f}% "
              f"(captures {rl_gap / oracle_gap * 100.0:.0f}% of the CE-AC edge)")
        print(f"  RL mean gap vs smart-static: {vs_static_gap:+.2f}%  "
              f"[{vs_static_gap - vs_static_half:+.2f}%, {vs_static_gap + vs_static_half:+.2f}%]")
        print("  (negative vs smart-static => the agent's conditioning adds value beyond")
        print("   ANY static schedule; positive => its edge over naive AC was mostly")
        print("   profile knowledge a static classical schedule already captures.)")


if __name__ == "__main__":
    main()
