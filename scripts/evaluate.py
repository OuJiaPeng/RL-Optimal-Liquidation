"""Evaluate PPO against the Almgren-Chriss baseline (and the analytical cost)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from rl_optimal_liquidation.baselines.almgren_chriss import ac_expected_cost
from rl_optimal_liquidation.diagnostics import ac_policy_fn, mean_cost
from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams
# Imported so pickle can resolve BetaActorCriticPolicy when loading a Beta-trained model.
from rl_optimal_liquidation.policies import BetaActorCriticPolicy  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="runs/phase1/best_model.zip",
                    help="Defaults to best-gap checkpoint; pass model.zip for the final one.")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10_000)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params = LiquidationParams(**cfg["env"])
    env = LiquidationEnv(params, seed=args.seed)

    print(f"Evaluating over {args.episodes} episodes (seed base {args.seed}).")

    # --- AC analytical (deterministic equivalent, drift = 0) ---
    analytic = ac_expected_cost(
        params.Q, params.T, params.N,
        params.lam, params.sigma, params.eta, params.gamma,
    )
    print(f"AC analytical cost   : {analytic:.4e}")

    # --- AC schedule rolled out in the env (so noise + terminal penalty included) ---
    ac_fn = ac_policy_fn(env)
    ac_mean, ac_std = mean_cost(env, ac_fn, args.episodes, seed=args.seed)
    print(f"AC rolled-out cost   : {ac_mean:.4e} +/- {ac_std:.4e}")

    # --- Learned PPO policy ---
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[warn] no model at {model_path}; skipping PPO eval.")
        return

    from stable_baselines3 import PPO
    model = PPO.load(str(model_path), device="cpu")

    def rl_fn(obs, deterministic=True):
        a, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    rl_mean, rl_std = mean_cost(env, rl_fn, args.episodes, seed=args.seed)
    print(f"PPO learned cost     : {rl_mean:.4e} +/- {rl_std:.4e}")
    gap_pct = (rl_mean - ac_mean) / ac_mean * 100.0
    print(f"Gap vs AC rollout    : {gap_pct:+.2f}%   (negative => PPO beats AC)")


if __name__ == "__main__":
    main()
