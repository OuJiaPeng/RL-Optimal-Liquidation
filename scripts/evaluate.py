"""Evaluate PPO against the Almgren-Chriss baseline (and the analytical cost)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from rl_optimal_liquidation.baselines.almgren_chriss import ac_expected_cost
from rl_optimal_liquidation.diagnostics import ac_policy_fn, episode_costs_over_seeds
from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams
# Imported so pickle can resolve BetaActorCriticPolicy when loading a Beta-trained model.
from rl_optimal_liquidation.policies import BetaActorCriticPolicy  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--model", default="runs/phase1_s0/best_model.zip",
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
    # Only a valid anchor in the constant-sigma noise-free regime; under the
    # Phase 2 configs the env charges sigma_k = sigma*scale*profile(t), which
    # makes this constant-sigma number ~30% below the true expected AC cost.
    analytic = ac_expected_cost(
        params.Q, params.T, params.N,
        params.lam, params.sigma, params.eta, params.gamma,
    )
    lq_regime = params.sigma_profile == "constant" and params.sigma_noise_std == 0.0
    if lq_regime:
        print(f"AC analytical cost   : {analytic:.4e}")
    else:
        print(f"AC analytical cost   : {analytic:.4e}  "
              f"[constant-sigma reference only — env runs profile="
              f"{params.sigma_profile!r}, noise={params.sigma_noise_std}; "
              f"compare against the rolled-out AC cost below]")

    # --- AC schedule rolled out in the env (so noise + terminal penalty included) ---
    ac_fn = ac_policy_fn(env)
    ac_costs = episode_costs_over_seeds(env, ac_fn, args.episodes, seed=args.seed)
    ac_mean, ac_std = float(ac_costs.mean()), float(ac_costs.std())
    print(f"AC rolled-out cost   : {ac_mean:.4e} +/- {ac_std:.4e}")

    # --- Learned PPO policy ---
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[warn] no model at {model_path}; skipping PPO eval.")
        return

    from stable_baselines3 import PPO
    # vec_normalize.pkl is deliberately not loaded: training used norm_obs=False
    # (obs are already dimensionless by construction), and reward normalization
    # only shapes training gradients — eval costs stay in true cost units.
    model = PPO.load(str(model_path), device="cpu")

    def rl_fn(obs, deterministic=True):
        a, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    rl_costs = episode_costs_over_seeds(env, rl_fn, args.episodes, seed=args.seed)
    rl_mean, rl_std = float(rl_costs.mean()), float(rl_costs.std())
    print(f"PPO learned cost     : {rl_mean:.4e} +/- {rl_std:.4e}")

    # Same seeds for both arms -> per-episode costs are paired; the paired
    # difference removes the shared scenario variance, so the CI is on the
    # gap itself rather than on two marginal means.
    diffs = rl_costs - ac_costs
    gap_pct = float(diffs.mean()) / ac_mean * 100.0
    n = len(diffs)
    se_pct = float(diffs.std(ddof=1)) / np.sqrt(n) / ac_mean * 100.0 if n > 1 else float("nan")
    print(f"Gap vs AC rollout    : {gap_pct:+.2f}%  (95% CI [{gap_pct - 1.96 * se_pct:+.2f}%, "
          f"{gap_pct + 1.96 * se_pct:+.2f}%], paired, n={n})   (negative => PPO beats AC)")


if __name__ == "__main__":
    main()
