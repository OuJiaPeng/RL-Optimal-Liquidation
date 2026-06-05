"""Run a trained PPO checkpoint through the full PDF §6 diagnostic suite.

Outputs to `--output`:
    inventory.png         RL inventory band + (RL − AC) residual panel
    actions.png           per-step fraction sold f_k = a_k / q_k, RL vs AC
    costs.png             episode cost histograms RL vs AC with gap % in title
    summary.yaml          numeric summary (means, stds, IS gap, inventory L1 deviation)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from rl_optimal_liquidation.baselines.almgren_chriss import ac_expected_cost
from rl_optimal_liquidation.diagnostics import ac_policy_fn, episode_cost, rollout
from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams
from rl_optimal_liquidation.plots import (
    plot_action_fraction,
    plot_cost_distribution,
    plot_inventory_trajectory,
)
# Imported so pickle can resolve BetaActorCriticPolicy when loading a Beta-trained model.
from rl_optimal_liquidation.policies import BetaActorCriticPolicy  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="runs/phase1/best_model.zip",
                    help="Defaults to best-gap checkpoint; pass model.zip for the final one.")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--output", default="runs/phase1/diagnostics")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params = LiquidationParams(**cfg["env"])

    from stable_baselines3 import PPO
    # MlpPolicy is faster on CPU; loading with device='auto' would unnecessarily
    # go to CUDA and trigger SB3's warning.
    model = PPO.load(args.model, device="cpu")

    def rl_fn(obs, deterministic=True):
        a, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    env = LiquidationEnv(params, seed=args.seed)
    ac_fn = ac_policy_fn(env)

    N = args.episodes
    rl_costs = np.empty(N)
    ac_costs = np.empty(N)
    rl_inv = np.empty((N, params.N + 1))
    ac_inv = np.empty((N, params.N + 1))
    rl_act = np.empty((N, params.N))
    ac_act = np.empty((N, params.N))

    print(f"running {N} matched RL/AC rollouts ...")
    for i in range(N):
        seed = args.seed + i
        env.reset(seed=seed)
        r = rollout(env, rl_fn)
        rl_costs[i] = episode_cost(r["rewards"])
        rl_inv[i] = r["inventory"]
        rl_act[i] = r["actions"]

        env.reset(seed=seed)
        a = rollout(env, ac_fn)
        ac_costs[i] = episode_cost(a["rewards"])
        ac_inv[i] = a["inventory"]
        ac_act[i] = a["actions"]

    dt = params.T / params.N
    analytic = ac_expected_cost(
        params.Q, params.T, params.N,
        params.lam, params.sigma, params.eta, params.gamma,
    )
    plot_inventory_trajectory(rl_inv, ac_inv, dt=dt).savefig(out_dir / "inventory.png", dpi=150)
    plot_action_fraction(rl_inv, rl_act, ac_inv, ac_act).savefig(out_dir / "actions.png", dpi=150)
    plot_cost_distribution(rl_costs, ac_costs, ac_analytical=analytic).savefig(
        out_dir / "costs.png", dpi=150
    )

    ac_mean = float(ac_costs.mean())
    summary = {
        "episodes": int(N),
        "rl_cost_mean": float(rl_costs.mean()),
        "rl_cost_std": float(rl_costs.std()),
        "ac_cost_mean": ac_mean,
        "ac_cost_std": float(ac_costs.std()),
        "is_gap_pct": float((rl_costs.mean() - ac_mean) / ac_mean * 100.0) if ac_mean != 0 else None,
        "inv_l1_dev_mean": float(np.abs(rl_inv - ac_inv).mean()),
    }
    with open(out_dir / "summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"\nwrote diagnostics to {out_dir}")
    for k, v in summary.items():
        print(f"  {k:>18}: {v}")


if __name__ == "__main__":
    main()
