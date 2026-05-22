"""Probe whether the trained Phase 2 agent reads sigma_hat from its observation
or merely memorized a time-keyed schedule that happens to coincide with the
U-shaped vol pattern it trained on.

Two complementary checks:

1. Out-of-distribution rollout. The training profile is u_shaped (high at
   open/close, low midday). At eval, swap in two profiles the agent never saw:
   flat (sigma_amplitude=0) and inverted-U (sigma_amplitude=-0.5; low at
   open/close, high midday). If the agent's per-step trading fraction f_k
   tracks the new sigma_hat pattern, it's reading the observation. If f_k
   is the same shape across all three profiles, it memorized the time schedule.

2. Controlled sensitivity. At a fixed (k/N, q/Q), query the policy at varying
   sigma_hat values. The action should be monotonically increasing in sigma_hat
   (high vol -> sell faster). Direct isolation, no rollout dynamics confound.

Saves a 2-panel plot (sigma_hat trajectories on top, f_k trajectories on
bottom) and prints the sensitivity table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import PPO

from rl_optimal_liquidation.diagnostics import rollout
from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams
from rl_optimal_liquidation.policies import BetaActorCriticPolicy  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/phase2_final_s0/best_model.zip")
    ap.add_argument("--config", default="configs/phase2_vol.yaml",
                    help="Must be the phase2_vol config (model expects 3-D obs with"
                         " sigma_hat); the phase1/default configs have include_vol_obs=False"
                         " and would produce a 2-D obs that mismatches the model.")
    ap.add_argument("--output", default=None,
                    help="Plot path. Defaults to <model_dir>/diagnostics/vol_conditioning.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_path = Path(args.model)
    out_path = Path(args.output) if args.output else model_path.parent / "diagnostics" / "vol_conditioning.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    base_env_kwargs = dict(cfg["env"])

    model = PPO.load(str(model_path), device="cpu")

    def rl_fn(obs, deterministic=True):
        a, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    profiles = [
        ("u-shape (trained)",  "u_shaped",  0.5),
        ("flat",               "constant",  0.0),
        ("inverted-U (OOD)",   "u_shaped", -0.5),
    ]

    results = {}
    for name, profile, amp in profiles:
        # Force sigma_noise_std=0 at eval so each rollout has a deterministic σ
        # trajectory (otherwise we can't read a clean action-vs-σ̂ signal).
        kwargs = {
            **base_env_kwargs,
            "sigma_profile": profile,
            "sigma_amplitude": amp,
            "sigma_noise_std": 0.0,
        }
        params = LiquidationParams(**kwargs)
        env = LiquidationEnv(params, seed=args.seed)
        env.reset(seed=args.seed)
        out = rollout(env, rl_fn)

        sigma_hat = np.array([
            params.sigma_at(k * params.dt()) / params.sigma for k in range(params.N)
        ])
        q_at_step = out["inventory"][:-1]
        f_k = out["actions"] / np.maximum(q_at_step, 1e-9)
        results[name] = {
            "sigma_hat": sigma_hat,
            "f_k": f_k,
            "q_over_Q": out["inventory"] / params.Q,  # length N+1
        }

    # --- Plot: 3 panels (sigma, inventory, action) ---
    n_steps = results[profiles[0][0]]["sigma_hat"].shape[0]
    steps = np.arange(n_steps)
    steps_q = np.arange(n_steps + 1)  # inventory has one extra point (q_0..q_N)

    fig, (ax_sigma, ax_q, ax_f) = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 3, 3]},
    )
    colors = ["C0", "C1", "C2"]

    for i, (name, _, _) in enumerate(profiles):
        ax_sigma.plot(steps, results[name]["sigma_hat"], color=colors[i], label=name)
        ax_q.plot(steps_q, results[name]["q_over_Q"], color=colors[i], label=name)
        ax_f.plot(steps, results[name]["f_k"], color=colors[i], label=name, marker="o", markersize=3)

    ax_sigma.set_ylabel(r"$\hat{\sigma}_k / \sigma_0$")
    ax_sigma.set_title("Vol-conditioning probe: same agent, three sigma profiles")
    ax_sigma.legend(loc="upper right")
    ax_sigma.grid(alpha=0.3)

    ax_q.set_ylabel(r"remaining inventory $q_k / Q$")
    ax_q.legend(loc="upper right")
    ax_q.grid(alpha=0.3)
    # Inventory trajectories diverge MUCH more visibly than f_k because they
    # integrate the conditioning. This is where closed-loop control is most legible.

    # Zoom the action plot to where the conditioning signal lives (early/mid),
    # then break the y-axis or just use log if terminal hits 1.0. Simpler: clip
    # the y-axis to [0, 0.15] which covers everything except the terminal spike.
    ax_f.set_ylim(0, 0.16)
    ax_f.set_xlabel("step $k$")
    ax_f.set_ylabel(r"trading fraction $f_k$  (zoomed)")
    ax_f.legend(loc="upper left")
    ax_f.grid(alpha=0.3)
    # Annotate that terminal spikes go off-scale
    ax_f.text(0.99, 0.95, "terminal f_k -> 1.0\n(off-scale; see q_k panel)",
              transform=ax_f.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")

    # --- Controlled sensitivity at fixed (k, q) ---
    print("\nControlled sensitivity at fixed state (k/N=0.2, q/Q=0.5):")
    print(f"  {'sigma_hat':>10}  {'action f':>10}")
    for sh in [0.5, 0.75, 1.0, 1.25, 1.5]:
        obs = np.array([0.2, 0.5, sh], dtype=np.float32)
        a = rl_fn(obs)
        print(f"  {sh:>10.2f}  {float(a[0]):>10.4f}")
    print("\n(If f monotonically increases with sigma_hat, the policy reads the vol obs.)")


if __name__ == "__main__":
    main()
