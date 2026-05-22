"""Re-evaluate existing best_model checkpoints under the textbook-AC env.

Run this after the textbook-AC cleanup to confirm previously-shipped numbers
(Phase 1 ~2.84% mean gap, Phase 2 vol ~−1.22% mean gap, etc.) still hold
under the updated cost formulation. We expect the gap to shift by ≲ 0.5%
relative — well inside the per-seed variance bands documented in the journal.

The cleanup removed `gamma * a^2` from the env's per-step reward (textbook
AC: permanent impact is a schedule-independent constant and drops from the
optimization). Existing checkpoints were trained against the slightly
different cost; this script measures whether the policies they learned are
still near-optimal under the corrected cost.

Output: one row per `runs/<phase>_s<seed>` directory containing a
`best_model.zip`, with both the pre-cleanup gap (read from the run's
`diagnostics/summary.yaml` if present) and the freshly-measured post-cleanup
gap on 200 matched-pair episodes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from rl_optimal_liquidation.diagnostics import ac_policy_fn, episode_cost, rollout
from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams
# Imported so pickle can resolve BetaActorCriticPolicy when loading.
from rl_optimal_liquidation.policies import BetaActorCriticPolicy  # noqa: F401


def measure_gap(model_path: Path, config_path: Path, episodes: int, seed_base: int):
    """Return (rl_mean, ac_mean, gap_pct) on `episodes` matched-pair rollouts."""
    from stable_baselines3 import PPO

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    params = LiquidationParams(**cfg["env"])
    env = LiquidationEnv(params, seed=seed_base)
    ac_fn = ac_policy_fn(env)
    model = PPO.load(str(model_path), device="cpu")

    def rl_fn(obs, deterministic=True):
        a, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    rl_costs = np.empty(episodes)
    ac_costs = np.empty(episodes)
    for i in range(episodes):
        s = seed_base + i
        env.reset(seed=s)
        rl_costs[i] = episode_cost(rollout(env, rl_fn)["rewards"])
        env.reset(seed=s)
        ac_costs[i] = episode_cost(rollout(env, ac_fn)["rewards"])
    rl_mean = float(rl_costs.mean())
    ac_mean = float(ac_costs.mean())
    gap = (rl_mean - ac_mean) / ac_mean * 100.0 if ac_mean != 0.0 else float("nan")
    return rl_mean, ac_mean, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs",
                    help="Parent dir containing <name>_s<seed>/best_model.zip subdirs")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--phase", choices=["phase1", "phase2_vol", "phase2_spread", "auto"],
                    default="auto",
                    help="Which config to use. 'auto' guesses from dir-name prefix.")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"no runs dir at {runs_dir}")

    phase_to_config = {
        "phase1": Path("configs/phase1.yaml"),
        "phase2_vol": Path("configs/phase2_vol.yaml"),
        "phase2_spread": Path("configs/phase2_spread.yaml"),
    }

    def guess_phase(name: str) -> str | None:
        # name like "phase1_final_s0", "phase2_n30_s0", "phase2_eta2_s0"
        if name.startswith("phase1"):
            return "phase1"
        if "vol" in name or "n30" in name or "n15" in name:
            # n30/n15 = old sigma_noise=0.3/0.15 vol runs
            return "phase2_vol"
        if "eta" in name or "spread" in name:
            return "phase2_spread"
        return None

    print(f"{'run':<28}  {'config':<20}  {'old_gap':>10}  {'new_gap':>10}  {'shift':>8}")
    print("-" * 90)

    for run in sorted(runs_dir.iterdir()):
        model = run / "best_model.zip"
        if not model.exists():
            continue
        phase = args.phase if args.phase != "auto" else guess_phase(run.name)
        if phase is None:
            print(f"{run.name:<28}  (skip: can't guess phase)")
            continue
        cfg_path = phase_to_config[phase]
        old_gap = None
        summary_path = run / "diagnostics" / "summary.yaml"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = yaml.safe_load(f) or {}
            old_gap = summary.get("is_gap_pct")
        rl_mean, ac_mean, new_gap = measure_gap(model, cfg_path, args.episodes, args.seed)
        if old_gap is None:
            shift_str = "       n/a"
            old_str = "       n/a"
        else:
            shift = new_gap - old_gap
            shift_str = f"{shift:+8.3f}pp"
            old_str = f"{old_gap:+9.3f}%"
        print(f"{run.name:<28}  {phase:<20}  {old_str:>10}  {new_gap:+9.3f}%  {shift_str:>8}")


if __name__ == "__main__":
    main()
