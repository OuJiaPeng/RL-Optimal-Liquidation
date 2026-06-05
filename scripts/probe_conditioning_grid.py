"""Grid sweep of closed-loop vol-conditioning: does the trained policy respond to its
sigma_hat observation monotonically in the correct direction, across the WHOLE
(k/N, q/Q) state space — not just one point?

This is the script that backs the "monotonic, direction-correct on N/N grid cells"
claim. `probe_vol_conditioning.py` only checks a single (k/N=0.2, q/Q=0.5) state;
this sweeps a full grid and counts how many cells pass.

obs = [k/N, q/Q, sigma_hat]; correct direction is f INCREASING in sigma_hat
(high vol -> sell faster).

Reports: cells passing monotonic+direction-correct / total, and the max & mean
per-cell sensitivity = |f(hat_max) - f(hat_min)|.
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml
from stable_baselines3 import PPO

from rl_optimal_liquidation.policies import BetaActorCriticPolicy  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k-grid", type=int, default=6, help="number of k/N grid points")
    ap.add_argument("--q-grid", type=int, default=5, help="number of q/Q grid points")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]
    # Sanity: the model's obs must include the conditioning channel.
    if not env_cfg.get("include_vol_obs"):
        print("WARNING: config has include_vol_obs=false; obs will lack sigma_hat")

    model = PPO.load(args.model, device="cpu")

    def act(obs):
        a, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return float(np.asarray(a).reshape(-1)[0])

    # Conditioning-variable sweep values and the "correct" monotone direction.
    hats = [0.5, 0.75, 1.0, 1.25, 1.5]
    direction = +1  # f should increase with sigma_hat
    hat_name = "sigma_hat"

    ks = np.linspace(0.1, 0.9, args.k_grid)
    qs = np.linspace(0.2, 1.0, args.q_grid)

    total = 0
    passed = 0
    sensitivities = []
    rows = []
    for kN in ks:
        for qQ in qs:
            fs = [act([kN, qQ, h]) for h in hats]
            # direction-correct monotonicity: sign(f[i+1]-f[i]) == direction (allow ties)
            diffs = np.diff(fs)
            mono = bool(np.all(direction * diffs >= -1e-6))  # tolerate tiny numerical wobble
            sens = abs(fs[-1] - fs[0])
            sensitivities.append(sens)
            total += 1
            passed += int(mono)
            rows.append((kN, qQ, fs, mono, sens))

    print(f"\nGrid vol-conditioning probe ({hat_name}), model={args.model}")
    print(f"  grid = {args.k_grid} (k/N) x {args.q_grid} (q/Q) = {total} cells")
    print(f"  sweep {hat_name} over {hats}, correct direction = increasing f\n")
    print(f"  {'k/N':>5} {'q/Q':>5}   " + "  ".join(f"{h:>6}" for h in hats) + "   mono?   sens")
    for kN, qQ, fs, mono, sens in rows:
        fstr = "  ".join(f"{v:6.3f}" for v in fs)
        print(f"  {kN:5.2f} {qQ:5.2f}   {fstr}   {'OK ' if mono else 'NO '}   {sens:.4f}")

    print(f"\n  monotonic direction-correct cells: {passed}/{total}")
    print(f"  max per-cell sensitivity:  {max(sensitivities):.4f}")
    print(f"  mean per-cell sensitivity: {np.mean(sensitivities):.4f}")


if __name__ == "__main__":
    main()
