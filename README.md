# RL Optimal Liquidation

This repo studies a simple execution question:

Can a reinforcement-learning agent improve on Almgren-Chriss liquidation, or does the classical schedule
already solve the problem?

The answer is mostly classical. When the assumptions of Almgren-Chriss hold, PPO recovers the optimal
schedule. When volatility varies by episode, the agent learns useful closed-loop conditioning and beats every
static schedule. It still does not beat certainty-equivalence AC, which replans from the observed volatility
and remains the exact ceiling in this linear-quadratic setting.

## The Problem

You need to sell `Q` shares over a fixed horizon.

Sell too fast and you pay market impact. Sell too slowly and you carry inventory risk. Almgren-Chriss gives a
closed-form schedule for this tradeoff under linear impact and quadratic costs.

This repo asks what happens when a learned policy is measured against that classical ladder instead of against
a weak baseline.

## Results

| Pass | Question | Result |
|---|---|---|
| Recover AC | Can PPO find the known optimum? | Yes. In the discriminating `kappa T = 3` regime, PPO lands within **0.26-0.36%** of AC across 3 seeds. |
| Vol conditioning | Can PPO react when volatility changes? | Yes. It beats the best static schedule on all 5 seeds and captures about **73%** of the CE-AC edge. |
| Ceiling check | Does PPO beat the best classical causal method? | No. CE-AC remains best, as the linear-quadratic theory predicts. |

The useful result is the middle one. The agent learns to sell faster when observed volatility is higher and
slower when it is lower. Probe-grid tests confirm the conditioning at fixed state, not just in rollout cost.

![Vol-conditioning probe](docs/figures/vol_conditioning.png)

## What Gets Compared

The Phase 2 ladder uses matched scenarios:

| Method | Meaning | Gap vs naive AC |
|---|---|---:|
| Naive AC | Fixed schedule from base volatility | 0 |
| Smart-static | Best single schedule for the volatility distribution | **-1.33%** |
| PPO | Learned policy conditioned on observed volatility | **-3.41%** |
| CE-AC | Replan after observing episode volatility | **-4.66%** |

Lower cost is better. CE-AC is not a strawman or a loose benchmark; it is the ceiling for this setup.

## How It Works

- **State:** time step and remaining inventory; Phase 2 also includes observed volatility.
- **Action:** fraction of remaining inventory to sell.
- **Reward:** negative execution cost: temporary impact, permanent impact, and inventory risk.
- **Algorithm:** PPO with Stable-Baselines3.
- **Baselines:** analytic AC, exact tridiagonal solver, smart-static schedules, CE-AC.

## Claim Boundary

This project stays inside the linear-quadratic family on purpose. That gives exact classical baselines. It
also means RL should not beat the best classical method.

The finding is a boundary: PPO can learn the right conditioning signal, but in this model class the clean
certainty-equivalence solution still wins.

## Quickstart

```bash
make install
make test
make phase1          # recover AC
make phase1-kt3      # featured recovery regime
make phase2-vol      # train the volatility-conditioning agent
make ladder          # compare naive AC / smart-static / PPO / CE-AC
make diagnose        # plots and run summary
```

Training outputs land in `runs/`, which is local and ignored. The longer account is in
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).
