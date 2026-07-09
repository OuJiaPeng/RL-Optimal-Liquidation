# RL Optimal Liquidation - Project Status

This is the longer version of the README.

## Short Version

The repo asks whether PPO adds value to Almgren-Chriss execution.

The answer:

- When the Almgren-Chriss assumptions hold, PPO recovers the classical schedule.
- When volatility varies by episode, PPO learns to condition on observed volatility and beats static
  schedules.
- Certainty-equivalence AC still wins because the problem remains linear-quadratic.

That last point matters. The agent learns something real, but the best classical method is still the ceiling
for this model class.

## Setup

The task is to liquidate `Q` shares over a fixed horizon. The cost has three pieces:

- temporary impact,
- permanent impact,
- inventory risk.

The environment is short and deterministic in time. Each episode ends after `N = 50` steps. The action is the
fraction of remaining inventory to sell, so inventory cannot go negative by construction.

## Pass 1: Recover Almgren-Chriss

The first test keeps the textbook assumptions. Almgren-Chriss is optimal, so PPO should recover it.

The useful recovery regime is `kappa T = 3`. In easier regimes, many schedules look almost optimal, so
recovery does not tell you much. In `kappa T = 3`, AC beats TWAP by about 32%, and the PPO agents land within
**0.26-0.36%** of AC across 3 seeds.

An independent 50-parameter direct optimizer also reproduces the environment optimum. That checks that the
reward and the classical cost formula agree.

One caveat: final checkpoints are less stable than best checkpoints. The last-step terminal penalty creates a
cliff, so small late-training drift can leave inventory and cause a large cost jump. The reported recovery
number is the best validated checkpoint, which is the version you would actually deploy.

## Pass 2: Volatility Conditioning

Phase 2 changes one input: volatility varies over time and by episode. A fixed AC schedule computed at base
volatility goes stale.

The comparison ladder is:

| Method | Description | Gap vs naive AC |
|---|---|---:|
| Naive AC | Fixed schedule from base volatility | 0 |
| Smart-static | Best single static schedule for the volatility distribution | **-1.33%** |
| PPO | Policy conditioned on observed volatility | **-3.41%** |
| CE-AC | Replan from observed volatility | **-4.66%** |

Lower is better. PPO beats the static schedules on all 5 seeds. It captures about **73%** of the CE-AC edge.

The conditioning is not just a rollout artifact. Probe-grid tests fix `(time, inventory)` and vary observed
volatility. The action increases with volatility on 26-30 of 30 probe cells on every seed.

## Why CE-AC Still Wins

The problem stays linear-quadratic. Volatility uncertainty enters through the inventory-risk term, and the
episode scale is observable. Re-estimate volatility, solve the AC schedule again, and you get the best causal
policy in this setting.

That is why CE-AC beats PPO. There is no hidden nonlinear structure for the agent to exploit.

## Why The Project Stops Here

You can make execution problems where RL is more tempting: nonlinear impact, hidden regimes, cross-impact,
stochastic arrivals, tail-risk objectives. The tradeoff is that the exact classical ladder disappears.

This project keeps the exact ladder and accepts the boundary it reveals:

- PPO can recover the known optimum.
- PPO can learn a useful conditioning signal.
- PPO does not beat the exact certainty-equivalence solution when that solution exists.

That is a cleaner result than forcing a noisier benchmark win.

## Method Notes

- PPO uses Stable-Baselines3.
- The featured Phase 2 policy is Gaussian; it discovered volatility conditioning more reliably than the Beta
  policy in this setup.
- Evaluation uses matched scenarios and paired confidence intervals.
- Tests pin the AC formulas, exact solver, and environment-cost agreement.

## Reproduce

```bash
make test
python scripts/direct_schedule_opt.py
make phase1
make phase1-kt3
make phase2-vol
make ladder
```
