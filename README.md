# RL Optimal Liquidation

> **A framework for measuring *when* reinforcement learning adds value over Almgren–Chriss execution — and a structural reason it usually doesn't. When an input of the quadratic execution problem changes (here, time-varying volatility), the problem stays in the linear-quadratic family, so the smart classical method — certainty-equivalence AC — is already near-optimal and RL only *ties* it. RL's sole edge is the small value of conditioning over a *naive* fixed AC schedule.**

Reinforcement-learning framework for optimal trade execution. Liquidate `Q` shares over horizon
`[0, T]` minimizing market impact + inventory risk. Validates against Almgren–Chriss (1999) in the
linear regime where AC is *provably optimal*, then changes one input so AC's fixed schedule begins
to degrade — and tests whether the agent learns genuine closed-loop control that keeps up with the
adaptive classical method.

> **📋 Single source of truth — full status, the three-phase arc, and the unifying certainty-equivalence finding: [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).**

This execution module takes positions determined by an upstream allocator (see
[RL-Portfolio-Optimization](https://github.com/OuJiaPeng/RL-Portfolio-Optimization)) and
liquidates them under real-time market conditions. Volatility forecasts from
[DL-Volatility-Forecasting](https://github.com/OuJiaPeng/DL-Volatility-Forecasting) can
serve as the σ̂ input to the Phase 2 vol-conditional policy.

## Results

| Phase | Result |
|---|---|
| **Phase 1** — recover (linear impact, AC provably optimal) | RL **recovers** AC to **2.84% mean** (5 seeds, best-validated checkpoint; range [1.26%, 4.37%]). Finals don't converge — a documented terminal-penalty *cliff*, not the env. Direct-opt validates the env + AC to **−0.094%**. *The instrument works.* |
| **Phase 2** — degrade one input (the **certainty-equivalence ceiling**) | Change a single input — time-varying σ — and AC's *fixed* schedule degrades. The problem stays in the linear-quadratic family, so certainty-equivalence AC is near-optimal and **RL only ties it**; RL's edge is the value of conditioning over *naive* AC. Best-checkpoint **−1.22% mean vs naive AC**; closed-loop σ-conditioning verified on a **30×30 (k, q) grid**, all 5 seeds (oracle 3.79%, agent captures ~32%). |
| **Phase 3** — open | Blank slate — where, *if anywhere*, RL becomes genuinely *necessary* over the best classical method. Not yet started. |

Mechanism and the full finding: [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

![Vol-conditioning probe](docs/figures/vol_conditioning.png)

*Same agent, three σ profiles. Inventory trajectories (middle panel) diverge in the predicted direction: under inverted-U (σ high midday), the agent sells faster midday; under u-shape (σ high at boundaries), it sells faster at the ends. Direction-correct closed-loop control verified on **30×30 state-grid cells**, monotone in σ̂ on all 5 seeds (reproduce: `scripts/probe_conditioning_grid.py`).*

> **Run-dir naming:** the vol experiment trains from `configs/phase2_vol.yaml`; local run dirs are named `runs/phase2_n30_*` (`n30` = per-episode σ-noise **0.30**, with N=50 — *noise-0.30*, not step count).

## Methodology

| Component | Choice |
|---|---|
| Algorithm | PPO (Stable-Baselines3) with `VecNormalize`, `target_kl`, cost-based save-best |
| Policy | Custom **Beta(α, β)** on `[0, 1]` — avoids the clipped-Gaussian boundary bias that capped a default-PPO Gaussian at 5–9% gap |
| Reward | Direct execution cost (no shaping); `γ=1.0` for deterministically-terminating episodes |
| Baselines | Analytical AC + **50-parameter direct-opt** (validated env to −0.094%) + per-realization MC oracle (3.79% vol value of conditioning) |
| Validation | 5-seed multi-run (best AND final); state-space sensitivity probe (30×30 conditioning grid); regression suite |

## Quickstart

```bash
make install
make test
make phase1          # ~5 min, validates against AC
make phase2-vol      # ~5 min, demonstrates closed-loop σ-conditioning
make diagnose        # plots + summary.yaml for the most recent phase1 run
make oracle          # matched-pair value-of-conditioning oracle
```

For the full story — debugging chronology, dead-end pivots, methodology lessons —
see [`docs/phase1_journal.md`](docs/phase1_journal.md). The journal documents what we
tried, what failed, and what the diagnostic framework revealed about when on-policy RL
adds value over classical execution algorithms.
