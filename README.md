# RL Optimal Liquidation

> **PPO matches Almgren–Chriss to ~3% mean gap in the linear regime, and beats AC by 1.2% mean under stochastic volatility, with verified closed-loop σ-conditioning.**

Reinforcement-learning framework for optimal trade execution. Liquidate `Q` shares over horizon `[0, T]` minimizing impact + inventory cost. Validates against Almgren–Chriss (1999) in the linear regime, then extends to settings where AC's closed form doesn't apply — and tests whether the agent learns genuine closed-loop control.

This execution module takes positions determined by an upstream allocator (see
[RL-Portfolio-Optimization](https://github.com/OuJiaPeng/RL-Portfolio-Optimization)) and
liquidates them under real-time market conditions. Volatility forecasts from
[DL-Volatility-Forecasting](https://github.com/OuJiaPeng/DL-Volatility-Forecasting) can
serve as the σ̂ input to the Phase 2 vol-conditional policy.

## Results

| Setting | Result | Notes |
|---|---|---|
| **Phase 1** — linear impact, AC has closed form | **2.84% mean gap to AC** across 5 seeds (range [1.26%, 4.37%]) | Validates pipeline against a known-optimal baseline |
| **Phase 2 vol** — stochastic σ, u-shape profile | **−1.22% mean gap (beats AC)** across 5 seeds | Agent captures 32% of the 3.79% available oracle value via closed-loop σ̂-conditioning |
| **Phase 2 spread** — stochastic η (per-step AR(1)) | Mixed: agent learns direction but mistunes magnitude | Informative negative — characterizes when on-policy RL needs recurrent state |

![Vol-conditioning probe](runs/phase2_n30_s0/diagnostics/vol_conditioning.png)

*Same agent, three σ profiles. Inventory trajectories (middle panel) diverge in the predicted direction: under inverted-U (σ high midday), the agent sells faster midday; under u-shape (σ high at boundaries), it sells faster at the ends. Direction-correct closed-loop control verified on **30/30 state grid cells**.*

## Methodology

| Component | Choice |
|---|---|
| Algorithm | PPO (Stable-Baselines3) with `VecNormalize`, `target_kl`, cost-based save-best |
| Policy | Custom **Beta(α, β)** on `[0, 1]` — avoids the clipped-Gaussian boundary bias that capped a default-PPO Gaussian at 5–9% gap |
| Reward | Direct execution cost (no shaping); `γ=1.0` for deterministically-terminating episodes |
| Baselines | Analytical AC + **50-parameter direct-opt** (validated env to −0.094%) + per-realization MC oracles (3.79% vol value, 7.55% spread value) |
| Validation | 5-seed multi-run on every shipped result; 17-test regression suite |

## Quickstart

```bash
make install
make test
make phase1          # ~5 min, validates against AC
make phase2-vol      # ~5 min, demonstrates closed-loop σ-conditioning
make phase2-spread   # ~5 min, the informative-negative spread experiment
make diagnose        # plots + summary.yaml for the most recent phase1 run
make oracle          # matched-pair value-of-conditioning oracles
```

For the full story — debugging chronology, dead-end pivots, methodology lessons —
see [`docs/phase1_journal.md`](docs/phase1_journal.md). The journal documents what we
tried, what failed, and what the diagnostic framework revealed about when on-policy RL
adds value over classical execution algorithms.
