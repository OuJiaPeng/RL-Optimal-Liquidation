# RL Optimal Liquidation

> **This project measures when reinforcement learning adds value over Almgren–Chriss execution — and it usually doesn't.** We change one input of the problem (time-varying volatility), the problem stays in the linear-quadratic family, and certainty-equivalence AC (CE-AC) — re-estimate volatility, then re-solve the schedule — remains the optimal method. RL still earns a real result: the trained agent learns closed-loop volatility conditioning, beats the best static classical schedule on every seed (by **2.1 percentage points** on average), and captures **~73%** of the CE-AC edge. It never reaches the ceiling itself, because in this family nothing can. Every baseline in the comparison is an exact optimum, not a strawman.

The problem: liquidate `Q` shares over a horizon `[0, T]` while minimizing market impact plus
inventory risk — sell too fast and you pay impact, sell too slow and you carry risk. The classical
answer is Almgren–Chriss (2001): a deterministic selling schedule, solved in closed form before
trading starts, provably optimal under linear impact and quadratic costs. RL is worth trying
because that optimality is conditional on the model's inputs — when one drifts, the fixed schedule
goes stale, and a learned policy can in principle adapt closed-loop without deriving a new closed
form. Whether that flexibility buys anything measurable is the question this project answers, in
two phases: first recover AC where it is provably optimal (validating the pipeline), then degrade
one input and measure what closed-loop control is worth.

> **Full account — mechanism, caveats, and why it stops here: [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).**

## Results

| Phase | Result |
|---|---|
| **Phase 1** · recover (linear impact, AC provably optimal) | RL recovers AC to **+0.26–0.36%** (3 seeds) in the high-urgency regime κT=3, where recovery actually discriminates: there AC undercuts naive TWAP by ~32%, so landing near AC means finding the schedule's shape, not just any schedule. (κT is AC's urgency × horizon; at low κT nearly every schedule ties the optimum.) An independent 50-parameter direct optimizer reproduces the same optimum, confirming the environment math. |
| **Phase 2** · degrade one input (the certainty-equivalence ceiling) | Time-varying σ(t) with per-episode scale noise makes AC's *fixed* schedule stale. An exact classical ladder prices each rung on matched scenarios (paired 95% CIs; cost gaps vs naive AC): naive AC 0 · smart-static −1.33% (best *single* schedule knowing the σ distribution, zero conditioning) · **RL −3.41%** [−3.93, −2.89] (5/5 seeds) · CE-AC −4.66% (re-plan at the observed σ̂ — the ceiling for any causal policy). The agent beats every static schedule and captures most of the conditioning share of the ceiling. |

We stop at two phases deliberately. Inside the linear-quadratic family, RL can at best tie the
smart classical method, and making RL *necessary* means leaving that family. But the same break
that weakens the classical benchmark also removes the exact ladder that makes RL's gain
verifiable — and the agent's own noise floor doesn't shrink: it already leaves ~27% of the CE-AC
edge unclaimed in the one regime where everything is still exact. A harder problem has to clear a
wider floor with no referee to prove it did. The boundary is the finding.

![Vol-conditioning probe](docs/figures/vol_conditioning.png)

*The same agent under three σ profiles, two never seen in training. The inventory trajectories
(middle panel) diverge in the predicted direction: under the inverted-U (σ high midday) the agent
sells faster midday; under the U-shape (σ high at the boundaries) it sells faster at the ends. The
conditioning is verified causally, independently of cost: at fixed state, the action is monotone
in σ̂ on 26–30 of the 30 probe-grid cells on every seed. Reproduce (after `make phase2-vol`):
`python scripts/probe_conditioning_grid.py --model runs/p2_gauss_s0/best_model.zip --config configs/phase2_vol_gaussian.yaml`*

## Formulation

| | |
|---|---|
| **State** | (k/N, q_k/Q) — step and remaining inventory; Phase 2 adds the observed volatility σ̂_k/σ |
| **Action** | f_k ∈ [0, 1], fraction of remaining inventory to sell: a_k = f_k·q_k, so inventory can't go negative by construction |
| **Reward** | per-step execution cost, negated: −(η·a²/Δt + γ·a² + λ·σ(t)²·q²·Δt) — temporary impact, permanent impact, inventory risk — plus a soft terminal penalty −M·q_N² |
| **Horizon** | N = 50 steps, undiscounted (discount factor 1; γ above is the impact coefficient): episodes terminate deterministically, no reward shaping anywhere |

## Methodology

| Component | Choice |
|---|---|
| Algorithm | PPO (Stable-Baselines3). On-policy fits short, deterministically terminating episodes, and Phase 2 shows the binding constraint here is exploration, not sample reuse. `VecNormalize` reward scaling, `target_kl`, cost-based save-best. |
| Policy | Gaussian (shared log-σ), featured for Phase 2: 5/5 seeds discover σ̂-conditioning versus the Beta policy's 2/5, because state-independent exploration wins on a discovery-limited task. Beta(α, β) (`policy: beta`) recovers AC cleanly in Phase 1. |
| Baselines | Analytical AC + exact tridiagonal schedule solver + classical ladder (naive / smart-static / CE-AC on matched scenarios) + independent 50-parameter direct-opt validation |
| Validation | 5-seed protocol reporting best AND final · 30-cell conditioning probe grid · 15-test regression suite |

## Quickstart

```bash
make install
make test
make phase1          # ~5 min, validates against AC
make phase1-kt3      # ~5 min, the discriminating κT=3 regime (featured Phase 1 result)
make phase2-vol      # ~5 min, trains the featured vol-conditioning agent
make ladder          # classical ladder: naive AC / smart-static / CE-AC vs trained RL
make diagnose        # plots + summary.yaml for the phase1 run
```

Training outputs land in `runs/` (local, not tracked). Full derivations and the original
write-up: [`docs/RL_Optimal_Execution.pdf`](docs/RL_Optimal_Execution.pdf).
