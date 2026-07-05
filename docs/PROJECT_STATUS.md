# RL Optimal Liquidation — Project Status

**Single source of truth** for the project's status, results, and the reasoning behind them.
The README summarizes; this file is the complete account.

## Thesis

Optimal trade execution: liquidate `Q` shares over a horizon, minimizing market impact plus
inventory risk. The classical answer is **Almgren–Chriss (AC)**, which is *provably optimal* in
the linear-impact / quadratic-cost regime. This project validates reinforcement learning against
AC where AC is optimal, then asks **whether, where, and why** RL adds value beyond the *best
classical method a desk would actually deploy*, never a strawman. The deliverable is an honest
*structural* boundary rather than a horse-race win. "RL does not beat the best classical method
here" counts as a result when it comes with a precise reason.

## The arc — what RL can recover

Two phases:

1. **Recover.** Where AC is provably optimal, RL recovers it, which validates the pipeline as a
   sound measurement instrument.
2. **Degrade one input.** Change a single input, time-varying volatility, so that AC's *fixed*
   schedule goes stale. The right classical fix is **certainty-equivalence AC** (re-estimate, then
   act), and the problem stays linear-quadratic, so CE-AC is the ceiling. RL learns genuine
   closed-loop conditioning: it beats every *static* classical schedule and captures ~73% of the
   CE-AC edge without reaching it.

The boundary is where the arc stops, and that stopping point is itself the finding (see
[Why it stops here](#why-it-stops-here)).

### Phase 1 — recover
RL **recovers** the AC optimum in the linear-impact regime where AC is provably optimal. The
best-validated checkpoint lands **~2.84% mean** from AC (5 seeds, range 1.26–4.37%), and the
recovery holds where it discriminates: at κT=3 (λ=1e-4), AC undercuts naive TWAP by ~32% (versus
only 0.017% in the original κT=0.3 regime, where any schedule is nearly as good as AC), and
retrained agents still land **+0.26–0.36% from AC** (3 seeds, `configs/phase1_kt3.yaml`).

Two structural caveats keep the numbers honest. Final-of-training models do **not** reliably
converge, a **terminal-penalty cliff** reconfirmed on the 2026-07 retrains (κT=3 finals +0.6%,
+1.4%, **+47%** against bests of +0.26–0.36%), so every recovery figure is a save-best *deployment*
number rather than a convergence number. And the independent 50-parameter direct optimizer
reproduces AC to **−0.094%**, a residual that is structural rather than optimizer noise: the *soft*
terminal penalty (M=1e-3) makes leaving ~4.8% of the final step's inventory optimal, worth exactly
−0.094% against the hard-liquidation closed form (the exact solver in
`scripts/eval_phase2_baselines.py` reproduces it to four digits). The environment math is confirmed.

### Phase 2 — degrade one input (the certainty-equivalence ceiling)
RL beats every static classical schedule and captures ~73% of the certainty-equivalence ceiling,
which it never reaches. The perturbation is a single input, **time-varying volatility σ(t)** with
per-episode scale noise, which leaves the fixed AC schedule (computed once, at the base σ) stale. A
full classical ladder prices each rung *exactly* on matched scenarios
(`scripts/eval_phase2_baselines.py`: exact tridiagonal schedules, paired 95% CIs, 400 episodes):

| Arm | Gap vs naive AC |
|---|---|
| **Smart-static** — best *single* schedule knowing the σ-profile + noise distribution, zero conditioning | **−1.33%** [−1.62, −1.05] |
| **RL** (PPO, Gaussian policy, mean of 5 seeds; range −3.03…−3.70) | **−3.41%** [−3.93, −2.89] |
| **CE-AC (= per-episode oracle)** — re-plan at the observed σ̂; σ̂₀ reveals the episode scale exactly, so this is the ceiling for *any* causal policy | **−4.66%** [−5.17, −4.14] |

> **CE-AC  >  RL  >  smart-static  >  naive AC.** The agent beats the best static classical
> schedule by **2.1pp** [1.7, 2.5] on *every* seed and captures **~73%** of the CE-AC edge. Of
> CE-AC's 4.66%, 1.33pp is pure model knowledge (capturable with zero conditioning) and 3.32pp is
> genuine per-episode conditioning; the agent's edge over smart-static shows it captures most of
> the conditioning share. CE-AC itself remains unbeaten, as LQ theory requires.

The conditioning is verified *causally, independently of cost*: at fixed (k, q) the action is
monotone increasing in σ̂ on 26–30 of 30 probe-grid cells on every seed, and rolled out on σ
profiles never seen in training (flat, inverted-U) the agent tracks the new σ̂ pattern rather than
a memorized clock.

**Policy-family finding** (from the 2026-07 retrain batch, 16 runs + 2 extras): the featured
Gaussian policy succeeds **5/5 seeds**; the custom Beta policy is bimodal (2/5 discover
conditioning), and a CE control-variate reward deepens capture when discovery happens (best
−3.65%) but does not fix discovery — the bottleneck is exploration, not gradient variance. Likely
mechanism: SB3's Gaussian keeps one state-independent log-σ (exploration stays alive everywhere),
while Beta's state-dependent concentration can sharpen prematurely. The earlier "Beta > Gaussian"
result was specific to Phase 1's κT=0.3 boundary regime — a claim inherited across sessions until
re-measured.

**Why CE wins.** The cost is quadratic and the volatility uncertainty enters the inventory penalty
only through its mean, so "estimate the unknown, then act on the estimate" (certainty-equivalence)
is already near-optimal. The problem never leaves the linear-quadratic family, which leaves little
for a cleverer policy to exploit. Naive AC underperforms only because it never re-estimates, while
CE-AC does so cheaply. Because the cost surface is smooth and convex, conditioning on the realized
path buys a small, bounded edge and never a structural win. That is why the result is a **ceiling
rather than a vol-specific quirk**: it is a property of certainty-equivalence in a quadratic
objective, not of volatility.

### Why it stops here
The project stops at two phases by design, and the reason isn't that RL "fails" outside the
linear-quadratic family; it's that the same break which weakens the classical benchmark also
weakens RL's ability to close the gap, so a wider fight doesn't hand RL a cleaner win.

Two things are already visible in the Phase 2 numbers. First, RL's own imprecision is real: it
captures ~73% of the CE-AC edge, not all of it, so roughly a quarter of a 4.66% opportunity
(about 1.25pp) is left on the table by the optimizer, not by the problem. That gap is PPO's noise
floor, and nothing about leaving the LQ family shrinks it; nonlinear cost surfaces make value
estimation noisier and credit assignment harder, so the floor likely grows. Second, leaving the
family removes the exact benchmark (naive AC, smart-static, CE-AC) that makes every number above
verifiable rather than asserted. Break AC enough to make RL *necessary*, and both the referee and
the agent's precision degrade together: the extra opportunity a harder problem opens up has to
clear a wider noise floor, on a court with no lines.

That is the reasoned stopping point, not an unexplored one. (Earlier exploratory chapters,
including spreads, regimes, cross-impact, arrivals, and CVaR, live under `archive/`, each with its
own README.)

## Why this is a conclusion, not abandonment

The contribution is the structural boundary **plus** the measurement methodology the RL-execution
literature largely lacks: a **value-of-conditioning oracle** (matched-pair Monte Carlo that
separates "no value to capture" from "agent failed to capture it"), a **classical baseline ladder**
(naive / smart-static / CE-AC as exact schedules on matched scenarios — so "beats a baseline" can
never mean "beats a strawman"), **state-space sensitivity-probe grids**, and a **5-seed
best-AND-final protocol** that diagnoses convergence honestly rather than reporting a lucky
checkpoint. The discipline of the boundary is what would make any future GO
credible.

## Headline rule

No "RL beats AC" unqualified, anywhere. Frame as **"measuring when adaptive RL does and doesn't
add value over AC, and why."** The Phase-2 line is **CE-AC > RL > smart-static > naive AC**: the
agent beats every static classical schedule and captures ~73% of the certainty-equivalence
ceiling, it does not beat CE-AC, and in this family nothing can. The project stops at two phases
because leaving the linear-quadratic family means leaving the regime where an exact benchmark
exists. State the boundary as the finding, not as "future work."

## Out of scope (this release)

WRDS / real limit-order-book data (synthetic only) · off-policy or recurrent architectures (the
terminal-penalty cliff and the limits of per-step on-policy conditioning are *characterized* here,
not re-engineered).

## Methodology stack (the transferable contribution)

one execution-cost formula, defined in `baselines/exact_lq.py` and mirrored exactly by the env ·
analytic AC + exact tridiagonal schedule solver + 50-parameter direct-opt (env validation to
−0.094%, a residual the soft terminal penalty fully explains) · classical baseline ladder — naive /
smart-static / CE-AC on matched scenarios with paired CIs (`scripts/eval_phase2_baselines.py`;
the CE-AC rung doubles as the per-realization value-of-conditioning oracle) · state-space
sensitivity-probe grid (`scripts/probe_conditioning_grid.py`) · 5-seed protocol diagnosing
**best AND final** · regression suite (15 tests) pinning the AC formulas, the exact solver,
env-vs-analytic cost equivalence, and the κT-regime discrimination gap.

## Reproduce

```bash
make test                                # regression suite
python scripts/direct_schedule_opt.py    # env / AC validation (-0.094%)
make phase1                              # Phase 1 — recover AC                (~5 min)
make phase2-vol                          # Phase 2 — vol-conditioning           (~5 min)
make ladder                              # classical ladder: naive AC / smart-static / CE-AC vs RL
```
