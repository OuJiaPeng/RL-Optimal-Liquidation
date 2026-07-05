# RL Optimal Liquidation — Project Status

**Single source of truth** for the project's status, results, and the reasoning behind them.
The README summarizes; this file is the complete account.

## Thesis

Optimal trade execution: liquidate `Q` shares over a horizon, minimizing market impact plus
inventory risk. The classical answer is **Almgren–Chriss (AC)** — a deterministic selling
schedule, solved in closed form before trading starts, that trades impact cost against inventory
risk, and is *provably optimal* under linear impact and quadratic costs. RL is worth trying
because that optimality is conditional on the model's inputs: when one drifts, the precomputed
schedule no longer fits, and a learned policy can in principle adapt closed-loop without a new
closed form. This project validates RL against AC where AC is optimal, then asks **whether, where, and
why** RL adds value beyond the *best classical method a desk would actually deploy*, never a
strawman. The deliverable is a structural boundary rather than a benchmark win: "RL does not beat
the best classical method here" counts as a result when it comes with a precise reason.

(The MDP — state, action, reward, horizon — is stated in the README's Formulation table and in
the env docstring, `src/rl_optimal_liquidation/envs/liquidation_env.py`.)

## The arc — two phases

1. **Recover.** Where AC is provably optimal, RL recovers it, which validates the pipeline as a
   sound measurement instrument.
2. **Degrade one input.** Change a single input, time-varying volatility, so that AC's *fixed*
   schedule goes stale. The right classical fix is **certainty-equivalence AC** (re-estimate, then
   act), and because the problem stays linear-quadratic, CE-AC is the ceiling. RL learns real
   closed-loop conditioning and lands between the static schedules and that ceiling.

The arc stops there, and the stopping point is itself the finding (see
[Why it stops here](#why-it-stops-here)).

### Phase 1 — recover

RL **recovers** the AC optimum in the linear-impact regime where AC is provably optimal. The
best-validated checkpoint lands **~2.84% mean** from AC (5 seeds, range 1.26–4.37%), and the
recovery holds where it discriminates: at κT=3 (AC's urgency × horizon; λ=1e-4), AC undercuts naive TWAP by ~32% (versus
only 0.017% in the original κT=0.3 regime, where any schedule is nearly as good as AC), and
retrained agents land **+0.26–0.36% from AC** (3 seeds, `configs/phase1_kt3.yaml`, `make phase1-kt3`).

Two structural caveats qualify these numbers. Final-of-training models do **not** reliably
converge — a **terminal-penalty cliff** reconfirmed when the agents were retrained for this
release (κT=3 finals ranged from +0.6% to +47% against bests of +0.26–0.36%). The mechanism: the
optimum sits against the f→1 boundary at the last step, where the quadratic penalty M·q_N² turns
small late-training policy drift into large cost swings. Every recovery figure is therefore a
save-best *deployment* number rather than a convergence number. Second, the independent
50-parameter direct optimizer reproduces AC to **−0.094%**, a residual with a structural cause:
the *soft* terminal penalty (M=1e-3) makes leaving ~4.8% of the final step's inventory optimal,
worth exactly −0.094% against the hard-liquidation closed form. The exact solver in
`src/rl_optimal_liquidation/baselines/exact_lq.py` reproduces it to four digits — the environment
math is confirmed.

### Phase 2 — degrade one input (the certainty-equivalence ceiling)

The perturbation is a single input, **time-varying volatility σ(t)** with per-episode scale noise,
which leaves the fixed AC schedule (computed once, at the base σ) stale. A full classical ladder
prices each rung *exactly* on matched scenarios (`scripts/eval_phase2_baselines.py`: exact
tridiagonal schedules, paired 95% CIs, 400 episodes):

| Arm | Gap vs naive AC |
|---|---|
| **Smart-static** — best *single* schedule knowing the σ profile + noise distribution, zero conditioning | **−1.33%** [−1.62, −1.05] |
| **RL** (PPO, Gaussian policy, mean of 5 seeds; seed range −3.70 to −3.03) | **−3.41%** [−3.93, −2.89] |
| **CE-AC (= per-episode oracle)** — re-plan at the observed σ̂; σ̂₀ reveals the episode scale exactly, so this is the ceiling for *any* causal policy | **−4.66%** [−5.17, −4.14] |

> **CE-AC > RL > smart-static > naive AC.** The agent beats the best static classical schedule on
> *every* seed — by **2.1 percentage points (pp)** on average [1.7, 2.5] — and captures **~73%**
> of the CE-AC edge. Of that 4.66pp edge, 1.33pp is pure model knowledge (capturable with zero
> conditioning) and the remaining ~3.3pp is per-episode conditioning; the agent's margin over
> smart-static shows it captures most of the conditioning share. CE-AC itself remains unbeaten,
> as LQ theory requires.

The conditioning is verified *causally, independently of cost*: at fixed (k, q) the action is
monotone increasing in σ̂ on 26–30 of the 30 probe-grid cells on every seed, and rolled out on σ
profiles never seen in training (flat, inverted-U) the agent tracks the new σ̂ pattern rather
than a memorized clock.

**Policy-family finding.** The featured Gaussian policy succeeds **5/5 seeds**; the Beta policy is
bimodal (2/5 discover conditioning). Likely mechanism: SB3's Gaussian keeps one state-independent
log-σ, so exploration stays alive everywhere, while Beta's state-dependent concentration can
sharpen prematurely — the bottleneck is exploration, not gradient variance. An earlier
"Beta > Gaussian" result was specific to Phase 1's κT=0.3 boundary regime and did not survive
re-measurement.

**Why CE wins.** The cost is quadratic and the volatility uncertainty enters the inventory penalty
only through its mean, so "estimate the unknown, then act on the estimate" (certainty-equivalence)
is optimal here — σ̂₀ reveals the episode scale exactly, and the problem never leaves the
linear-quadratic family, which leaves nothing for a cleverer policy to exploit. Naive AC
underperforms only because it never re-estimates; CE-AC does so cheaply. Because the cost surface
is smooth and convex, conditioning on the realized path buys a small, bounded edge and never a
structural win. That is why the result is a **ceiling**: it is a property of certainty-equivalence
in a quadratic objective, not of volatility.

### Why it stops here

The project stops at two phases by design. The reason isn't that RL "fails" outside the
linear-quadratic family; it's that the same break which weakens the classical benchmark also
weakens RL's ability to close the gap, so a harder problem doesn't produce a cleaner win.

Two things are already visible in the Phase 2 numbers. First, RL's imprecision is real: it leaves
roughly a quarter of the CE-AC edge (about 1.25pp) on the table, and that shortfall belongs to the
optimizer, not the problem. It is PPO's noise floor, and nothing about leaving the LQ family
shrinks it — nonlinear cost surfaces make value estimation noisier and credit assignment harder,
so the floor likely grows. Second, leaving the family removes the exact benchmark (naive AC,
smart-static, CE-AC) that makes every number above verifiable. Break AC enough to make RL
*necessary*, and the referee and the agent's precision degrade together: the extra opportunity a
harder problem opens up has to clear a wider noise floor, with no exact baseline left to certify
the result.

The stopping point is reasoned, not arbitrary. Earlier exploratory chapters (spreads, hidden
regimes, cross-impact, stochastic arrivals, CVaR objectives) were explored and then archived out
of the shipped repo; this release keeps only the two-phase spine.

## What the methodology contributes

The contribution is the structural boundary **plus** the measurement machinery the RL-execution
literature largely lacks: a **value-of-conditioning oracle** (matched-pair Monte Carlo that
separates "no value to capture" from "agent failed to capture it"), a **classical baseline
ladder** (naive / smart-static / CE-AC as exact schedules on matched scenarios), state-space
sensitivity-probe grids, and a 5-seed best-AND-final protocol that diagnoses convergence instead
of reporting a lucky checkpoint. The discipline of the boundary is what would make any future
extension credible.

## Out of scope (this release)

WRDS / real limit-order-book data — synthetic by design, not by omission: the exact classical
ladder that makes every result verifiable only exists when the environment matches the model
class, and real data has no ground-truth optimum to measure against. Also out: off-policy and
recurrent architectures — the terminal-penalty cliff and the limits of per-step on-policy
conditioning are *characterized* here; engineering around them is a different project.

## Methodology stack (index)

- One execution-cost formula — `src/rl_optimal_liquidation/baselines/exact_lq.py`, mirrored exactly by the env
- Analytic AC + exact tridiagonal schedule solver — `src/rl_optimal_liquidation/baselines/`
- Independent 50-parameter direct-opt env validation — `scripts/direct_schedule_opt.py` (details in Phase 1)
- Classical baseline ladder with paired CIs — `scripts/eval_phase2_baselines.py` (the CE-AC rung doubles as the value-of-conditioning oracle)
- State-space sensitivity-probe grid — `scripts/probe_conditioning_grid.py`
- 5-seed protocol diagnosing best AND final — `scripts/train_ppo.py` + `src/rl_optimal_liquidation/callbacks.py`
- Regression suite (15 tests) pinning the AC formulas, the exact solver, env-vs-analytic cost equivalence, and the κT-regime discrimination gap — `tests/`

## Reproduce

```bash
make test                                # regression suite
python scripts/direct_schedule_opt.py    # env / AC validation (expect -0.094%)
make phase1                              # Phase 1 — recover AC              (~5 min)
make phase1-kt3                          # Phase 1 — the discriminating κT=3 regime
make phase2-vol                          # Phase 2 — vol-conditioning        (~5 min)
make ladder                              # naive AC / smart-static / CE-AC vs trained RL
```
