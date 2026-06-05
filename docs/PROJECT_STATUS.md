# RL Optimal Liquidation — Project Status

**Single source of truth.** The README points here; `docs/phase1_journal.md` is the detail
underneath.

## Thesis

Optimal trade execution: liquidate `Q` shares over a horizon, minimizing market impact plus
inventory risk. The classical answer is **Almgren–Chriss (AC)**, which is *provably optimal* in
the linear-impact / quadratic-cost regime. This project validates reinforcement learning against
AC where AC is optimal, then asks **whether, where, and why** RL adds value beyond the *best
classical method a desk would actually deploy* — never a strawman. The deliverable is an honest
*structural* boundary, not a horse-race win. "RL does not beat the best classical method here" is
a result — stated with a precise reason.

## The arc — what RL can recover

Three phases:

1. **Recover.** Where AC is provably optimal, RL recovers it — validating the pipeline as a sound
   measurement instrument.
2. **Degrade one input.** Change a single input — time-varying volatility — so AC's *fixed*
   schedule begins to degrade. The right classical fix is **certainty-equivalence AC** (re-estimate,
   then act). RL **keeps up** with it but does not beat it: the problem stays linear-quadratic, so
   CE-AC is the ceiling, and RL only **ties** it — beating only the *naive* fixed schedule, by the
   small value of conditioning.
3. **Open.** Where, *if anywhere*, does changing the problem make RL genuinely *necessary* over the
   best classical method? Deliberate blank slate — not yet started.

### Phase 1 — recover
RL **recovers** the AC optimum in the linear-impact regime where AC is provably optimal. The
best-validated checkpoint lands **~2.84% mean** from AC (5 seeds, range 1.26–4.37%); final-of-
training models do **not** converge (gaps 8.5–559%) — a structural **terminal-penalty cliff**,
documented openly, so 2.84% is a save-best *deployment* number, not a convergence number. An
independent 50-parameter direct optimizer reproduces AC to **−0.094%**, confirming the environment
math and AC's optimality. *The pipeline is a validated measurement instrument.*

### Phase 2 — degrade one input (the certainty-equivalence ceiling)
Change a single input: **time-varying volatility σ(t)** with per-episode scale noise, so a *fixed*
AC schedule (computed once, at the mean σ) leaves value on the table. The right classical response
is **certainty-equivalence AC** — re-estimate σ̂, then act on the estimate. The finding:

> **RL ≈ CE-AC (ties)  >  naive fixed AC (beaten only by the value of conditioning).**

Trained PPO (custom Beta policy, 5 seeds) lands **−1.22% mean vs naive AC** at the best-validated
checkpoint, with genuine closed-loop control verified *independently of cost*: the action is
monotone and direction-correct in σ̂ across a **30×30 (k, q) state grid on all 5 seeds** — it sells
faster when volatility is high. A per-realization matched-pair oracle puts the full value of
*perfect* σ-conditioning at **3.79%** over naive AC; the agent captures **~32%** of it. The agent
**matches** CE-AC and beats only the naive baseline.

**Why CE wins, in one breath.** Because the cost is quadratic and the volatility uncertainty enters
the inventory penalty only through its mean, "estimate the unknown, then act on the estimate"
(certainty-equivalence) is already near-optimal — the problem never leaves the linear-quadratic
family, so there is little left for a cleverer policy to exploit. Naive AC underperforms only
because it never re-estimates; CE-AC does, cheaply. And because the cost surface is smooth and
convex, conditioning on the realized path buys a small, *bounded* edge — never a structural win.
This is the structural reason the result is a **ceiling, not a vol-specific quirk**: it is a
property of certainty-equivalence in a quadratic objective, not of volatility.

### Phase 3 — open
The first two phases establish a boundary: where the problem stays in the linear-quadratic family,
perturbing an input never makes RL *necessary* — it only ties the best classical method. The honest
next question — *where, if anywhere, does RL become necessary over the best classical method a desk
would deploy?* — is genuinely open. **Phase 3 is a blank slate: no committed objective, not yet
started.**

## Why this is a conclusion, not abandonment

The contribution is the structural boundary **plus** the measurement methodology the RL-execution
literature largely lacks: a **value-of-conditioning oracle** (matched-pair Monte Carlo that
separates "no value to capture" from "agent failed to capture it"), **state-space sensitivity-probe
grids**, and a **5-seed best-AND-final protocol** that diagnoses convergence honestly rather than
reporting a lucky checkpoint. The discipline of the boundary is what would make any future GO
credible.

## Headline rule

No "RL beats AC" anywhere. Frame as **"measuring when adaptive RL does and doesn't add value over
AC, and why."** The Phase-2 line is: **RL ties CE-AC (the smart adaptive classical method); it
beats only *naive* AC, by the value of conditioning.** Phase 3 is **open** — not yet started. Any
claim that RL beats the *optimal* policy is wrong — it ties it.

## Out of scope (this release)

WRDS / real limit-order-book data (synthetic only) · off-policy or recurrent architectures (the
terminal-penalty cliff and the limits of per-step on-policy conditioning are *characterized* here,
not re-engineered).

## Methodology stack (the transferable contribution)

`cost.py` single-source execution cost · analytic AC + 50-parameter direct-opt (env validation to
−0.094%) · per-realization matched-pair MC oracle (the value of conditioning) · state-space
sensitivity-probe grid (`scripts/probe_conditioning_grid.py`) · 5-seed protocol diagnosing **best
AND final** · regression suite with degenerate-reproduces-base guards.

## Reading order

README → **this file** → `docs/phase1_journal.md` (the Phase 1 debugging chronology and the
Phase 2 vol result).

## Reproduce

```bash
make test                                # regression suite
python scripts/direct_schedule_opt.py    # env / AC validation (-0.094%)
make phase1                              # Phase 1 — recover AC                (~5 min)
make phase2-vol                          # Phase 2 — vol-conditioning           (~5 min)
make oracle                              # value-of-conditioning oracle
```
