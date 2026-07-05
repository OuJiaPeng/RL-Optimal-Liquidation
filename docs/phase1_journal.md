# Phase 1 Journal: PPO learning Almgren–Chriss

A retrospective on the iterations between "PPO hoards inventory and pays a 10⁹ terminal
penalty" and "PPO matches AC to within ~7% across seeds, deterministically and reproducibly."

The goal of Phase 1 was never to *beat* AC — AC's solution is provably optimal in the
linear-impact regime. The goal was to validate that our PPO + Gymnasium + diagnostics
pipeline can recover that known optimum, so we have ground to stand on when we add
frictions in Phase 2 where AC no longer applies.

## Starting point

Discrete-time liquidation env per PDF §4:

- 50 steps, `Q=10⁶` shares, `T=1`, `σ=0.3`, `η=10⁻⁶`, `γ_perm=10⁻⁷`, `λ=10⁻⁶`
- State `(t/N, q/Q, S/S₀)`, action `f∈[0,1]` (fraction of inventory to sell)
- Reward = negative per-step cost (impact + inventory penalty), terminal penalty `−M·q_N²`
  for unliquidated inventory
- `MlpPolicy` PPO with SB3 defaults; 8 parallel envs; 500k timesteps

The AC analytical baseline for this config: `cost ≈ 1.03 × 10⁶`. That's the number to beat
(or match, more accurately).

## Run 1 — Reward-scale catastrophe (gap: 96,739%)

The first training run was an unmitigated failure:

| Metric | Value | Diagnosis |
|---|---|---|
| `std` | 1.05 (≈ init) | Policy never moved |
| `entropy_loss` | −1.47 (≈ init) | Same |
| `clip_fraction` | 0 | PPO's clip never activated — updates too small to need it |
| `approx_kl` | 1e-7 | Updates were essentially no-ops |
| `explained_variance` | 0 | Value function totally useless |
| `value_loss` | 7.5e13 | V-head chasing returns of order 10⁹, can't catch up |
| `rl_cost_mean` | 1.0 × 10⁹ | ≈ `M · Q² = 10⁻³ · (10⁶)²` — agent hoarded everything |

**Root cause: reward magnitudes were O(10⁶–10⁹).** A freshly-initialized value network has
weights near zero and predicts ~0, while the truth is ~10⁹. Squared-error value loss
exploded to O(10¹⁸) and dominated the gradient budget. The policy head got no useful
signal; PPO's per-batch advantage normalization was insufficient because every episode
ended the same way (terminal cliff) and so there was nothing to *differ* over.

**Fix**: `VecNormalize(norm_reward=True, clip_reward=10.0)`. Tracks a running std of
returns and rescales rewards to O(1) for the agent. The unwrapped env used by our callback
and `diagnose.py` keeps reporting cost in original units, so all AC comparisons stay
meaningful.

This is the canonical PPO-on-LQ-cost gotcha. Single biggest change of the project.

## Run 2 — Learning happens, but late drift (gap: best 5.7% at 400k, ended at 10.9%)

With VecNormalize:

| Step | RL cost | Gap | std |
|---|---|---|---|
| 80k | 1.0e9 | 96739% | — |
| 160k | 3.5e8 | 33395% | 0.49 |
| 240k | 1.15e6 | 11.3% | 0.12 |
| 320k | 1.11e6 | 7.7% | 0.06 |
| **400k** | 1.09e6 | **5.7%** | 0.02 |
| 480k | 1.14e6 | **10.9%** ← regressed | 0.015 |

The policy *did* find the right neighborhood at 400k, then random-walked back out. Why?

PPO's mid-training oscillation: even with our settings, `approx_kl` spiked to 5.38 on one
of the late iterations. The KL between two Gaussians with the same `σ` is
`(Δμ)²/(2σ²)` — so as `σ` collapses toward 0, even microscopic mean shifts produce huge
KL. The optimizer's fixed-size steps overshoot.

**Fix**: `target_kl: 0.03`. SB3 early-stops the inner PPO update loop when KL exceeds this
threshold, preventing the runaway. Worked: no more KL=5 spikes.

But target_kl alone doesn't *prevent drift* — it caps per-update magnitude, not
trajectory direction. The policy still wandered around the 5.7% basin, sometimes
better, sometimes worse.

## Run 3 — Save-best (gap: 5.71% on the captured checkpoint)

If the policy *visits* the right point but doesn't *stay* there, we can just record the
best checkpoint we ever saw:

- `ACBaselineCallback` now writes `best_model.zip` whenever the current
  policy beats the lowest cost it's seen on a matched eval set.
- `evaluate.py` and `diagnose.py` default to `best_model.zip`.

This made `diagnose.py` (200 episodes, deterministic eval) report a clean **5.71% gap**.

**Important generalization for Phase 2**: the save-best metric is *absolute RL cost on the
eval set*, not "gap vs AC." Phase 2 won't have AC, but it will have an eval rollout cost,
so the same callback machinery works.

At this point I argued Phase 1 was essentially validated. The user pushed back: 5–7% is
still "real money on the table in a regime where the answer is known exactly," and
save-best is a workaround, not stable convergence. Right call. We kept exploring.

## Misadventure — Linear LR decay made it worse (gap: best 8.92%)

Predicted fix: linearly decay LR from 3e-4 → 0. Textbook stochastic-approximation cure
for "fixed-size optimizer steps random-walking around the optimum."

Result: regressed. With LR decay, `std` stayed at 0.028 instead of collapsing to 0.015 —
the policy never committed as hard. Mid-training the gap was *worse* (50% gap at 240k vs
11% in the constant-LR run).

**Lesson learned**: PPO's noisy updates aren't pure waste — on this LQ problem they
provide *exploration*. Smoothing the trajectory with LR decay starves that exploration,
so the policy commits earlier to a slightly-worse local optimum.

Reverted. The empirical lesson outweighed the textbook prescription here.

## Run 4 — γ=1.0 (gap: best 8.4%)

User correctly pointed out the bigger issue: `γ=0.99` over 50 steps means the terminal
penalty is discounted by `0.99⁴⁹ ≈ 0.61` in the value target and `(γλ)⁴⁹ ≈ 0.05` in the
GAE-shaped policy gradient. For a *deterministically-terminating* episode this is just
wrong — AC's objective is undiscounted, so `γ=1.0` is the principled match.

I expected this to close the gap to 1–2%. It didn't:

| | γ=0.99 (Run 3) | γ=1.0 (this run) |
|---|---|---|
| Best gap | 5.7% | 8.4% |
| Final std | 0.015 | 0.008 |
| Final `value_loss` | 2e-5 | 2e-6 |
| Final `explained_variance` | 0.997 | 0.999 |

The internal metrics all *improved* — value function fits 10× tighter, std collapses
further, EV higher. But the policy landed in a slightly different local optimum with a
slightly worse gap. The discount-induced "undervaluing of terminal penalty" was real but
not the operative friction.

Kept γ=1.0 anyway — it's the right choice on principle, and within-seed variance
is large enough that the apparent regression isn't decisive.

## Run 5 — Drop the price observation (gap: best 8.0%)

AC's optimal policy is `f*(t, q)` — independent of `S`. Our state was `(t/N, q/Q, S/S₀)`,
forcing the MLP to *learn* to ignore S. Hypothesis: this is contributing to the residual
gap.

Added `include_price_obs: bool` flag on `LiquidationParams`. Set to `false` in the config.
State is now `(t/N, q/Q)`.

Result: gap moved 8.4% → 8.0%. Marginal. Hypothesis only weakly supported.

One notable side effect: `rl_cost_std` across 200 eval episodes is now **2e-10** (zero,
modulo float precision). With no price input, the deterministic policy on deterministic
state evolution gives byte-identical rollouts. The 0.04% noise band we saw before *was*
the policy reacting to price noise — just at a magnitude that didn't matter.

Kept `include_price_obs: false` for Phase 1; will flip back for Phase 2 if any friction
makes price genuinely informative (e.g. signed drift with concave impact).

## Multi-seed sweep — Characterizing variance

To know whether 5.7% vs 8.0% was a hyperparameter effect or just stochastic-optimization
luck, ran three seeds with the settled config (γ=1.0, 2D obs):

| Seed | Best gap | inv_l1_dev |
|---|---|---|
| 0 | 8.03% | 26.7k |
| 1 | 5.76% | 27.0k |
| 2 | 8.70% | 14.2k |

**Mean ≈ 7.5%, range [5.8%, 8.7%].** PPO stochasticity dominates over the hyperparameter
changes we tried within this regime.

That 5.7% from the original Run 3 (γ=0.99, 3D obs) wasn't evidence that config was
better — it was the lucky end of *this same distribution* on a different seed. None of
the changes we made escape the [5.8, 8.7] band.

## What is the 5–9% gap, actually?

This section was originally written as "MLP approximation error — model class's
expressive ceiling." That was wrong, and the diagnostic that settles it lives in
[`scripts/direct_schedule_opt.py`](../scripts/direct_schedule_opt.py): a 50-parameter
direct optimization of the trade schedule via Adam, run on the same env cost
formula.

The result:

```
Direct-opt cost                  : 1.031755e+06
AC analytic cost (continuous)    : 1.032728e+06
AC schedule costed through env   : 1.032728e+06

Gap direct vs AC analytic        : -0.0942%
Gap AC-via-env vs AC analytic    : +0.0000%
```

Three takeaways:

1. **The env/reward math is correct.** The AC discrete schedule costed through our
   per-step cost formula agrees with `ac_expected_cost(...)` to 0.0000%.
2. **A 50-parameter direct optimization beats AC by 0.094%.** Direct-opt finds the true
   discrete-time optimum; the gap is *negative* because AC's continuous-time formula is
   slightly suboptimal in our discrete-50-step setting. The one place the two schedules
   differ meaningfully: at `k=49`, direct-opt sells 95.2% of remaining inventory while AC
   says 100%. Direct-opt correctly notices `M=10⁻³` is finite, so the *exact* optimum
   leaves a sliver of inventory rather than incurring the marginal impact cost of
   liquidating it.
3. **The 5–9% PPO gap is not an expressivity ceiling.** A 4,600-parameter MLP can fit
   the 50-point AC schedule to arbitrary precision. Direct-opt proves the loss surface
   is benign. The gap is purely policy parameterization + PPO dynamics.

The most likely culprit, given the action space `Box(0, 1)`: SB3's PPO uses a Gaussian
distribution clipped to the action bounds. During rollout it samples from `N(μ, σ²)`,
clips the sample to `[0, 1]`, but computes `log_prob` on the *unclipped* sample. That
mismatch creates a systematic bias near the boundaries — and AC's optimum *has* boundary
behavior (`f_k → 1` at the terminal step). A Beta(α, β) distribution, which lives on
`[0, 1]` natively with no clipping, should sidestep this and is the obvious next
experiment.

**Implication for Phase 1**: the gap is reducible, not a floor. Whether to spend the
~100 lines of custom-policy code on a Beta distribution in Phase 1 vs accepting ~7% and
moving on is a judgment call — but the journal should not claim the residual is an
expressive ceiling, because it isn't.

## Run 6 — Beta policy at 500k timesteps (gap: 1.5%–15.5% across 3 seeds)

Implemented `BetaActorCriticPolicy` (~120 lines): `Beta(α, β)` with α, β = softplus + 1
for unimodality. Reparameterized samples (`rsample`). Override `_build` and
`_get_action_dist_from_latent` on SB3's `ActorCriticPolicy`.

First run across 3 seeds with the existing budget (500k timesteps):

| Seed | Gaussian best | Beta best |
|---|---|---|
| 0 | 8.03% | 5.12% |
| 1 | 5.76% | **1.49%** |
| 2 | 8.70% | 15.54% |

Seed 1 at 1.49% confirms the diagnosis: Beta *can* reach the direct-opt floor. But the
spread is huge (1.5%–15.5%) and `explained_variance` ended at 0.58 on seed 0 — value
function losing track of the policy. The seed-0 trajectory shows a 1622% spike at
step 320k followed by recovery to 5.12% by 480k. Policy was still actively improving
when training ran out.

## Misadventure (again) — Multi-variable tuning over-corrected

Tried to stabilize Beta with three simultaneous changes: `LR 3e-4 → 1e-4`,
`target_kl 0.03 → 0.01`, `n_steps 256 → 512`. Result: all three seeds at 30–55% gap.
Total update count had dropped from ~244 to ~122 and effective per-step gradient
budget by ~3×; the agent was massively undertrained.

**Lesson, restated**: same rule as multi-seed sanity checks — change one variable at a
time when debugging hyperparameters. With three simultaneous changes, a failed result
tells you nothing about which knob is responsible. This was the second time on this
project I bundled fixes; first time I called it "principled scoping," this time it was
clearer cope.

## Run 7 — Beta at 1M timesteps (gap: [1.3%, 4.4%] across 5 seeds — settled)

Reverted the bundled tuning. Single change: doubled `total_timesteps` from 500k to 1M,
keeping all Beta defaults (LR=3e-4, target_kl=0.03, n_steps=256). Justified by the
seed-0 evidence above: Beta's policy was still improving at 480k.

Final 5-seed result (the shipping numbers in `runs/phase1_final_s*`):

| Seed | Best gap | inv_l1_dev |
|---|---|---|
| 0 | 1.54% | 13.1k |
| 1 | 3.80% | 28.4k |
| 2 | 3.22% | 21.1k |
| 3 | 4.37% | 22.8k |
| 4 | 1.26% | 10.7k |

**Mean 2.84%, range [1.26%, 4.37%].** All 5 seeds land within ~5% of AC. Range
is wider than initial 3-seed estimate (3pp → 3.1pp) but in the same band.

The 500k Beta variance was undertraining, not instability. Beta's concentration
parameters take longer to settle than Gaussian's (μ, σ) because they affect both the
mode and the variance simultaneously through `softplus + 1` — but given the time,
they get there cleanly.

## Settled Phase 1 baseline

`policy=beta` + `total_timesteps=1M`, all other hyperparameters at SB3 defaults. Stable
across seeds in the [1.3%, 4.4%] band, mean 2.84%. Phase 2 inherits this config.

The journey from `gap=96,739%` (run 1) to `gap≈2.8% mean` (run 7) took roughly the order
listed in the lessons section below — reward scale dominated, then convergence
mechanisms (target_kl, save-best, γ=1.0), then policy class (Beta), then training
duration. Each ordering matters: bumping timesteps before fixing reward scale would
have done nothing.

### What the ~2.8% number actually represents

Not approximation error and not policy-class expressivity (direct-opt with 50
parameters reaches AC to −0.094%; the network is far more expressive than needed).
Eval rollouts *are* deterministic (`rl_cost_std ≈ 2 × 10⁻¹⁰` across 200 episodes),
so the number is exact — not eval noise.

But it is **not the gradient-variance "equilibrium ball" noise floor either**. An
earlier revision of this section made that claim; the trajectory data contradicts
it. The `ac_eval.csv` for `runs/phase1_final_s0` (constant LR, single seed):

| Step | Gap to AC |
|---|---|
| 80k  | 140.8% |
| 160k | 9.6%   |
| 240k | 339.1% |
| 320k | 963.7% |
| 400k | 465.9% |
| 480k | 38.5%  |
| 560k | 5.9%   |
| 640k | 190.3% |
| 720k | 355.8% |
| **800k** | **1.54%** ← save-best fires here |
| 880k | 123.9% |
| 960k | 509.3% |

The policy is not in a tight ball around AC. It cycles between brief near-AC transits
(1–10% gap) and large pathological excursions (100–1000%), all the way to the end of
training. **Save-best captures the lucky transits; everything else is discarded.**

Final-model gaps across all 5 shipped Phase 1 seeds (diagnosed on `model.zip`, not
`best_model.zip`):

| Seed | Best-model gap | Final-model gap |
|---|---|---|
| 0 | 1.54% | 57.1% |
| 1 | 3.80% | 272.0% |
| 2 | 3.22% | 559.5% |
| 3 | 4.37% | **8.5%** |
| 4 | 1.26% | 538.3% |

So the **2.84% mean is a *deployment* number**, not a convergence number. PPO + Beta
+ 64×64 MLP on this LQ env does not stably converge to AC's neighborhood. It passes
through it periodically; we snapshot it via save-best.

### The cliff-instability mechanism

Gradient-variance jitter wouldn't produce 1000% excursions; the mechanism is
the terminal-penalty cliff `M · q_N²`. If the policy under-sells by 5% of `Q`,
terminal penalty = `1e-3 · (5e4)² = 2.5 × 10⁶`, which exceeds the entire AC cost
(`1.03 × 10⁶`). Small changes to the last few actions flip whether the policy
fully liquidates or stops short, and PPO's stochastic updates regularly tip the
policy across that knife-edge in both directions.

### Two LR-anneal experiments, both confirming non-convergence

**Late anneal** (`configs/phase1_anneal.yaml`: constant LR for first 80%, anneal over
last 20%). Save-best identical to constant-LR run because the captured checkpoint is
from step 800k, before annealing engages (the two runs share an identical trajectory
through 800k under the same seed). Final-model gap: **100.5%** — the annealed tail
diverges from the constant-LR baseline (whose final is 57.1%); only the save-best
checkpoint is shared. Single seed. The point stands: annealing the last 20% does not
prevent the cliff-fall.

**Early anneal** (`configs/phase1_anneal_early.yaml`: constant for first 20%, anneal
linearly over remaining 80%). Save-best gap: 1.37% — basically the same lucky-transit
floor. **Final-model gap: 313.7%**, even with LR down to ~4.5 × 10⁻⁵ by step 880k.
Cliff-falls persist regardless of step size.

The cliff is a structural property of the cost function, not an LR artifact. Smaller
step sizes don't damp it within tested budgets.

### What this means for the Phase 1 result

PPO + Beta on this env produces a *deployable* policy (within ~3% of AC) via save-best
on a wandering trajectory — not a converged policy. Honest framing: "best validated
checkpoint within 1.3–4.4% of AC across 5 seeds; the policy itself does not stably
converge to that neighborhood under tested configurations." Closing the gap further
would require either changing the cost function (softer terminal handling), changing
the algorithm class (off-policy + replay buffer for smoother updates, or pretrain →
fine-tune), or constraining the policy parameterization (e.g., explicit terminal-
forced-liquidation, removing the cliff). None tested in this release.

## Settled Phase 1 config

```yaml
env:
  Q: 1.0e+6
  T: 1.0
  N: 50
  sigma: 0.3
  eta: 1.0e-6
  gamma: 1.0e-7        # permanent-impact coefficient
  lam: 1.0e-6
  terminal_penalty: 1.0e-3
  include_price_obs: false   # drop S/S₀ — irrelevant for LQ optimum

ppo:
  policy: beta               # Beta(α,β) on [0,1], no clipping; Gaussian was 5–9% floor
  device: cpu
  learning_rate: 3.0e-4
  lr_schedule: constant      # linear decay starved exploration; constant won
  gamma: 1.0                 # episodes terminate at N=50; undiscounted matches AC's objective
  gae_lambda: 0.95
  clip_range: 0.2
  target_kl: 0.03            # caps per-update KL once std collapses
```

Training duration: `--total-timesteps 1000000` (Beta needs ~2× the budget of Gaussian
to converge; 500k is undertrained for Beta, see Run 6).

## Lessons (in rough order of importance)

1. **Reward scale matters more than anything else.** `VecNormalize` was the difference
   between "policy never moves" and "policy learns the right thing." Until rewards
   sit at O(1), nothing else PPO has matters.
2. **Theoretically-correct fixes don't always move the empirical needle much.**
   `γ=1.0` and dropping the price obs were both principled, and both moved the
   gap by < 1 percentage point. Kept them anyway because the *reason* to make them
   doesn't depend on whether they help — they make the model match the problem.
3. **PPO's noisy updates do useful early exploration; global LR decay starves that.**
   Linearly decaying LR from step 0 made things worse because the policy hadn't yet
   reached the right basin when its step size was already cut. The Robbins–Monro
   argument for decaying LR still applies in principle — but the right schedule is
   *two-phase* (constant to find the basin, then anneal to close the noise-floor
   ball), not global linear decay. We did not test the two-phase variant; the global
   one was a misadventure, not a refutation of the underlying noise-floor theory.
4. **Single-run results are unreliable on stochastic-optimization problems.** Three seeds
   is the minimum to know whether a number is typical or lucky. The 5.7% that looked
   like a great result on seed 0 was just the bottom of a band that spans [5.8, 8.7]
   across seeds.
5. **Save-best generalizes to Phase 2 if you key it on cost, not on gap-vs-baseline.**
   Phase 2 won't have AC, but it always has rollout cost.
6. **Don't write off a residual gap as an "expressive ceiling" without a diagnostic
   that actually localizes the bottleneck.** I argued the 5–9% gap was MLP approximation
   error; the direct-schedule diagnostic proved the loss surface is benign and a small
   network can fit the schedule to arbitrary precision. The residual was policy
   parameterization (clipped Gaussian on a bounded box). Beta on [0, 1] + enough
   timesteps closed the gap to <4% on every seed. "We've explored the easy knobs and
   ran out of ideas" is not the same statement as "this is the model's ceiling."
7. **Change one variable at a time when debugging hyperparameters.** The Beta-tuning
   attempt that bundled three changes (LR, target_kl, n_steps) blew up and produced
   no actionable signal about which knob was responsible. The clean follow-up — keep
   all defaults, just double the timesteps — gave the decisive answer. Same lesson as
   multi-seed sanity checks, just one level up.

# Phase 2 — Where AC doesn't apply

> **Note.** This journal's `Phase 2.x` headings are *build order* — the sequence of things tried.
> The shipped Phase 2 result is the **vol-conditioning** experiment (2.2); 2.1 (the κT sweep) is
> the key insight that led to it. See
> [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the consolidated certainty-equivalence finding.

The framework is validated at Phase 1. Now we test whether the agent can learn
closed-loop control in regimes AC can't describe. The trajectory below is *not*
"add a friction, train, measure." It's a sequence of dead-ends, diagnostic
sweeps, and partial successes that progressively localized where on-policy RL
adds value over AC and where it doesn't.

## Phase 2.1 — The κT regime sweep (key insight)

Before training under time-varying σ, ran a diagnostic that we should have
done much earlier: for a sweep of λ values, optimize a single-parameter
(constant fraction) policy and compare to the 50-parameter optimal schedule:

| λ | κT | const_f | const_cost | full_cost | **gap %** |
|---|---|---|---|---|---|
| 1e-6 | 0.30 | 0.080 | 2.35e6 | 1.04e6 | **+125.4%** |
| 1e-5 | 0.95 | 0.080 | 2.54e6 | 1.39e6 | +82.5% |
| 5e-5 | 2.12 | 0.083 | 3.39e6 | 2.74e6 | +23.6% |
| **1e-4** | **3.00** | **0.088** | **4.42e6** | **4.11e6** | **+7.7%** |
| 5e-4 | 6.71 | 0.167 | 1.08e7 | 1.08e7 | +0.44% |
| 1e-3 | 9.49 | 0.236 | 1.62e7 | 1.62e7 | +0.12% |
| 2e-3 | 13.4 | 0.323 | 2.44e7 | 2.44e7 | +0.029% |
| 1e-2 | 30.0 | 0.579 | 6.96e7 | 6.96e7 | +0.001% |

**The result is decisive.** At κT > ~6, the cost surface is *flat in the
schedule-shape direction* — a constant policy at the right fraction matches
the optimal full schedule to within 0.5%. So PPO at high κT correctly finds
"best constant" rather than "best schedule shape." At κT ∈ [2, 4], schedule
structure matters meaningfully (7–24% gap), and that's the regime where
closed-loop conditioning can pay off.

This explains a sub-experiment we ran earlier at λ=2e-3 (κT=13.4) under
u-shaped σ: the agent converged to a degenerate constant policy outputting
f=0.324 at every (k, q, σ̂). Initially read as failure; the sweep showed it
was the *correct* answer — there's nothing to learn at that κT. Picked κT=3
(λ=1e-4) for all subsequent Phase 2 experiments.

## Phase 2.2 — Vol-conditioning (positive result)

Setup: time-varying σ(t) with u-shape (high at open/close, low midday) plus
per-episode multiplicative scale ~ N(1, σ_noise_std). The scale randomization
*decorrelates σ̂ from t* — t alone can't predict σ̂, the agent must read its
σ̂_k observation to plan the trajectory.

**First attempt (σ_noise=0.15)**: 5 seeds, all converged but each with
**zero σ̂ sensitivity** at the controlled-probe test. Sensitivity grid:
action was identical across σ̂ ∈ {0.5, 0.75, 1.0, 1.25, 1.5} at every
(k, q) test point. Cost matched AC (gap −0.03%) but for the wrong reason:
the agent learned the AC-shape schedule *without using σ̂*. The conditioning
that could have been profitable wasn't discovered.

**Bumped to σ_noise=0.3**: signal-to-noise ratio increases (σ_scale range
[0.4, 1.6] vs [0.55, 1.45]). 5 seeds, all converged, **all 5 beat *naive* AC**
(4 substantially, 1 barely; comparable to CE-strengthened AC, not a beat of the best
classical method — see docs/PROJECT_STATUS.md):

| Seed | RL cost | AC cost | Gap |
|---|---|---|---|
| 0 | 4.63e6 | 4.65e6 | −0.58% |
| 1 | 4.63e6 | 4.65e6 | −0.45% |
| 2 | 4.65e6 | 4.65e6 | −0.10% |
| 3 | 4.57e6 | 4.65e6 | **−1.66%** |
| 4 | 4.50e6 | 4.65e6 | **−3.32%** |
| **mean** | **4.60e6** | **4.65e6** | **−1.22%** |

Sensitivity grid now shows **monotonic, direction-correct σ̂-conditioning
across the entire (k, q) state space**: max sensitivity 0.060, with action
increasing in σ̂ at every test point (high vol → sell faster). The agent
genuinely learned closed-loop control.

Oracle diagnostic (per-realization direct-opt knowing σ_scale, matched-pair MC):

| σ_noise_std | AC cost | Oracle cost | **Value of conditioning** |
|---|---|---|---|
| 0.15 | 4.23e6 | 4.15e6 | **+1.98%** |
| 0.3 | 4.38e6 | 4.21e6 | **+3.79%** |

At σ_noise=0.15, the maximum possible value of perfect σ̂-conditioning is
only 1.98% over AC. The agent's failure to discover σ̂-conditioning at this
noise level is consistent with "the available signal is below the
gradient-noise floor of PPO": the conditioning gradient and the policy-
gradient-estimator noise are the same order of magnitude.

At σ_noise=0.3 the available value nearly doubles to 3.79%, and the agent
captures **1.22% mean across 5 seeds — about 32% of the oracle's value**.

**Decomposition (the rigorous way to read this result):** the agent's
distance-to-true-optimum is essentially constant across phases —
Phase 1 had a ~2.94% gap to direct-opt; Phase 2 has a 2.57% gap to oracle
(3.79% − 1.22%). PPO's best-checkpoint imprecision (Phase 1 section above) is
the *same* in both phases. What changes is the **location of the optimum**:
in Phase 1 it sits at AC, in Phase 2 it sits 3.79% below *naive* AC. The
"−1.22% vs naive AC" result is the derived consequence of "PPO finds a target
~2.5% short of optimum, and the optimum has shifted 3.79% away from naive AC"
(the agent only *matches* CE-strengthened AC; the win is over the naive
baseline). This explains *why* RL beats naive AC rather than just reporting
that it does, and predicts the margin over naive AC scales linearly with the
oracle's distance from AC
(modulo the constant ~3% PPO imprecision).

**Phase 2.2 conclusion:** closed-loop σ-conditioning is learnable by PPO
when the signal-to-noise ratio is large enough (0.15 → fails, 0.3 → works).

## Unifying interpretation

The controlling variable across the Phase 2 probes is the **value of conditioning
relative to policy variance**, not horizon length or PPO architecture:

| Experiment | Stochasticity | Available value | Captured? |
|---|---|---|---|
| Vol, σ_noise=0.15 | per-episode | **1.98%** | No — agent ignores σ̂ (signal below the noise floor) |
| **Vol, σ_noise=0.3** | **per-episode** | **3.79%** | **Yes — 1.22% mean (≈32% capture)** |

At σ_noise=0.15 the maximum value of perfect σ̂-conditioning (1.98%) sits at the same
order as PPO's policy-gradient noise, so the agent rationally ignores σ̂ and just
reproduces the AC-shape schedule. Doubling the available value to 3.79% (σ_noise=0.3)
lets the conditioning gradient clear the noise floor; the agent then captures ≈32% of
it — **matching certainty-equivalence AC, beating only *naive* fixed AC**. The lesson is
structural: what RL can capture is bounded by the value of conditioning the problem
actually contains, not by horizon or architecture.

## Methodology lessons from Phase 2 (additions to Phase 1's lessons)

1. **Run the value-of-conditioning oracle BEFORE concluding what a negative result
   means.** The vol@0.15 case looked like "the agent failed to learn"
   until the per-realization oracle showed the available value was below the
   gradient-noise floor. "No value to capture" is a fundamentally different result
   from "PPO failed to capture it" — and only the oracle distinguishes them.
2. **A 50-parameter direct-opt is the most useful single diagnostic in the project.**
   It validated the env in Phase 1 (−0.094% vs AC), characterized the κT regime in
   Phase 2.1, and provided the value-of-conditioning oracle in 2.2. Cheap (≈30 lines,
   runs in seconds) and decisive.

## Settled configs (final state)

The shipped Phase configs are `configs/phase1.yaml` and `configs/phase1_kt3.yaml` (Phase 1)
and `configs/phase2_vol_gaussian.yaml` (the featured Phase 2 run; `phase2_vol.yaml` and
`phase2_vol_cv.yaml` are the Beta and control-variate arms). Each is documented inline and
reproduces its result with a 5-seed `python scripts/train_ppo.py --config <file>` sweep.
The LR-anneal experiment configs now live in `archive/superseded/phase1_anneal*.yaml`.

# Addendum (2026-07-03) — audit corrections and the exact baseline ladder

A code audit found measurement bugs and a missing-baseline gap; all are fixed in this commit.

1. **`--eval-freq` was off by a factor of n_envs.** The callback gated on `n_calls`, which SB3
   increments once per *VecEnv* step while `num_timesteps` advances by n_envs. With the shipped
   `n_envs: 8`, every run evaluated and save-bested every **80k** timesteps, not the documented
   10k (the committed `ac_eval.csv` rows confirm it). Now gated on `num_timesteps`.
2. **Beta `mode()` NaN guard.** For raw params ≤ ~−17, float32 softplus underflows to α=β=1
   exactly and the mode is 0/0 = NaN, which `np.clip` passes into the env on every
   deterministic-eval path. `mode()` now falls back to the mean when the denominator vanishes.
3. **The −0.094% direct-opt residual is structural, not optimizer noise.** With the *soft*
   terminal penalty M=1e-3, the true optimum leaves q_N = q_{N−1}·c_a/(c_a+M) ≈ 4.8% of the last
   step's inventory. An exact tridiagonal solve of the quadratic objective reproduces −0.0943%
   vs the hard-liquidation AC closed form. Phase 1's "direct-opt reproduces AC" story is intact —
   the residual now has its reason.
4. **"RL ties CE-AC" was never measured — and is false.** `scripts/eval_phase2_baselines.py`
   prices the full classical ladder *exactly* on matched scenarios (paired 95% CIs, 400
   episodes): naive AC 0 · smart-static (profile + noise-distribution knowledge, zero
   conditioning) **−1.33%** · CE-AC **−4.66%** (σ̂₀ reveals the episode scale, so CE-AC = the
   per-episode oracle) · RL mean of 5 seeds **−0.74%** (−1.22% on the original eval seeds — mild
   save-best selection optimism). The mean agent captures ~16% of the CE-AC edge and is **0.60%
   [0.43, 0.78] worse than the best static schedule**; seeds 3–4 (−1.50%, −2.96%) do clear the
   static bar. Corrected Phase 2 line: **RL beats only naive AC; it does not tie CE-AC, and on
   average does not clear even the smart static baseline.** The earlier "value of conditioning =
   3.79%" also conflated profile knowledge (capturable with zero conditioning) with genuine
   per-episode conditioning; the ladder decomposes it (1.33pp + 3.32pp).
5. **Smaller corrections.** The conditioning probe grid is 30 cells (6×5), not "30×30"; the
   matched-pair oracle gap at noise=0.15 is 1.98% (a stale 5.10% figure sat in the probe
   docstring); `pyproject.toml` now declares tqdm/rich (`progress_bar=True` crashed fresh
   installs); the env-vs-analytic cost test now also runs at σ=0.3, pinning the risk term the
   σ=0-only version never touched; `evaluate.py`/`diagnose.py` report paired 95% CIs and flag
   the constant-σ analytic anchor as non-comparable under Phase 2 configs (it sits ~29% low
   there); bool params in `LiquidationParams` now parse YAML strings ("false" was truthy).

# Addendum (2026-07-04) — the retrain batch: Gaussian inverts Beta, RL clears the static bar

Retrained everything under the corrected protocol (18 runs: 5 Beta, 5 Beta+control-variate,
5 Gaussian, 3 Phase-1b), all scored on the exact ladder with matched scenarios:

1. **The featured result.** Gaussian policy, 5/5 seeds: **−3.41%** [−3.93, −2.89] vs naive AC
   (range −3.03…−3.70), beating smart-static (−1.33%) by 2.1pp on every seed and capturing ~73%
   of the CE-AC ceiling (−4.66%). Conditioning verified causally: action monotone in σ̂ on 26–30
   of 30 grid cells per seed; OOD profile rollouts track σ̂, not the clock.
2. **The Beta-vs-Gaussian inversion.** The journal's earlier "clipped Gaussian capped at 5–9%,
   Beta closed it" was a Phase 1 (κT=0.3) result that did not transfer: in the Phase 2 regime
   Beta is bimodal (2/5 seeds discover conditioning; retrain mean −0.83%), Gaussian is uniform
   (5/5). Hypothesis: SB3's Gaussian shares one state-independent log-σ so exploration survives
   everywhere; Beta's state-dependent concentration sharpens prematurely and can kill σ̂
   exploration before discovery. Lesson: distribution choice is regime-dependent — re-measure
   inherited claims.
3. **Control variate: hypothesis half-right.** Subtracting the per-scenario CE cost from the
   reward (returns ≈ regret) deepens capture when discovery happens (best −3.65%) but does not
   raise the discovery rate (2/5) — the binding constraint is exploration, not gradient variance.
4. **Phase 1b (κT=3, `configs/phase1_kt3.yaml`).** Beta recovers AC to +0.26–0.36% (3 seeds) in
   the regime where shape is worth 7.7% — the discriminating version of the Phase 1 claim.
5. **Phase 3 sqrt-impact direction measured dead.** Best-LQ-family vs true square-root-impact
   optimum: ≤0.85% across participation sizes 0.25×–64× (peak ~4×, converging again as risk
   dominance front-loads both families) and ~0.70% under stochastic liquidity correlated with
   vol. Below RL's own imprecision — no honest crossover available there. Superseded tools moved
   to `archive/superseded/`.
