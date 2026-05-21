"""Diagnostic plots for comparing a learned policy against the AC baseline.

Each function takes raw rollout arrays and returns a `matplotlib.figure.Figure`.
Plots include a residual subplot where a deviation panel makes small differences
between RL and AC visible without manual zooming.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _two_panel(figsize=(8, 6)):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    return fig, ax_top, ax_bot


def plot_inventory_trajectory(
    rl_inv: np.ndarray, ac_inv: np.ndarray, dt: float = 1.0
) -> Figure:
    """Top: RL band vs AC line. Bottom: RL − AC residual.
    rl_inv, ac_inv shape: (n_episodes, N+1).
    """
    t = np.arange(rl_inv.shape[1]) * dt
    rl_mean, rl_std = rl_inv.mean(axis=0), rl_inv.std(axis=0)
    ac_mean = ac_inv.mean(axis=0)

    fig, ax_top, ax_bot = _two_panel()
    ax_top.plot(t, rl_mean, color="C0", label="RL (mean)")
    ax_top.fill_between(t, rl_mean - rl_std, rl_mean + rl_std,
                        alpha=0.2, color="C0", label="RL ±1σ")
    ax_top.plot(t, ac_mean, "--", color="C1", label="AC")
    ax_top.set_ylabel("inventory $q_t$")
    ax_top.set_title("Inventory trajectory: RL vs AC")
    ax_top.legend()
    ax_top.grid(alpha=0.3)

    residual = rl_mean - ac_mean
    ax_bot.plot(t, residual, color="C2")
    ax_bot.fill_between(t, residual - rl_std, residual + rl_std,
                        alpha=0.2, color="C2")
    ax_bot.axhline(0.0, color="black", lw=0.5)
    ax_bot.set_xlabel("time")
    ax_bot.set_ylabel("RL − AC")
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_action_fraction(
    rl_inv: np.ndarray, rl_act: np.ndarray,
    ac_inv: np.ndarray, ac_act: np.ndarray,
) -> Figure:
    """Per-step fraction-of-inventory sold f_k = a_k / q_k, RL band vs AC line.
    Residual panel below. Bounded y-axis [0, 1] makes the comparison readable.

    rl_inv, ac_inv shape (n_episodes, N+1); rl_act, ac_act shape (n_episodes, N).
    """
    q_rl = np.maximum(rl_inv[:, :-1], 1e-9)
    q_ac = np.maximum(ac_inv[:, :-1], 1e-9)
    rl_frac = rl_act / q_rl
    ac_frac = ac_act / q_ac

    steps = np.arange(rl_act.shape[1])
    rl_mean, rl_std = rl_frac.mean(axis=0), rl_frac.std(axis=0)
    ac_mean = ac_frac.mean(axis=0)

    fig, ax_top, ax_bot = _two_panel()
    ax_top.plot(steps, rl_mean, "o-", color="C0", markersize=3, label="RL (mean)")
    ax_top.fill_between(steps, rl_mean - rl_std, rl_mean + rl_std,
                        alpha=0.2, color="C0", label="RL ±1σ")
    ax_top.plot(steps, ac_mean, "s--", color="C1", markersize=3, label="AC")
    ax_top.set_ylabel(r"fraction sold $f_k = a_k / q_k$")
    ax_top.set_title("Per-step trading fraction: RL vs AC")
    ax_top.legend()
    ax_top.grid(alpha=0.3)

    ax_bot.plot(steps, rl_mean - ac_mean, color="C2")
    ax_bot.axhline(0.0, color="black", lw=0.5)
    ax_bot.set_xlabel("step $k$")
    ax_bot.set_ylabel("RL − AC")
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_cost_distribution(
    rl_costs: np.ndarray, ac_costs: np.ndarray,
    ac_analytical: float | None = None,
    bins: int = 30,
) -> Figure:
    """Histogram of episode cost with mean reference lines and gap in the title."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(rl_costs, bins=bins, alpha=0.5, color="C0",
            label=f"RL  μ={rl_costs.mean():.3e}  σ={rl_costs.std():.3e}")
    ax.hist(ac_costs, bins=bins, alpha=0.5, color="C1",
            label=f"AC  μ={ac_costs.mean():.3e}  σ={ac_costs.std():.3e}")
    ax.axvline(rl_costs.mean(), color="C0", lw=1, ls="--")
    ax.axvline(ac_costs.mean(), color="C1", lw=1, ls="--")
    if ac_analytical is not None:
        ax.axvline(ac_analytical, color="black", lw=1, ls=":",
                   label=f"AC analytic  {ac_analytical:.3e}")

    gap = (rl_costs.mean() - ac_costs.mean()) / ac_costs.mean() * 100.0
    ax.set_xlabel("episode cost")
    ax.set_ylabel("count")
    ax.set_title(f"Episode cost distribution  (gap RL vs AC = {gap:+.2f}%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
