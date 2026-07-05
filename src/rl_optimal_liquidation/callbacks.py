"""SB3 callback that benchmarks the learning policy against the AC baseline.

Maps directly to the PDF §6 diagnostics suite:
    Implementation shortfall vs AC   ->  ac/is_gap_pct
    Episode cost variance            ->  ac/rl_cost_std
    Inventory trajectory shape       ->  ac/inv_l1_dev   (full plot in scripts/diagnose.py)

SB3 already logs policy entropy, approx KL, and clip fraction under train/* — those
diagnostics from §6 are covered without extra code.

If `log_path` is provided, each evaluation also appends one row to a CSV — useful for
inspecting training history after the fact without scraping TB event files.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .diagnostics import ac_policy_fn, episode_cost, rollout
from .envs.liquidation_env import LiquidationEnv


class ACBaselineCallback(BaseCallback):
    """Periodically compare the current policy to the AC schedule on matched seeds."""

    CSV_FIELDS = [
        "timestep", "rl_cost_mean", "rl_cost_std",
        "ac_cost_mean", "ac_cost_std", "is_gap_pct", "inv_l1_dev",
    ]

    def __init__(
        self,
        eval_env_fn: Callable[[], LiquidationEnv],
        n_eval_episodes: int = 20,
        eval_freq: int = 10_000,
        seed_base: int = 1_000_000,
        log_path: Path | str | None = None,
        save_best_path: Path | str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.seed_base = seed_base
        self.log_path = Path(log_path) if log_path is not None else None
        self.save_best_path = Path(save_best_path) if save_best_path is not None else None
        # Save-best is triggered by lowest RL cost on the matched eval seeds.
        # Cost generalizes to Phase 2 (no AC needed); in Phase 1, lowest cost
        # is equivalent to lowest gap because AC cost is constant.
        self.best_cost = float("inf")
        self.best_gap_at_best = float("nan")
        self._next_eval = eval_freq

    def _init_callback(self) -> None:
        # Start each training run with a fresh CSV and reset best-tracking.
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writeheader()
        self.best_cost = float("inf")
        self.best_gap_at_best = float("nan")
        self._next_eval = self.eval_freq

    def _on_step(self) -> bool:
        # Gate on num_timesteps, not n_calls: SB3 increments n_calls once per
        # VecEnv step while num_timesteps advances by n_envs, so an n_calls
        # gate would evaluate every eval_freq * n_envs timesteps.
        if self.num_timesteps >= self._next_eval:
            while self._next_eval <= self.num_timesteps:
                self._next_eval += self.eval_freq
            self._evaluate()
        return True

    def _predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _evaluate(self) -> None:
        env = self.eval_env_fn()
        ac_fn = ac_policy_fn(env)
        N = self.n_eval_episodes
        H = env.unwrapped.p.N

        rl_costs = np.empty(N)
        ac_costs = np.empty(N)
        rl_inv = np.empty((N, H + 1))
        ac_inv = np.empty((N, H + 1))

        for i in range(N):
            seed = self.seed_base + i
            env.reset(seed=seed)
            r = rollout(env, self._predict)
            rl_costs[i] = episode_cost(r["rewards"])
            rl_inv[i] = r["inventory"]

            env.reset(seed=seed)
            a = rollout(env, ac_fn)
            ac_costs[i] = episode_cost(a["rewards"])
            ac_inv[i] = a["inventory"]

        rl_mean = float(rl_costs.mean())
        rl_std = float(rl_costs.std())
        ac_mean = float(ac_costs.mean())
        ac_std = float(ac_costs.std())
        gap_pct = (rl_mean - ac_mean) / ac_mean * 100.0 if ac_mean != 0.0 else float("nan")
        inv_dev = float(np.abs(rl_inv - ac_inv).mean())

        self.logger.record("ac/rl_cost_mean", rl_mean)
        self.logger.record("ac/rl_cost_std", rl_std)
        self.logger.record("ac/ac_cost_mean", ac_mean)
        self.logger.record("ac/is_gap_pct", gap_pct)
        self.logger.record("ac/inv_l1_dev", inv_dev)

        if self.log_path is not None:
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writerow({
                    "timestep": int(self.num_timesteps),
                    "rl_cost_mean": rl_mean,
                    "rl_cost_std": rl_std,
                    "ac_cost_mean": ac_mean,
                    "ac_cost_std": ac_std,
                    "is_gap_pct": gap_pct,
                    "inv_l1_dev": inv_dev,
                })

        new_best = False
        if (
            self.save_best_path is not None
            and np.isfinite(rl_mean)
            and rl_mean < self.best_cost
        ):
            self.best_cost = rl_mean
            self.best_gap_at_best = gap_pct
            self.save_best_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_best_path))
            new_best = True

        if self.verbose:
            tag = "  *NEW BEST*" if new_best else ""
            print(
                f"[step {self.num_timesteps:>9}]  RL {rl_mean:.4e}  AC {ac_mean:.4e}  "
                f"gap {gap_pct:+.2f}%  inv_dev {inv_dev:.2f}{tag}"
            )
