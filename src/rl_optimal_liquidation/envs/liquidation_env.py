"""Discrete-time liquidation environment matching the PDF spec (§4)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class LiquidationParams:
    Q: float = 1.0e6              # total shares to liquidate
    T: float = 1.0                # horizon (units of time, e.g. one trading day)
    N: int = 50                   # number of discrete steps
    S0: float = 100.0             # initial midprice
    mu: float = 0.0               # drift
    sigma: float = 0.3            # midprice volatility
    eta: float = 1.0e-6           # temporary impact coefficient (h(v) = eta * v)
    gamma: float = 1.0e-7         # permanent impact coefficient (g(v) = gamma * v)
    lam: float = 1.0e-6           # risk aversion lambda
    terminal_penalty: float = 1.0e-3  # M in -M * q_N^2
    include_price_obs: bool = True    # add S/S0 to obs (irrelevant for LQ optimum)
    include_vol_obs: bool = False     # add sigma(t)/sigma to obs (informative when sigma_profile != constant)
    sigma_profile: str = "constant"   # "constant" or "u_shaped"
    sigma_amplitude: float = 0.0      # A in sigma(t) = sigma * (1 + A * cos(2 pi t / T)) for u_shaped
    sigma_noise_std: float = 0.0      # Per-episode multiplicative noise: scale ~ N(1, σ_noise). With
                                      # deterministic σ(t), t alone predicts σ̂ and the agent can't be
                                      # forced to read its σ̂ observation. Adding scale randomness
                                      # decorrelates σ̂ from t and tests genuine closed-loop conditioning.

    def __post_init__(self):
        # YAML 1.1 parses "1.0e6" as a string (no sign on the exponent).
        # Coerce here so the env can't silently get a string-typed param.
        self.Q = float(self.Q)
        self.T = float(self.T)
        self.N = int(self.N)
        self.S0 = float(self.S0)
        self.mu = float(self.mu)
        self.sigma = float(self.sigma)
        self.eta = float(self.eta)
        self.gamma = float(self.gamma)
        self.lam = float(self.lam)
        self.terminal_penalty = float(self.terminal_penalty)
        # bool("false") is True, so string-typed YAML values need explicit parsing.
        self.include_price_obs = self._coerce_bool(self.include_price_obs)
        self.include_vol_obs = self._coerce_bool(self.include_vol_obs)
        self.sigma_amplitude = float(self.sigma_amplitude)
        self.sigma_noise_std = float(self.sigma_noise_std)
        if self.sigma_profile not in ("constant", "u_shaped"):
            raise ValueError(
                f"sigma_profile must be 'constant' or 'u_shaped', got {self.sigma_profile!r}"
            )

    @staticmethod
    def _coerce_bool(v) -> bool:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "yes", "on"):
                return True
            if s in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"cannot interpret {v!r} as a bool")
        return bool(v)

    def dt(self) -> float:
        return self.T / self.N

    def sigma_at(self, t: float, scale: float = 1.0) -> float:
        """Time-varying volatility at time t with optional per-episode scale multiplier.

        For sigma_profile='u_shaped': sigma(t) = sigma * scale * (1 + A cos(2 pi t / T)).
        The scale defaults to 1.0; the env samples a per-episode value from
        N(1, sigma_noise_std) in reset() and passes it here, decorrelating
        σ̂ from t so the agent must read its σ̂ observation to act optimally.
        """
        if self.sigma_profile == "constant":
            return self.sigma * scale
        return self.sigma * scale * (
            1.0 + self.sigma_amplitude * float(np.cos(2.0 * np.pi * t / self.T))
        )


class LiquidationEnv(gym.Env):
    """Liquidation env supporting Phase 1 (linear impact, constant sigma) and
    Phase 2 (time-varying sigma) via params on LiquidationParams.

    State:    (k/N, q_k/Q)                            — minimal
              + S_k/S_0                                — if include_price_obs
              + sigma(t_k)/sigma                       — if include_vol_obs
    Action:   f_k in [0, 1], shares sold this step a_k = f_k * q_k
    Reward:   -(eta a_k^2 / dt  +  gamma a_k^2  +  lam sigma(t_k)^2 q_k^2 dt)
              with terminal penalty -M * q_N^2 if inventory remains at step N.
    """

    metadata = {"render_modes": []}

    def __init__(self, params: LiquidationParams | None = None, seed: int | None = None):
        super().__init__()
        self.p = params or LiquidationParams()
        self._dt = self.p.dt()

        low = [0.0, 0.0]
        high = [1.0, 1.0]
        if self.p.include_price_obs:
            low.append(0.0)
            high.append(np.finfo(np.float32).max)
        if self.p.include_vol_obs:
            low.append(0.0)
            high.append(np.finfo(np.float32).max)
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.k = 0
        self.q = self.p.Q
        self.S = self.p.S0
        self._sigma_scale = 1.0  # populated each reset; multiplier on the sigma profile.

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.k = 0
        self.q = self.p.Q
        self.S = self.p.S0
        if self.p.sigma_noise_std > 0.0:
            # Clamp to a small positive floor to avoid pathological near-zero sigma.
            self._sigma_scale = max(
                1.0 + float(self._rng.normal(0.0, self.p.sigma_noise_std)), 0.1
            )
        else:
            self._sigma_scale = 1.0
        return self._obs(), {
            "params": asdict(self.p),
            "sigma_scale": self._sigma_scale,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        f = float(np.clip(action[0], 0.0, 1.0))
        a = f * self.q

        t_k = self.k * self._dt
        sigma_k = self.p.sigma_at(t_k, scale=self._sigma_scale)

        temp_cost = self.p.eta * a * a / self._dt
        perm_cost = self.p.gamma * a * a
        perm_price_drop = self.p.gamma * a
        inv_pen = self.p.lam * (sigma_k ** 2) * (self.q ** 2) * self._dt
        reward = -(temp_cost + perm_cost + inv_pen)

        self.q = max(self.q - a, 0.0)

        eps = self._rng.standard_normal()
        self.S = self.S + self.p.mu * self._dt \
                       + sigma_k * np.sqrt(self._dt) * eps \
                       - perm_price_drop

        self.k += 1
        terminated = self.k >= self.p.N
        if terminated and self.q > 0.0:
            reward -= self.p.terminal_penalty * (self.q ** 2)

        info = {
            "executed": a,
            "inventory": self.q,
            "price": self.S,
            "temp_cost": temp_cost,
            "perm_cost": perm_cost,
            "inv_pen": inv_pen,
        }
        return self._obs(), float(reward), bool(terminated), False, info

    def _obs(self) -> np.ndarray:
        base = [self.k / self.p.N, self.q / self.p.Q]
        if self.p.include_price_obs:
            base.append(self.S / self.p.S0)
        if self.p.include_vol_obs:
            base.append(
                self.p.sigma_at(self.k * self._dt, scale=self._sigma_scale) / self.p.sigma
            )
        return np.array(base, dtype=np.float32)
