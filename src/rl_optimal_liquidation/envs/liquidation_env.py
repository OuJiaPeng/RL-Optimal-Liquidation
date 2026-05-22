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
    gamma: float = 1.0e-7         # permanent impact coefficient (g(v) = gamma * v).
                                  # Linear branch: enters ONLY through midprice dynamics
                                  # (S -= gamma*a), not through the per-step cost. This
                                  # matches textbook Almgren-Chriss, where the permanent-
                                  # impact contribution to total cost integrates to the
                                  # schedule-independent constant (1/2)*gamma*Q^2 and so
                                  # drops out of the optimization. The agent only sees
                                  # the price drift if include_price_obs=True; with the
                                  # default include_price_obs=False, gamma has no observable
                                  # effect on training in the linear regime.
                                  # Sqrt branch (Phase-2 dead-end, not shipped) keeps
                                  # gamma*a^(3/2) in the per-step cost as a stylized
                                  # concave-impact penalty.
    lam: float = 1.0e-6           # risk aversion lambda
    terminal_penalty: float = 1.0e-3  # M in -M * q_N^2
    include_price_obs: bool = True    # add S/S0 to obs (irrelevant for LQ optimum)
    include_vol_obs: bool = False     # add sigma(t)/sigma to obs (informative when sigma_profile != constant)
    impact_type: str = "linear"       # "linear" (Phase 1/AC) or "sqrt" (concave perm impact)
    sigma_profile: str = "constant"   # "constant" or "u_shaped"
    sigma_amplitude: float = 0.0      # A in sigma(t) = sigma * (1 + A * cos(2 pi t / T)) for u_shaped
    sigma_noise_std: float = 0.0      # Per-episode multiplicative noise: scale ~ N(1, σ_noise). With
                                      # deterministic σ(t), t alone predicts σ̂ and the agent can't be
                                      # forced to read its σ̂ observation. Adding scale randomness
                                      # decorrelates σ̂ from t and tests genuine closed-loop conditioning.
    eta_rho: float = 0.0              # AR(1) persistence on log(η). 0 = constant η = self.eta.
    eta_noise_std: float = 0.0        # Std of per-step Gaussian innovation on log(η). > 0 turns η into a
                                      # log-AR(1) process around log(self.eta) — tests whether the agent
                                      # can condition on a temp-impact (spread) observation that affects
                                      # cost *this step*, not 30 steps later (unlike σ̂ via inventory penalty).
    include_eta_obs: bool = False     # Add η_k / self.eta to obs when η is stochastic.

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
        self.include_price_obs = bool(self.include_price_obs)
        self.include_vol_obs = bool(self.include_vol_obs)
        self.sigma_amplitude = float(self.sigma_amplitude)
        self.sigma_noise_std = float(self.sigma_noise_std)
        self.eta_rho = float(self.eta_rho)
        self.eta_noise_std = float(self.eta_noise_std)
        self.include_eta_obs = bool(self.include_eta_obs)
        if self.impact_type not in ("linear", "sqrt"):
            raise ValueError(
                f"impact_type must be 'linear' or 'sqrt', got {self.impact_type!r}"
            )
        if self.sigma_profile not in ("constant", "u_shaped"):
            raise ValueError(
                f"sigma_profile must be 'constant' or 'u_shaped', got {self.sigma_profile!r}"
            )

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
    Phase 2 wedges (sqrt impact, time-varying sigma) via params on LiquidationParams.

    State:    (k/N, q_k/Q)                            — minimal
              + S_k/S_0                                — if include_price_obs
              + sigma(t_k)/sigma                       — if include_vol_obs
    Action:   f_k in [0, 1], shares sold this step a_k = f_k * q_k
    Reward:   -(eta a_k^2 / dt  +  perm(a_k)  +  lam sigma(t_k)^2 q_k^2 dt)
              with terminal penalty -M * q_N^2 if inventory remains at step N.
              perm(a) = 0 in the linear branch (textbook AC: permanent impact's
              contribution to total cost is the schedule-independent constant
              (1/2)*gamma*Q^2, which drops from the optimization);
              perm(a) = gamma * a^(3/2) in the sqrt branch (stylized penalty).
              In both branches, gamma still updates the midprice via the price-
              dynamics term S -= gamma*a (linear) or gamma*sqrt(a) (sqrt).
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
        if self.p.include_eta_obs:
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
        self._eta_k = float(self.p.eta)  # current η; evolves per step if eta_noise_std > 0.

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
        # eta state: start at the mean. With log-AR(1), η_0 = eta_mean.
        self._eta_k = float(self.p.eta)
        return self._obs(), {
            "params": asdict(self.p),
            "sigma_scale": self._sigma_scale,
            "eta_0": self._eta_k,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        f = float(np.clip(action[0], 0.0, 1.0))
        a = f * self.q

        t_k = self.k * self._dt
        sigma_k = self.p.sigma_at(t_k, scale=self._sigma_scale)

        # Reward is the negative of this step's cost contribution. Use self._eta_k
        # which matches the η in the agent's current obs (set at end of prev step
        # or in reset). After cost is computed we evolve η for the next obs.
        temp_cost = self._eta_k * a * a / self._dt
        if self.p.impact_type == "linear":
            # Textbook AC: permanent impact contributes the schedule-independent
            # constant (1/2)*γ*Q^2 to total cost and so drops from the optimization.
            # We omit it from the per-step reward entirely; γ still affects the
            # simulated midprice via perm_price_drop below.
            perm_cost = 0.0
            perm_price_drop = self.p.gamma * a
        else:  # sqrt: g(v) = γ√v -> stylized per-step cost γ·a^(3/2). Dead-end
               # arm from Phase 2.1; kept for completeness but not shipped.
            sqrt_a = np.sqrt(max(a, 0.0))
            perm_cost = self.p.gamma * a * sqrt_a
            perm_price_drop = self.p.gamma * sqrt_a
        inv_pen = self.p.lam * (sigma_k ** 2) * (self.q ** 2) * self._dt
        reward = -(temp_cost + perm_cost + inv_pen)

        self.q = max(self.q - a, 0.0)

        eps = self._rng.standard_normal()
        self.S = self.S + self.p.mu * self._dt \
                       + sigma_k * np.sqrt(self._dt) * eps \
                       - perm_price_drop

        # Evolve η for the NEXT step (so the next obs we return shows the η
        # that will price the agent's next action). Uses a separate RNG draw
        # from the price noise.
        if self.p.eta_noise_std > 0.0:
            eps_eta = float(self._rng.standard_normal())
            log_mean = float(np.log(self.p.eta))
            log_eta = float(np.log(max(self._eta_k, 1e-300)))
            new_log_eta = (
                (1.0 - self.p.eta_rho) * log_mean
                + self.p.eta_rho * log_eta
                + self.p.eta_noise_std * np.sqrt(self._dt) * eps_eta
            )
            self._eta_k = float(np.exp(new_log_eta))

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
        if self.p.include_eta_obs:
            base.append(self._eta_k / self.p.eta)
        return np.array(base, dtype=np.float32)
