"""Train PPO on the liquidation env (Phase 1 or Phase 2, selected via --config)."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_optimal_liquidation.callbacks import ACBaselineCallback
from rl_optimal_liquidation.envs import LiquidationEnv, LiquidationParams
from rl_optimal_liquidation.policies import BetaActorCriticPolicy
from rl_optimal_liquidation.wrappers import ControlVariateReward


def make_env_thunk(env_cfg: dict, control_variate: str | None = None):
    def thunk():
        env = LiquidationEnv(LiquidationParams(**env_cfg))
        if control_variate:
            env = ControlVariateReward(env, mode=control_variate)
        return Monitor(env)
    return thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--total-timesteps", type=int, default=1_000_000,
                    help="1M is the stabilized Beta-policy baseline; 500k is enough"
                         " for Gaussian but Beta needs the extra time to converge.")
    ap.add_argument("--output", default="runs/phase1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-freq", type=int, default=10_000,
                    help="Steps between AC-baseline evaluations")
    ap.add_argument("--eval-episodes", type=int, default=20,
                    help="Episodes per AC-baseline evaluation")
    ap.add_argument("--no-progress-bar", action="store_true",
                    help="Disable the tqdm/rich progress bar (for logged batch runs)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_envs = int(cfg.get("n_envs", 4))
    # Optional training-time control variate: reward becomes -(cost - classical
    # reference cost) for the same scenario. Pure variance reduction — the
    # reference never depends on the agent's actions, so the optimum is
    # unchanged. Eval (callback / evaluate.py) always uses raw-cost envs.
    control_variate = cfg.get("control_variate")
    env = DummyVecEnv([make_env_thunk(cfg["env"], control_variate) for _ in range(n_envs)])

    ppo_kwargs = dict(cfg.get("ppo", {}))

    # Optional LR schedule: SB3 calls the callable with progress_remaining
    # (1.0 at start, 0.0 at end). Linear-to-zero is the standard stochastic-
    # approximation cure for late-training drift around a deterministic optimum.
    lr_schedule = ppo_kwargs.pop("lr_schedule", "constant")
    if lr_schedule == "linear":
        base_lr = float(ppo_kwargs["learning_rate"])
        ppo_kwargs["learning_rate"] = lambda progress_remaining: progress_remaining * base_lr
    elif lr_schedule == "two_phase":
        base_lr = float(ppo_kwargs["learning_rate"])
        # Constant LR for the first lr_const_fraction of training, then anneal
        # linearly to 0 over the remaining portion. lr_const_fraction=0.8 leaves
        # only the last 20% for annealing (safe but mostly post-save-best);
        # lr_const_fraction=0.2 anneals for 80% of training (much more aggressive,
        # save-best samples from the annealed phase).
        const_frac = float(ppo_kwargs.pop("lr_const_fraction", 0.8))
        anneal_threshold = 1.0 - const_frac
        def schedule(progress_remaining):
            if progress_remaining > anneal_threshold:
                return base_lr
            return base_lr * (progress_remaining / anneal_threshold)
        ppo_kwargs["learning_rate"] = schedule
    elif lr_schedule != "constant":
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}; use 'constant', 'linear', or 'two_phase'")

    # Policy choice: gaussian (SB3 default MlpPolicy) or beta (our custom Box(0,1) policy).
    policy_choice = ppo_kwargs.pop("policy", "gaussian")
    if policy_choice == "gaussian":
        policy_class = "MlpPolicy"
    elif policy_choice == "beta":
        policy_class = BetaActorCriticPolicy
    else:
        raise ValueError(f"unknown policy {policy_choice!r}; use 'gaussian' or 'beta'")

    # Rewards in this env are O(1e6)-O(1e9), which swamps the value head.
    # VecNormalize tracks a running std of returns and rescales rewards to ~O(1)
    # so the value function can learn. norm_obs=False: the env already emits
    # normalized features (k/N, q/Q, sigma_hat/sigma), so eval can run on raw
    # envs and the callback / diagnose.py report true cost units.
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=float(ppo_kwargs.get("gamma", 0.99)),
    )

    # tensorboard_log is omitted: we configure the logger explicitly below so
    # we can also emit progress.csv alongside the TB events.
    model = PPO(
        policy_class,
        env,
        verbose=1,
        seed=args.seed,
        **ppo_kwargs,
    )

    # SB3 logger -> stdout, CSV (out_dir/tb/progress.csv), TensorBoard events.
    tb_dir = out_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(tb_dir), ["stdout", "csv", "tensorboard"]))

    callback = ACBaselineCallback(
        eval_env_fn=lambda: LiquidationEnv(LiquidationParams(**cfg["env"])),
        n_eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        log_path=out_dir / "ac_eval.csv",
        save_best_path=out_dir / "best_model.zip",
        verbose=1,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=not args.no_progress_bar,
    )
    model.save(out_dir / "model")
    env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"saved final model to {out_dir / 'model.zip'}")
    best_path = out_dir / "best_model.zip"
    if best_path.exists():
        print(f"best model saved to {best_path} "
              f"(cost {callback.best_cost:.4e}, gap {callback.best_gap_at_best:+.2f}%)")


if __name__ == "__main__":
    main()
