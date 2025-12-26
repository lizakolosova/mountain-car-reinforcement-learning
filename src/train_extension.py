from custom_env_wrapper import CustomRewardWrapper
from stable_baselines3 import SAC
from utils import load_config

import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_extension(cfg_path: str = "./config/config_extension.yaml"):
    cfg = load_config(cfg_path)

    env_id = cfg["environment"]
    algo_name = cfg.get("algorithm", "SAC")
    total_timesteps = cfg["total_timesteps"]

    hyperparam_name = cfg["hyperparam_name"]
    hyperparam_values = cfg["hyperparam_values"]

    base_log_dir = cfg.get("log_dir", "logs/extension")

    print("=" * 60)
    print("TRAINING EXTENSION: HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Environment: {env_id}")
    print(f"Algorithm: {algo_name}")
    print(f"Hyperparameter: {hyperparam_name}")
    print(f"Values: {hyperparam_values}")
    print(f"Trials per value: 1  (extension requirement)")
    print(f"Timesteps per run: {total_timesteps:,}")
    print("=" * 60 + "\n")

    algo = SAC

    for value in hyperparam_values:
        print(f"\n{'#' * 60}")
        print(f"{hyperparam_name} = {value}")
        print(f"{'#' * 60}\n")

        # single run per value
        trial = 0

        log_dir = os.path.join(
            base_log_dir,
            f"{hyperparam_name}_{value}",
            f"{algo_name}_trial{trial}",
        )
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        env = gym.make(env_id)
        env = CustomRewardWrapper(env, cfg)
        print("✓ Environment wrapped with custom reward function (extension run)")

        sac_kwargs = {hyperparam_name: value}

        model = algo(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **sac_kwargs,
        )
        print("✓ Model initialized")

        cb = CheckpointCallback(
            save_freq=total_timesteps // 20,
            save_path=checkpoint_dir,
            name_prefix="rl_model",
        )
        print(f"✓ Checkpoints will be saved to: {checkpoint_dir}")

        print(f"\nStarting training for {total_timesteps:,} timesteps...")
        print(f"Monitor progress with: tensorboard --logdir={base_log_dir}\n")

        model.learn(total_timesteps=total_timesteps, callback=cb)
        print(f"\n✓ Training complete for {hyperparam_name} = {value}")

        results_dir = os.path.join(
            "results", "extension", f"{hyperparam_name}_{value}", algo_name
        )
        os.makedirs(results_dir, exist_ok=True)
        model.save(os.path.join(results_dir, f"trial{trial}"))
        print(f"✓ Final model saved to: {results_dir}/trial{trial}.zip")

        env.close()

    print("\n" + "=" * 60)
    print("ALL EXTENSION RUNS COMPLETE!")


if __name__ == "__main__":
    train_extension()
