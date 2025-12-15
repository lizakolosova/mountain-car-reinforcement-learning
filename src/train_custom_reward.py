from custom_env_wrapper import CustomRewardWrapper
from stable_baselines3 import SAC
from utils import load_config
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_custom(cfg_path="config/config_custom.yaml"):
    cfg = load_config(cfg_path)

    print("=" * 60)
    print("TRAINING WITH CUSTOM REWARD")
    print("=" * 60)
    print(f"Environment: {cfg['environment']}")
    print(f"Algorithm: {cfg['algorithm']}")
    print(f"Number of trials: {cfg['num_trials']}")
    print(f"Timesteps per trial: {cfg['timesteps']:,}")
    print("=" * 60 + "\n")

    for trial in range(cfg['num_trials']):
        print(f"\n{'#' * 60}")
        print(f"Starting Trial {trial}")
        print(f"{'#' * 60}\n")

        env = gym.make(cfg['environment'])

        env = CustomRewardWrapper(env, cfg)
        print("✓ Environment wrapped with custom reward function")

        algo = SAC

        model = algo(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"logs/custom/{cfg['algorithm']}_trial{trial}/"
        )
        print("✓ Model initialized")

        checkpoint_dir = f"logs/custom/{cfg['algorithm']}_trial{trial}/checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)

        cb = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=checkpoint_dir,
            name_prefix="rl_model"
        )
        print(f"✓ Checkpoints will be saved to: {checkpoint_dir}")

        print(f"\nStarting training for {cfg['timesteps']:,} timesteps...")
        print("Monitor progress with: tensorboard --logdir=logs/custom\n")

        model.learn(total_timesteps=cfg["timesteps"], callback=cb)

        print(f"\n✓ Training complete for trial {trial}")

        results_dir = f"results/custom/{cfg['algorithm']}/"
        os.makedirs(results_dir, exist_ok=True)
        model.save(f"{results_dir}trial{trial}")
        print(f"✓ Final model saved to: {results_dir}trial{trial}.zip")

        env.close()

    print("\n" + "=" * 60)
    print("ALL TRIALS COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Evaluate models: python evaluate.py")
    print("2. Compare with baseline using TensorBoard")
    print("3. Analyze results for your report")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    train_custom()