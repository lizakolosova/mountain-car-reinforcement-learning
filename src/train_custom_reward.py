import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from utils import load_config
from custom_env_wrapper import CustomRewardWrapper

def train_custom(cfg_path="config/config_custom.yaml"):
    cfg = load_config(cfg_path)

    for trial in range(cfg['num_trials']):

        log_path = f"logs/custom/SAC_trial{trial}/"
        os.makedirs(f"{log_path}checkpoints/", exist_ok=True)

        env = CustomRewardWrapper(gym.make(cfg['environment']), cfg)
        env = Monitor(env)

        eval_env = Monitor(CustomRewardWrapper(gym.make(cfg['environment']), cfg))

        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

        cb = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=f"{log_path}checkpoints/",
            name_prefix="rl_model"
        )

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=f"{log_path}best_model",
            log_path=f"{log_path}results",
            eval_freq=5000,
            deterministic=True
        )

        model.learn(total_timesteps=cfg["timesteps"], callback=[cb, eval_cb])
        model.save(f"results/custom/SAC_trial{trial}")

        env.close()
        eval_env.close()


if __name__ == '__main__':
    train_custom()