from stable_baselines3 import PPO, DQN, A2C, SAC
import gymnasium as gym
from utils import load_config
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_baseline(cfg_path="config/config_baseline.yaml"):
    cfg = load_config(cfg_path)
    for trial in cfg['num_trials']:
        env =  # Create environment
        algo =  # Pick Algorithm
        model = algo("MlpPolicy", env, verbose=1, tensorboard_log=f"logs/baseline/{cfg['algorithm']}_trial{trial}/")
        cb = CheckpointCallback(save_freq=cfg["checkpoint_freq"],
                                save_path=f"logs/baseline/{cfg['algorithm']}_trial{trial}/checkpoints/")
        model.learn(total_timesteps=cfg["timesteps"], callback=cb)
        model.save(f"results/baseline/{cfg['algorithm']}_trial{1}")


if __name__ == '__main__': train_baseline()
