import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC


def evaluate(model_path, env_id, episodes=10):
    model =  # Load your model
    env = gym.make(env_id)
    for _ in range(episodes):
        # Reset environment, initialize episode
        while not finished:
            # Predict actions with model (without exploration!!), apply them to environment, log metrics