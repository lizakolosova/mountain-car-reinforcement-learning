import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np


def evaluate(model_path, env_id, episodes=10):
    model = SAC.load(model_path)
    env = gym.make(env_id)
    returns = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        returns.append(ep_ret)

    env.close()
    returns = np.array(returns)
    mean_ret = returns.mean()
    std_ret = returns.std()
    print(f"Evaluated {episodes} episodes: mean return = {mean_ret:.2f}, std = {std_ret:.2f}")
    return mean_ret, std_ret, returns


if __name__ == "__main__":
    evaluate(
        # replace the path with the model path
        model_path="../src/logs/baseline/SAC_trial0/checkpoints/model.zip",
        env_id="MountainCarContinuous-v0",
        episodes=10,
    )

if __name__ == "__main__":
    env_id = "MountainCarContinuous-v0"

    print("\n=== learning_rate = 0.0001 ===")
    evaluate(
        model_path="results/extension/learning_rate_0.0001/SAC/trial0.zip",
        env_id=env_id,
        episodes=20,
    )

    print("\n=== learning_rate = 0.0003 ===")
    evaluate(
        model_path="results/extension/learning_rate_0.0003/SAC/trial0.zip",
        env_id=env_id,
        episodes=20,
    )

    print("\n=== learning_rate = 0.001 ===")
    evaluate(
        model_path="results/extension/learning_rate_0.001/SAC/trial0.zip",
        env_id=env_id,
        episodes=20,
    )
