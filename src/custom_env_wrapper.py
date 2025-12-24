import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np


class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)

        reward_cfg = cfg.get("reward_shaping", {})

        self.goal_position = 0.45
        self.min_position = -1.2
        self.episode_steps = 0
        self.last_action = None

        self.position_scale = reward_cfg.get("position_scale", 8.0)
        self.velocity_scale = reward_cfg.get("velocity_scale", 4.0)
        self.height_scale = reward_cfg.get("height_scale", 6.0)
        self.energy_penalty_scale = reward_cfg.get("energy_penalty_scale", 0.05)
        self.goal_bonus = reward_cfg.get("goal_bonus", 100.0)

        self.time_penalty = reward_cfg.get("time_penalty", 0.05)
        self.near_goal_threshold = reward_cfg.get("near_goal_threshold", 0.35)
        self.near_goal_scale = reward_cfg.get("near_goal_scale", 50.0)

    def reset(self, **kwargs):
        self.episode_steps = 0
        self.last_action = None
        return super().reset(**kwargs)

    def step(self, action):
        self.last_action = action
        self.episode_steps += 1
        return super().step(action)

    def reward(self, original_reward: float) -> float:
        position, velocity = self.env.unwrapped.state
        action = self.last_action if self.last_action is not None else np.array([0.0])

        pos_norm = (position - self.min_position) / (self.goal_position - self.min_position)
        pos_norm = np.clip(pos_norm, 0.0, 1.0)

        position_reward = pos_norm * self.position_scale
        velocity_reward = max(0.0, velocity) * self.velocity_scale
        height_reward = max(0.0, position) * self.height_scale

        energy_penalty = -np.square(action).sum() * self.energy_penalty_scale
        time_penalty = -self.time_penalty

        near_goal_bonus = 0.0
        if position > self.near_goal_threshold:
            near_goal_bonus = self.near_goal_scale * (position - self.near_goal_threshold)

        goal_bonus = self.goal_bonus if position >= self.goal_position else 0.0

        return (
            original_reward
            + position_reward
            + velocity_reward
            + height_reward
            + near_goal_bonus
            + energy_penalty
            + time_penalty
            + goal_bonus
        )