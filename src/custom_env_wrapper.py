import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np


class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

        self.goal_position = 0.45
        self.min_position = -1.2
        self.last_action = None
        self.episode_steps = 0

        reward_cfg = cfg.get('reward_shaping', {})

        self.position_scale       = reward_cfg.get('position_scale', 1.0)
        self.velocity_scale       = reward_cfg.get('velocity_scale', 0.25)
        self.height_scale         = reward_cfg.get('height_scale', 0.5)
        self.energy_penalty_scale = reward_cfg.get('energy_penalty_scale', 0.05)
        self.goal_bonus           = reward_cfg.get('goal_bonus', 10.0)

        self.time_penalty         = reward_cfg.get('time_penalty', 0.01)
        self.near_goal_threshold  = reward_cfg.get('near_goal_threshold', 0.20)
        self.near_goal_scale      = reward_cfg.get('near_goal_scale', 1.0)

    def reset(self, **kwargs):
        self.episode_steps = 0
        self.last_action = None
        return super().reset(**kwargs)

    def step(self, action):
        self.last_action = action
        self.episode_steps += 1
        return super().step(action)

    def reward(self, original_reward: float) -> float:
        position = self.env.unwrapped.state[0]
        velocity = self.env.unwrapped.state[1]
        action = self.last_action if self.last_action is not None else np.array([0.0])

        position_normalized = (position - self.min_position) / (self.goal_position - self.min_position)
        position_normalized = np.clip(position_normalized, 0.0, 1.0)

        position_reward = (np.exp(position_normalized) - 1.0) * self.position_scale

        near_goal_bonus = 0.0
        if position > self.near_goal_threshold:
            distance_to_goal = max(self.goal_position - position, 1e-4)
            near_goal_bonus = self.near_goal_scale * np.exp(-5.0 * distance_to_goal)

        if position < 0.0:
            velocity_reward = abs(velocity) * self.velocity_scale
        else:
            velocity_reward = max(0.0, velocity) * self.velocity_scale

        height_reward = max(0.0, position) * self.height_scale

        energy_penalty = -np.square(action).sum() * self.energy_penalty_scale

        time_penalty = -self.time_penalty

        goal_bonus = 0.0
        if position >= self.goal_position:
            speed_bonus = max(0, 200 - self.episode_steps) * 0.05
            goal_bonus = self.goal_bonus + speed_bonus

        shaped_reward = (
            position_reward
            + velocity_reward
            + height_reward
            + near_goal_bonus
            + energy_penalty
            + time_penalty
            + goal_bonus
        )

        return original_reward * 0.1 + shaped_reward