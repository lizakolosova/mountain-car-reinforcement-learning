import gymnasium as gym
from gymnasium import RewardWrapper


class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

    def reward(self, reward: float) -> float:
        """
        The reward parameter is the default reward
        """
        # TODO: Implement your own reward
        return reward