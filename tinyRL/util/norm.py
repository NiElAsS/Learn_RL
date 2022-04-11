import gym
import numpy as np


class ActionNormalizer(gym.ActionWrapper):

    """Normalize the range of actions"""

    def action(self, action: np.ndarray):
        """Rescale the range [-1, 1] to [low, high]

        :action: TODO

        """

        self._action = action
        low = self.action_space.low
        high = self.action_space.high

        action_scale = (high - low) / 2
        action_mid = high - action_scale

        action = action * action_scale + action_mid
        action = np.clip(action, low, high)
        return action

    def normAction(self, action: np.ndarray):
        """Rescale the action [low, high] to [-1, 1]"""
        low = self.action_space.low
        high = self.action_space.high

        action_scale = (high - low) / 2
        action_mid = high - action_scale

        action = (action - action_mid) / action_scale
        action = np.clip(action, 1.0, -1.0)
        return action
