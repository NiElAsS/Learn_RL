import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNnet(nn.Module):

    """The Deep Network for training Deep Q-Network"""

    def __init__(self, input_dim, output_dim):
        """Initialize the network

        :input_dim: TODO
        :output_dim: TODO

        """
        nn.Module.__init__(self)

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._linear_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, state):
        """forward calculate"""
        return self._linear_layer(state)
