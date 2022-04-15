import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNnet(nn.Module):

    """The Deep Network for training Deep Q-Network"""

    def __init__(self, input_dim, output_dim):
        """Initialize the network

        :input_dim: The dim of the states
        :output_dim: The dim of the actions

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


class Actor(nn.Module):

    """Neural network for actor-critic algorithm"""

    def __init__(self, input_dim: int, output_dim: int):
        """Initialize the network

        :input_dim: input dimension
        :output_dim: output dimentsion

        """
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._linear_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, state: torch.Tensor):
        """forward calculation"""
        x = F.relu(self._linear_layer(state))
        return x.tanh()


class Critic(nn.Module):

    """Critic network(action-value) for actor-critic algorithm"""

    def __init__(self, state_dim: int, action_dim: int):
        """Initialize the network

        :state_dim: state dimension
        :action_dim: action dimension

        """
        nn.Module.__init__(self)

        self._state_dim = state_dim
        self._action_dim = action_dim

        self._linear_layer = nn.Sequential(
            nn.Linear(self._state_dim + self._action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """forward calculation"""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self._linear_layer(x))
        return x
