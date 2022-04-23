import torch
import torch.nn as nn
from torch.distributions import Normal


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
        self._fully_conn_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, state):
        """forward calculate"""
        return self._fully_conn_layer(state)


class ActorDet(nn.Module):

    """Neural network for actor-critic algorithm"""

    def __init__(self, input_dim: int, output_dim: int):
        """Initialize the network

        :input_dim: input dimension
        :output_dim: output dimension

        """
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._fully_conn_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, state: torch.Tensor):
        """forward calculation"""
        x = self._fully_conn_layer(state)
        return x.tanh()


class ActorSto(nn.Module):

    """stochastic actor NN for actor-critic algorithm"""

    def __init__(self, input_dim: int, output_dim: int):
        """Init the network, use Normal distribution

        :input_dim: Input dimension(state dimension)
        :output_dim: Output dimenstion
        :max_std: Maximum value of standard deviation of Normal distribution

        """
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._fully_conn_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # compute the mu and std for Normal distribution
        self._mu = nn.Linear(32, output_dim)
        self._std = nn.Linear(32, output_dim)

    def forward(self, state: torch.Tensor) -> tuple:
        """Forward calculation"""
        x = self._fully_conn_layer(state)

        # map action mean to (-1, 1)
        mu = torch.tanh(self._mu(x))

        # map action std to (0,max_std)
        log_std = torch.tanh(self._std(x))
        std = log_std.exp()

        # build the distribution and sample the action
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class CriticQ(nn.Module):

    """Critic network(action-value) for actor-critic algorithm"""

    def __init__(self, state_dim: int, action_dim: int):
        """Initialize the network

        :state_dim: state dimension
        :action_dim: action dimension

        """
        super().__init__()

        self._state_dim = state_dim
        self._action_dim = action_dim

        self._fully_conn_layer = nn.Sequential(
            nn.Linear(self._state_dim + self._action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """forward calculation"""
        x = torch.cat((state, action), dim=-1)
        x = self._fully_conn_layer(x)
        return x


class CriticV(nn.Module):

    """state-value critic network for actor-critic algorithm"""

    def __init__(self, input_dim: int):
        """Init the network

        :input_dim: input dimension of state

        """
        super().__init__()

        self._input_dim = input_dim
        self._fully_conn_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor):
        """Forward calculation"""
        return self._fully_conn_layer(state)
