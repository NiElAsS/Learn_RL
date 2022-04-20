from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tinyRL.util.net import ActorDet, CriticQ
from tinyRL.util.replay_buffer import ReplayBuffer


class BaseAgent():

    """Base class for agent"""

    def __init__(
            self,
            env,
            actor,
            critic,
            replay_buffer,
            gamma: float = 0.99,
            tau: float = 1e-3,
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3
    ):
        """Init the paras for agent

        :env: A gym.Env
        :actor: A NN as actor
        :critic: A NN as critic
        :replay_buffer: A replay buffer for saving and sampling transition
        :gamma: Discount factor
        :tau: Soft update factor
        :actor_learning_rate: Learning rate for actor
        :critic_learning_rate: Learning rate for critic

        """

        self._env = env
        self._replay_buffer = replay_buffer
        self._gamma = gamma
        self._tau = tau
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate

        # use gpu or cpu
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # dimension info
        self._observation_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]

        # init the network
        self._actor = actor.to(self._device)
        self._actor_target = deepcopy(self._actor).to(self._device)
        self._critic = critic.to(self._device)
        self._critic_target = deepcopy(self._critic).to(self._device)

        # optimizer
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._actor_learning_rate
        )
        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=self._critic_learning_rate
        )

        # store tmp transitions
        self._transitions = list()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """select action with respect to state

        :state: a numpy array of states
        :returns: a numpy array of actions

        """

        action = self._actor(
            torch.FloatTensor(state).to(self._device)
        ).detach().cpu().numpy()

        # save the transition info
        self._transition = [state, action]

        return action

    def step(self, action: np.ndarray) -> tuple:
        """Take the given action and return the data

        :action: the action need to be taken
        :returns: tuple(next_state, reward, done)

        """
        next_state, reward, done, _ = self._env.step(action)

        # save the transition info
        self._transition += [next_state, reward, done]
        self._replay_buffer.save(*self._transition)

        return next_state, reward, done

    def exploreEnv(self, max_step):
        """explore the env, save the trajectory to replay buffer

        :max_step: The maximum steps taking on env

        """

        done = False
        step = 0
        state = self._env.reset()
        while step < max_step or not done:
            action = self.select_action(state)
            _, _, done = self.step(action)
            step += 1

    @staticmethod
    def applyUpdate(optimizer: torch.optim.Optimizer, loss_func):
        """A help function to calculate the gradient and step.

        :optimizer: torch optimizer
        :loss_func: corresponding loss function

        """
        optimizer.zero_grad()
        loss_func.backward()
        optimizer.step()

    @staticmethod
    def softUpdate(target_network, network, tau):
        """Applying soft(delayed) update for target network"""

        for target_param, param in zip(
                target_network.parameters(), network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)


    def train(self):
        raise NotImplementedError
    def update(self):
        raise NotImplementedError
