from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim


class BaseAgent():

    """Base class for agent"""

    def __init__(
            self,
            env,
            actor,
            critic,
            buffer,
            gamma: float = 0.99,
            tau: float = 1e-3,
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3
    ):
        """Init the paras for agent

        :env: gym.Env
        :actor: NN as actor
        :critic: NN as critic
        :replay_buffer: The replay buffer for saving and sampling transition
        :gamma: Discount factor
        :tau: Soft update factor
        :actor_learning_rate: Learning rate for actor
        :critic_learning_rate: Learning rate for critic

        """

        self._env = env
        self._buffer = buffer
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

        # data recorder
        self._scores = list()
        self._actor_loss = list()
        self._critic_loss = list()

        # store tmp transitions
        self._transitions = list()
        self._curr_step = 1

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
        self._buffer.save(*self._transition)

        return next_state, reward, done

    def exploreEnv(self, max_step):
        """explore the env, save the trajectory to buffer

        :max_step: The maximum steps taking on env

        """

        step = 0
        score = 0
        state = self._env.reset()

        while step < max_step:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state

            score += reward
            step += 1
            self._curr_step += 1

            if done:
                state = self._env.reset()
                self._scores.append(score)
                score = 0

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
