from copy import deepcopy

from tinyRL.util.replay_buffer import ReplayBufferDev
from tinyRL.util.net import ActorDet, CriticQ

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BaseAgent():

    """Base class for agent"""

    def __init__(
            self,
            config
    ):
        """Init the agent"""

        # use gpu or cpu
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # env
        self._env = getattr(config, "env")
        self._state_dim = getattr(config, "state_dim")
        self._action_dim = getattr(config, "action_dim")

        # buffer
        self._max_buffer_size = getattr(config, "max_buffer_size")
        self._batch_size = getattr(config, "buffer_batch_size")
        self._buffer = ReplayBufferDev(
            self._state_dim,
            self._action_dim,
            self._batch_size,
            self._max_buffer_size
        )

        self._gamma = getattr(config, "gamma")
        self._tau = getattr(config, "tau_soft_update")
        self._actor_lr = getattr(config, "actor_learning_rate")
        self._critic_lr = getattr(config, "critic_learning_rate")

        # training
        self._max_train_step = getattr(config, "max_train_step")
        self._rollout_step = getattr(config, "rollout_step")
        self._update_repeat_times = getattr(config, "update_repeat_times")

        # init the network
        self._actor = ActorDet(
            self._state_dim, self._action_dim).to(self._device)
        self._actor_target = deepcopy(self._actor).to(self._device)

        self._critic = CriticQ(
            self._state_dim, self._action_dim).to(self._device)
        self._critic_target = deepcopy(self._critic).to(self._device)

        # optimizer
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._actor_lr
        )
        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=self._critic_lr
        )

        # loss func
        self._loss_func = nn.SmoothL1Loss()

        # data recorder
        self._scores = list()
        self._actor_loss = list()
        self._critic_loss = list()

        # store tmp transitions
        self._transition = list()
        self._curr_step = 0

    def selectAction(self, state: torch.Tensor) -> torch.Tensor:
        """select action with respect to state

        :state: a torch.Tensor of state
        :returns: a torch.Tensor of action

        """

        action_tensor = self._actor(state)

        return action_tensor

    def step(self, action: np.ndarray) -> tuple:
        """Take the given action and return the data

        :action: the action need to be taken
        :returns: tuple(next_state, reward, done)

        """
        next_state, reward, done, _ = self._env.step(action)

        return next_state, reward, done

    def exploreEnv(self):
        """explore the env, save the trajectory to buffer

        :max_step: The maximum steps taking on env

        """

        step = 0
        score = 0
        traj = []
        state = self._env.reset()
        done = False

        while step < self._rollout_step or not done:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32
            )

            action_tensor = self.selectAction(
                state_tensor.to(self._device)
            ).detach().cpu()  # cpu

            next_state, reward, done = self.step(action_tensor.numpy())

            transition = [state_tensor, action_tensor,
                          reward, 1-done]
            traj.append(self.transToTensor(transition))

            # update the vars
            state = next_state
            score += reward
            step += 1
            if done:
                self._scores.append(score)

                score = 0
                state = self._env.reset()

        self._buffer.saveTrajectory(traj)
        self._curr_step += step

    @staticmethod
    def transToTensor(transition) -> tuple:
        return tuple(torch.as_tensor(item).reshape(1, -1) for item in transition)

    @staticmethod
    def applyUpdate(optimizer: torch.optim.Optimizer, loss_func):
        """A help function to calculate the gradient and step.

        :optimizer: torch optimizer
        :loss_func: corresponding loss function

        """
        optimizer.zero_grad()
        loss_func.backward()
        optimizer.step()

    @ staticmethod
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
