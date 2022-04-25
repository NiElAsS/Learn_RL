import os
import sys

from tinyRL.util import ReplayBuffer, ReplayBufferDev
from tinyRL.util import DQNnet
from tinyRL.util.configurator import Configurator
from tinyRL.agents import BaseAgent
from tqdm import trange
from copy import deepcopy
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import gym


class DQN(BaseAgent):

    """Create a DQN agent for training and evaluation"""

    def __init__(self, config):
        """Init the agent

        :config: TODO

        """
        super().__init__(config)

        # use DQNnet
        self._action_number = getattr(
            config, "action_number", self._env.action_space.n
        )

        self._critic = DQNnet(
            self._state_dim, self._action_number
        ).to(self._device)
        self._critic_target = deepcopy(self._critic).to(self._device)
        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=self._critic_lr
        )

        # use epsilon_greedy
        self._eps = getattr(config, "max_epsilon_greedy_rate", 0.95)
        self._max_eps = getattr(config, "max_epsilon_greedy_rate", 0.95)
        self._min_eps = getattr(config, "min_epsilon_greedy_rate", 0.05)
        self._eps_decay_step = getattr(
            config, "epsilon_greedy_rate_decay", 10000)
        self._eps_decay_rate = (
            self._max_eps - self._min_eps) / self._eps_decay_step

    def selectAction(self, state: torch.Tensor) -> torch.Tensor:
        """select action using epsilon greedy

        :state: a torch.Tensor of state
        :returns: a torch.Tensor of action

        """
        if torch.rand(1) < self._eps:
            action = self._env.action_space.sample()
            action = torch.as_tensor(action, dtype=torch.int64)
        else:
            action = self._critic(state).argmax()

        return action

    def update(self):
        """Update the agent network"""

        updates_times = int(
            1+self._update_repeat_times * self._buffer._curr_size / self._batch_size
        )
        for _ in range(updates_times):
            with torch.no_grad():
                state, action, next_state, reward, mask = (
                    self._buffer.sampleBatch()
                )
                # compute target for training
                target_q = self._critic_target(
                    next_state).max(dim=1, keepdim=True)[0]
                y = reward + mask * self._gamma * target_q

            # compute loss
            curr_q = torch.gather(self._critic(state), 1,
                                  action.to(dtype=torch.int64))
            loss = self._loss_func(curr_q, y)

            # update the network
            self.applyUpdate(self._critic_optimizer, loss)

            # update the target network
            self.softUpdate(self._critic_target, self._critic, self._tau)

    def train(self):
        """Train the agent"""

        # init
        self._scores = []
        self._actor_losses = []
        self._critic_losses = []
        print_count = 0

        while self._curr_step < self._max_train_step:
            self.exploreEnv()
            self.update()

            self._eps = self._max_eps - self._curr_step * self._eps_decay_rate
            self._eps = max(self._min_eps, self._eps)

            print_count += 1
            if print_count % 10 == 0:
                print(
                    f"Current step: {self._curr_step} Last 100 exploration mean score: {torch.tensor(self._scores[-100:]).mean()}"
                )

        self._env.close()


class DQNagent():

    """Create a DQN agent for training"""

    def __init__(self,
                 env: gym.Env,
                 buffer_size: int,
                 batch_size: int,
                 target_update_period: int,
                 epsilon_decay: float,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 gamma: float = 0.99):
        """Initialize the agents

        :env: The environment in which the agents are training
        :buffer_size: The size of ReplayBuffer
        :batch_size: The size of each batch drawn from ReplayBuffer
        :target_update_period: The period that target network clones current network
        :epsilon_decay: The step that epsilon decays to min_epsilon
        :max_epsilon: The maximum value of epsilon
        :min_epsilon: The minimum value of epsilon
        :gamma: The reward discount

        """

        self._env = env
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._target_update_period = target_update_period
        self._epsilon_decay = epsilon_decay
        self._eps = max_epsilon
        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._gamma = gamma

        self._state_dim = self._env.observation_space.shape[0]
        self._action_dim = self._env.action_space.n

        self._buffer = ReplayBuffer(
            self._state_dim, self._buffer_size, self._batch_size)

        # choose gpu or cpu
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Current devices: {}".format(self._device))

        # init the network
        self._dqn_net = DQNnet(
            self._state_dim, self._action_dim).to(self._device)
        self._target_dqn_net = DQNnet(
            self._state_dim, self._action_dim).to(self._device)
        self._target_dqn_net.load_state_dict(self._dqn_net.state_dict())
        self._target_dqn_net.eval()

        # set the optimizer for training
        self._optimizer = optim.Adam(self._dqn_net.parameters())

        # set the Loss function
        self._loss_func = F.smooth_l1_loss

        # set the transition for ReplayBuffer
        self._transition = list()

    def selectAction(self, state: np.ndarray):
        """selec a action, epsilon_greedy apply"""

        if self._eps > np.random.random():
            selected_action = self._env.action_space.sample()
        else:
            selected_action = self._dqn_net(
                torch.FloatTensor(state).to(self._device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        self._transition = [state, selected_action]
        return selected_action

    def step(self, action: int):
        """Given an action, return next state. Save the transition into ReplayBuffer"""

        next_state, reward, done, _ = self._env.step(action)
        self._transition += [next_state, reward, done]
        self._buffer.save(*self._transition)

        return next_state, reward, done

    def computeLoss(self, samples: dict):
        """compute and return the Loss for training"""
        states = torch.FloatTensor(samples["states"]).to(self._device)
        next_states = torch.FloatTensor(
            samples["next_states"]).to(self._device)
        actions = torch.LongTensor(
            samples["actions"].reshape(-1, 1)).to(self._device)
        rewards = torch.FloatTensor(
            samples["rewards"]).reshape(-1, 1).to(self._device)
        done = torch.FloatTensor(
            samples["done"]).reshape(-1, 1).to(self._device)

        curr_q_values = torch.gather(self._dqn_net(states), 1, actions)
        next_q_values = self._target_dqn_net(next_states).max(dim=1, keepdim=True)[
            0].detach()  # the max() return the max value and corresponding index

        # if done, the target is R. if not done, it is R + \gamma * Q
        mask = 1 - done
        target = (rewards + self._gamma *
                  next_q_values * mask).to(self._device)

        loss = self._loss_func(curr_q_values, target)
        return loss

    def update(self):
        """sample the batch from ReplayBuffer, update the DQN network paras and return the loss value"""
        samples = self._buffer.sample_batch()
        loss = self.computeLoss(samples)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def train(self, num_episode: int):
        state = self._env.reset()
        update_count = 0
        score = 0

        with trange(num_episode) as pbar:
            for i in pbar:

                action = self.selectAction(state)
                next_state, reward, done = self.step(action)

                score += reward
                state = next_state

                if done:
                    state = self._env.reset()
                    score = 0

                if len(self._buffer) >= self._buffer_size:

                    loss = self.update()
                    update_count += 1

                    curr_eps = self._eps - \
                        (self._max_epsilon - self._min_epsilon) * \
                        self._epsilon_decay
                    self._eps = max(self._min_epsilon, curr_eps)

                    if update_count % self._target_update_period == 0:
                        self._target_dqn_net.load_state_dict(
                            self._dqn_net.state_dict())

        self._env.close()


if __name__ == "__main__":
    env = "CartPole-v0"
    env = gym.make(env)
    config = Configurator(env)
    config.max_train_step = 40000
    config.rollout_step = 2
    config.max_buffer_size = 3000
    agent = DQN(config)
    agent.train()
    # EPISODES = 10000
    # BUFFER_SIZE = 1000
    # BATCH_SIZE = 128
    # TARGET_UPDATE_PERIOD = 100
    # EPSILON_DECAY = 1 / 2000

    # agent = DQNagent(
    # env,
    # BUFFER_SIZE,
    # BATCH_SIZE,
    # TARGET_UPDATE_PERIOD,
    # EPSILON_DECAY
    # )

    # agent.train(EPISODES)
