import os
import sys
sys.path.append(
        os.path.dirname(
        os.path.dirname(
        os.path.dirname(
        os.path.realpath(__file__)))))

from tinyRL.util import ReplayBuffer
from tinyRL.util import DQNnet
from tqdm import trange
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import gym


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
    EPISODES = 10000
    BUFFER_SIZE = 1000
    BATCH_SIZE = 128
    TARGET_UPDATE_PERIOD = 100
    EPSILON_DECAY = 1 / 2000

    agent = DQNagent(
        env,
        BUFFER_SIZE,
        BATCH_SIZE,
        TARGET_UPDATE_PERIOD,
        EPSILON_DECAY
    )

    agent.train(EPISODES)
