from typing import Tuple
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from tinyRL.agents import BaseAgent
from tinyRL.util.net import ActorSto, CriticV
from tinyRL.util.norm import ActionNormalizer
from tinyRL.util.configurator import Configurator

from tinyRL.util.gae import gae
from tinyRL.util.replay_buffer import BufferPPO


class PPO(BaseAgent):

    """Create a PPO agent for training and evaluation"""

    def __init__(self, config):
        """Init the agent

        :config: Configurator for init all paras

        """
        super().__init__(config)

        # init buffer, use list()
        self._buffer = list()

        # init the net
        self._actor = ActorSto(
            self._state_dim, self._action_dim
        ).to(self._device)

        self._critic = CriticV(self._state_dim).to(self._device)

        # init the optim
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._actor_lr
        )
        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=self._critic_lr
        )

        self._clip_eps = getattr(config, "ppo_clip_eps", 0.25)

    def selectAction(self, state: torch.Tensor) -> tuple:
        """select action with respect to state

        :state: a torch.Tensor of state
        :returns: a Tuple[torch.Tensor, torch.Distribution]

        """

        action_tensor, dist = self._actor(state)

        return action_tensor, dist

    def trajToBuffer(self, traj):
        """save the trajectory into buffer(List)"""
        self._buffer = list(map(list, zip(*traj)))
        state, action, reward, mask, value, log_prob = [
            torch.cat(i, dim=0).to(self._device) for i in self._buffer
        ]
        self._buffer = [state, action, reward, mask, value, log_prob]

    def getReturn(self) -> torch.Tensor:
        """compute the return for each sample"""
        reward = self._buffer[2]
        mask = self._buffer[3]
        n_sample = reward.shape[0]  # type:ignore
        ret = torch.zeros(
            (n_sample, 1),
            dtype=torch.float32,
            device=self._device
        )
        tmp = 0
        for i in reversed(range(0, n_sample)):
            ret[i] = reward[i] + mask[i] * tmp * self._gamma
            tmp = ret[i]

        return ret

    def exploreEnv(self):
        """explore the env, save the trajectory to buffer"""

        step = 0
        score = 0
        traj = []
        state = self._env.reset()
        done = False

        while step < self._rollout_step or not done:

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self._device
            )

            action_tensor, dist = self.selectAction(
                state_tensor
            )
            value = self._critic(state_tensor)
            log_prob = dist.log_prob(action_tensor)
            action = action_tensor.detach().cpu()

            next_state, reward, done = self.step(action.numpy())

            transition = [state_tensor,
                          action_tensor,
                          reward,
                          1-done,
                          value,
                          log_prob]
            traj.append(self.transToTensor(transition))

            # update the vars
            state = next_state
            score += reward
            step += 1
            if done:
                self._scores.append(score)

                score = 0
                state = self._env.reset()

        self.trajToBuffer(traj)
        self._curr_step += step

    def update(self):
        """update the agent"""
        with torch.no_grad():
            # get samples
            state = self._buffer[0]
            action = self._buffer[1]
            # reward = self._buffer[2]
            # mask = self._buffer[3]
            value = self._buffer[4]
            log_prob = self._buffer[5]

            # get returns
            ret = self.getReturn()

            # get advantages
            adv = ret - value
            # adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        n_sample = value.shape[0]  # type:ignore
        assert n_sample >= self._batch_size

        updates_times = int(
            1 + self._update_repeat_times * n_sample / self._batch_size
        )

        for _ in range(updates_times):
            # sample the batch
            indices = torch.randint(
                n_sample,
                size=(self._batch_size,),
                requires_grad=False,
                device=self._device
            )
            batch_state = state[indices]
            batch_action = action[indices]
            batch_return = ret[indices]
            batch_adv = adv[indices].detach()
            batch_log_prob = log_prob[indices]

            # compute r = pi / pi_old
            _, curr_dist = self._actor(batch_state)
            curr_log_prob = curr_dist.log_prob(batch_action)
            r = (curr_log_prob - batch_log_prob.detach()).exp()

            # compute surrogate objective
            surr = r * batch_adv
            clip_surr = torch.clamp(r, 1.0-self._clip_eps, 1.0+self._clip_eps)

            # compute entropy
            entropy = curr_dist.entropy().mean()

            # compute actor_loss
            actor_loss = (
                torch.min(surr, clip_surr).mean()
                - entropy * self._entropy_weight
            )

            # compute critic_loss
            curr_value = self._critic(batch_state)
            critic_loss = self._loss_func(curr_value, batch_return)

            # applyUpdate
            self.applyUpdate(self._actor_optimizer, -actor_loss)
            self.applyUpdate(self._critic_optimizer, critic_loss)

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
            print_count += 1
            if print_count % 2 == 0:
                print(
                    f"Current step: {self._curr_step} Last 100 exploration mean score: {torch.tensor(self._scores[-100:]).mean()}"
                )

        self._env.close()


if __name__ == '__main__':

    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)

    config = Configurator(env)
    config.max_train_step = 100000
    config.rollout_step = 2000
    config.update_repeat_times = 64
    config.buffer_batch_size = 256
    config.gamma = 0.9
    agent = PPO(config)
    agent.train()

    # act_dim = env.action_space.shape[0]
    # obs_dim = env.observation_space.shape[0]
    # actor = ActorSto(obs_dim, act_dim)
    # critic = CriticV(obs_dim)
    # buffer = BufferPPO()
    # agent = PPOagent(
    # env,
    # actor,
    # critic,
    # buffer
    # )

    # n_step = 100000

    # agent.train(n_step)
