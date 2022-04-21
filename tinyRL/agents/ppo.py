from typing import Tuple
import gym
import numpy as np
import torch
import torch.nn.functional as F

from tinyRL.agents import BaseAgent
from tinyRL.util.net import ActorSto, CriticV
from tinyRL.util.norm import ActionNormalizer
from tinyRL.util.gae import gae
from tinyRL.util.replay_buffer import BufferPPO


class PPOagent(BaseAgent):

    """Agent of PPO algorithm"""

    def __init__(
            self,
            env,
            actor,
            critic,
            buffer,
            gamma: float = 0.99,
            lambd: float = 0.01,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-3,
            epsilon: float = 0.25,
            T_step: int = 2**11,
            epoch: int = 2**6,
            entropy_weight: float = 1e-2
    ):
        """Init the PPOagent

        :env: gym.Env
        :actor: NN as actor
        :critic: NN as critic
        :replay_buffer: The replay buffer for saving and sampling transition
        :gamma: Discount factor
        :tau: Soft update factor
        :actor_learning_rate: Learning rate for actor
        :critic_learning_rate: Learning rate for critic
        :epsilon: clipping surrogate object
        :T_step: the number of rollout
        :epoch: the number of epoch updating the networks
        :entropy_weight: ratio of weighting entropy into loss func

        """
        super().__init__(env, actor, critic, buffer, gamma, 0.1, actor_lr, critic_lr)

        self._lambda = lambd
        self._epsilon = epsilon
        self._T_step = T_step
        self._epoch = epoch
        self._entropy_weight = entropy_weight

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """select action with respect to state

        :state: a numpy array of states
        :returns: a numpy array of actions

        """

        # reshape 1-dim state to 2-dim
        # for torch.cat()
        state_tensor = torch.FloatTensor(state).reshape(1, -1).to(self._device)

        action, dist = self._actor(state_tensor)
        action_tensor = action.reshape(-1, 1)
        value = self._critic(state_tensor).reshape(-1, 1)
        log_prob = dist.log_prob(action).reshape(-1, 1)

        # save the transition info
        # [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        # on self._device
        self._transition = [state_tensor, action_tensor, value, log_prob]

        return action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take the given action and return the data

        :action: the action need to be taken
        :returns: tuple(next_state, reward, done)

        """
        next_state, reward, done, _ = self._env.step(action)

        # save the transition info
        reward_tenosr = torch.FloatTensor(
            reward).reshape(-1, 1).to(self._device)
        mask_tensor = torch.tensor(1-done).reshape(-1, 1).to(self._device)
        next_state_tensor = torch.FloatTensor(
            next_state).reshape(1, -1).to(self._device)

        self._transition += [next_state_tensor, reward_tenosr, mask_tensor]
        self._buffer.save(*self._transition)

        return next_state, reward, done

    def update(self):
        """Update the network

        :returns: TODO

        """
        # Get data
        data = self._buffer.data()
        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        values = data['values'].detach()
        next_states = data['next_states']
        masks = data['masks']
        log_probs = data['log_prob'].detach()

        # for GAE
        last_next_state = next_states[-1]
        last_next_value = self._critic(last_next_state).reshape(1, -1)

        # compute gae
        returns = gae(
            last_next_value,
            rewards,
            masks,
            values,
            self._gamma,
            self._lambda
        )
        returns = torch.cat(returns).reshape(-1, 1).detach()
        advantages = returns - values

        # optimize objective function wrt \theta with K epoch and batch_size M \lq NT
        data_size = len(self._buffer)
        batch_size = self._buffer.batchSize()
        for _ in range(self._epoch):
            for _ in range(data_size // batch_size):
                index = np.random.choice(
                    data_size, size=batch_size, replace=False)
                state = states[index]
                action = actions[index]
                # value = values[index]
                log_prob = log_probs[index]
                ret = returns[index]
                adv = advantages[index]

                # r = pi / pi_old
                _, curr_dist = self._actor(state)
                curr_log_prob = curr_dist.log_prob(action)
                r = (curr_log_prob - log_prob).exp()

                # compute surrogate objective
                surr = r * adv
                clipped_surr = torch.clamp(
                    r, 1.0-self._epsilon, 1.0+self._epsilon
                ) * adv

                # add entropy
                entropy = curr_dist.entropy().mean()
                actor_loss = (
                    -torch.min(surr, clipped_surr).mean()
                    - entropy * self._entropy_weight
                )

                # compute critic loss
                value = self._critic(state)
                critic_loss = F.mse_loss(ret, value)

                # apply update
                self.applyUpdate(self._critic_optimizer, critic_loss)
                self.applyUpdate(self._actor_optimizer, actor_loss)

        self._buffer.clear()

    def train(self, n_step: int):
        """Train the agent

        :arg1: TODO
        :returns: TODO

        """
        while self._curr_step <= n_step:
            self.exploreEnv(self._T_step)

            self.update()

        self._env.close()


if __name__ == '__main__':

    env = gym.make("Pendulum-v1")
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    env = ActionNormalizer(env)
    actor = ActorSto(obs_dim, act_dim)
    critic = CriticV(obs_dim)
    buffer = BufferPPO()
    agent = PPOagent(
        env,
        actor,
        critic,
        buffer
    )

    n_step = 100000

    agent.train(n_step)
