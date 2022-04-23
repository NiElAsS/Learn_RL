import gym
import matplotlib.pyplot as plt
import torch
from tinyRL.util import ActionNormalizer
from tinyRL.util.configurator import Configurator
from tinyRL.agents.base import BaseAgent


class DDPG(BaseAgent):

    """DDPG agent for training and evaluating"""

    def __init__(self, config):
        """Init the agent

        :config: Configurator for setting up the agent

        """
        super().__init__(config)

        self._normal_noise_std = getattr(config, "normal_noise_std", 0.05)
        self._noise = getattr(config, "noise", torch.distributions.Normal)

    def selectAction(self, state: torch.Tensor):
        """Get the action with Normal noise"""
        action = self._actor(state)

        # add noise for exploration
        action += self._noise(0, self._normal_noise_std).sample()
        return action.clamp(-1.0, 1.0)

    def update(self):
        """Update the agent network"""

        # go throught samples 'self._update_repeat_times'
        updates_times = int(
            1 + self._update_repeat_times * self._buffer._curr_size / self._batch_size
        )

        for _ in range(updates_times):
            with torch.no_grad():
                state, action, next_state, reward, mask = (
                    self._buffer.sampleBatch()
                )
                # calculate target for critic
                next_action = self._actor_target(next_state)
                next_q = self._critic_target(next_state, next_action)
                y = reward + mask * self._gamma * next_q

            # compute loss for critic
            q = self._critic(state, action)
            critic_loss = self.loss_func(y, q)
            self.applyUpdate(self._critic_optimizer, critic_loss)

            # compute loss for actor
            # maximize J(\theta) == minimize -J
            actor_loss = -self._critic(state, self._actor(state)).mean()
            self.applyUpdate(self._actor_optimizer, actor_loss)

            # update target networks
            self.softUpdate(self._critic_target, self._critic, self._tau)
            self.softUpdate(self._actor_target, self._actor, self._tau)

            # save the loss values
            self._actor_loss.append(actor_loss.item())
            self._critic_loss.append(critic_loss.item())

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
            if print_count % 10 == 0:
                print(
                    f"Current step: {self._curr_step} Last 100 exploration mean score: {torch.tensor(self._scores[-100:]).mean()}"
                )

        self._env.close()


if __name__ == '__main__':
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)

    config = Configurator(env)
    agent = DDPG(config)
    agent.train()
