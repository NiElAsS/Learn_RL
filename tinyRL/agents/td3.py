import gym
from tinyRL.agents import BaseAgent
from tinyRL.util import Configurator, ActionNormalizer
from tinyRL.util import CriticQ
import torch
from copy import deepcopy


class TD3(BaseAgent):

    """Create a TD3 agent for training and evaluation"""

    def __init__(self, config):
        """Init the agent

        :config: Configurator for setting up paras

        """
        super().__init__(config)

        # set up second Q function
        self._critic_2 = CriticQ(
            self._state_dim, self._action_dim).to(self._device)
        self._critic_target_2 = deepcopy(self._critic_2).to(self._device)

        # cat the critic parameters
        q1_q2_paras = (
            list(self._critic.parameters()) + list(self._critic_2.parameters())
        )
        # use one optimizer
        self._critic_optimizer = torch.optim.Adam(
            q1_q2_paras,
            lr=self._critic_lr
        )

        # set up td3 paras
        self._action_noise_std = getattr(
            config,
            "action_noise_std",
            0.1
        )
        self._target_policy_noise_std = getattr(
            config,
            "target_policy_noise_std",
            0.2
        )
        self._target_policy_noise_clip = getattr(
            config,
            "target_policy_noise_clip",
            0.5
        )
        self._delay_update_freq = getattr(
            config,
            "delay_update_freq",
            2
        )

    def selectAction(self, state: torch.Tensor) -> torch.Tensor:
        """select action with respect to state

        :state: a torch.Tensor of state
        :returns: a torch.Tensor of action

        """

        action_tensor = self._actor(state)
        noise = torch.distributions.Normal(0, self._action_noise_std)
        action_tensor += noise.sample()

        return action_tensor

    def update(self):
        """ update the network"""

        updates_times = int(
            1 + self._update_repeat_times * self._buffer._curr_size / self._batch_size
        )
        for u in range(updates_times):
            with torch.no_grad():
                state, action, next_state, reward, mask = (
                    self._buffer.sampleBatch()
                )
                n_sample = state.shape[0]

                # sample and clip the noise
                noise = torch.distributions.Normal(
                    0, self._target_policy_noise_std
                )
                c = self._target_policy_noise_clip
                clip_noise = noise.sample(
                    (n_sample, 1)  # type:ignore
                ).clamp(-c, c)

                # compute action_tilde
                next_action = self._actor_target(next_state)
                action_tilde = next_action + clip_noise.to(self._device)

                # find the min q_value
                q_value_1 = self._critic_target(next_state, action_tilde)
                q_value_2 = self._critic_target_2(next_state, action_tilde)
                q_value = torch.min(q_value_1, q_value_2)

                # compute critic target
                y = reward + q_value*self._gamma

            # compute critic loss
            curr_q_value_1 = self._critic(state, action)
            curr_q_value_2 = self._critic_2(state, action)
            loss_1 = self._loss_func(curr_q_value_1, y)
            loss_2 = self._loss_func(curr_q_value_2, y)
            critic_loss = loss_1 + loss_2

            # apply critic update
            self.applyUpdate(self._critic_optimizer, critic_loss)

            # delay update
            if u % self._delay_update_freq == 0:
                # update actor
                actor_loss = -self._critic(state, self._actor(state)).mean()
                self.applyUpdate(self._actor_optimizer, actor_loss)

                # update target network
                self.softUpdate(self._critic_target, self._critic, self._tau)
                self.softUpdate(self._critic_target_2,
                                self._critic_2, self._tau)
                self.softUpdate(self._actor_target, self._actor, self._tau)

    def train(self):
        """Train the agent"""

        # init
        self._scores = []
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
    agent = TD3(config)
    agent.train()
