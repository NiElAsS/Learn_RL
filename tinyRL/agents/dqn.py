from tinyRL.util import DQNnet
from tinyRL.util.configurator import Configurator
from tinyRL.agents import BaseAgent
from copy import deepcopy
import torch.optim as optim
import torch
import gym


class DQN(BaseAgent):

    """Create a DQN agent for training and evaluation"""

    def __init__(self, config):
        """Init the agent

        :config: Configurator for initialization

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


if __name__ == "__main__":
    env = "CartPole-v0"
    env = gym.make(env)
    config = Configurator(env)
    config.max_train_step = 40000
    config.rollout_step = 2
    config.max_buffer_size = 3000
    agent = DQN(config)
    agent.train()
