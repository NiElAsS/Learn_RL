from tinyRL.agents import DQN
from tinyRL.util import Configurator
import torch
import gym


class DoubleDQN(DQN):

    """Create a double DQN agent for training. Inherit from DQN"""

    def __init__(self, config):
        """Init the agent"""
        super().__init__(config)

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
                # compute target for training(double dqn)
                target_q = torch.gather(
                    self._critic_target(next_state),
                    1,
                    self._critic(next_state).argmax(dim=1, keepdim=True)
                )
                y = reward + mask * self._gamma * target_q

            # compute loss
            curr_q = torch.gather(self._critic(state), 1,
                                  action.to(dtype=torch.int64))
            loss = self._loss_func(curr_q, y)

            # update the network
            self.applyUpdate(self._critic_optimizer, loss)

            # update the target network
            self.softUpdate(self._critic_target, self._critic, self._tau)


if __name__ == "__main__":
    env = "CartPole-v0"
    env = gym.make(env)
    config = Configurator(env)
    config.max_train_step = 40000
    config.rollout_step = 2
    config.max_buffer_size = 3000
    agent = DQN(config)
    agent.train()
