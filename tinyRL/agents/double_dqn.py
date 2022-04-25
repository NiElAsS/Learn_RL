from tinyRL.agents import DQN, DQNagent
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


class DoubleDQNagent(DQNagent):

    """Create a double DQN agent for training. Inherit from DQNagent"""

    def __init__(self,
                 env: gym.Env,
                 buffer_size: int,
                 batch_size: int,
                 target_update_period: int,
                 epsilon_decay: float,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 gamma: float = 0.99):

        super().__init__(env, buffer_size, batch_size, target_update_period,
                         epsilon_decay, max_epsilon, min_epsilon, gamma)

    def computeLoss(self, samples: dict):
        """override the computerLoss part, evaluate loss in double dqn way"""

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
        # double dqn
        next_q_values = torch.gather(
            self._target_dqn_net(next_states),
            1,
            self._dqn_net(next_states).argmax(dim=1, keepdim=True)
        ).detach()
        # dqn
        # next_q_values = self._target_dqn_net(next_states).max(dim=1, keepdim=True)[0].detach()

        # if done, the target is R. if not done, it is R + \gamma * Q
        mask = 1 - done
        target = (rewards + self._gamma *
                  next_q_values * mask).to(self._device)

        loss = self._loss_func(curr_q_values, target)
        return loss


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

    # agent = DoubleDQNagent(
    # env,
    # BUFFER_SIZE,
    # BATCH_SIZE,
    # TARGET_UPDATE_PERIOD,
    # EPSILON_DECAY
    # )

    # agent.train(EPISODES)
