import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F
from tinyRL.util.noise import OUNoise
from tinyRL.util.net import Critic, Actor
from tinyRL.util import ReplayBuffer, ActionNormalizer


class DDPGagent(object):

    """Create a DDPG agent for training"""

    def __init__(self, env,
                 buffer_size: int,
                 batch_size: int,
                 ou_noise_theta: float,
                 ou_noise_sigma: float,
                 gamma: float = 0.99,
                 tau: float = 5e-3
                 ):
        """Init the paras for agent

        :env: The Gym environment
        :buffer_size: The maximum size of ReplayBuffer
        :batch_size: The number of samples of each batch
        :ou_noise_theta: Parameter for OUNoise
        :ou_noise_sigma: Parameter for OUNoise
        :gamma: Discount factor
        :tau: Parameter for delayed update of parameters of target net

        """
        self._env = env
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._ou_noise_theta = ou_noise_theta
        self._ou_noise_sigma = ou_noise_sigma
        self._gamma = gamma
        self._tau = tau
        self._step_count = 0

        self._observation_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]

        # ReplayBuffer
        self._buffer = ReplayBuffer(
            self._observation_dim, self._buffer_size, self._batch_size)

        # define the noise
        # use OU-noise
        self._noise = OUNoise(
            self._action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma
        )

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # (input, output)
        # pi(a|s)
        self._actor = Actor(self._observation_dim,
                            self._action_dim).to(self._device)
        self._actor_target = deepcopy(self._actor).to(self._device)

        # (state dim, action dim)
        # Q(s,a)
        self._critic = Critic(self._observation_dim,
                              self._action_dim).to(self._device)
        self._critic_target = deepcopy(self._critic).to(
            self._device).to(self._device)

        # optimizer
        self._actor_opt = optim.Adam(self._actor.parameters(), lr=1e-4)
        self._critic_opt = optim.Adam(self._critic.parameters(), lr=1e-3)

        # store transition
        self._transition = list()

        # data recorder
        self._scores = list()
        self._actor_losses = list()
        self._critic_losses = list()

    def selectAction(self, state: np.ndarray) -> np.ndarray:
        """Select a batch of actions given states

        :state: state
        :returns: Batch of actions given states

        """

        action = self._actor(
            torch.FloatTensor(state).to(self._device)
        ).detach().cpu().numpy()

        noise = self._noise.sample()
        action = np.clip(action + noise, -1.0, 1.0)

        # save the transition info
        self._transition = [state, action]

        return action

    def step(self, action: np.ndarray) -> tuple:
        """Take the given action and return the data

        :action: the action need to be taken
        :returns: tuple(next_state, reward, done)

        """
        next_state, reward, done, _ = self._env.step(action)

        # save the transition info
        self._transition += [next_state, reward, done]
        self._buffer.save(*self._transition)

        return next_state, reward, done

    @staticmethod
    def applyUpdate(optimizer: torch.optim.Optimizer, loss_func):
        """A help function to calculate the gradient and step.

        :optimizer: torch optimizer
        :loss_func: corresponding loss function

        """
        optimizer.zero_grad()
        loss_func.backward()
        optimizer.step()

    def update(self) -> tuple:
        """Update the agent given data"""

        # sample the data
        samples = self._buffer.sample_batch()
        states = torch.FloatTensor(samples['states']).to(self._device)
        next_states = torch.FloatTensor(
            samples['next_states']).to(self._device)
        actions = torch.FloatTensor(
            samples['actions']).reshape(-1, 1).to(self._device)
        rewards = torch.FloatTensor(
            samples['rewards']).reshape(-1, 1).to(self._device)
        dones = torch.FloatTensor(
            samples['done']).reshape(-1, 1).to(self._device)

        masks = 1 - dones
        next_actions = self._actor_target(next_states)
        next_q_values = self._critic_target(next_states, next_actions)
        target_values = rewards + self._gamma * next_q_values * masks

        # update critic
        current_values = self._critic(states, actions)
        critic_loss = F.mse_loss(current_values, target_values)

        self.applyUpdate(self._critic_opt, critic_loss)

        # update actor
        # we want to max J, so we can min the -J
        actor_loss = -self._critic(states, self._actor(states)).mean()
        self.applyUpdate(self._actor_opt, actor_loss)

        # update the target network
        self.softUpdate(self._critic_target, self._critic, self._tau)
        self.softUpdate(self._actor_target, self._actor, self._tau)

        return actor_loss, critic_loss

    @staticmethod
    def softUpdate(target_network, network, tau):
        """Applying soft(delayed) update for target network"""

        for target_param, param in zip(
                target_network.parameters(), network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)

    def train(self, n_step: int):
        """Train the agent

        :n_step: the number of training step

        """
        state = self._env.reset()
        self._scores = []
        self._actor_losses = []
        self._critic_losses = []
        score = 0

        for i in range(1, n_step + 1):

            action = self.selectAction(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
            self._step_count += 1

            if done:
                state = self._env.reset()
                self._scores.append(score)
                score = 0

            if(len(self._buffer) >= self._batch_size):
                critic_loss, actor_loss = self.update()
                self._critic_losses.append(critic_loss)
                self._actor_losses.append(actor_loss)

            if(i % 1000 == 0):
                print(
                    f"Current step: {i} Last 5000 Mean: {np.array(self._scores[-5000:]).mean()}")

        self._env.close()

    def plot(self):
        """plot the result(scores, critic losses and actor losses)"""

        plt.title("Scores")
        plt.plot(self._scores)
        plt.show()


if __name__ == '__main__':
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)

    n_step = 50000
    buffer_size = 100000
    batch_size = 256
    # ou_noise_theta = 0.15
    # ou_noise_sigma = 0.2
    ou_noise_theta = 1.0
    ou_noise_sigma = 0.1

    agent = DDPGagent(
        env,
        buffer_size,
        batch_size,
        ou_noise_theta,
        ou_noise_sigma
    )

    agent.train(n_step)

    agent.plot()
