from gym.spaces.discrete import Discrete


class Configurator():

    """Configurator for training agents, setting the parameters"""

    def __init__(self, env):
        """Init all parameters """

        self.env = env
        self.discrete_action = isinstance(
            env.action_space, Discrete
        )
        self.state_dim = env.observation_space.shape[0]
        if self.discrete_action:
            self.action_dim = 1
        else:
            self.action_dim = env.action_space.shape[0]

        """for training"""
        self.gamma = 0.99  # discount factor
        self.max_train_step = 5e4  # maximum training step
        self.tau_soft_update = 1e-3  # target network soft update factor
        self.rollout_step = 200  # maximum steps of each exploration
        self.update_repeat_times = 1  # number of going through all samples
        self.actor_learning_rate = 1e-4  # actor_learning_rate
        self.critic_learning_rate = 1e-3  # critic_learning_rate

        """for epsilon-greedy"""
        self.max_epsilon_greedy_rate = 1.0
        self.min_epsilon_greedy_rate = 0.1
        self.epsilon_greedy_rate_decay = 10000

        """for GAE"""
        self.gae_lambda = 0.02

        """for PPO clipped"""
        self.ppo_clipped_eps = 0.25

        """for replay buffer"""
        self.buffer_batch_size = 128
        self.max_buffer_size = 1e5

        """for devices"""
        self.random_seed = 42
