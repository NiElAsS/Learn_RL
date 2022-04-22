
class Configurator():

    """Configurator for training agents, setting the parameters"""

    def __init__(self, env):
        """Init all parameters """

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]  # gym continuous env

        """for training"""
        self.gamma = 0.99
        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3
        self.tau_soft_update = 1e-3
        self.rollout_step = 2**10
        self.update_repeat_times = 1

        """for epsilon-greedy"""
        self.epsilon_greedy_rate = 0.25
        self.max_epsilon_greedy_rate = 1.0
        self.min_epsilon_greedy_rate = 0.05

        """for GAE"""
        self.gae_lambda = 0.02

        """for PPO clipped"""
        self.ppo_clipped_eps = 0.25

        """for replay buffer"""
        self.buffer_batch_size = 256
        self.max_buffer_size = 1e5

        """for devices"""
        self.random_seed = 42
