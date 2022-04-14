from tinyRL.agents import dqn


class DDPGagent(object):

    """Create a DDPG agent for training"""

    def __init__(self, env:gym.Env,
            buffer_size: int,
            batch_size: int,
            ou_noise_theta: float,
            ou_noise_sigma: float,
            gamma: float = 0.99,
            tau: float=5e-3,
            init_rand_steps: int = 1e4):
        """Init the paras for agent

        :env: TODO
        :buffer_size: TODO
        :batch_size: TODO
        :ou_noise_theta: TODO
        :ou_noise_sigma: TODO
        :gamma: TODO
        :tau: TODO
        :init_rand_steps: TODO

        """
        self._env = env
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._ou_noise_theta = ou_noise_theta
        self._ou_noise_sigma = ou_noise_sigma
        self._gamma = gamma
        self._tau = tau
        self._init_rand_steps = init_rand_steps
        

