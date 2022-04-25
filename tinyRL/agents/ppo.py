import gym
import torch
import torch.optim as optim

from tinyRL.agents import BaseAgent
from tinyRL.util.net import ActorSto, CriticV
from tinyRL.util.norm import ActionNormalizer
from tinyRL.util.configurator import Configurator


class PPO(BaseAgent):

    """Create a PPO agent for training and evaluation"""

    def __init__(self, config):
        """Init the agent

        :config: Configurator for init all paras

        """
        super().__init__(config)

        # init buffer, use list()
        self._buffer = list()

        # init the net
        self._actor = ActorSto(
            self._state_dim, self._action_dim
        ).to(self._device)

        self._critic = CriticV(self._state_dim).to(self._device)

        # init the optim
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._actor_lr
        )
        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=self._critic_lr
        )

        self._clip_eps = getattr(config, "ppo_clip_eps", 0.25)
        self._if_gae = getattr(config, "if_gae", True)
        self._gae_lambda = getattr(config, "gae_lambda", 0.95)
        self._ret_estimator = self.getReturnAndGAE if self._if_gae else self.getDiscountReturn

    def selectAction(self, state: torch.Tensor) -> tuple:
        """select action with respect to state

        :state: a torch.Tensor of state
        :returns: a Tuple[torch.Tensor, torch.Distribution]

        """

        action_tensor, dist = self._actor(state)

        return action_tensor, dist

    def trajToBuffer(self, traj):
        """save the trajectory into buffer(List)"""
        self._buffer = list(map(list, zip(*traj)))
        state, action, reward, mask, value, log_prob = [
            torch.cat(i, dim=0).to(self._device) for i in self._buffer
        ]
        self._buffer = [state, action, reward, mask, value, log_prob]

    def getDiscountReturn(self) -> tuple:
        """compute the return advantages for each sample"""
        reward = self._buffer[2]
        mask = self._buffer[3]
        value = self._buffer[4]
        n_sample = reward.shape[0]  # type:ignore
        ret = torch.zeros(
            (n_sample, 1),
            dtype=torch.float32,
            device=self._device
        )
        tmp = 0
        for i in reversed(range(0, n_sample)):
            ret[i] = reward[i] + mask[i] * tmp * self._gamma
            tmp = ret[i]

        return ret, ret-value

    def getReturnAndGAE(self) -> tuple:
        """compute the returns and generalized advantage estimator(GAE)"""
        reward = self._buffer[2]
        mask = self._buffer[3]
        value = self._buffer[4]
        n_sample = reward.shape[0]  # type:ignore
        ret = torch.zeros(
            (n_sample, 1),
            dtype=torch.float32,
            device=self._device
        )
        adv = torch.zeros(
            (n_sample, 1),
            dtype=torch.float32,
            device=self._device
        )
        tmp_ret = 0
        tmp_adv = 0
        for i in reversed(range(0, n_sample)):
            ret[i] = reward[i] + mask[i] * tmp_ret * self._gamma
            tmp_ret = ret[i]

            adv[i] = reward[i] + mask[i] * tmp_adv * self._gamma - value[i]
            tmp_adv = value[i] + adv[i] * self._gae_lambda

        return ret, adv

    def exploreEnv(self):
        """explore the env, save the trajectory to buffer"""

        step = 0
        score = 0
        traj = []
        state = self._env.reset()
        done = False

        while step < self._rollout_step or not done:

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self._device
            )

            action_tensor, dist = self.selectAction(
                state_tensor
            )
            value = self._critic(state_tensor)
            log_prob = dist.log_prob(action_tensor)
            action = action_tensor.detach().cpu()

            next_state, reward, done = self.step(action.numpy())

            transition = [state_tensor,
                          action_tensor,
                          reward,
                          1-done,
                          value,
                          log_prob]
            traj.append(self.transToTensor(transition))

            # update the vars
            state = next_state
            score += reward
            step += 1
            if done:
                self._scores.append(score)

                score = 0
                state = self._env.reset()

        self.trajToBuffer(traj)
        self._curr_step += step

    def update(self):
        """update the agent"""
        with torch.no_grad():
            # get samples
            state = self._buffer[0]
            action = self._buffer[1]
            # reward = self._buffer[2]
            # mask = self._buffer[3]
            value = self._buffer[4]
            log_prob = self._buffer[5]

            # get returns and advantages
            ret, adv = self._ret_estimator()
            # adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        n_sample = value.shape[0]  # type:ignore
        assert n_sample >= self._batch_size

        updates_times = int(
            1 + self._update_repeat_times * n_sample / self._batch_size
        )

        for _ in range(updates_times):
            # sample the batch
            indices = torch.randint(
                n_sample,
                size=(self._batch_size,),
                requires_grad=False,
                device=self._device
            )
            batch_state = state[indices]
            batch_action = action[indices]
            batch_return = ret[indices]
            batch_adv = adv[indices].detach()
            batch_log_prob = log_prob[indices]

            # compute r = pi / pi_old
            _, curr_dist = self._actor(batch_state)
            curr_log_prob = curr_dist.log_prob(batch_action)
            r = (curr_log_prob - batch_log_prob.detach()).exp()

            # compute surrogate objective
            surr = r * batch_adv
            clip_surr = torch.clamp(r, 1.0-self._clip_eps, 1.0+self._clip_eps)

            # compute entropy
            entropy = curr_dist.entropy().mean()

            # compute actor_loss
            actor_loss = (
                torch.min(surr, clip_surr).mean()
                + entropy * self._entropy_weight
            )

            # compute critic_loss
            curr_value = self._critic(batch_state)
            critic_loss = self._loss_func(curr_value, batch_return)

            # applyUpdate
            self.applyUpdate(self._actor_optimizer, -actor_loss)
            self.applyUpdate(self._critic_optimizer, critic_loss)

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
            if print_count % 2 == 0:
                print(
                    f"Current step: {self._curr_step} Last 100 exploration mean score: {torch.tensor(self._scores[-100:]).mean()}"
                )

        self._env.close()


if __name__ == '__main__':

    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)

    config = Configurator(env)
    config.max_train_step = 100000
    config.rollout_step = 2000
    config.update_repeat_times = 16
    config.buffer_batch_size = 128
    config.gamma = 0.9
    config.gae_lambda = 0.8
    config.if_gae = True

    agent = PPO(config)
    agent.train()
