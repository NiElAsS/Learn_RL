import torch


class ReplayBuffer():

    """ReplayBuffer for training off-policy agent"""

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            batch_size: int,
            max_buffer_size: int
    ):
        """Init the Buffer

        :state_dim: Dimenstion of states
        :action_dim: Dimenstion of actions
        :batch_size: Size of each batch
        :max_buffer_size: The maximum number of samples in buffer

        """

        # init the paras
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._batch_size = int(batch_size)
        self._max_buffer_size = int(max_buffer_size)

        # init the gpu/cpu
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # init the buffer
        self._state_buffer = torch.zeros(
            (self._max_buffer_size, self._state_dim),
            dtype=torch.float32,
            device=self._device
        )  # use state_buffer both for state and next_state(see sampleBatch)
        self._action_buffer = torch.zeros(
            (self._max_buffer_size, self._action_dim),
            dtype=torch.float32,
            device=self._device
        )
        self._reward_buffer = torch.zeros(
            (self._max_buffer_size, 1),
            dtype=torch.float32,
            device=self._device
        )
        self._mask_buffer = torch.zeros(
            (self._max_buffer_size, 1),
            dtype=torch.float32,
            device=self._device
        )

        # record the curr state of buffer
        self._curr_size = 0
        self._ptr = 0

    def saveTrajectory(self, trajectory: list):
        """Given the trajectory(list), save the each transition into buffer"""

        """
        convert
        [[state, action, next_state, reward, mask], [...],...]
        to
        [[state1, ...], [action1, ...], [next_state1, ...],...]
        """
        traj = list(map(list, zip(*trajectory)))

        """convert list to torch.Tensor"""
        state, action, reward, mask = [
            torch.cat(i, dim=0) for i in traj
        ]
        traj_size = reward.shape[0]
        require_size = self._ptr + traj_size

        """save the traj"""
        if require_size > self._max_buffer_size:
            """space not enough, separate traj into 2 parts"""
            self._state_buffer[self._ptr:self._max_buffer_size] = (
                state[:self._max_buffer_size - self._ptr]
            )
            self._action_buffer[self._ptr:self._max_buffer_size] = (
                action[:self._max_buffer_size - self._ptr]
            )
            self._reward_buffer[self._ptr:self._max_buffer_size] = (
                reward[:self._max_buffer_size - self._ptr]
            )
            self._mask_buffer[self._ptr:self._max_buffer_size] = (
                mask[:self._max_buffer_size - self._ptr]
            )

            """2nd part"""
            require_size = require_size - self._max_buffer_size
            self._state_buffer[0:require_size] = state[-require_size:]
            self._action_buffer[0:require_size] = action[-require_size:]
            self._reward_buffer[0:require_size] = reward[-require_size:]
            self._mask_buffer[0:require_size] = mask[-require_size:]
        else:
            """we have enough space"""
            self._state_buffer[self._ptr:require_size] = state
            self._action_buffer[self._ptr:require_size] = action
            self._reward_buffer[self._ptr:require_size] = reward
            self._mask_buffer[self._ptr:require_size] = mask

        self._ptr = require_size
        self._curr_size = min(
            self._curr_size + traj_size, self._max_buffer_size
        )

    def saveTransition(self, transition: list):
        """Saving one transition into buffer

        :transiton: A list [state, action, next_state, reward, mask]

        """
        self._state_buffer[self._ptr] = transition[0]
        self._action_buffer[self._ptr] = transition[1]
        self._reward_buffer[self._ptr] = transition[3]
        self._mask_buffer[self._ptr] = transition[4]

        self._curr_size = min(self._curr_size + 1, self._max_buffer_size)
        self._ptr = (self._ptr + 1) % self._max_buffer_size

    def sampleBatch(self) -> tuple:
        """sample a batch of data with size self._batch_size

        :returns: Tuple[states, actions, next_states, rewards, masks]

        """

        indices = torch.randint(
            self._curr_size - 1,  # -1 bcs (indices + 1) is needed below
            size=(self._batch_size,),
            device=self._device
        )
        return (
            self._state_buffer[indices],
            self._action_buffer[indices],
            self._state_buffer[indices + 1],
            self._reward_buffer[indices],
            self._mask_buffer[indices]
        )
