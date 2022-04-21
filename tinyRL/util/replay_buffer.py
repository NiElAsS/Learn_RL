import numpy as np


class Buffer():

    """This is the buffer for saving samples during training"""

    def __init__(self, batch_size: int):
        """Init the Buffer

        :batch_size: The number of sample in a batch

        """

        self._batch_size = batch_size

        # count the current number of samples
        self._curr_size = 0

        # data
        self._state = []
        self._action = []
        self._reward = []
        self._next_state = []
        self._mask = []
        self._value = []
        self._log_prob = []

    def __len__(self):
        """return the current size of Buffer"""
        return self._curr_size

    def clear(self):
        """clear all data list"""
        self._curr_size = 0

        self._state.clear()
        self._action.clear()
        self._reward.clear()
        self._next_state.clear()
        self._mask.clear()
        self._value.clear()
        self._log_prob.clear()

    def sample_batch(self):
        """sample a batch of data

        :returns: a dict with (data_keyword, np.ndarray) pair

        """
        raise NotImplementedError

    def save(self, arg1):
        """Save the data into Buffer"""
        raise NotImplementedError


class BufferPPO(Buffer):

    """The data Buffer for PPOagent"""

    def __init__(self, batch_size: int):
        """Init the Buffer

        :max_buffer_size: The maximum number of sample in the Buffer
        :batch_size: The number of sample in a batch

        """
        super().__init__(batch_size)

    def save(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            value: np.ndarray,
            mask: np.ndarray,
            log_prob: np.ndarray
    ):
        """save the data

        :state: np.ndarray,
        :action: np.ndarray,
        :reward: np.ndarray,
        :value: np.ndarray,
        :mask: np.ndarray,
        :log_prob: np.ndarray

        """
        self._state.append(state)
        self._action.append(action)
        self._reward.append(reward)
        self._value.append(value)
        self._mask.append(mask)
        self._log_prob.append(log_prob)


class ReplayBuffer():

    """This is a normal replay buffer for training DQN.
    Implementing using numpy
    """

    def __init__(self, state_size: int, buffer_size: int, batch_size: int = 32):
        """Initialize the buffer, setting up the paras

        :state_size: The dim of the states
        :buffer_size: The maximum size of the buffer
        :batch_size: The size of each sample batch

        """

        self._state_size = state_size
        self._buffer_size = buffer_size
        self._max_buffer_size = buffer_size
        self._batch_size = batch_size

        """
        each np array stores one kind of data
        """
        self._state_buffer = np.zeros(
            (self._buffer_size, self._state_size), dtype=np.float32)
        self._next_state_buffer = np.zeros(
            (self._buffer_size, self._state_size), dtype=np.float32)
        self._actions_buffer = np.zeros((self._buffer_size,), dtype=np.float32)
        self._rewards_buffer = np.zeros((self._buffer_size,), dtype=np.float32)
        self._done_buffer = np.zeros((self._buffer_size,), dtype=np.float32)

        self._curr_size = 0
        self._ptr = 0

    def save(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, done: bool):
        """Saving the data(samples) into the buffer"""
        self._state_buffer[self._ptr] = state
        self._next_state_buffer[self._ptr] = next_state
        self._actions_buffer[self._ptr] = action
        self._rewards_buffer[self._ptr] = reward
        self._done_buffer[self._ptr] = done

        self._curr_size = min(self._curr_size + 1, self._max_buffer_size)
        self._ptr = (self._ptr + 1) % self._max_buffer_size

    def sample_batch(self):
        """sample a batch from current buffer

        :returns: a dictionary with samples

        """
        i = np.random.choice(
            self._curr_size, size=self._batch_size, replace=False)
        return dict(states=self._state_buffer[i],
                    next_states=self._next_state_buffer[i],
                    actions=self._actions_buffer[i],
                    rewards=self._rewards_buffer[i],
                    done=self._done_buffer[i])

    def __len__(self):
        """return the current size of buffer"""
        return self._curr_size
