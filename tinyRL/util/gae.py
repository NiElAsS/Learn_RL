from collections import deque
import torch


def gae(
        last_next_value: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        lambd: float
) -> list:
    """Compute the GAE

    :arg1: TODO
    :returns: TODO

    """
    values = torch.cat((values, last_next_value))
    gae = 0
    res = deque()

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lambd * masks[i] * gae
        res.appendleft(gae + values[i])

    return list(res)
