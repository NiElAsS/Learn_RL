from collections import deque


def gae(
        next_value: list,
        rewards: list,
        masks: list,
        values: list,
        gamma: float,
        lambd: float
) -> list:
    """Compute the GAE

    :arg1: TODO
    :returns: TODO

    """
    values = values + [next_value]
    gae = 0
    res = deque()

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lambd * masks[i] * gae
        res.appendleft(gae + values[i])

    return list(res)
