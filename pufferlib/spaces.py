import numpy as np
import gymnasium

Box = gymnasium.spaces.Box
Dict = gymnasium.spaces.Dict
MultiBinary = gymnasium.spaces.MultiBinary
Tuple = gymnasium.spaces.Tuple


def joint_space(space, n):
    if not isinstance(space, Box):
        raise ValueError(f"Unsupported space: {space}")
    low = np.repeat(space.low[None], n, axis=0)
    high = np.repeat(space.high[None], n, axis=0)
    return gymnasium.spaces.Box(low=low, high=high, shape=(n, *space.shape), dtype=space.dtype)
