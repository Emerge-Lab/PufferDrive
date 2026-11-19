import numpy as np


class Box:
    def __init__(self, low, high, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.low = np.full(self.shape, low, dtype=self.dtype)
        self.high = np.full(self.shape, high, dtype=self.dtype)

    def contains(self, x):
        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def __eq__(self, other):
        return (
            isinstance(other, Box)
            and self.shape == other.shape
            and self.dtype == other.dtype
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    def __repr__(self):
        return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"


def joint_space(space, n):
    if not isinstance(space, Box):
        raise ValueError(f"Unsupported space: {space}")
    low = np.repeat(space.low[None], n, axis=0)
    high = np.repeat(space.high[None], n, axis=0)
    return Box(low=low, high=high, shape=(n, *space.shape), dtype=space.dtype)
