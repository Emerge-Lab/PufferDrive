"""Kinematic feature computation utilities for WOSAC metrics."""

import numpy as np


def central_diff(t: np.ndarray, pad_value: float) -> np.ndarray:
    """Computes the central difference along the last axis.

    This function is used to compute 1st order derivatives (speeds) when called
    once. Calling this function twice is used to compute 2nd order derivatives
    (accelerations) instead.
    This function returns the central difference as
    df(x)/dx = [f(x+h)-f(x-h)] / 2h.

    Args:
        t: A float array of shape [..., steps].
        pad_value: To maintain the original tensor shape, this value is prepended
            once and appended once to the difference.

    Returns:
        An array of shape [..., steps] containing the central differences,
        appropriately prepended and appended with `pad_value` to maintain the
        original shape.
    """
    pad_shape = (*t.shape[:-1], 1)
    pad_array = np.full(pad_shape, pad_value)
    diff_t = (t[..., 2:] - t[..., :-2]) / 2
    return np.concatenate([pad_array, diff_t, pad_array], axis=-1)


def central_logical_and(t: np.ndarray, pad_value: bool) -> np.ndarray:
    """Computes the central `logical_and` along the last axis.

    This function is used to compute the validity tensor for 1st and 2nd order
    derivatives using central difference, where element [i] is valid only if
    both elements [i-1] and [i+1] are valid.

    Args:
        t: A bool array of shape [..., steps].
        pad_value: To maintain the original tensor shape, this value is prepended
            once and appended once to the difference.

    Returns:
        An array of shape [..., steps] containing the central `logical_and`,
        appropriately prepended and appended with `pad_value` to maintain the
        original shape.
    """
    pad_shape = (*t.shape[:-1], 1)
    pad_array = np.full(pad_shape, pad_value)
    diff_t = np.logical_and(t[..., 2:], t[..., :-2])
    return np.concatenate([pad_array, diff_t, pad_array], axis=-1)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wraps angles in the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
