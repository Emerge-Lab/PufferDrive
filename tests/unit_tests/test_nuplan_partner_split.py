"""Long nuPlan partners (buses) are shown to the policy as training-sized boxes along their heading."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402


def _partners():
    x = np.array([10.0, 50.0], np.float32)
    y = np.array([0.0, 5.0], np.float32)
    z = np.zeros(2, np.float32)
    h = np.array([0.0, math.pi / 2], np.float32)
    vx = np.array([3.0, 0.0], np.float32)
    vy = np.array([0.0, 1.0], np.float32)
    types = np.array([1, 1], np.int32)
    lengths = np.array([4.5, 12.7], np.float32)
    widths = np.array([1.9, 2.4], np.float32)
    return x, y, z, h, vx, vy, types, lengths, widths


def test_only_partners_over_the_limit_are_split_into_equal_pieces_along_their_heading():
    x, y, z, h, vx, vy, types, lengths, widths = nb.split_long_partners(*_partners(), max_length_m=7.0)
    assert len(x) == 3
    assert (x[0], y[0], lengths[0]) == (10.0, 0.0, pytest.approx(4.5))
    assert lengths[1] == pytest.approx(6.35) and lengths[2] == pytest.approx(6.35)
    assert x[1] == pytest.approx(50.0) and x[2] == pytest.approx(50.0)
    assert sorted((y[1], y[2])) == pytest.approx([5.0 - 3.175, 5.0 + 3.175])
    assert list(widths) == pytest.approx([1.9, 2.4, 2.4])
    assert list(vy) == pytest.approx([0.0, 1.0, 1.0]) and list(types) == [1, 1, 1]
    assert h.dtype == np.float32 and types.dtype == np.int32


def test_split_is_the_identity_above_the_longest_partner():
    before = _partners()
    after = nb.split_long_partners(*before, max_length_m=20.0)
    for a, b in zip(before, after):
        assert np.array_equal(a, b)


def test_nearest_partner_slots_keeps_the_devkit_order_until_the_slots_run_out():
    x = np.array([30.0, 5.0, 20.0], np.float32)
    y = np.zeros(3, np.float32)
    assert list(nb.nearest_partner_slots(x, y, 0.0, 0.0, 3)) == [0, 1, 2]
    assert list(nb.nearest_partner_slots(x, y, 0.0, 0.0, 2)) == [1, 2]
