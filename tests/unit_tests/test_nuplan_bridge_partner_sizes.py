"""Partner size eval hacks: the VRU floor grows pedestrian/bicycle boxes to the floor on each axis, the pad adds
to pedestrian boxes only; vehicles keep their size and the inputs are never modified."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim import nuplan_bridge as nb


def test_floor_grows_only_vru_boxes_below_the_floor():
    types = np.array([1, 2, 3, 2], np.int32)  # vehicle, pedestrian, bicycle, pedestrian
    lengths = np.array([4.5, 0.5, 1.7, 0.9], np.float32)
    widths = np.array([1.9, 0.6, 0.5, 0.7], np.float32)
    ln, wd = nb.floor_vru_partner_sizes(types, lengths, widths, 0.8)
    np.testing.assert_allclose(ln, [4.5, 0.8, 1.7, 0.9])
    np.testing.assert_allclose(wd, [1.9, 0.8, 0.8, 0.8])
    assert ln.dtype == np.float32 and wd.dtype == np.float32


def test_floor_leaves_inputs_untouched():
    types = np.array([2], np.int32)
    lengths = np.array([0.4], np.float32)
    widths = np.array([0.4], np.float32)
    nb.floor_vru_partner_sizes(types, lengths, widths, 0.8)
    assert lengths[0] == pytest.approx(0.4) and widths[0] == pytest.approx(0.4)


def test_pad_grows_only_pedestrian_boxes():
    types = np.array([1, 2, 3, 2], np.int32)  # vehicle, pedestrian, bicycle, pedestrian
    lengths = np.array([4.5, 0.5, 1.7, 0.9], np.float32)
    widths = np.array([1.9, 0.6, 0.5, 0.7], np.float32)
    ln, wd = nb.pad_pedestrian_partner_sizes(types, lengths, widths, 0.8)
    np.testing.assert_allclose(ln, [4.5, 1.3, 1.7, 1.7], rtol=1e-6)
    np.testing.assert_allclose(wd, [1.9, 1.4, 0.5, 1.5], rtol=1e-6)
    assert ln.dtype == np.float32 and wd.dtype == np.float32
    assert lengths[1] == pytest.approx(0.5) and widths[1] == pytest.approx(0.6)


def test_pad_stacks_on_the_floor():
    types = np.array([2, 3], np.int32)  # pedestrian, bicycle
    lengths = np.array([0.4, 1.7], np.float32)
    widths = np.array([0.4, 0.5], np.float32)
    ln, wd = nb.floor_vru_partner_sizes(types, lengths, widths, 0.8)
    ln, wd = nb.pad_pedestrian_partner_sizes(types, ln, wd, 0.8)
    np.testing.assert_allclose(ln, [1.6, 1.7], rtol=1e-6)
    np.testing.assert_allclose(wd, [1.6, 0.8], rtol=1e-6)
