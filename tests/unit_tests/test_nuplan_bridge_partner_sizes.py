"""floor_vru_partner_sizes: pedestrian/bicycle boxes grow to the floor on each axis, vehicles keep their size."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

nuplan = pytest.importorskip("nuplan")

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
