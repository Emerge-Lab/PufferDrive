"""pad_vru_partner_sizes grows pedestrian and bicycle boxes by the pad on both axes and leaves its inputs untouched."""

import numpy as np
import pytest

from opengigaflow.ocean.cosim import nuplan_bridge as nb


def test_pad_grows_pedestrian_and_bicycle_boxes_only():
    types = np.array([1, 2, 3, 2], np.int32)  # vehicle, pedestrian, bicycle, pedestrian
    lengths = np.array([4.5, 0.5, 1.7, 0.9], np.float32)
    widths = np.array([1.9, 0.6, 0.5, 0.7], np.float32)
    ln, wd = nb.pad_vru_partner_sizes(types, lengths, widths, 0.8)
    np.testing.assert_allclose(ln, [4.5, 1.3, 2.5, 1.7], rtol=1e-6)
    np.testing.assert_allclose(wd, [1.9, 1.4, 1.3, 1.5], rtol=1e-6)
    assert ln.dtype == np.float32 and wd.dtype == np.float32


def test_pad_leaves_inputs_untouched():
    types = np.array([2], np.int32)
    lengths = np.array([0.4], np.float32)
    widths = np.array([0.4], np.float32)
    nb.pad_vru_partner_sizes(types, lengths, widths, 0.8)
    assert lengths[0] == pytest.approx(0.4) and widths[0] == pytest.approx(0.4)
