"""nearest_partner_slots: nuPlan partners keep the devkit order until the shadow env's slots run out."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402


def test_nearest_partner_slots_keeps_the_devkit_order_until_the_slots_run_out():
    x = np.array([30.0, 5.0, 20.0], np.float32)
    y = np.zeros(3, np.float32)
    assert list(nb.nearest_partner_slots(x, y, 0.0, 0.0, 3)) == [0, 1, 2]
    assert list(nb.nearest_partner_slots(x, y, 0.0, 0.0, 2)) == [1, 2]
