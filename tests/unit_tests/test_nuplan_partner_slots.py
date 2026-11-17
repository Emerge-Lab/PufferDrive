"""nearest_partner_slots: nuPlan partners keep the devkit order until the shadow env's slots run out."""

import numpy as np

from opengigaflow.ocean.cosim import nuplan_bridge as nb


def test_nearest_partner_slots_keeps_the_devkit_order_until_the_slots_run_out():
    x = np.array([30.0, 5.0, 20.0], np.float32)
    y = np.zeros(3, np.float32)
    assert list(nb.nearest_partner_slots(x, y, 0.0, 0.0, 3)) == [0, 1, 2]
    assert list(nb.nearest_partner_slots(x, y, 0.0, 0.0, 2)) == [1, 2]
