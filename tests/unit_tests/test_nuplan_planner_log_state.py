"""The planner's pickled state (nuPlan writes it into every simulation log) carries no policy, env or goal window."""

import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("nuplan")
pytest.importorskip("carl_nuplan")

from pufferlib.ocean.cosim.nuplan.planner import LOG_EXCLUDED_STATE, PufferDrivePlanner

WEIGHT_BYTES = 1 << 20


def test_pickled_planner_drops_its_runtime_state():
    planner = PufferDrivePlanner.__new__(PufferDrivePlanner)
    planner.__dict__.update({"_checkpoint_path": "model.pt", "_arch": {"num_agents": 2}})
    planner.__dict__.update({name: {"weights": bytes(WEIGHT_BYTES)} for name in LOG_EXCLUDED_STATE})
    pickled = pickle.dumps(planner)
    assert len(pickled) < WEIGHT_BYTES
    restored = pickle.loads(pickled)
    assert all(getattr(restored, name) is None for name in LOG_EXCLUDED_STATE)
    assert restored._checkpoint_path == "model.pt" and restored._arch == {"num_agents": 2}
    assert all(getattr(planner, name) is not None for name in LOG_EXCLUDED_STATE)
