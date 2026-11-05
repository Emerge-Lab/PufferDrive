"""A nuPlan partner's stopped time follows its track token, not the shadow-env slot it happens to occupy."""

import os
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("nuplan")
pytest.importorskip("carl_nuplan")

from pufferlib.ocean.cosim.nuplan.planner import PufferDrivePlanner


STEP_SECONDS = 0.1


def _planner():
    planner = PufferDrivePlanner.__new__(PufferDrivePlanner)
    planner.__dict__.update({"_arch": {"dt": STEP_SECONDS}, "_partner_stopped_s": {}})
    return planner


def _track(track_token, speed_mps):
    return SimpleNamespace(track_token=track_token, velocity=SimpleNamespace(x=speed_mps, y=0.0))


def _static_object(track_token):
    return SimpleNamespace(track_token=track_token)


def test_stopped_time_survives_a_change_of_list_order():
    planner = _planner()
    planner._advance_partner_stopped_s([_track("walker", 0.0), _track("car", 5.0)])
    planner._advance_partner_stopped_s([_track("car", 5.0), _track("new", 3.0), _track("walker", 0.05)])
    assert planner._partner_stopped_s == pytest.approx({"walker": 2 * STEP_SECONDS, "car": 0.0, "new": 0.0})


def test_moving_restarts_the_count_and_static_objects_always_count():
    planner = _planner()
    for walker_speed_mps in (0.0, 0.0, 1.0, 0.0):
        planner._advance_partner_stopped_s([_track("walker", walker_speed_mps), _static_object("cone")])
    assert planner._partner_stopped_s == pytest.approx({"walker": STEP_SECONDS, "cone": 4 * STEP_SECONDS})


def test_a_track_missing_from_one_observation_starts_over():
    planner = _planner()
    planner._advance_partner_stopped_s([_track("walker", 0.0)])
    planner._advance_partner_stopped_s([])
    planner._advance_partner_stopped_s([_track("walker", 0.0)])
    assert planner._partner_stopped_s == pytest.approx({"walker": STEP_SECONDS})
