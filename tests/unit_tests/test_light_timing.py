"""YellowCountdown (co-sim remaining-yellow fallback) and the CARLA phase-timer helper."""

import pytest

from pufferlib.ocean.cosim import carla_bridge as cb
from pufferlib.ocean.cosim.light_timing import YellowCountdown


def test_countdown_runs_from_onset_and_floors_at_one_step():
    countdown = YellowCountdown(assumed_total_s=2.0, min_remaining_s=0.1)
    assert countdown.tick({7}, 10.0) == {7: 2.0}
    assert countdown.tick({7}, 10.5) == {7: pytest.approx(1.5)}
    assert countdown.tick({7}, 12.5) == {7: pytest.approx(0.1)}  # past the assumed total: one step stays
    assert countdown.tick(set(), 12.6) == {}
    assert countdown.tick({7}, 20.0) == {7: 2.0}  # restarts after a non-yellow tick


def test_countdown_tracks_lights_independently():
    countdown = YellowCountdown(2.0, 0.1)
    countdown.tick({1}, 0.0)
    assert countdown.tick({1, 2}, 1.0) == {1: pytest.approx(1.0), 2: 2.0}


def test_countdown_rejects_nonpositive_durations():
    with pytest.raises(ValueError):
        YellowCountdown(0.0, 0.1)
    with pytest.raises(ValueError):
        YellowCountdown(2.0, 0.0)


def test_carla_timer_gives_remaining_or_none_when_frozen():
    assert cb.carla_yellow_remaining_s(elapsed_s=0.5, yellow_time_s=3.1) == pytest.approx(2.6)
    assert cb.carla_yellow_remaining_s(elapsed_s=3.3, yellow_time_s=3.1) == 0.0
    assert cb.carla_yellow_remaining_s(elapsed_s=0.5, yellow_time_s=1e9) is None
