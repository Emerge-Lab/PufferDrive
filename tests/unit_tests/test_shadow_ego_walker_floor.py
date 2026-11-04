import math
from types import SimpleNamespace

from pufferlib.ocean.cosim.carla.shadow_ego import WALKER_SPEED_FLOOR_WINDOW_S, ShadowEgo


def _shadow(floor_mps):
    return SimpleNamespace(walker_speed_floor_mps=floor_mps, _walker_first_seen_s={})


def test_fresh_slow_walker_reads_floor_speed_along_its_facing():
    shadow = _shadow(3.8)
    vx, vy = ShadowEgo._floored_walker_velocity(shadow, 7, 10.0, math.pi / 2, 0.0, 0.0)
    assert abs(vx) < 1e-6 and abs(vy - 3.8) < 1e-6
    vx, vy = ShadowEgo._floored_walker_velocity(shadow, 7, 10.5, math.pi / 2, 0.0, 1.0)
    assert abs(vx) < 1e-6 and abs(vy - 3.8) < 1e-6


def test_fast_walker_and_expired_window_keep_raw_velocity():
    shadow = _shadow(3.8)
    assert ShadowEgo._floored_walker_velocity(shadow, 7, 10.0, 0.0, 4.0, 0.5) == (4.0, 0.5)
    late_s = 10.0 + WALKER_SPEED_FLOOR_WINDOW_S + 0.1
    assert ShadowEgo._floored_walker_velocity(shadow, 7, late_s, 0.0, 0.0, 0.0) == (0.0, 0.0)
    assert shadow._walker_first_seen_s == {7: 10.0}
