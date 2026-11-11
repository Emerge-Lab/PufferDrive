import math
from types import SimpleNamespace

from pufferlib.ocean.cosim import carla_bridge as cb
from pufferlib.ocean.cosim.carla.shadow_ego import ShadowEgo


YELLOW_LIGHT_ID = 7
YELLOW_ELEMENT_IDX = 1


def _reported_yellow_seconds(yellow_margin_s, carla_remaining_s):
    shadow = SimpleNamespace(
        num_traffic=2,
        light_bin_indices_by_id={YELLOW_LIGHT_ID: [YELLOW_ELEMENT_IDX]},
        _yellow_margin_s=yellow_margin_s,
        _yellow_remaining_by_id=lambda snapshot, state_by_id: {YELLOW_LIGHT_ID: carla_remaining_s},
    )
    snapshot = {"light_states": [[YELLOW_LIGHT_ID, "Yellow"]], "ego_light_id": YELLOW_LIGHT_ID}
    states, yellow_remaining_s = ShadowEgo._read_light_states(shadow, snapshot)
    assert states[YELLOW_ELEMENT_IDX] == cb.TRAFFIC_LIGHT_STATE_YELLOW
    return float(yellow_remaining_s[YELLOW_ELEMENT_IDX])


def test_without_a_margin_the_policy_sees_the_yellow_time_carla_has_left():
    assert _reported_yellow_seconds(0.0, 2.0) == 2.0


def test_the_margin_shortens_the_yellow_time_the_policy_sees():
    assert math.isclose(_reported_yellow_seconds(0.5, 2.0), 1.5)


def test_the_reported_yellow_time_never_goes_negative():
    assert _reported_yellow_seconds(0.5, 0.2) == 0.0
