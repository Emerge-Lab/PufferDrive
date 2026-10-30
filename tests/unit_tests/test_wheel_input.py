import ast
import json
import math
import time
from pathlib import Path

import pytest

from pufferlib.wheel_demo.input.base import (
    HAT_NEUTRAL,
    INPUT_STALE_SECONDS,
    WheelCalibration,
    WheelInput,
)
from pufferlib.wheel_demo.input.g920 import (
    G920_REPORT_ID,
    G920_REPORT_LENGTH,
    G920_PRODUCT_ID,
    G920_VENDOR_ID,
    ButtonMap,
    decode_button_mask,
    parse_g920_report,
)
from pufferlib.wheel_demo.input.trace import (
    SyntheticWheelInput,
    TraceWheelInput,
    load_report_trace,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "g920" / "wheel_sweep_10byte.bin"

FIXTURE_REPORT_COUNT = 6572
FIXTURE_CONSTANT_TAIL_BYTE = 0x05
FIXTURE_BUTTON_FRAME_IDX = 6571
FIXTURE_BUTTON_BIT_IDX = 0

# First frame on which each pedal byte leaves its released value, in the order a human pressed them.
FIXTURE_THROTTLE_FIRST_PRESS_FRAME = 5082
FIXTURE_BRAKE_FIRST_PRESS_FRAME = 5206
FIXTURE_CLUTCH_FIRST_PRESS_FRAME = 5231

PEDAL_RAW_RELEASED = 255
PEDAL_RAW_FLOORED = 0
PEDAL_PRESSED_NORM = 0.15
STEERING_SWEEP_MIN_NORM = -0.9
STEERING_SWEEP_MAX_NORM = 0.9

G920_RAW_MIN = 0
G920_RAW_CENTER = 32768
G920_RAW_MAX = 65535

SHIFT_UP_BIT_IDX = 0
SHIFT_DOWN_BIT_IDX = 1

STALE_MARGIN_SECONDS = 0.1
READER_ERROR_TIMEOUT_SECONDS = 2.0
READER_POLL_INTERVAL_SECONDS = 0.005

PEDAL_NAMES = ("throttle", "brake", "clutch")


def _make_report(
    steering_raw=G920_RAW_CENTER,
    throttle_raw=PEDAL_RAW_RELEASED,
    brake_raw=PEDAL_RAW_RELEASED,
    clutch_raw=PEDAL_RAW_RELEASED,
    button_nibble=0x0,
    button_byte2=0x00,
    button_byte3=0x00,
    hat=HAT_NEUTRAL,
    report_id=G920_REPORT_ID,
):
    return bytes(
        [
            report_id,
            (button_nibble << 4) | hat,
            button_byte2,
            button_byte3,
            steering_raw & 0xFF,
            (steering_raw >> 8) & 0xFF,
            throttle_raw,
            brake_raw,
            clutch_raw,
            FIXTURE_CONSTANT_TAIL_BYTE,
        ]
    )


@pytest.fixture(scope="module")
def fixture_reports():
    return load_report_trace(FIXTURE_PATH)


@pytest.fixture(scope="module")
def fixture_decoded(fixture_reports):
    calibration = WheelCalibration()
    return [parse_g920_report(report, calibration) for report in fixture_reports]


def test_steering_center_maps_to_exact_zero():
    calibration = WheelCalibration()
    assert calibration.steering_to_norm(G920_RAW_CENTER) == 0.0


def test_steering_full_lock_maps_to_exact_unit():
    calibration = WheelCalibration()
    assert calibration.steering_to_norm(G920_RAW_MIN) == -1.0
    assert calibration.steering_to_norm(G920_RAW_MAX) == 1.0


def test_steering_out_of_range_raw_clamps_to_unit():
    calibration = WheelCalibration()
    assert calibration.steering_to_norm(G920_RAW_MAX + 10_000) == 1.0
    assert calibration.steering_to_norm(G920_RAW_MIN - 10_000) == -1.0


def test_steering_asymmetric_center_reaches_unit_on_both_sides():
    calibration = WheelCalibration(steering_raw_min=1281, steering_raw_center=32755, steering_raw_max=65535)
    assert calibration.steering_to_norm(1281) == -1.0
    assert calibration.steering_to_norm(32755) == 0.0
    assert calibration.steering_to_norm(65535) == 1.0

    left_half_raw = 32755 - (32755 - 1281) // 2
    right_half_raw = 32755 + (65535 - 32755) // 2
    assert calibration.steering_to_norm(left_half_raw) == pytest.approx(-0.5, abs=1e-4)
    assert calibration.steering_to_norm(right_half_raw) == pytest.approx(0.5, abs=1e-4)


def test_steering_asymmetric_center_does_not_leak_short_side_scale():
    calibration = WheelCalibration(steering_raw_min=0, steering_raw_center=10_000, steering_raw_max=65_535)
    assert calibration.steering_to_norm(5_000) == pytest.approx(-0.5)
    assert calibration.steering_to_norm(20_000) == pytest.approx(10_000 / 55_535)


def test_steering_deadzone_zeroes_small_input_and_still_reaches_full_lock():
    deadzone_norm = 0.1
    calibration = WheelCalibration(steering_deadzone_norm=deadzone_norm)
    inside_deadzone_raw = G920_RAW_CENTER + int(0.5 * deadzone_norm * (G920_RAW_MAX - G920_RAW_CENTER))

    assert calibration.steering_to_norm(inside_deadzone_raw) == 0.0
    assert calibration.steering_to_norm(G920_RAW_MAX) == 1.0
    assert calibration.steering_to_norm(G920_RAW_MIN) == -1.0


def test_steering_deadzone_rescales_edge_to_zero_and_midpoint_linearly():
    deadzone_norm = 0.2
    calibration = WheelCalibration(steering_deadzone_norm=deadzone_norm)
    right_span = G920_RAW_MAX - G920_RAW_CENTER

    at_deadzone_raw = G920_RAW_CENTER + int(deadzone_norm * right_span)
    assert calibration.steering_to_norm(at_deadzone_raw) == 0.0

    half_travel_raw = G920_RAW_CENTER + int((deadzone_norm + 0.5 * (1.0 - deadzone_norm)) * right_span)
    assert calibration.steering_to_norm(half_travel_raw) == pytest.approx(0.5, abs=1e-4)


def test_steering_deadzone_is_symmetric_about_center():
    calibration = WheelCalibration(steering_deadzone_norm=0.1)
    offset_raw = 10_000
    left = calibration.steering_to_norm(G920_RAW_CENTER - offset_raw)
    right = calibration.steering_to_norm(G920_RAW_CENTER + offset_raw)
    assert left < 0.0 < right
    assert left == pytest.approx(-right, abs=1e-4)


@pytest.mark.parametrize("pedal", PEDAL_NAMES)
def test_pedal_rest_maps_to_zero_and_floor_maps_to_one(pedal):
    calibration = WheelCalibration()
    assert calibration.pedal_to_norm(PEDAL_RAW_RELEASED, pedal) == 0.0
    assert calibration.pedal_to_norm(PEDAL_RAW_FLOORED, pedal) == 1.0


@pytest.mark.parametrize("pedal", PEDAL_NAMES)
def test_pedal_out_of_range_raw_clamps(pedal):
    calibration = WheelCalibration()
    assert calibration.pedal_to_norm(PEDAL_RAW_RELEASED + 100, pedal) == 0.0
    assert calibration.pedal_to_norm(PEDAL_RAW_FLOORED - 100, pedal) == 1.0


def test_pedal_deadzone_zeroes_small_travel_and_rescales_to_full_range():
    deadzone_norm = 0.25
    calibration = WheelCalibration(pedal_deadzone_norm=deadzone_norm)
    inside_deadzone_raw = int(round(PEDAL_RAW_RELEASED * (1.0 - 0.5 * deadzone_norm)))
    half_travel_raw = int(round(PEDAL_RAW_RELEASED * (1.0 - (deadzone_norm + 0.5 * (1.0 - deadzone_norm)))))

    assert calibration.pedal_to_norm(inside_deadzone_raw, "throttle") == 0.0
    assert calibration.pedal_to_norm(PEDAL_RAW_FLOORED, "throttle") == 1.0
    assert calibration.pedal_to_norm(half_travel_raw, "throttle") == pytest.approx(0.5, abs=1e-2)


def test_pedal_with_non_inverted_calibration_still_maps_rest_to_zero():
    calibration = WheelCalibration(brake_raw_rest=12, brake_raw_floor=240, pedal_deadzone_norm=0.0)
    assert calibration.pedal_to_norm(12, "brake") == 0.0
    assert calibration.pedal_to_norm(240, "brake") == 1.0
    assert calibration.pedal_to_norm(126, "brake") == pytest.approx(0.5, abs=1e-2)


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"steering_raw_min": G920_RAW_CENTER},
        {"steering_raw_min": G920_RAW_CENTER + 1},
        {"steering_raw_max": G920_RAW_CENTER},
        {"steering_raw_max": G920_RAW_CENTER - 1},
        {"steering_raw_min": 100, "steering_raw_center": 100, "steering_raw_max": 100},
        {"throttle_raw_rest": 128, "throttle_raw_floor": 128},
        {"brake_raw_rest": 0, "brake_raw_floor": 0},
        {"clutch_raw_rest": 255, "clutch_raw_floor": 255},
        {"steering_deadzone_norm": -0.01},
        {"steering_deadzone_norm": 1.0},
        {"steering_deadzone_norm": 1.5},
        {"pedal_deadzone_norm": -0.5},
        {"pedal_deadzone_norm": 1.0},
        {"pedal_deadzone_norm": math.nan},
    ],
)
def test_invalid_calibration_raises(invalid_kwargs):
    with pytest.raises(ValueError):
        WheelCalibration(**invalid_kwargs)


def test_calibration_json_round_trip_preserves_values(tmp_path):
    calibration = WheelCalibration(
        steering_raw_min=1281,
        steering_raw_center=32755,
        steering_raw_max=65535,
        steering_deadzone_norm=0.035,
        throttle_raw_rest=254,
        throttle_raw_floor=3,
        brake_raw_rest=250,
        brake_raw_floor=10,
        clutch_raw_rest=253,
        clutch_raw_floor=1,
        pedal_deadzone_norm=0.04,
    )
    path = tmp_path / "calibration.json"
    calibration.to_json(path)

    assert WheelCalibration.from_json(path) == calibration
    assert set(json.loads(path.read_text())) == set(WheelCalibration.__dataclass_fields__)


def test_calibration_json_unknown_key_raises(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"steering_raw_center": 32755, "steering_raw_centre": 32755}))

    with pytest.raises(ValueError, match="unknown calibration keys"):
        WheelCalibration.from_json(path)


def test_calibration_json_invalid_values_still_raise(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"steering_raw_min": 60_000, "steering_raw_center": 100}))

    with pytest.raises(ValueError):
        WheelCalibration.from_json(path)


@pytest.mark.parametrize("report_length", [0, 1, G920_REPORT_LENGTH - 1, G920_REPORT_LENGTH + 1, 64])
def test_parse_rejects_wrong_report_length(report_length):
    calibration = WheelCalibration()
    report = bytes([G920_REPORT_ID]) + bytes(max(report_length - 1, 0))

    with pytest.raises(ValueError, match="must be 10 bytes"):
        parse_g920_report(report[:report_length], calibration)


@pytest.mark.parametrize("report_id", [0x00, 0x02, 0xFF])
def test_parse_rejects_wrong_report_id(report_id):
    calibration = WheelCalibration()

    with pytest.raises(ValueError, match="report id"):
        parse_g920_report(_make_report(report_id=report_id), calibration)


def test_steering_decodes_little_endian():
    calibration = WheelCalibration()
    low_byte, high_byte = 0x34, 0x12
    report = _make_report()
    report = report[:4] + bytes([low_byte, high_byte]) + report[6:]

    decoded = parse_g920_report(report, calibration)
    assert decoded["steering_raw"] == (high_byte << 8) | low_byte
    assert decoded["steering_raw"] != (low_byte << 8) | high_byte


def test_steering_decode_uses_full_sixteen_bit_range():
    calibration = WheelCalibration()
    assert parse_g920_report(_make_report(steering_raw=G920_RAW_MAX), calibration)["steering_raw"] == G920_RAW_MAX
    assert parse_g920_report(_make_report(steering_raw=G920_RAW_MIN), calibration)["steering_raw"] == G920_RAW_MIN


def test_button_mask_assembles_bytes_one_two_three_at_distinct_offsets():
    report = _make_report(button_nibble=0x5, button_byte2=0xAA, button_byte3=0x55)
    assert decode_button_mask(report) == 0x5 | (0xAA << 4) | (0x55 << 12)


@pytest.mark.parametrize(
    "button_nibble,button_byte2,button_byte3,expected_bit_idx",
    [
        (0x1, 0x00, 0x00, 0),
        (0x8, 0x00, 0x00, 3),
        (0x0, 0x01, 0x00, 4),
        (0x0, 0x80, 0x00, 11),
        (0x0, 0x00, 0x01, 12),
        (0x0, 0x00, 0x80, 19),
    ],
)
def test_button_mask_maps_each_source_bit_to_one_mask_bit(button_nibble, button_byte2, button_byte3, expected_bit_idx):
    report = _make_report(button_nibble=button_nibble, button_byte2=button_byte2, button_byte3=button_byte3)
    assert decode_button_mask(report) == 1 << expected_bit_idx


def test_hat_decode_ignores_button_nibble():
    calibration = WheelCalibration()
    decoded = parse_g920_report(_make_report(button_nibble=0xF, hat=0x03), calibration)
    assert decoded["hat"] == 0x03
    assert parse_g920_report(_make_report(button_nibble=0xF), calibration)["hat"] == HAT_NEUTRAL


def test_parse_returns_raw_and_normalized_pedals_from_their_own_bytes():
    calibration = WheelCalibration()
    decoded = parse_g920_report(
        _make_report(throttle_raw=PEDAL_RAW_FLOORED, brake_raw=PEDAL_RAW_RELEASED, clutch_raw=128), calibration
    )

    assert decoded["throttle_raw"] == PEDAL_RAW_FLOORED
    assert decoded["brake_raw"] == PEDAL_RAW_RELEASED
    assert decoded["clutch_raw"] == 128
    assert decoded["throttle_norm"] == 1.0
    assert decoded["brake_norm"] == 0.0
    assert 0.0 < decoded["clutch_norm"] < 1.0


def test_fixture_has_expected_report_count(fixture_reports):
    assert len(fixture_reports) == FIXTURE_REPORT_COUNT
    assert all(len(report) == G920_REPORT_LENGTH for report in fixture_reports)


def test_fixture_report_id_and_tail_byte_are_constant(fixture_reports):
    assert {report[0] for report in fixture_reports} == {G920_REPORT_ID}
    assert {report[9] for report in fixture_reports} == {FIXTURE_CONSTANT_TAIL_BYTE}


def test_fixture_has_exactly_one_button_frame(fixture_reports):
    pressed_frames = [idx for idx, report in enumerate(fixture_reports) if decode_button_mask(report)]
    assert pressed_frames == [FIXTURE_BUTTON_FRAME_IDX]
    assert decode_button_mask(fixture_reports[FIXTURE_BUTTON_FRAME_IDX]) == 1 << FIXTURE_BUTTON_BIT_IDX


def test_fixture_decodes_every_frame_within_normalized_bounds(fixture_decoded):
    assert len(fixture_decoded) == FIXTURE_REPORT_COUNT
    for frame_idx, decoded in enumerate(fixture_decoded):
        assert -1.0 <= decoded["steering_norm"] <= 1.0, frame_idx
        for pedal in PEDAL_NAMES:
            assert 0.0 <= decoded[f"{pedal}_norm"] <= 1.0, (frame_idx, pedal)
        assert decoded["hat"] == HAT_NEUTRAL, frame_idx


def test_fixture_steering_sweeps_near_full_lock(fixture_decoded):
    steering_norms = [decoded["steering_norm"] for decoded in fixture_decoded]
    assert min(steering_norms) <= STEERING_SWEEP_MIN_NORM
    assert max(steering_norms) >= STEERING_SWEEP_MAX_NORM
    assert abs(steering_norms[0]) < 0.05


@pytest.mark.parametrize("pedal", PEDAL_NAMES)
def test_fixture_each_pedal_reaches_a_pressed_state(fixture_decoded, pedal):
    pedal_norms = [decoded[f"{pedal}_norm"] for decoded in fixture_decoded]
    assert min(pedal_norms) == 0.0
    assert max(pedal_norms) > PEDAL_PRESSED_NORM


@pytest.mark.parametrize("pedal", ("throttle", "clutch"))
def test_fixture_throttle_and_clutch_reach_the_floor(fixture_decoded, pedal):
    assert max(decoded[f"{pedal}_norm"] for decoded in fixture_decoded) == 1.0


def _first_press_frame(fixture_decoded, pedal):
    for frame_idx, decoded in enumerate(fixture_decoded):
        if decoded[f"{pedal}_raw"] != PEDAL_RAW_RELEASED:
            return frame_idx
    raise AssertionError(f"{pedal} never left its released value in the fixture")


def test_fixture_pedal_press_order_is_throttle_then_brake_then_clutch(fixture_decoded):
    throttle_frame = _first_press_frame(fixture_decoded, "throttle")
    brake_frame = _first_press_frame(fixture_decoded, "brake")
    clutch_frame = _first_press_frame(fixture_decoded, "clutch")

    assert throttle_frame < brake_frame < clutch_frame
    assert throttle_frame == FIXTURE_THROTTLE_FIRST_PRESS_FRAME
    assert brake_frame == FIXTURE_BRAKE_FIRST_PRESS_FRAME
    assert clutch_frame == FIXTURE_CLUTCH_FIRST_PRESS_FRAME


def test_fixture_pedals_are_released_before_the_pedal_phase(fixture_decoded):
    for decoded in fixture_decoded[:FIXTURE_THROTTLE_FIRST_PRESS_FRAME]:
        for pedal in PEDAL_NAMES:
            assert decoded[f"{pedal}_raw"] == PEDAL_RAW_RELEASED


def test_trace_steps_threadlessly_and_counts_reports_monotonically(fixture_reports):
    wheel = TraceWheelInput(fixture_reports[:5])
    assert wheel.latest().report_count == 0

    previous_count = 0
    for expected_count in range(1, 6):
        wheel._read_state()
        state = wheel.latest()
        assert state.report_count == expected_count
        assert state.report_count > previous_count
        previous_count = state.report_count


def test_trace_replays_the_decoded_values_in_order(fixture_reports):
    calibration = WheelCalibration()
    wheel = TraceWheelInput(fixture_reports[:20], calibration=calibration)
    for report in fixture_reports[:20]:
        wheel._read_state()
        expected = parse_g920_report(report, calibration)
        state = wheel.latest()
        assert state.steering_norm == expected["steering_norm"]
        assert state.throttle_norm == expected["throttle_norm"]
        assert state.brake_norm == expected["brake_norm"]
        assert state.clutch_norm == expected["clutch_norm"]


def test_trace_exhausts_at_the_end_without_loop(fixture_reports):
    wheel = TraceWheelInput(fixture_reports[:3], loop=False)
    assert not wheel.exhausted

    for _ in range(3):
        wheel._read_state()
    assert wheel.exhausted
    assert wheel.latest().report_count == 3

    wheel._read_state()
    assert wheel.latest().report_count == 3
    assert not wheel._running


def test_trace_wraps_when_looping(fixture_reports):
    reports = fixture_reports[:3]
    wheel = TraceWheelInput(reports, loop=True)
    for _ in range(7):
        wheel._read_state()

    assert not wheel.exhausted
    assert wheel.latest().report_count == 7
    assert wheel.frame_idx == 7

    calibration = WheelCalibration()
    expected = parse_g920_report(reports[(7 - 1) % len(reports)], calibration)
    assert wheel.latest().steering_norm == expected["steering_norm"]


def test_trace_from_file_loads_the_whole_fixture():
    wheel = TraceWheelInput.from_file(FIXTURE_PATH)
    assert len(wheel.reports) == FIXTURE_REPORT_COUNT
    wheel._read_state()
    assert wheel.latest().report_count == 1


def test_trace_file_with_ragged_length_raises(tmp_path):
    path = tmp_path / "ragged.bin"
    path.write_bytes(bytes(G920_REPORT_LENGTH * 2 + 3))

    with pytest.raises(ValueError, match="not a whole number"):
        load_report_trace(path)


def test_trace_file_that_is_empty_raises(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="not a whole number"):
        load_report_trace(path)


def test_trace_with_empty_report_list_raises():
    with pytest.raises(ValueError, match="at least one report"):
        TraceWheelInput([])


def test_trace_counts_a_button_edge_once():
    pressed = _make_report(button_nibble=1 << SHIFT_UP_BIT_IDX)
    released = _make_report()
    button_map = ButtonMap(shift_up_bit=SHIFT_UP_BIT_IDX, shift_down_bit=SHIFT_DOWN_BIT_IDX)
    wheel = TraceWheelInput([released, pressed, pressed, released, pressed], button_map=button_map)

    for _ in range(5):
        wheel._read_state()

    assert wheel.latest().shift_up_count == 2
    assert wheel.latest().shift_down_count == 0


def test_trace_of_the_fixture_sees_its_single_button_edge(fixture_reports):
    button_map = ButtonMap(shift_up_bit=FIXTURE_BUTTON_BIT_IDX, shift_down_bit=SHIFT_DOWN_BIT_IDX)
    wheel = TraceWheelInput(fixture_reports, button_map=button_map)
    for _ in range(FIXTURE_REPORT_COUNT):
        wheel._read_state()

    assert wheel.exhausted
    assert wheel.latest().report_count == FIXTURE_REPORT_COUNT
    assert wheel.latest().shift_up_count == 1
    assert wheel.latest().shift_down_count == 0


def test_trace_without_mapped_shift_bits_never_counts_a_shift(fixture_reports):
    wheel = TraceWheelInput(fixture_reports)
    for _ in range(FIXTURE_REPORT_COUNT):
        wheel._read_state()

    assert wheel.latest().buttons == 1 << FIXTURE_BUTTON_BIT_IDX
    assert wheel.latest().shift_up_count == 0
    assert wheel.latest().shift_down_count == 0


def test_synthetic_set_controls_publishes_immediately():
    wheel = SyntheticWheelInput()
    assert wheel.latest().report_count == 0

    wheel.set_controls(steering_norm=-0.25, throttle_norm=0.5, brake_norm=0.125, clutch_norm=0.75, buttons=3, hat=2)
    state = wheel.latest()

    assert state.steering_norm == -0.25
    assert state.throttle_norm == 0.5
    assert state.brake_norm == 0.125
    assert state.clutch_norm == 0.75
    assert state.buttons == 3
    assert state.hat == 2
    assert state.report_count == 1


def test_synthetic_set_controls_keeps_unspecified_channels():
    wheel = SyntheticWheelInput()
    wheel.set_controls(steering_norm=0.4, throttle_norm=0.6)
    wheel.set_controls(brake_norm=0.2)
    state = wheel.latest()

    assert state.steering_norm == 0.4
    assert state.throttle_norm == 0.6
    assert state.brake_norm == 0.2
    assert state.report_count == 2


def test_synthetic_shift_requests_are_counted_once_and_are_monotonic():
    wheel = SyntheticWheelInput()
    wheel.request_shift_up()
    assert wheel.latest().shift_up_count == 1

    wheel.set_controls(throttle_norm=1.0)
    assert wheel.latest().shift_up_count == 1
    assert wheel.latest().shift_down_count == 0

    wheel.request_shift_up()
    wheel.request_shift_down()
    assert wheel.latest().shift_up_count == 2
    assert wheel.latest().shift_down_count == 1

    wheel.set_controls(brake_norm=1.0)
    assert wheel.latest().shift_up_count == 2
    assert wheel.latest().shift_down_count == 1


def test_synthetic_report_count_is_monotonic_across_every_publish():
    wheel = SyntheticWheelInput()
    previous_count = 0
    for step_idx in range(10):
        wheel.set_controls(steering_norm=0.1 * step_idx)
        count = wheel.latest().report_count
        assert count == previous_count + 1
        previous_count = count


@pytest.mark.parametrize(
    "button_map",
    [
        ButtonMap(),
        ButtonMap(shift_up_bit=SHIFT_UP_BIT_IDX),
        ButtonMap(shift_down_bit=SHIFT_DOWN_BIT_IDX),
    ],
)
def test_require_shift_bits_raises_when_a_paddle_is_unmapped(button_map):
    with pytest.raises(RuntimeError, match="unmapped"):
        button_map.require_shift_bits()


def test_require_shift_bits_accepts_a_fully_mapped_button_map():
    ButtonMap(shift_up_bit=SHIFT_UP_BIT_IDX, shift_down_bit=SHIFT_DOWN_BIT_IDX).require_shift_bits()


def test_button_map_json_round_trip(tmp_path):
    path = tmp_path / "buttons.json"
    button_map = ButtonMap(shift_up_bit=4, shift_down_bit=5)
    button_map.to_json(path)

    assert ButtonMap.from_json(path) == button_map


def test_is_stale_before_any_report():
    wheel = SyntheticWheelInput()
    assert wheel.latest().report_count == 0
    assert wheel.is_stale()
    assert wheel.is_stale(now_s=0.0)


def test_is_stale_once_the_latest_report_is_older_than_the_threshold():
    wheel = SyntheticWheelInput()
    wheel.set_controls(steering_norm=0.0)
    published_s = wheel.latest().timestamp_s

    assert not wheel.is_stale(now_s=published_s)
    assert not wheel.is_stale(now_s=published_s + INPUT_STALE_SECONDS - STALE_MARGIN_SECONDS)
    assert wheel.is_stale(now_s=published_s + INPUT_STALE_SECONDS + STALE_MARGIN_SECONDS)


def test_is_stale_clears_after_a_fresh_report():
    wheel = SyntheticWheelInput()
    wheel.set_controls(steering_norm=0.0)
    stale_now_s = wheel.latest().timestamp_s + INPUT_STALE_SECONDS + STALE_MARGIN_SECONDS
    assert wheel.is_stale(now_s=stale_now_s)

    wheel.set_controls(steering_norm=0.1)
    assert not wheel.is_stale(now_s=wheel.latest().timestamp_s)


class _FailingWheelInput(WheelInput):
    """Stands in for a device that dies after opening; the reader thread must not swallow it."""

    def _open(self):
        pass

    def _read_state(self):
        raise RuntimeError("device went away")


def test_latest_reraises_the_reader_thread_error():
    wheel = _FailingWheelInput()
    wheel.start()
    deadline_s = time.monotonic() + READER_ERROR_TIMEOUT_SECONDS
    while wheel._error is None and time.monotonic() < deadline_s:
        time.sleep(READER_POLL_INTERVAL_SECONDS)

    try:
        with pytest.raises(RuntimeError, match="device went away"):
            wheel.latest()
        with pytest.raises(RuntimeError, match="device went away"):
            wheel.latest()
    finally:
        wheel.stop()

    with pytest.raises(RuntimeError, match="device went away"):
        wheel.latest()


class _IdleWheelInput(WheelInput):
    def _open(self):
        pass

    def _read_state(self):
        time.sleep(READER_POLL_INTERVAL_SECONDS)


def test_start_twice_raises():
    wheel = _IdleWheelInput()
    wheel.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            wheel.start()
    finally:
        wheel.stop()


INPUT_MODULE_PATHS = (
    Path(WheelInput.__module__.replace(".", "/") + ".py"),
    Path(ButtonMap.__module__.replace(".", "/") + ".py"),
    Path(TraceWheelInput.__module__.replace(".", "/") + ".py"),
)


@pytest.mark.parametrize("module_path", INPUT_MODULE_PATHS, ids=lambda path: path.stem)
def test_input_module_does_not_import_hid_at_module_scope(module_path):
    source_path = Path(__file__).resolve().parents[2] / module_path
    tree = ast.parse(source_path.read_text())
    module_scope_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]

    imported_names = set()
    for node in module_scope_imports:
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        else:
            imported_names.add(node.module or "")
    assert "hid" not in imported_names


def test_g920_backend_constructs_without_touching_hardware():
    from pufferlib.wheel_demo.input.g920 import G920HidInput

    wheel = G920HidInput()
    assert wheel.latest().report_count == 0
    assert wheel.is_stale()
    assert wheel._device is None


def _g920_is_connected():
    try:
        import hid
    except ImportError:
        return None
    try:
        return bool(hid.enumerate(G920_VENDOR_ID, G920_PRODUCT_ID))
    except Exception:
        return None


def test_hid_backend_fails_fast_when_no_wheel_is_connected():
    connected = _g920_is_connected()
    if connected is None:
        pytest.skip("hidapi is unavailable, so the raw-HID backend cannot be exercised")
    if connected:
        pytest.skip("a G920 is connected, so the absent-device path cannot be exercised")

    from pufferlib.wheel_demo.input.g920 import G920HidInput, find_g920_interfaces

    assert find_g920_interfaces()["axes"] is None
    with pytest.raises(RuntimeError, match="No G920 axis interface found"):
        G920HidInput()._open()
