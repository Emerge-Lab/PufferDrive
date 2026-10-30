"""Wheel calibration checks and town selection, with no wheel and no prompt.

Every rule in `validate_calibration` exists because that exact mistake was actually made during
development, so each one gets a calibration that reproduces the mistake and an assertion that the
problem names the right thing. A test that lets one of these back through is worthless.
"""

import importlib.util
import os

import pytest

from pufferlib.wheel_demo.calibrate import (
    CAPTURE_SECONDS,
    MAX_STEERING_CENTER_OFFSET_COUNTS,
    MIN_BRAKE_TRAVEL_FRACTION,
    MIN_PEDAL_TRAVEL_COUNTS,
    MIN_STEERING_SPAN_COUNTS,
    PEDAL_BYTE_IDX,
    STEERING_FULL_SCALE,
    STEERING_HIGH_BYTE,
    STEERING_LOW_BYTE,
    capture_calibration,
    collect,
    validate_calibration,
)
from pufferlib.wheel_demo.input.base import WheelCalibration
from pufferlib.wheel_demo.input.g920 import G920_REPORT_ID, G920_REPORT_LENGTH
from pufferlib.wheel_demo.maps import CARLA_TOWNS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DEMO_PATH = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "run_demo.py")

PEDAL_RAW_RELEASED = 255
PEDAL_RAW_FLOORED = 0
FULL_STEERING_MIN = 0
FULL_STEERING_MAX = STEERING_FULL_SCALE
FULL_STEERING_CENTER = (FULL_STEERING_MIN + FULL_STEERING_MAX) // 2

SHORT_SWEEP_MIN = 18_000
SHORT_SWEEP_MAX = 46_000
OFF_CENTRE_EXTRA_COUNTS = MAX_STEERING_CENTER_OFFSET_COUNTS + 1_000
STUCK_PEDAL_FLOOR = PEDAL_RAW_RELEASED - (MIN_PEDAL_TRAVEL_COUNTS - 20)
WEAK_BRAKE_FLOOR = PEDAL_RAW_RELEASED - 100

FAKE_CLOCK_TICK_S = 0.001


def good_calibration(**overrides):
    fields = dict(
        steering_raw_min=FULL_STEERING_MIN,
        steering_raw_center=FULL_STEERING_CENTER,
        steering_raw_max=FULL_STEERING_MAX,
        throttle_raw_rest=PEDAL_RAW_RELEASED,
        throttle_raw_floor=PEDAL_RAW_FLOORED,
        brake_raw_rest=PEDAL_RAW_RELEASED,
        brake_raw_floor=PEDAL_RAW_FLOORED,
        clutch_raw_rest=PEDAL_RAW_RELEASED,
        clutch_raw_floor=PEDAL_RAW_FLOORED,
    )
    fields.update(overrides)
    return WheelCalibration(**fields)


def problems_mentioning(problems, *words):
    return [p for p in problems if all(word in p.lower() for word in words)]


# ---------------------------------------------------------------------------------------------
# validate_calibration
# ---------------------------------------------------------------------------------------------


def test_a_good_calibration_has_no_problems():
    assert validate_calibration(good_calibration()) == []


def test_default_calibration_has_no_problems():
    assert validate_calibration(WheelCalibration()) == []


def test_steering_sweep_short_of_the_stops_is_caught():
    calibration = good_calibration(
        steering_raw_min=SHORT_SWEEP_MIN,
        steering_raw_center=(SHORT_SWEEP_MIN + SHORT_SWEEP_MAX) // 2,
        steering_raw_max=SHORT_SWEEP_MAX,
    )
    problems = validate_calibration(calibration)

    assert len(problems) == 1
    assert problems_mentioning(problems, "steering", "lock")
    assert str(SHORT_SWEEP_MAX - SHORT_SWEEP_MIN) in problems[0]


def test_a_sweep_just_over_the_minimum_span_passes():
    span = MIN_STEERING_SPAN_COUNTS + 2
    calibration = good_calibration(
        steering_raw_min=0,
        steering_raw_center=span // 2,
        steering_raw_max=span,
    )

    assert validate_calibration(calibration) == []


def test_off_centre_steering_centre_is_caught():
    calibration = good_calibration(steering_raw_center=FULL_STEERING_CENTER + OFF_CENTRE_EXTRA_COUNTS)
    problems = validate_calibration(calibration)

    assert len(problems) == 1
    assert problems_mentioning(problems, "centre")


def test_off_centre_steering_centre_is_caught_on_either_side():
    low = good_calibration(steering_raw_center=FULL_STEERING_CENTER - OFF_CENTRE_EXTRA_COUNTS)

    assert problems_mentioning(validate_calibration(low), "centre")


def test_a_centre_inside_the_tolerance_passes():
    calibration = good_calibration(steering_raw_center=FULL_STEERING_CENTER + MAX_STEERING_CENTER_OFFSET_COUNTS - 1)

    assert validate_calibration(calibration) == []


@pytest.mark.parametrize("pedal", sorted(PEDAL_BYTE_IDX))
def test_a_pedal_with_almost_no_travel_is_caught(pedal):
    calibration = good_calibration(**{f"{pedal}_raw_floor": STUCK_PEDAL_FLOOR})
    problems = validate_calibration(calibration)

    assert problems_mentioning(problems, pedal, "travel")
    assert all(pedal in p for p in problems)


def test_a_brake_far_short_of_the_other_pedals_is_caught():
    calibration = good_calibration(brake_raw_floor=WEAK_BRAKE_FLOOR)
    problems = validate_calibration(calibration)

    assert len(problems) == 1
    assert problems_mentioning(problems, "brake", "progressive")


def test_a_brake_just_inside_the_fraction_passes():
    full_travel = PEDAL_RAW_RELEASED - PEDAL_RAW_FLOORED
    brake_travel = int(MIN_BRAKE_TRAVEL_FRACTION * full_travel) + 2
    calibration = good_calibration(brake_raw_floor=PEDAL_RAW_RELEASED - brake_travel)

    assert validate_calibration(calibration) == []


def test_several_mistakes_are_all_reported():
    calibration = good_calibration(
        steering_raw_min=SHORT_SWEEP_MIN,
        steering_raw_center=SHORT_SWEEP_MIN + 1,
        steering_raw_max=SHORT_SWEEP_MAX,
        brake_raw_floor=WEAK_BRAKE_FLOOR,
    )
    problems = validate_calibration(calibration)

    assert len(problems) == 3
    assert problems_mentioning(problems, "steering", "lock")
    assert problems_mentioning(problems, "centre")
    assert problems_mentioning(problems, "brake", "progressive")


def test_problems_are_plain_sentences():
    problems = validate_calibration(good_calibration(brake_raw_floor=WEAK_BRAKE_FLOOR))

    assert all(isinstance(p, str) and p.strip() and p.endswith(".") for p in problems)


# ---------------------------------------------------------------------------------------------
# capture_calibration, over a fake device and a fake clock.
# ---------------------------------------------------------------------------------------------


class FakeClock:
    """Virtual wall clock. Every reading advances it, so every bounded wait in calibrate.py ends."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        self.now += FAKE_CLOCK_TICK_S
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeWheelDevice:
    """Replays one scripted report queue per capture phase, then goes quiet like a still G920.

    `advance` doubles as the prompt callback, so each phase's reports start when capture_calibration
    asks the operator for that phase and never bleed into the next one.
    """

    def __init__(self, phases):
        self.phases = [list(phase) for phase in phases]
        self.phase_idx = -1
        self.prompts = []

    def advance(self, prompt=""):
        self.prompts.append(prompt)
        self.phase_idx += 1

    def read(self, length):
        if not 0 <= self.phase_idx < len(self.phases) or not self.phases[self.phase_idx]:
            return []
        return list(self.phases[self.phase_idx].pop(0))


def wheel_report(steering_raw, throttle_raw, brake_raw, clutch_raw):
    return bytes(
        [
            G920_REPORT_ID,
            0x0F,
            0x00,
            0x00,
            steering_raw & 0xFF,
            (steering_raw >> 8) & 0xFF,
            throttle_raw,
            brake_raw,
            clutch_raw,
            0x05,
        ]
    )


@pytest.fixture
def fake_clock(monkeypatch):
    from pufferlib.wheel_demo import calibrate

    clock = FakeClock()
    monkeypatch.setattr(calibrate, "time", clock)
    return clock


def test_collect_keeps_only_well_formed_reports(fake_clock):
    good = wheel_report(32_768, 255, 255, 255)
    device = FakeWheelDevice([[good, b"\x01\x02", good, b"", good]])
    device.advance()

    reports = collect(device, 0.05)

    assert reports == [good, good, good]


def test_capture_calibration_builds_the_calibration_the_sweep_implies(fake_clock):
    sweep = [
        wheel_report(1_000, 255, 255, 255),
        wheel_report(64_000, 255, 255, 255),
        wheel_report(32_500, 255, 255, 255),
    ]
    throttle = [wheel_report(32_500, 40, 255, 255), wheel_report(32_500, 255, 255, 255)]
    brake = [wheel_report(32_500, 255, 60, 255), wheel_report(32_500, 255, 255, 255)]
    clutch = [wheel_report(32_500, 255, 255, 10), wheel_report(32_500, 255, 255, 255)]
    device = FakeWheelDevice([sweep, throttle, brake, clutch])
    logs = []

    calibration = capture_calibration(device, prompt=device.advance, log=logs.append)

    assert calibration.steering_raw_min == 1_000
    assert calibration.steering_raw_max == 64_000
    assert calibration.steering_raw_center == (1_000 + 64_000) // 2
    assert calibration.throttle_raw_floor == 40
    assert calibration.brake_raw_floor == 60
    assert calibration.clutch_raw_floor == 10
    assert calibration.throttle_raw_rest == 255
    assert calibration.brake_raw_rest == 255
    assert calibration.clutch_raw_rest == 255
    assert len(device.prompts) == 1 + len(PEDAL_BYTE_IDX)


def test_capture_calibration_takes_pedal_rest_from_every_phase(fake_clock):
    """Rest is the maximum across all phases, so a throttle never released in its own window is
    still rested correctly from the phases where the foot was off it."""
    sweep = [wheel_report(1_000, 200, 255, 255), wheel_report(64_000, 200, 255, 255)]
    throttle = [wheel_report(32_500, 30, 255, 255), wheel_report(32_500, 120, 255, 255)]
    brake = [wheel_report(32_500, 255, 20, 255)]
    clutch = [wheel_report(32_500, 255, 255, 25)]
    device = FakeWheelDevice([sweep, throttle, brake, clutch])

    calibration = capture_calibration(device, prompt=device.advance, log=lambda _: None)

    assert calibration.throttle_raw_rest == 255
    assert calibration.throttle_raw_floor == 30


def test_capture_calibration_refuses_a_silent_steering_sweep(fake_clock):
    device = FakeWheelDevice([[]])

    with pytest.raises(RuntimeError, match="steering sweep"):
        capture_calibration(device, prompt=device.advance, log=lambda _: None)


def test_capture_calibration_refuses_a_silent_pedal_phase(fake_clock):
    sweep = [wheel_report(1_000, 255, 255, 255), wheel_report(64_000, 255, 255, 255)]
    device = FakeWheelDevice([sweep, []])

    with pytest.raises(RuntimeError, match="throttle"):
        capture_calibration(device, prompt=device.advance, log=lambda _: None)


def test_capture_seconds_are_positive():
    assert CAPTURE_SECONDS > 0.0
    assert STEERING_LOW_BYTE < STEERING_HIGH_BYTE < G920_REPORT_LENGTH


# ---------------------------------------------------------------------------------------------
# run_demo.choose_town
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_demo():
    spec = importlib.util.spec_from_file_location("wheel_demo_run_demo", RUN_DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def answers(monkeypatch):
    queued = []

    def fake_input(_prompt=""):
        assert queued, "choose_town asked for more input than the test queued"
        return queued.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return queued


@pytest.mark.parametrize("town", CARLA_TOWNS)
def test_a_named_town_passes_straight_through(run_demo, town, answers):
    assert run_demo.choose_town(town) == town
    assert answers == []


@pytest.mark.parametrize("index,town", list(enumerate(CARLA_TOWNS, start=1)))
def test_every_town_is_reachable_by_number(run_demo, answers, index, town):
    answers.append(str(index))

    assert run_demo.choose_town("ask") == town


def test_empty_input_takes_the_default(run_demo, answers):
    answers.append("")

    assert run_demo.choose_town("ask") == CARLA_TOWNS[-1]


def test_whitespace_only_input_takes_the_default(run_demo, answers):
    answers.append("   ")

    assert run_demo.choose_town("ask") == CARLA_TOWNS[-1]


def test_surrounding_whitespace_is_tolerated(run_demo, answers):
    answers.append("  2  ")

    assert run_demo.choose_town("ask") == CARLA_TOWNS[1]


@pytest.mark.parametrize("bad", ["0", str(len(CARLA_TOWNS) + 1), "99", "Town99", "-1", "2.5", "two"])
def test_invalid_input_reprompts(run_demo, answers, bad):
    answers.extend([bad, "3"])

    assert run_demo.choose_town("ask") == CARLA_TOWNS[2]
    assert answers == []


def test_repeated_invalid_input_keeps_reprompting(run_demo, answers):
    answers.extend(["x", "0", "", ""])

    assert run_demo.choose_town("ask") == CARLA_TOWNS[-1]
    assert len(answers) == 1
