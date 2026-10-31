"""Force feedback (M6), with no wheel attached anywhere.

The priority here is the SIGN CHAIN. This project already shipped one inverted-sign bug that every
test was blind to because the assertions were on an internal angle rather than on observable
motion. Force feedback has the identical exposure and a worse failure mode: a backwards sign makes
the wheel amplify the driver's input and drive itself to full lock against their hands.

So the chain is pinned end to end, in terms a driver would recognise: start from SIM-convention car
state (what `env.get_human_state()` actually reports), run it through the real `car_feel`, the real
`compute_force` and the real `G920ForceFeedback.send`, decode the 16-bit force word that would have
gone down the wire, and assert the wheel gets pushed back TOWARDS CENTRE. An inversion in any link
fails these tests.
"""

import inspect
import math
import os
import threading
import time
from dataclasses import replace

import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.wheel_demo import ffb as ffb_module
from pufferlib.wheel_demo.ffb import (
    ALIGN_TORQUE_GAIN,
    ASSIST_FULL_FORCE,
    ASSIST_FULL_FULL_ERROR_FRACTION,
    ASSIST_NUDGE_FORCE,
    ASSIST_NUDGE_FULL_ERROR_FRACTION,
    DEFAULT_FORCE_GAIN,
    DAMPING_GAIN,
    DEVICE_FORCE_FULL_SCALE,
    FORCE_LIMIT,
    GRIP_FALLOFF,
    HIDPP_DEVICE_INDEX,
    HIDPP_FF_DOWNLOAD_EFFECT,
    HIDPP_REPORT_LONG,
    IMPACT_DECAY_PER_SECOND,
    IMPACT_FORCE,
    IMPACT_WINDOW_SECONDS,
    OFFROAD_RUMBLE_FORCE,
    OFFROAD_RUMBLE_MIN_SPEED_MPS,
    STEADY_FORCE_LIMIT,
    CarFeel,
    ForceFeedbackLoop,
    G920ForceFeedback,
    RecordingForceOutput,
    centring_spring,
    compute_force,
    grip_utilisation,
    saturated_grip_factor,
    self_aligning_torque,
)
from pufferlib.wheel_demo.loop import WheelDemoLoop

STEERING_LIMIT_RAD = binding.HUMAN_STEERING_LIMIT_RAD
TIRE_GRIP_MPS2 = binding.HUMAN_TIRE_GRIP_MPS2

# ---------------------------------------------------------------------------------------------
# Conventions this suite pins. These are deliberately NOT imported from ffb.py: they are the
# external facts the code must agree with, so flipping a constant in the module must fail a test.
# ---------------------------------------------------------------------------------------------

# Sim convention, pinned by tests/drive/test_drive_steering_convention.c: positive curvature and
# therefore positive accel_lat is a LEFT turn, and steering_angle = -steering_norm * limit, so a
# driver holding the wheel RIGHT (steering_norm > 0) shows a NEGATIVE sim steering_angle.
SIM_LEFT = +1
SIM_RIGHT = -1

# Measured on the real G920 (see the DEVICE_FORCE_SIGN comment in ffb.py): a POSITIVE device force
# word turns the wheel towards NEGATIVE steering_norm, i.e. anticlockwise, i.e. left.
POSITIVE_DEVICE_FORCE_TURNS_WHEEL_TOWARDS_NORM = -1

CORNERING_SPEED_MPS = 20.0
CORNERING_ACCEL_LAT_MPS2 = 5.0
PARKING_SPEED_MPS = 1.0
HELD_STEERING_RAD = 0.3

FAKE_FEATURE_INDEX = 0x0D
FAKE_SLOT = 0x01
HIDPP_ROOT_GET_FEATURE_FUNCTION = 0x0A
DOWNLOAD_FORCE_HIGH_BYTE_IDX = 10
DOWNLOAD_FORCE_LOW_BYTE_IDX = 11
REPLY_LENGTH_BYTES = 20
INT16_WRAP = 0x10000
INT16_MAX = 0x7FFF

TEST_FORCE_GAIN = 1.0
LOOP_UPDATE_HZ = 200.0
LOOP_RUN_SECONDS = 0.4
LOOP_RATE_TOLERANCE_LOW = 0.25
LOOP_RATE_TOLERANCE_HIGH = 1.3
THREAD_JOIN_TIMEOUT_S = 2.0


def sign_of(value):
    if value > 0.0:
        return +1
    if value < 0.0:
        return -1
    return 0


# ---------------------------------------------------------------------------------------------
# Fake hardware. start() opens whatever g920.hid_module() returns, so the interception is: swap
# hid_module for a recorder and stub the g920 helpers start() imports. No HID device is ever opened.
# ---------------------------------------------------------------------------------------------


class FakeHidDevice:
    """Records every written frame and answers each request with a plausible HID++ echo."""

    def __init__(self):
        self.written = []
        self.closed = False
        self.reads = 0
        self._replies = []

    def open_path(self, path):
        self.path = path

    def set_nonblocking(self, value):
        self.nonblocking = value

    def write(self, data):
        frame = bytes(data)
        self.written.append(frame)
        reply = [frame[0], HIDPP_DEVICE_INDEX, frame[2], frame[3]] + [0] * (REPLY_LENGTH_BYTES - 4)
        if frame[3] == HIDPP_ROOT_GET_FEATURE_FUNCTION:
            reply[4] = FAKE_FEATURE_INDEX
        elif frame[3] == HIDPP_FF_DOWNLOAD_EFFECT:
            reply[4] = FAKE_SLOT
        self._replies.append(reply)
        return len(frame)

    def read(self, length):
        self.reads += 1
        return self._replies.pop(0) if self._replies else []

    def close(self):
        self.closed = True

    def download_frames(self):
        return [f for f in self.written if f[0] == HIDPP_REPORT_LONG and f[3] == HIDPP_FF_DOWNLOAD_EFFECT]


class FakeHidModule:
    """What g920.hid_module() returns: hands out the one recording device."""

    def __init__(self, device):
        self._device = device

    def device(self):
        return self._device


def decode_device_force(frame):
    """The signed 16-bit constant-force magnitude carried by a DOWNLOAD_EFFECT frame."""
    unsigned = (frame[DOWNLOAD_FORCE_HIGH_BYTE_IDX] << 8) | frame[DOWNLOAD_FORCE_LOW_BYTE_IDX]
    return unsigned - INT16_WRAP if unsigned > INT16_MAX else unsigned


@pytest.fixture
def fake_wheel(monkeypatch):
    """A started G920ForceFeedback whose transport is a recorder, via the real start() path."""
    from pufferlib.wheel_demo.input import g920

    device = FakeHidDevice()
    monkeypatch.setattr(g920, "hid_module", lambda: FakeHidModule(device))
    monkeypatch.setattr(g920, "set_darwin_non_exclusive", lambda: None)
    monkeypatch.setattr(g920, "find_g920_interfaces", lambda: {"vendor": [b"fake-vendor-path"], "axes": b"fake-axes"})
    output = G920ForceFeedback(force_gain=TEST_FORCE_GAIN).start()
    yield output, device
    output.stop()


def sim_assist_state(mode=binding.ASSIST_MODE_OFF, reference_valid=0, reference_steering_rad=0.0):
    """What env.get_assist_state() reports, in SIM signs: positive = LEFT."""
    return {
        "mode": mode,
        "blend_target": 0.0,
        "blend_weight": 0.0,
        "reference_valid": reference_valid,
        "reference_steering_rad": reference_steering_rad,
        "reference_accel_mps2": 0.0,
        "human_steering_rad": 0.0,
        "commanded_steering_rad": 0.0,
    }


class FakeHumanEnv:
    """Stands in for the Drive env: returns one fixed SIM-convention human state."""

    def __init__(self, human_state, assist_state=None):
        self.human_state = human_state
        self.assist_state = assist_state if assist_state is not None else sim_assist_state()

    def get_human_state(self):
        return self.human_state

    def get_assist_state(self):
        return self.assist_state


def sim_human_state(
    steering_angle_rad,
    accel_lat_mps2,
    speed_mps,
    collision=0.0,
    offroad=0.0,
):
    """A human state in the signs the sim actually reports: positive = LEFT."""
    return {
        "steering_angle_rad": steering_angle_rad,
        "accel_lat_mps2": accel_lat_mps2,
        "speed_mps": speed_mps,
        "collision": collision,
        "offroad": offroad,
    }


def feel_from_sim_state(human_state, assist_state=None):
    """Run the real WheelDemoLoop.car_feel over a fake env, with the steering rate settled to zero.

    car_feel is called twice so the second call differences the same angle against itself: the rate
    term then cannot mask the steering and lateral-acceleration signs this suite is pinning.
    """
    loop = WheelDemoLoop.__new__(WheelDemoLoop)
    loop.env = FakeHumanEnv(human_state, assist_state)
    loop._last_steering_angle_rad = 0.0
    loop._last_feel_time_s = time.perf_counter()
    loop.car_feel()
    return loop.car_feel()


def device_force_for_sim_state(output, device, human_state, assist_state=None):
    before = len(device.download_frames())
    output.send(compute_force(feel_from_sim_state(human_state, assist_state), STEERING_LIMIT_RAD))
    frames = device.download_frames()
    assert len(frames) == before + 1
    return decode_device_force(frames[-1])


def wheel_push_direction(device_force):
    """Which way along steering_norm this device force drives the wheel: +1 right, -1 left."""
    return sign_of(device_force) * POSITIVE_DEVICE_FORCE_TURNS_WHEEL_TOWARDS_NORM


def driver_wheel_direction(sim_steering_angle_rad):
    """Where the driver is holding the wheel along steering_norm, given the sim's steering angle."""
    return -sign_of(sim_steering_angle_rad)


# ---------------------------------------------------------------------------------------------
# 1. The sign chain, end to end.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["cornering_left", "cornering_right"])
def test_cornering_pushes_the_wheel_back_towards_centre(fake_wheel, turn):
    """Sim state -> car_feel -> compute_force -> the bytes on the wire. An inversion anywhere fails."""
    output, device = fake_wheel
    human_state = sim_human_state(
        steering_angle_rad=turn * HELD_STEERING_RAD,
        accel_lat_mps2=turn * CORNERING_ACCEL_LAT_MPS2,
        speed_mps=CORNERING_SPEED_MPS,
    )
    device_force = device_force_for_sim_state(output, device, human_state)

    assert device_force != 0
    assert wheel_push_direction(device_force) == -driver_wheel_direction(turn * HELD_STEERING_RAD)


@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["steered_left", "steered_right"])
def test_parking_speed_self_centres_the_wheel(fake_wheel, turn):
    """At parking speed there is no lateral force, so this pins the steering sign alone."""
    output, device = fake_wheel
    human_state = sim_human_state(
        steering_angle_rad=turn * HELD_STEERING_RAD,
        accel_lat_mps2=0.0,
        speed_mps=PARKING_SPEED_MPS,
    )
    device_force = device_force_for_sim_state(output, device, human_state)

    assert device_force != 0
    assert wheel_push_direction(device_force) == -driver_wheel_direction(turn * HELD_STEERING_RAD)


def test_straight_car_with_centred_wheel_sends_zero_force(fake_wheel):
    output, device = fake_wheel
    human_state = sim_human_state(steering_angle_rad=0.0, accel_lat_mps2=0.0, speed_mps=CORNERING_SPEED_MPS)

    assert device_force_for_sim_state(output, device, human_state) == 0


def test_straight_car_with_centred_wheel_computes_zero_force():
    assert compute_force(feel_from_sim_state(sim_human_state(0.0, 0.0, CORNERING_SPEED_MPS)), STEERING_LIMIT_RAD) == 0.0


# Light enough that the spring stays below its saturation point and leaves the clamp headroom.
LIGHT_STEERING_RAD = 0.02
NOMINAL_WHEELBASE_M = 2.8


@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["left", "right"])
def test_wheel_goes_light_once_the_front_lets_go_with_a_light_hand(fake_wheel, turn):
    output, device = fake_wheel
    at_limit = sim_human_state(turn * LIGHT_STEERING_RAD, turn * TIRE_GRIP_MPS2, CORNERING_SPEED_MPS)
    past_limit = sim_human_state(turn * LIGHT_STEERING_RAD, turn * 1.5 * TIRE_GRIP_MPS2, CORNERING_SPEED_MPS)

    force_at_limit = device_force_for_sim_state(output, device, at_limit)
    force_past_limit = device_force_for_sim_state(output, device, past_limit)

    assert abs(force_past_limit) < abs(force_at_limit)
    assert wheel_push_direction(force_past_limit) == -driver_wheel_direction(turn * LIGHT_STEERING_RAD)


@pytest.mark.parametrize("speed_mps", [8.0, 10.0, 15.0])
def test_grip_loss_cue_survives_the_steering_a_real_corner_needs(speed_mps):
    """Kinematic steering for a corner at the grip limit; past it the wheel must weigh less."""
    steering_rad = math.atan(NOMINAL_WHEELBASE_M * TIRE_GRIP_MPS2 / speed_mps**2)
    at_limit = CarFeel(steering_angle_rad=steering_rad, speed_mps=speed_mps, accel_lat_mps2=TIRE_GRIP_MPS2)
    past_limit = replace(at_limit, accel_lat_mps2=1.5 * TIRE_GRIP_MPS2)
    assert abs(compute_force(past_limit, STEERING_LIMIT_RAD)) < abs(compute_force(at_limit, STEERING_LIMIT_RAD))


@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["left", "right"])
def test_wheel_goes_light_once_the_front_lets_go(fake_wheel, turn):
    """The grip-loss cue, felt through the whole chain: past the limit the wheel weighs less."""
    output, device = fake_wheel
    at_limit = sim_human_state(turn * HELD_STEERING_RAD, turn * TIRE_GRIP_MPS2, CORNERING_SPEED_MPS)
    past_limit = sim_human_state(turn * HELD_STEERING_RAD, turn * 1.5 * TIRE_GRIP_MPS2, CORNERING_SPEED_MPS)

    force_at_limit = device_force_for_sim_state(output, device, at_limit)
    force_past_limit = device_force_for_sim_state(output, device, past_limit)

    assert abs(force_past_limit) < abs(force_at_limit)
    assert wheel_push_direction(force_past_limit) == -driver_wheel_direction(turn * HELD_STEERING_RAD)


def test_device_force_scales_with_gain_and_stays_inside_full_scale(fake_wheel):
    output, device = fake_wheel
    output.send(FORCE_LIMIT)
    full = decode_device_force(device.download_frames()[-1])
    output.send(-FORCE_LIMIT)
    full_opposite = decode_device_force(device.download_frames()[-1])

    assert abs(full) == DEVICE_FORCE_FULL_SCALE * TEST_FORCE_GAIN
    assert full_opposite == -full


def test_send_rejects_force_outside_the_device_range(fake_wheel):
    output, _device = fake_wheel
    with pytest.raises(ValueError):
        output.send(1.5)
    with pytest.raises(ValueError):
        output.send(-1.5)


def test_start_resolves_the_feature_index_and_opens_one_effect(fake_wheel):
    output, device = fake_wheel

    assert output._feature_index == FAKE_FEATURE_INDEX
    assert output._slot == FAKE_SLOT
    assert len(device.download_frames()) == 1


def test_stop_closes_the_device(monkeypatch):
    from pufferlib.wheel_demo.input import g920

    device = FakeHidDevice()
    monkeypatch.setattr(g920, "hid_module", lambda: FakeHidModule(device))
    monkeypatch.setattr(g920, "set_darwin_non_exclusive", lambda: None)
    monkeypatch.setattr(g920, "find_g920_interfaces", lambda: {"vendor": [b"fake-vendor-path"]})
    G920ForceFeedback().start().stop()

    assert device.closed


# ---------------------------------------------------------------------------------------------
# 2. self_aligning_torque: the model whose whole point is going light at the limit.
# ---------------------------------------------------------------------------------------------


def test_aligning_torque_is_zero_without_lateral_acceleration():
    assert self_aligning_torque(0.0, TIRE_GRIP_MPS2) == 0.0


@pytest.mark.parametrize("direction", [+1, -1])
def test_aligning_torque_rises_with_utilisation_up_to_the_limit(direction):
    utilisations = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    magnitudes = [abs(self_aligning_torque(direction * u * TIRE_GRIP_MPS2, TIRE_GRIP_MPS2)) for u in utilisations]

    assert magnitudes == sorted(magnitudes)
    assert all(b > a for a, b in zip(magnitudes, magnitudes[1:]))


@pytest.mark.parametrize("direction", [+1, -1])
def test_aligning_torque_falls_away_above_the_grip_limit(direction):
    utilisations = [1.0, 1.1, 1.3, 1.6, 2.0, 2.5]
    magnitudes = [abs(self_aligning_torque(direction * u * TIRE_GRIP_MPS2, TIRE_GRIP_MPS2)) for u in utilisations]

    assert all(b < a for a, b in zip(magnitudes, magnitudes[1:]))


def test_aligning_torque_peaks_exactly_at_the_grip_limit():
    peak = abs(self_aligning_torque(TIRE_GRIP_MPS2, TIRE_GRIP_MPS2))

    assert peak == pytest.approx(ALIGN_TORQUE_GAIN)
    assert abs(self_aligning_torque(0.95 * TIRE_GRIP_MPS2, TIRE_GRIP_MPS2)) < peak
    assert abs(self_aligning_torque(1.05 * TIRE_GRIP_MPS2, TIRE_GRIP_MPS2)) < peak


def test_aligning_torque_collapses_to_zero_far_past_the_limit():
    far_past = (1.0 + 1.0 / GRIP_FALLOFF) * TIRE_GRIP_MPS2

    assert self_aligning_torque(far_past, TIRE_GRIP_MPS2) == pytest.approx(0.0, abs=1e-12)
    assert self_aligning_torque(3.0 * far_past, TIRE_GRIP_MPS2) == pytest.approx(0.0, abs=1e-12)


def test_aligning_torque_never_exceeds_its_gain_and_is_never_nan():
    for step in range(-400, 401):
        accel_lat = 0.05 * step * TIRE_GRIP_MPS2
        torque = self_aligning_torque(accel_lat, TIRE_GRIP_MPS2)
        assert not math.isnan(torque)
        assert abs(torque) <= ALIGN_TORQUE_GAIN


@pytest.mark.parametrize("accel_lat", [0.5, 4.0, 8.5, 12.0, 20.0])
def test_aligning_torque_always_opposes_the_turn(accel_lat):
    assert sign_of(self_aligning_torque(accel_lat, TIRE_GRIP_MPS2)) == -1
    assert sign_of(self_aligning_torque(-accel_lat, TIRE_GRIP_MPS2)) == +1


@pytest.mark.parametrize("grip", [0.0, -1.0, -8.5])
def test_aligning_torque_rejects_non_positive_grip(grip):
    with pytest.raises(ValueError):
        self_aligning_torque(3.0, grip)
    with pytest.raises(ValueError):
        grip_utilisation(3.0, grip)


def test_grip_utilisation_is_one_at_the_limit():
    assert grip_utilisation(TIRE_GRIP_MPS2, TIRE_GRIP_MPS2) == pytest.approx(1.0)
    assert grip_utilisation(-TIRE_GRIP_MPS2, TIRE_GRIP_MPS2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# 3. The force model, pinned against an independent transcription of the contract.
# ---------------------------------------------------------------------------------------------

CONTRACT_ALIGN_TORQUE_GAIN = 0.55
CONTRACT_GRIP_FALLOFF = 0.55
CONTRACT_CENTRING_PEAK_FORCE = 0.7
CONTRACT_CENTRING_SATURATION_FRACTION = 0.2
CONTRACT_CENTRING_AT_SPEED_FRACTION = 0.3
CONTRACT_CENTRING_FADE_SPEED_MPS = 10.0
CONTRACT_DAMPING_GAIN = 0.10
CONTRACT_STEADY_FORCE_LIMIT = 0.75
CONTRACT_NUDGE_FORCE = 0.35
CONTRACT_NUDGE_FULL_ERROR_FRACTION = 0.15
CONTRACT_FULL_FORCE = 0.75
CONTRACT_FULL_FULL_ERROR_FRACTION = 0.04
CONTRACT_OFFROAD_RUMBLE_FORCE = 0.22
CONTRACT_OFFROAD_RUMBLE_HZ = 22.0
CONTRACT_OFFROAD_RUMBLE_MIN_SPEED_MPS = 0.5
CONTRACT_DEFAULT_FORCE_GAIN = 0.7
CENTRING_SPEEDS_MPS = (0.0, 1.0, 6.0, 12.0, 30.0, 60.0, -10.0)
FULL_LOCK_RAD = STEERING_LIMIT_RAD
# sin(2 pi f t) peaks at a quarter period and troughs at three quarters
RUMBLE_PEAK_ELAPSED_S = 1.0 / (4.0 * CONTRACT_OFFROAD_RUMBLE_HZ)
RUMBLE_TROUGH_ELAPSED_S = 3.0 / (4.0 * CONTRACT_OFFROAD_RUMBLE_HZ)


def contract_clamp(value, limit=1.0):
    return max(-limit, min(limit, value))


def contract_grip_factor(utilisation):
    return 1.0 if utilisation <= 1.0 else max(0.0, 1.0 - CONTRACT_GRIP_FALLOFF * (utilisation - 1.0))


def contract_steady(feel, limit_rad):
    utilisation = abs(feel.accel_lat_mps2) / feel.tire_grip_mps2
    grip = contract_grip_factor(utilisation)
    align_magnitude = utilisation if utilisation <= 1.0 else grip
    steady = -math.copysign(CONTRACT_ALIGN_TORQUE_GAIN * align_magnitude, feel.accel_lat_mps2)
    speed_scale = 1.0 - min(1.0, abs(feel.speed_mps) / CONTRACT_CENTRING_FADE_SPEED_MPS) * (
        1.0 - CONTRACT_CENTRING_AT_SPEED_FRACTION
    )
    full = feel.assist_full and feel.assist_target_steering_rad is not None
    spring_centre_rad = feel.assist_target_steering_rad if full else 0.0
    steady += (
        -CONTRACT_CENTRING_PEAK_FORCE
        * speed_scale
        * grip
        * contract_clamp(
            (feel.steering_angle_rad - spring_centre_rad) / (CONTRACT_CENTRING_SATURATION_FRACTION * limit_rad)
        )
    )
    steady += -CONTRACT_DAMPING_GAIN * feel.steering_rate_rad_s
    if feel.assist_target_steering_rad is not None:
        pull = CONTRACT_FULL_FORCE if full else CONTRACT_NUDGE_FORCE
        full_error_fraction = CONTRACT_FULL_FULL_ERROR_FRACTION if full else CONTRACT_NUDGE_FULL_ERROR_FRACTION
        error = (feel.assist_target_steering_rad - feel.steering_angle_rad) / (full_error_fraction * limit_rad)
        steady += pull * contract_clamp(error)
    return steady


def contract_force(feel, limit_rad, impact_age_s=None, elapsed_s=0.0):
    steady = contract_steady(feel, limit_rad)
    force = CONTRACT_STEADY_FORCE_LIMIT * math.tanh(steady / CONTRACT_STEADY_FORCE_LIMIT)
    if feel.offroad and abs(feel.speed_mps) > CONTRACT_OFFROAD_RUMBLE_MIN_SPEED_MPS:
        force += CONTRACT_OFFROAD_RUMBLE_FORCE * math.sin(2.0 * math.pi * CONTRACT_OFFROAD_RUMBLE_HZ * elapsed_s)
    if impact_age_s is not None and impact_age_s >= 0.0:
        direction = -1.0 if steady > 0.0 else 1.0
        force += direction * IMPACT_FORCE * math.exp(-IMPACT_DECAY_PER_SECOND * impact_age_s)
    return contract_clamp(force)


def test_force_model_constants_match_the_contract():
    assert ALIGN_TORQUE_GAIN == CONTRACT_ALIGN_TORQUE_GAIN
    assert GRIP_FALLOFF == CONTRACT_GRIP_FALLOFF
    assert STEADY_FORCE_LIMIT == CONTRACT_STEADY_FORCE_LIMIT
    assert ASSIST_NUDGE_FORCE == CONTRACT_NUDGE_FORCE
    assert ASSIST_NUDGE_FULL_ERROR_FRACTION == CONTRACT_NUDGE_FULL_ERROR_FRACTION
    assert ASSIST_FULL_FORCE == CONTRACT_FULL_FORCE
    assert ASSIST_FULL_FULL_ERROR_FRACTION == CONTRACT_FULL_FULL_ERROR_FRACTION
    assert DAMPING_GAIN == CONTRACT_DAMPING_GAIN
    assert OFFROAD_RUMBLE_FORCE == CONTRACT_OFFROAD_RUMBLE_FORCE
    assert DEFAULT_FORCE_GAIN == CONTRACT_DEFAULT_FORCE_GAIN


def test_the_overspeed_haptic_is_gone():
    assert not [name for name in dir(ffb_module) if "OVERSPEED" in name.upper()]
    assert not hasattr(ffb_module, "speed_excess_mps")
    assert {field for field in CarFeel.__dataclass_fields__} == {
        "steering_angle_rad",
        "steering_rate_rad_s",
        "speed_mps",
        "accel_lat_mps2",
        "tire_grip_mps2",
        "collision",
        "offroad",
        "assist_target_steering_rad",
        "assist_full",
    }
    assert CarFeel().assist_full is False


def test_speeding_no_longer_moves_the_wheel():
    for speed in (5.0, 30.0, 60.0):
        assert rumble_samples(CarFeel(speed_mps=speed)) == [0.0] * RUMBLE_SAMPLE_COUNT


@pytest.mark.parametrize(
    "utilisation,expected",
    [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0 - 0.55 * 0.5), (2.0, 0.45), (1.0 + 1.0 / 0.55, 0.0), (9.0, 0.0)],
)
def test_saturated_grip_factor_follows_the_contract(utilisation, expected):
    assert saturated_grip_factor(utilisation) == pytest.approx(expected, abs=1e-12)


def test_centring_spring_takes_speed_and_grip():
    assert list(inspect.signature(centring_spring).parameters) == [
        "steering_angle_rad",
        "steering_limit_rad",
        "speed_mps",
        "grip_factor",
    ]


@pytest.mark.parametrize("speed", CENTRING_SPEEDS_MPS)
@pytest.mark.parametrize("lock_fraction", [-1.7, -1.0, -0.2, -0.13, -0.02, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.7])
@pytest.mark.parametrize("grip_factor", [1.0, 0.6, 0.0])
def test_centring_spring_follows_the_contract(speed, lock_fraction, grip_factor):
    angle = lock_fraction * STEERING_LIMIT_RAD
    speed_scale = 1.0 - min(1.0, abs(speed) / CONTRACT_CENTRING_FADE_SPEED_MPS) * (1.0 - CONTRACT_CENTRING_AT_SPEED_FRACTION)
    expected = (
        -CONTRACT_CENTRING_PEAK_FORCE
        * speed_scale
        * grip_factor
        * contract_clamp(lock_fraction / CONTRACT_CENTRING_SATURATION_FRACTION)
    )
    assert centring_spring(angle, STEERING_LIMIT_RAD, speed, grip_factor) == pytest.approx(expected, abs=1e-12)


def test_centring_spring_is_full_at_rest_and_eases_to_its_at_speed_share():
    at_rest = centring_spring(STEERING_LIMIT_RAD, STEERING_LIMIT_RAD, 0.0, 1.0)
    at_speed = centring_spring(STEERING_LIMIT_RAD, STEERING_LIMIT_RAD, CONTRACT_CENTRING_FADE_SPEED_MPS, 1.0)
    beyond = centring_spring(STEERING_LIMIT_RAD, STEERING_LIMIT_RAD, 3.0 * CONTRACT_CENTRING_FADE_SPEED_MPS, 1.0)
    assert at_rest == pytest.approx(-CONTRACT_CENTRING_PEAK_FORCE)
    assert at_speed == pytest.approx(-CONTRACT_CENTRING_PEAK_FORCE * CONTRACT_CENTRING_AT_SPEED_FRACTION)
    assert beyond == at_speed
    speeds = [0.5 * i for i in range(30)]
    magnitudes = [abs(centring_spring(0.3, STEERING_LIMIT_RAD, s, 1.0)) for s in speeds]
    assert all(later <= earlier for earlier, later in zip(magnitudes, magnitudes[1:]))


@pytest.mark.parametrize("angle", [0.05, 0.2, 0.55, 1.0])
def test_centring_spring_opposes_the_steering_angle(angle):
    assert sign_of(centring_spring(angle, STEERING_LIMIT_RAD, 5.0, 1.0)) == -1
    assert sign_of(centring_spring(-angle, STEERING_LIMIT_RAD, 5.0, 1.0)) == +1


@pytest.mark.parametrize("limit", [0.0, -0.55])
def test_centring_spring_rejects_non_positive_steering_limit(limit):
    with pytest.raises(ValueError):
        centring_spring(0.1, limit, 0.0, 1.0)


@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["steered_left", "steered_right"])
def test_motorway_speed_still_self_centres_the_wheel(fake_wheel, turn):
    """The spring eases with speed but never vanishes, so a held wheel is pulled back at any speed."""
    output, device = fake_wheel
    human_state = sim_human_state(steering_angle_rad=turn * HELD_STEERING_RAD, accel_lat_mps2=0.0, speed_mps=25.0)
    device_force = device_force_for_sim_state(output, device, human_state)

    assert wheel_push_direction(device_force) == -driver_wheel_direction(turn * HELD_STEERING_RAD)


def contract_sweep_feels():
    for speed in (0.0, 0.4, 3.0, 12.0, 30.0, 60.0):
        for accel_lat in (-40.0, -17.0, -8.5, -2.0, 0.0, 2.0, 8.5, 17.0, 40.0):
            for steering in (-1.2, -0.55, -0.1, 0.0, 0.1, 0.55, 1.2):
                for steering_rate in (-40.0, -5.0, 0.0, 5.0, 40.0):
                    for nudge_target, assist_full in ((None, False), (None, True), (-0.6, False), (0.0, False),
                                                      (0.3, False), (-0.6, True), (0.0, True), (0.3, True)):
                        for offroad in (False, True):
                            yield CarFeel(
                                assist_full=assist_full,
                                steering_angle_rad=steering,
                                steering_rate_rad_s=steering_rate,
                                speed_mps=speed,
                                accel_lat_mps2=accel_lat,
                                tire_grip_mps2=TIRE_GRIP_MPS2,
                                offroad=offroad,
                                assist_target_steering_rad=nudge_target,
                            )


def test_compute_force_matches_the_contract_model_everywhere():
    checked = 0
    for feel in contract_sweep_feels():
        for impact_age in (None, 0.0, 0.05):
            for elapsed in (0.0, RUMBLE_PEAK_ELAPSED_S, 0.013):
                expected = contract_force(feel, STEERING_LIMIT_RAD, impact_age_s=impact_age, elapsed_s=elapsed)
                actual = compute_force(feel, STEERING_LIMIT_RAD, impact_age_s=impact_age, elapsed_s=elapsed)
                assert actual == pytest.approx(expected, abs=1e-12)
                checked += 1
    assert checked > 50_000


def test_compute_force_stays_inside_the_device_range_across_a_wide_sweep():
    checked = 0
    for feel in contract_sweep_feels():
        for impact_age in (None, 0.0, 0.05, 1.0):
            force = compute_force(feel, STEERING_LIMIT_RAD, impact_age_s=impact_age, elapsed_s=0.013)
            assert not math.isnan(force)
            assert -FORCE_LIMIT <= force <= FORCE_LIMIT
            checked += 1
    assert checked > 10_000


def test_steady_force_never_reaches_the_soft_limit():
    for feel in contract_sweep_feels():
        if feel.offroad:
            continue
        assert abs(compute_force(feel, STEERING_LIMIT_RAD)) < STEADY_FORCE_LIMIT


@pytest.mark.parametrize("speed_mps", [5.0, 10.0, 20.0, 30.0])
@pytest.mark.parametrize("lock_fraction", [0.05, 0.3, 1.0])
@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["left", "right"])
def test_grip_loss_lightens_a_held_wheel_monotonically(speed_mps, lock_fraction, turn):
    """Past the limit, more demand must never feel heavier, even with the wheel held at lock."""
    steering_rad = -turn * lock_fraction * STEERING_LIMIT_RAD
    utilisations = [1.0 + 0.05 * i for i in range(60)]
    forces = [
        compute_force(
            CarFeel(
                steering_angle_rad=steering_rad,
                speed_mps=speed_mps,
                accel_lat_mps2=-turn * u * TIRE_GRIP_MPS2,
                tire_grip_mps2=TIRE_GRIP_MPS2,
            ),
            STEERING_LIMIT_RAD,
        )
        for u in utilisations
    ]
    magnitudes = [abs(force) for force in forces]
    gone_idx = next(i for i, u in enumerate(utilisations) if contract_grip_factor(u) == 0.0)
    assert all(later < earlier for earlier, later in zip(magnitudes[: gone_idx + 1], magnitudes[1 : gone_idx + 1]))
    assert all(magnitude == pytest.approx(0.0, abs=1e-12) for magnitude in magnitudes[gone_idx:])
    assert magnitudes[0] - magnitudes[int(0.5 / 0.05)] > 0.05
    assert all(sign_of(force) in (0, turn) for force in forces)


@pytest.mark.parametrize("speed_mps", [2.0, 20.0])
@pytest.mark.parametrize("driver_norm", [+1, -1], ids=["holding_right", "holding_left"])
def test_offroad_rumble_keeps_full_peak_to_peak_at_full_lock(speed_mps, driver_norm):
    loaded = CarFeel(
        steering_angle_rad=driver_norm * FULL_LOCK_RAD,
        speed_mps=speed_mps,
        accel_lat_mps2=driver_norm * TIRE_GRIP_MPS2,
        offroad=True,
        steering_rate_rad_s=driver_norm * 3.0,
    )
    on_road = replace(loaded, offroad=False)
    peak = compute_force(loaded, STEERING_LIMIT_RAD, elapsed_s=RUMBLE_PEAK_ELAPSED_S)
    trough = compute_force(loaded, STEERING_LIMIT_RAD, elapsed_s=RUMBLE_TROUGH_ELAPSED_S)
    steady = compute_force(on_road, STEERING_LIMIT_RAD)
    assert abs(steady) > 0.5
    assert peak - trough == pytest.approx(2.0 * CONTRACT_OFFROAD_RUMBLE_FORCE, abs=1e-9)
    assert peak - steady == pytest.approx(CONTRACT_OFFROAD_RUMBLE_FORCE, abs=1e-9)


@pytest.mark.parametrize("driver_norm", [+1, -1], ids=["holding_right", "holding_left"])
def test_impact_jolt_is_felt_in_full_with_the_wheel_loaded_at_lock(driver_norm):
    loaded = CarFeel(
        steering_angle_rad=driver_norm * FULL_LOCK_RAD,
        speed_mps=CORNERING_SPEED_MPS,
        accel_lat_mps2=driver_norm * TIRE_GRIP_MPS2,
    )
    steady = compute_force(loaded, STEERING_LIMIT_RAD)
    struck = compute_force(loaded, STEERING_LIMIT_RAD, impact_age_s=0.0)
    assert abs(steady) > 0.5
    assert struck - steady == pytest.approx(-sign_of(steady) * IMPACT_FORCE, abs=1e-12)
    assert sign_of(struck) == -sign_of(steady)


NUDGE_SPEED_MPS = 10.0


def nudge_only(target_rad, current_rad=0.0):
    """The nudge's share of the steady force: the same car with and without a target."""
    with_target = CarFeel(steering_angle_rad=current_rad, speed_mps=NUDGE_SPEED_MPS, assist_target_steering_rad=target_rad)
    steady_with = contract_steady(with_target, STEERING_LIMIT_RAD)
    steady_without = contract_steady(replace(with_target, assist_target_steering_rad=None), STEERING_LIMIT_RAD)
    return compute_force(with_target, STEERING_LIMIT_RAD), steady_with - steady_without


def test_no_target_means_no_nudge():
    feel = CarFeel(steering_angle_rad=0.1, speed_mps=NUDGE_SPEED_MPS)
    assert feel.assist_target_steering_rad is None
    assert compute_force(feel, STEERING_LIMIT_RAD) == pytest.approx(contract_force(feel, STEERING_LIMIT_RAD), abs=1e-12)


@pytest.mark.parametrize("current_rad", [-0.3, 0.0, 0.2])
def test_target_where_the_wheel_already_is_adds_nothing(current_rad):
    feel = CarFeel(steering_angle_rad=current_rad, speed_mps=NUDGE_SPEED_MPS)
    assert compute_force(replace(feel, assist_target_steering_rad=current_rad), STEERING_LIMIT_RAD) == compute_force(
        feel, STEERING_LIMIT_RAD
    )


@pytest.mark.parametrize("error_share", [0.1, 0.5, 0.9])
def test_nudge_is_linear_below_its_saturation(error_share):
    error_rad = error_share * CONTRACT_NUDGE_FULL_ERROR_FRACTION * STEERING_LIMIT_RAD
    _, nudge = nudge_only(error_rad)
    assert nudge == pytest.approx(CONTRACT_NUDGE_FORCE * error_share)
    _, nudge_left = nudge_only(-error_rad)
    assert nudge_left == pytest.approx(-CONTRACT_NUDGE_FORCE * error_share)


@pytest.mark.parametrize("error_multiple", [1.0, 2.0, 10.0])
def test_nudge_saturates_at_its_force(error_multiple):
    error_rad = error_multiple * CONTRACT_NUDGE_FULL_ERROR_FRACTION * STEERING_LIMIT_RAD
    assert nudge_only(error_rad)[1] == pytest.approx(CONTRACT_NUDGE_FORCE)
    assert nudge_only(-error_rad)[1] == pytest.approx(-CONTRACT_NUDGE_FORCE)


def test_policy_wanting_further_right_pushes_clockwise():
    """Driver terms: positive is RIGHT and a positive force turns the wheel clockwise."""
    centred = CarFeel(speed_mps=NUDGE_SPEED_MPS)
    wants_right = replace(centred, assist_target_steering_rad=0.2)
    wants_left = replace(centred, assist_target_steering_rad=-0.2)
    assert compute_force(wants_right, STEERING_LIMIT_RAD) > 0.0
    assert compute_force(wants_left, STEERING_LIMIT_RAD) < 0.0


def test_nudge_is_soft_limited_with_the_steady_force():
    held_left = CarFeel(steering_angle_rad=-FULL_LOCK_RAD, speed_mps=0.0, assist_target_steering_rad=FULL_LOCK_RAD)
    force = compute_force(held_left, STEERING_LIMIT_RAD)
    assert force == pytest.approx(CONTRACT_STEADY_FORCE_LIMIT * math.tanh((0.7 + 0.35) / CONTRACT_STEADY_FORCE_LIMIT))
    assert force < STEADY_FORCE_LIMIT


def nudge_assist(reference_sim_rad, mode=binding.ASSIST_MODE_NUDGE, reference_valid=1):
    return sim_assist_state(mode=mode, reference_valid=reference_valid, reference_steering_rad=reference_sim_rad)


def test_car_feel_hands_the_policy_angle_over_in_driver_terms_only_when_nudging_or_full():
    straight = sim_human_state(0.0, 0.0, NUDGE_SPEED_MPS)
    policy_left_sim = SIM_LEFT * 0.12
    for mode in (binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL):
        feel = feel_from_sim_state(straight, nudge_assist(policy_left_sim, mode=mode))
        assert feel.assist_target_steering_rad == -policy_left_sim
        assert feel.assist_full == (mode == binding.ASSIST_MODE_FULL)
        invalid = nudge_assist(policy_left_sim, mode=mode, reference_valid=0)
        assert feel_from_sim_state(straight, invalid).assist_target_steering_rad is None
    for mode in (binding.ASSIST_MODE_OFF, binding.ASSIST_MODE_DISPLAY, binding.ASSIST_MODE_BLEND):
        feel = feel_from_sim_state(straight, nudge_assist(policy_left_sim, mode=mode))
        assert feel.assist_target_steering_rad is None
        assert feel.assist_full is False


FULL_SPEEDS_MPS = (0.0, 3.0, NUDGE_SPEED_MPS, 30.0)
FULL_TARGETS_RAD = (-0.4, -0.05, 0.0, 0.12, 0.4)


def full_feel(target_rad, current_rad, speed_mps=NUDGE_SPEED_MPS):
    return CarFeel(steering_angle_rad=current_rad, speed_mps=speed_mps, assist_target_steering_rad=target_rad,
                   assist_full=True)


@pytest.mark.parametrize("speed_mps", FULL_SPEEDS_MPS)
@pytest.mark.parametrize("target_rad", FULL_TARGETS_RAD)
def test_full_holds_the_wheel_at_the_target_with_zero_force(speed_mps, target_rad):
    assert compute_force(full_feel(target_rad, target_rad, speed_mps), STEERING_LIMIT_RAD) == 0.0


def test_full_without_a_target_is_the_plain_wheel():
    plain = CarFeel(steering_angle_rad=0.2, speed_mps=NUDGE_SPEED_MPS)
    assert compute_force(replace(plain, assist_full=True), STEERING_LIMIT_RAD) == compute_force(plain, STEERING_LIMIT_RAD)


@pytest.mark.parametrize("speed_mps", FULL_SPEEDS_MPS)
@pytest.mark.parametrize("target_rad", FULL_TARGETS_RAD)
@pytest.mark.parametrize("offset_rad", [0.001, 0.01, 0.1, 0.5])
def test_full_pulls_toward_the_target_from_both_sides(speed_mps, target_rad, offset_rad):
    left_of_target = compute_force(full_feel(target_rad, target_rad - offset_rad, speed_mps), STEERING_LIMIT_RAD)
    right_of_target = compute_force(full_feel(target_rad, target_rad + offset_rad, speed_mps), STEERING_LIMIT_RAD)
    assert left_of_target > 0.0
    assert right_of_target < 0.0


@pytest.mark.parametrize("error_lock_fraction", [0.01, 0.04, 0.1, 0.3])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_full_pulls_harder_than_nudge_at_the_same_error(error_lock_fraction, direction):
    target_rad = direction * error_lock_fraction * STEERING_LIMIT_RAD
    full = compute_force(full_feel(target_rad, 0.0), STEERING_LIMIT_RAD)
    nudge = compute_force(replace(full_feel(target_rad, 0.0), assist_full=False), STEERING_LIMIT_RAD)
    assert sign_of(full) == sign_of(nudge) == sign_of(direction)
    assert abs(full) > abs(nudge)


@pytest.mark.parametrize("error_share", [0.25, 0.5, 1.0, 3.0])
def test_full_pull_saturates_at_its_force_within_a_few_percent_of_lock(error_share):
    error_rad = error_share * CONTRACT_FULL_FULL_ERROR_FRACTION * STEERING_LIMIT_RAD
    held = CarFeel(speed_mps=NUDGE_SPEED_MPS, assist_full=True)
    with_target = contract_steady(replace(held, assist_target_steering_rad=error_rad), STEERING_LIMIT_RAD)
    spring_only = contract_steady(replace(held, steering_angle_rad=-error_rad), STEERING_LIMIT_RAD)
    assert with_target - spring_only == pytest.approx(CONTRACT_FULL_FORCE * min(1.0, error_share))


def test_full_spring_is_centred_on_the_target_not_on_zero():
    target_rad = 0.3
    nudge_at_target = CarFeel(steering_angle_rad=target_rad, speed_mps=NUDGE_SPEED_MPS,
                              assist_target_steering_rad=target_rad)
    assert compute_force(nudge_at_target, STEERING_LIMIT_RAD) < 0.0
    assert compute_force(replace(nudge_at_target, assist_full=True), STEERING_LIMIT_RAD) == 0.0


@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["policy_left", "policy_right"])
def test_full_turns_the_real_wheel_towards_the_policy(fake_wheel, turn):
    output, device = fake_wheel
    straight = sim_human_state(0.0, 0.0, NUDGE_SPEED_MPS)
    full_force = device_force_for_sim_state(output, device, straight, nudge_assist(turn * 0.2, mode=binding.ASSIST_MODE_FULL))
    nudge_force = device_force_for_sim_state(output, device, straight, nudge_assist(turn * 0.2))
    assert wheel_push_direction(full_force) == driver_wheel_direction(turn * 0.2)
    assert abs(full_force) > abs(nudge_force)


@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["policy_left", "policy_right"])
def test_nudge_turns_the_real_wheel_towards_the_policy(fake_wheel, turn):
    """Sim reference -> car_feel -> compute_force -> wire: the wheel moves the way the policy steers."""
    output, device = fake_wheel
    straight = sim_human_state(0.0, 0.0, NUDGE_SPEED_MPS)
    device_force = device_force_for_sim_state(output, device, straight, nudge_assist(turn * 0.2))
    assert device_force != 0
    assert wheel_push_direction(device_force) == driver_wheel_direction(turn * 0.2)


@pytest.mark.parametrize("steering_rate", [0.5, 2.0, 8.0])
def test_damping_opposes_the_steering_rate(steering_rate):
    turning_one_way = compute_force(CarFeel(steering_rate_rad_s=steering_rate), STEERING_LIMIT_RAD)
    turning_the_other = compute_force(CarFeel(steering_rate_rad_s=-steering_rate), STEERING_LIMIT_RAD)
    soft = CONTRACT_STEADY_FORCE_LIMIT * math.tanh(DAMPING_GAIN * steering_rate / CONTRACT_STEADY_FORCE_LIMIT)

    assert turning_one_way == pytest.approx(-soft)
    assert turning_the_other == pytest.approx(soft)


def test_offroad_threshold_is_unchanged():
    assert OFFROAD_RUMBLE_MIN_SPEED_MPS == CONTRACT_OFFROAD_RUMBLE_MIN_SPEED_MPS
    just_above = CarFeel(speed_mps=CONTRACT_OFFROAD_RUMBLE_MIN_SPEED_MPS + 0.01, offroad=True)
    assert_oscillates(rumble_samples(just_above))


def test_offroad_rumble_oscillates():
    feel = CarFeel(speed_mps=10.0, offroad=True)
    samples = [compute_force(feel, STEERING_LIMIT_RAD, elapsed_s=0.001 * i) for i in range(200)]

    assert max(samples) > 0.0
    assert min(samples) < 0.0
    assert sum(1 for a, b in zip(samples, samples[1:]) if sign_of(a) != sign_of(b)) >= 2


@pytest.mark.parametrize("speed", [0.0, 0.25, 0.5])
def test_offroad_rumble_is_absent_below_the_speed_threshold(speed):
    on_road = CarFeel(speed_mps=speed, offroad=False)
    off_road = CarFeel(speed_mps=speed, offroad=True)
    for elapsed in (0.0, 0.007, 0.011, 0.02):
        assert compute_force(off_road, STEERING_LIMIT_RAD, elapsed_s=elapsed) == compute_force(
            on_road, STEERING_LIMIT_RAD, elapsed_s=elapsed
        )


def test_collision_pulse_decays_monotonically():
    feel = CarFeel(speed_mps=10.0)
    ages = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    magnitudes = [abs(compute_force(feel, STEERING_LIMIT_RAD, impact_age_s=age)) for age in ages]

    assert all(b <= a for a, b in zip(magnitudes, magnitudes[1:]))
    assert all(b < a for a, b in zip(magnitudes[1:], magnitudes[2:]))
    assert magnitudes[0] > 0.0
    assert magnitudes[-1] < 0.01


def test_no_collision_pulse_when_no_impact_age_is_given():
    feel = CarFeel(speed_mps=10.0)

    assert compute_force(feel, STEERING_LIMIT_RAD, impact_age_s=None) == 0.0


# ---------------------------------------------------------------------------------------------
# 5. RecordingForceOutput.
# ---------------------------------------------------------------------------------------------


def test_recording_output_rejects_send_before_start():
    output = RecordingForceOutput()
    with pytest.raises(RuntimeError):
        output.send(0.1)


def test_recording_output_rejects_send_after_stop():
    output = RecordingForceOutput().start()
    output.send(0.1)
    output.stop()
    with pytest.raises(RuntimeError):
        output.send(0.1)


@pytest.mark.parametrize("force", [1.01, -1.01, 5.0, float("nan"), float("inf")])
def test_recording_output_rejects_out_of_range_force(force):
    output = RecordingForceOutput().start()
    with pytest.raises(ValueError):
        output.send(force)


@pytest.mark.parametrize("force", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_recording_output_keeps_in_range_force(force):
    output = RecordingForceOutput().start()
    output.send(force)

    assert output.sent == [force]


# ---------------------------------------------------------------------------------------------
# 6. ForceFeedbackLoop.
# ---------------------------------------------------------------------------------------------


def test_loop_drives_the_output_at_roughly_the_configured_rate():
    output = RecordingForceOutput()
    loop = ForceFeedbackLoop(output, lambda: CarFeel(speed_mps=10.0), STEERING_LIMIT_RAD, update_hz=LOOP_UPDATE_HZ)
    loop.start()
    time.sleep(LOOP_RUN_SECONDS)
    loop.stop()

    nominal = LOOP_UPDATE_HZ * LOOP_RUN_SECONDS
    assert loop.error is None
    assert LOOP_RATE_TOLERANCE_LOW * nominal <= len(output.sent) <= LOOP_RATE_TOLERANCE_HIGH * nominal


def test_loop_skips_a_none_feel_without_crashing():
    output = RecordingForceOutput()
    loop = ForceFeedbackLoop(output, lambda: None, STEERING_LIMIT_RAD, update_hz=LOOP_UPDATE_HZ)
    loop.start()
    time.sleep(LOOP_RUN_SECONDS / 4.0)
    loop.stop()

    assert loop.error is None
    assert output.sent == []


def test_loop_sends_the_force_the_model_computed():
    feel = CarFeel(steering_angle_rad=0.3, speed_mps=0.0)
    output = RecordingForceOutput()
    loop = ForceFeedbackLoop(output, lambda: feel, STEERING_LIMIT_RAD, update_hz=LOOP_UPDATE_HZ)
    loop.start()
    time.sleep(LOOP_RUN_SECONDS / 4.0)
    loop.stop()

    assert output.sent
    assert all(sent == pytest.approx(compute_force(feel, STEERING_LIMIT_RAD)) for sent in output.sent)


def test_loop_parks_a_thread_exception_in_error_and_stops():
    class ExplodingOutput(RecordingForceOutput):
        def send(self, force):
            raise RuntimeError("wheel unplugged")

    output = ExplodingOutput()
    loop = ForceFeedbackLoop(output, lambda: CarFeel(speed_mps=10.0), STEERING_LIMIT_RAD, update_hz=LOOP_UPDATE_HZ)
    loop.start()
    loop._thread.join(timeout=THREAD_JOIN_TIMEOUT_S)

    assert not loop._thread.is_alive()
    assert isinstance(loop.error, RuntimeError)
    assert str(loop.error) == "wheel unplugged"
    assert loop._running is False


def test_loop_parks_a_feel_source_exception_in_error():
    def exploding_feel():
        raise ValueError("env went away")

    loop = ForceFeedbackLoop(RecordingForceOutput(), exploding_feel, STEERING_LIMIT_RAD, update_hz=LOOP_UPDATE_HZ)
    loop.start()
    loop._thread.join(timeout=THREAD_JOIN_TIMEOUT_S)

    assert isinstance(loop.error, ValueError)
    assert loop._running is False


def test_loop_stop_joins_the_thread_and_stops_the_output():
    output = RecordingForceOutput()
    loop = ForceFeedbackLoop(output, lambda: CarFeel(speed_mps=10.0), STEERING_LIMIT_RAD, update_hz=LOOP_UPDATE_HZ)
    loop.start()
    assert output.started
    threads_before = {t.name for t in threading.enumerate()}
    assert "ffb" in threads_before
    loop.stop()

    assert not loop._thread.is_alive()
    assert output.started is False


def test_loop_starts_the_output_before_sending():
    class StartOrderOutput(RecordingForceOutput):
        def __init__(self):
            super().__init__()
            self.sends_before_start = 0

        def send(self, force):
            if not self.started:
                self.sends_before_start += 1
            super().send(force)

    output = StartOrderOutput()
    loop = ForceFeedbackLoop(output, lambda: CarFeel(speed_mps=10.0), STEERING_LIMIT_RAD, update_hz=LOOP_UPDATE_HZ)
    loop.start()
    time.sleep(LOOP_RUN_SECONDS / 4.0)
    loop.stop()

    assert output.sends_before_start == 0


# ---------------------------------------------------------------------------------------------
# 7. G920ForceFeedback construction.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gain", [0.0, -0.1, -1.0, 1.0001, 2.0])
def test_g920_rejects_a_force_gain_outside_the_unit_interval(gain):
    with pytest.raises(ValueError):
        G920ForceFeedback(force_gain=gain)


@pytest.mark.parametrize("gain", [0.001, 0.55, 1.0])
def test_g920_accepts_a_force_gain_inside_the_unit_interval(gain):
    assert G920ForceFeedback(force_gain=gain).force_gain == gain


def test_g920_construction_opens_no_device():
    output = G920ForceFeedback()

    assert output._device is None
    assert output._feature_index is None
    assert output._slot is None


# ---------------------------------------------------------------------------------------------
# 8. car_feel, in its own right.
# ---------------------------------------------------------------------------------------------


def test_car_feel_returns_none_when_nobody_is_driving():
    loop = WheelDemoLoop.__new__(WheelDemoLoop)
    loop.env = FakeHumanEnv(None)
    loop._last_steering_angle_rad = 0.0
    loop._last_feel_time_s = time.perf_counter()

    assert loop.car_feel() is None


def test_car_feel_passes_speed_and_grip_through_unchanged():
    feel = feel_from_sim_state(sim_human_state(0.0, 0.0, CORNERING_SPEED_MPS))

    assert feel.speed_mps == CORNERING_SPEED_MPS
    assert feel.tire_grip_mps2 == TIRE_GRIP_MPS2


def test_car_feel_reports_collision():
    assert feel_from_sim_state(sim_human_state(0.0, 0.0, 5.0, collision=1.0)).collision is True
    assert feel_from_sim_state(sim_human_state(0.0, 0.0, 5.0, collision=0.0)).collision is False


def test_car_feel_steering_rate_follows_the_steering_angle():
    """A wheel turned one way between two samples must report a rate pointing the same way."""
    loop = WheelDemoLoop.__new__(WheelDemoLoop)
    loop.env = FakeHumanEnv(sim_human_state(0.0, 0.0, 5.0))
    loop._last_steering_angle_rad = 0.0
    loop._last_feel_time_s = time.perf_counter()
    loop.car_feel()
    loop.env.human_state = sim_human_state(SIM_RIGHT * HELD_STEERING_RAD, 0.0, 5.0)
    turning_right = loop.car_feel()

    loop.env.human_state = sim_human_state(0.0, 0.0, 5.0)
    loop.car_feel()
    loop.env.human_state = sim_human_state(SIM_LEFT * HELD_STEERING_RAD, 0.0, 5.0)
    turning_left = loop.car_feel()

    assert sign_of(turning_right.steering_rate_rad_s) == sign_of(turning_right.steering_angle_rad)
    assert sign_of(turning_left.steering_rate_rad_s) == sign_of(turning_left.steering_angle_rad)
    assert sign_of(turning_right.steering_rate_rad_s) == -sign_of(turning_left.steering_rate_rad_s)


# ---------------------------------------------------------------------------------------------
# 9. The collision pulse through the LIVE loop.
#
# compute_force only ever sees an impact age; deciding WHEN an impact started is the loop's job,
# and it was once not done at all, so the pulse never fired on a real car. These run the real
# ForceFeedbackLoop thread over a scripted feel source and read what reached the output.
# ---------------------------------------------------------------------------------------------

FRAME_SECONDS = 1.0 / LOOP_UPDATE_HZ
# compute_force is in DRIVER signs, like car_feel: a positive force pushes the wheel clockwise,
# i.e. towards positive steering_norm, i.e. RIGHT. With the wheel centred the jolt kicks right.
JOLT_AT_CENTRE_TURNS_WHEEL_TOWARDS_NORM = +1
SEND_BURST_FRAMES = 50


class FakeLoopClock:
    """Stands in for the `time` the loop uses. The loop's own sleep is what advances it, so a
    scripted run covers an exact span of virtual seconds with no wall-clock waiting."""

    def __init__(self):
        self.now_s = 0.0

    def monotonic(self):
        return self.now_s

    def sleep(self, seconds):
        self.now_s += seconds


class ScriptedFeelSource:
    """Hands the loop one CarFeel per iteration, then goes quiet and signals that it is done."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.next_frame_idx = 0
        self.exhausted = threading.Event()

    def __call__(self):
        if self.next_frame_idx >= len(self.frames):
            self.exhausted.set()
            return None
        feel = self.frames[self.next_frame_idx]
        self.next_frame_idx += 1
        return feel


def run_scripted_loop(monkeypatch, frames):
    """The forces the real ForceFeedbackLoop thread produces for `frames`, one per iteration."""
    from pufferlib.wheel_demo import ffb as ffb_module

    monkeypatch.setattr(ffb_module, "time", FakeLoopClock())
    source = ScriptedFeelSource(frames)
    output = RecordingForceOutput()
    loop = ForceFeedbackLoop(output, source, STEERING_LIMIT_RAD, update_hz=LOOP_UPDATE_HZ)
    loop.start()
    assert source.exhausted.wait(timeout=THREAD_JOIN_TIMEOUT_S)
    loop.stop()

    assert loop.error is None
    assert len(output.sent) == len(frames)
    return output.sent


# Cruising straight and level: every steady term is zero, so anything non-zero is the pulse.
QUIET_FRAME = CarFeel(speed_mps=CORNERING_SPEED_MPS)
STRUCK_FRAME = CarFeel(speed_mps=CORNERING_SPEED_MPS, collision=True)


def test_loop_sends_no_impact_pulse_before_any_collision(monkeypatch):
    assert run_scripted_loop(monkeypatch, [QUIET_FRAME] * 20) == [0.0] * 20


def test_loop_fires_a_full_pulse_on_the_frame_the_collision_rises(monkeypatch):
    sent = run_scripted_loop(monkeypatch, [QUIET_FRAME] * 5 + [STRUCK_FRAME] + [QUIET_FRAME] * 5)

    assert sent[:5] == [0.0] * 5
    assert sent[5] == pytest.approx(IMPACT_FORCE)
    assert sign_of(sent[5]) == JOLT_AT_CENTRE_TURNS_WHEEL_TOWARDS_NORM


def test_loop_pulse_decays_from_the_impact_frame_on_the_loop_clock(monkeypatch):
    sent = run_scripted_loop(monkeypatch, [QUIET_FRAME] + [STRUCK_FRAME] + [QUIET_FRAME] * 20)
    pulse = sent[1:]

    assert all(later < earlier for earlier, later in zip(pulse, pulse[1:]))
    for frames_since_impact, force in enumerate(pulse):
        age_s = frames_since_impact * FRAME_SECONDS
        assert force == pytest.approx(IMPACT_FORCE * math.exp(-IMPACT_DECAY_PER_SECOND * age_s))


def test_loop_stops_the_pulse_once_the_impact_window_has_passed(monkeypatch):
    frames_in_window = int(IMPACT_WINDOW_SECONDS / FRAME_SECONDS)
    sent = run_scripted_loop(monkeypatch, [STRUCK_FRAME] * (frames_in_window + 5))

    assert sent[0] == pytest.approx(IMPACT_FORCE)
    assert sent[frames_in_window - 5] != 0.0
    assert sent[frames_in_window + 2 :] == [0.0] * 3


def test_loop_does_not_retrigger_while_one_collision_stays_latched(monkeypatch):
    """A collision flag held True for many frames is ONE impact, so it must feel identical to a
    flag that rose for a single frame and cleared."""
    latched = run_scripted_loop(monkeypatch, [QUIET_FRAME] + [STRUCK_FRAME] * 20)
    single_frame = run_scripted_loop(monkeypatch, [QUIET_FRAME] + [STRUCK_FRAME] + [QUIET_FRAME] * 19)

    assert latched[1] == pytest.approx(IMPACT_FORCE)
    assert latched == single_frame


def test_loop_retriggers_the_pulse_for_a_second_separate_collision(monkeypatch):
    sent = run_scripted_loop(
        monkeypatch,
        [QUIET_FRAME] * 2 + [STRUCK_FRAME] * 2 + [QUIET_FRAME] * 2 + [STRUCK_FRAME] * 2 + [QUIET_FRAME] * 4,
    )

    assert sent[:2] == [0.0, 0.0]
    assert sent[2] == pytest.approx(IMPACT_FORCE)
    first_pulse = sent[2:6]
    assert all(later < earlier for earlier, later in zip(first_pulse, first_pulse[1:]))
    assert sent[6] == pytest.approx(IMPACT_FORCE)
    assert sent[6] > sent[5]
    second_pulse = sent[6:]
    assert all(later < earlier for earlier, later in zip(second_pulse, second_pulse[1:]))


def test_loop_retriggers_after_the_impact_window_has_closed(monkeypatch):
    """The window reset must re-arm the edge detector, not disable it for the rest of the drive."""
    frames_in_window = int(IMPACT_WINDOW_SECONDS / FRAME_SECONDS)
    sent = run_scripted_loop(
        monkeypatch,
        [STRUCK_FRAME] + [QUIET_FRAME] * (frames_in_window + 4) + [STRUCK_FRAME] + [QUIET_FRAME] * 3,
    )
    second_impact_idx = frames_in_window + 5

    assert sent[0] == pytest.approx(IMPACT_FORCE)
    assert sent[second_impact_idx - 1] == 0.0
    assert sent[second_impact_idx] == pytest.approx(IMPACT_FORCE)


def test_loop_adds_the_pulse_on_top_of_the_steady_force(monkeypatch):
    """Damping gives a non-zero steady force the jolt must add to rather than replace."""
    turning = CarFeel(steering_rate_rad_s=2.0, speed_mps=CORNERING_SPEED_MPS)
    turning_and_struck = CarFeel(steering_rate_rad_s=2.0, speed_mps=CORNERING_SPEED_MPS, collision=True)
    sent = run_scripted_loop(monkeypatch, [turning, turning_and_struck])

    soft_damping = -STEADY_FORCE_LIMIT * math.tanh(DAMPING_GAIN * 2.0 / STEADY_FORCE_LIMIT)
    assert sent[0] == pytest.approx(soft_damping)
    assert sent[1] == pytest.approx(soft_damping + IMPACT_FORCE)


@pytest.mark.parametrize("driver_norm", [+1, -1], ids=["holding_right", "holding_left"])
@pytest.mark.parametrize("steering_rad", [LIGHT_STEERING_RAD, HELD_STEERING_RAD])
def test_impact_jolt_kicks_against_the_steady_force(driver_norm, steering_rad):
    """car_feel is in driver signs: a held-right wheel is centred leftwards, so the jolt kicks right."""
    feel = CarFeel(steering_angle_rad=driver_norm * steering_rad, speed_mps=CORNERING_SPEED_MPS)
    steady = compute_force(feel, STEERING_LIMIT_RAD)
    jolt = compute_force(feel, STEERING_LIMIT_RAD, impact_age_s=0.0) - steady

    assert sign_of(steady) == -driver_norm
    assert sign_of(jolt) == driver_norm
    assert jolt == pytest.approx(driver_norm * IMPACT_FORCE)


def test_impact_jolt_treats_both_signed_zeroes_as_a_centred_wheel():
    """-0.0 is a real float the sim can report, and it is not the driver holding left."""
    centred = CarFeel(steering_angle_rad=0.0, speed_mps=CORNERING_SPEED_MPS)
    negative_zero = CarFeel(steering_angle_rad=-0.0, speed_mps=CORNERING_SPEED_MPS)

    assert compute_force(centred, STEERING_LIMIT_RAD, impact_age_s=0.0) == pytest.approx(IMPACT_FORCE)
    assert compute_force(negative_zero, STEERING_LIMIT_RAD, impact_age_s=0.0) == pytest.approx(IMPACT_FORCE)


# ---------------------------------------------------------------------------------------------
# 10. Offroad, from the sim's own flag to a rumbling wheel.
#
# This path was dead: car_feel hardcoded offroad=False and the binding exported no offroad field,
# so no state the sim could reach ever produced a rumble.
# ---------------------------------------------------------------------------------------------

RUMBLE_SAMPLE_COUNT = 200
RUMBLE_SAMPLE_SECONDS = 0.001
OFFROAD_MAP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pufferlib",
    "resources",
    "drive",
    "binaries",
    "sdc_replay_test",
)
OFFROAD_SCENARIO_LENGTH = 400
OFFROAD_MAX_STEPS = 120
FULL_LOCK_STEERING_NORM = 1.0
FULL_THROTTLE = 1.0


def rumble_samples(feel):
    return [
        compute_force(feel, STEERING_LIMIT_RAD, elapsed_s=RUMBLE_SAMPLE_SECONDS * i)
        for i in range(RUMBLE_SAMPLE_COUNT)
    ]


def rumble_component(feel):
    """The offroad contribution alone: the same car state sampled with and without the flag.

    A car far enough off the road to trip the flag is usually also cornering hard, and that steady
    force is larger than the rumble, so the rumble shows up as a wobble about it rather than as a
    force that changes sign.
    """
    on_road = replace(feel, offroad=False)
    return [off - on for off, on in zip(rumble_samples(feel), rumble_samples(on_road))]


def assert_oscillates(samples):
    assert max(samples) > 0.0
    assert min(samples) < 0.0
    assert sum(1 for a, b in zip(samples, samples[1:]) if sign_of(a) != sign_of(b)) >= 2


def test_car_feel_reports_offroad():
    assert feel_from_sim_state(sim_human_state(0.0, 0.0, 5.0, offroad=1.0)).offroad is True
    assert feel_from_sim_state(sim_human_state(0.0, 0.0, 5.0, offroad=0.0)).offroad is False


def test_offroad_sim_state_rumbles_the_wheel():
    """Sim state -> car_feel -> compute_force, with the on-road state as the control."""
    off_road = feel_from_sim_state(sim_human_state(0.0, 0.0, CORNERING_SPEED_MPS, offroad=1.0))
    on_road = feel_from_sim_state(sim_human_state(0.0, 0.0, CORNERING_SPEED_MPS, offroad=0.0))

    assert rumble_samples(on_road) == [0.0] * RUMBLE_SAMPLE_COUNT
    assert_oscillates(rumble_samples(off_road))


def test_loop_rumbles_on_its_own_clock_when_the_car_is_offroad(monkeypatch):
    sent = run_scripted_loop(monkeypatch, [CarFeel(speed_mps=CORNERING_SPEED_MPS, offroad=True)] * 40)

    assert_oscillates(sent)
    assert all(abs(force) <= OFFROAD_RUMBLE_FORCE for force in sent)


def test_offroad_rumble_reaches_the_wire(fake_wheel):
    output, device = fake_wheel
    off_road = feel_from_sim_state(sim_human_state(0.0, 0.0, CORNERING_SPEED_MPS, offroad=1.0))
    device_forces = []
    for i in range(RUMBLE_SAMPLE_COUNT):
        output.send(compute_force(off_road, STEERING_LIMIT_RAD, elapsed_s=RUMBLE_SAMPLE_SECONDS * i))
        device_forces.append(decode_device_force(device.download_frames()[-1]))

    assert max(device_forces) > 0
    assert min(device_forces) < 0


def test_a_real_car_driven_off_the_road_rumbles_the_wheel():
    """The whole offroad path on the real C sim: drive agent 0 off the road at full lock, then run
    the state the binding reports through the real car_feel and the real force model."""
    env = Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=OFFROAD_MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=OFFROAD_SCENARIO_LENGTH,
        resample_frequency=1_000_000,
        termination_mode=False,
        goal_source="gt",
        report_interval=1,
        num_goals=3,
        goal_radius=2.0,
        reset_accel_on_stop=True,
    )
    try:
        env.reset()
        env.select_human_agent(0)
        env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
        assert env.get_human_state()["offroad"] == 0.0

        feel_loop = WheelDemoLoop.__new__(WheelDemoLoop)
        feel_loop.env = env
        feel_loop._last_steering_angle_rad = 0.0
        feel_loop._last_feel_time_s = time.perf_counter()

        env.set_human_pedals(FULL_LOCK_STEERING_NORM, FULL_THROTTLE, 0.0, 0.0)
        no_action = env.action_space.sample() * 0
        off_road = None
        for _ in range(OFFROAD_MAX_STEPS):
            env.step(no_action)
            feel = feel_loop.car_feel()
            if feel.offroad:
                off_road = feel
                break

        assert off_road is not None, "full lock and full throttle never took the car off the road"
        assert off_road.speed_mps > 1.0, "the rumble needs the car to be moving"
        assert_oscillates(rumble_component(off_road))
        assert rumble_samples(replace(off_road, offroad=False)) == [
            compute_force(replace(off_road, offroad=False), STEERING_LIMIT_RAD)
        ] * RUMBLE_SAMPLE_COUNT
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------
# 11. The HID transport: started before use, and never waiting on a reply per frame.
# ---------------------------------------------------------------------------------------------


def test_g920_send_before_start_raises():
    with pytest.raises(RuntimeError):
        G920ForceFeedback().send(0.5)


def test_g920_started_guard_fires_before_the_range_check():
    """Out-of-range on an unopened device is still a use-before-start, not a bad force."""
    with pytest.raises(RuntimeError):
        G920ForceFeedback().send(5.0)


def test_g920_send_after_stop_raises(fake_wheel):
    output, _device = fake_wheel
    output.send(0.5)
    output.stop()

    with pytest.raises(RuntimeError):
        output.send(0.5)


def test_g920_send_raises_when_no_effect_slot_is_open(fake_wheel):
    """An open device whose effect download never came back has no magnitude to rewrite."""
    output, _device = fake_wheel
    output._slot = None

    with pytest.raises(RuntimeError):
        output.send(0.5)


def test_send_does_not_wait_for_a_device_reply(fake_wheel):
    """At 200 Hz a dropped reply would stall the wheel for a whole HIDPP_REPLY_TIMEOUT_S."""
    output, device = fake_wheel
    reads_before = device.reads
    frames_before = len(device.download_frames())
    for _ in range(SEND_BURST_FRAMES):
        output.send(0.5)

    assert device.reads == reads_before
    assert len(device.download_frames()) == frames_before + SEND_BURST_FRAMES
    assert wheel_push_direction(decode_device_force(device.download_frames()[-1])) == +1


def test_start_still_waits_for_the_replies_it_needs(fake_wheel):
    output, device = fake_wheel

    assert device.reads > 0
    assert output._feature_index == FAKE_FEATURE_INDEX
    assert output._slot == FAKE_SLOT


def test_stop_does_not_wait_for_a_reply(fake_wheel):
    output, device = fake_wheel
    reads_before = device.reads
    output.stop()

    assert device.closed
    assert device.reads == reads_before


# ---------------------------------------------------------------------------------------------
# 12. Characterisation of a known, accepted artefact. These pin what the code DOES, not what it
# should do: the steering rate is differentiated against the wall clock, so the damping term is a
# property of how fast the force thread polls rather than of how fast the car's wheels are turning.
# ---------------------------------------------------------------------------------------------

# Matched to loop.py: 18 sub-steps inside a 0.3 s policy step, against the 200 Hz force thread.
SIM_SUBSTEP_SECONDS = 1.0 / 60.0
FFB_POLLS_PER_SIM_SUBSTEP = 3


class FakeWallClock:
    """Stands in for the `time` car_feel differentiates against."""

    def __init__(self):
        self.now_s = 0.0

    def perf_counter(self):
        return self.now_s


def steering_rate_over_one_interval(monkeypatch, interval_s):
    from pufferlib.wheel_demo import loop as loop_module

    clock = FakeWallClock()
    monkeypatch.setattr(loop_module, "time", clock)
    feel_loop = WheelDemoLoop.__new__(WheelDemoLoop)
    feel_loop.env = FakeHumanEnv(sim_human_state(0.0, 0.0, 5.0))
    feel_loop._last_steering_angle_rad = 0.0
    feel_loop._last_feel_time_s = 0.0
    clock.now_s += interval_s
    feel_loop.env.human_state = sim_human_state(SIM_RIGHT * HELD_STEERING_RAD, 0.0, 5.0)
    return feel_loop.car_feel().steering_rate_rad_s


def test_car_feel_steering_rate_follows_the_polling_interval_not_the_sim(monkeypatch):
    """The force thread polls faster than the sim moves the steering, so most polls see no change
    and the poll after a sub-step carries the whole delta at three times the true rate."""
    sim_paced = steering_rate_over_one_interval(monkeypatch, SIM_SUBSTEP_SECONDS)
    force_paced = steering_rate_over_one_interval(monkeypatch, SIM_SUBSTEP_SECONDS / FFB_POLLS_PER_SIM_SUBSTEP)

    assert abs(force_paced) == pytest.approx(FFB_POLLS_PER_SIM_SUBSTEP * abs(sim_paced))
    assert sign_of(force_paced) == sign_of(sim_paced)


def test_car_feel_reports_no_steering_rate_when_the_sim_has_not_moved(monkeypatch):
    from pufferlib.wheel_demo import loop as loop_module

    clock = FakeWallClock()
    monkeypatch.setattr(loop_module, "time", clock)
    feel_loop = WheelDemoLoop.__new__(WheelDemoLoop)
    feel_loop.env = FakeHumanEnv(sim_human_state(SIM_RIGHT * HELD_STEERING_RAD, 0.0, 5.0))
    feel_loop._last_steering_angle_rad = 0.0
    feel_loop._last_feel_time_s = 0.0
    clock.now_s += SIM_SUBSTEP_SECONDS
    feel_loop.car_feel()

    repolled = []
    for _ in range(FFB_POLLS_PER_SIM_SUBSTEP):
        clock.now_s += SIM_SUBSTEP_SECONDS / FFB_POLLS_PER_SIM_SUBSTEP
        repolled.append(feel_loop.car_feel().steering_rate_rad_s)

    assert repolled == [0.0] * FFB_POLLS_PER_SIM_SUBSTEP
