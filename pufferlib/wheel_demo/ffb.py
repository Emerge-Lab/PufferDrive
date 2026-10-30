"""Force feedback: what the wheel should push back with, and how to send it.

`compute_force` is a pure function of the car's state, so the feel can be tested and tuned without
a wheel attached. The transport is separate and lives behind ForceFeedbackOutput.

The effect that matters is self-aligning torque: the wheel loads up as the front tyres build
lateral force, and -- crucially -- goes LIGHT once they saturate. That drop is the only honest way a
wheel can tell you the front has let go, and it is why the grip model exists at all.
"""

import math
import threading
import time
from dataclasses import dataclass

# Peak self-aligning torque, as a fraction of the device's full output, at the grip limit.
ALIGN_TORQUE_GAIN = 0.85
# Below this speed the aligning torque has nothing to work with, so a centring spring stands in.
CENTRING_SPEED_MPS = 6.0
CENTRING_GAIN = 0.28
# Opposes rotation; without it the wheel oscillates around centre.
DAMPING_GAIN = 0.10
# Past the grip limit the tyre's self-aligning torque collapses rather than growing.
GRIP_FALLOFF = 0.55
# A collision is a sharp jolt that decays over a few frames.
IMPACT_FORCE = 1.0
IMPACT_DECAY_PER_SECOND = 6.0
# Past this the pulse has decayed to nothing, so it stops being applied.
IMPACT_WINDOW_SECONDS = 1.0
# Offroad is a buzz, not a push.
OFFROAD_RUMBLE_FORCE = 0.22
OFFROAD_RUMBLE_HZ = 22.0

FORCE_LIMIT = 1.0


@dataclass(frozen=True)
class CarFeel:
    """The slice of car state the wheel can be made to express."""

    steering_angle_rad: float = 0.0
    steering_rate_rad_s: float = 0.0
    speed_mps: float = 0.0
    accel_lat_mps2: float = 0.0
    tire_grip_mps2: float = 8.5
    collision: bool = False
    offroad: bool = False


def _clamp(value, limit=FORCE_LIMIT):
    return max(-limit, min(limit, value))


def grip_utilisation(accel_lat_mps2, tire_grip_mps2):
    """0 at rest, 1 at the limit of adhesion, above 1 when the demand exceeds it."""
    if tire_grip_mps2 <= 0.0:
        raise ValueError(f"tire_grip_mps2 must be positive, got {tire_grip_mps2}")
    return abs(accel_lat_mps2) / tire_grip_mps2


def self_aligning_torque(accel_lat_mps2, tire_grip_mps2):
    """Rises with lateral force, peaks at the limit, then falls away as the tyre saturates.

    The sign opposes the turn, which is what makes the wheel want to straighten itself.
    """
    utilisation = grip_utilisation(accel_lat_mps2, tire_grip_mps2)
    if utilisation <= 1.0:
        magnitude = utilisation
    else:
        magnitude = max(0.0, 1.0 - GRIP_FALLOFF * (utilisation - 1.0))
    return -math.copysign(ALIGN_TORQUE_GAIN * magnitude, accel_lat_mps2)


def centring_spring(steering_angle_rad, speed_mps, steering_limit_rad):
    """Stands in for aligning torque below CENTRING_SPEED_MPS, fading out as speed builds."""
    if steering_limit_rad <= 0.0:
        raise ValueError(f"steering_limit_rad must be positive, got {steering_limit_rad}")
    blend = max(0.0, 1.0 - abs(speed_mps) / CENTRING_SPEED_MPS)
    return -CENTRING_GAIN * blend * (steering_angle_rad / steering_limit_rad)


def compute_force(feel, steering_limit_rad, impact_age_s=None, elapsed_s=0.0):
    """Signed device force in [-1, 1]. Positive pushes the wheel clockwise."""
    force = self_aligning_torque(feel.accel_lat_mps2, feel.tire_grip_mps2)
    force += centring_spring(feel.steering_angle_rad, feel.speed_mps, steering_limit_rad)
    force += -DAMPING_GAIN * feel.steering_rate_rad_s

    if feel.offroad and abs(feel.speed_mps) > 0.5:
        force += OFFROAD_RUMBLE_FORCE * math.sin(2.0 * math.pi * OFFROAD_RUMBLE_HZ * elapsed_s)

    if impact_age_s is not None and impact_age_s >= 0.0:
        decay = math.exp(-IMPACT_DECAY_PER_SECOND * impact_age_s)
        # The jolt kicks away from where the driver is holding the wheel; centred, it kicks right.
        jolt_direction = -1.0 if feel.steering_angle_rad > 0.0 else 1.0
        force += jolt_direction * IMPACT_FORCE * decay

    return _clamp(force)


class ForceFeedbackOutput:
    """Transport interface. Backends: G920 over HID++ (macOS), evdev/SDL (Linux), and a recorder
    used by tests and by anyone without a wheel."""

    def start(self):
        raise NotImplementedError

    def send(self, force):
        raise NotImplementedError

    def stop(self):
        pass


class RecordingForceOutput(ForceFeedbackOutput):
    """Captures what would have been sent. This is what makes the feel testable with no hardware."""

    def __init__(self):
        self.sent = []
        self.started = False

    def start(self):
        self.started = True
        return self

    def send(self, force):
        if not self.started:
            raise RuntimeError("RecordingForceOutput.send before start()")
        if not -FORCE_LIMIT <= force <= FORCE_LIMIT:
            raise ValueError(f"force {force} outside [-{FORCE_LIMIT}, {FORCE_LIMIT}]")
        self.sent.append(force)

    def stop(self):
        self.started = False


# ---------------------------------------------------------------------------------------------
# Logitech G920 transport, HID++ 2.0 feature 0x8123.
# Byte layouts follow drivers/hid/hid-logitech-hidpp.c; they are NOT guessed. Verified on hardware:
# the feature resolves to an index, the aperture reads back 900 degrees, and a constant force moves
# the wheel.
# ---------------------------------------------------------------------------------------------

HIDPP_REPORT_SHORT = 0x10
HIDPP_REPORT_LONG = 0x11
HIDPP_DEVICE_INDEX = 0xFF
HIDPP_SOFTWARE_ID = 0x0A
HIDPP_ROOT_FEATURE_INDEX = 0x00
HIDPP_ROOT_GET_FEATURE = 0x00
HIDPP_ERROR_FEATURE_INDEX = 0xFF
FORCE_FEEDBACK_FEATURE = (0x81, 0x23)

HIDPP_FF_RESET_ALL = 0x11
HIDPP_FF_DOWNLOAD_EFFECT = 0x21
HIDPP_FF_SET_EFFECT_STATE = 0x31
HIDPP_FF_SET_APERTURE = 0x61
HIDPP_FF_EFFECT_CONSTANT = 0x00
HIDPP_FF_EFFECT_AUTOSTART = 0x80
HIDPP_FF_EFFECT_STATE_STOP = 0x01
HIDPP_FF_EFFECT_STATE_PLAY = 0x02

HIDPP_SHORT_PAYLOAD_BYTES = 3
HIDPP_LONG_PAYLOAD_BYTES = 16
HIDPP_REPLY_TIMEOUT_S = 0.4

# Measured on hardware: a POSITIVE device force turns the wheel towards negative steering_norm.
# Getting this backwards makes the feedback amplify every input instead of opposing it.
DEVICE_FORCE_SIGN = -1
# Full scale is 0x7FFF. The G920 is strong, so the demo opens well below it.
DEVICE_FORCE_FULL_SCALE = 0x7FFF
DEFAULT_FORCE_GAIN = 0.55
FFB_UPDATE_HZ = 200.0


class G920ForceFeedback(ForceFeedbackOutput):
    """Holds one constant-force effect open and rewrites its magnitude as the car changes."""

    def __init__(self, force_gain=DEFAULT_FORCE_GAIN, wheel_range_deg=None):
        if not 0.0 < force_gain <= 1.0:
            raise ValueError(f"force_gain must be in (0, 1], got {force_gain}")
        self.force_gain = force_gain
        self.wheel_range_deg = wheel_range_deg
        self._device = None
        self._feature_index = None
        self._slot = None

    def _request(self, report_id, feature_index, function_id, params, payload_bytes, want_reply=True):
        body = list(params) + [0] * (payload_bytes - len(params))
        self._device.write(bytes([report_id, HIDPP_DEVICE_INDEX, feature_index, function_id] + body))
        if not want_reply:
            return None
        deadline = time.monotonic() + HIDPP_REPLY_TIMEOUT_S
        while time.monotonic() < deadline:
            reply = self._device.read(64)
            # The vendor interface also carries asynchronous notifications, so only the frame that
            # echoes our device, feature and function is ours.
            if (
                reply
                and len(reply) > 4
                and reply[1] == HIDPP_DEVICE_INDEX
                and reply[2] == feature_index
                and reply[3] == function_id
            ):
                return bytes(reply)
            time.sleep(0.002)
        return None

    def start(self):
        from pufferlib.wheel_demo.input.g920 import find_g920_interfaces, set_darwin_non_exclusive

        import hid

        set_darwin_non_exclusive()
        interfaces = find_g920_interfaces()
        if not interfaces["vendor"]:
            raise RuntimeError("no Logitech vendor HID interface found; force feedback needs the G920 connected")
        self._device = hid.device()
        self._device.open_path(interfaces["vendor"][0])
        self._device.set_nonblocking(True)

        reply = self._request(
            HIDPP_REPORT_SHORT,
            HIDPP_ROOT_FEATURE_INDEX,
            (HIDPP_ROOT_GET_FEATURE << 4) | HIDPP_SOFTWARE_ID,
            FORCE_FEEDBACK_FEATURE,
            HIDPP_SHORT_PAYLOAD_BYTES,
        )
        if reply is None or reply[2] == HIDPP_ERROR_FEATURE_INDEX or reply[4] == 0:
            raise RuntimeError("this device does not expose HID++ feature 0x8123 (force feedback)")
        self._feature_index = reply[4]

        self._request(HIDPP_REPORT_SHORT, self._feature_index, HIDPP_FF_RESET_ALL, (), HIDPP_SHORT_PAYLOAD_BYTES)
        if self.wheel_range_deg is not None:
            self._request(
                HIDPP_REPORT_SHORT,
                self._feature_index,
                HIDPP_FF_SET_APERTURE,
                (int(self.wheel_range_deg) >> 8, int(self.wheel_range_deg) & 0xFF),
                HIDPP_SHORT_PAYLOAD_BYTES,
            )
        self._slot = self._download(0, 0)
        if self._slot is None:
            raise RuntimeError("the wheel refused the constant-force effect download")
        self._request(
            HIDPP_REPORT_SHORT,
            self._feature_index,
            HIDPP_FF_SET_EFFECT_STATE,
            (self._slot, HIDPP_FF_EFFECT_STATE_PLAY),
            HIDPP_SHORT_PAYLOAD_BYTES,
            want_reply=False,
        )
        return self

    def _download(self, slot, device_force, want_reply=True):
        unsigned = device_force & 0xFFFF
        params = [
            slot,
            HIDPP_FF_EFFECT_CONSTANT | HIDPP_FF_EFFECT_AUTOSTART,
            0,
            0,
            0,
            0,
            (unsigned >> 8) & 0xFF,
            unsigned & 0xFF,
        ] + [0] * 6
        reply = self._request(
            HIDPP_REPORT_LONG,
            self._feature_index,
            HIDPP_FF_DOWNLOAD_EFFECT,
            params,
            HIDPP_LONG_PAYLOAD_BYTES,
            want_reply=want_reply,
        )
        return reply[4] if reply else None

    def send(self, force):
        if self._device is None or self._slot is None:
            raise RuntimeError("G920ForceFeedback.send() before start()")
        if not -FORCE_LIMIT <= force <= FORCE_LIMIT:
            raise ValueError(f"force {force} outside [-{FORCE_LIMIT}, {FORCE_LIMIT}]")
        device_force = int(DEVICE_FORCE_SIGN * force * self.force_gain * DEVICE_FORCE_FULL_SCALE)
        # No reply is awaited per frame: one dropped reply would stall the wheel for a full timeout.
        self._download(self._slot, device_force, want_reply=False)

    def stop(self):
        if self._device is None:
            return
        if self._feature_index is not None:
            self._request(
                HIDPP_REPORT_SHORT,
                self._feature_index,
                HIDPP_FF_RESET_ALL,
                (),
                HIDPP_SHORT_PAYLOAD_BYTES,
                want_reply=False,
            )
        self._device.close()
        self._device = None
        self._slot = None


class ForceFeedbackLoop:
    """Drives the output from a car-state callback on its own thread, so the sim never waits on HID."""

    def __init__(self, output, feel_source, steering_limit_rad, update_hz=FFB_UPDATE_HZ):
        self.output = output
        self.feel_source = feel_source
        self.steering_limit_rad = steering_limit_rad
        self.interval_s = 1.0 / update_hz
        self._thread = None
        self._running = False
        self.error = None

    def start(self):
        self.output.start()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="ffb", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        started = time.monotonic()
        impact_started_s = None
        was_colliding = False
        try:
            while self._running:
                feel = self.feel_source()
                if feel is not None:
                    now_s = time.monotonic()
                    if feel.collision and not was_colliding:
                        impact_started_s = now_s
                    was_colliding = feel.collision
                    impact_age_s = None
                    if impact_started_s is not None:
                        impact_age_s = now_s - impact_started_s
                        if impact_age_s > IMPACT_WINDOW_SECONDS:
                            impact_started_s, impact_age_s = None, None
                    self.output.send(
                        compute_force(
                            feel,
                            self.steering_limit_rad,
                            impact_age_s=impact_age_s,
                            elapsed_s=now_s - started,
                        )
                    )
                time.sleep(self.interval_s)
        except Exception as error:
            self.error = error
            self._running = False

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.output.stop()
