"""Wheel input interface shared by every backend (raw HID, SDL, recorded trace, synthetic)."""

import json
import math
import threading
import time
from dataclasses import dataclass

HAT_NEUTRAL = 0x0F

# A backend that has opened a device but produced no report for this long is treated as dead.
INPUT_STALE_SECONDS = 2.0


@dataclass(frozen=True)
class WheelState:
    """One sampled instant of the physical controls.

    steering_norm is -1 at full left lock, +1 at full right lock. The pedals are 0 released,
    1 fully depressed. shift_up_count/shift_down_count are monotonic so a consumer running
    slower than the reader thread diffs them instead of missing an edge.
    """

    steering_norm: float = 0.0
    throttle_norm: float = 0.0
    brake_norm: float = 0.0
    clutch_norm: float = 0.0
    buttons: int = 0
    hat: int = HAT_NEUTRAL
    shift_up_count: int = 0
    shift_down_count: int = 0
    report_count: int = 0
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class WheelCalibration:
    """Raw-counts-to-normalized mapping for one physical wheel.

    Defaults cover the G920's nominal ranges; capture_* in calibrate.py narrows them to the
    unit actually plugged in. Pedal rest/floor are stored unordered-safe: the G920 reports
    255 released and 0 depressed, so rest > floor is the normal case.
    """

    steering_raw_min: int = 0
    steering_raw_center: int = 32768
    steering_raw_max: int = 65535
    steering_deadzone_norm: float = 0.0
    throttle_raw_rest: int = 255
    throttle_raw_floor: int = 0
    brake_raw_rest: int = 255
    brake_raw_floor: int = 0
    clutch_raw_rest: int = 255
    clutch_raw_floor: int = 0
    pedal_deadzone_norm: float = 0.02
    # The G920 brake is a stiff progressive rubber stop, so linear travel puts most of the braking
    # in the last part of the press. Below 1.0 this bends the curve so light presses bite harder.
    brake_response_gamma: float = 0.4
    # Above 1.0 part throttle gives proportionally less torque; full travel is still full throttle.
    throttle_response_gamma: float = 1.6

    def __post_init__(self):
        if not self.steering_raw_min < self.steering_raw_center < self.steering_raw_max:
            raise ValueError(
                f"steering calibration must satisfy min < center < max, got "
                f"{self.steering_raw_min} < {self.steering_raw_center} < {self.steering_raw_max}"
            )
        for pedal in ("throttle", "brake", "clutch"):
            rest = getattr(self, f"{pedal}_raw_rest")
            floor = getattr(self, f"{pedal}_raw_floor")
            if rest == floor:
                raise ValueError(f"{pedal} calibration has zero travel: rest == floor == {rest}")
        if not 0.0 <= self.steering_deadzone_norm < 1.0:
            raise ValueError(f"steering_deadzone_norm must be in [0, 1), got {self.steering_deadzone_norm}")
        if not 0.0 <= self.pedal_deadzone_norm < 1.0:
            raise ValueError(f"pedal_deadzone_norm must be in [0, 1), got {self.pedal_deadzone_norm}")
        if not 0.0 < self.brake_response_gamma <= 1.0:
            raise ValueError(f"brake_response_gamma must be in (0, 1], got {self.brake_response_gamma}")
        if not 1.0 <= self.throttle_response_gamma < math.inf:
            raise ValueError(f"throttle_response_gamma must be finite and >= 1, got {self.throttle_response_gamma}")

    def steering_to_norm(self, raw):
        if raw >= self.steering_raw_center:
            span = self.steering_raw_max - self.steering_raw_center
            norm = (raw - self.steering_raw_center) / span
        else:
            span = self.steering_raw_center - self.steering_raw_min
            norm = (raw - self.steering_raw_center) / span
        norm = _clamp(norm, -1.0, 1.0)
        return _apply_deadzone(norm, self.steering_deadzone_norm)

    def pedal_to_norm(self, raw, pedal):
        rest = getattr(self, f"{pedal}_raw_rest")
        floor = getattr(self, f"{pedal}_raw_floor")
        norm = (raw - rest) / (floor - rest)
        norm = _clamp(norm, 0.0, 1.0)
        if norm <= self.pedal_deadzone_norm:
            return 0.0
        norm = (norm - self.pedal_deadzone_norm) / (1.0 - self.pedal_deadzone_norm)
        if pedal == "brake":
            return norm**self.brake_response_gamma
        if pedal == "throttle":
            return norm**self.throttle_response_gamma
        return norm

    def to_json(self, path):
        with open(path, "w") as handle:
            json.dump(self.__dict__, handle, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path):
        with open(path) as handle:
            stored = json.load(handle)
        unknown = set(stored) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path} has unknown calibration keys: {sorted(unknown)}")
        return cls(**stored)


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


def _apply_deadzone(norm, deadzone):
    if deadzone <= 0.0:
        return norm
    magnitude = abs(norm)
    if magnitude <= deadzone:
        return 0.0
    rescaled = (magnitude - deadzone) / (1.0 - deadzone)
    return rescaled if norm > 0.0 else -rescaled


class WheelInput:
    """Base class: a reader thread publishes immutable WheelState snapshots.

    Subclasses implement _open/_read_state/_close. latest() never blocks, so the simulation loop
    is decoupled from device latency; it does re-raise a reader-thread failure rather than serving
    a frozen state forever.
    """

    def __init__(self, calibration=None):
        self.calibration = calibration or WheelCalibration()
        self._state = WheelState()
        self._thread = None
        self._running = False
        self._error = None

    @property
    def name(self):
        return type(self).__name__

    def start(self):
        if self._running:
            raise RuntimeError(f"{self.name} already started")
        if self._error is not None:
            raise RuntimeError(f"{self.name} reader already failed and cannot be restarted") from self._error
        self._open()
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=INPUT_STALE_SECONDS)
            self._thread = None
        self._close()

    def latest(self):
        """Most recent snapshot. Assignment of an immutable dataclass is atomic under the GIL,
        so no lock is taken on the simulation loop's side."""
        if self._error is not None:
            raise self._error
        return self._state

    def is_stale(self, now_s=None):
        state = self._state
        if state.report_count == 0:
            return True
        return (now_s if now_s is not None else time.monotonic()) - state.timestamp_s > INPUT_STALE_SECONDS

    def _publish(
        self,
        previous,
        steering_norm,
        throttle_norm,
        brake_norm,
        clutch_norm,
        buttons,
        hat,
        shift_up,
        shift_down,
        timestamp_s,
    ):
        self._state = WheelState(
            steering_norm=steering_norm,
            throttle_norm=throttle_norm,
            brake_norm=brake_norm,
            clutch_norm=clutch_norm,
            buttons=buttons,
            hat=hat,
            shift_up_count=previous.shift_up_count + shift_up,
            shift_down_count=previous.shift_down_count + shift_down,
            report_count=previous.report_count + 1,
            timestamp_s=timestamp_s,
        )

    def _run(self):
        try:
            while self._running:
                self._read_state()
        except Exception as error:  # surfaced on the next latest() rather than dying silently
            self._error = error
            self._running = False

    def _open(self):
        raise NotImplementedError

    def _read_state(self):
        raise NotImplementedError

    def _close(self):
        pass
