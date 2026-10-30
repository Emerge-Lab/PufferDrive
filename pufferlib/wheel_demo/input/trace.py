"""Hardware-free wheel inputs: replay of a recorded HID trace, and a programmable source.

Every test and the headless loop run on these, so nothing above the backend layer needs a wheel
plugged in. Both support threadless use: call _read_state() directly for deterministic stepping.
"""

import time

from pufferlib.wheel_demo.input.base import HAT_NEUTRAL, WheelInput
from pufferlib.wheel_demo.input.g920 import G920_REPORT_LENGTH, ButtonMap, parse_g920_report

TRACE_REPLAY_INTERVAL_S = 1.0 / 190.0  # the rate the G920 was observed to report at


def load_report_trace(path, report_length=G920_REPORT_LENGTH):
    """Concatenated fixed-length HID reports -> list of bytes objects."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) == 0 or len(blob) % report_length != 0:
        raise ValueError(f"{path} is {len(blob)} bytes, not a whole number of {report_length}-byte reports")
    return [blob[i : i + report_length] for i in range(0, len(blob), report_length)]


class TraceWheelInput(WheelInput):
    """Replays a recorded G920 trace. Exhausting a non-looping trace stops the reader."""

    def __init__(self, reports, calibration=None, button_map=None, loop=False, realtime=False):
        super().__init__(calibration)
        if not reports:
            raise ValueError("TraceWheelInput needs at least one report")
        self.reports = list(reports)
        self.button_map = button_map or ButtonMap()
        self.loop = loop
        self.realtime = realtime
        self.frame_idx = 0

    @classmethod
    def from_file(cls, path, **kwargs):
        return cls(load_report_trace(path), **kwargs)

    @property
    def exhausted(self):
        return not self.loop and self.frame_idx >= len(self.reports)

    def _open(self):
        self.frame_idx = 0

    def _read_state(self):
        if self.exhausted:
            self._running = False
            return
        report = self.reports[self.frame_idx % len(self.reports)]
        self.frame_idx += 1
        decoded = parse_g920_report(report, self.calibration)
        previous = self._state
        rising = decoded["buttons"] & ~previous.buttons
        up_bit = self.button_map.shift_up_bit
        down_bit = self.button_map.shift_down_bit
        self._publish(
            previous,
            decoded["steering_norm"],
            decoded["throttle_norm"],
            decoded["brake_norm"],
            decoded["clutch_norm"],
            decoded["buttons"],
            decoded["hat"],
            1 if up_bit is not None and rising & (1 << up_bit) else 0,
            1 if down_bit is not None and rising & (1 << down_bit) else 0,
            time.monotonic(),
        )
        if self.realtime:
            time.sleep(TRACE_REPLAY_INTERVAL_S)


class SyntheticWheelInput(WheelInput):
    """Controls driven from code. The headless loop and most tests use this."""

    def __init__(self, calibration=None):
        super().__init__(calibration)
        self._pending = {
            "steering_norm": 0.0,
            "throttle_norm": 0.0,
            "brake_norm": 0.0,
            "clutch_norm": 0.0,
            "buttons": 0,
            "hat": HAT_NEUTRAL,
            "shift_up": 0,
            "shift_down": 0,
        }

    def set_controls(
        self, steering_norm=None, throttle_norm=None, brake_norm=None, clutch_norm=None, buttons=None, hat=None
    ):
        for key, value in (
            ("steering_norm", steering_norm),
            ("throttle_norm", throttle_norm),
            ("brake_norm", brake_norm),
            ("clutch_norm", clutch_norm),
            ("buttons", buttons),
            ("hat", hat),
        ):
            if value is not None:
                self._pending[key] = value
        self._read_state()

    def request_shift_up(self):
        self._pending["shift_up"] = 1
        self._read_state()

    def request_shift_down(self):
        self._pending["shift_down"] = 1
        self._read_state()

    def _open(self):
        pass

    def _read_state(self):
        pending = self._pending
        self._publish(
            self._state,
            pending["steering_norm"],
            pending["throttle_norm"],
            pending["brake_norm"],
            pending["clutch_norm"],
            pending["buttons"],
            pending["hat"],
            pending["shift_up"],
            pending["shift_down"],
            time.monotonic(),
        )
        pending["shift_up"] = 0
        pending["shift_down"] = 0
