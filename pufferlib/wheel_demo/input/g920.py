"""Logitech G920 raw-HID input.

Report layout was established on hardware (6572 reports, ~190 Hz) and the pedal identities by
pressing them in a known order and correlating which byte moved first -- see
docs/wheel_demo_plan.md. Button bit positions are NOT guessed: they come from a button map
captured by scripts/wheel_demo/probe_g920.py.
"""

import ctypes
import json
import sys
import time
from dataclasses import dataclass

from pufferlib.wheel_demo.input.base import WheelInput

G920_VENDOR_ID = 0x046D
G920_PRODUCT_ID = 0xC262

GENERIC_DESKTOP_USAGE_PAGE = 0x0001
JOYSTICK_USAGE = 0x0004
LOGITECH_VENDOR_USAGE_PAGE = 0xFF43

G920_REPORT_ID = 0x01
G920_REPORT_LENGTH = 10
HAT_MASK = 0x0F
BUTTON_NIBBLE_SHIFT = 4
BUTTON_BYTE2_SHIFT = 4
BUTTON_BYTE3_SHIFT = 12
BUTTON_MASK_BIT_COUNT = 20
HAT_DIRECTION_COUNT = 8

HID_READ_BUFFER_BYTES = 64
HID_POLL_INTERVAL_S = 0.001


FACE_BUTTON_NAMES = ("a", "b", "x", "y", "view", "menu", "lsb", "rsb")
DPAD_DIRECTION_NAMES = ("up", "right", "down", "left")


@dataclass(frozen=True)
class ButtonMap:
    """Bit positions within the 20-bit button mask, and the hat value of each D-pad direction.
    None = not yet mapped on this unit."""

    shift_up_bit: int = None
    shift_down_bit: int = None
    a_bit: int = None
    b_bit: int = None
    x_bit: int = None
    y_bit: int = None
    view_bit: int = None
    menu_bit: int = None
    lsb_bit: int = None
    rsb_bit: int = None
    dpad_up_hat: int = None
    dpad_right_hat: int = None
    dpad_down_hat: int = None
    dpad_left_hat: int = None

    def __post_init__(self):
        bit_fields = ["shift_up_bit", "shift_down_bit"] + [f"{name}_bit" for name in FACE_BUTTON_NAMES]
        for field_name in bit_fields:
            bit = getattr(self, field_name)
            if bit is not None and not 0 <= bit < BUTTON_MASK_BIT_COUNT:
                raise ValueError(
                    f"{field_name}={bit} is outside the {BUTTON_MASK_BIT_COUNT}-bit button mask; "
                    "such a bit can never match, so the control would silently never fire"
                )
        for direction in DPAD_DIRECTION_NAMES:
            hat = getattr(self, f"dpad_{direction}_hat")
            if hat is not None and not 0 <= hat < HAT_DIRECTION_COUNT:
                raise ValueError(f"dpad_{direction}_hat={hat} is not a hat direction (0..{HAT_DIRECTION_COUNT - 1})")
        mapped = [getattr(self, name) for name in bit_fields if getattr(self, name) is not None]
        if len(mapped) != len(set(mapped)):
            raise ValueError(f"two controls share one button bit: {mapped}")
        hats = [getattr(self, f"dpad_{d}_hat") for d in DPAD_DIRECTION_NAMES if getattr(self, f"dpad_{d}_hat") is not None]
        if len(hats) != len(set(hats)):
            raise ValueError(f"two D-pad directions share one hat value: {hats}")

    def pressed(self, buttons, hat):
        """Names of the mapped face buttons and D-pad directions held in this report."""
        names = [name for name in FACE_BUTTON_NAMES if getattr(self, f"{name}_bit") is not None
                 and buttons & (1 << getattr(self, f"{name}_bit"))]
        names += [f"dpad_{direction}" for direction in DPAD_DIRECTION_NAMES
                  if getattr(self, f"dpad_{direction}_hat") == hat]
        return frozenset(names)

    def require_shift_bits(self):
        if self.shift_up_bit is None or self.shift_down_bit is None:
            raise RuntimeError(
                "Paddle shift bits are unmapped. Run scripts/wheel_demo/probe_g920.py and pass the "
                "resulting button map; manual gearbox mode cannot run without it."
            )

    def to_json(self, path):
        with open(path, "w") as handle:
            json.dump(self.__dict__, handle, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path):
        with open(path) as handle:
            stored = json.load(handle)
        unknown = set(stored) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path} has unknown button-map keys: {sorted(unknown)}")
        return cls(**stored)


def decode_button_mask(report):
    return (report[1] >> BUTTON_NIBBLE_SHIFT) | (report[2] << BUTTON_BYTE2_SHIFT) | (report[3] << BUTTON_BYTE3_SHIFT)


def parse_g920_report(report, calibration):
    """Pure decode of one 10-byte report into raw and normalized control values."""
    if len(report) != G920_REPORT_LENGTH:
        raise ValueError(f"G920 report must be {G920_REPORT_LENGTH} bytes, got {len(report)}")
    if report[0] != G920_REPORT_ID:
        raise ValueError(f"G920 report id must be 0x{G920_REPORT_ID:02x}, got 0x{report[0]:02x}")

    steering_raw = report[4] | (report[5] << 8)
    return {
        "steering_raw": steering_raw,
        "throttle_raw": report[6],
        "brake_raw": report[7],
        "clutch_raw": report[8],
        "buttons": decode_button_mask(report),
        "hat": report[1] & HAT_MASK,
        "steering_norm": calibration.steering_to_norm(steering_raw),
        "throttle_norm": calibration.pedal_to_norm(report[6], "throttle"),
        "brake_norm": calibration.pedal_to_norm(report[7], "brake"),
        "clutch_norm": calibration.pedal_to_norm(report[8], "clutch"),
    }


def hid_module():
    # On Linux pip's `hid` is the libusb backend: usages read 0 and opening detaches the kernel driver.
    if sys.platform == "linux":
        import hidraw

        return hidraw
    import hid

    return hid


def set_darwin_non_exclusive():
    """macOS hidapi seizes a device on open, which would make the axis interface and the HID++
    force-feedback interface mutually exclusive. The toggle is absent from the Python binding but
    exported by its bundled library."""
    if sys.platform != "darwin":
        return False
    import hid

    library = ctypes.CDLL(hid.__file__)
    library.hid_darwin_set_open_exclusive.argtypes = [ctypes.c_int]
    library.hid_darwin_set_open_exclusive(0)
    return library.hid_darwin_get_open_exclusive() == 0


def find_g920_interfaces():
    found = {"axes": None, "vendor": []}
    for entry in hid_module().enumerate(G920_VENDOR_ID, G920_PRODUCT_ID):
        if entry["usage_page"] == GENERIC_DESKTOP_USAGE_PAGE and entry["usage"] == JOYSTICK_USAGE:
            found["axes"] = entry["path"]
        elif entry["usage_page"] == LOGITECH_VENDOR_USAGE_PAGE:
            found["vendor"].append(entry["path"])
    return found


class G920HidInput(WheelInput):
    """Reads the G920's joystick interface directly over raw HID."""

    def __init__(self, calibration=None, button_map=None):
        super().__init__(calibration)
        self.button_map = button_map or ButtonMap()
        self._device = None

    def _open(self):
        set_darwin_non_exclusive()
        interfaces = find_g920_interfaces()
        if interfaces["axes"] is None:
            raise RuntimeError(
                f"No G920 axis interface found (vid=0x{G920_VENDOR_ID:04x} pid=0x{G920_PRODUCT_ID:04x}). "
                "Check the wheel is connected and, on macOS, that Input Monitoring is granted; on Linux, "
                "that the udev rule in docs/wheel_demo_setup.md is installed."
            )
        self._device = hid_module().device()
        self._device.open_path(interfaces["axes"])
        self._device.set_nonblocking(True)

    def _read_state(self):
        report = self._device.read(HID_READ_BUFFER_BYTES)
        if not report:
            time.sleep(HID_POLL_INTERVAL_S)
            return
        # One HID interface carries more than one report type. Anything that is not the 10-byte
        # axis report is another report, not a malformed one, so it is skipped rather than raised on.
        if len(report) != G920_REPORT_LENGTH or report[0] != G920_REPORT_ID:
            return
        decoded = parse_g920_report(report, self.calibration)
        previous = self._state
        shift_up, shift_down = self._shift_edges(previous.buttons, decoded["buttons"])
        self._publish(
            previous,
            decoded["steering_norm"],
            decoded["throttle_norm"],
            decoded["brake_norm"],
            decoded["clutch_norm"],
            decoded["buttons"],
            decoded["hat"],
            shift_up,
            shift_down,
            time.monotonic(),
        )

    def _shift_edges(self, previous_buttons, buttons):
        rising = buttons & ~previous_buttons
        up_bit = self.button_map.shift_up_bit
        down_bit = self.button_map.shift_down_bit
        shift_up = 1 if up_bit is not None and rising & (1 << up_bit) else 0
        shift_down = 1 if down_bit is not None and rising & (1 << down_bit) else 0
        return shift_up, shift_down

    def _close(self):
        if self._device is not None:
            self._device.close()
            self._device = None
