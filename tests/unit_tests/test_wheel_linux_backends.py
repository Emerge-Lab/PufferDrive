"""Linux transports for the G920: the hidraw HID backend and kernel evdev force feedback.

No device is ever opened. hid modules are swapped for recorders in sys.modules, and the evdev path
runs against a tmp sysfs tree with os.open/os.write/fcntl.ioctl replaced by recorders. Struct
layouts are checked against linux/input.h through ctypes and, where gcc is present, the header itself.
"""

import ctypes
import importlib.util
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import types

import pytest

from pufferlib.wheel_demo import calibrate
from pufferlib.wheel_demo import ffb
from pufferlib.wheel_demo.ffb import (
    EvdevForceFeedback,
    G920ForceFeedback,
    compute_force,
    find_g920_event_device,
    platform_force_output,
)
from pufferlib.wheel_demo.input import g920
from pufferlib.wheel_demo.loop import WheelDemoLoop

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE_SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "probe_g920.py")

# linux/input.h and linux/input-event-codes.h, written out here so a typo in ffb.py cannot hide.
LINUX_EV_FF = 0x15
LINUX_FF_CONSTANT = 0x52
LINUX_FF_GAIN = 0x60
LINUX_FF_AUTOCENTER = 0x61
LINUX_IOC_WRITE = 1
LINUX_IOC_DIR_SHIFT = 30
LINUX_IOC_SIZE_SHIFT = 16
LINUX_IOC_TYPE_SHIFT = 8
EVDEV_IOC_TYPE = ord("E")
EVIOCSFF_NR = 0x80
EVIOCRMFF_NR = 0x81
FF_EFFECT_SIZE_BYTES = 48
INPUT_EVENT_SIZE_BYTES = 24
FF_EFFECT_LEVEL_OFFSET_BYTES = 16
FF_EFFECT_ID_OFFSET_BYTES = 2
FF_GAIN_FULL = 0xFFFF
QUARTER_TURN_DIRECTION = 0x4000
DEVICE_FULL_SCALE = 0x7FFF
G920_RANGE_DEG = 900
G920_VENDOR_HEX = "046d"
G920_PRODUCT_HEX = "c262"

# Same measured fact as the HID++ path: the kernel passes level * sin(direction) through, so a
# POSITIVE level turns the wheel anticlockwise, towards negative steering_norm.
POSITIVE_LEVEL_TURNS_WHEEL_TOWARDS_NORM = -1
SIM_LEFT = +1
SIM_RIGHT = -1
HELD_STEERING_RAD = 0.3
PARKING_SPEED_MPS = 1.0
STEERING_LIMIT_RAD = 0.55

FAKE_FD = 4242
FAKE_EFFECT_ID = 3
AXES_PATH = b"/dev/hidraw-axes"
VENDOR_PATH = b"/dev/hidraw-vendor"
GENERIC_DESKTOP_USAGE_PAGE = 0x0001
JOYSTICK_USAGE = 0x0004
LOGITECH_VENDOR_USAGE_PAGE = 0xFF43


def ioc_write(nr, size_bytes):
    return (
        (LINUX_IOC_WRITE << LINUX_IOC_DIR_SHIFT)
        | (size_bytes << LINUX_IOC_SIZE_SHIFT)
        | (EVDEV_IOC_TYPE << LINUX_IOC_TYPE_SHIFT)
        | nr
    )


class FfTrigger(ctypes.Structure):
    _fields_ = [("button", ctypes.c_uint16), ("interval", ctypes.c_uint16)]


class FfReplay(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint16), ("delay", ctypes.c_uint16)]


class FfEnvelope(ctypes.Structure):
    _fields_ = [
        ("attack_length", ctypes.c_uint16),
        ("attack_level", ctypes.c_uint16),
        ("fade_length", ctypes.c_uint16),
        ("fade_level", ctypes.c_uint16),
    ]


class FfConstantEffect(ctypes.Structure):
    _fields_ = [("level", ctypes.c_int16), ("envelope", FfEnvelope)]


class FfPeriodicEffect(ctypes.Structure):
    _fields_ = [
        ("waveform", ctypes.c_uint16),
        ("period", ctypes.c_uint16),
        ("magnitude", ctypes.c_int16),
        ("offset", ctypes.c_int16),
        ("phase", ctypes.c_uint16),
        ("envelope", FfEnvelope),
        ("custom_len", ctypes.c_uint32),
        ("custom_data", ctypes.POINTER(ctypes.c_int16)),
    ]


class FfEffectUnion(ctypes.Union):
    _fields_ = [("constant", FfConstantEffect), ("periodic", FfPeriodicEffect)]


class FfEffect(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint16),
        ("id", ctypes.c_int16),
        ("direction", ctypes.c_uint16),
        ("trigger", FfTrigger),
        ("replay", FfReplay),
        ("u", FfEffectUnion),
    ]


class Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class InputEvent(ctypes.Structure):
    _fields_ = [("time", Timeval), ("type", ctypes.c_uint16), ("code", ctypes.c_uint16), ("value", ctypes.c_int32)]


def decode_effect(raw):
    return FfEffect.from_buffer_copy(bytes(raw))


def decode_event(raw):
    return InputEvent.from_buffer_copy(bytes(raw))


# ---------------------------------------------------------------------------------------------
# Fake hid backends.
# ---------------------------------------------------------------------------------------------


class DeviceOpened(Exception):
    pass


class RecordingHidModule(types.ModuleType):
    """Enumerates like pip's hidraw backend (real usages) and refuses to actually open a device."""

    def __init__(self, name):
        super().__init__(name)
        self.opened_paths = []
        self.enumerate_calls = []

    def enumerate(self, vendor_id, product_id):
        self.enumerate_calls.append((vendor_id, product_id))
        return [
            {"usage_page": GENERIC_DESKTOP_USAGE_PAGE, "usage": JOYSTICK_USAGE, "path": AXES_PATH},
            {"usage_page": LOGITECH_VENDOR_USAGE_PAGE, "usage": 1, "path": VENDOR_PATH},
        ]

    def device(self):
        module = self

        class Device:
            def open_path(self, path):
                module.opened_paths.append(path)
                raise DeviceOpened(path)

        return Device()


class ForbiddenHidModule(types.ModuleType):
    """pip's libusb `hid` on Linux: touching it would detach the kernel driver."""

    def enumerate(self, *args):
        raise AssertionError("the libusb hid backend was used on Linux")

    def device(self):
        raise AssertionError("the libusb hid backend was used on Linux")


@pytest.fixture
def linux_hid(monkeypatch):
    hidraw = RecordingHidModule("hidraw")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "hidraw", hidraw)
    monkeypatch.setitem(sys.modules, "hid", ForbiddenHidModule("hid"))
    return hidraw


def load_probe_script():
    spec = importlib.util.spec_from_file_location("wheel_demo_probe_g920", PROBE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hid_module_is_hidraw_on_linux(linux_hid):
    assert g920.hid_module() is linux_hid


def test_hid_module_is_hid_elsewhere(monkeypatch):
    hid = RecordingHidModule("hid")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "hid", hid)
    monkeypatch.setitem(sys.modules, "hidraw", ForbiddenHidModule("hidraw"))
    assert g920.hid_module() is hid


def test_the_real_linux_hid_module_is_the_hidraw_backend():
    if sys.platform != "linux":
        pytest.skip("Linux only")
    pytest.importorskip("hidraw")
    assert g920.hid_module().__name__ == "hidraw"


def test_find_g920_interfaces_enumerates_through_hidraw(linux_hid):
    found = g920.find_g920_interfaces()
    assert found == {"axes": AXES_PATH, "vendor": [VENDOR_PATH]}
    assert linux_hid.enumerate_calls == [(g920.G920_VENDOR_ID, g920.G920_PRODUCT_ID)]


def test_axis_reader_opens_through_hidraw(linux_hid):
    with pytest.raises(DeviceOpened):
        g920.G920HidInput()._open()
    assert linux_hid.opened_paths == [AXES_PATH]


def test_hidpp_force_feedback_opens_through_hidraw(linux_hid):
    with pytest.raises(DeviceOpened):
        G920ForceFeedback().start()
    assert linux_hid.opened_paths == [VENDOR_PATH]


def test_calibration_opens_through_hidraw(linux_hid):
    with pytest.raises(DeviceOpened):
        calibrate.open_wheel()
    assert linux_hid.opened_paths == [AXES_PATH]


def test_probe_script_opens_through_hidraw(linux_hid):
    probe = load_probe_script()
    with pytest.raises(DeviceOpened):
        probe.open_axis_device()
    assert linux_hid.opened_paths == [AXES_PATH]


# ---------------------------------------------------------------------------------------------
# struct and ioctl layout, independent of the format strings in ffb.py.
# ---------------------------------------------------------------------------------------------


def test_ff_effect_layout_matches_linux_input_h():
    assert ctypes.sizeof(FfEffect) == FF_EFFECT_SIZE_BYTES
    assert FfEffect.u.offset + FfEffectUnion.constant.offset + FfConstantEffect.level.offset == (
        FF_EFFECT_LEVEL_OFFSET_BYTES
    )
    assert struct.calcsize(ffb.EVDEV_FF_EFFECT_FORMAT) == FF_EFFECT_SIZE_BYTES


def test_input_event_layout_matches_linux_input_h():
    assert ctypes.sizeof(InputEvent) == INPUT_EVENT_SIZE_BYTES
    assert struct.calcsize(ffb.EVDEV_INPUT_EVENT_FORMAT) == INPUT_EVENT_SIZE_BYTES


def test_packed_effect_decodes_field_for_field():
    level = -12345
    packed = struct.pack(
        ffb.EVDEV_FF_EFFECT_FORMAT, LINUX_FF_CONSTANT, 7, QUARTER_TURN_DIRECTION, 1, 2, 3, 4, level, 5, 6, 8, 9
    )
    effect = decode_effect(packed)
    assert (effect.type, effect.id, effect.direction) == (LINUX_FF_CONSTANT, 7, QUARTER_TURN_DIRECTION)
    assert (effect.trigger.button, effect.trigger.interval, effect.replay.length, effect.replay.delay) == (1, 2, 3, 4)
    assert effect.u.constant.level == level
    envelope = effect.u.constant.envelope
    assert (envelope.attack_length, envelope.attack_level, envelope.fade_length, envelope.fade_level) == (5, 6, 8, 9)


def test_ioctl_numbers_and_event_codes_match_linux():
    assert ffb.EVDEV_EVIOCSFF == ioc_write(EVIOCSFF_NR, FF_EFFECT_SIZE_BYTES)
    assert ffb.EVDEV_EVIOCRMFF == ioc_write(EVIOCRMFF_NR, ctypes.sizeof(ctypes.c_int))
    assert (ffb.EVDEV_EV_FF, ffb.EVDEV_FF_CONSTANT) == (LINUX_EV_FF, LINUX_FF_CONSTANT)
    assert (ffb.EVDEV_FF_GAIN, ffb.EVDEV_FF_AUTOCENTER) == (LINUX_FF_GAIN, LINUX_FF_AUTOCENTER)
    assert ffb.EVDEV_DIRECTION_PASS_THROUGH == QUARTER_TURN_DIRECTION


HEADER_PROBE_SOURCE = """
#include <linux/input.h>
#include <stddef.h>
#include <stdio.h>
int main(void) {
    printf("%zu %zu %zu %lu %lu %d %d %d %d\\n", sizeof(struct ff_effect),
           offsetof(struct ff_effect, u.constant.level), sizeof(struct input_event),
           (unsigned long) EVIOCSFF, (unsigned long) EVIOCRMFF, EV_FF, FF_CONSTANT, FF_GAIN, FF_AUTOCENTER);
    return 0;
}
"""


def test_layout_agrees_with_the_installed_kernel_header(tmp_path):
    if sys.platform != "linux" or shutil.which("gcc") is None or not os.path.isfile("/usr/include/linux/input.h"):
        pytest.skip("needs gcc and linux/input.h")
    source = tmp_path / "probe.c"
    binary = tmp_path / "probe"
    source.write_text(HEADER_PROBE_SOURCE)
    subprocess.run(["gcc", str(source), "-o", str(binary)], check=True, capture_output=True)
    fields = [int(value) for value in subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.split()]
    effect_size, level_offset, event_size, sff, rmff, ev_ff, ff_constant, ff_gain, ff_autocenter = fields
    assert struct.calcsize(ffb.EVDEV_FF_EFFECT_FORMAT) == effect_size
    packed = struct.pack(ffb.EVDEV_FF_EFFECT_FORMAT, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0)
    assert struct.unpack_from("=h", packed, level_offset)[0] == -2
    assert struct.calcsize(ffb.EVDEV_INPUT_EVENT_FORMAT) == event_size
    assert (ffb.EVDEV_EVIOCSFF, ffb.EVDEV_EVIOCRMFF) == (sff, rmff)
    assert (ffb.EVDEV_EV_FF, ffb.EVDEV_FF_CONSTANT, ffb.EVDEV_FF_GAIN, ffb.EVDEV_FF_AUTOCENTER) == (
        ev_ff,
        ff_constant,
        ff_gain,
        ff_autocenter,
    )


# ---------------------------------------------------------------------------------------------
# EvdevForceFeedback over a fake sysfs tree and recording syscalls.
# ---------------------------------------------------------------------------------------------


def write_input_node(sysfs_dir, name, vendor_hex, product_hex, range_deg=None):
    id_dir = sysfs_dir / name / "device" / "id"
    id_dir.mkdir(parents=True)
    (id_dir / "vendor").write_text(vendor_hex + "\n")
    (id_dir / "product").write_text(product_hex + "\n")
    if range_deg is not None:
        hid_dir = sysfs_dir / name / "device" / "device"
        hid_dir.mkdir()
        (hid_dir / "range").write_text(f"{range_deg}\n")


def make_sysfs(tmp_path, range_deg=G920_RANGE_DEG, with_wheel=True):
    sysfs_dir = tmp_path / "input"
    sysfs_dir.mkdir()
    (sysfs_dir / "input4").mkdir()
    (sysfs_dir / "mouse0").mkdir()
    write_input_node(sysfs_dir, "event2", "1234", "5678")
    write_input_node(sysfs_dir, "event5", G920_VENDOR_HEX, "c261")
    if with_wheel:
        write_input_node(sysfs_dir, "event9", G920_VENDOR_HEX, G920_PRODUCT_HEX, range_deg)
    return sysfs_dir


class RecordingEvdev:
    """Stands in for the os and fcntl calls ffb makes on the event node."""

    def __init__(self):
        self.opened = []
        self.closed = []
        self.events = []
        self.uploads = []
        self.removed = []
        self.calls = []

    def open(self, path, flags):
        self.opened.append((path, flags))
        self.calls.append("open")
        return FAKE_FD

    def write(self, fd, data):
        assert fd == FAKE_FD
        self.events.append(decode_event(data))
        self.calls.append("write")
        return len(data)

    def close(self, fd):
        assert fd == FAKE_FD
        self.closed.append(fd)
        self.calls.append("close")

    def ioctl(self, fd, request, arg=0, mutate_flag=True):
        assert fd == FAKE_FD
        if request == ffb.EVDEV_EVIOCSFF:
            effect = decode_effect(arg)
            self.uploads.append(effect)
            self.calls.append("upload")
            if effect.id == -1:
                struct.pack_into("=h", arg, FF_EFFECT_ID_OFFSET_BYTES, FAKE_EFFECT_ID)
            return 0
        if request == ffb.EVDEV_EVIOCRMFF:
            self.removed.append(arg)
            self.calls.append("remove")
            return 0
        raise AssertionError(f"unexpected ioctl {request:#x}")


@pytest.fixture
def evdev(monkeypatch):
    recorder = RecordingEvdev()
    fake_os = types.SimpleNamespace(
        path=os.path,
        listdir=os.listdir,
        O_RDWR=os.O_RDWR,
        open=recorder.open,
        write=recorder.write,
        close=recorder.close,
    )
    monkeypatch.setattr(ffb, "os", fake_os)
    monkeypatch.setattr(ffb, "fcntl", types.SimpleNamespace(ioctl=recorder.ioctl))
    return recorder


def started_output(tmp_path, gain=1.0):
    return EvdevForceFeedback(force_gain=gain, sysfs_input_dir=str(make_sysfs(tmp_path))).start()


def test_find_event_device_picks_the_g920_and_its_range_attribute(tmp_path):
    sysfs_dir = make_sysfs(tmp_path)
    event_path, range_path = find_g920_event_device(str(sysfs_dir))
    assert event_path == "/dev/input/event9"
    assert range_path == str(sysfs_dir / "event9" / "device" / "device" / "range")


def test_find_event_device_reports_no_wheel(tmp_path):
    assert find_g920_event_device(str(make_sysfs(tmp_path, with_wheel=False))) is None


def test_start_opens_the_event_node_read_write(tmp_path, evdev):
    started_output(tmp_path)
    assert evdev.opened == [("/dev/input/event9", os.O_RDWR)]


def test_start_disables_autocentre_and_sets_full_gain(tmp_path, evdev):
    started_output(tmp_path)
    settings = {(event.type, event.code): event.value for event in evdev.events if event.code != FAKE_EFFECT_ID}
    assert settings[(LINUX_EV_FF, LINUX_FF_AUTOCENTER)] == 0
    assert settings[(LINUX_EV_FF, LINUX_FF_GAIN)] == FF_GAIN_FULL


def test_start_uploads_one_new_quarter_turn_constant_effect_at_zero(tmp_path, evdev):
    started_output(tmp_path)
    assert len(evdev.uploads) == 1
    effect = evdev.uploads[0]
    assert (effect.type, effect.id, effect.direction) == (LINUX_FF_CONSTANT, -1, QUARTER_TURN_DIRECTION)
    assert effect.u.constant.level == 0


def test_start_plays_the_effect_id_the_kernel_assigned(tmp_path, evdev):
    started_output(tmp_path)
    play = evdev.events[-1]
    assert (play.type, play.code, play.value) == (LINUX_EV_FF, FAKE_EFFECT_ID, 1)
    assert evdev.calls.index("upload") < len(evdev.calls) - 1


@pytest.mark.parametrize("range_deg", [450, 899, 901, 1080])
def test_start_refuses_any_range_but_900(tmp_path, evdev, range_deg):
    output = EvdevForceFeedback(sysfs_input_dir=str(make_sysfs(tmp_path, range_deg=range_deg)))
    with pytest.raises(RuntimeError, match="900"):
        output.start()
    assert evdev.opened == []


class FailingEvdev(RecordingEvdev):
    """The node opens, then the kernel refuses the setup at the named step."""

    def __init__(self, failing_call):
        super().__init__()
        self.failing_call = failing_call

    def write(self, fd, data):
        if self.failing_call == "write":
            raise OSError("EIO")
        return super().write(fd, data)

    def ioctl(self, fd, request, arg=0, mutate_flag=True):
        if self.failing_call == "upload" and request == ffb.EVDEV_EVIOCSFF:
            raise OSError("ENOSPC")
        return super().ioctl(fd, request, arg, mutate_flag)


@pytest.mark.parametrize("failing_call", ["write", "upload"])
def test_start_closes_the_node_when_setup_fails(tmp_path, monkeypatch, failing_call):
    recorder = FailingEvdev(failing_call)
    fake_os = types.SimpleNamespace(
        path=os.path, listdir=os.listdir, O_RDWR=os.O_RDWR, open=recorder.open, write=recorder.write,
        close=recorder.close,
    )
    monkeypatch.setattr(ffb, "os", fake_os)
    monkeypatch.setattr(ffb, "fcntl", types.SimpleNamespace(ioctl=recorder.ioctl))
    output = EvdevForceFeedback(sysfs_input_dir=str(make_sysfs(tmp_path)))
    with pytest.raises(OSError):
        output.start()
    assert recorder.opened and recorder.closed == [FAKE_FD]
    with pytest.raises(RuntimeError):
        output.send(0.1)
    output.stop()
    assert recorder.closed == [FAKE_FD]


def test_start_without_a_wheel_raises_and_opens_nothing(tmp_path, evdev):
    output = EvdevForceFeedback(sysfs_input_dir=str(make_sysfs(tmp_path, with_wheel=False)))
    with pytest.raises(RuntimeError):
        output.start()
    assert evdev.opened == []


@pytest.mark.parametrize("gain", [0.0, -0.1, 1.01, math.nan])
def test_rejects_force_gain_outside_the_unit_interval(gain):
    with pytest.raises(ValueError):
        EvdevForceFeedback(force_gain=gain)


def test_send_before_start_raises():
    with pytest.raises(RuntimeError):
        EvdevForceFeedback().send(0.5)


@pytest.mark.parametrize("force", [1.0001, -1.0001, 5.0, math.nan, math.inf])
def test_send_rejects_force_outside_the_unit_range(tmp_path, evdev, force):
    output = started_output(tmp_path)
    with pytest.raises(ValueError):
        output.send(force)


@pytest.mark.parametrize("force", [1.0, -1.0, 0.5, -0.25, 0.0])
@pytest.mark.parametrize("gain", [1.0, 0.55])
def test_send_rewrites_the_same_effect_with_the_negated_scaled_level(tmp_path, evdev, force, gain):
    output = started_output(tmp_path, gain=gain)
    output.send(force)
    effect = evdev.uploads[-1]
    assert effect.id == FAKE_EFFECT_ID
    assert (effect.type, effect.direction) == (LINUX_FF_CONSTANT, QUARTER_TURN_DIRECTION)
    assert effect.u.constant.level == int(-force * gain * DEVICE_FULL_SCALE)


def test_full_force_fits_in_a_signed_level(tmp_path, evdev):
    output = started_output(tmp_path)
    output.send(1.0)
    output.send(-1.0)
    assert [effect.u.constant.level for effect in evdev.uploads[-2:]] == [-DEVICE_FULL_SCALE, DEVICE_FULL_SCALE]


def test_stop_removes_the_effect_and_closes(tmp_path, evdev):
    output = started_output(tmp_path)
    output.stop()
    assert evdev.removed == [FAKE_EFFECT_ID]
    assert evdev.closed == [FAKE_FD]
    assert evdev.calls[-2:] == ["remove", "close"]
    with pytest.raises(RuntimeError):
        output.send(0.1)


def test_stop_twice_and_stop_before_start_are_harmless(tmp_path, evdev):
    EvdevForceFeedback().stop()
    output = started_output(tmp_path)
    output.stop()
    output.stop()
    assert evdev.closed == [FAKE_FD]


class FakeHumanEnv:
    def __init__(self, human_state):
        self.human_state = human_state

    def get_human_state(self):
        return self.human_state

    def get_assist_state(self):
        return {"mode": 0, "reference_valid": 0, "reference_steering_rad": 0.0}


def sim_feel(sim_steering_angle_rad):
    loop = WheelDemoLoop.__new__(WheelDemoLoop)
    loop.env = FakeHumanEnv(
        {
            "steering_angle_rad": sim_steering_angle_rad,
            "accel_lat_mps2": 0.0,
            "speed_mps": PARKING_SPEED_MPS,
            "collision": 0.0,
            "offroad": 0.0,
        }
    )
    loop._last_steering_angle_rad = 0.0
    loop._last_feel_time_s = time.perf_counter()
    loop.car_feel()
    return loop.car_feel()


@pytest.mark.parametrize("turn", [SIM_LEFT, SIM_RIGHT], ids=["steered_left", "steered_right"])
def test_evdev_pushes_a_held_wheel_back_to_centre(tmp_path, evdev, turn):
    """Sim state -> car_feel -> compute_force -> the level the kernel receives, in driver terms."""
    output = started_output(tmp_path)
    output.send(compute_force(sim_feel(turn * HELD_STEERING_RAD), STEERING_LIMIT_RAD))
    level = evdev.uploads[-1].u.constant.level
    driver_holds_towards_norm = -turn
    wheel_pushed_towards_norm = (1 if level > 0 else -1) * POSITIVE_LEVEL_TURNS_WHEEL_TOWARDS_NORM
    assert level != 0
    assert wheel_pushed_towards_norm == -driver_holds_towards_norm


# ---------------------------------------------------------------------------------------------
# platform_force_output.
# ---------------------------------------------------------------------------------------------


def test_platform_output_is_evdev_on_linux(monkeypatch, evdev):
    monkeypatch.setattr(sys, "platform", "linux")
    output = platform_force_output(force_gain=0.4)
    assert isinstance(output, EvdevForceFeedback)
    assert output.force_gain == 0.4
    assert output.sysfs_input_dir == ffb.SYSFS_INPUT_DIR
    assert evdev.opened == []


def test_platform_output_is_hidpp_with_a_900_degree_range_elsewhere(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    output = platform_force_output(force_gain=0.4)
    assert isinstance(output, G920ForceFeedback)
    assert output.force_gain == 0.4
    assert output.wheel_range_deg == G920_RANGE_DEG
    assert output._device is None
