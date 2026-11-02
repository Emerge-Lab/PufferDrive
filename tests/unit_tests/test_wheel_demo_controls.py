"""Wheel buttons and demo commands, with no wheel attached: the button map, the press-edge
command stream, the force-feedback gain steps, the assist cycle, the CLI defaults, and the
probe's --only re-capture (driven through a fake device and a scripted input())."""

import argparse
import builtins
import importlib.util
import json
import os

import pytest

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo.input.base import HAT_NEUTRAL, WheelState
from pufferlib.wheel_demo.input.g920 import (
    BUTTON_MASK_BIT_COUNT,
    DPAD_DIRECTION_NAMES,
    FACE_BUTTON_NAMES,
    G920_REPORT_ID,
    G920_REPORT_LENGTH,
    ButtonMap,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMITTED_BUTTON_MAP_PATH = os.path.join(REPO_ROOT, "wheel_buttons.json")
CONTRACT_FACE_BUTTONS = ("a", "b", "x", "y", "view", "menu", "lsb", "rsb")
CONTRACT_DPAD = ("up", "right", "down", "left")
CONTRACT_COMMANDS = {
    "a": "cycle_camera",
    "b": "toggle_night",
    "x": "cycle_level",
    "y": "next_agent",
    "view": "restart_engine",
    "menu": "cycle_assist",
    "dpad_up": "ffb_gain_up",
    "dpad_down": "ffb_gain_down",
    "dpad_right": "cycle_style",
    "dpad_left": "cycle_full_steering",
}
CONTRACT_GAIN_STEP = 0.05
CONTRACT_GAIN_MIN = 0.05
CONTRACT_GAIN_MAX = 1.0
CONTRACT_DEFAULT_FFB_GAIN = 0.7
HAT_DIRECTION_COUNT = 8
MAPPED_FIELD_COUNT = 14
TEST_BUTTON_MAP = ButtonMap(
    shift_up_bit=4, shift_down_bit=5, a_bit=0, b_bit=1, x_bit=2, y_bit=3, view_bit=7, menu_bit=6, lsb_bit=9,
    rsb_bit=8, dpad_up_hat=0, dpad_right_hat=2, dpad_down_hat=4, dpad_left_hat=6,
)


def load_script(name):
    path = os.path.join(REPO_ROOT, "scripts", "wheel_demo", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"wheel_demo_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_demo():
    return load_script("run_demo")


@pytest.fixture(scope="module")
def probe():
    return load_script("probe_g920")


# ---------------------------------------------------------------------------------------------
# ButtonMap
# ---------------------------------------------------------------------------------------------


def test_button_names_are_the_contracts():
    assert FACE_BUTTON_NAMES == CONTRACT_FACE_BUTTONS
    assert DPAD_DIRECTION_NAMES == CONTRACT_DPAD
    fields = set(ButtonMap.__dataclass_fields__)
    assert {f"{name}_bit" for name in CONTRACT_FACE_BUTTONS} <= fields
    assert {f"dpad_{direction}_hat" for direction in CONTRACT_DPAD} <= fields


@pytest.mark.parametrize("name", CONTRACT_FACE_BUTTONS)
@pytest.mark.parametrize("bit", [-1, BUTTON_MASK_BIT_COUNT, 64])
def test_face_button_bit_outside_the_mask_is_rejected(name, bit):
    with pytest.raises(ValueError):
        ButtonMap(**{f"{name}_bit": bit})


@pytest.mark.parametrize("direction", CONTRACT_DPAD)
@pytest.mark.parametrize("hat", [-1, HAT_DIRECTION_COUNT, HAT_NEUTRAL])
def test_dpad_hat_outside_the_eight_directions_is_rejected(direction, hat):
    with pytest.raises(ValueError):
        ButtonMap(**{f"dpad_{direction}_hat": hat})


@pytest.mark.parametrize(
    "first,second", [("a_bit", "b_bit"), ("shift_up_bit", "menu_bit"), ("lsb_bit", "shift_down_bit"), ("x_bit", "rsb_bit")]
)
def test_two_controls_on_one_bit_are_rejected(first, second):
    with pytest.raises(ValueError):
        ButtonMap(**{first: 3, second: 3})


@pytest.mark.parametrize(
    "first,second", [("up", "down"), ("right", "left"), ("up", "left"), ("down", "right")]
)
@pytest.mark.parametrize("hat", [0, 3, HAT_DIRECTION_COUNT - 1])
def test_two_dpad_directions_on_one_hat_value_are_rejected(first, second, hat):
    with pytest.raises(ValueError):
        ButtonMap(**{f"dpad_{first}_hat": hat, f"dpad_{second}_hat": hat})


def test_a_hat_value_may_equal_a_button_bit():
    ButtonMap(a_bit=2, dpad_up_hat=2)


def test_every_hat_value_and_bit_edge_is_accepted():
    ButtonMap(a_bit=0, b_bit=BUTTON_MASK_BIT_COUNT - 1, dpad_up_hat=0, dpad_left_hat=HAT_DIRECTION_COUNT - 1)


def test_pressed_reports_held_buttons_and_the_dpad_direction():
    buttons = (1 << TEST_BUTTON_MAP.a_bit) | (1 << TEST_BUTTON_MAP.menu_bit) | (1 << TEST_BUTTON_MAP.shift_up_bit)
    assert TEST_BUTTON_MAP.pressed(buttons, TEST_BUTTON_MAP.dpad_down_hat) == frozenset({"a", "menu", "dpad_down"})


def test_pressed_is_empty_at_rest():
    assert TEST_BUTTON_MAP.pressed(0, HAT_NEUTRAL) == frozenset()


@pytest.mark.parametrize("name", CONTRACT_FACE_BUTTONS)
def test_each_face_button_reads_alone(name):
    bit = getattr(TEST_BUTTON_MAP, f"{name}_bit")
    assert TEST_BUTTON_MAP.pressed(1 << bit, HAT_NEUTRAL) == frozenset({name})


@pytest.mark.parametrize("direction", CONTRACT_DPAD)
def test_each_dpad_direction_reads_alone(direction):
    hat = getattr(TEST_BUTTON_MAP, f"dpad_{direction}_hat")
    assert TEST_BUTTON_MAP.pressed(0, hat) == frozenset({f"dpad_{direction}"})


def test_paddles_are_not_face_buttons():
    paddles = (1 << TEST_BUTTON_MAP.shift_up_bit) | (1 << TEST_BUTTON_MAP.shift_down_bit)
    assert TEST_BUTTON_MAP.pressed(paddles, HAT_NEUTRAL) == frozenset()


def test_unmapped_controls_never_read_as_pressed():
    every_bit = (1 << BUTTON_MASK_BIT_COUNT) - 1
    for hat in range(HAT_DIRECTION_COUNT):
        assert ButtonMap().pressed(every_bit, hat) == frozenset()


def test_committed_button_map_loads_with_every_control_mapped():
    button_map = ButtonMap.from_json(COMMITTED_BUTTON_MAP_PATH)
    with open(COMMITTED_BUTTON_MAP_PATH) as handle:
        stored = json.load(handle)
    assert len(stored) == MAPPED_FIELD_COUNT
    assert all(getattr(button_map, field) is not None for field in stored)
    bits = [getattr(button_map, field) for field in stored if field.endswith("_bit")]
    hats = [getattr(button_map, field) for field in stored if field.endswith("_hat")]
    assert len(set(bits)) == len(bits)
    assert len(set(hats)) == len(hats)
    button_map.require_shift_bits()


def test_committed_dpad_hats_follow_the_clockwise_hat_convention():
    button_map = ButtonMap.from_json(COMMITTED_BUTTON_MAP_PATH)
    hats = [getattr(button_map, f"dpad_{direction}_hat") for direction in CONTRACT_DPAD]
    assert hats == [0, 2, 4, 6]


def test_full_button_map_round_trips_through_json(tmp_path):
    path = tmp_path / "buttons.json"
    TEST_BUTTON_MAP.to_json(path)
    assert ButtonMap.from_json(path) == TEST_BUTTON_MAP


# ---------------------------------------------------------------------------------------------
# WheelButtonCommands: commands fire on the press edge only.
# ---------------------------------------------------------------------------------------------


def wheel_with(*names):
    buttons, hat = 0, HAT_NEUTRAL
    for name in names:
        if name.startswith("dpad_"):
            hat = getattr(TEST_BUTTON_MAP, f"{name}_hat")
        else:
            buttons |= 1 << getattr(TEST_BUTTON_MAP, f"{name}_bit")
    return WheelState(buttons=buttons, hat=hat)


def test_button_command_table_is_the_contracts(run_demo):
    assert run_demo.WHEEL_BUTTON_COMMANDS == CONTRACT_COMMANDS


@pytest.mark.parametrize("name,command", sorted(CONTRACT_COMMANDS.items()))
def test_a_press_emits_its_command_once_and_holding_does_not_repeat(run_demo, name, command):
    commands = run_demo.WheelButtonCommands(TEST_BUTTON_MAP)
    assert commands.poll(wheel_with()) == {}
    assert commands.poll(wheel_with(name)) == {command: True}
    for _ in range(5):
        assert commands.poll(wheel_with(name)) == {}
    assert commands.poll(wheel_with()) == {}
    assert commands.poll(wheel_with(name)) == {command: True}


def test_a_button_held_from_the_first_poll_counts_as_a_press(run_demo):
    commands = run_demo.WheelButtonCommands(TEST_BUTTON_MAP)
    assert commands.poll(wheel_with("menu")) == {"cycle_assist": True}


def test_a_second_button_pressed_while_one_is_held_fires_alone(run_demo):
    commands = run_demo.WheelButtonCommands(TEST_BUTTON_MAP)
    commands.poll(wheel_with("a"))
    assert commands.poll(wheel_with("a", "b")) == {"toggle_night": True}
    assert commands.poll(wheel_with("b")) == {}


def test_dpad_sweeping_up_to_down_fires_each_direction(run_demo):
    commands = run_demo.WheelButtonCommands(TEST_BUTTON_MAP)
    assert commands.poll(wheel_with("dpad_up")) == {"ffb_gain_up": True}
    assert commands.poll(wheel_with("dpad_down")) == {"ffb_gain_down": True}
    assert commands.poll(wheel_with("dpad_up")) == {"ffb_gain_up": True}


def test_unbound_buttons_and_paddles_emit_nothing(run_demo):
    commands = run_demo.WheelButtonCommands(TEST_BUTTON_MAP)
    assert commands.poll(wheel_with("lsb", "rsb")) == {}
    paddles = (1 << TEST_BUTTON_MAP.shift_up_bit) | (1 << TEST_BUTTON_MAP.shift_down_bit)
    assert commands.poll(WheelState(buttons=paddles)) == {}


# ---------------------------------------------------------------------------------------------
# Force-feedback gain steps and the assist cycle.
# ---------------------------------------------------------------------------------------------


class FakeOutput:
    def __init__(self, force_gain):
        self.force_gain = force_gain


class FakeForceLoop:
    def __init__(self, force_gain):
        self.output = FakeOutput(force_gain)


def test_gain_step_constants_are_the_contracts(run_demo):
    assert (run_demo.FFB_GAIN_STEP, run_demo.FFB_GAIN_MIN, run_demo.FFB_GAIN_MAX) == (
        CONTRACT_GAIN_STEP,
        CONTRACT_GAIN_MIN,
        CONTRACT_GAIN_MAX,
    )


def test_gain_steps_up_by_a_twentieth_and_stops_at_full(run_demo):
    force_loop = FakeForceLoop(CONTRACT_DEFAULT_FFB_GAIN)
    seen = []
    for _ in range(10):
        run_demo.adjust_ffb_gain(force_loop, CONTRACT_GAIN_STEP)
        seen.append(force_loop.output.force_gain)
    assert seen[:6] == pytest.approx([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    assert seen[6:] == [CONTRACT_GAIN_MAX] * 4


def test_gain_steps_down_and_never_reaches_zero(run_demo):
    force_loop = FakeForceLoop(0.2)
    for _ in range(10):
        run_demo.adjust_ffb_gain(force_loop, -CONTRACT_GAIN_STEP)
        assert force_loop.output.force_gain >= CONTRACT_GAIN_MIN
    assert force_loop.output.force_gain == pytest.approx(CONTRACT_GAIN_MIN)


def test_gain_steps_land_on_exact_twentieths(run_demo):
    force_loop = FakeForceLoop(CONTRACT_GAIN_MIN)
    for twentieths in range(2, 21):
        run_demo.adjust_ffb_gain(force_loop, CONTRACT_GAIN_STEP)
        assert force_loop.output.force_gain == round(twentieths * CONTRACT_GAIN_STEP, 2)


def test_gain_steps_do_not_drift_over_many_presses(run_demo):
    force_loop = FakeForceLoop(CONTRACT_DEFAULT_FFB_GAIN)
    for _ in range(37):
        run_demo.adjust_ffb_gain(force_loop, CONTRACT_GAIN_STEP)
        run_demo.adjust_ffb_gain(force_loop, -CONTRACT_GAIN_STEP)
    assert force_loop.output.force_gain == CONTRACT_DEFAULT_FFB_GAIN


def test_gain_steps_keep_every_real_output_valid(run_demo):
    from pufferlib.wheel_demo.ffb import EvdevForceFeedback, G920ForceFeedback

    for output_class in (EvdevForceFeedback, G920ForceFeedback):
        force_loop = FakeForceLoop(0.0)
        force_loop.output = output_class(force_gain=CONTRACT_GAIN_MIN)
        for step in [-CONTRACT_GAIN_STEP] * 3 + [CONTRACT_GAIN_STEP] * 30:
            run_demo.adjust_ffb_gain(force_loop, step)
            assert 0.0 < force_loop.output.force_gain <= 1.0


def test_gain_step_without_force_feedback_is_a_no_op(run_demo):
    run_demo.adjust_ffb_gain(None, CONTRACT_GAIN_STEP)


class AssistEnv:
    def __init__(self):
        self.mode = binding.ASSIST_MODE_OFF
        self.calls = []
        self.engine_restarts = 0

    def get_assist_state(self):
        return {"mode": self.mode}

    def set_assist_mode(self, mode, blend_target):
        self.calls.append((mode, blend_target))
        self.mode = mode

    def restart_human_engine(self):
        self.engine_restarts += 1


class FakeLoop:
    def __init__(self):
        self.env = AssistEnv()
        self.human_agent_idx = 0


class FakeWindow:
    def __init__(self, keys=None):
        self.keys = dict(keys or {})
        self.camera_mode = 0
        self.camera_modes_set = []
        self.night_toggles = 0

    def poll(self):
        return dict(self.keys)

    def set_camera_mode(self, camera_mode):
        self.camera_modes_set.append(camera_mode)
        self.camera_mode = camera_mode

    def toggle_night(self):
        self.night_toggles += 1


def test_assist_cycles_off_display_blend_nudge_full_off(run_demo):
    loop = FakeLoop()
    seen = []
    for _ in range(6):
        run_demo.cycle_assist(loop, 0.4)
        seen.append(loop.env.mode)
    assert seen == [
        binding.ASSIST_MODE_DISPLAY,
        binding.ASSIST_MODE_BLEND,
        binding.ASSIST_MODE_NUDGE,
        binding.ASSIST_MODE_FULL,
        binding.ASSIST_MODE_OFF,
        binding.ASSIST_MODE_DISPLAY,
    ]
    assert all(target == 0.4 for _, target in loop.env.calls)


def test_menu_button_and_m_key_cycle_the_assist(run_demo):
    loop = FakeLoop()
    run_demo.apply_window_keys(FakeWindow(), loop, 0, {"cycle_assist": True}, None, 0.6)
    assert loop.env.calls == [(binding.ASSIST_MODE_DISPLAY, 0.6)]
    run_demo.apply_window_keys(FakeWindow({"cycle_assist": True}), loop, 0, None, None, 0.6)
    assert loop.env.calls[-1] == (binding.ASSIST_MODE_BLEND, 0.6)


def test_m_key_is_bound_to_the_assist_cycle():
    from pufferlib.wheel_demo import carla_window

    source = open(carla_window.__file__).read()
    assert "pygame.K_m: \"cycle_assist\"" in source


def test_a_button_cycles_the_camera_through_all_three_views(run_demo):
    window = FakeWindow()
    for _ in range(4):
        run_demo.apply_window_keys(window, FakeLoop(), 0, {"cycle_camera": True}, None, 0.5)
    assert window.camera_modes_set == [1, 2, 0, 1]


def test_dpad_steps_the_force_feedback_gain_through_window_keys(run_demo):
    force_loop = FakeForceLoop(CONTRACT_DEFAULT_FFB_GAIN)
    run_demo.apply_window_keys(FakeWindow(), FakeLoop(), 0, {"ffb_gain_up": True}, force_loop, 0.5)
    assert force_loop.output.force_gain == pytest.approx(0.75)
    run_demo.apply_window_keys(FakeWindow(), FakeLoop(), 0, {"ffb_gain_down": True}, force_loop, 0.5)
    run_demo.apply_window_keys(FakeWindow(), FakeLoop(), 0, {"ffb_gain_down": True}, force_loop, 0.5)
    assert force_loop.output.force_gain == pytest.approx(0.65)


def test_view_and_b_buttons_restart_the_engine_and_toggle_night(run_demo):
    window, loop = FakeWindow(), FakeLoop()
    run_demo.apply_window_keys(window, loop, 0, {"restart_engine": True, "toggle_night": True}, None, 0.5)
    assert loop.env.engine_restarts == 1
    assert window.night_toggles == 1


def test_wheel_commands_do_not_leak_into_the_next_poll(run_demo):
    window, loop = FakeWindow(), FakeLoop()
    run_demo.apply_window_keys(window, loop, 0, {"cycle_assist": True}, None, 0.5)
    run_demo.apply_window_keys(window, loop, 0, None, None, 0.5)
    assert len(loop.env.calls) == 1


# ---------------------------------------------------------------------------------------------
# CLI defaults. main() builds its own parser, so it is captured at parse_args.
# ---------------------------------------------------------------------------------------------


class ParserCaptured(Exception):
    pass


@pytest.fixture(scope="module")
def demo_parser(run_demo):
    captured = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise ParserCaptured

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(ParserCaptured):
            run_demo.main()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


def test_ffb_gain_defaults_to_the_force_models_default(demo_parser):
    from pufferlib.wheel_demo.ffb import DEFAULT_FORCE_GAIN

    assert demo_parser.get_default("ffb_gain") == DEFAULT_FORCE_GAIN == CONTRACT_DEFAULT_FFB_GAIN


def test_assist_flag_offers_the_five_modes_and_starts_off(demo_parser, run_demo):
    action = next(a for a in demo_parser._actions if a.dest == "assist")
    assert set(action.choices) == {"off", "display", "blend", "nudge", "full"}
    assert action.default == "off"
    assert demo_parser.parse_args(["--assist", "full"]).assist == "full"
    assert run_demo.ASSIST_MODES_BY_NAME == {
        "off": binding.ASSIST_MODE_OFF,
        "display": binding.ASSIST_MODE_DISPLAY,
        "blend": binding.ASSIST_MODE_BLEND,
        "nudge": binding.ASSIST_MODE_NUDGE,
        "full": binding.ASSIST_MODE_FULL,
    }


def test_assist_blend_flag_parses_a_float(demo_parser):
    assert demo_parser.parse_args(["--assist", "blend", "--assist-blend", "0.8"]).assist_blend == 0.8
    with pytest.raises(SystemExit):
        demo_parser.parse_args(["--assist", "always"])


# ---------------------------------------------------------------------------------------------
# probe_g920 buttons --only, against a fake device and a scripted operator.
# ---------------------------------------------------------------------------------------------


def report_holding(buttons=0, hat=HAT_NEUTRAL):
    """A G920 axis report with the given 20-bit mask and hat (layout as decode_button_mask reads it)."""
    report = bytearray(G920_REPORT_LENGTH)
    report[0] = G920_REPORT_ID
    report[1] = ((buttons & 0xF) << 4) | hat
    report[2] = (buttons >> 4) & 0xFF
    report[3] = (buttons >> 12) & 0xFF
    return bytes(report)


class ScriptedOperator:
    """Answers each probe prompt by holding the control it names, as wired on TEST_WIRING."""

    def __init__(self, wiring):
        self.wiring = wiring
        self.held = report_holding()
        self.prompts = []

    def input(self, message):
        self.prompts.append(message)
        self.held = report_holding()
        for phrase, report in self.wiring.items():
            if phrase in message:
                self.held = report
        return ""

    def read(self, length):
        return self.held


TEST_WIRING = {
    "RIGHT paddle": report_holding(buttons=1 << 4),
    "LEFT paddle": report_holding(buttons=1 << 5),
    "A button (green)": report_holding(buttons=1 << 11),
    "B button (red)": report_holding(buttons=1 << 1),
    "D-pad UP": report_holding(hat=0),
    "D-pad DOWN": report_holding(hat=4),
}


@pytest.fixture
def scripted_probe(probe, monkeypatch):
    operator = ScriptedOperator(TEST_WIRING)
    monkeypatch.setattr(builtins, "input", operator.input)
    monkeypatch.setattr(probe, "SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(probe, "CAPTURE_SECONDS", 0.005)
    return probe, operator


def test_only_recaptures_the_named_fields_and_keeps_the_rest(scripted_probe, tmp_path):
    probe, operator = scripted_probe
    path = tmp_path / "buttons.json"
    TEST_BUTTON_MAP.to_json(path)
    probe.probe_buttons(operator, path, only={"a_bit", "dpad_down_hat"})
    updated = ButtonMap.from_json(path)
    assert updated.a_bit == 11
    assert updated.dpad_down_hat == 4
    unchanged = {name: getattr(TEST_BUTTON_MAP, name) for name in ButtonMap.__dataclass_fields__}
    unchanged.update(a_bit=11)
    assert updated == ButtonMap(**unchanged)
    asked_for = [message for message in operator.prompts if "Hold" in message]
    assert len(asked_for) == 2
    assert "A button" in asked_for[0]
    assert "D-pad DOWN" in asked_for[1]


def test_without_only_every_control_is_asked_for(scripted_probe, tmp_path):
    probe, operator = scripted_probe
    path = tmp_path / "buttons.json"
    probe.probe_buttons(operator, path)
    asked_for = [message for message in operator.prompts if "Hold" in message]
    assert len(asked_for) == 2 + len(CONTRACT_FACE_BUTTONS) + len(CONTRACT_DPAD)
    captured = ButtonMap.from_json(path)
    assert (captured.shift_up_bit, captured.shift_down_bit, captured.a_bit, captured.b_bit) == (4, 5, 11, 1)
    assert (captured.dpad_up_hat, captured.dpad_down_hat) == (0, 4)
    assert captured.x_bit is None
    assert captured.dpad_left_hat is None


def test_a_failed_recapture_keeps_the_existing_mapping(scripted_probe, tmp_path):
    probe, operator = scripted_probe
    path = tmp_path / "buttons.json"
    TEST_BUTTON_MAP.to_json(path)
    probe.probe_buttons(operator, path, only={"x_bit", "dpad_left_hat"})
    assert ButtonMap.from_json(path) == TEST_BUTTON_MAP


def test_a_failed_recapture_of_an_unmapped_field_leaves_it_unmapped(scripted_probe, tmp_path):
    probe, operator = scripted_probe
    path = tmp_path / "buttons.json"
    ButtonMap(shift_up_bit=4, shift_down_bit=5).to_json(path)
    probe.probe_buttons(operator, path, only={"x_bit", "a_bit"})
    assert ButtonMap.from_json(path) == ButtonMap(shift_up_bit=4, shift_down_bit=5, a_bit=11)


# ---------------------------------------------------------------------------------------------
# Full screen: the flag, its forwarding into the window, and the set_mode flags. No real window.
# ---------------------------------------------------------------------------------------------


def test_fullscreen_flag_is_off_unless_given(demo_parser):
    assert demo_parser.parse_args([]).fullscreen is False
    assert demo_parser.parse_args(["--fullscreen"]).fullscreen is True
    with pytest.raises(SystemExit):
        demo_parser.parse_args(["--fullscreen", "yes"])


class WindowBuilt(Exception):
    pass


class FakeDemoLoop:
    def __init__(self, config, wheel):
        self.env = AssistEnv()
        self.human_agent_idx = 0
        self.substep_dt = 0.05


@pytest.mark.parametrize("extra_args,expected", [(["--fullscreen"], True), ([], False)])
def test_main_forwards_fullscreen_to_the_window(run_demo, monkeypatch, tmp_path, extra_args, expected):
    built = {}

    def fake_window(*args, **kwargs):
        built.update(kwargs)
        raise WindowBuilt

    monkeypatch.setattr(run_demo, "build_wheel_input", lambda *args: (object(), None))
    monkeypatch.setattr(run_demo, "WheelDemoLoop", FakeDemoLoop)
    monkeypatch.setattr(run_demo, "choose_town", lambda town: town or "Town01")
    monkeypatch.setattr(run_demo, "CarlaDemoWindow", fake_window)
    argv = ["run_demo.py", "--input", "synthetic", "--render", "--town", "Town01", "--calibration",
            str(tmp_path / "none.json"), "--buttons", str(tmp_path / "none_buttons.json")] + extra_args
    monkeypatch.setattr(run_demo.sys, "argv", argv)
    with pytest.raises(WindowBuilt):
        run_demo.main()
    assert built["fullscreen"] is expected


class FakeScene:
    instances = []

    def __init__(self, carla, town, env, human_agent_idx, substep_dt_s, host, port, image_size_px):
        self.closed = False
        self.camera_mode = None
        FakeScene.instances.append(self)

    def set_night(self, night):
        self.night = night

    def set_camera_mode(self, camera_mode):
        self.camera_mode = camera_mode

    def close(self):
        self.closed = True


@pytest.fixture
def headless_window_module(monkeypatch):
    import sys
    import types

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from pufferlib.wheel_demo import carla_window

    calls = []

    def set_mode(size_px, flags=0, *args, **kwargs):
        calls.append((tuple(size_px), flags))
        return pygame.Surface(size_px)

    monkeypatch.setitem(sys.modules, "carla", types.ModuleType("carla"))
    monkeypatch.setattr(carla_window, "CarlaScene", FakeScene)
    monkeypatch.setattr(pygame.display, "set_mode", set_mode)
    monkeypatch.setattr(pygame.display, "set_caption", lambda *args: None)
    FakeScene.instances.clear()
    return carla_window, pygame, calls


@pytest.mark.parametrize("fullscreen", [True, False])
def test_window_asks_for_scaled_full_screen_only_when_told(headless_window_module, fullscreen):
    carla_window, pygame, calls = headless_window_module
    window = carla_window.CarlaDemoWindow(object(), "Town01", 0, 0.05, fullscreen=fullscreen)
    expected_flags = pygame.FULLSCREEN | pygame.SCALED if fullscreen else 0
    assert calls == [(carla_window.DEFAULT_WINDOW_SIZE_PX, expected_flags)]
    assert window.screen.get_size() == carla_window.DEFAULT_WINDOW_SIZE_PX
    assert not FakeScene.instances[-1].closed


def test_window_defaults_to_windowed(headless_window_module):
    carla_window, _pygame, calls = headless_window_module
    carla_window.CarlaDemoWindow(object(), "Town01", 0, 0.05)
    assert calls[-1][1] == 0


def test_window_closes_the_scene_when_full_screen_fails(headless_window_module, monkeypatch):
    carla_window, pygame, _calls = headless_window_module

    def refuse(*args, **kwargs):
        raise pygame.error("no full-screen mode")

    monkeypatch.setattr(pygame.display, "set_mode", refuse)
    with pytest.raises(pygame.error):
        carla_window.CarlaDemoWindow(object(), "Town01", 0, 0.05, fullscreen=True)
    assert FakeScene.instances[-1].closed


# ---------------------------------------------------------------------------------------------
# Driving style and FULL steering presets: cycled from the wheel, refused outside FULL.
# ---------------------------------------------------------------------------------------------

CONTRACT_DRIVING_STYLES = ("off", "calm", "normal", "aggressive")
CONTRACT_FULL_STEERING_PRESETS = {
    "car": (binding.FULL_ASSIST_DYNAMICS_SET_STEERING, 0.6),
    "wheel": (binding.FULL_ASSIST_DYNAMICS_JERK_SPEED, 0.9),
}
STYLE_LIMIT_LOW = 0.0
STYLE_LIMIT_HIGH = 1.0


class StyleEnv(AssistEnv):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.style_calls = []

    def get_reward_coef_limits(self):
        import numpy as np

        return (np.full(binding.NUM_REWARD_COEFS, STYLE_LIMIT_LOW, dtype=np.float32),
                np.full(binding.NUM_REWARD_COEFS, STYLE_LIMIT_HIGH, dtype=np.float32))

    def set_human_driving_style(self, min_values=None, max_values=None):
        self.style_calls.append(None if min_values is None else (min_values.copy(), max_values.copy()))


class StyleLoop:
    def __init__(self, mode):
        self.env = StyleEnv(mode)
        self.human_agent_idx = 0
        self.full_steering = "car"
        self.full_steering_calls = []

    def set_full_steering(self, full_assist_dynamics, lookahead_window_seconds):
        self.full_steering_calls.append((full_assist_dynamics, lookahead_window_seconds))
        self.full_steering = next((name for name, preset in CONTRACT_FULL_STEERING_PRESETS.items()
                                   if preset == (full_assist_dynamics, lookahead_window_seconds)), "custom")


def test_driving_styles_are_off_then_the_levels(run_demo):
    assert run_demo.DRIVING_STYLES == CONTRACT_DRIVING_STYLES


def test_driving_style_cycles_off_calm_normal_aggressive_off_under_full(run_demo):
    from pufferlib.wheel_demo import levels

    loop = StyleLoop(binding.ASSIST_MODE_FULL)
    style_state = {"index": 0}
    seen = []
    for _ in range(len(CONTRACT_DRIVING_STYLES)):
        run_demo.cycle_driving_style(loop, style_state)
        seen.append(run_demo.DRIVING_STYLES[style_state["index"]])
    assert seen == ["calm", "normal", "aggressive", "off"]
    assert loop.env.style_calls[-1] is None
    for call, name in zip(loop.env.style_calls[:3], seen[:3]):
        low, high = levels.level_bounds(loop.env, levels.level_by_name(name))
        assert (call[0] == low).all() and (call[1] == high).all()


@pytest.mark.parametrize(
    "mode", [binding.ASSIST_MODE_OFF, binding.ASSIST_MODE_DISPLAY, binding.ASSIST_MODE_BLEND, binding.ASSIST_MODE_NUDGE]
)
def test_driving_style_is_refused_outside_full(run_demo, mode, capsys):
    loop = StyleLoop(mode)
    style_state = {"index": 1}
    run_demo.cycle_driving_style(loop, style_state)
    assert style_state == {"index": 1}
    assert loop.env.style_calls == []
    assert "FULL" in capsys.readouterr().out


def test_full_steering_presets_are_the_contract():
    from pufferlib.wheel_demo import loop as loop_module

    assert loop_module.FULL_STEERING_PRESETS == CONTRACT_FULL_STEERING_PRESETS
    assert loop_module.DEFAULT_FULL_STEERING == "car"


@pytest.mark.parametrize("start,expected", [("car", "wheel"), ("wheel", "car"), ("custom", "car")])
def test_full_steering_cycles_through_the_presets_under_full(run_demo, start, expected):
    loop = StyleLoop(binding.ASSIST_MODE_FULL)
    loop.full_steering = start
    run_demo.cycle_full_steering(loop)
    assert loop.full_steering_calls == [CONTRACT_FULL_STEERING_PRESETS[expected]]
    assert loop.full_steering == expected


@pytest.mark.parametrize(
    "mode", [binding.ASSIST_MODE_OFF, binding.ASSIST_MODE_DISPLAY, binding.ASSIST_MODE_BLEND, binding.ASSIST_MODE_NUDGE]
)
def test_full_steering_is_refused_outside_full(run_demo, mode, capsys):
    loop = StyleLoop(mode)
    run_demo.cycle_full_steering(loop)
    assert loop.full_steering_calls == []
    assert loop.full_steering == "car"
    assert "FULL" in capsys.readouterr().out


def test_window_keys_route_style_and_full_steering(run_demo):
    loop = StyleLoop(binding.ASSIST_MODE_FULL)
    style_state = {"index": 0}
    run_demo.apply_window_keys(FakeWindow(), loop, 0, {"cycle_style": True, "cycle_full_steering": True}, None, 0.5,
                               style_state)
    assert style_state == {"index": 1}
    assert loop.full_steering_calls == [CONTRACT_FULL_STEERING_PRESETS["wheel"]]


def test_window_keys_without_style_state_still_work(run_demo):
    loop = StyleLoop(binding.ASSIST_MODE_FULL)
    run_demo.apply_window_keys(FakeWindow({"cycle_style": True}), loop, 0, {"cycle_assist": True}, None, 0.5)
    assert loop.env.style_calls == []
    assert loop.env.calls == [(binding.ASSIST_MODE_OFF, 0.5)]


def test_full_steering_style_and_gearbox_flags(demo_parser, run_demo):
    defaults = demo_parser.parse_args([])
    assert defaults.full_steering == "car"
    assert defaults.full_assist_dynamics is None
    assert defaults.lookahead_window is None
    assert defaults.driving_style == "off"
    assert defaults.gearbox == "automatic"
    actions = {a.dest: a for a in demo_parser._actions}
    assert set(actions["full_steering"].choices) == set(CONTRACT_FULL_STEERING_PRESETS)
    assert tuple(actions["driving_style"].choices) == CONTRACT_DRIVING_STYLES
    assert set(actions["gearbox"].choices) == {"automatic", "manual", "none"}
    assert run_demo.GEARBOX_MODES_BY_NAME == {
        "automatic": binding.GEARBOX_MODE_AUTOMATIC,
        "manual": binding.GEARBOX_MODE_MANUAL,
        "none": binding.GEARBOX_MODE_NONE,
    }
    with pytest.raises(SystemExit):
        demo_parser.parse_args(["--driving-style", "reckless"])


class ConfigCaptured(Exception):
    pass


def captured_demo_config(run_demo, monkeypatch, tmp_path, extra_args):
    captured = {}

    def capture_loop(config, wheel):
        captured["config"] = config
        raise ConfigCaptured

    monkeypatch.setattr(run_demo, "build_wheel_input", lambda *args: (object(), None))
    monkeypatch.setattr(run_demo, "WheelDemoLoop", capture_loop)
    monkeypatch.setattr(run_demo, "choose_town", lambda town: town or "Town01")
    argv = ["run_demo.py", "--input", "synthetic", "--town", "Town01", "--calibration", str(tmp_path / "none.json"),
            "--buttons", str(tmp_path / "none_buttons.json")] + extra_args
    monkeypatch.setattr(run_demo.sys, "argv", argv)
    with pytest.raises(ConfigCaptured):
        run_demo.main()
    return captured["config"]


@pytest.mark.parametrize("preset", sorted(CONTRACT_FULL_STEERING_PRESETS))
def test_full_steering_preset_sets_dynamics_and_window(run_demo, monkeypatch, tmp_path, preset):
    config = captured_demo_config(run_demo, monkeypatch, tmp_path, ["--full-steering", preset])
    assert (config.full_assist_dynamics, config.lookahead_window_seconds) == CONTRACT_FULL_STEERING_PRESETS[preset]


@pytest.mark.parametrize(
    "name,dynamics",
    [("jerk-speed", binding.FULL_ASSIST_DYNAMICS_JERK_SPEED), ("bicycle", binding.FULL_ASSIST_DYNAMICS_HUMAN),
     ("jerk", binding.FULL_ASSIST_DYNAMICS_JERK), ("lookahead", binding.FULL_ASSIST_DYNAMICS_SET_STEERING)],
)
def test_explicit_dynamics_and_window_override_the_preset(run_demo, monkeypatch, tmp_path, name, dynamics):
    config = captured_demo_config(run_demo, monkeypatch, tmp_path,
                                  ["--full-steering", "car", "--full-assist-dynamics", name, "--lookahead-window", "1.5"])
    assert (config.full_assist_dynamics, config.lookahead_window_seconds) == (dynamics, 1.5)


@pytest.mark.parametrize(
    "name,grip_mode",
    [("realistic", binding.GRIP_MODE_REALISTIC), ("kinematic", binding.GRIP_MODE_KINEMATIC), ("skid", binding.GRIP_MODE_SKID)],
)
def test_grip_flag_reaches_the_config(run_demo, monkeypatch, tmp_path, name, grip_mode):
    from pufferlib.wheel_demo import loop as loop_module

    assert loop_module.GRIP_MODE_SKID == binding.GRIP_MODE_SKID
    assert run_demo.GRIP_MODES_BY_NAME[name] == grip_mode
    config = captured_demo_config(run_demo, monkeypatch, tmp_path, ["--grip", name])
    assert config.grip_mode == grip_mode


def test_grip_flag_offers_skid_and_defaults_to_realistic(demo_parser, run_demo):
    action = next(a for a in demo_parser._actions if a.dest == "grip")
    assert tuple(action.choices) == ("realistic", "kinematic", "skid")
    assert action.default == "realistic"
    assert "spin" in action.help
    with pytest.raises(SystemExit):
        demo_parser.parse_args(["--grip", "drift"])


def test_gearless_flag_reaches_the_config(run_demo, monkeypatch, tmp_path):
    config = captured_demo_config(run_demo, monkeypatch, tmp_path, ["--gearbox", "none"])
    assert config.gearbox_mode == binding.GEARBOX_MODE_NONE


def test_window_binds_s_and_f_and_documents_the_dpad():
    from pufferlib.wheel_demo import carla_window

    source = open(carla_window.__file__).read()
    assert "pygame.K_s: \"cycle_style\"" in source
    assert "pygame.K_f: \"cycle_full_steering\"" in source
    assert "D-pad right style" in carla_window.HELP_TEXT
    assert "D-pad left full steering" in carla_window.HELP_TEXT


class FakeFrameClock:
    def __init__(self):
        self.now_s = 0.0

    def perf_counter(self):
        return self.now_s

    def sleep(self, seconds):
        self.now_s += seconds


WINDOW_SUBSTEP_SECONDS = 0.05
SKID_MARK_DRAWN = ((1.0, 2.0), (3.0, 4.0), 2, 0.5)


@pytest.fixture
def drawing_window(headless_window_module, monkeypatch):
    carla_window, pygame, _calls = headless_window_module
    clock = FakeFrameClock()
    monkeypatch.setattr(carla_window, "time", clock)
    monkeypatch.setattr(pygame.display, "flip", lambda: None)
    window = carla_window.CarlaDemoWindow(object(), "Town01", 0, WINDOW_SUBSTEP_SECONDS)
    drawn = {"infos": [], "update_dts": [], "skid_queries": [], "skid_marks": []}

    def update(human, wheel_state, dt_s):
        drawn["update_dts"].append(dt_s)
        return {"main": None}

    window.scene.update = update
    window.scene.goal_marker = lambda human, image: None
    window.scene.skid_marks = lambda image, depth_image: drawn["skid_queries"].append((image, depth_image)) or [
        SKID_MARK_DRAWN]
    window._surface = lambda image: pygame.Surface((1, 1))
    window._mirror_layout = lambda images: []

    def hud_draw(surface, human, wheel_state, info, elapsed, **kwargs):
        drawn["infos"].append(info)
        drawn["skid_marks"].append(kwargs.get("skid_marks"))

    window.hud.draw = hud_draw
    return window, clock, drawn


def test_draw_passes_the_style_and_full_steering_to_the_hud(drawing_window):
    window, _clock, drawn = drawing_window
    assert window.draw({}, WheelState(), 0.0, driving_style="calm", full_steering="wheel")
    info = drawn["infos"][-1]
    assert (info.driving_style, info.full_steering) == ("calm", "wheel")
    window.draw({}, WheelState(), 0.0)
    assert (drawn["infos"][-1].driving_style, drawn["infos"][-1].full_steering) == ("off", "car")


def test_a_late_frame_is_skipped_and_the_next_one_covers_the_gap(drawing_window):
    window, clock, drawn = drawing_window
    window.draw({}, WheelState(), 0.0)
    clock.now_s += 10.0 * WINDOW_SUBSTEP_SECONDS
    assert window.draw({}, WheelState(), 0.0) is False
    assert window.frames_skipped == 1
    clock.now_s = window._next_frame_s
    assert window.draw({}, WheelState(), 0.0) is True
    assert drawn["update_dts"][-1] == pytest.approx(2.0 * WINDOW_SUBSTEP_SECONDS)


def test_lockstep_never_skips_a_frame(drawing_window):
    window, clock, drawn = drawing_window
    window.lockstep = True
    for _ in range(5):
        clock.now_s += 10.0 * WINDOW_SUBSTEP_SECONDS
        assert window.draw({}, WheelState(), 0.0) is True
    assert window.frames_skipped == 0
    assert drawn["update_dts"] == [pytest.approx(WINDOW_SUBSTEP_SECONDS)] * 5


def test_recording_turns_on_lockstep(headless_window_module, monkeypatch, tmp_path):
    carla_window, _pygame, _calls = headless_window_module
    started = []
    monkeypatch.setattr(carla_window.CarlaDemoWindow, "_start_recorder", lambda self, path: started.append(path))
    record_path = str(tmp_path / "demo.mp4")
    assert carla_window.CarlaDemoWindow(object(), "Town01", 0, WINDOW_SUBSTEP_SECONDS, record_path=record_path).lockstep
    assert started == [record_path]
    assert not carla_window.CarlaDemoWindow(object(), "Town01", 0, WINDOW_SUBSTEP_SECONDS).lockstep


def test_draw_hands_the_scenes_skid_marks_for_the_main_and_depth_images_to_the_hud(drawing_window):
    window, _clock, drawn = drawing_window
    images = {"main": "main image", "depth": "depth image"}
    window.scene.update = lambda human, wheel_state, dt_s: images
    window.draw({}, WheelState(), 0.0)
    assert drawn["skid_queries"] == [("main image", "depth image")]
    assert drawn["skid_marks"] == [[SKID_MARK_DRAWN]]
    images.pop("depth")
    window.draw({}, WheelState(), 0.0)
    assert drawn["skid_queries"][-1] == ("main image", None)
