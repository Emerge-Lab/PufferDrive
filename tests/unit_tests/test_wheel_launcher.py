"""The desktop launcher's pure parts: the run_demo command it builds, the towns it finds on disk, and
the desktop entry. Nothing here opens a window, starts CARLA or writes outside tmp_path."""

import argparse
import importlib.util
import itertools
import os
import shlex
import subprocess
import sys
import time

import pytest

from pufferlib.wheel_demo.maps import CARLA_TOWNS


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts", "wheel_demo")
CONTRACT_DEFAULTS = {
    "town": "Town10HD",
    "camera": "chase",
    "night": "day",
    "level": "normal",
    "assist": "off",
    "assist_blend": 50,
    "full_steering": "car",
    "driving_style": "off",
    "ffb": True,
    "ffb_gain": 70,
    "gearbox": "automatic",
    "grip": "skid",
    "input": "wheel",
    "minutes": 20,
    "window": "fullscreen",
}
CONTRACT_POLICY_STEP_SECONDS = 0.3
SECONDS_PER_MINUTE = 60
SUBPROCESS_TIMEOUT_S = 60
CONTRACT_FULL_STEERINGS = ("car", "wheel")
CONTRACT_DRIVING_STYLES = ("off", "calm", "normal", "aggressive")
CONTRACT_GEARBOXES = ("automatic", "manual", "none")
CONTRACT_GRIPS = ("realistic", "kinematic", "skid")
FULL_ONLY_ROWS = ("Self-driving steering", "Your style")


def load_script(name):
    path = os.path.join(SCRIPTS_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"wheel_demo_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def launcher():
    pytest.importorskip("tkinter")
    return load_script("launcher")


@pytest.fixture(scope="module")
def install_launcher():
    return load_script("install_launcher")


class ParserCaptured(Exception):
    pass


@pytest.fixture(scope="module")
def demo_parser():
    run_demo = load_script("run_demo")
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


def settings_with(**overrides):
    settings = dict(CONTRACT_DEFAULTS)
    settings.update(overrides)
    return settings


def flag_value(command, flag):
    assert command.count(flag) == 1
    return command[command.index(flag) + 1]


def test_defaults_are_the_contract(launcher):
    assert launcher.DEFAULTS == CONTRACT_DEFAULTS


def test_every_offered_town_is_one_run_demo_knows(launcher):
    assert {town for town, _ in launcher.TOWNS} <= set(CARLA_TOWNS)
    assert launcher.DEFAULTS["town"] in {town for town, _ in launcher.TOWNS}


def test_default_command_runs_run_demo_with_this_interpreter(launcher):
    command = launcher.demo_command(dict(CONTRACT_DEFAULTS))
    assert command[0] == sys.executable
    assert os.path.realpath(command[1]) == os.path.realpath(os.path.join(SCRIPTS_DIR, "run_demo.py"))
    assert os.path.isfile(command[1])
    assert "--render" in command
    assert flag_value(command, "--town") == "Town10HD"
    assert "--night" not in command
    assert command.count("--fullscreen") == 1
    assert flag_value(command, "--ffb-gain") == "0.70"
    assert "--ffb" in command
    assert flag_value(command, "--assist") == "off"
    assert flag_value(command, "--assist-blend") == "0.50"
    assert flag_value(command, "--full-steering") == "car"
    assert flag_value(command, "--driving-style") == "off"
    assert flag_value(command, "--steps") == "4000"


@pytest.mark.parametrize("town", CARLA_TOWNS)
def test_town_and_render_are_always_passed(launcher, town):
    command = launcher.demo_command(settings_with(town=town, input="synthetic", ffb=False))
    assert flag_value(command, "--town") == town
    assert command.count("--render") == 1


@pytest.mark.parametrize("night,expected", [("night", True), ("day", False)])
def test_night_flag_only_for_night(launcher, night, expected):
    assert ("--night" in launcher.demo_command(settings_with(night=night))) == expected


@pytest.mark.parametrize(
    "ffb,input_name,expected",
    [(True, "wheel", True), (False, "wheel", False), (True, "synthetic", False), (False, "synthetic", False),
     (True, "remote", False), (False, "remote", False)],
)
def test_force_feedback_only_with_the_wheel_and_ffb_on(launcher, ffb, input_name, expected):
    command = launcher.demo_command(settings_with(ffb=ffb, input=input_name))
    assert ("--ffb" in command) == expected
    assert ("--ffb-gain" in command) == expected


@pytest.mark.parametrize("input_name", ["wheel", "synthetic", "remote"])
def test_input_is_passed_through(launcher, input_name):
    assert flag_value(launcher.demo_command(settings_with(input=input_name)), "--input") == input_name


def test_remote_command_is_the_wheel_command_without_force_feedback(launcher):
    wheel_command = launcher.demo_command(settings_with(input="wheel", ffb=False))
    remote_command = launcher.demo_command(settings_with(input="remote", ffb=True))
    assert remote_command == [("remote" if part == "wheel" else part) for part in wheel_command]


@pytest.mark.parametrize("gain_percent,expected", [(5, "0.05"), (33, "0.33"), (70, "0.70"), (100, "1.00")])
def test_ffb_gain_is_the_percent_over_100_to_two_decimals(launcher, gain_percent, expected):
    assert flag_value(launcher.demo_command(settings_with(ffb_gain=gain_percent)), "--ffb-gain") == expected


@pytest.mark.parametrize("blend_percent,expected", [(0, "0.00"), (37, "0.37"), (50, "0.50"), (100, "1.00")])
def test_assist_blend_is_the_percent_over_100(launcher, blend_percent, expected):
    assert flag_value(launcher.demo_command(settings_with(assist_blend=blend_percent)), "--assist-blend") == expected


@pytest.mark.parametrize("minutes", range(1, 61))
def test_steps_cover_the_chosen_minutes_at_the_policy_step(launcher, minutes):
    steps = int(flag_value(launcher.demo_command(settings_with(minutes=minutes)), "--steps"))
    assert steps * CONTRACT_POLICY_STEP_SECONDS == pytest.approx(minutes * SECONDS_PER_MINUTE)


@pytest.mark.parametrize("full_steering", CONTRACT_FULL_STEERINGS)
@pytest.mark.parametrize("driving_style", CONTRACT_DRIVING_STYLES)
def test_full_steering_and_style_are_always_passed(launcher, full_steering, driving_style):
    command = launcher.demo_command(settings_with(full_steering=full_steering, driving_style=driving_style))
    assert flag_value(command, "--full-steering") == full_steering
    assert flag_value(command, "--driving-style") == driving_style


def every_launcher_choice():
    choices = itertools.product(
        ("chase", "driver", "agent"),
        ("day", "night"),
        ("calm", "normal", "aggressive"),
        ("off", "display", "blend", "nudge", "full", "co-adapt", "dynamic"),
        (True, False),
        ("wheel", "synthetic", "remote"),
        CONTRACT_GEARBOXES,
        CONTRACT_GRIPS,
        ("fullscreen", "windowed"),
    )
    for idx, (camera, night, level, assist, ffb, input_name, gearbox, grip, window) in enumerate(choices):
        full_steering = CONTRACT_FULL_STEERINGS[idx % len(CONTRACT_FULL_STEERINGS)]
        driving_style = CONTRACT_DRIVING_STYLES[idx % len(CONTRACT_DRIVING_STYLES)]
        yield settings_with(camera=camera, night=night, level=level, assist=assist, ffb=ffb, input=input_name,
                            gearbox=gearbox, grip=grip, window=window, ffb_gain=45, assist_blend=80, minutes=7,
                            full_steering=full_steering, driving_style=driving_style)


def test_run_demo_accepts_every_command_the_launcher_can_build(launcher, demo_parser):
    checked = 0
    for settings in every_launcher_choice():
        args = demo_parser.parse_args(launcher.demo_command(settings)[2:])
        assert (args.camera, args.level, args.assist, args.gearbox, args.grip, args.input) == (
            settings["camera"], settings["level"], settings["assist"], settings["gearbox"], settings["grip"],
            settings["input"],
        )
        assert args.town == settings["town"] and args.render
        assert args.night == (settings["night"] == "night")
        assert args.fullscreen == (settings["window"] == "fullscreen")
        assert args.ffb == (settings["ffb"] and settings["input"] == "wheel")
        assert args.assist_blend == pytest.approx(0.8)
        assert (args.full_steering, args.driving_style) == (settings["full_steering"], settings["driving_style"])
        assert args.steps == 1400
        if args.ffb:
            assert args.ffb_gain == pytest.approx(0.45)
        checked += 1
    assert checked == 3 * 2 * 3 * 7 * 2 * 3 * len(CONTRACT_GEARBOXES) * len(CONTRACT_GRIPS) * 2


@pytest.mark.parametrize("window,expected", [("fullscreen", True), ("windowed", False)])
def test_fullscreen_flag_only_for_full_screen(launcher, window, expected):
    command = launcher.demo_command(settings_with(window=window))
    assert command.count("--fullscreen") == (1 if expected else 0)


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeWidget:
    created = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.children = []
        if args and isinstance(args[0], FakeWidget):
            args[0].children.append(self)
        FakeWidget.created.append(self)

    def winfo_children(self):
        return list(self.children)

    def __getitem__(self, key):
        return self.kwargs.get(key, "#000000")

    def grid(self, *args, **kwargs):
        pass

    def pack(self, *args, **kwargs):
        pass

    def bind(self, *args, **kwargs):
        pass

    def configure(self, *args, **kwargs):
        self.kwargs.update(kwargs)

    def columnconfigure(self, *args, **kwargs):
        pass


class FakeTk:
    Frame = Label = Button = Radiobutton = Scale = FakeWidget


@pytest.fixture
def built_launcher(launcher, monkeypatch):
    """Runs the launcher's layout against stand-in widgets, so no Tk window is ever created."""
    FakeWidget.created.clear()
    monkeypatch.setattr(launcher, "tk", FakeTk)
    app = launcher.Launcher.__new__(launcher.Launcher)
    app.root = FakeWidget(bg="#000000")
    app.vars = {key: FakeVar(value) for key, value in launcher.DEFAULTS.items()}
    app.town_cards = {}
    monkeypatch.setattr(app, "_on_change", lambda: None, raising=False)
    app._build()
    return app


def segment_options(key, app):
    return [(w.kwargs["value"], w.kwargs["text"]) for w in FakeWidget.created
            if "value" in w.kwargs and w.kwargs.get("variable") is app.vars[key]]


def test_launcher_has_a_window_row_offering_full_screen_and_windowed(built_launcher):
    assert "Window" in built_launcher.widgets
    labels = [w.kwargs.get("text") for w in FakeWidget.created]
    assert "Window" in labels
    assert segment_options("window", built_launcher) == [("fullscreen", "Full screen"), ("windowed", "Windowed")]


def test_every_setting_has_a_control(built_launcher, launcher):
    controlled = {key for key, var in built_launcher.vars.items()
                  if any(w.kwargs.get("variable") is var for w in FakeWidget.created)}
    assert controlled == set(launcher.DEFAULTS) - {"town"}


def make_maps_dir(root):
    maps_dir = root / "CarlaUE4" / "Content" / "Carla" / "Maps"
    maps_dir.mkdir(parents=True)
    return maps_dir


def test_installed_towns_reads_the_umap_names(launcher, tmp_path):
    maps_dir = make_maps_dir(tmp_path)
    for name in ("Town01.umap", "Town10HD.umap", "Town03_Opt.umap", "Town02.uexp", "README.txt"):
        (maps_dir / name).write_text("")
    (maps_dir / "TestMaps").mkdir()
    assert launcher.installed_towns(str(tmp_path)) == {"Town01", "Town10HD", "Town03_Opt"}


def test_installed_towns_is_empty_without_a_carla_install(launcher, tmp_path):
    assert launcher.installed_towns(str(tmp_path)) == set()
    assert launcher.installed_towns(str(tmp_path / "missing")) == set()


def test_installed_towns_is_empty_for_an_empty_maps_dir(launcher, tmp_path):
    make_maps_dir(tmp_path)
    assert launcher.installed_towns(str(tmp_path)) == set()


def test_desktop_entry_runs_the_launcher_with_this_interpreter(install_launcher, tmp_path):
    icon_path = str(tmp_path / "icon.png")
    lines = install_launcher.desktop_entry(icon_path).splitlines()
    assert lines[0] == "[Desktop Entry]"
    launcher_path = os.path.join(install_launcher.REPO_ROOT, "scripts", "wheel_demo", "launcher.py")
    assert os.path.realpath(launcher_path) == os.path.realpath(os.path.join(SCRIPTS_DIR, "launcher.py"))
    exec_lines = [line for line in lines if line.startswith("Exec=")]
    assert len(exec_lines) == 1
    exec_value = exec_lines[0][len("Exec="):]
    assert exec_value == f'"{sys.executable}" "{launcher_path}"'
    assert shlex.split(exec_value) == [sys.executable, launcher_path]
    assert f"Icon={icon_path}" in lines
    assert "Terminal=false" in lines
    assert "Type=Application" in lines


@pytest.mark.parametrize("name", ["launcher", "install_launcher"])
def test_scripts_import_without_a_display(name):
    pytest.importorskip("tkinter")
    env = {key: value for key, value in os.environ.items() if key not in ("DISPLAY", "WAYLAND_DISPLAY")}
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('probe', {os.path.join(SCRIPTS_DIR, name + '.py')!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "tkinter = sys.modules.get('tkinter')\n"
        "assert tkinter is None or tkinter._default_root is None\n"
        "assert 'pygame' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
                            timeout=SUBPROCESS_TIMEOUT_S)
    assert result.returncode == 0, result.stderr


def test_desktop_entry_quotes_paths_with_spaces(install_launcher, monkeypatch, tmp_path):
    spaced_python = "/opt/my python/bin/python3"
    spaced_repo = str(tmp_path / "my repo")
    monkeypatch.setattr(install_launcher.sys, "executable", spaced_python)
    monkeypatch.setattr(install_launcher, "REPO_ROOT", spaced_repo)
    lines = install_launcher.desktop_entry(str(tmp_path / "icon.png")).splitlines()
    exec_value = next(line for line in lines if line.startswith("Exec="))[len("Exec="):]
    launcher_path = os.path.join(spaced_repo, "scripts", "wheel_demo", "launcher.py")
    assert shlex.split(exec_value) == [spaced_python, launcher_path]


def test_launcher_offers_the_full_steering_style_and_gearbox_choices(built_launcher):
    assert segment_options("full_steering", built_launcher) == [("car", "Car steers, wheel shows"), ("wheel", "Wheel steers")]
    assert [value for value, _ in segment_options("driving_style", built_launcher)] == list(CONTRACT_DRIVING_STYLES)
    assert [value for value, _ in segment_options("gearbox", built_launcher)] == list(CONTRACT_GEARBOXES)
    assert ("none", "None (policy's model)") in segment_options("gearbox", built_launcher)
    assert [value for value, _ in segment_options("grip", built_launcher)] == list(CONTRACT_GRIPS)
    assert ("skid", "Skid (can spin)") in segment_options("grip", built_launcher)


@pytest.mark.parametrize("grip", CONTRACT_GRIPS)
def test_grip_is_always_passed(launcher, grip):
    assert flag_value(launcher.demo_command(settings_with(grip=grip)), "--grip") == grip


@pytest.mark.parametrize("assist", ["off", "display", "blend", "nudge", "full", "co-adapt", "dynamic"])
def test_full_steering_and_style_rows_are_enabled_only_under_full(built_launcher, launcher, assist):
    built_launcher.vars["assist"] = FakeVar(assist)
    launcher.Launcher._on_change(built_launcher)
    expected_state = "normal" if assist == "full" else "disabled"
    for name in FULL_ONLY_ROWS:
        buttons = built_launcher.widgets[name].winfo_children()
        assert buttons
        assert {button.kwargs["state"] for button in buttons} == {expected_state}


# ---------------------------------------------------------------------------------------------
# Calibration from the launcher: save_with_backup and the wizard, driven under a real Tk root with
# the wheel and the clock faked (no device is opened).
# ---------------------------------------------------------------------------------------------

CONTRACT_CALIBRATION_FILE = "wheel_calibration.json"
CONTRACT_BUTTON_MAP_FILE = "wheel_buttons.json"
CONTRACT_BACKUP_SUFFIX = ".previous.json"
WIZARD_MAX_PUMPS = 20_000
CALIBRATION_PHASE_COUNT = 4
DPAD_DIRECTION_COUNT = 4
REPORT_ID = 0xF8
HAT_NEUTRAL_NIBBLE = 0x0F


def test_calibration_files_are_the_ones_run_demo_reads(launcher):
    assert launcher.CALIBRATION_PATH == os.path.join(REPO_ROOT, CONTRACT_CALIBRATION_FILE)
    assert launcher.BUTTON_MAP_PATH == os.path.join(REPO_ROOT, CONTRACT_BUTTON_MAP_FILE)
    assert launcher.CALIBRATION_BACKUP_SUFFIX == CONTRACT_BACKUP_SUFFIX


def test_save_with_backup_keeps_the_replaced_file(launcher, tmp_path):
    path = tmp_path / "wheel_calibration.json"
    path.write_text("old")
    launcher.save_with_backup(lambda target: open(target, "w").write("new"), str(path))
    assert path.read_text() == "new"
    assert (tmp_path / "wheel_calibration.previous.json").read_text() == "old"


def test_save_with_backup_without_an_old_file_just_writes(launcher, tmp_path):
    path = tmp_path / "wheel_buttons.json"
    launcher.save_with_backup(lambda target: open(target, "w").write("new"), str(path))
    assert path.read_text() == "new"
    assert not (tmp_path / "wheel_buttons.previous.json").exists()


def axis_report(steering_raw=32_500, throttle_raw=255, brake_raw=255, clutch_raw=255, buttons=0, hat=HAT_NEUTRAL_NIBBLE):
    return bytes([REPORT_ID, ((buttons & 0xF) << 4) | hat, (buttons >> 4) & 0xFF, (buttons >> 12) & 0xFF,
                  steering_raw & 0xFF, (steering_raw >> 8) & 0xFF, throttle_raw, brake_raw, clutch_raw, 0x05])


class ScriptedWheel:
    """Answers each wizard prompt with the reports a driver doing that instruction would produce."""

    def __init__(self, calibrate):
        self.calibrate = calibrate
        self.prompts = []
        self.closed = False

    def reports_for(self, instruction):
        if instruction.startswith("Turn the wheel"):
            return [axis_report(steering_raw=1_000), axis_report(steering_raw=64_000), axis_report()]
        for label, pedal in (("THROTTLE", "throttle"), ("BRAKE", "brake"), ("CLUTCH", "clutch")):
            if label in instruction:
                return [axis_report(**{f"{pedal}_raw": 0}), axis_report()]
        for bit, (_, control) in enumerate(self.calibrate.BUTTON_CONTROLS):
            if instruction == f"Hold the {control}":
                return [axis_report(), axis_report(buttons=1 << bit)]
        for hat, direction in enumerate(self.calibrate.DPAD_DIRECTION_NAMES):
            if instruction == f"Hold the D-pad {direction.upper()}":
                return [axis_report(), axis_report(hat=2 * hat)]
        return [axis_report(), axis_report(steering_raw=33_000)]

    def collect(self, device, seconds):
        return self.reports_for(self.prompts[-1])

    def close(self):
        self.closed = True


@pytest.fixture
def tk_root():
    import tkinter

    try:
        root = tkinter.Tk()
    except tkinter.TclError as error:
        pytest.skip(f"no display for Tk: {error}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def wizard_world(launcher, monkeypatch, tmp_path, tk_root):
    from pufferlib.wheel_demo import calibrate

    wheel = ScriptedWheel(calibrate)
    monkeypatch.setattr(calibrate, "open_wheel", lambda: wheel)
    monkeypatch.setattr(calibrate, "collect", wheel.collect)
    monkeypatch.setattr(calibrate, "SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(launcher, "BUTTON_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(launcher, "CALIBRATION_PATH", str(tmp_path / CONTRACT_CALIBRATION_FILE))
    monkeypatch.setattr(launcher, "BUTTON_MAP_PATH", str(tmp_path / CONTRACT_BUTTON_MAP_FILE))
    closed_with = []
    wizard = launcher.CalibrationWizard(tk_root, on_close=closed_with.append)
    real_prompt = wizard._prompt

    def prompt(instruction):
        wheel.prompts.append(instruction)
        real_prompt(instruction)

    wizard._prompt = prompt
    return wizard, wheel, closed_with, tk_root, tmp_path


def pump_until(root, condition):
    for _ in range(WIZARD_MAX_PUMPS):
        root.update()
        if condition():
            return
        time.sleep(0.001)
    pytest.fail("the wizard never reached the expected state")


def run_wizard(wizard, root):
    """Press Start at every prompt until the wizard shows its result or stops; returns the Start presses."""
    wizard._start()
    presses = 0
    while True:
        pump_until(root, lambda: wizard.result is not None or wizard.step_label.cget("text") == "CALIBRATION STOPPED"
                   or (str(wizard.start_button.cget("state")) == "normal" and wizard.messages.empty()))
        if wizard.result is not None or wizard.step_label.cget("text") == "CALIBRATION STOPPED":
            return presses
        wizard._start()
        presses += 1



def test_wizard_walks_every_step_then_saves_with_backups(wizard_world, launcher):
    wizard, wheel, closed_with, root, tmp_path = wizard_world
    (tmp_path / CONTRACT_CALIBRATION_FILE).write_text("{}")
    presses = run_wizard(wizard, root)
    button_steps = 1 + len(wizard.calibrate.BUTTON_CONTROLS) + DPAD_DIRECTION_COUNT
    assert presses == CALIBRATION_PHASE_COUNT + button_steps
    assert len(wizard.phase_seconds) == presses
    calibration, problems, button_map = wizard.result
    assert calibration.steering_raw_min == 1_000 and calibration.steering_raw_max == 64_000
    assert all(value is not None for value in button_map.__dict__.values())
    wizard._save()
    assert closed_with == [[launcher.CALIBRATION_PATH, launcher.BUTTON_MAP_PATH]]
    assert (tmp_path / CONTRACT_CALIBRATION_FILE).read_text() != "{}"
    assert (tmp_path / ("wheel_calibration" + CONTRACT_BACKUP_SUFFIX)).read_text() == "{}"
    assert (tmp_path / CONTRACT_BUTTON_MAP_FILE).is_file()
    wizard.worker.join(timeout=5.0)
    assert wheel.closed


def test_wizard_without_buttons_saves_only_the_calibration(wizard_world, launcher):
    wizard, wheel, closed_with, root, tmp_path = wizard_world
    wizard.include_buttons.set(False)
    presses = run_wizard(wizard, root)
    assert presses == CALIBRATION_PHASE_COUNT == len(wizard.phase_seconds)
    assert wizard.result[2] is None
    wizard._save()
    assert closed_with == [[launcher.CALIBRATION_PATH]]
    assert not (tmp_path / CONTRACT_BUTTON_MAP_FILE).exists()


def test_cancelling_at_a_prompt_saves_nothing_and_closes_the_wheel(wizard_world):
    wizard, wheel, closed_with, root, tmp_path = wizard_world
    wizard._start()
    pump_until(root, lambda: str(wizard.start_button.cget("state")) == "normal")
    wizard._cancel()
    wizard.worker.join(timeout=5.0)
    assert not wizard.worker.is_alive()
    assert wheel.closed
    assert closed_with == [[]]
    assert list(tmp_path.iterdir()) == []


def test_a_wheel_that_will_not_open_stops_and_saves_nothing(wizard_world, monkeypatch):
    wizard, wheel, closed_with, root, tmp_path = wizard_world

    def refuse():
        raise RuntimeError("no G920 found")

    monkeypatch.setattr(wizard.calibrate, "open_wheel", refuse)
    run_wizard(wizard, root)
    assert wizard.step_label.cget("text") == "CALIBRATION STOPPED"
    assert "no G920 found" in wizard.instruction.cget("text")
    assert wizard.result is None
    assert closed_with == []
    assert list(tmp_path.iterdir()) == []


def test_done_warns_when_the_paddles_are_unmapped(wizard_world):
    wizard, wheel, closed_with, root, tmp_path = wizard_world
    real_reports_for = wheel.reports_for
    paddle_controls = {control for field, control in wizard.calibrate.BUTTON_CONTROLS if field.startswith("shift_")}

    def without_paddles(instruction):
        if any(instruction == f"Hold the {control}" for control in paddle_controls):
            return [axis_report()]
        return real_reports_for(instruction)

    wheel.reports_for = without_paddles
    run_wizard(wizard, root)
    text = wizard.instruction.cget("text")
    assert "shift_up_bit" in text and "shift_down_bit" in text
    assert "paddles are unmapped" in text


def test_launcher_reenables_and_reports_what_was_saved(built_launcher, launcher):
    built_launcher.status = FakeWidget()
    built_launcher.root.update_idletasks = lambda: None
    built_launcher.drive_button.configure(state="disabled")
    built_launcher.calibrate_button.configure(state="disabled")
    built_launcher._calibration_closed([launcher.CALIBRATION_PATH, launcher.BUTTON_MAP_PATH])
    assert built_launcher.drive_button.kwargs["state"] == "normal"
    assert built_launcher.calibrate_button.kwargs["state"] == "normal"
    assert CONTRACT_CALIBRATION_FILE in built_launcher.status.kwargs["text"]
    assert CONTRACT_BUTTON_MAP_FILE in built_launcher.status.kwargs["text"]


def test_the_default_town_is_town10hd(launcher):
    assert launcher.DEFAULTS["town"] == "Town10HD"
    assert "Town10HD" in {town for town, _ in launcher.TOWNS}


def test_the_assist_choice_names_full_self_driving(built_launcher):
    assert ("full", "Full Self Driving") in segment_options("assist", built_launcher)
    assert segment_options("assist", built_launcher) == [
        ("off", "Off"), ("display", "Display"), ("blend", "Blend"), ("nudge", "Nudge"), ("dynamic", "Dynamic blend"),
        ("full", "Full Self Driving")]
    assert "Self-driving steering" in built_launcher.widgets
    assert "Full steering" not in built_launcher.widgets
