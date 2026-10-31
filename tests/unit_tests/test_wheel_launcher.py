"""The desktop launcher's pure parts: the run_demo command it builds, the towns it finds on disk, and
the desktop entry. Nothing here opens a window, starts CARLA or writes outside tmp_path."""

import argparse
import importlib.util
import itertools
import os
import shlex
import subprocess
import sys

import pytest

from pufferlib.wheel_demo.maps import CARLA_TOWNS


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts", "wheel_demo")
CONTRACT_DEFAULTS = {
    "town": "Town03",
    "camera": "chase",
    "night": "day",
    "level": "normal",
    "assist": "off",
    "assist_blend": 50,
    "ffb": True,
    "ffb_gain": 70,
    "gearbox": "automatic",
    "grip": "realistic",
    "input": "wheel",
    "minutes": 20,
}
CONTRACT_POLICY_STEP_SECONDS = 0.3
SECONDS_PER_MINUTE = 60
SUBPROCESS_TIMEOUT_S = 60


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
    assert flag_value(command, "--town") == "Town03"
    assert "--night" not in command
    assert flag_value(command, "--ffb-gain") == "0.70"
    assert "--ffb" in command
    assert flag_value(command, "--assist") == "off"
    assert flag_value(command, "--assist-blend") == "0.50"
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
    [(True, "wheel", True), (False, "wheel", False), (True, "synthetic", False), (False, "synthetic", False)],
)
def test_force_feedback_only_with_the_wheel_and_ffb_on(launcher, ffb, input_name, expected):
    command = launcher.demo_command(settings_with(ffb=ffb, input=input_name))
    assert ("--ffb" in command) == expected
    assert ("--ffb-gain" in command) == expected


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


def every_launcher_choice():
    choices = itertools.product(
        ("chase", "driver", "agent"),
        ("day", "night"),
        ("calm", "normal", "aggressive"),
        ("off", "display", "blend", "nudge", "full"),
        (True, False),
        ("wheel", "synthetic"),
        ("automatic", "manual"),
        ("realistic", "kinematic"),
    )
    for camera, night, level, assist, ffb, input_name, gearbox, grip in choices:
        yield settings_with(camera=camera, night=night, level=level, assist=assist, ffb=ffb, input=input_name,
                            gearbox=gearbox, grip=grip, ffb_gain=45, assist_blend=80, minutes=7)


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
        assert args.ffb == (settings["ffb"] and settings["input"] == "wheel")
        assert args.assist_blend == pytest.approx(0.8)
        assert args.steps == 1400
        if args.ffb:
            assert args.ffb_gain == pytest.approx(0.45)
        checked += 1
    assert checked == 3 * 2 * 3 * 5 * 2 * 2 * 2 * 2


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
