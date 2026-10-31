"""Python side of the wheel demo's renderer (M5), with no window anywhere.

CI is headless, so nothing here may reach demo_render_init: the camera and cockpit maths are
covered in C (tests/drive/test_drive_render.c, `make -C tests/drive render`). What is left for
Python is the import contract, the key handling, the sub-step callback, and the guarantee that
pufferlib.ocean.drive.binding is still free of raylib.
"""

import glob
import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

from pufferlib.wheel_demo import render as render_module

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_PATH = os.path.expanduser("~/Downloads/k_scaled_0050_1000/final_model.pt")
CARLA_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "carla")
BUILD_COMMAND = "make -C pufferlib/ocean/drive/render"

# Symbols raylib defines. Present in the demo renderer, and never in the training extension.
RAYLIB_SYMBOLS = ("InitWindow", "BeginDrawing", "DrawModelEx", "rlPushMatrix", "LoadModel")
WINDOWING_LIBRARIES = ("raylib", "OpenGL", "libGL", "Cocoa", "libX11")


def _drive_render_or_skip():
    try:
        return render_module._import_render_module()
    except ImportError:
        pytest.skip(f"drive_render is not built; run `{BUILD_COMMAND}`")


def _shared_object_or_skip(pattern, label):
    matches = sorted(glob.glob(os.path.join(REPO_ROOT, pattern)))
    if not matches:
        pytest.skip(f"{label} is not built (no match for {pattern})")
    return matches[0]


def _symbol_dump_or_skip(shared_object_path):
    tool = "nm"
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not available to inspect {shared_object_path}")
    result = subprocess.run([tool, shared_object_path], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"{tool} failed on {shared_object_path}: {result.stderr.strip()}")
    return result.stdout


def _link_dump_or_skip(shared_object_path):
    tool = "otool" if sys.platform == "darwin" else "ldd"
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not available to inspect {shared_object_path}")
    args = [tool, "-L", shared_object_path] if tool == "otool" else [tool, shared_object_path]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"{tool} failed on {shared_object_path}: {result.stderr.strip()}")
    return result.stdout


class FakeWindow:
    """Stands in for DemoWindow: same surface, no raylib."""

    def __init__(self, keys=None, camera_mode=render_module.CAMERA_MODE_CHASE, night_mode=False):
        self.keys = keys or {}
        self.camera_mode = camera_mode
        self.night_mode = night_mode
        self.poll_count = 0
        self.camera_modes_set = []
        self.night_toggles = 0
        self.human_agents_set = []

    def poll(self):
        self.poll_count += 1
        return self.keys

    def set_camera_mode(self, camera_mode):
        self.camera_modes_set.append(camera_mode)
        self.camera_mode = camera_mode

    def toggle_night(self):
        self.night_mode = not self.night_mode
        self.night_toggles += 1

    def set_human_agent(self, agent_idx):
        self.human_agents_set.append(agent_idx)


class FakeEnv:
    def __init__(self):
        self.engine_restarts = 0

    def restart_human_engine(self):
        self.engine_restarts += 1


class FakeObservations:
    def __init__(self, agent_count):
        self.shape = (agent_count, 1)


class FakeLoop:
    def __init__(self, agent_count=8, human_agent_idx=0):
        self.env = FakeEnv()
        self.observations = FakeObservations(agent_count)
        self.human_agent_idx = human_agent_idx
        self.selected = []

    def select_human_agent(self, agent_idx):
        self.selected.append(agent_idx)
        self.human_agent_idx = agent_idx


def _load_run_demo():
    path = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "run_demo.py")
    spec = importlib.util.spec_from_file_location("wheel_demo_run_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_demo():
    return _load_run_demo()


def test_import_without_the_built_extension_names_the_build_command(tmp_path, monkeypatch):
    """The failure a contributor hits first: the renderer was never built."""
    monkeypatch.setattr(render_module, "RENDER_MODULE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if "drive/render" not in entry])
    monkeypatch.delitem(sys.modules, "drive_render", raising=False)

    with pytest.raises(ImportError) as error:
        render_module._import_render_module()

    message = str(error.value)
    assert BUILD_COMMAND in message, message
    assert str(tmp_path) in message, message
    assert isinstance(error.value.__cause__, ImportError)


def test_import_failure_does_not_silently_fall_back(tmp_path, monkeypatch):
    """DemoWindow must raise, not construct a window-less object that fails later."""
    monkeypatch.setattr(render_module, "RENDER_MODULE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if "drive/render" not in entry])
    monkeypatch.delitem(sys.modules, "drive_render", raising=False)

    with pytest.raises(ImportError) as error:
        render_module.DemoWindow()
    assert BUILD_COMMAND in str(error.value)


def test_camera_mode_names_cover_every_mode():
    names = render_module.CAMERA_MODE_NAMES
    assert len(names) == len(set(names))
    assert names.index("chase") == render_module.CAMERA_MODE_CHASE
    assert names.index("driver") == render_module.CAMERA_MODE_DRIVER
    assert names.index("agent") == render_module.CAMERA_MODE_AGENT
    assert sorted(
        (render_module.CAMERA_MODE_CHASE, render_module.CAMERA_MODE_DRIVER, render_module.CAMERA_MODE_AGENT)
    ) == list(range(len(names)))


def test_camera_mode_constants_match_the_extension():
    drive_render = _drive_render_or_skip()
    assert drive_render.CAMERA_MODE_CHASE == render_module.CAMERA_MODE_CHASE
    assert drive_render.CAMERA_MODE_DRIVER == render_module.CAMERA_MODE_DRIVER
    assert drive_render.CAMERA_MODE_AGENT == render_module.CAMERA_MODE_AGENT
    assert drive_render.CAMERA_MODE_COUNT == len(render_module.CAMERA_MODE_NAMES)


def test_set_camera_mode_rejects_out_of_range():
    drive_render = _drive_render_or_skip()
    with pytest.raises(ValueError):
        drive_render.set_camera_mode(drive_render.CAMERA_MODE_COUNT)
    with pytest.raises(ValueError):
        drive_render.set_camera_mode(-1)


def test_frame_rejects_a_handle_that_is_not_an_env():
    drive_render = _drive_render_or_skip()
    with pytest.raises(TypeError):
        drive_render.frame("not a handle", 0.0)
    with pytest.raises(ValueError):
        drive_render.frame(0, 0.0)


def test_poll_keys_is_empty_without_a_window():
    """No window open means no key events, rather than a crash into raylib's input state."""
    drive_render = _drive_render_or_skip()
    assert drive_render.poll_keys() == {}
    assert drive_render.should_close() is True


@pytest.mark.parametrize(
    "key_name,expected_mode",
    [("camera_chase", 0), ("camera_driver", 1), ("camera_agent", 2)],
)
def test_camera_keys_select_their_mode(run_demo, key_name, expected_mode):
    window = FakeWindow(keys={key_name: True}, camera_mode=99)
    loop = FakeLoop()
    run_demo.apply_window_keys(window, loop)
    assert window.camera_modes_set == [expected_mode]
    assert window.camera_mode == expected_mode
    assert render_module.CAMERA_MODE_NAMES[expected_mode] == key_name.split("_", 1)[1]
    assert loop.selected == []
    assert loop.env.engine_restarts == 0
    assert window.night_toggles == 0


def test_no_keys_changes_nothing(run_demo):
    window = FakeWindow(keys={})
    loop = FakeLoop()
    run_demo.apply_window_keys(window, loop)
    assert window.poll_count == 1
    assert window.camera_modes_set == []
    assert window.night_toggles == 0
    assert loop.selected == []
    assert loop.env.engine_restarts == 0
    assert window.human_agents_set == []


def test_false_valued_keys_are_ignored(run_demo):
    window = FakeWindow(
        keys={
            name: False
            for name in (
                "camera_chase",
                "camera_driver",
                "camera_agent",
                "toggle_night",
                "next_agent",
                "restart_engine",
            )
        }
    )
    loop = FakeLoop()
    run_demo.apply_window_keys(window, loop)
    assert window.camera_modes_set == []
    assert window.night_toggles == 0
    assert loop.selected == []
    assert loop.env.engine_restarts == 0


def test_night_key_flips_each_press(run_demo):
    window = FakeWindow(keys={"toggle_night": True}, night_mode=False)
    loop = FakeLoop()
    run_demo.apply_window_keys(window, loop)
    assert window.night_mode is True
    run_demo.apply_window_keys(window, loop)
    assert window.night_mode is False
    assert window.night_toggles == 2


def test_restart_engine_key_calls_through(run_demo):
    window = FakeWindow(keys={"restart_engine": True})
    loop = FakeLoop()
    run_demo.apply_window_keys(window, loop)
    run_demo.apply_window_keys(window, loop)
    assert loop.env.engine_restarts == 2
    assert loop.selected == []


def test_next_agent_wraps_around_the_agent_count(run_demo):
    agent_count = 5
    window = FakeWindow(keys={"next_agent": True})
    loop = FakeLoop(agent_count=agent_count, human_agent_idx=0)
    for _ in range(agent_count + 1):
        run_demo.apply_window_keys(window, loop)
    assert loop.selected == [1, 2, 3, 4, 0, 1]
    assert loop.human_agent_idx == 1
    assert window.human_agents_set == loop.selected


def test_next_agent_ring_is_sized_by_the_observation_rows(run_demo):
    """Pins today's behaviour so the known drift stays visible: apply_window_keys sizes the ring
    from observations.shape[0], while c_select_human_agent validates against num_total_agents."""
    window = FakeWindow(keys={"next_agent": True})
    loop = FakeLoop(agent_count=3, human_agent_idx=2)
    loop.total_agents = 10  # what the C side would accept, were they ever to differ
    run_demo.apply_window_keys(window, loop)
    assert loop.selected == [0]


def test_next_agent_from_the_last_agent_returns_to_zero(run_demo):
    agent_count = 3
    window = FakeWindow(keys={"next_agent": True})
    loop = FakeLoop(agent_count=agent_count, human_agent_idx=agent_count - 1)
    run_demo.apply_window_keys(window, loop)
    assert loop.selected == [0]
    assert loop.human_agent_idx == 0


def test_one_press_can_do_several_things(run_demo):
    window = FakeWindow(
        keys={"camera_agent": True, "toggle_night": True, "next_agent": True, "restart_engine": True}, night_mode=False
    )
    loop = FakeLoop(agent_count=4, human_agent_idx=3)
    run_demo.apply_window_keys(window, loop)
    assert window.camera_modes_set == [2]
    assert window.night_mode is True
    assert loop.env.engine_restarts == 1
    assert loop.selected == [0]
    assert window.human_agents_set == [0]


def test_next_agent_tells_the_window_the_index_the_loop_settled_on(run_demo):
    window = FakeWindow(keys={"next_agent": True})
    loop = FakeLoop(agent_count=4, human_agent_idx=1)
    run_demo.apply_window_keys(window, loop)
    assert window.human_agents_set == [loop.human_agent_idx] == [2]


def test_restart_and_camera_keys_do_not_retarget_the_window(run_demo):
    window = FakeWindow(keys={"restart_engine": True, "camera_driver": True, "toggle_night": True})
    run_demo.apply_window_keys(window, FakeLoop())
    assert window.human_agents_set == []


def test_run_demo_camera_names_select_the_carla_scene_modes(run_demo):
    from pufferlib.wheel_demo import carla_scene

    names = run_demo.CAMERA_MODE_NAMES
    assert names.index("chase") == carla_scene.CAMERA_MODE_CHASE
    assert names.index("driver") == carla_scene.CAMERA_MODE_DRIVER
    assert names.index("agent") == carla_scene.CAMERA_MODE_AGENT


def test_run_demo_imports_without_carla_or_pygame():
    """The CARLA client and pygame load only when a window is opened (beyond what the env already pulls in)."""
    path = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "run_demo.py")
    probe = (
        "import importlib.util, sys\n"
        "import pufferlib.ocean.drive.drive\n"
        "baseline = {name for name in ('carla', 'pygame') if name in sys.modules}\n"
        f"spec = importlib.util.spec_from_file_location('probe_run_demo', {path!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print(sorted(name for name in ('carla', 'pygame') if name in sys.modules and name not in baseline))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


# ---- the terminal HUD ----

TERMINAL_HUD_DT_S = 0.1


class FakeWheelState:
    def __init__(self, throttle_norm=0.0, brake_norm=0.0, steering_norm=0.37, clutch_norm=0.0):
        self.throttle_norm = throttle_norm
        self.brake_norm = brake_norm
        self.steering_norm = steering_norm
        self.clutch_norm = clutch_norm


def terminal_human(**overrides):
    human = {
        "speed_mps": 10.0,
        "gear": 3,
        "gearbox_mode": 1,
        "collision": 0.0,
        "offroad": 0.0,
        "engine_stalled": 0,
        "engine_rpm": 4321.0,
        "steering_angle_rad": 0.2,
        "accel_long_mps2": 1.5,
        "accel_lat_mps2": -2.5,
    }
    human.update(overrides)
    return human


def print_hud_line(run_demo, capsys, human, wheel_state):
    run_demo.print_hud(7, wheel_state, human, TERMINAL_HUD_DT_S)
    return capsys.readouterr().out


def test_terminal_hud_shows_speed_gear_and_pedals(run_demo, capsys):
    line = print_hud_line(run_demo, capsys, terminal_human(), FakeWheelState(throttle_norm=0.42, brake_norm=0.07))
    assert "36.0 km/h" in line
    assert "D3" in line
    assert "throttle  42%" in line
    assert "brake   7%" in line


def test_terminal_hud_shows_reverse_speed_as_positive(run_demo, capsys):
    line = print_hud_line(run_demo, capsys, terminal_human(speed_mps=-2.5, gear=-1), FakeWheelState())
    assert "9.0 km/h" in line
    assert "-9.0" not in line
    assert " R " in line


def test_terminal_hud_has_no_rpm_and_no_angles(run_demo, capsys):
    line = print_hud_line(run_demo, capsys, terminal_human(), FakeWheelState(throttle_norm=0.5))
    for banned in ("rpm", "4321", "deg", "a_lat", "a_long", "wheel"):
        assert banned not in line.lower(), line


def test_terminal_hud_lists_every_raised_flag(run_demo, capsys):
    line = print_hud_line(
        run_demo, capsys, terminal_human(collision=1.0, offroad=1.0, engine_stalled=1), FakeWheelState()
    )
    assert "COLLISION" in line and "OFF ROAD" in line and "STALLED" in line
    quiet = print_hud_line(run_demo, capsys, terminal_human(), FakeWheelState())
    assert "COLLISION" not in quiet and "OFF ROAD" not in quiet and "STALLED" not in quiet


def test_terminal_hud_uses_the_manual_gear_label(run_demo, capsys):
    line = print_hud_line(run_demo, capsys, terminal_human(gear=2, gearbox_mode=0), FakeWheelState())
    assert "gear   2" in line
    assert "D2" not in line


def test_terminal_hud_reports_nobody_driving(run_demo, capsys):
    assert "nobody is driving" in print_hud_line(run_demo, capsys, None, FakeWheelState())


# ---- the sub-step callback, which is where the renderer draws ----


def _demo_loop_or_skip(substeps_per_policy_step):
    if not os.path.isfile(CHECKPOINT_PATH):
        pytest.skip(f"checkpoint not present: {CHECKPOINT_PATH}")
    from pufferlib.wheel_demo.input.trace import SyntheticWheelInput
    from pufferlib.wheel_demo.loop import DemoConfig, WheelDemoLoop

    config = DemoConfig(
        checkpoint_path=CHECKPOINT_PATH,
        town="Town01",
        map_dir=CARLA_MAP_DIR,
        num_agents=8,
        substeps_per_policy_step=substeps_per_policy_step,
        seed=0,
    )
    return WheelDemoLoop(config, SyntheticWheelInput())


@pytest.fixture(scope="module")
def demo_loop():
    loop = _demo_loop_or_skip(4)
    yield loop
    loop.close()


def test_on_substep_fires_once_per_substep_in_order(demo_loop):
    seen = []
    demo_loop.step_once(on_substep=lambda substep_idx, wheel_state: seen.append(substep_idx))
    assert seen == list(range(demo_loop.substeps_per_policy_step))

    seen.clear()
    demo_loop.step_once(on_substep=lambda substep_idx, wheel_state: seen.append(substep_idx))
    assert seen == list(range(demo_loop.substeps_per_policy_step))


def test_on_substep_receives_the_wheel_state(demo_loop):
    demo_loop.wheel.set_controls(steering_norm=0.25, throttle_norm=0.5)
    states = []
    demo_loop.step_once(on_substep=lambda substep_idx, wheel_state: states.append(wheel_state))
    assert len(states) == demo_loop.substeps_per_policy_step
    assert all(state.steering_norm == pytest.approx(0.25) for state in states)
    assert all(state.throttle_norm == pytest.approx(0.5) for state in states)


def test_step_once_without_a_callback_still_steps(demo_loop):
    before = demo_loop.telemetry.policy_steps
    wheel_state, info = demo_loop.step_once()
    assert demo_loop.telemetry.policy_steps == before + 1
    assert wheel_state is not None


def test_render_poses_sized_for_every_agent(demo_loop):
    agent_count = demo_loop.observations.shape[0]
    x, y, heading = demo_loop.render_poses(0)
    assert x.shape[0] >= agent_count
    assert y.shape[0] >= agent_count and heading.shape[0] >= agent_count

    # Same buffers reused: the renderer draws at 60 Hz and must not allocate per frame.
    again = demo_loop.render_poses(1)
    assert all(new is old for new, old in zip(again, (x, y, heading)))

    import numpy as np

    assert np.isfinite(x[:agent_count]).all()
    assert np.isfinite(y[:agent_count]).all()
    assert np.isfinite(heading[:agent_count]).all()
    assert (np.abs(heading[:agent_count]) <= np.pi + 1e-5).all()


def test_demo_traffic_is_car_sized_without_eval_mode(demo_loop):
    """The renderer scales each .glb by the agent's sim_length/width/height, so the training size
    profile (0.8 m to 7 m boxes) would draw stretched cars. VEHICLE_SIZE_PROFILE_CAR gives the
    demo car-sized traffic, and must do so without eval_mode, which the demo turns off."""
    from pufferlib.wheel_demo import policy_host

    vehicle_size_profile_car = 1
    assert policy_host.DEMO_ENV_OVERRIDES["vehicle_size_profile"] == vehicle_size_profile_car
    assert policy_host.DEMO_ENV_OVERRIDES["eval_mode"] is False
    assert demo_loop.env.vehicle_size_profile == vehicle_size_profile_car
    assert not demo_loop.env.eval_mode


def test_telemetry_separates_callback_time_from_sim_time():
    """Rendering is wall-clock paced, so a slow callback must not be charged to the sim."""
    import time

    loop = _demo_loop_or_skip(3)
    try:
        sleep_seconds = 0.02
        substeps = loop.substeps_per_policy_step
        slept_seconds = substeps * sleep_seconds

        loop.step_once()
        telemetry = loop.telemetry
        assert telemetry.substep_callback_seconds == 0.0
        assert telemetry.mean_substep_callback_ms() == 0.0
        env_seconds_without_callback = telemetry.env_seconds
        assert env_seconds_without_callback < slept_seconds

        def slow_callback(substep_idx, wheel_state):
            time.sleep(sleep_seconds)

        loop.step_once(on_substep=slow_callback)
        assert telemetry.policy_steps == 2

        # The sleep is charged to the callback, in full.
        assert telemetry.substep_callback_seconds >= slept_seconds
        assert telemetry.mean_substep_callback_ms() == pytest.approx(1000.0 * telemetry.substep_callback_seconds / 2.0)

        # ... and not to the sim, which did the same work it did on the first step.
        env_seconds_of_slow_step = telemetry.env_seconds - env_seconds_without_callback
        assert env_seconds_of_slow_step >= 0.0
        assert env_seconds_of_slow_step < slept_seconds
        assert telemetry.mean_env_ms() == pytest.approx(1000.0 * telemetry.env_seconds / 2.0)
        assert telemetry.mean_policy_ms() > 0.0
    finally:
        loop.close()


# ---- the isolation guarantee: the training extension must never link raylib (#593) ----


def test_training_binding_has_no_raylib_symbols():
    binding_so = _shared_object_or_skip("pufferlib/ocean/drive/binding*.so", "drive binding")
    symbols = _symbol_dump_or_skip(binding_so)
    found = [symbol for symbol in RAYLIB_SYMBOLS if symbol in symbols]
    assert not found, f"{binding_so} references raylib symbols {found}; #593 removed raylib from the training build"
    assert "rlgl" not in symbols.lower()


def test_the_raylib_symbol_probe_actually_finds_raylib():
    """Positive control: a typo in RAYLIB_SYMBOLS would make the test above vacuous."""
    render_so = _shared_object_or_skip("pufferlib/ocean/drive/render/drive_render*.so", "drive_render")
    symbols = _symbol_dump_or_skip(render_so)
    missing = [symbol for symbol in RAYLIB_SYMBOLS if symbol not in symbols]
    assert not missing, f"{render_so} is missing {missing}; the probe cannot detect raylib"


def test_training_binding_links_no_windowing_library():
    binding_so = _shared_object_or_skip("pufferlib/ocean/drive/binding*.so", "drive binding")
    links = _link_dump_or_skip(binding_so)
    found = [name for name in WINDOWING_LIBRARIES if name in links]
    assert not found, f"{binding_so} links {found}"


def test_importing_the_training_binding_does_not_import_the_renderer():
    probe = (
        "import sys\n"
        "import pufferlib.ocean.drive.binding\n"
        "from pufferlib.ocean.drive.drive import Drive\n"
        "print('drive_render' in sys.modules)\n"
        "print(any('drive/render' in entry for entry in sys.path))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[-2:] == ["False", "False"], result.stdout


def test_setup_py_does_not_build_the_renderer():
    """The renderer has its own Makefile precisely so setup.py never pulls raylib in."""
    with open(os.path.join(REPO_ROOT, "setup.py")) as handle:
        setup_source = handle.read()
    assert "raylib" not in setup_source.lower()
    assert "drive_render" not in setup_source
    assert "drive/render" not in setup_source
