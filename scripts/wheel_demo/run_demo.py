"""Drive one car among frozen-policy traffic, rendered by CARLA.

    scripts/wheel_demo/start_carla.sh                          # once, in another terminal
    python scripts/wheel_demo/run_demo.py --render --ffb
    python scripts/wheel_demo/run_demo.py --input synthetic --steps 100

--render needs a running CARLA server and the CARLA Python client (Python 3.10-3.12). Without it the
demo prints terminal telemetry. With --input wheel it reads the real G920 and must run in the
FOREGROUND (macOS attributes Input Monitoring to the responsible process). With --input synthetic it
drives a scripted lap and needs no hardware, which is how the loop is exercised in CI.
"""

import argparse
import math
import os
import sys
import time

import torch

# The demo imports the repo's top-level data_utils, which an editable install does not expose.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo.input.base import WheelCalibration
from pufferlib.wheel_demo.input.g920 import ButtonMap
from pufferlib.wheel_demo.input.trace import SyntheticWheelInput
from pufferlib.wheel_demo.loop import (
    GEARBOX_MODE_AUTOMATIC,
    GEARBOX_MODE_MANUAL,
    GRIP_MODE_KINEMATIC,
    GRIP_MODE_REALISTIC,
    DemoConfig,
    WheelDemoLoop,
)
from pufferlib.wheel_demo.calibrate import calibrate_interactively, validate_calibration
from pufferlib.wheel_demo.ffb import DEFAULT_FORCE_GAIN, ForceFeedbackLoop, platform_force_output
from pufferlib.wheel_demo.levels import LEVELS, LEVEL_NAMES, apply_level, level_by_name
from pufferlib.wheel_demo.maps import CARLA_TOWNS
from pufferlib.wheel_demo.carla_window import CAMERA_MODE_NAMES, CarlaDemoWindow
from pufferlib.wheel_demo.hud import MPS_TO_KMH, gear_label

# Shipped with the repo, so the demo runs with no flags. Overridable with --checkpoint.
DEFAULT_CHECKPOINT = "weights/k_scaled_0050_1000/final_model.pt"

HUD_EVERY_STEPS = 3
CAMERA_MODE_COUNT = 3
FFB_GAIN_STEP = 0.05
FFB_GAIN_MIN = 0.05
FFB_GAIN_MAX = 1.0
ASSIST_MODES_BY_NAME = {
    "off": binding.ASSIST_MODE_OFF,
    "display": binding.ASSIST_MODE_DISPLAY,
    "blend": binding.ASSIST_MODE_BLEND,
    "nudge": binding.ASSIST_MODE_NUDGE,
    "full": binding.ASSIST_MODE_FULL,
}
ASSIST_MODE_CYCLE = tuple(ASSIST_MODES_BY_NAME.values())
# Wheel button -> demo command. The keyboard keys keep working alongside these.
WHEEL_BUTTON_COMMANDS = {
    "a": "cycle_camera",
    "b": "toggle_night",
    "x": "cycle_level",
    "y": "next_agent",
    "view": "restart_engine",
    "menu": "cycle_assist",
    "dpad_up": "ffb_gain_up",
    "dpad_down": "ffb_gain_down",
}
SYNTHETIC_STEER_PERIOD_STEPS = 40.0


def build_wheel_input(kind, calibration_path, button_map_path):
    if kind == "synthetic":
        return SyntheticWheelInput(), None
    calibration = WheelCalibration.from_json(calibration_path) if calibration_path else WheelCalibration()
    button_map = ButtonMap.from_json(button_map_path) if button_map_path else ButtonMap()
    from pufferlib.wheel_demo.input.g920 import G920HidInput

    return G920HidInput(calibration, button_map).start(), button_map


def drive_synthetic(wheel, step_idx):
    """A scripted lap: accelerate, then weave, so the loop is exercised without hardware."""
    wheel.set_controls(
        steering_norm=0.6 * math.sin(2.0 * math.pi * step_idx / SYNTHETIC_STEER_PERIOD_STEPS),
        throttle_norm=1.0 if step_idx < 20 else 0.45,
        brake_norm=0.0,
    )


def print_hud(step_idx, wheel_state, human, dt):
    """The same readout as the window: speed, gear, throttle and brake."""
    if human is None:
        print(f"step {step_idx:5d}  (nobody is driving)")
        return
    flags = [name for name, key in (("COLLISION", "collision"), ("OFF ROAD", "offroad")) if human[key] > 0.0]
    if human["engine_stalled"]:
        flags.append("STALLED")
    print(
        f"t={step_idx * dt:6.1f}s  {abs(human['speed_mps']) * MPS_TO_KMH:5.1f} km/h  "
        f"gear {gear_label(human['gear'], human['gearbox_mode']):>3s}  "
        f"throttle {100 * wheel_state.throttle_norm:3.0f}%  brake {100 * wheel_state.brake_norm:3.0f}%  "
        + "  ".join(flags)
    )


def choose_town(requested):
    """All eight towns are playable; `ask` lets the driver pick at the prompt."""
    if requested != "ask":
        return requested
    print("\nChoose a town:")
    for index, town in enumerate(CARLA_TOWNS, start=1):
        flat = " (flat)" if town in ("Town01", "Town10HD") else ""
        print(f"  {index}. {town}{flat}")
    while True:
        answer = input(f"  number, or Enter for {CARLA_TOWNS[-1]}: ").strip()
        if not answer:
            return CARLA_TOWNS[-1]
        if answer.isdigit() and 1 <= int(answer) <= len(CARLA_TOWNS):
            return CARLA_TOWNS[int(answer) - 1]
        print("  not one of those.")


def cycle_level(loop, current_index):
    """Next difficulty level, applied to the live traffic. The human car is unaffected."""
    next_index = (current_index + 1) % len(LEVELS)
    level = LEVELS[next_index]
    changed = apply_level(loop.env, level)
    print(f"  level -> {level.name}: {level.description} ({changed} cars resampled)")
    return next_index


def cycle_assist(loop, blend_target):
    current = loop.env.get_assist_state()["mode"]
    next_mode = ASSIST_MODE_CYCLE[(ASSIST_MODE_CYCLE.index(current) + 1) % len(ASSIST_MODE_CYCLE)]
    loop.env.set_assist_mode(next_mode, blend_target)
    print(f"  assist -> {next(name for name, mode in ASSIST_MODES_BY_NAME.items() if mode == next_mode)}")


class WheelButtonCommands:
    """Turns wheel buttons into demo commands on the press, so holding a button does not repeat it."""

    def __init__(self, button_map):
        self.button_map = button_map
        self._held = frozenset()

    def poll(self, wheel_state):
        held = self.button_map.pressed(wheel_state.buttons, wheel_state.hat)
        pressed_now, self._held = held - self._held, held
        return {WHEEL_BUTTON_COMMANDS[name]: True for name in pressed_now if name in WHEEL_BUTTON_COMMANDS}


def adjust_ffb_gain(force_loop, step):
    if force_loop is None:
        return
    output = force_loop.output
    output.force_gain = min(FFB_GAIN_MAX, max(FFB_GAIN_MIN, round(output.force_gain + step, 2)))
    print(f"  force feedback gain -> {output.force_gain:.2f}")


def apply_window_keys(window, loop, level_index=0, wheel_commands=None, force_loop=None, assist_blend=0.5):
    """Window keys and wheel buttons drive the demo: cameras, night, level, assist, car takeover, engine restart."""
    keys = window.poll()
    keys.update(wheel_commands or {})
    if keys.get("cycle_assist"):
        cycle_assist(loop, assist_blend)
    if keys.get("cycle_camera"):
        window.set_camera_mode((window.camera_mode + 1) % CAMERA_MODE_COUNT)
    if keys.get("ffb_gain_up"):
        adjust_ffb_gain(force_loop, FFB_GAIN_STEP)
    if keys.get("ffb_gain_down"):
        adjust_ffb_gain(force_loop, -FFB_GAIN_STEP)
    if keys.get("cycle_level"):
        level_index = cycle_level(loop, level_index)
    for name, camera_mode in (("camera_chase", 0), ("camera_driver", 1), ("camera_agent", 2)):
        if keys.get(name):
            window.set_camera_mode(camera_mode)
    if keys.get("toggle_night"):
        window.toggle_night()
    if keys.get("restart_engine"):
        loop.env.restart_human_engine()
    if keys.get("next_agent"):
        total = loop.observations.shape[0]
        loop.select_human_agent((loop.human_agent_idx + 1) % total)
        window.set_human_agent(loop.human_agent_idx)
    return level_index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="policy driving the traffic; defaults to the one shipped in weights/")
    parser.add_argument("--town", default="ask", choices=("ask",) + tuple(CARLA_TOWNS))
    parser.add_argument("--agents", type=int, default=64)
    parser.add_argument("--human-agent", type=int, default=0)
    parser.add_argument("--grip", default="realistic", choices=("realistic", "kinematic"))
    parser.add_argument("--gearbox", default="automatic", choices=("automatic", "manual"))
    parser.add_argument("--input", default="wheel", choices=("wheel", "synthetic"))
    parser.add_argument("--calibration", default="wheel_calibration.json")
    parser.add_argument("--buttons", default="wheel_buttons.json")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="where the traffic policy runs; the GPU removes a once-per-step render hitch")
    parser.add_argument("--render", action="store_true", help="open the CARLA-rendered window (needs a CARLA server)")
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--camera", default="chase", choices=CAMERA_MODE_NAMES)
    parser.add_argument("--night", action="store_true")
    parser.add_argument("--ffb", action="store_true", help="force feedback (needs the wheel connected)")
    parser.add_argument("--ffb-gain", type=float, default=DEFAULT_FORCE_GAIN, help="overall force strength, 0-1")
    parser.add_argument("--calibrate", action="store_true", help="calibrate the wheel before driving")
    parser.add_argument("--level", default="normal", choices=LEVEL_NAMES, help="how the surrounding traffic behaves")
    parser.add_argument("--assist", default="off", choices=tuple(ASSIST_MODES_BY_NAME), help="steering assist mode")
    parser.add_argument("--assist-blend", type=float, default=0.5, help="blend mode: share of steering the policy gets")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_interactively(args.calibration)
    calibration_path = args.calibration if os.path.isfile(args.calibration) else None
    if calibration_path is not None:
        for problem in validate_calibration(WheelCalibration.from_json(calibration_path)):
            print(f"calibration warning: {problem}", file=sys.stderr)
    button_map_path = args.buttons if os.path.isfile(args.buttons) else None
    if args.input == "wheel" and calibration_path is None:
        print(f"warning: {args.calibration} not found, using default calibration", file=sys.stderr)

    wheel, button_map = build_wheel_input(args.input, calibration_path, button_map_path)
    wheel_buttons = WheelButtonCommands(button_map) if button_map is not None else None
    config = DemoConfig(
        checkpoint_path=os.path.expanduser(args.checkpoint),
        town=choose_town(args.town),
        num_agents=args.agents,
        human_agent_idx=args.human_agent,
        grip_mode=GRIP_MODE_REALISTIC if args.grip == "realistic" else GRIP_MODE_KINEMATIC,
        gearbox_mode=GEARBOX_MODE_AUTOMATIC if args.gearbox == "automatic" else GEARBOX_MODE_MANUAL,
        device=args.device,
        seed=args.seed,
    )

    loop = WheelDemoLoop(config, wheel)
    loop.env.set_assist_mode(ASSIST_MODES_BY_NAME[args.assist], args.assist_blend)
    window = None
    if args.render:
        window = CarlaDemoWindow(
            loop.env,
            config.town,
            loop.human_agent_idx,
            loop.substep_dt,
            camera_mode=CAMERA_MODE_NAMES.index(args.camera),
            night_mode=args.night,
            host=args.carla_host,
            port=args.carla_port,
        )
    force_loop = None
    if args.ffb:
        force_loop = ForceFeedbackLoop(
            platform_force_output(force_gain=args.ffb_gain),
            loop.car_feel,
            binding.HUMAN_STEERING_LIMIT_RAD,
        ).start()
    print(
        f"{config.town}: {loop.observations.shape[0]} cars, you are agent {loop.human_agent_idx}, "
        f"grip={args.grip}, gearbox={args.gearbox}, dt={loop.dt}s, "
        f"{loop.substeps_per_policy_step} substeps/step"
    )
    print("-" * 118)

    starting_level = level_by_name(args.level)
    state = {"level_index": LEVELS.index(starting_level)}
    changed = apply_level(loop.env, starting_level)
    print(f"level: {starting_level.name} - {starting_level.description} ({changed} cars)")

    def on_substep(substep_idx, wheel_state):
        if window is None:
            return
        window.draw(
            loop.human_state(),
            wheel_state,
            substep_idx * loop.substep_dt,
            assist=loop.env.get_assist_state(),
            level_name=LEVELS[state["level_index"]].name,
            ffb_gain=force_loop.output.force_gain if force_loop is not None else None,
        )
        wheel_commands = wheel_buttons.poll(wheel_state) if wheel_buttons is not None else None
        state["level_index"] = apply_window_keys(
            window, loop, state["level_index"], wheel_commands, force_loop, args.assist_blend
        )

    try:
        for step_idx in range(args.steps):
            if args.input == "synthetic":
                drive_synthetic(wheel, step_idx)
            started = time.perf_counter()
            wheel_state, _info = loop.step_once(on_substep=on_substep)
            if step_idx % HUD_EVERY_STEPS == 0 and window is None:
                print_hud(step_idx, wheel_state, loop.human_state(), loop.dt)
            if window is not None and window.should_close():
                break
            if args.input == "wheel" and window is None:
                remaining = loop.dt - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if force_loop is not None:
            force_loop.stop()
            if force_loop.error is not None:
                print(f"force feedback stopped: {force_loop.error}", file=sys.stderr)
        if window is not None:
            window.close()
        telemetry = loop.telemetry
        print("-" * 118)
        compute_ms = telemetry.mean_policy_ms() + telemetry.mean_env_ms()
        print(
            f"{telemetry.policy_steps} steps: policy {telemetry.mean_policy_ms():.1f} ms + "
            f"sim {telemetry.mean_env_ms():.1f} ms = {compute_ms:.1f} ms compute "
            f"(budget {loop.dt * 1000:.0f} ms)"
        )
        if telemetry.mean_substep_callback_ms() > 0.0:
            print(
                f"  render: {telemetry.mean_substep_callback_ms():.1f} ms per policy step across "
                f"{loop.substeps_per_policy_step} frames (paced to the display, not compute)"
            )
        loop.close()
        if args.input != "synthetic":
            wheel.stop()


if __name__ == "__main__":
    main()
