"""Drive one car among frozen-policy traffic. Terminal telemetry only until the renderer (M5).

    python scripts/wheel_demo/run_demo.py --checkpoint ~/Downloads/k_scaled_0050_1000/final_model.pt
    python scripts/wheel_demo/run_demo.py --checkpoint ... --input synthetic --steps 100

With --input wheel it reads the real G920 and must run in the FOREGROUND (macOS attributes Input
Monitoring to the responsible process). With --input synthetic it drives a scripted lap and needs
no hardware, which is how the loop is exercised in CI.
"""

import argparse
import math
import os
import sys
import time

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
from pufferlib.wheel_demo.ffb import ForceFeedbackLoop, G920ForceFeedback
from pufferlib.wheel_demo.levels import LEVELS, LEVEL_NAMES, apply_level, level_by_name
from pufferlib.wheel_demo.maps import CARLA_TOWNS
from pufferlib.wheel_demo.render import CAMERA_MODE_NAMES, DemoWindow

HUD_EVERY_STEPS = 3
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
    if human is None:
        print(f"step {step_idx:5d}  (nobody is driving)")
        return
    # Driver-facing sign: positive is a RIGHT turn, matching the wheel in your hands and the
    # window HUD. The sim's steering_angle is positive-left, so it is negated here too.
    steering_deg = -math.degrees(human["steering_angle_rad"])
    wheel_deg = steering_deg * binding.HUMAN_STEERING_RATIO
    print(
        f"t={step_idx * dt:6.1f}s  "
        f"wheel {wheel_state.steering_norm:+.2f} ({wheel_deg:+7.1f}deg) -> road {steering_deg:+6.2f}deg  |  "
        f"thr {wheel_state.throttle_norm:.2f} brk {wheel_state.brake_norm:.2f} clu {wheel_state.clutch_norm:.2f}  |  "
        f"{human['speed_mps'] * 3.6:5.1f} km/h  g{human['gear']:+d} {human['engine_rpm']:5.0f}rpm"
        f"{' STALL' if human['engine_stalled'] else '     '}  a_long {human['accel_long_mps2']:+5.2f}  "
        f"a_lat {human['accel_lat_mps2']:+6.2f}" + ("  [COLLISION]" if human["collision"] > 0 else "")
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


def apply_window_keys(window, loop, level_index=0):
    """Window keys drive the demo: cameras, night, level, car takeover, engine restart."""
    keys = window.poll()
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
    return level_index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--render", action="store_true", help="open a window (needs make -C pufferlib/ocean/drive/render)"
    )
    parser.add_argument("--camera", default="chase", choices=CAMERA_MODE_NAMES)
    parser.add_argument("--night", action="store_true")
    parser.add_argument("--ffb", action="store_true", help="force feedback (needs the wheel connected)")
    parser.add_argument("--ffb-gain", type=float, default=0.55, help="overall force strength, 0-1")
    parser.add_argument("--calibrate", action="store_true", help="calibrate the wheel before driving")
    parser.add_argument("--level", default="normal", choices=LEVEL_NAMES, help="how the surrounding traffic behaves")
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

    wheel, _ = build_wheel_input(args.input, calibration_path, button_map_path)
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
    window = DemoWindow(CAMERA_MODE_NAMES.index(args.camera), args.night) if args.render else None
    force_loop = None
    if args.ffb:
        force_loop = ForceFeedbackLoop(
            G920ForceFeedback(force_gain=args.ffb_gain, wheel_range_deg=900),
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

    def on_substep(substep_idx, _wheel_state):
        if window is None:
            return
        window.draw(loop.env, substep_idx * loop.substep_dt)
        state["level_index"] = apply_window_keys(window, loop, state["level_index"])

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
