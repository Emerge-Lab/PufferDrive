"""Map the G920's buttons and capture its calibration, one labelled control at a time.

    python scripts/wheel_demo/probe_g920.py buttons   --out wheel_buttons.json
    python scripts/wheel_demo/probe_g920.py calibrate --out wheel_calibration.json
    python scripts/wheel_demo/probe_g920.py record    --out trace.bin --seconds 30

Nothing here guesses: a bit is only assigned to a control the operator was asked to press.
Must run in the foreground -- macOS attributes Input Monitoring to the responsible process, so a
detached run fails to open the device.
"""

import argparse
import sys
import time

from pufferlib.wheel_demo.calibrate import capture_button_map
from pufferlib.wheel_demo.input.base import WheelCalibration
from pufferlib.wheel_demo.input.g920 import (
    G920_REPORT_LENGTH,
    ButtonMap,
    find_g920_interfaces,
    hid_module,
    set_darwin_non_exclusive,
)

SETTLE_SECONDS = 1.0
CAPTURE_SECONDS = 4.0
POLL_INTERVAL_S = 0.002


def open_axis_device():
    set_darwin_non_exclusive()
    interfaces = find_g920_interfaces()
    if interfaces["axes"] is None:
        sys.exit("No G920 axis interface found. Connect the wheel; on macOS grant Input Monitoring.")
    device = hid_module().device()
    device.open_path(interfaces["axes"])
    device.set_nonblocking(True)
    return device


def collect(device, seconds):
    reports = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        report = device.read(64)
        if report and len(report) == G920_REPORT_LENGTH:
            reports.append(bytes(report))
        else:
            time.sleep(POLL_INTERVAL_S)
    return reports


def require_reports(reports):
    """The G920 only emits a report when something changes, so an empty phase means the operator
    held still or the device is not readable at all."""
    if not reports:
        sys.exit(
            "No reports received during that step. Keep the wheel moving while it records.\n"
            "If it never records anything, grant Input Monitoring to this terminal "
            "(System Settings -> Privacy & Security -> Input Monitoring) and run it in the foreground."
        )


def prompt(message):
    input(f"\n>>> {message}\n    press Enter when ready, then do it for {CAPTURE_SECONDS:.0f}s ...")
    time.sleep(SETTLE_SECONDS)


def probe_buttons(device, out_path, only=None):
    """only: field names to re-capture; the rest of an existing map at out_path is kept."""
    # a failed re-capture keeps the previous value rather than unmapping a working control
    existing = ButtonMap.from_json(out_path) if only else None
    try:
        button_map = capture_button_map(
            device, prompt, lambda reader: collect(reader, CAPTURE_SECONDS), print, only=only, existing=existing
        )
    except RuntimeError as error:
        sys.exit(str(error))
    button_map.to_json(out_path)
    print(f"\nwrote {out_path}: {button_map}")
    if button_map.shift_up_bit is None or button_map.shift_down_bit is None:
        print("WARNING: paddles unmapped -- manual gearbox mode will refuse to run.")


PEDAL_BYTE_IDX = {"throttle": 6, "brake": 7, "clutch": 8}
MIN_PEDAL_TRAVEL_COUNTS = 32


def probe_calibration(device, out_path):
    # Each pedal's rest is the max over EVERY phase, not over its own press window: during the
    # steering sweep and the other pedals' windows this pedal is released, and a press-window max
    # reads a mid-press value whenever the operator is already on the pedal.
    prompt("Turn the wheel fully LEFT, fully RIGHT, then centre it -- keep BOTH FEET OFF the pedals")
    sweep = collect(device, CAPTURE_SECONDS * 2)
    require_reports(sweep)
    steering = [r[4] | (r[5] << 8) for r in sweep]
    all_reports = list(sweep)

    pedal_floor = {}
    for pedal, label in (("throttle", "THROTTLE"), ("brake", "BRAKE (press it hard)"), ("clutch", "CLUTCH")):
        prompt(f"Press the {label} to the floor and hold, then RELEASE it")
        window = collect(device, CAPTURE_SECONDS)
        require_reports(window)
        all_reports += window
        pedal_floor[pedal] = min(r[PEDAL_BYTE_IDX[pedal]] for r in window)

    pedal_raw = {}
    for pedal, byte_idx in PEDAL_BYTE_IDX.items():
        rest = max(r[byte_idx] for r in all_reports)
        floor = pedal_floor[pedal]
        travel = rest - floor
        pedal_raw[pedal] = (rest, floor)
        print(f"    {pedal}: rest={rest} floor={floor} travel={travel}")
        if travel < MIN_PEDAL_TRAVEL_COUNTS:
            sys.exit(f"{pedal} travel is only {travel} counts -- was the pedal actually pressed?")

    calibration = WheelCalibration(
        steering_raw_min=min(steering),
        steering_raw_center=(min(steering) + max(steering)) // 2,
        steering_raw_max=max(steering),
        throttle_raw_rest=pedal_raw["throttle"][0],
        throttle_raw_floor=pedal_raw["throttle"][1],
        brake_raw_rest=pedal_raw["brake"][0],
        brake_raw_floor=pedal_raw["brake"][1],
        clutch_raw_rest=pedal_raw["clutch"][0],
        clutch_raw_floor=pedal_raw["clutch"][1],
    )
    calibration.to_json(out_path)
    print(f"\nsteering raw range {min(steering)}..{max(steering)}")
    print(f"wrote {out_path}")


def record_trace(device, out_path, seconds):
    print(f"Recording {seconds:.0f}s -- exercise every control.")
    reports = collect(device, seconds)
    with open(out_path, "wb") as handle:
        handle.write(b"".join(reports))
    print(f"wrote {len(reports)} reports to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("buttons", "calibrate", "record"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--only", help="buttons mode: comma-separated fields to re-capture, e.g. a_bit,shift_down_bit")
    args = parser.parse_args()

    device = open_axis_device()
    try:
        if args.mode == "buttons":
            probe_buttons(device, args.out, only=set(args.only.split(",")) if args.only else None)
        elif args.mode == "calibrate":
            probe_calibration(device, args.out)
        else:
            record_trace(device, args.out, args.seconds)
    finally:
        device.close()


if __name__ == "__main__":
    main()
