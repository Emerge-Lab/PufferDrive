"""Guided wheel calibration, plus checks for the ways it has actually gone wrong.

Every rule in `validate_calibration` exists because that exact mistake was made during development:
a pedal whose "rest" was captured mid-press, a steering sweep that never reached the stops, a brake
whose usable travel was a fifth of the others. Catching them here costs a second; discovering them
while driving costs an afternoon.
"""

import time

from pufferlib.wheel_demo.input.base import WheelCalibration
from pufferlib.wheel_demo.input.g920 import (
    BUTTON_MASK_BIT_COUNT,
    DPAD_DIRECTION_NAMES,
    G920_REPORT_LENGTH,
    HAT_MASK,
    ButtonMap,
    decode_button_mask,
    find_g920_interfaces,
    hid_module,
    set_darwin_non_exclusive,
)

PEDAL_BYTE_IDX = {"throttle": 6, "brake": 7, "clutch": 8}
STEERING_LOW_BYTE, STEERING_HIGH_BYTE = 4, 5
STEERING_FULL_SCALE = 65535

SETTLE_SECONDS = 0.6
CAPTURE_SECONDS = 4.0
POLL_INTERVAL_S = 0.002

# A sweep that does not reach the stops produces a calibration where full lock is unreachable.
MIN_STEERING_SPAN_COUNTS = int(0.90 * STEERING_FULL_SCALE)
# Off-centre by more than this and left and right lock will not match.
MAX_STEERING_CENTER_OFFSET_COUNTS = int(0.06 * STEERING_FULL_SCALE)
MIN_PEDAL_TRAVEL_COUNTS = 60
# The brake is progressive and always shorter than the others, but not by this much.
MIN_BRAKE_TRAVEL_FRACTION = 0.45

BUTTON_FACE_LABELS = (
    ("a", "A button (green)"),
    ("b", "B button (red)"),
    ("x", "X button (blue)"),
    ("y", "Y button (yellow)"),
    ("view", "VIEW button (two overlapping squares, left of centre)"),
    ("menu", "MENU button (three lines, right of centre)"),
    ("lsb", "LSB button (round, lower left)"),
    ("rsb", "RSB button (round, lower right)"),
)
BUTTON_CONTROLS = (("shift_up_bit", "RIGHT paddle (upshift)"), ("shift_down_bit", "LEFT paddle (downshift)")) + tuple(
    (f"{name}_bit", label) for name, label in BUTTON_FACE_LABELS
)
NO_BUTTON_REPORTS_MESSAGE = (
    "No reports received during that step. Keep the wheel moving while it records.\n"
    "If it never records anything, check the wheel is connected and readable (Linux: the udev rule; "
    "macOS: Input Monitoring for this program)."
)


def collect(device, seconds):
    """The G920 only reports on change, so a phase where nothing moves returns nothing."""
    reports = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        report = device.read(64)
        if report and len(report) == G920_REPORT_LENGTH:
            reports.append(bytes(report))
        else:
            time.sleep(POLL_INTERVAL_S)
    return reports


def open_wheel():
    set_darwin_non_exclusive()
    interfaces = find_g920_interfaces()
    if interfaces["axes"] is None:
        raise RuntimeError(
            "no G920 axis interface found. Connect the wheel; on macOS grant Input Monitoring to this "
            "terminal and run in the foreground."
        )
    device = hid_module().device()
    device.open_path(interfaces["axes"])
    device.set_nonblocking(True)
    return device


def terminal_prompt(instruction):
    input(f"\n>>> {instruction}\n    Press Enter to start ...")


def capture_calibration(device, prompt=terminal_prompt, log=print):
    """Walk the operator through the phases and build a WheelCalibration.

    Pedal rest comes from the maximum across EVERY phase, not from each pedal's own press window:
    during the steering sweep and the other pedals' phases this pedal is released, so rest is
    observed without ever asking anyone to hold still (which would yield no reports at all).
    """
    prompt(
        "Turn the wheel FULL LEFT, FULL RIGHT, then centre it. Keep BOTH FEET OFF the pedals. "
        f"(records for {2 * CAPTURE_SECONDS:.0f} s)"
    )
    time.sleep(SETTLE_SECONDS)
    sweep = collect(device, CAPTURE_SECONDS * 2)
    if not sweep:
        raise RuntimeError("no reports during the steering sweep: keep the wheel moving while it records")
    steering = [r[STEERING_LOW_BYTE] | (r[STEERING_HIGH_BYTE] << 8) for r in sweep]
    all_reports = list(sweep)

    pedal_floor = {}
    for pedal, label in (("throttle", "THROTTLE"), ("brake", "BRAKE, as hard as you can"), ("clutch", "CLUTCH")):
        prompt(f"Press the {label} to the floor, hold, then RELEASE. (records for {CAPTURE_SECONDS:.0f} s)")
        time.sleep(SETTLE_SECONDS)
        window = collect(device, CAPTURE_SECONDS)
        if not window:
            raise RuntimeError(f"no reports during the {pedal} phase: the pedal must move while it records")
        all_reports += window
        pedal_floor[pedal] = min(r[PEDAL_BYTE_IDX[pedal]] for r in window)

    rest = {pedal: max(r[idx] for r in all_reports) for pedal, idx in PEDAL_BYTE_IDX.items()}
    for pedal in PEDAL_BYTE_IDX:
        log(
            f"    {pedal:9s} rest={rest[pedal]:3d} floor={pedal_floor[pedal]:3d} "
            f"travel={rest[pedal] - pedal_floor[pedal]:3d}"
        )
    log(f"    steering  {min(steering)} .. {max(steering)}")

    return WheelCalibration(
        steering_raw_min=min(steering),
        steering_raw_center=(min(steering) + max(steering)) // 2,
        steering_raw_max=max(steering),
        throttle_raw_rest=rest["throttle"],
        throttle_raw_floor=pedal_floor["throttle"],
        brake_raw_rest=rest["brake"],
        brake_raw_floor=pedal_floor["brake"],
        clutch_raw_rest=rest["clutch"],
        clutch_raw_floor=pedal_floor["clutch"],
    )


def validate_calibration(calibration):
    """Problems worth telling the driver about, as a list of plain sentences. Empty means good."""
    problems = []

    span = calibration.steering_raw_max - calibration.steering_raw_min
    if span < MIN_STEERING_SPAN_COUNTS:
        problems.append(
            f"steering sweep only covered {span} of {STEERING_FULL_SCALE} counts "
            f"({100.0 * span / STEERING_FULL_SCALE:.0f}%): full lock will be unreachable. Re-run and turn "
            "the wheel all the way to both stops."
        )

    midpoint = (calibration.steering_raw_min + calibration.steering_raw_max) // 2
    offset = abs(calibration.steering_raw_center - midpoint)
    if offset > MAX_STEERING_CENTER_OFFSET_COUNTS:
        problems.append(f"steering centre is {offset} counts off the midpoint: left and right lock will not match.")

    travel = {
        pedal: getattr(calibration, f"{pedal}_raw_rest") - getattr(calibration, f"{pedal}_raw_floor")
        for pedal in PEDAL_BYTE_IDX
    }
    for pedal, counts in travel.items():
        if abs(counts) < MIN_PEDAL_TRAVEL_COUNTS:
            problems.append(
                f"{pedal} travel is only {counts} counts: it was probably not pressed to the floor, or its "
                "rest value was captured mid-press."
            )

    reference = max(abs(travel["throttle"]), abs(travel["clutch"]))
    if reference > 0 and abs(travel["brake"]) < MIN_BRAKE_TRAVEL_FRACTION * reference:
        problems.append(
            f"brake travel ({abs(travel['brake'])}) is far short of the other pedals ({reference}): the brake "
            "is progressive, so press it genuinely hard during calibration or full braking will be out of reach."
        )

    return problems


def most_common(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def capture_button_map(device, prompt, capture, log=print, only=None, existing=None):
    """Ask for each control in turn and assign the one bit (or hat value) that changed while it was held.

    `prompt(instruction)` returns when the operator is ready; `capture(device)` returns that phase's reports.
    `only` names the fields to re-capture, keeping the rest of `existing`; a control that changes no bit, or
    more than one, is left as it was rather than guessed.
    """
    prompt("Rock the wheel gently left and right, touching NO buttons or paddles")
    baseline_reports = capture(device)
    if not baseline_reports:
        raise RuntimeError(NO_BUTTON_REPORTS_MESSAGE)
    baseline = most_common(decode_button_mask(r) for r in baseline_reports)
    log(f"    baseline button mask = 0x{baseline:05x}")

    assigned = {}
    if existing is not None:
        assigned = {name: value for name, value in existing.__dict__.items() if value is not None}
    for field_name, control in BUTTON_CONTROLS:
        if only and field_name not in only:
            continue
        prompt(f"Hold the {control}")
        changed = 0
        for mask in (decode_button_mask(r) for r in capture(device)):
            changed |= mask ^ baseline
        bits = [b for b in range(BUTTON_MASK_BIT_COUNT) if changed & (1 << b)]
        log(f"    bits that changed: {bits if bits else 'NONE'}")
        if len(bits) != 1:
            log(f"    !! expected exactly one bit for {control}, got {len(bits)} -- not assigning")
            continue
        assigned[field_name] = bits[0]

    baseline_hat = most_common(r[1] & HAT_MASK for r in baseline_reports)
    for direction in DPAD_DIRECTION_NAMES:
        if only and f"dpad_{direction}_hat" not in only:
            continue
        prompt(f"Hold the D-pad {direction.upper()}")
        held = [r[1] & HAT_MASK for r in capture(device) if r[1] & HAT_MASK != baseline_hat]
        if not held:
            log(f"    !! the hat never left neutral for D-pad {direction} -- not assigning")
            continue
        assigned[f"dpad_{direction}_hat"] = most_common(held)
        log(f"    hat value = {assigned[f'dpad_{direction}_hat']}")
    return ButtonMap(**assigned)


def calibrate_interactively(output_path, prompt=terminal_prompt, log=print):
    device = open_wheel()
    try:
        calibration = capture_calibration(device, prompt=prompt, log=log)
    finally:
        device.close()
    problems = validate_calibration(calibration)
    for problem in problems:
        log(f"  WARNING: {problem}")
    calibration.to_json(output_path)
    log(f"  wrote {output_path}" + ("" if problems else "  (no problems found)"))
    return calibration, problems
