"""Kinematic bicycle used to invert and replay logged poses.

Cars and trucks share one model and one command box. The rear axle has
no sideslip and yaws at v tan(steer) / wheelbase. The body center is
half a wheelbase ahead of that axle. This matches bicycle_integrate in
ocean/drive/drive.h.

The command box is the policy grid in ocean/drive/constants.h: accel
±16 m/s², steer ±0.70 rad, 257 bins with a center of 0, and a 45 m/s
speed clamp. Those bounds contain the signed 1st and 99th percentiles
of the I24 inverse for both vehicles and trucks. A box of 20 m or more
is a tractor-trailer: the tractor is that bicycle with a Cascadia
wheelbase, and the trailer heading is towed from the hitch. There is
no mass and no tire force. The logged pose is the tractor center. The
log has no separate trailer heading, so a replay starts the trailer
aligned with the tractor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from utils.common import TRUCK_MIN_LENGTH_M

SIM_DT = 0.1
NUM_ACCEL_BINS = 257
NUM_STEER_BINS = 257

# Shared policy grid. Matches ACCEL_MIN/MAX, STEER_MAX, and
# DYN_SPEED_MAX in ocean/drive/constants.h. The center bin is 0.
ACCEL_MIN = -16.0
ACCEL_MAX = 16.0
STEER_MAX = 0.70
MAX_SPEED = 45.0

# Same length cut as classify_partner. A 20 m box is a tractor-trailer.
TRAILER_MIN_LENGTH_M = 20.0

# Freightliner Cascadia sleeper, the tractor Torc runs.
# 232 in from the front axle to the tandem center.
TRACTOR_WHEELBASE_M = 232.0 * 0.0254
# Fifth wheel 10 in ahead of the tandem center.
HITCH_AHEAD_OF_REAR_AXLE_M = 10.0 * 0.0254
# 53 ft van with the tandem about 41 ft behind the kingpin.
TRAILER_WHEELBASE_M = 41.0 * 0.3048


@dataclass(frozen=True)
class Limits:
    accel_min: float
    accel_max: float
    steer_max: float
    max_speed: float

    @property
    def steer_min(self) -> float:
        return -self.steer_max


LIMITS = Limits(ACCEL_MIN, ACCEL_MAX, STEER_MAX, MAX_SPEED)
# Cars and trucks use one box. The names stay so callers can ask by class.
CAR_LIMITS = LIMITS
TRUCK_LIMITS = LIMITS


def vehicle_kind(length: float) -> str:
    if length >= TRAILER_MIN_LENGTH_M:
        return "trailer"
    if length >= TRUCK_MIN_LENGTH_M:
        return "truck"
    return "car"


def limits_for(length: float) -> Limits:
    del length
    return LIMITS


def wheelbase_for(length: float) -> float:
    """Tractor wheelbase for a combination, otherwise 0.6 of the body."""
    if vehicle_kind(length) == "trailer":
        return TRACTOR_WHEELBASE_M
    wheelbase = 0.6 * length
    if wheelbase < 1.0:
        return 1.0
    return wheelbase


def wrap_heading(heading: float) -> float:
    if heading > math.pi:
        heading -= 2.0 * math.pi
    if heading < -math.pi:
        heading += 2.0 * math.pi
    return heading


def clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def bicycle_step(
    x: float,
    y: float,
    heading: float,
    vx: float,
    vy: float,
    wheelbase: float,
    accel: float,
    steer: float,
    limits: Limits,
    trailer_heading: float | None = None,
) -> tuple[float, float, float, float, float, float | None]:
    """One 0.1 s step. Returns the trailer heading when one was passed in."""
    cosine = math.cos(heading)
    sine = math.sin(heading)
    speed = vx * cosine + vy * sine + accel * SIM_DT
    half = 0.5 * wheelbase
    rear_x = x - half * cosine
    rear_y = y - half * sine
    speed = clip(speed, -limits.max_speed, limits.max_speed)
    steer = clip(steer, limits.steer_min, limits.steer_max)
    tan_steer = math.tan(steer)
    d_heading = speed * tan_steer / wheelbase * SIM_DT
    if abs(d_heading) < 1.0e-4:
        dx_body = speed * SIM_DT
        dy_body = 0.0
    else:
        radius = speed * SIM_DT / d_heading
        dx_body = radius * math.sin(d_heading)
        dy_body = radius * (1.0 - math.cos(d_heading))
    rear_x += dx_body * cosine - dy_body * sine
    rear_y += dx_body * sine + dy_body * cosine
    heading = wrap_heading(heading + d_heading)
    cosine = math.cos(heading)
    sine = math.sin(heading)
    x = rear_x + half * cosine
    y = rear_y + half * sine
    v_lat = speed * tan_steer * 0.5
    vx = speed * cosine - v_lat * sine
    vy = speed * sine + v_lat * cosine
    if trailer_heading is None:
        return x, y, heading, vx, vy, None
    # Hitch yaw rate about the rear axle. gamma is tractor minus trailer.
    yaw_rate = speed * tan_steer / wheelbase
    gamma = wrap_heading(heading - trailer_heading)
    trailer_rate = (
        speed * math.sin(gamma)
        + HITCH_AHEAD_OF_REAR_AXLE_M * yaw_rate * math.cos(gamma)
    ) / TRAILER_WHEELBASE_M
    trailer_heading = wrap_heading(trailer_heading + trailer_rate * SIM_DT)
    return x, y, heading, vx, vy, trailer_heading


def inverse_action(
    heading0: float,
    vx0: float,
    vy0: float,
    heading1: float,
    vx1: float,
    vy1: float,
    wheelbase: float,
    limits: Limits,
) -> tuple[float, float]:
    """Accel and steer that bicycle_step applies on this logged pair.

    The command is clipped to the vehicle limits. A near-zero end speed
    forces steer to 0, matching offline_inverse.
    """
    speed0 = vx0 * math.cos(heading0) + vy0 * math.sin(heading0)
    speed1 = vx1 * math.cos(heading1) + vy1 * math.sin(heading1)
    d_heading = wrap_heading(heading1 - heading0)
    accel = (speed1 - speed0) / SIM_DT
    if abs(speed1) < 1.0e-3:
        steer = 0.0
    else:
        steer = math.atan2(d_heading * wheelbase, speed1 * SIM_DT)
    accel = clip(accel, limits.accel_min, limits.accel_max)
    steer = clip(steer, limits.steer_min, limits.steer_max)
    return accel, steer


def quantize_bin(value: float, lo: float, hi: float, count: int) -> int:
    value = clip(value, lo, hi)
    step = (hi - lo) / float(count - 1)
    bucket = int(math.floor((value - lo) / step + 0.5))
    return min(max(bucket, 0), count - 1)


def bin_center(lo: float, hi: float, count: int, bucket: int) -> float:
    if count % 2 == 1 and bucket == count // 2 and lo == -hi:
        return 0.0
    step = (hi - lo) / float(count - 1)
    return lo + bucket * step


def discretize_action(accel: float, steer: float) -> tuple[float, float]:
    """Nearest center of the shared 257-bin command grid."""
    accel_bin = quantize_bin(accel, ACCEL_MIN, ACCEL_MAX, NUM_ACCEL_BINS)
    steer_bin = quantize_bin(steer, -STEER_MAX, STEER_MAX, NUM_STEER_BINS)
    return (
        bin_center(ACCEL_MIN, ACCEL_MAX, NUM_ACCEL_BINS, accel_bin),
        bin_center(-STEER_MAX, STEER_MAX, NUM_STEER_BINS, steer_bin),
    )


def simulator_wheelbase(length: float) -> float:
    """Wheelbase used by bicycle_integrate: 0.6 of the body, at least 1 m."""
    wheelbase = 0.6 * float(length)
    if wheelbase < 1.0:
        return 1.0
    return wheelbase


def displacement_errors(
    positions: list[tuple[float, float, float, float, float]],
    length: float,
    discrete: bool,
    wheelbase: float | None = None,
) -> list[float]:
    """Open-loop position error at each step after the first.

    positions are contiguous valid (x, y, heading, vx, vy). Actions are
    the inverse of each logged pair. The rollout starts at the first
    pose and never snaps back to the log.
    """
    if len(positions) < 2:
        return []
    if wheelbase is None:
        wheelbase = wheelbase_for(length)
    limits = limits_for(length)
    articulated = vehicle_kind(length) == "trailer"
    x, y, heading, vx, vy = positions[0]
    trailer = heading if articulated else None
    errors = []
    for index, nxt in enumerate(positions[1:], start=1):
        logged = positions[index - 1]
        accel, steer = inverse_action(
            logged[2], logged[3], logged[4], nxt[2], nxt[3], nxt[4],
            wheelbase, limits,
        )
        if discrete:
            accel, steer = discretize_action(accel, steer)
        x, y, heading, vx, vy, trailer = bicycle_step(
            x, y, heading, vx, vy, wheelbase, accel, steer, limits, trailer,
        )
        errors.append(math.hypot(x - nxt[0], y - nxt[1]))
    return errors


def contiguous_poses(obj: dict) -> list[list[tuple[float, float, float, float, float]]]:
    """Valid stretches of (x, y, heading, vx, vy). A gap starts a new run."""
    runs: list[list[tuple[float, float, float, float, float]]] = []
    current: list[tuple[float, float, float, float, float]] = []
    for valid, pos, heading, vel in zip(
        obj["valid"], obj["position"], obj["heading"], obj["velocity"]
    ):
        if not valid:
            if current:
                runs.append(current)
                current = []
            continue
        current.append((
            float(pos["x"]), float(pos["y"]), float(heading),
            float(vel["x"]), float(vel["y"]),
        ))
    if current:
        runs.append(current)
    return runs


def scene_errors(objects: list[dict], discrete: bool) -> tuple[float, int, dict[str, tuple[float, int]]]:
    """Mean open-loop displacement in meters, and the same split by kind."""
    total = 0.0
    count = 0
    by_kind: dict[str, list[float]] = {"car": [], "truck": [], "trailer": []}
    for obj in objects:
        kind = vehicle_kind(float(obj["length"]))
        for run in contiguous_poses(obj):
            errors = displacement_errors(run, float(obj["length"]), discrete)
            total += sum(errors)
            count += len(errors)
            by_kind[kind].extend(errors)
    summary = {
        kind: (
            sum(values) / len(values) if values else 0.0,
            len(values),
        )
        for kind, values in by_kind.items()
    }
    mean = total / count if count else 0.0
    return mean, count, summary
