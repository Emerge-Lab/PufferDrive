"""Kinematic target -> (steer, throttle, brake) tracking controller.

The policy's jerk action is integrated by the shadow PufferDrive env into a
target (speed, yaw) one policy-dt ahead; this controller makes CARLA's physics
chase that target. CARLA (not the shadow env) moves the ego, so all leaderboard
collision/infraction machinery sees a normally-driven vehicle. The simulator
side wraps the returned values in carla.VehicleControl; this module never
imports carla so it also runs in the PufferDrive venv behind lb1/policy_server.py.

Longitudinal: PI on speed error -> throttle/brake.
Lateral: desired yaw rate over the policy horizon -> inverse kinematic bicycle
-> front wheel angle, normalized by the vehicle's max steer angle.

Tracking error is accumulated per route (see stats()) — it is the number that
disambiguates "bad policy" from "bad controller" when scores are low.
"""

import math
from typing import NamedTuple

import numpy as np

from pufferlib.ocean.cosim.carla_bridge import wrap_deg_180

WHEEL_ROW_COLUMNS = 4  # max_steer_deg, position x/y/z (cm, CARLA's physics control units)
WHEEL_POSITION_CM_PER_M = 100.0
DEFAULT_WHEELBASE_M = 2.85  # Lincoln MKZ-ish fallbacks for implausible physics controls
DEFAULT_MAX_STEER_DEG = 70.0
WHEELBASE_RANGE_M = (1.5, 5.0)
MAX_STEER_RANGE_DEG = (10.0, 90.0)


def wheel_rows(vehicle):
    """[[max_steer_deg, x_cm, y_cm, z_cm], ...] of a live CARLA vehicle's wheels (read-only physics control)."""
    return [[w.max_steer_angle, w.position.x, w.position.y, w.position.z] for w in vehicle.get_physics_control().wheels]


def vehicle_geometry_from_wheels(rows):
    """(wheelbase_m, max_steer_rad) from wheel_rows; defaults for anything implausible."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, WHEEL_ROW_COLUMNS)
    if len(rows) == 0:
        return DEFAULT_WHEELBASE_M, math.radians(DEFAULT_MAX_STEER_DEG)
    max_steer_deg = float(rows[:, 0].max())
    steering, fixed = rows[rows[:, 0] > 0.0], rows[rows[:, 0] <= 0.0]
    wheelbase = DEFAULT_WHEELBASE_M
    if len(steering) and len(fixed):
        axle_offset_cm = steering[:, 1:].mean(axis=0) - fixed[:, 1:].mean(axis=0)
        wheelbase = float(np.linalg.norm(axle_offset_cm)) / WHEEL_POSITION_CM_PER_M
    if not (WHEELBASE_RANGE_M[0] <= wheelbase <= WHEELBASE_RANGE_M[1]):
        wheelbase = DEFAULT_WHEELBASE_M
    if not (MAX_STEER_RANGE_DEG[0] <= max_steer_deg <= MAX_STEER_RANGE_DEG[1]):
        max_steer_deg = DEFAULT_MAX_STEER_DEG
    return wheelbase, math.radians(max_steer_deg)


def read_vehicle_geometry(vehicle):
    """(wheelbase_m, max_steer_rad) of a live CARLA vehicle; defaults on any surprise."""
    try:
        return vehicle_geometry_from_wheels(wheel_rows(vehicle))
    except Exception:
        return DEFAULT_WHEELBASE_M, math.radians(DEFAULT_MAX_STEER_DEG)


class TrackingController:
    def __init__(
        self,
        wheelbase_m=DEFAULT_WHEELBASE_M,
        max_steer_rad=math.radians(DEFAULT_MAX_STEER_DEG),
        horizon_s=0.1,
        kp_speed=0.7,
        ki_speed=0.15,
        kp_brake=0.5,
        max_throttle=0.75,
        stop_speed=0.1,
    ):
        # stop_speed must sit BELOW the first integrated target speed of a
        # jerk-dynamics launch from rest (~j*dt^2, ~0.36 m/s at dt=0.3),
        # otherwise the brake-hold swallows the ramp-up: ego never moves, the
        # shadow env re-syncs to zero speed, and the target never grows.
        self.wheelbase = wheelbase_m
        self.max_steer = max_steer_rad
        self.horizon = horizon_s  # time in which the target yaw should be reached (policy dt)
        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kp_brake = kp_brake
        self.max_throttle = max_throttle
        self.stop_speed = stop_speed
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._speed_errors = []
        self._yaw_errors = []

    def step(self, current_speed, current_yaw_deg, target_speed, target_yaw_deg, tick_dt):
        """One control tick -> (steer, throttle, brake). current_* from CARLA ground truth, target_* from
        the shadow env's integrated policy action (held between policy steps)."""
        # --- longitudinal ---
        err = target_speed - current_speed
        self._speed_errors.append(err)
        throttle, brake = 0.0, 0.0
        if target_speed < self.stop_speed and current_speed < 2.0 * self.stop_speed:
            # Policy wants a stop and we are (almost) there: hold the brake so
            # the car doesn't creep into intersections/red lights.
            brake = 1.0
            self._integral = 0.0
        elif err >= 0.0:
            self._integral = float(np.clip(self._integral + err * tick_dt, 0.0, 2.0))
            throttle = float(np.clip(self.kp_speed * err + self.ki_speed * self._integral, 0.0, self.max_throttle))
        else:
            self._integral = 0.0
            brake = float(np.clip(-self.kp_brake * err, 0.0, 1.0))

        # --- lateral ---
        yaw_err_rad = math.radians(wrap_deg_180(target_yaw_deg - current_yaw_deg))
        self._yaw_errors.append(yaw_err_rad)
        desired_yaw_rate = yaw_err_rad / self.horizon
        # Kinematic bicycle: yaw_rate = v / L * tan(steer_angle)
        steer_angle = math.atan2(self.wheelbase * desired_yaw_rate, max(current_speed, 1.0))
        steer = float(np.clip(steer_angle / self.max_steer, -1.0, 1.0))

        return steer, throttle, brake

    def stats(self):
        """Per-route tracking fidelity: mean/max absolute speed error (m/s) and
        yaw error (deg) between the policy's target and what CARLA achieved."""
        if not self._speed_errors:
            return {}
        se = np.abs(np.array(self._speed_errors))
        ye = np.degrees(np.abs(np.array(self._yaw_errors)))
        return {
            "speed_err_mean": float(se.mean()),
            "speed_err_max": float(se.max()),
            "yaw_err_mean": float(ye.mean()),
            "yaw_err_max": float(ye.max()),
            "ticks": len(se),
        }


YAW_TRACKING_MIN_SPEED_MPS = 2.0
REVERSE_GEAR = -1  # what CARLA reports as the gear while VehicleControl.reverse is set


class GearGains(NamedTuple):
    speeds_mps: tuple  # span where the automatic gearbox uses the gear; the values below are at these speeds
    throttle_holding_speed: tuple
    accel_per_throttle_mps2: tuple


class PedalGains(NamedTuple):
    gears: tuple  # GearGains of forward gear 1, 2, ...
    reverse: GearGains
    gear_shift_seconds: float  # the automatic gearbox gives no drive for this long while it shifts
    max_engine_braking_mps2: float  # the brake starts beyond this even in a gear that brakes harder: it acts at once
    brake_per_mps2: float
    curvature_per_steer_at_rest: float
    curvature_per_steer_per_mps: float
    rear_axle_to_center_m: float  # the car turns about its rear axle, so its centre travels at an angle to its heading


# measured open loop on a straight road with each gear held (pedal steps from rest and from speed, steer steps):
# CARLA 0.9.15 and 0.9.10.1. First gear and reverse hold at rest the throttle at which the standing car starts to move.
PEDAL_GAINS_BY_BLUEPRINT = {
    "vehicle.lincoln.mkz_2020": PedalGains(
        gears=(
            GearGains((0.0, 2.6, 5.3), (0.30, 0.35, 0.47), (4.4, 8.0, 10.4)),
            GearGains((5.7, 10.2), (0.40, 0.45), (8.4, 13.2)),
            GearGains((10.0, 16.8), (0.37, 0.47), (7.8, 9.7)),
            GearGains((12.7, 22.1), (0.38, 0.47), (5.8, 8.0)),
            GearGains((18.3, 26.5), (0.38, 0.48), (4.3, 5.3)),
            GearGains((24.5, 28.0), (0.44, 0.52), (3.2, 3.6)),
        ),
        reverse=GearGains((0.0, 2.0), (0.21, 0.21), (4.8, 4.8)),
        gear_shift_seconds=0.1,
        max_engine_braking_mps2=2.0,
        brake_per_mps2=0.22,
        curvature_per_steer_at_rest=0.376,
        curvature_per_steer_per_mps=-0.0023,
        rear_axle_to_center_m=1.25,
    ),
    "vehicle.lincoln.mkz2017": PedalGains(
        gears=(
            GearGains((0.0, 2.0, 4.6), (0.27, 0.30, 0.46), (4.7, 5.6, 6.9)),
            GearGains((1.3, 8.4), (0.18, 0.49), (4.1, 5.9)),
            GearGains((2.6, 12.3), (0.21, 0.49), (3.0, 4.3)),
            GearGains((10.2, 15.3), (0.38, 0.49), (2.8, 3.2)),
            GearGains((13.9, 28.0), (0.44, 0.71), (2.7, 2.7)),
        ),
        reverse=GearGains((0.0, 2.0), (0.20, 0.20), (4.4, 4.4)),
        gear_shift_seconds=0.5,
        max_engine_braking_mps2=1.0,
        brake_per_mps2=0.16,
        curvature_per_steer_at_rest=0.387,
        curvature_per_steer_per_mps=-0.0031,
        rear_axle_to_center_m=1.3,
    ),
}


def pedal_gains_for(blueprint):
    if blueprint not in PEDAL_GAINS_BY_BLUEPRINT:
        raise ValueError(
            f"no measured pedal gains for ego blueprint {blueprint!r}; known: {sorted(PEDAL_GAINS_BY_BLUEPRINT)}"
        )
    return PEDAL_GAINS_BY_BLUEPRINT[blueprint]


def pedals(gear_gains, max_engine_braking_mps2, brake_per_mps2, shifting, target_accel_mps2, speed_mps):
    """(throttle, brake) for a target acceleration in one gear: one straight line each. Acceleration and speed are
    along the direction the gear drives in."""
    accel_per_throttle = float(np.interp(speed_mps, gear_gains.speeds_mps, gear_gains.accel_per_throttle_mps2))
    throttle_holding_speed = float(np.interp(speed_mps, gear_gains.speeds_mps, gear_gains.throttle_holding_speed))
    # a released throttle brakes by the holding throttle's worth; while the gearbox shifts the engine is disconnected
    engine_braking_mps2 = 0.0 if shifting else min(throttle_holding_speed * accel_per_throttle, max_engine_braking_mps2)
    throttle = float(np.clip(throttle_holding_speed + target_accel_mps2 / accel_per_throttle, 0.0, 1.0))
    brake = float(np.clip(brake_per_mps2 * (-target_accel_mps2 - engine_braking_mps2), 0.0, 1.0))
    return throttle, brake


class PedalMap:
    """Jerk-integrator targets -> (steer, throttle, brake, reverse) through the gains of the gear the automatic gearbox
    is in, or of reverse; the policy is the only feedback loop."""

    def __init__(self, gains):
        self.gains = gains
        self.reset()

    def reset(self):
        self._engaged_gear = 1
        self._command = None
        self._accel_rows = []
        self._yaw_rate_rows = []

    def observe(self, speed_mps, yaw_rate_radps, tick_dt):
        """CARLA's response to the previous tick's command, kept for stats()."""
        if self._command is None:
            return
        target_accel_mps2, target_curvature_per_m, commanded_speed_mps, brake, gear = self._command
        self._accel_rows.append((target_accel_mps2, (speed_mps - commanded_speed_mps) / tick_dt, brake, gear))
        if speed_mps > YAW_TRACKING_MIN_SPEED_MPS:
            self._yaw_rate_rows.append((target_curvature_per_m * speed_mps, yaw_rate_radps))

    def step(self, target_accel_mps2, target_curvature_per_m, speed_mps, gear, hold_standstill, reverse):
        """One tick -> (steer, throttle, brake, reverse) of the carla.VehicleControl. Speed and acceleration are along
        the car's forward axis, curvature and yaw rate positive to the right, like CARLA's steer. gear is the one CARLA
        reports for the ego: 1.. forward, 0 while the gearbox shifts, REVERSE_GEAR in reverse."""
        if gear > 0:
            self._engaged_gear = gear
        forward_speed_mps = max(speed_mps, 0.0)
        curvature_per_steer = (
            self.gains.curvature_per_steer_at_rest + self.gains.curvature_per_steer_per_mps * forward_speed_mps
        )
        steer = float(np.clip(target_curvature_per_m / curvature_per_steer, -1.0, 1.0))
        if hold_standstill:
            self._command = None
            return steer, 0.0, 1.0, False
        drive_direction = -1.0 if reverse else 1.0
        throttle, brake = pedals(
            self.gains.reverse if reverse else self.gains.gears[self._engaged_gear - 1],
            self.gains.max_engine_braking_mps2,
            self.gains.brake_per_mps2,
            gear == 0,
            drive_direction * target_accel_mps2,
            max(drive_direction * speed_mps, 0.0),
        )
        self._command = (target_accel_mps2, target_curvature_per_m, speed_mps, brake, REVERSE_GEAR if reverse else gear)
        return steer, throttle, brake, reverse

    def stats(self):
        """How far the gains are off: realised minus target acceleration per pedal and per gear (ticks, bias),
        realised over target yaw rate."""
        if not self._accel_rows:
            return {}
        target_accel, realised_accel, brake, gear = np.array(self._accel_rows).T
        accel_error = realised_accel - target_accel
        on_brake = brake > 0.0
        out = {"accel_abs_err_mps2": float(np.abs(accel_error).mean()), "ticks": len(accel_error)}
        out["accel_bias_by_gear_mps2"] = {
            int(g): (int((gear == g).sum()), round(float(accel_error[gear == g].mean()), 2)) for g in np.unique(gear)
        }
        if not on_brake.all():
            out["accel_bias_throttle_mps2"] = float(accel_error[~on_brake].mean())
        if on_brake.any():
            out["accel_bias_brake_mps2"] = float(accel_error[on_brake].mean())
        if self._yaw_rate_rows:
            target_yaw_rate, realised_yaw_rate = np.array(self._yaw_rate_rows).T
            target_power = float(target_yaw_rate @ target_yaw_rate)
            if target_power > 0.0:
                out["yaw_rate_gain"] = float(target_yaw_rate @ realised_yaw_rate) / target_power
        return out
