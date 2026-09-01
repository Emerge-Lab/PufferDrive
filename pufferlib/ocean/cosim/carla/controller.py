"""Kinematic target -> carla.VehicleControl tracking controller.

The policy's jerk action is integrated by the shadow PufferDrive env into a
target (speed, yaw) one policy-dt ahead; this controller makes CARLA's physics
chase that target. CARLA (not the shadow env) moves the ego, so all leaderboard
collision/infraction machinery sees a normally-driven vehicle.

Longitudinal: PI on speed error -> throttle/brake.
Lateral: desired yaw rate over the policy horizon -> inverse kinematic bicycle
-> front wheel angle, normalized by the vehicle's max steer angle.

Tracking error is accumulated per route (see stats()) — it is the number that
disambiguates "bad policy" from "bad controller" when scores are low.
"""

import math

import carla
import numpy as np

from pufferlib.ocean.cosim.carla_bridge import wrap_deg_180


def read_vehicle_geometry(vehicle):
    """(wheelbase_m, max_steer_rad) from the vehicle's physics parameters
    (read-only). Falls back to Lincoln MKZ-ish defaults on any surprise."""
    try:
        phys = vehicle.get_physics_control()
        wheels = phys.wheels
        max_steer_deg = max(w.max_steer_angle for w in wheels)
        steering = [w for w in wheels if w.max_steer_angle > 0.0]
        fixed = [w for w in wheels if w.max_steer_angle <= 0.0]
        if steering and fixed:
            front = np.mean([[w.position.x, w.position.y, w.position.z] for w in steering], axis=0)
            rear = np.mean([[w.position.x, w.position.y, w.position.z] for w in fixed], axis=0)
            wheelbase = float(np.linalg.norm(front - rear)) / 100.0  # positions are in cm
        else:
            wheelbase = 2.85
        if not (1.5 <= wheelbase <= 5.0):
            wheelbase = 2.85
        if not (10.0 <= max_steer_deg <= 90.0):
            max_steer_deg = 70.0
        return wheelbase, math.radians(max_steer_deg)
    except Exception:
        return 2.85, math.radians(70.0)


class TrackingController:
    def __init__(
        self,
        wheelbase_m=2.85,
        max_steer_rad=math.radians(70.0),
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
        """One control tick. current_* from CARLA ground truth, target_* from the
        shadow env's integrated policy action (held between policy steps)."""
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

        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

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
