"""Commands for nuPlan's KinematicBicycleModel from the shadow Drive env's post-step ego targets.

The devkit model low-passes acceleration and steering with gain dt/(dt+tau) before a forward Euler step and
has neither a speed floor nor a stop snap: both are enforced here on the realised (low-passed) acceleration.
"""

import math
from dataclasses import dataclass
from typing import Optional

DEVKIT_ACCEL_TIME_CONSTANT_S = 0.2
DEVKIT_STEERING_TIME_CONSTANT_S = 0.05
YAW_JERK_MIN_SPEED_MPS = 0.1  # below this the yaw rate is ~0 whatever the wheel does; the atan bounds degenerate
FLOOR_SPEED_TOL_MPS = 1e-6  # the floor lands the model on min_speed up to float rounding


@dataclass(frozen=True)
class BicycleCommandConfig:
    dt_s: float
    min_speed_mps: float  # floor on the model's next speed; 0 = CaRL's no-reverse driving
    undo_low_pass: bool  # pre-compensate the command lag so one model step lands on the targets
    accel_time_constant_s: float = DEVKIT_ACCEL_TIME_CONSTANT_S
    steering_time_constant_s: float = DEVKIT_STEERING_TIME_CONSTANT_S
    center_accel_cap_mps2: float = 0.0  # cap on the devkit's center accel (rear + d*yaw_rate^2 + d*yaw_accel); 0 = off
    max_yaw_jerk_radps3: float = 0.0  # per-step change of yaw accel bounded via the steering command; 0 = off
    wheel_base_m: float = 0.0
    rear_axle_to_center_m: float = 0.0

    def __post_init__(self):
        if not self.dt_s > 0:
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")
        if not self.min_speed_mps <= 0:
            raise ValueError(f"min_speed_mps must be <= 0, got {self.min_speed_mps}")
        if not (self.accel_time_constant_s >= 0 and self.steering_time_constant_s >= 0):
            raise ValueError("the low-pass time constants must be >= 0")
        if not self.center_accel_cap_mps2 >= 0:
            raise ValueError(f"center_accel_cap_mps2 must be >= 0, got {self.center_accel_cap_mps2}")
        if self.center_accel_cap_mps2 > 0 and not (self.wheel_base_m > 0 and self.rear_axle_to_center_m >= 0):
            raise ValueError("center_accel_cap_mps2 needs wheel_base_m > 0 and rear_axle_to_center_m >= 0")
        if not self.max_yaw_jerk_radps3 >= 0:
            raise ValueError(f"max_yaw_jerk_radps3 must be >= 0, got {self.max_yaw_jerk_radps3}")
        if self.max_yaw_jerk_radps3 > 0 and not self.wheel_base_m > 0:
            raise ValueError("max_yaw_jerk_radps3 needs wheel_base_m > 0")


def low_pass_gain(dt_s: float, time_constant_s: float) -> float:
    return dt_s / (dt_s + time_constant_s)


def low_pass_response(current: float, command: float, gain: float) -> float:
    return current + gain * (command - current)


def low_pass_command(current: float, target: float, gain: float) -> float:
    return current + (target - current) / gain


def floor_clamped_intent_accel(cfg: BicycleCommandConfig, speed_mps: float, intent_accel_mps2: float) -> float:
    """Drive zeroes the acceleration when the car stops; with the model held on the floor a negative intent would
    wind down to the clip while the speed never moves, a state the policy never saw."""
    if speed_mps <= cfg.min_speed_mps + FLOOR_SPEED_TOL_MPS and intent_accel_mps2 < 0:
        return 0.0
    return intent_accel_mps2


def center_accel_shift(
    cfg: BicycleCommandConfig, accel_mps2: float, speed_mps: float, steering_next_rad: float, yaw_rate_radps: float
) -> float:
    """What the devkit adds to the rear-axle acceleration at the vehicle center after one model step."""
    speed_next = speed_mps + accel_mps2 * cfg.dt_s
    yaw_rate_next = speed_next * math.tan(steering_next_rad) / cfg.wheel_base_m
    yaw_accel_next = (yaw_rate_next - yaw_rate_radps) / cfg.dt_s
    return cfg.rear_axle_to_center_m * (yaw_rate_next * yaw_rate_next + yaw_accel_next)


def clip_steering_for_yaw_jerk(
    cfg: BicycleCommandConfig,
    steering_next_rad: float,
    speed_next_mps: float,
    yaw_rate_radps: float,
    yaw_rate_prev_radps: float,
) -> float:
    """Next steering angle such that the yaw acceleration changes by at most max_yaw_jerk * dt within the step."""
    if speed_next_mps < YAW_JERK_MIN_SPEED_MPS:
        return steering_next_rad
    yaw_rate_window = cfg.max_yaw_jerk_radps3 * cfg.dt_s * cfg.dt_s
    yaw_rate_center = 2.0 * yaw_rate_radps - yaw_rate_prev_radps
    steering_lo = math.atan((yaw_rate_center - yaw_rate_window) * cfg.wheel_base_m / speed_next_mps)
    steering_hi = math.atan((yaw_rate_center + yaw_rate_window) * cfg.wheel_base_m / speed_next_mps)
    return min(max(steering_next_rad, steering_lo), steering_hi)


def bicycle_commands(
    cfg: BicycleCommandConfig,
    target_accel_mps2: float,
    target_steering_rad: float,
    current_accel_mps2: float,
    current_steering_rad: float,
    speed_mps: float,
    yaw_rate_radps: float = 0.0,
    yaw_rate_prev_radps: Optional[float] = None,
    stop_this_step: bool = False,
):
    """-> (acceleration [m/s^2], steering rate [rad/s]) for a CaRL ActionTrajectory over the next dt.
    stop_this_step: the shadow env snapped its speed to zero this step, so the realised acceleration brings
    the model to exactly zero speed too. With center_accel_cap_mps2 the realised acceleration is reduced
    while turning so the center acceleration nuPlan scores stays under the cap (CaRL clip_angular_adjustment);
    with max_yaw_jerk_radps3 the steering command is slowed so the yaw acceleration (and with it the center
    acceleration's derivative, nuPlan's longitudinal jerk) cannot jump within a step."""
    accel_gain = low_pass_gain(cfg.dt_s, cfg.accel_time_constant_s)
    steering_gain = low_pass_gain(cfg.dt_s, cfg.steering_time_constant_s)
    accel_cmd = target_accel_mps2
    steering_cmd = target_steering_rad
    if cfg.undo_low_pass:
        accel_cmd = low_pass_command(current_accel_mps2, target_accel_mps2, accel_gain)
        steering_cmd = low_pass_command(current_steering_rad, target_steering_rad, steering_gain)
    if stop_this_step:
        accel_cmd = low_pass_command(current_accel_mps2, -speed_mps / cfg.dt_s, accel_gain)
    accel_realised = low_pass_response(current_accel_mps2, accel_cmd, accel_gain)
    if cfg.max_yaw_jerk_radps3 > 0 and yaw_rate_prev_radps is not None:
        steering_next = clip_steering_for_yaw_jerk(
            cfg,
            low_pass_response(current_steering_rad, steering_cmd, steering_gain),
            speed_mps + accel_realised * cfg.dt_s,
            yaw_rate_radps,
            yaw_rate_prev_radps,
        )
        steering_cmd = low_pass_command(current_steering_rad, steering_next, steering_gain)
    if cfg.center_accel_cap_mps2 > 0:
        steering_next = low_pass_response(current_steering_rad, steering_cmd, steering_gain)
        shift = center_accel_shift(cfg, accel_realised, speed_mps, steering_next, yaw_rate_radps)
        if accel_realised > cfg.center_accel_cap_mps2 - shift:
            accel_cmd = low_pass_command(current_accel_mps2, cfg.center_accel_cap_mps2 - shift, accel_gain)
    floor_accel_mps2 = (cfg.min_speed_mps - speed_mps) / cfg.dt_s
    accel_cmd = max(accel_cmd, low_pass_command(current_accel_mps2, floor_accel_mps2, accel_gain))
    return accel_cmd, (steering_cmd - current_steering_rad) / cfg.dt_s


def bicycle_response(
    cfg: BicycleCommandConfig, current_accel_mps2: float, accel_cmd_mps2: float, speed_mps: float, elapsed_s: float
):
    """(acceleration, speed) the devkit model reaches after elapsed_s under the command: the tracking guard."""
    accel = low_pass_response(current_accel_mps2, accel_cmd_mps2, low_pass_gain(elapsed_s, cfg.accel_time_constant_s))
    return accel, speed_mps + accel * elapsed_s
