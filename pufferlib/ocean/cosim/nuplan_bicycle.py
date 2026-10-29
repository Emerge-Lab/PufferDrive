"""Commands for nuPlan's KinematicBicycleModel from the shadow Drive env's post-step ego targets.

The devkit model low-passes acceleration and steering with gain dt/(dt+tau) before a forward Euler step and
has neither a speed floor nor a stop snap: both are enforced here on the realised (low-passed) acceleration.
"""

from dataclasses import dataclass

DEVKIT_ACCEL_TIME_CONSTANT_S = 0.2
DEVKIT_STEERING_TIME_CONSTANT_S = 0.05


@dataclass(frozen=True)
class BicycleCommandConfig:
    dt_s: float
    min_speed_mps: float  # floor on the model's next speed; 0 = CaRL's no-reverse driving
    undo_low_pass: bool  # pre-compensate the command lag so one model step lands on the targets
    accel_time_constant_s: float = DEVKIT_ACCEL_TIME_CONSTANT_S
    steering_time_constant_s: float = DEVKIT_STEERING_TIME_CONSTANT_S

    def __post_init__(self):
        if not self.dt_s > 0:
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")
        if not self.min_speed_mps <= 0:
            raise ValueError(f"min_speed_mps must be <= 0, got {self.min_speed_mps}")
        if not (self.accel_time_constant_s >= 0 and self.steering_time_constant_s >= 0):
            raise ValueError("the low-pass time constants must be >= 0")


def low_pass_gain(dt_s: float, time_constant_s: float) -> float:
    return dt_s / (dt_s + time_constant_s)


def low_pass_response(current: float, command: float, gain: float) -> float:
    return current + gain * (command - current)


def low_pass_command(current: float, target: float, gain: float) -> float:
    return current + (target - current) / gain


def bicycle_commands(
    cfg: BicycleCommandConfig,
    target_accel_mps2: float,
    target_steering_rad: float,
    current_accel_mps2: float,
    current_steering_rad: float,
    speed_mps: float,
    stop_this_step: bool = False,
):
    """-> (acceleration [m/s^2], steering rate [rad/s]) for a CaRL ActionTrajectory over the next dt.
    stop_this_step: the shadow env snapped its speed to zero this step, so the realised acceleration brings
    the model to exactly zero speed too."""
    accel_gain = low_pass_gain(cfg.dt_s, cfg.accel_time_constant_s)
    steering_gain = low_pass_gain(cfg.dt_s, cfg.steering_time_constant_s)
    accel_cmd = target_accel_mps2
    steering_cmd = target_steering_rad
    if cfg.undo_low_pass:
        accel_cmd = low_pass_command(current_accel_mps2, target_accel_mps2, accel_gain)
        steering_cmd = low_pass_command(current_steering_rad, target_steering_rad, steering_gain)
    if stop_this_step:
        accel_cmd = low_pass_command(current_accel_mps2, -speed_mps / cfg.dt_s, accel_gain)
    floor_accel_mps2 = (cfg.min_speed_mps - speed_mps) / cfg.dt_s
    accel_cmd = max(accel_cmd, low_pass_command(current_accel_mps2, floor_accel_mps2, accel_gain))
    return accel_cmd, (steering_cmd - current_steering_rad) / cfg.dt_s


def bicycle_response(
    cfg: BicycleCommandConfig, current_accel_mps2: float, accel_cmd_mps2: float, speed_mps: float, elapsed_s: float
):
    """(acceleration, speed) the devkit model reaches after elapsed_s under the command: the tracking guard."""
    accel = low_pass_response(current_accel_mps2, accel_cmd_mps2, low_pass_gain(elapsed_s, cfg.accel_time_constant_s))
    return accel, speed_mps + accel * elapsed_s
