"""The demo loop: your car on the wheel, every other car on the frozen policy.

Runs on a FIXED timestep. The sub-step count is a constant, never derived from wall-clock, so the
same seed and the same recorded input trace replay to the same trajectory. When the machine falls
behind, frames are dropped -- the physics step never stretches.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pufferlib.wheel_demo.ffb import CarFeel
from pufferlib.wheel_demo.maps import DEFAULT_MAP_DIR, resolve_carla_map_files
from pufferlib.wheel_demo.policy_host import FrozenTraffic, build_demo_env, load_checkpoint_config

from pufferlib.ocean.drive import binding

GRIP_MODE_KINEMATIC = binding.GRIP_MODE_KINEMATIC
GRIP_MODE_REALISTIC = binding.GRIP_MODE_REALISTIC
GRIP_MODE_SKID = binding.GRIP_MODE_SKID
GEARBOX_MODE_MANUAL = binding.GEARBOX_MODE_MANUAL
GEARBOX_MODE_AUTOMATIC = binding.GEARBOX_MODE_AUTOMATIC
GEARBOX_MODE_NONE = binding.GEARBOX_MODE_NONE
SUBSTEP_MODE_ALL = binding.SUBSTEP_MODE_ALL
SUBSTEP_MODE_HUMAN_ONLY = binding.SUBSTEP_MODE_HUMAN_ONLY

# 60 Hz physics inside a 3.33 Hz policy step: matched to the display, and above the rate at
# which anything the driver can feel changes.
DEFAULT_SUBSTEPS_PER_POLICY_STEP = 18
# Steering a stopped car moves nothing, so the policy's angle there is noise; the wheel holds its last target.
ASSIST_WHEEL_HOLD_SPEED_MPS = 1.0
# Soft deadband on the policy's angle: a change this size moves the wheel at half speed, smaller ones barely at all.
ASSIST_WHEEL_DEADBAND_RAD = 8.0 / 450.0 * binding.HUMAN_STEERING_LIMIT_RAD
ASSIST_WHEEL_SMOOTHING_SECONDS = 0.15
# Look-ahead: the policy drives an imagined copy of your car forward, decisions at its own 0.3 s rate and physics
# at 100 Hz, and the mean angle over a window ahead of now (or centred on now) becomes the wheel's target, or under
# FULL_ASSIST_DYNAMICS_SET_STEERING the car's own steering.
WHEEL_TARGETS = ("lookahead", "filtered", "car")
LOOKAHEAD_PHYSICS_HZ = 100
LOOKAHEAD_WINDOW_SECONDS = 0.6
FULL_ASSIST_DYNAMICS_SET_STEERING = binding.FULL_ASSIST_DYNAMICS_SET_STEERING
# FULL steering, both with speed from the jerk model: the car steers itself with the 0.6 s look-ahead mean and the
# wheel shows it, or the wheel steers the car toward a 0.9 s look-ahead target. Picked from the all-towns ablation.
FULL_STEERING_PRESETS = {
    "car": (binding.FULL_ASSIST_DYNAMICS_SET_STEERING, 0.6),
    "wheel": (binding.FULL_ASSIST_DYNAMICS_JERK_SPEED, 0.9),
}
DEFAULT_FULL_STEERING = "car"


@dataclass
class DemoConfig:
    checkpoint_path: str
    town: str = "Town10HD"
    map_dir: str = DEFAULT_MAP_DIR
    num_agents: int = 64
    human_agent_idx: int = 0
    grip_mode: int = GRIP_MODE_REALISTIC
    gearbox_mode: int = GEARBOX_MODE_AUTOMATIC
    start_gear: int = 1
    substep_mode: int = SUBSTEP_MODE_ALL
    substeps_per_policy_step: int = DEFAULT_SUBSTEPS_PER_POLICY_STEP
    device: str = "cpu"
    action_selection: str = "mode"
    wheel_target: str = "lookahead"
    full_assist_dynamics: int = FULL_STEERING_PRESETS[DEFAULT_FULL_STEERING][0]
    lookahead_window_seconds: float = FULL_STEERING_PRESETS[DEFAULT_FULL_STEERING][1]
    # half the window from the car's own recent steering, half imagined ahead, so the mean does not lead
    lookahead_centred: bool = False
    seed: int = 0
    env_overrides: dict = field(default_factory=dict)

    def map_file(self):
        matches = [p for p in resolve_carla_map_files(self.map_dir) if p.endswith(f"__{self.town}.bin")]
        if not matches:
            raise ValueError(f"town {self.town!r} not found in {self.map_dir}")
        return matches[0]


@dataclass
class DemoTelemetry:
    policy_steps: int = 0
    dropped_frames: int = 0
    policy_seconds: float = 0.0
    env_seconds: float = 0.0
    substep_callback_seconds: float = 0.0

    def mean_policy_ms(self):
        return 1000.0 * self.policy_seconds / max(self.policy_steps, 1)

    def mean_env_ms(self):
        return 1000.0 * self.env_seconds / max(self.policy_steps, 1)

    def mean_substep_callback_ms(self):
        """Rendering lives here. With a window open it paces to the display, so this is wall-clock
        by design rather than compute cost."""
        return 1000.0 * self.substep_callback_seconds / max(self.policy_steps, 1)


class WheelDemoLoop:
    """Owns the env, the frozen policy and the wheel. One policy step per call to step_once()."""

    def __init__(self, config, wheel_input):
        self.config = config
        self.wheel = wheel_input
        self.telemetry = DemoTelemetry()

        checkpoint_config, _ = load_checkpoint_config(config.checkpoint_path)
        self.checkpoint_config = checkpoint_config
        self.env = build_demo_env(checkpoint_config, config.map_file(), config.num_agents, config.env_overrides)
        self.dt = float(self.env.dt)
        self.observations, _ = self.env.reset(seed=config.seed)
        self.last_rewards = None
        self.traffic = FrozenTraffic(
            config.checkpoint_path, self.env, checkpoint_config, config.device, config.action_selection
        )

        self.env.set_substepping(config.substep_mode, config.substeps_per_policy_step)
        self.substeps_per_policy_step = config.substeps_per_policy_step
        self.substep_dt = self.dt / config.substeps_per_policy_step
        self.env.set_human_grip_mode(config.grip_mode)
        self.env.set_human_gearbox_mode(config.gearbox_mode)
        self.env.set_human_gear(config.start_gear)
        self.lookahead_dt = 1.0 / LOOKAHEAD_PHYSICS_HZ
        self.set_full_steering(config.full_assist_dynamics, config.lookahead_window_seconds)
        self.select_human_agent(config.human_agent_idx)
        self._last_shift_up = 0
        self._last_shift_down = 0
        self._pending_shifts = 0
        self._last_steering_angle_rad = 0.0
        self._last_feel_time_s = time.perf_counter()
        self._assist_wheel_target_rad = None
        # (target, sim-time rate, when) published together, read by the force-feedback thread
        self._assist_wheel_target_motion = (None, 0.0, time.perf_counter())
        # sim seconds per wall second over the last policy step: 1 in real time, below it when recording lockstep
        self._sim_speed_ratio = 1.0
        self._last_step_start_s = None
        self._previous_lookahead_rad = None

    def set_full_steering(self, full_assist_dynamics, lookahead_window_seconds):
        """What steers your car under FULL and how far the look-ahead averages; safe between policy steps."""
        self.config.full_assist_dynamics = full_assist_dynamics
        self.config.lookahead_window_seconds = lookahead_window_seconds
        self.env.set_full_assist_dynamics(full_assist_dynamics)
        self.lookahead_steers_car = full_assist_dynamics == FULL_ASSIST_DYNAMICS_SET_STEERING
        centred = self.config.lookahead_centred
        ahead_seconds = lookahead_window_seconds * (0.5 if centred else 1.0)
        self.lookahead_window_count = round(ahead_seconds / self.lookahead_dt)
        # the previous drive, one step older, must still cover the window at this step's last sub-step
        self.lookahead_policy_steps = int(np.ceil((2.0 * self.dt + ahead_seconds) / self.dt - 1e-9))
        past_count = round(0.5 * lookahead_window_seconds / self.substep_dt) if centred else 0
        self._recent_car_steering_rad = deque(maxlen=max(past_count, 1))
        # drives imagined for the old window are the wrong length for the new one
        self._lookahead_rad = None
        self._previous_lookahead_rad = None

    @property
    def full_steering(self):
        """The FULL_STEERING_PRESETS name in use, or "custom" when set by hand."""
        current = (self.config.full_assist_dynamics, self.config.lookahead_window_seconds)
        return next((name for name, preset in FULL_STEERING_PRESETS.items() if preset == current), "custom")

    def restart(self):
        """Back to the start: the session's seed puts every car and goal where they began. Assist, steering preset,
        driving style and traffic level live in the env and carry over."""
        self.observations, _ = self.env.reset(seed=self.config.seed)
        self.select_human_agent(self.human_agent_idx)
        self.env.set_human_gear(self.config.start_gear)
        self.env.restart_human_engine()
        self._pending_shifts = 0
        self._previous_lookahead_rad = None
        self._last_step_start_s = None

    def select_human_agent(self, agent_idx):
        self.env.select_human_agent(agent_idx)
        self.human_agent_idx = agent_idx
        self._assist_wheel_target_rad = None
        self._lookahead_rad = None
        self._recent_car_steering_rad.clear()
        self._assist_wheel_target_motion = (None, 0.0, time.perf_counter())

    def human_state(self):
        return self.env.get_human_state()

    def consume_shift_requests(self, wheel_state):
        """Paddle pulls arrive as counts and are queued, so pulls that land between two policy
        steps are applied one per step rather than collapsed into one."""
        self._pending_shifts += wheel_state.shift_up_count - self._last_shift_up
        self._pending_shifts -= wheel_state.shift_down_count - self._last_shift_down
        self._last_shift_up = wheel_state.shift_up_count
        self._last_shift_down = wheel_state.shift_down_count
        if self._pending_shifts > 0:
            self._pending_shifts -= 1
            return 1
        if self._pending_shifts < 0:
            self._pending_shifts += 1
            return -1
        return 0

    def step_once(self, on_substep=None):
        """One policy decision, then `substeps_per_policy_step` physics sub-steps that each re-read
        the wheel. `on_substep(substep_idx, wheel_state)` is where the renderer draws a frame."""
        policy_start = time.perf_counter()
        if self._last_step_start_s is not None:
            self._sim_speed_ratio = min(1.0, self.dt / max(policy_start - self._last_step_start_s, 1e-6))
        self._last_step_start_s = policy_start
        actions = self.traffic.act(self.observations)
        if self.config.wheel_target == "lookahead" or self.lookahead_steers_car:
            self._look_ahead(actions)
        self.env.actions[:] = actions  # the sub-steps integrate this; step() only scores it
        env_start = time.perf_counter()

        previous_callback_seconds = self.telemetry.substep_callback_seconds
        wheel_state = self.wheel.latest()
        for substep_idx in range(self.substeps_per_policy_step):
            wheel_state = self.wheel.latest()
            self.env.set_human_pedals(
                wheel_state.steering_norm,
                wheel_state.throttle_norm,
                wheel_state.brake_norm,
                wheel_state.clutch_norm,
                self.consume_shift_requests(wheel_state),
            )
            if self.lookahead_steers_car:
                lookahead_rad = self._lookahead_mean_rad(substep_idx)
                self.env.set_assist_steering(None if lookahead_rad is None else -lookahead_rad)
            self.env.substep()
            self._recent_car_steering_rad.append(-self.env.get_human_state()["steering_angle_rad"])
            previous_target_rad = self._assist_wheel_target_rad
            assist_mode = self.env.get_assist_state()["mode"]
            car_steers_itself = self.lookahead_steers_car and assist_mode == binding.ASSIST_MODE_FULL
            if assist_mode == binding.ASSIST_MODE_BLEND:
                self._update_blend_wheel_target()
            elif car_steers_itself:
                # only under FULL does the car steer itself; under NUDGE the car is the wheel, so it would pull nowhere
                self._update_car_wheel_target()
            elif self.config.wheel_target == "lookahead":
                self._update_lookahead_wheel_target(substep_idx)
            elif self.config.wheel_target == "car":
                self._update_car_wheel_target()
            else:
                self._update_assist_wheel_target()
            self._publish_wheel_target_motion(previous_target_rad)
            if on_substep is not None:
                callback_start = time.perf_counter()
                on_substep(substep_idx, wheel_state)
                self.telemetry.substep_callback_seconds += time.perf_counter() - callback_start

        self.observations, rewards, terminals, truncations, info = self.env.step(actions)
        self.last_rewards = rewards  # every car's reward for this policy step (co-adapt scores yours)
        done = time.perf_counter()

        self.telemetry.policy_steps += 1
        self.telemetry.policy_seconds += env_start - policy_start
        self.telemetry.env_seconds += (
            (done - env_start) - self.telemetry.substep_callback_seconds + previous_callback_seconds
        )
        return wheel_state, info

    def _publish_wheel_target_motion(self, previous_target_rad):
        target_rad = self._assist_wheel_target_rad
        moving = target_rad is not None and previous_target_rad is not None
        rate_rad_s = (target_rad - previous_target_rad) / self.substep_dt if moving else 0.0
        self._assist_wheel_target_motion = (target_rad, rate_rad_s, time.perf_counter())

    def car_feel(self):
        """Car state in DRIVER-facing signs for the force model: positive means RIGHT.

        The sim is positive-left (positive curvature turns left), so steering and lateral
        acceleration are both negated here. Getting this backwards makes the wheel fight the
        driver instead of centring -- the same inversion that made the car steer the wrong way.
        """
        human = self.env.get_human_state()
        if human is None:
            return None
        now_s = time.perf_counter()
        # the wheel itself, not the sim's copy: a frame of lag is enough to make the assist servo overshoot
        steering_angle_rad = self.wheel.latest().steering_norm * binding.HUMAN_STEERING_LIMIT_RAD
        interval_s = max(now_s - self._last_feel_time_s, 1e-4)
        steering_rate_rad_s = (steering_angle_rad - self._last_steering_angle_rad) / interval_s
        self._last_steering_angle_rad = steering_angle_rad
        self._last_feel_time_s = now_s
        target_rad, target_sim_rate_rad_s, published_s = self._assist_wheel_target_motion
        # the wheel moves in wall time, so the target's rate is too; at most one sub-step of motion is carried on
        target_rate_rad_s = target_sim_rate_rad_s * self._sim_speed_ratio
        if target_rad is not None:
            # carry the target on through the policy's pause at each step start, so the wheel is not left behind
            target_rad += target_sim_rate_rad_s * min((now_s - published_s) * self._sim_speed_ratio, self.substep_dt)
        return CarFeel(
            steering_angle_rad=steering_angle_rad,
            steering_rate_rad_s=steering_rate_rad_s,
            speed_mps=human["speed_mps"],
            accel_lat_mps2=self._felt_lateral_accel_mps2(human, steering_angle_rad),
            tire_grip_mps2=binding.HUMAN_TIRE_GRIP_MPS2,
            collision=human["collision"] > 0.0,
            offroad=human["offroad"] > 0.0,
            assist_target_steering_rad=target_rad,
            assist_full=self.env.get_assist_state()["mode"] == binding.ASSIST_MODE_FULL,
            assist_target_rate_rad_s=target_rate_rad_s,
        )

    def _felt_lateral_accel_mps2(self, human, steering_angle_rad):
        """Cornering the wheel's aligning torque is built from, driver sign. Under BLEND the car's front wheels are
        partly the policy's, not this wheel's, so the torque comes from the cornering this wheel's own angle asks for:
        the car's would push the wheel against the policy's turn."""
        if self.env.get_assist_state()["mode"] != binding.ASSIST_MODE_BLEND:
            return -human["accel_lat_mps2"]
        speed_mps = human["speed_mps"]
        wheel_accel_mps2 = speed_mps * speed_mps * math.tan(steering_angle_rad) / human["wheelbase_m"]
        return float(np.clip(wheel_accel_mps2, -binding.HUMAN_TIRE_GRIP_MPS2, binding.HUMAN_TIRE_GRIP_MPS2))

    def _update_assist_wheel_target(self):
        """The policy's angle in driver terms for the wheel to follow (NUDGE, FULL), held while nearly stopped.
        Low-passed with a soft deadband: small back-and-forth changes are mostly ignored, steady or large
        ones are followed, and the weight between the two is smooth rather than a threshold."""
        assist = self.env.get_assist_state()
        steers_the_wheel = assist["mode"] in (binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL)
        if not steers_the_wheel or not assist["reference_valid"]:
            self._assist_wheel_target_rad = None
            return
        policy_rad = -assist["reference_steering_rad"]
        if self._assist_wheel_target_rad is None:
            self._assist_wheel_target_rad = policy_rad
        if abs(self.env.get_human_state()["speed_mps"]) < ASSIST_WHEEL_HOLD_SPEED_MPS:
            return
        gap_rad = policy_rad - self._assist_wheel_target_rad
        deadband_weight = (
            gap_rad * gap_rad / (gap_rad * gap_rad + ASSIST_WHEEL_DEADBAND_RAD * ASSIST_WHEEL_DEADBAND_RAD)
        )
        smoothing = 1.0 - np.exp(-self.substep_dt / ASSIST_WHEEL_SMOOTHING_SECONDS)
        self._assist_wheel_target_rad += deadband_weight * smoothing * gap_rad

    def _update_blend_wheel_target(self):
        """Under BLEND the wheel is pulled toward the angle the car is actually steered at, so it lines up with where
        the car goes. That gap is blend_weight * (policy - wheel): no pull at 0%, the policy's at 100%."""
        assist = self.env.get_assist_state()
        if not assist["reference_valid"]:
            self._assist_wheel_target_rad = None
            return
        self._assist_wheel_target_rad = -assist["commanded_steering_rad"]

    def _update_car_wheel_target(self):
        """The angle your car is actually steering at (NUDGE, FULL); under FULL on the jerk model that is the policy's."""
        assist = self.env.get_assist_state()
        steers_the_wheel = assist["mode"] in (binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL)
        if not steers_the_wheel or not assist["reference_valid"]:
            self._assist_wheel_target_rad = None
            return
        self._assist_wheel_target_rad = -self.env.get_human_state()["steering_angle_rad"]

    def _look_ahead(self, actions):
        """The policy's own drive of your car from where it is now, imagined at LOOKAHEAD_PHYSICS_HZ with
        every other car holding still. Steering only, in driver sign; the env is left exactly as it was."""
        pieces_per_step = round(self.dt / self.lookahead_dt)
        self._previous_lookahead_rad = None if self._lookahead_rad is None else self._lookahead_rad[pieces_per_step:]
        self._lookahead_rad = None
        slot = self.env.begin_preview()
        if slot < 0:
            return
        steering_rad = np.zeros((self.lookahead_policy_steps, pieces_per_step), dtype=np.float32)
        try:
            self.env.actions[slot] = actions[slot]
            for step_idx in range(self.lookahead_policy_steps):
                if step_idx > 0:
                    self.env.actions[slot] = self.traffic.act(self.env.observations[slot : slot + 1])[0]
                self.env.preview_step(steering_rad[step_idx])
        finally:
            self.env.end_preview()
        self._lookahead_rad = -steering_rad.reshape(-1)

    def _lookahead_mean_rad(self, substep_idx):
        """Mean imagined angle (driver sign) over the window ahead of this sub-step's end, crossfaded across the
        policy step from the previous drive to this one so a new plan never jumps; centred, it is averaged equally
        with the car's own steering over the half-window behind. None when there is no drive to average."""
        if self._lookahead_rad is None:
            return None
        start_s = (substep_idx + 1) * self.substep_dt
        ahead_rad = self._window_mean_rad(self._lookahead_rad, start_s)
        if self._previous_lookahead_rad is not None:
            new_plan_weight = (substep_idx + 1) / self.substeps_per_policy_step
            previous_rad = self._window_mean_rad(self._previous_lookahead_rad, start_s)
            ahead_rad = new_plan_weight * ahead_rad + (1.0 - new_plan_weight) * previous_rad
        if not self.config.lookahead_centred or not self._recent_car_steering_rad:
            return ahead_rad
        return 0.5 * ahead_rad + 0.5 * float(np.mean(self._recent_car_steering_rad))

    def _window_mean_rad(self, samples_rad, start_s):
        """Mean of a drive sampled every lookahead_dt (sample k at (k + 1) * lookahead_dt) over the window from start_s.
        Taken from the running integral at exact times, so the window slides smoothly even though 60 Hz sub-steps
        do not land on 100 Hz samples; rounding to whole samples steps it by 2, 2, 1 and ripples the angle."""
        sample_times_s = (np.arange(len(samples_rad)) + 1) * self.lookahead_dt
        integral_rad_s = np.concatenate(([0.0], np.cumsum(samples_rad) * self.lookahead_dt))
        integral_times_s = np.concatenate(([0.0], sample_times_s))
        window_s = self.lookahead_window_count * self.lookahead_dt
        end_s = start_s + window_s
        assert end_s <= integral_times_s[-1] + 1e-9, "look-ahead drive is shorter than the window"
        return float(
            (np.interp(end_s, integral_times_s, integral_rad_s) - np.interp(start_s, integral_times_s, integral_rad_s))
            / window_s
        )

    def _update_lookahead_wheel_target(self, substep_idx):
        """The look-ahead mean as the wheel's target (NUDGE, FULL)."""
        assist = self.env.get_assist_state()
        steers_the_wheel = assist["mode"] in (binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL)
        if not steers_the_wheel or not assist["reference_valid"]:
            self._assist_wheel_target_rad = None
            return
        self._assist_wheel_target_rad = self._lookahead_mean_rad(substep_idx)

    def render_poses(self, substep_idx):
        """World poses for drawing. Under SUBSTEP_MODE_ALL these are the real poses; under
        HUMAN_ONLY partners are dead-reckoned forward from the last policy step."""
        count = self.observations.shape[0]
        if not hasattr(self, "_pose_buffers") or self._pose_buffers[0].shape[0] < count:
            self._pose_buffers = tuple(np.zeros(count, dtype=np.float32) for _ in range(3))
        x, y, heading = self._pose_buffers
        self.env.get_render_poses(substep_idx * self.substep_dt, x, y, heading)
        return x, y, heading

    def run(self, policy_steps, realtime=False):
        """Fixed-timestep run. In realtime mode a step that overruns its budget drops the wait
        rather than stretching dt, so the physics stays deterministic."""
        next_deadline = time.perf_counter() + self.dt
        for _ in range(policy_steps):
            self.step_once()
            if not realtime:
                continue
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                self.telemetry.dropped_frames += 1
            next_deadline += self.dt

    def close(self):
        self.env.close()
