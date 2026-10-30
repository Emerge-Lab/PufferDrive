"""The demo loop: your car on the wheel, every other car on the frozen policy.

Runs on a FIXED timestep. The sub-step count is a constant, never derived from wall-clock, so the
same seed and the same recorded input trace replay to the same trajectory. When the machine falls
behind, frames are dropped -- the physics step never stretches.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from pufferlib.wheel_demo.ffb import CarFeel
from pufferlib.wheel_demo.maps import DEFAULT_MAP_DIR, resolve_carla_map_files
from pufferlib.wheel_demo.policy_host import FrozenTraffic, build_demo_env, load_checkpoint_config

from pufferlib.ocean.drive import binding

GRIP_MODE_KINEMATIC = binding.GRIP_MODE_KINEMATIC
GRIP_MODE_REALISTIC = binding.GRIP_MODE_REALISTIC
GEARBOX_MODE_MANUAL = binding.GEARBOX_MODE_MANUAL
GEARBOX_MODE_AUTOMATIC = binding.GEARBOX_MODE_AUTOMATIC
SUBSTEP_MODE_ALL = binding.SUBSTEP_MODE_ALL
SUBSTEP_MODE_HUMAN_ONLY = binding.SUBSTEP_MODE_HUMAN_ONLY

# 60 Hz physics inside a 3.33 Hz policy step: matched to the display, and above the rate at
# which anything the driver can feel changes.
DEFAULT_SUBSTEPS_PER_POLICY_STEP = 18


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
        self.traffic = FrozenTraffic(config.checkpoint_path, self.env, checkpoint_config, config.device)

        self.env.set_substepping(config.substep_mode, config.substeps_per_policy_step)
        self.substeps_per_policy_step = config.substeps_per_policy_step
        self.substep_dt = self.dt / config.substeps_per_policy_step
        self.env.set_human_grip_mode(config.grip_mode)
        self.env.set_human_gearbox_mode(config.gearbox_mode)
        self.env.set_human_gear(config.start_gear)
        self.select_human_agent(config.human_agent_idx)
        self._last_shift_up = 0
        self._last_shift_down = 0
        self._pending_shifts = 0
        self._last_steering_angle_rad = 0.0
        self._last_feel_time_s = time.perf_counter()

    def select_human_agent(self, agent_idx):
        self.env.select_human_agent(agent_idx)
        self.human_agent_idx = agent_idx

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
        actions = self.traffic.act(self.observations)
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
            self.env.substep()
            if on_substep is not None:
                callback_start = time.perf_counter()
                on_substep(substep_idx, wheel_state)
                self.telemetry.substep_callback_seconds += time.perf_counter() - callback_start

        self.observations, rewards, terminals, truncations, info = self.env.step(actions)
        done = time.perf_counter()

        self.telemetry.policy_steps += 1
        self.telemetry.policy_seconds += env_start - policy_start
        self.telemetry.env_seconds += (
            (done - env_start) - self.telemetry.substep_callback_seconds + previous_callback_seconds
        )
        return wheel_state, info

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
        steering_angle_rad = -human["steering_angle_rad"]
        interval_s = max(now_s - self._last_feel_time_s, 1e-4)
        steering_rate_rad_s = (steering_angle_rad - self._last_steering_angle_rad) / interval_s
        self._last_steering_angle_rad = steering_angle_rad
        self._last_feel_time_s = now_s
        return CarFeel(
            steering_angle_rad=steering_angle_rad,
            steering_rate_rad_s=steering_rate_rad_s,
            speed_mps=human["speed_mps"],
            accel_lat_mps2=-human["accel_lat_mps2"],
            tire_grip_mps2=binding.HUMAN_TIRE_GRIP_MPS2,
            collision=human["collision"] > 0.0,
            offroad=False,
        )

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
