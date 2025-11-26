import math
import numpy as np


def _linear_schedule(start, end, progress):
    return start + (end - start) * progress


def _quadratic_schedule(start, end, progress):
    return start + (end - start) * progress**2


def _sqrt_schedule(start, end, progress):
    return start + (end - start) * np.sqrt(progress)

SCHEDULE_BUILDERS = {
    "linear": _linear_schedule,
    "quadratic": _quadratic_schedule,
    "sqrt": _sqrt_schedule,
}

GOAL_CURRICULUM_SCHEDULES = tuple(SCHEDULE_BUILDERS.keys())


class GoalCurriculum:
    def __init__(self, start_distance, end_distance, steps, schedule="linear", total_timesteps=None):
        steps = int(steps)
        if steps < 1:
            raise ValueError("goal_curriculum_steps must be >= 1")

        schedule_key = str(schedule).lower()
        if schedule_key not in SCHEDULE_BUILDERS:
            raise ValueError(f"Unknown goal curriculum schedule: {schedule}")

        start_distance = float(start_distance)
        end_distance = float(end_distance)
        progress = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        builder = SCHEDULE_BUILDERS[schedule_key]
        self.distances = builder(start_distance, end_distance, progress).astype(np.float32)

        self.current_idx = 0
        self.agent_steps = 0

        if isinstance(total_timesteps, str):
            if total_timesteps.lower() in ("auto", "none"):
                total_timesteps = None
            else:
                total_timesteps = total_timesteps.replace("_", "")

        self.total_timesteps = None if total_timesteps is None else int(total_timesteps)
        self.step_interval = None
        if self.total_timesteps is not None:
            transitions = max(1, steps - 1)
            self.step_interval = max(1, math.ceil(self.total_timesteps / transitions))

    @property
    def current_distance(self):
        return float(self.distances[self.current_idx])

    def consume_agent_steps(self, agent_steps):
        """Advance based on cumulative agent-steps budget if configured."""
        if self.step_interval is None:
            return False

        self.agent_steps += max(0, int(agent_steps))
        target_idx = min(self.agent_steps // self.step_interval, len(self.distances) - 1)
        changed = target_idx != self.current_idx
        self.current_idx = target_idx
        return changed

    def advance_one_stage(self):
        """Advance a single curriculum step (used when not timestep-driven)."""
        if self.step_interval is not None:
            return False

        if self.current_idx < len(self.distances) - 1:
            self.current_idx += 1
            return True

        return False

    def reset(self):
        self.current_idx = 0
        self.agent_steps = 0
