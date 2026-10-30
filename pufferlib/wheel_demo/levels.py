"""Difficulty levels: how the traffic around you behaves.

Every car draws its reward conditioning from a window, and the policy reads those coefficients as
part of its observation. Shifting the window therefore changes how the surrounding cars drive,
mid-episode, with no reload and no retraining.

Two deliberate choices:

- A level shifts the WINDOW, it does not pin a value. Pinning every car to one coefficient vector
  would make all the traffic behave identically, which reads as obviously fake. Shifting the window
  moves the average while keeping the per-driver variety.
- Windows are expressed as FRACTIONS of the policy's own trained range, so a level can never ask
  for a coefficient the policy has not seen. Outside that range the conditioning signal saturates
  at +-1 and the level silently stops having any effect -- which looks like the feature is broken
  rather than like a bad setting.
"""

from dataclasses import dataclass, field

from pufferlib.ocean.drive import binding

COEF_COLLISION = binding.REWARD_COEF_COLLISION
COEF_OFFROAD = binding.REWARD_COEF_OFFROAD
COEF_COMFORT = binding.REWARD_COEF_COMFORT
COEF_LANE_ALIGN = binding.REWARD_COEF_LANE_ALIGN
COEF_LANE_CENTER = binding.REWARD_COEF_LANE_CENTER
COEF_STOP_LINE = binding.REWARD_COEF_STOP_LINE
COEF_OVERSPEED = binding.REWARD_COEF_OVERSPEED
COEF_VEL_ALIGN = binding.REWARD_COEF_VEL_ALIGN
# C_vel: the per-car speed cap. Its trained window is [1/conditioning_speed_scale, scale].
COEF_SPEED = binding.REWARD_COEF_SPEED


@dataclass(frozen=True)
class Level:
    """`windows` maps a coefficient index to (low, high) as fractions of its trained range.
    Coefficients left out keep the full trained range."""

    name: str
    description: str
    windows: dict = field(default_factory=dict)


# Careful drivers: they brake for you, hold their lane, and stop at the line.
CALM = Level(
    name="calm",
    description="patient traffic that yields and keeps its distance",
    windows={
        COEF_COLLISION: (0.70, 1.00),
        COEF_OFFROAD: (0.70, 1.00),
        COEF_COMFORT: (0.55, 1.00),
        COEF_LANE_ALIGN: (0.55, 1.00),
        COEF_LANE_CENTER: (0.55, 1.00),
        COEF_STOP_LINE: (0.75, 1.00),
        COEF_OVERSPEED: (0.60, 1.00),
        COEF_VEL_ALIGN: (0.50, 1.00),
        COEF_SPEED: (0.00, 0.45),
    },
)

# The distribution the policy was actually trained on: the full spread of driver personalities.
NORMAL = Level(name="normal", description="the training distribution, every kind of driver")

# Low regard for collisions, lane discipline, stop lines and speed limits.
AGGRESSIVE = Level(
    name="aggressive",
    description="impatient traffic that crowds you and runs lines",
    windows={
        COEF_COLLISION: (0.00, 0.30),
        COEF_OFFROAD: (0.00, 0.35),
        COEF_COMFORT: (0.00, 0.25),
        COEF_LANE_ALIGN: (0.00, 0.30),
        COEF_LANE_CENTER: (0.00, 0.30),
        COEF_STOP_LINE: (0.00, 0.25),
        COEF_OVERSPEED: (0.00, 0.20),
        COEF_VEL_ALIGN: (0.00, 0.40),
        COEF_SPEED: (0.60, 1.00),
    },
)

LEVELS = (CALM, NORMAL, AGGRESSIVE)
LEVEL_NAMES = tuple(level.name for level in LEVELS)


def level_by_name(name):
    for level in LEVELS:
        if level.name == name:
            return level
    raise ValueError(f"unknown level {name!r}; choose one of {LEVEL_NAMES}")


def level_bounds(env, level):
    """Absolute (min, max) coefficient arrays for this level, from the env's trained limits."""
    trained_min, trained_max = env.get_reward_coef_limits()
    low = trained_min.copy()
    high = trained_max.copy()
    for coef_idx, (low_fraction, high_fraction) in level.windows.items():
        if not 0.0 <= low_fraction <= high_fraction <= 1.0:
            raise ValueError(
                f"level {level.name!r} coefficient {coef_idx} has window ({low_fraction}, {high_fraction}); "
                "fractions must satisfy 0 <= low <= high <= 1")
        span = trained_max[coef_idx] - trained_min[coef_idx]
        low[coef_idx] = trained_min[coef_idx] + low_fraction * span
        high[coef_idx] = trained_min[coef_idx] + high_fraction * span
    return low, high


def apply_level(env, level):
    """Set the window and redraw every policy car from it. Returns how many cars changed.

    The human car is untouched: its coefficients are pinned to a fixed reference profile, so your
    own car's capability never shifts when the difficulty does.
    """
    env.set_reward_coef_bounds(*level_bounds(env, level))
    return env.resample_reward_coefs()
