"""Shadow-env Drive kwargs from a checkpoint config (CARLA + nuPlan co-sim).

The shadow env adopts EVERY Drive-accepted key from the checkpoint's
config.yaml env section -- full parity with the training env, no allowlist to
fall out of date -- then applies CLEAN_EVAL_OVERRIDES (the repo's clean-eval
profile), then the caller's structural co-sim keys (map_dir, num_agents,
scenario_length, ...) win.
"""

import inspect
from pathlib import Path

from pufferlib.ocean.drive.drive import Drive

# Mirror of the noise/light/conditioning keys in
# pufferlib/config/evaluation/benchmark.yaml, duplicated so the co-sim venvs
# (CARLA cp310 / nuPlan) never import the training stack;
# tests/unit_tests/test_cosim_config_contract.py pins the two equal.
# The reward_*/goal_* keys pin the conditioning obs to the GigaFlow paper's
# eval values (Table A2) instead of adopting the training config's
# reward_randomization=true, which would sample a random command profile.
CLEAN_EVAL_OVERRIDES = {
    "obs_dropout_lane": 0.0,
    "obs_dropout_boundary": 0.0,
    "partner_blindness_prob": 0.0,
    "partner_blindness_trigger_prob": 0.0,
    "phantom_braking_prob": 0.0,
    "phantom_braking_trigger_prob": 0.0,
    "traffic_light_behavior": "stop",
    "reward_randomization": False,
    "goal_speed": 3.0,
    "goal_radius": 10.0,
    "reward_collision": 3.0,
    "reward_offroad": 3.0,
    "reward_stop_line": 1.0,
    "reward_ade": 0.0,
    "reward_goal": 1.0,
    "reward_overspeed": 0.05,
    "reward_comfort": 0.05,
    "reward_velocity": 0.0025,
    "reward_lane_align": 0.025,
    "reward_lane_center": 0.0075,  # training-range max (REWARD_BOUNDS), not the paper's 0.0038
    "reward_timestep": 0.000025,
    "reward_reverse": 0.005,
    "pose_noise_xy_m": 0.0,
    "pose_noise_yaw_deg": 0.0,
}


# Some saved checkpoint configs store the infraction behaviors as resolved
# binding enums (0/1/2) instead of the strings Drive.__init__ accepts (e.g.
# weights/mimolette: collision_behavior: 1); normalize either spelling.
_INFRACTION_BEHAVIOR_NAMES = {0: "ignore", 1: "stop", 2: "remove"}
_INFRACTION_BEHAVIOR_KEYS = ("collision_behavior", "offroad_behavior", "traffic_light_behavior")


def checkpoint_config_path(checkpoint):
    """config.yaml saved with a checkpoint: next to a run-dir `final_model.pt`, or one level up for
    `models/model_*.pt`. Raises when neither exists (a shadow env built from Drive defaults would be
    silently wrong)."""
    checkpoint = Path(checkpoint).resolve()
    for candidate in (checkpoint.parent / "config.yaml", checkpoint.parents[1] / "config.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no config.yaml next to or one level above {checkpoint}")


# Applied to every co-sim shadow env after CLEAN_EVAL_OVERRIDES: the eval-only semantics the native
# benchmark gets from eval_mode (scripts/kesai/7_eval_8node_slurm.sh passes the 0.2 m margin), which the
# co-sim env cannot enter because eval_mode's scenario batching needs a full agent pool.
COSIM_EVAL_OVERRIDES = {
    "cosim_eval_semantics": True,
    "eval_perceived_size_margin_m": 0.2,
}


def shadow_env_kwargs(cfg, defaults=None, overrides=None):
    """Drive kwargs for a co-sim shadow env.

    Precedence: `defaults` (no-checkpoint fallback arch) < checkpoint env
    config (every Drive-accepted key) < CLEAN_EVAL_OVERRIDES < COSIM_EVAL_OVERRIDES < `overrides`
    (the co-sim's structural keys)."""
    accepted = set(inspect.signature(Drive.__init__).parameters)
    adopted = {k: v for k, v in ((cfg or {}).get("env") or {}).items() if k in accepted}
    for key in _INFRACTION_BEHAVIOR_KEYS:
        if key in adopted and isinstance(adopted[key], int):
            adopted[key] = _INFRACTION_BEHAVIOR_NAMES[adopted[key]]
    return {**(defaults or {}), **adopted, **CLEAN_EVAL_OVERRIDES, **COSIM_EVAL_OVERRIDES, **(overrides or {})}
