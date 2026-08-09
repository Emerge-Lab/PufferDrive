"""Shadow-env Drive kwargs from a checkpoint config (CARLA + nuPlan co-sim).

The shadow env adopts EVERY Drive-accepted key from the checkpoint's
config.yaml env section -- full parity with the training env, no allowlist to
fall out of date -- then applies CLEAN_EVAL_OVERRIDES (the repo's clean-eval
profile), then the caller's structural co-sim keys (map_dir, num_agents,
scenario_length, ...) win.
"""

import inspect

from pufferlib.ocean.drive.drive import Drive

# Mirror of the noise/light keys in pufferlib/config/evaluation/benchmark.yaml,
# duplicated so the co-sim venvs (CARLA cp310 / nuPlan) never import the training
# stack; tests/unit_tests/test_cosim_config_contract.py pins the two equal.
# Train-time observation noise is disabled, and traffic_light_behavior="stop"
# keeps the benchmark's red-light semantics: a shadow-detected violation
# latches the ego stopped (drive.h apply_infraction_behavior).
CLEAN_EVAL_OVERRIDES = {
    "obs_dropout_lane": 0.0,
    "obs_dropout_boundary": 0.0,
    "partner_blindness_prob": 0.0,
    "partner_blindness_trigger_prob": 0.0,
    "phantom_braking_prob": 0.0,
    "phantom_braking_trigger_prob": 0.0,
    "traffic_light_behavior": "stop",
}


# Some saved checkpoint configs store the infraction behaviors as resolved
# binding enums (0/1/2) instead of the strings Drive.__init__ accepts (e.g.
# weights/mimolette: collision_behavior: 1); normalize either spelling.
_INFRACTION_BEHAVIOR_NAMES = {0: "ignore", 1: "stop", 2: "remove"}
_INFRACTION_BEHAVIOR_KEYS = ("collision_behavior", "offroad_behavior", "traffic_light_behavior")


def shadow_env_kwargs(cfg, defaults=None, overrides=None):
    """Drive kwargs for a co-sim shadow env.

    Precedence: `defaults` (no-checkpoint fallback arch) < checkpoint env
    config (every Drive-accepted key) < CLEAN_EVAL_OVERRIDES < `overrides`
    (the co-sim's structural keys)."""
    accepted = set(inspect.signature(Drive.__init__).parameters)
    adopted = {k: v for k, v in ((cfg or {}).get("env") or {}).items() if k in accepted}
    for key in _INFRACTION_BEHAVIOR_KEYS:
        if key in adopted and isinstance(adopted[key], int):
            adopted[key] = _INFRACTION_BEHAVIOR_NAMES[adopted[key]]
    return {**(defaults or {}), **adopted, **CLEAN_EVAL_OVERRIDES, **(overrides or {})}
