"""The frozen PufferDrive policy that drives every car except yours.

Everything about the env -- observation layout, dt, goal count, conditioning ranges -- is adopted
from the checkpoint's own config.yaml, never hand-copied. A demo env whose observation layout
disagrees with the weights does not crash; it feeds the policy nonsense and the traffic merely
drives badly, which is the worst kind of bug to chase.
"""

import os

import numpy as np
import yaml

from pufferlib.ocean.cosim import arch
from pufferlib.ocean.drive.drive import Drive

# triton ships no macOS wheels. The fused slot encoder is a kernel choice with identical bf16 math
# and the same weight shapes, so turning it off loads the same checkpoint and changes no output.
DEMO_POLICY_OVERRIDES = {"fused_slot_encoder": False}

DEMO_ENV_OVERRIDES = {
    # The co-sim profile arch.py layers on is for scoring a replayed route, not for driving one.
    "cosim_eval_semantics": False,
    "eval_mode": False,
    # The world must not reset itself out from under the driver.
    "termination_mode": False,
    "terminate_on_goal": False,
    # Traffic personality: every car draws its own reward conditioning, so the cars around you
    # behave like different drivers. The level presets shift these bounds (M7).
    "reward_randomization": True,
    # Visible jitter reads as a rendering bug when you are looking at the cars rather than a metric.
    "pose_noise_xy_m": 0.0,
    "pose_noise_yaw_deg": 0.0,
    # Erratic-driver features are off by default; the level presets turn them up.
    "partner_blindness_prob": 0.0,
    "partner_blindness_prob_min": 0.0,
    "phantom_braking_prob": 0.0,
    "simulation_mode": "gigaflow",
    "control_mode": "control_vehicles",
    "goal_source": "map",
    # Training spawns 0.8 m to 7 m boxes on purpose; a demo wants cars.
    "vehicle_size_profile": 1,
}

DEMO_SCENARIO_LENGTH_STEPS = 20000


def load_checkpoint_config(checkpoint_path):
    config_path = arch.checkpoint_config_path(checkpoint_path)
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    if "env" not in config:
        raise ValueError(f"{config_path} has no 'env' section; it is not a PufferDrive run config")
    return config, config_path


def demo_env_kwargs(config, map_file, num_agents, overrides=None):
    """Drive kwargs for the demo: checkpoint parity, then the demo's own structural keys."""
    if not os.path.isfile(map_file):
        raise FileNotFoundError(f"map file does not exist: {map_file}")
    if num_agents < 2:
        raise ValueError(f"the demo needs at least 2 agents (you plus traffic), got {num_agents}")
    structural = {
        **DEMO_ENV_OVERRIDES,
        "map_dir": map_file,
        "num_maps": 1,
        "allow_map_subset": True,
        "num_agents": num_agents,
        "max_agents_per_env": num_agents,
        "min_agents_per_env": num_agents,
        "scenario_length": DEMO_SCENARIO_LENGTH_STEPS,
        **(overrides or {}),
    }
    return arch.shadow_env_kwargs(config, overrides=structural)


def build_demo_env(config, map_file, num_agents, overrides=None):
    return Drive(**demo_env_kwargs(config, map_file, num_agents, overrides))


class FrozenTraffic:
    """Runs the checkpoint's policy over every active agent once per policy step.

    The human car's row is computed like any other and deliberately KEPT rather than discarded:
    pushed back through the jerk chain it is what the policy would have done from where you are
    sitting, which is the reference signal the steering assist needs (M9).
    """

    def __init__(self, checkpoint_path, env, config, device="cpu"):
        import torch

        import pufferlib.pytorch
        import pufferlib.spaces
        from pufferlib.ocean.cosim.carla.shadow_ego import clean_policy_state_dict

        self.env = env
        self.device = device
        self.config = config

        state_dict = clean_policy_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        self._assert_layout_matches(state_dict, env)

        import pufferlib.ocean.torch as drive_torch

        policy_config = {**config["policy"], **DEMO_POLICY_OVERRIDES}
        policy_config.setdefault("action_type", config["env"]["action_type"])
        policy_cls = getattr(drive_torch, config.get("policy_name", "Drive"))
        self.policy = policy_cls(env, **policy_config).to(device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        self.env_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        self.action_selection = (
            pufferlib.pytorch.ACTION_SELECT_MEAN
            if self.env_continuous and not self.policy.is_continuous
            else pufferlib.pytorch.ACTION_SELECT_MODE
        )

    @staticmethod
    def _assert_layout_matches(state_dict, env):
        """Fail loudly on an observation-layout mismatch instead of feeding the policy nonsense."""
        expected = {
            "actor_backbone.ego_encoder.0.weight": ("ego_features", env.ego_features),
            "actor_backbone.lane_encoder.0.weight": ("lane_features", env.lane_features),
            "actor_backbone.boundary_encoder.0.weight": ("boundary_features", env.boundary_features),
            "actor_backbone.partner_encoder.0.weight": ("partner_features", env.partner_features),
        }
        mismatches = []
        for key, (name, env_value) in expected.items():
            weight = state_dict.get(key)
            if weight is not None and int(weight.shape[1]) != int(env_value):
                mismatches.append(f"{name}: checkpoint expects {int(weight.shape[1])}, env emits {env_value}")
        if mismatches:
            raise ValueError("checkpoint and env observation layouts disagree -- " + "; ".join(mismatches))

    def act(self, observations):
        """Continuous actions for every active agent, shaped to the env's action buffer."""
        import torch

        import pufferlib.pytorch

        with torch.no_grad():
            logits, _value = self.policy.forward_eval(torch.as_tensor(observations).to(self.device))
            action, _log_probs, _entropy, continuous_action = pufferlib.pytorch.sample_logits(
                logits,
                action_selection=self.action_selection,
                env_continuous=self.env_continuous,
                policy=self.policy,
            )
        chosen = continuous_action if self.env_continuous and continuous_action is not None else action
        actions = chosen.cpu().numpy()
        if not self.env_continuous:
            return actions.astype(np.int32)
        return actions.astype(np.float32)
