"""Gradient-free policy ego controller for ReGentS generation and evaluation.

The ego is the only active agent in a ReGentS Drive, so this loads one checkpoint per
worker process and turns its observations into the continuous actions C consumes. No
gradient crosses this module: it is an alternative implementation of the ego rollout,
exactly like native IDM or log replay.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from pufferlib.ocean.drive import binding
from pufferlib.pytorch import ACTION_SELECT_MEAN, ACTION_SELECT_MODE, sample_logits

# Sampling would make an ego rollout depend on torch RNG state, which breaks the
# input+config+seed determinism contract across ego refreshes and spawn workers.
DETERMINISTIC_ACTION_SELECTIONS = (ACTION_SELECT_MEAN, ACTION_SELECT_MODE)

_POLICY_CACHE = {}


@dataclass(frozen=True)
class ReGentSPolicyEgoConfig:
    checkpoint_path: str
    config_path: str
    action_selection: str = ACTION_SELECT_MEAN
    device: str = "cpu"

    def __post_init__(self):
        for label, value in (("checkpoint_path", self.checkpoint_path), ("config_path", self.config_path)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Policy ego {label} must be a non-empty string")
            if not Path(value).is_file():
                raise ValueError(f"Policy ego {label} does not exist: {value}")
        if self.action_selection not in DETERMINISTIC_ACTION_SELECTIONS:
            raise ValueError(
                f"Policy ego action_selection must be one of {DETERMINISTIC_ACTION_SELECTIONS}, "
                f"got {self.action_selection!r}"
            )
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("Policy ego device must be a non-empty string")
        torch.device(self.device)


def checkpoint_digest(checkpoint_path):
    """Hash the checkpoint bytes so an artifact identifies the exact ego it targeted."""
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_policy(policy_config, drive):
    # Deferred: pufferl imports the generation entry point, so a module-level import cycles.
    from pufferlib.ocean.torch import Drive as DrivePolicy
    from pufferlib.ocean.torch import TargetDrive
    from pufferlib.pufferl import clean_policy_state_dict

    with open(policy_config.config_path) as handle:
        checkpoint_config = yaml.safe_load(handle)
    if not isinstance(checkpoint_config, dict) or "policy" not in checkpoint_config:
        raise ValueError(f"Checkpoint config {policy_config.config_path} has no 'policy' section")
    if checkpoint_config.get("policy_name") != "Drive":
        raise ValueError("ReGentS policy ego supports policy_name='Drive'")
    if checkpoint_config.get("rnn_name") is not None:
        raise ValueError("ReGentS policy ego does not support recurrent checkpoints")

    device = torch.device(policy_config.device)
    state_dict = torch.load(policy_config.checkpoint_path, map_location=device, weights_only=True)
    state_dict = clean_policy_state_dict(state_dict)
    partner_weight = state_dict.get("actor_backbone.partner_encoder.0.weight")
    if partner_weight is None or partner_weight.ndim != 2:
        raise ValueError("Policy ego checkpoint has no recognizable actor partner-encoder input layer")
    checkpoint_partner_feature_count = int(partner_weight.shape[1])
    if checkpoint_partner_feature_count == drive.partner_features:
        policy_class = DrivePolicy
    elif checkpoint_partner_feature_count == drive.partner_features - 1:
        policy_class = TargetDrive
    else:
        raise ValueError(
            f"Policy ego checkpoint expects {checkpoint_partner_feature_count} partner features, "
            f"but Drive exposes {drive.partner_features}"
        )
    policy = policy_class(drive, **checkpoint_config["policy"]).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


class PolicyEgoActor:
    """Turn Drive observations into the continuous ego action for one scenario."""

    def __init__(self, policy_config, drive):
        if not isinstance(policy_config, ReGentSPolicyEgoConfig):
            raise TypeError("policy_config must be a ReGentSPolicyEgoConfig")
        if drive.sdc_controller != binding.CONTROLLER_POLICY:
            raise ValueError("Policy ego requires sdc_controller='policy'")
        if drive._action_type_flag != binding.ACTION_TYPE_CONTINUOUS:
            raise ValueError("Policy ego requires action_type='continuous'")
        cache_key = (policy_config.checkpoint_path, policy_config.device)
        if cache_key not in _POLICY_CACHE:
            _POLICY_CACHE[cache_key] = _load_policy(policy_config, drive)
        self.policy = _POLICY_CACHE[cache_key]
        self.action_selection = policy_config.action_selection
        self.device = torch.device(policy_config.device)
        self.action_shape = drive.actions.shape

    def __call__(self, observations):
        with torch.no_grad():
            logits, _ = self.policy.forward_eval(torch.as_tensor(observations, device=self.device))
            _, _, _, continuous_action = sample_logits(
                logits,
                action_selection=self.action_selection,
                env_continuous=True,
                policy=self.policy,
            )
        actions = continuous_action.reshape(self.action_shape).float().cpu().numpy()
        if not np.isfinite(actions).all():
            raise RuntimeError("Policy ego emitted a non-finite action")
        return np.clip(actions, -1.0, 1.0)
