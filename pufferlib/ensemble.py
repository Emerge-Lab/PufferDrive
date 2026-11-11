"""Evaluation-time policy ensembles.

An ensemble checkpoint carries the weights of several trained policies of one architecture. load_policy_weights turns a
freshly built policy plus a loaded checkpoint into either that policy or an EnsemblePolicy: its mean action (what the
evals execute) is the mean of the members' mean actions, while its action distribution and value, used for mode and
sampled actions, logged probabilities and distillation labels, are the equal-weight averages over the members.
write_ensemble_checkpoint packs member checkpoints into a run-dir-shaped folder (final_model.pt + config.yaml) that
every eval harness takes like a training run.
"""

import copy
import math
import os
import threading

import torch
import yaml
from torch import nn

from pufferlib.pytorch import clean_policy_state_dict


ENSEMBLE_CHECKPOINT_KEY = "pufferdrive_ensemble"
ENSEMBLE_FORMAT_VERSION = 1
ENSEMBLE_CONFIG_KEY = "ensemble"
MIN_MEMBER_COUNT = 2
MIN_MIXTURE_VARIANCE = 1e-8
CHECKPOINT_FILE_NAME = "final_model.pt"
RUN_CONFIG_FILE_NAME = "config.yaml"
ARCHITECTURE_CONFIG_KEYS = ("policy_name", "rnn_name", "rnn")
POLICY_CONFIG_KEYS_IGNORED = ("fused_slot_encoder",)  # kernel choice, not architecture


class EnsemblePolicy(nn.Module):
    def __init__(self, members):
        super().__init__()
        if len(members) < MIN_MEMBER_COUNT:
            raise ValueError(f"an ensemble needs at least {MIN_MEMBER_COUNT} members, got {len(members)}")
        for member in members:
            if hasattr(member, "lstm"):
                raise ValueError("recurrent policies cannot be ensembled: every member would need its own state")
        self.members = nn.ModuleList(members)
        self.is_continuous = members[0].is_continuous
        self.fp32_heads = getattr(members[0], "fp32_heads", False)
        # sample_logits' mean-action hook only sees the averaged probabilities; it runs right after forward_eval
        self._member_probs_store = threading.local()

    @property
    def member_count(self):
        return len(self.members)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def forward_eval(self, observations, state=None):
        member_logits = []
        member_values = []
        for member in self.members:
            logits, value = member.forward_eval(observations, state)
            member_logits.append(logits)
            member_values.append(value.float())
        if isinstance(member_logits[0], torch.Tensor):
            self._member_probs_store.probs = [logits.float().softmax(dim=-1) for logits in member_logits]
        return average_member_predictions(member_logits), torch.stack(member_values).mean(dim=0)

    def discrete_actions_to_continuous(self, actions):
        return self.members[0].discrete_actions_to_continuous(actions)

    def discrete_probs_to_continuous_mean(self, probs):
        """Mean of the members' mean actions for the batch of the preceding forward_eval call."""
        member_probs = getattr(self._member_probs_store, "probs", None)
        if member_probs is None:
            raise RuntimeError("forward_eval must run before discrete_probs_to_continuous_mean")
        class_count = probs.shape[-1]
        if member_probs[0].shape != (probs.numel() // class_count, class_count):
            raise RuntimeError("discrete_probs_to_continuous_mean needs the member probabilities of the same batch")
        # averaged in physical units: the env's long scaling is piecewise, so normalised member means do not average
        mean_member_probs = torch.stack(member_probs).mean(dim=0)
        return self.members[0].discrete_probs_to_continuous_mean(mean_member_probs).reshape(*probs.shape[:-1], -1)


def _average_discrete_logits(member_logits):
    log_probs = torch.stack([logits.float() for logits in member_logits]).log_softmax(dim=-1)
    return torch.logsumexp(log_probs, dim=0) - math.log(len(member_logits))


def _average_normal_distributions(member_distributions):
    locs = torch.stack([distribution.loc.float() for distribution in member_distributions])
    scales = torch.stack([distribution.scale.float() for distribution in member_distributions])
    mean = locs.mean(dim=0)
    # moment-matched Gaussian: the mixture itself is not Gaussian
    variance = (scales.square() + locs.square()).mean(dim=0) - mean.square()
    return torch.distributions.Normal(mean, variance.clamp_min(MIN_MIXTURE_VARIANCE).sqrt())


def average_member_predictions(member_logits):
    """Logits of the equal-weight mixture of the members' action distributions, in the form sample_logits takes."""
    first = member_logits[0]
    if isinstance(first, torch.Tensor):
        return _average_discrete_logits(member_logits)
    if isinstance(first, torch.distributions.Normal):
        return _average_normal_distributions(member_logits)
    if isinstance(first, (tuple, list)):
        return type(first)(_average_discrete_logits(component) for component in zip(*member_logits))
    raise TypeError(f"cannot average policy outputs of type {type(first).__name__}")


def describe_policy(policy):
    if isinstance(policy, EnsemblePolicy):
        return f"ensemble of {policy.member_count} policies"
    return "policy"


def is_ensemble_checkpoint(checkpoint):
    return isinstance(checkpoint, dict) and ENSEMBLE_CHECKPOINT_KEY in checkpoint


def member_state_dicts(checkpoint):
    """Cleaned policy state dicts of a loaded checkpoint: one for a trained policy, every member of an ensemble."""
    if not is_ensemble_checkpoint(checkpoint):
        return [clean_policy_state_dict(checkpoint)]
    ensemble = checkpoint[ENSEMBLE_CHECKPOINT_KEY]
    format_version = ensemble.get("format_version")
    if format_version != ENSEMBLE_FORMAT_VERSION:
        raise ValueError(f"ensemble checkpoint format {format_version} is not the supported {ENSEMBLE_FORMAT_VERSION}")
    state_dicts = ensemble["member_state_dicts"]
    if len(state_dicts) < MIN_MEMBER_COUNT:
        raise ValueError(f"ensemble checkpoint holds {len(state_dicts)} members; at least {MIN_MEMBER_COUNT} needed")
    return [clean_policy_state_dict(state_dict) for state_dict in state_dicts]


def load_policy_weights(policy, checkpoint):
    """Load a torch.load()ed checkpoint into a freshly built policy: the policy itself, or an EnsemblePolicy of copies."""
    state_dicts = member_state_dicts(checkpoint)
    if len(state_dicts) == 1:
        policy.load_state_dict(state_dicts[0])
        return policy
    members = []
    for state_dict in state_dicts:
        member = copy.deepcopy(policy)
        member.load_state_dict(state_dict)
        members.append(member)
    return EnsemblePolicy(members)


def _policy_architecture(config):
    policy_config = config.get("policy") or {}
    return {key: value for key, value in policy_config.items() if key not in POLICY_CONFIG_KEYS_IGNORED}


def _check_member_matches_first(member_paths, member_configs, member_weights, member_idx, allow_env_mismatch):
    first_path, path = member_paths[0], member_paths[member_idx]
    first_config, config = member_configs[0], member_configs[member_idx]
    for key in ARCHITECTURE_CONFIG_KEYS:
        if first_config.get(key) != config.get(key):
            raise ValueError(f"{path}: {key} differs from {first_path}; an ensemble needs one architecture")
    if _policy_architecture(first_config) != _policy_architecture(config):
        raise ValueError(f"{path}: policy config differs from {first_path}; an ensemble needs one architecture")
    first_shapes = {name: tuple(tensor.shape) for name, tensor in member_weights[0].items()}
    shapes = {name: tuple(tensor.shape) for name, tensor in member_weights[member_idx].items()}
    if shapes != first_shapes:
        differing = sorted(
            name for name in shapes.keys() | first_shapes.keys() if shapes.get(name) != first_shapes.get(name)
        )
        raise ValueError(f"{path}: parameter shapes differ from {first_path}: {differing}")
    first_env, env = first_config.get("env") or {}, config.get("env") or {}
    differing_env_keys = sorted(key for key in env.keys() | first_env.keys() if env.get(key) != first_env.get(key))
    if differing_env_keys and not allow_env_mismatch:
        raise ValueError(
            f"{path}: env config differs from {first_path} in {differing_env_keys}; allow_env_mismatch evaluates the "
            "ensemble with the first member's env config"
        )


def write_ensemble_checkpoint(member_checkpoint_paths, output_dir, run_name, allow_env_mismatch=False):
    """Pack member checkpoints into <output_dir>/final_model.pt + config.yaml (the first member's training config)."""
    from pufferlib.ocean.cosim.arch import checkpoint_config_path

    member_paths = [os.path.abspath(path) for path in member_checkpoint_paths]
    if len(member_paths) < MIN_MEMBER_COUNT:
        raise ValueError(f"an ensemble needs at least {MIN_MEMBER_COUNT} member checkpoints, got {len(member_paths)}")
    if len(set(member_paths)) != len(member_paths):
        raise ValueError(f"member checkpoints repeat: {member_paths}")
    member_configs = []
    member_weights = []
    for path in member_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"member checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if is_ensemble_checkpoint(checkpoint):
            raise ValueError(f"{path} is itself an ensemble checkpoint; members must be trained policies")
        with open(checkpoint_config_path(path)) as config_file:
            member_configs.append(yaml.safe_load(config_file))
        member_weights.append(clean_policy_state_dict(checkpoint))
    if member_configs[0].get("rnn_name") is not None:
        raise ValueError("recurrent policies cannot be ensembled: every member would need its own state")
    for member_idx in range(1, len(member_paths)):
        _check_member_matches_first(member_paths, member_configs, member_weights, member_idx, allow_env_mismatch)

    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE_NAME)
    config_path = os.path.join(output_dir, RUN_CONFIG_FILE_NAME)
    if os.path.exists(checkpoint_path):
        existing = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not is_ensemble_checkpoint(existing):
            raise FileExistsError(f"{checkpoint_path} is a trained checkpoint, not an ensemble; refusing to overwrite")
    os.makedirs(output_dir, exist_ok=True)
    config = copy.deepcopy(member_configs[0])
    config["run_name"] = run_name
    config["wandb"] = False  # no training run to attach eval results to
    config[ENSEMBLE_CONFIG_KEY] = {
        "member_checkpoints": member_paths,
        "member_run_names": [member_config.get("run_name") for member_config in member_configs],
    }
    ensemble = {
        "format_version": ENSEMBLE_FORMAT_VERSION,
        "member_checkpoints": member_paths,
        "member_state_dicts": member_weights,
    }
    torch.save({ENSEMBLE_CHECKPOINT_KEY: ensemble}, checkpoint_path)
    with open(config_path, "w") as config_file:
        yaml.dump(config, config_file)
    return checkpoint_path, config_path
