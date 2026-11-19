import numpy as np
import torch
from torch.distributions.utils import logits_to_probs


numpy_to_torch_dtype_dict = {np.dtype("float32"): torch.float32}

ACTION_SELECT_SAMPLE = "sample"
ACTION_SELECT_MODE = "mode"  # argmax
ACTION_SELECT_MEAN = "mean"  # probability-weighted continuous mean
FLOPS_PER_MULTIPLY_ADD = 2.0


def layer_init(layer, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.zeros_(layer.bias)
    return layer


def clean_state_key(key):
    prefixes = ("module.", "_orig_mod.")
    while key.startswith(prefixes):
        key = key.split(".", 1)[1]
    return key


def clean_policy_state_dict(state_dict):
    return {clean_state_key(k): v for k, v in state_dict.items()}


def load_policy_weights(policy, checkpoint):
    """Load a torch.load()ed checkpoint into a freshly built policy and return that policy."""
    policy.load_state_dict(clean_policy_state_dict(checkpoint))
    return policy


def log_prob(logits, value):
    return logits.gather(-1, value.long().unsqueeze(-1)).squeeze(-1)


# taken from torch.distributions.Categorical
def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(logits, action=None, action_selection=ACTION_SELECT_SAMPLE, env_continuous=None, policy=None):
    if action_selection == ACTION_SELECT_MEAN and not (env_continuous and not policy.is_continuous):
        raise ValueError("action_selection='mean' requires a discrete policy on a continuous env")
    if isinstance(logits, torch.distributions.Normal):
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)
            if action_selection != ACTION_SELECT_SAMPLE:
                action = logits.loc.view(batch, -1)

        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy, None

    logits = logits.unsqueeze(0)

    # This can fail on nans etc
    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = logits_to_probs(logits)

    if action is None:
        if action_selection == ACTION_SELECT_SAMPLE:
            probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
            action = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, replacement=True).int()
            action = action.reshape(probs.shape[:-1])
        else:  # MODE and MEAN both use argmax as the *nominal* discrete action (for logging)
            action = torch.argmax(probs, -1)
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)
    logprob = log_prob(normalized_logits, action)
    logits_entropy = entropy(normalized_logits).sum(0)

    if env_continuous and not policy.is_continuous:
        if action_selection == ACTION_SELECT_MEAN:
            cont_actions = policy.discrete_probs_to_continuous_mean(probs)
        else:  # Mode and sample we already have an action from the code above.
            cont_actions = policy.discrete_actions_to_continuous(action)
        return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0), cont_actions.squeeze(0)

    return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0), None
