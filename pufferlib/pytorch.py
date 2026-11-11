import numpy as np
import torch
from torch.distributions.utils import logits_to_probs


numpy_to_torch_dtype_dict = {
    np.dtype("float64"): torch.float64,
    np.dtype("float32"): torch.float32,
    np.dtype("float16"): torch.float16,
    np.dtype("uint64"): torch.uint64,
    np.dtype("uint32"): torch.uint32,
    np.dtype("uint16"): torch.uint16,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int64"): torch.int64,
    np.dtype("int32"): torch.int32,
    np.dtype("int16"): torch.int16,
    np.dtype("int8"): torch.int8,
}

ACTION_SELECT_SAMPLE = "sample"
ACTION_SELECT_MODE = "mode"  # argmax
ACTION_SELECT_MEAN = "mean"  # probability-weighted continuous mean


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def clean_state_key(key):
    prefixes = ("module.", "_orig_mod.")
    while key.startswith(prefixes):
        key = key.split(".", 1)[1]
    return key


def clean_policy_state_dict(state_dict):
    return {clean_state_key(k): v for k, v in state_dict.items()}


# taken from torch.distributions.Categorical
def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


# taken from torch.distributions.Categorical
def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(
    logits, action=None, action_selection=ACTION_SELECT_SAMPLE, env_continuous=None, policy=None
):  # TODO discrete continuous
    is_discrete = isinstance(logits, torch.Tensor)
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
    elif is_discrete:
        logits = logits.unsqueeze(0)
    # TODO: Double check this
    else:  # multi-discrete
        logits = torch.nn.utils.rnn.pad_sequence(
            [l.transpose(0, 1) for l in logits], batch_first=False, padding_value=-torch.inf
        ).permute(1, 2, 0)

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
        if is_discrete:
            return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0), cont_actions.squeeze(0)
        return action.T, logprob.sum(0), logits_entropy, cont_actions

    if is_discrete:
        return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0), None

    return action.T, logprob.sum(0), logits_entropy, None
