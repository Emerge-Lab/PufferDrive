"""Loading the weights of a trained policy."""

from pufferlib.pytorch import clean_policy_state_dict


def load_policy_weights(policy, checkpoint):
    """Load a torch.load()ed checkpoint into a freshly built policy and return that policy."""
    policy.load_state_dict(clean_policy_state_dict(checkpoint))
    return policy
