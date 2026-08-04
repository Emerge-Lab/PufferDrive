"""Unit tests for evaluation-time action selection (sample / mode / mean).

The feature lets a user pick how eval turns policy logits into actions on a
*continuous* environment, for either a discrete or a continuous policy:

  - sample: draw from the policy distribution (stochastic)
  - mode:   argmax / Gaussian mean (deterministic "best guess")
  - mean:   probability-weighted continuous mean (discrete policy only)

The logic lives in `pufferlib.pytorch.sample_logits`; the mode is chosen by
`Evaluator.action_selection` (config -> validated -> passed to sample_logits).
Both are covered here without an env/C-sim/GPU.
"""

import sys

import pytest
import torch

import pufferlib.pytorch as P

# 3 discrete classes -> fixed 2D continuous embedding, mirroring Drive's
# action_table[num_classes, 2] (discrete action index -> (long, lat)).
ACTION_TABLE = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])


class _DiscretePolicy:
    """Minimal stand-in for a discrete Drive policy on a continuous env."""

    is_continuous = False

    def discrete_actions_to_continuous(self, actions):
        return ACTION_TABLE[actions.long()]

    def discrete_probs_to_continuous_mean(self, probs):
        return probs @ ACTION_TABLE.to(probs.dtype)


class _ContinuousPolicy:
    is_continuous = True


def _discrete_logits(rows):
    # Drive's discrete policy returns a tuple of one [batch, num_classes]
    # tensor (torch.split output); mirror that so we exercise the real path.
    return (torch.tensor(rows),)


# --------------------------------------------------------------------------
# Discrete policy on a continuous env
# --------------------------------------------------------------------------
def test_mode_selects_argmax_and_is_deterministic():
    # Row 0 argmax = class 2, row 1 argmax = class 0.
    logits = _discrete_logits([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])
    policy = _DiscretePolicy()

    action1, _, _, cont1 = P.sample_logits(
        logits, action_selection=P.ACTION_SELECT_MODE, env_continuous=True, policy=policy
    )
    action2, _, _, cont2 = P.sample_logits(
        logits, action_selection=P.ACTION_SELECT_MODE, env_continuous=True, policy=policy
    )

    assert action1.reshape(-1).tolist() == [2, 0]
    # Continuous action is the exact table row for the argmax class.
    expected_cont = torch.stack([ACTION_TABLE[2], ACTION_TABLE[0]])
    assert torch.allclose(cont1.reshape(-1, 2), expected_cont)
    # Deterministic: repeated calls are identical.
    assert torch.equal(action1, action2)
    assert torch.equal(cont1, cont2)


def test_mean_is_probability_weighted_and_differs_from_mode():
    # Uniform distribution over the 3 classes.
    logits = _discrete_logits([[0.0, 0.0, 0.0]])
    policy = _DiscretePolicy()

    action_mean, _, _, cont_mean = P.sample_logits(
        logits, action_selection=P.ACTION_SELECT_MEAN, env_continuous=True, policy=policy
    )
    action_mode, _, _, cont_mode = P.sample_logits(
        logits, action_selection=P.ACTION_SELECT_MODE, env_continuous=True, policy=policy
    )

    # mean == probability-weighted continuous mean == average of table rows == [0, 0].
    assert torch.allclose(cont_mean.reshape(-1), torch.zeros(2), atol=1e-6)
    # mode picks the argmax row ([-1, -1] for a uniform tie) -> distinct from mean.
    assert not torch.allclose(cont_mean.reshape(-1), cont_mode.reshape(-1))
    # The nominal discrete action reported for mean is the argmax (for logging).
    assert action_mean.reshape(-1).tolist() == action_mode.reshape(-1).tolist()


def test_sample_is_stochastic_and_stays_in_support():
    # Flat distribution so multinomial explores every class.
    logits = _discrete_logits([[0.0, 0.0, 0.0]])
    policy = _DiscretePolicy()

    torch.manual_seed(0)
    seen_classes = set()
    for _ in range(50):
        action, _, _, cont = P.sample_logits(
            logits, action_selection=P.ACTION_SELECT_SAMPLE, env_continuous=True, policy=policy
        )
        idx = int(action.reshape(-1).item())
        seen_classes.add(idx)
        assert idx in (0, 1, 2)
        # Continuous action must be the table row of the *sampled* class.
        assert torch.allclose(cont.reshape(-1), ACTION_TABLE[idx])
    # Genuinely sampling, not collapsing to argmax.
    assert len(seen_classes) > 1


# --------------------------------------------------------------------------
# Continuous policy on a continuous env
# --------------------------------------------------------------------------
def test_continuous_policy_mode_returns_gaussian_mean():
    loc = torch.tensor([[0.5, -0.5]])
    dist = torch.distributions.Normal(loc, torch.ones(1, 2))

    action, _, _, cont = P.sample_logits(
        dist, action_selection=P.ACTION_SELECT_MODE, env_continuous=True, policy=_ContinuousPolicy()
    )

    # A continuous policy already emits a continuous action -> no discrete->cont conversion.
    assert cont is None
    # Mode of a Gaussian is its mean (loc), deterministically.
    assert torch.allclose(action.reshape(1, 2), loc)


def test_continuous_policy_sample_differs_from_mode():
    loc = torch.tensor([[0.5, -0.5]])
    dist = torch.distributions.Normal(loc, torch.ones(1, 2))

    torch.manual_seed(0)
    action_sample, _, _, _ = P.sample_logits(
        dist, action_selection=P.ACTION_SELECT_SAMPLE, env_continuous=True, policy=_ContinuousPolicy()
    )

    assert not torch.allclose(action_sample.reshape(1, 2), loc)


# --------------------------------------------------------------------------
# Guards: `mean` is only valid for a discrete policy on a continuous env
# --------------------------------------------------------------------------
def test_mean_rejects_continuous_policy():
    dist = torch.distributions.Normal(torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError):
        P.sample_logits(dist, action_selection=P.ACTION_SELECT_MEAN, env_continuous=True, policy=_ContinuousPolicy())


def test_mean_rejects_discrete_env():
    logits = _discrete_logits([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        P.sample_logits(logits, action_selection=P.ACTION_SELECT_MEAN, env_continuous=False, policy=_DiscretePolicy())


# --------------------------------------------------------------------------
# Eval config plumbing: eval.action_selection -> validated -> sample_logits
# --------------------------------------------------------------------------
VALID_ACTION_SELECTIONS = (P.ACTION_SELECT_SAMPLE, P.ACTION_SELECT_MODE, P.ACTION_SELECT_MEAN)


def _eval_args(action_selection):
    # Minimal args for pufferl.eval; validation runs before any benchmark loading.
    return {
        "eval": {
            "action_selection": action_selection,
            "benchmark_config": "pufferlib/config/evaluation/benchmark.yaml",
            "benchmarks": None,
            "output_name": None,
            "render_scenarios": False,
            "render_filter": None,
            "max_rendered_failures": None,
            "failure_replay_csv": None,
            "max_sdc_replay_workers": 1,
        }
    }


def test_shipped_config_declares_a_valid_action_selection(monkeypatch):
    # Guards the key itself: a config refactor that drops or renames
    # eval.action_selection must fail here, not at the first eval run.
    import pufferlib.pufferl as pufferl

    # load_config treats everything left in argv as a Hydra override.
    monkeypatch.setattr(sys, "argv", ["puffer"])
    args = pufferl.load_config("puffer_drive")
    assert args["eval"]["action_selection"] in VALID_ACTION_SELECTIONS


def test_eval_rejects_invalid_action_selection():
    import pufferlib
    import pufferlib.pufferl as pufferl

    with pytest.raises(pufferlib.APIUsageError):
        pufferl.eval(env_name="puffer_drive", args=_eval_args("banana"))
