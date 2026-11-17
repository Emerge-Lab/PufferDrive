"""Eval-time action selection (sample / mode / mean) in opengigaflow.pytorch.sample_logits and eval.action_selection."""

import sys

import pytest
import torch

import opengigaflow
import opengigaflow.pytorch as P


# discrete action index -> (long, lat), like the policy's action_table[num_classes, 2]
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


def test_mode_selects_argmax_and_is_deterministic():
    logits = torch.tensor([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])
    policy = _DiscretePolicy()

    action1, _, _, cont1 = P.sample_logits(
        logits, action_selection=P.ACTION_SELECT_MODE, env_continuous=True, policy=policy
    )
    action2, _, _, cont2 = P.sample_logits(
        logits, action_selection=P.ACTION_SELECT_MODE, env_continuous=True, policy=policy
    )

    assert action1.reshape(-1).tolist() == [2, 0]
    expected_cont = torch.stack([ACTION_TABLE[2], ACTION_TABLE[0]])
    assert torch.allclose(cont1.reshape(-1, 2), expected_cont)
    assert torch.equal(action1, action2)
    assert torch.equal(cont1, cont2)


def test_mean_is_probability_weighted_and_differs_from_mode():
    logits = torch.tensor([[0.0, 0.0, 0.0]])
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
    logits = torch.tensor([[0.0, 0.0, 0.0]])
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


def test_mean_rejects_continuous_policy():
    dist = torch.distributions.Normal(torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError):
        P.sample_logits(dist, action_selection=P.ACTION_SELECT_MEAN, env_continuous=True, policy=_ContinuousPolicy())


def test_mean_rejects_discrete_env():
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        P.sample_logits(logits, action_selection=P.ACTION_SELECT_MEAN, env_continuous=False, policy=_DiscretePolicy())


VALID_ACTION_SELECTIONS = (P.ACTION_SELECT_SAMPLE, P.ACTION_SELECT_MODE, P.ACTION_SELECT_MEAN)


def test_shipped_config_declares_a_valid_action_selection():
    """A config refactor that drops or renames eval.action_selection fails here, not at the first eval run."""
    import opengigaflow.rl as rl

    args = rl.load_config("opengigaflow", overrides=[])
    assert args["eval"]["action_selection"] in VALID_ACTION_SELECTIONS


def test_config_rejects_invalid_action_selection(monkeypatch):
    import opengigaflow.rl as rl

    monkeypatch.setattr(sys, "argv", ["opengigaflow", "eval.action_selection=banana"])
    with pytest.raises(opengigaflow.APIUsageError):
        rl.load_config("opengigaflow")


def test_drive_mean_action_is_exact_in_physical_units():
    """Drive's mean is exact in physical units despite the piecewise jerk decode."""
    import opengigaflow.ocean.torch as ocean_torch
    import opengigaflow.rl as rl
    from opengigaflow.ocean.drive import binding
    from opengigaflow.ocean.drive.drive import Drive as DriveEnv

    args = rl.load_config("opengigaflow", overrides=[])
    env = DriveEnv(num_agents=64, map_dir="opengigaflow/resources/carla", num_maps=1, scenario_length=200)
    try:
        policy = ocean_torch.Drive(env, **args["policy"])

        num_lat = len(binding.JERK_LAT)
        lat_zero_idx = binding.JERK_LAT.index(0.0)
        brake_class = 0 * num_lat + lat_zero_idx
        accel_class = (len(binding.JERK_LONG) - 1) * num_lat + lat_zero_idx
        probs = torch.zeros(1, len(binding.JERK_LONG) * num_lat)
        probs[0, brake_class] = 0.5
        probs[0, accel_class] = 0.5

        normalized = policy.discrete_probs_to_continuous_mean(probs)
        # the continuous decode scales the brake half by -JERK_LONG[0], the accel half by JERK_LONG[-1]
        long_norm = float(normalized[0, 0])
        long_scale = -binding.JERK_LONG[0] if long_norm < 0 else binding.JERK_LONG[-1]
        expected_long_jerk = 0.5 * binding.JERK_LONG[0] + 0.5 * binding.JERK_LONG[-1]
        assert long_norm * long_scale == pytest.approx(expected_long_jerk, abs=1e-6)
        assert float(normalized[0, 1]) == pytest.approx(0.0, abs=1e-6)

        one_hot = torch.zeros_like(probs)
        one_hot[0, brake_class] = 1.0
        assert torch.allclose(
            policy.discrete_probs_to_continuous_mean(one_hot),
            policy.action_table[brake_class].unsqueeze(0),
            atol=1e-6,
        )
    finally:
        env.close()
