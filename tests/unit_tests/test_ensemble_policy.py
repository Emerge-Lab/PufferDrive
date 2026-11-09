"""Evaluation-time ensembles: averaged member predictions, loading from an ensemble checkpoint, the checkpoint builder."""

import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

import pufferlib.pytorch as P
from pufferlib import ensemble


ACTION_TABLE = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
ACTION_TABLE_PHYSICAL = torch.tensor([[-4.0, -1.0], [0.0, 0.0], [2.0, 1.0]])
LONG_NEG_SCALE = 4.0
LONG_POS_SCALE = 2.0
BATCH_SIZE = 4
OBS_DIM = 2
RUN_CONFIG = {
    "run_name": "member",
    "wandb_project": "nightly",
    "wandb": True,
    "policy_name": "Drive",
    "rnn_name": None,
    "rnn": {"input_size": 128, "hidden_size": 128},
    "policy": {"actor_hidden_size": 8, "fused_slot_encoder": True},
    "env": {"dt": 0.1, "num_maps": 128},
}


class _TinyPolicy(nn.Module):
    """Discrete head on a continuous env with Drive's piecewise longitudinal scale; constant outputs, checkable by hand."""

    is_continuous = False

    def __init__(self, logits=(0.0, 0.0, 0.0), value=0.0):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(logits))
        self.value_bias = nn.Parameter(torch.tensor([value]))

    def forward_eval(self, observations, state=None):
        batch = observations.shape[0]
        return self.logit_bias.expand(batch, -1), self.value_bias.expand(batch, -1)

    def discrete_actions_to_continuous(self, actions):
        return ACTION_TABLE[actions.long()]

    def discrete_probs_to_continuous_mean(self, probs):
        mean_physical = probs @ ACTION_TABLE_PHYSICAL.to(probs.dtype)
        mean_long = mean_physical[..., 0]
        mean_long_norm = torch.where(mean_long < 0.0, mean_long / LONG_NEG_SCALE, mean_long / LONG_POS_SCALE)
        return torch.stack([mean_long_norm, mean_physical[..., 1]], dim=-1)


def _observations():
    return torch.zeros(BATCH_SIZE, OBS_DIM)


def _ensemble_checkpoint(*members):
    return {
        ensemble.ENSEMBLE_CHECKPOINT_KEY: {
            "format_version": ensemble.ENSEMBLE_FORMAT_VERSION,
            "member_checkpoints": [f"member_{idx}.pt" for idx in range(len(members))],
            "member_state_dicts": [member.state_dict() for member in members],
        }
    }


def _write_run_dir(root, name, state_dict, config_overrides=None):
    run_dir = root / name
    run_dir.mkdir()
    config = {**copy.deepcopy(RUN_CONFIG), "run_name": name, **(config_overrides or {})}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    torch.save(state_dict, run_dir / "final_model.pt")
    return str(run_dir / "final_model.pt")


def test_discrete_average_is_the_mean_of_member_probabilities_not_a_vote():
    confident = _TinyPolicy(logits=(4.0, 0.0, 0.0))
    hesitant = _TinyPolicy(logits=(0.0, 1.0, 0.5))
    policy = ensemble.EnsemblePolicy([confident, hesitant])

    logits, _ = policy.forward_eval(_observations())
    member_probs = torch.stack([torch.softmax(member.logit_bias, -1) for member in (confident, hesitant)])

    assert torch.allclose(torch.softmax(logits, -1), member_probs.mean(0).expand(BATCH_SIZE, -1), atol=1e-6)
    mode, _, _, _ = P.sample_logits(logits, action_selection=P.ACTION_SELECT_MODE, env_continuous=True, policy=policy)
    assert mode.tolist() == [0] * BATCH_SIZE


def test_mean_action_selection_averages_the_member_mean_actions_not_the_mixture_expectation():
    braking = _TinyPolicy(logits=(9.0, 0.0, 0.0))
    accelerating = _TinyPolicy(logits=(0.0, 0.0, 9.0))
    policy = ensemble.EnsemblePolicy([braking, accelerating])

    logits, _ = policy.forward_eval(_observations())
    _, _, _, mean_action = P.sample_logits(
        logits, action_selection=P.ACTION_SELECT_MEAN, env_continuous=True, policy=policy
    )
    mixture_expectation = braking.discrete_probs_to_continuous_mean(torch.softmax(logits, -1))

    # members execute long -1 and +1: their mean is 0; the mixture expectation sits at -1 / LONG_NEG_SCALE
    assert torch.allclose(mean_action, torch.zeros(BATCH_SIZE, 2), atol=1e-3)
    assert torch.allclose(mixture_expectation[:, 0], torch.full((BATCH_SIZE,), -0.25), atol=1e-3)


def test_mean_action_needs_the_member_probabilities_of_the_same_batch():
    policy = ensemble.EnsemblePolicy([_TinyPolicy(), _TinyPolicy()])
    uniform = torch.full((1, BATCH_SIZE, 3), 1.0 / 3.0)

    with pytest.raises(RuntimeError, match="forward_eval"):
        policy.discrete_probs_to_continuous_mean(uniform)
    policy.forward_eval(_observations())
    with pytest.raises(RuntimeError, match="same batch"):
        policy.discrete_probs_to_continuous_mean(uniform[:, : BATCH_SIZE - 1])
    assert policy.discrete_probs_to_continuous_mean(uniform).shape == (1, BATCH_SIZE, 2)


def test_values_are_averaged():
    policy = ensemble.EnsemblePolicy([_TinyPolicy(value=1.0), _TinyPolicy(value=3.0)])

    _, value = policy.forward_eval(_observations())

    assert value.shape == (BATCH_SIZE, 1)
    assert torch.allclose(value, torch.full((BATCH_SIZE, 1), 2.0))


def test_normal_average_matches_the_mixture_moments():
    members = [
        torch.distributions.Normal(torch.zeros(BATCH_SIZE, 2), torch.ones(BATCH_SIZE, 2)),
        torch.distributions.Normal(torch.full((BATCH_SIZE, 2), 2.0), torch.ones(BATCH_SIZE, 2)),
    ]

    mixture = ensemble.average_member_predictions(members)

    assert torch.allclose(mixture.loc, torch.ones(BATCH_SIZE, 2))
    assert torch.allclose(mixture.scale, torch.full((BATCH_SIZE, 2), 2.0**0.5))


def test_multi_discrete_outputs_average_per_component():
    member_a = (torch.tensor([[4.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0]]))
    member_b = (torch.tensor([[0.0, 4.0]]), torch.tensor([[0.0, 0.0, 0.0]]))

    mixture = ensemble.average_member_predictions([member_a, member_b])

    assert isinstance(mixture, tuple) and len(mixture) == 2
    assert torch.allclose(torch.softmax(mixture[0], -1), torch.tensor([[0.5, 0.5]]), atol=1e-6)
    assert torch.allclose(torch.softmax(mixture[1], -1), torch.full((1, 3), 1.0 / 3.0), atol=1e-6)


def test_load_policy_weights_keeps_a_trained_checkpoint_as_the_same_policy():
    trained = _TinyPolicy(logits=(1.0, 2.0, 3.0), value=5.0)
    checkpoint = {f"_orig_mod.module.{key}": value for key, value in trained.state_dict().items()}
    policy = _TinyPolicy()

    loaded = ensemble.load_policy_weights(policy, checkpoint)

    assert loaded is policy
    assert torch.equal(loaded.logit_bias, trained.logit_bias)
    assert torch.equal(loaded.value_bias, trained.value_bias)


def test_load_policy_weights_builds_an_ensemble_of_independent_copies():
    trained = [_TinyPolicy(logits=(1.0, 0.0, 0.0), value=1.0), _TinyPolicy(logits=(0.0, 1.0, 0.0), value=3.0)]

    loaded = ensemble.load_policy_weights(_TinyPolicy(), _ensemble_checkpoint(*trained))

    assert isinstance(loaded, ensemble.EnsemblePolicy)
    assert loaded.member_count == 2
    for member, source in zip(loaded.members, trained):
        assert torch.equal(member.logit_bias, source.logit_bias)
    with torch.no_grad():
        loaded.members[0].logit_bias.fill_(7.0)
    assert torch.equal(loaded.members[1].logit_bias, trained[1].logit_bias)
    _, value = loaded.forward_eval(_observations())
    assert torch.allclose(value, torch.full((BATCH_SIZE, 1), 2.0))


def test_ensemble_checkpoints_of_another_format_or_a_single_member_are_refused():
    single = _ensemble_checkpoint(_TinyPolicy())
    with pytest.raises(ValueError, match="at least"):
        ensemble.member_state_dicts(single)
    other_format = _ensemble_checkpoint(_TinyPolicy(), _TinyPolicy())
    other_format[ensemble.ENSEMBLE_CHECKPOINT_KEY]["format_version"] = 99
    with pytest.raises(ValueError, match="format"):
        ensemble.member_state_dicts(other_format)


def test_recurrent_members_are_refused():
    recurrent = _TinyPolicy()
    recurrent.lstm = nn.LSTM(OBS_DIM, OBS_DIM)

    with pytest.raises(ValueError, match="recurrent"):
        ensemble.EnsemblePolicy([recurrent, _TinyPolicy()])


def test_write_ensemble_checkpoint_round_trip(tmp_path):
    trained = [_TinyPolicy(logits=(1.0, 0.0, 0.0), value=1.0), _TinyPolicy(logits=(0.0, 2.0, 0.0), value=3.0)]
    member_paths = [
        _write_run_dir(tmp_path, "run_1000", trained[0].state_dict()),
        _write_run_dir(
            tmp_path,
            "run_2000",
            trained[1].state_dict(),
            {"policy": {**RUN_CONFIG["policy"], "fused_slot_encoder": False}},
        ),
    ]
    output_dir = tmp_path / "ensemble_dir"

    checkpoint_path, config_path = ensemble.write_ensemble_checkpoint(member_paths, str(output_dir), "my_ensemble")

    assert checkpoint_path == str(output_dir / "final_model.pt") and config_path == str(output_dir / "config.yaml")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert ensemble.is_ensemble_checkpoint(checkpoint)
    assert checkpoint[ensemble.ENSEMBLE_CHECKPOINT_KEY]["member_checkpoints"] == member_paths
    state_dicts = ensemble.member_state_dicts(checkpoint)
    assert len(state_dicts) == 2
    assert torch.equal(state_dicts[1]["logit_bias"], trained[1].logit_bias)
    config = yaml.safe_load(Path(config_path).read_text())
    assert config["run_name"] == "my_ensemble"
    assert config["wandb"] is False
    assert config["policy"] == RUN_CONFIG["policy"]
    assert config[ensemble.ENSEMBLE_CONFIG_KEY] == {
        "member_checkpoints": member_paths,
        "member_run_names": ["run_1000", "run_2000"],
    }
    loaded = ensemble.load_policy_weights(_TinyPolicy(), checkpoint)
    _, value = loaded.forward_eval(_observations())
    assert torch.allclose(value, torch.full((BATCH_SIZE, 1), 2.0))

    ensemble.write_ensemble_checkpoint(member_paths, str(output_dir), "my_ensemble")  # a rebuild replaces the ensemble


def test_write_ensemble_checkpoint_refuses_members_that_do_not_fit_one_architecture(tmp_path):
    first = _write_run_dir(tmp_path, "first", _TinyPolicy().state_dict())
    other_shape = _write_run_dir(tmp_path, "other_shape", _TinyPolicy(logits=(0.0, 0.0)).state_dict())
    other_policy = _write_run_dir(
        tmp_path,
        "other_policy",
        _TinyPolicy().state_dict(),
        {"policy": {**RUN_CONFIG["policy"], "actor_hidden_size": 16}},
    )
    other_env = _write_run_dir(tmp_path, "other_env", _TinyPolicy().state_dict(), {"env": {"dt": 0.2, "num_maps": 128}})
    recurrent = _write_run_dir(tmp_path, "recurrent", _TinyPolicy().state_dict(), {"rnn_name": "Recurrent"})
    output_dir = str(tmp_path / "out")

    with pytest.raises(ValueError, match="parameter shapes.*logit_bias"):
        ensemble.write_ensemble_checkpoint([first, other_shape], output_dir, "ens")
    with pytest.raises(ValueError, match="policy config"):
        ensemble.write_ensemble_checkpoint([first, other_policy], output_dir, "ens")
    with pytest.raises(ValueError, match=r"env config.*\['dt'\]"):
        ensemble.write_ensemble_checkpoint([first, other_env], output_dir, "ens")
    ensemble.write_ensemble_checkpoint([first, other_env], output_dir, "ens", allow_env_mismatch=True)
    assert yaml.safe_load((tmp_path / "out" / "config.yaml").read_text())["env"]["dt"] == 0.1
    with pytest.raises(ValueError, match="rnn_name"):
        ensemble.write_ensemble_checkpoint([first, recurrent], output_dir, "ens")
    with pytest.raises(ValueError, match="at least"):
        ensemble.write_ensemble_checkpoint([first], output_dir, "ens")
    with pytest.raises(ValueError, match="repeat"):
        ensemble.write_ensemble_checkpoint([first, first], output_dir, "ens")


def test_write_ensemble_checkpoint_never_overwrites_a_trained_checkpoint(tmp_path):
    first = _write_run_dir(tmp_path, "first", _TinyPolicy().state_dict())
    second = _write_run_dir(tmp_path, "second", _TinyPolicy().state_dict())

    with pytest.raises(FileExistsError, match="trained checkpoint"):
        ensemble.write_ensemble_checkpoint([first, second], str(tmp_path / "first"), "ens")
    assert torch.load(first, map_location="cpu").keys() == _TinyPolicy().state_dict().keys()
