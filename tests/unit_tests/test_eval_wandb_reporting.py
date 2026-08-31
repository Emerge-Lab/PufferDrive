"""Unit tests for reporting standalone eval results onto the training wandb run.

A post-training `puffer eval` runs in its own process, so to land on the run that
produced the checkpoint it must resolve that run's identity from the checkpoint's
config.yaml rather than from its own config defaults. It also must not disturb what
training already published: no config overwrite, no model re-upload, and metrics are
logged step-free so wandb appends them after training's step history. All of that is
asserted here against a stub wandb module, with no env, C-sim, GPU, or network.
"""

import sys
import types
from collections import defaultdict

import pytest
import yaml

import pufferlib
from pufferlib import pufferl
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark


RUN_IDENTITY = {
    "run_name": "k_exp_0002_1000",
    "wandb_project": "nightly-multi-long",
    "wandb_group": "emerge_",
}


def _write_checkpoint_config(tmp_path, **overrides):
    config = {**RUN_IDENTITY, "policy_name": "Drive", "rnn_name": None}
    config.update(overrides)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


def _benchmark_result(**metrics_mean):
    return {
        "episodes": [],
        "summary": {"num_scenarios": 1000, "num_episodes": 998, "metrics_mean": metrics_mean},
    }


class _StubRun:
    def __init__(self):
        self.summary = {}
        self.id = "stub_run_id"

    def log_artifact(self, artifact):
        raise AssertionError("eval reporting must not upload a model artifact")


@pytest.fixture
def stub_wandb(monkeypatch):
    """Install a stub `wandb` module and record how the logger drives it."""
    stub = types.ModuleType("wandb")
    stub.run = _StubRun()
    stub.init_kwargs = None
    stub.logged = []
    stub.finished = False

    def init(**kwargs):
        stub.init_kwargs = kwargs

    def log(metrics, step=None):
        stub.logged.append((metrics, step))

    stub.init = init
    stub.log = log
    stub.Settings = lambda **kwargs: kwargs
    stub.finish = lambda: setattr(stub, "finished", True)
    monkeypatch.setitem(sys.modules, "wandb", stub)
    return stub


@pytest.fixture
def eval_args():
    """Args shaped like load_config's return: a defaultdict whose optional top-level
    keys (no_model_upload) are absent and materialize on access."""
    return defaultdict(
        dict,
        {
            "wandb": True,
            "run_name": "default_run",
            "wandb_project": "pufferlib",
            "wandb_group": "debug",
            "tag": None,
        },
    )


def test_identity_comes_from_checkpoint_not_eval_config(stub_wandb, eval_args):
    pufferl.report_eval_to_wandb(eval_args, {"carla": _benchmark_result(collision_rate=0.11)}, RUN_IDENTITY, "eval")

    assert stub_wandb.init_kwargs["id"] == RUN_IDENTITY["run_name"]
    assert stub_wandb.init_kwargs["name"] == RUN_IDENTITY["run_name"]
    assert stub_wandb.init_kwargs["project"] == RUN_IDENTITY["wandb_project"]
    assert stub_wandb.init_kwargs["group"] == RUN_IDENTITY["wandb_group"]
    # A missing target run must fail loudly rather than open a stray eval-only run.
    assert stub_wandb.init_kwargs["resume"] == "must"


def test_training_config_and_artifacts_are_left_alone(stub_wandb, eval_args):
    pufferl.report_eval_to_wandb(eval_args, {"carla": _benchmark_result(collision_rate=0.11)}, RUN_IDENTITY, "eval")

    assert stub_wandb.init_kwargs["config"] is None
    assert stub_wandb.finished is True
    # The caller's args are reused after eval returns, so they must not be rewritten.
    assert eval_args["run_name"] == "default_run"
    assert eval_args["wandb_project"] == "pufferlib"


def test_metrics_are_logged_step_free_under_the_final_eval_prefix(stub_wandb, eval_args):
    results = {
        "carla": _benchmark_result(collision_rate=0.11),
        "nuplan_single": _benchmark_result(collision_rate=0.04, offroad_rate=0.01),
    }
    pufferl.report_eval_to_wandb(eval_args, results, RUN_IDENTITY, "eval")

    expected_metrics = {
        "final_eval_carla/num_scenarios": 1000,
        "final_eval_carla/num_episodes": 998,
        "final_eval_carla/collision_rate": 0.11,
        "final_eval_nuplan_single/num_scenarios": 1000,
        "final_eval_nuplan_single/num_episodes": 998,
        "final_eval_nuplan_single/collision_rate": 0.04,
        "final_eval_nuplan_single/offroad_rate": 0.01,
    }
    assert stub_wandb.logged == [(expected_metrics, None)]


def test_no_metrics_opens_no_wandb_session(stub_wandb, eval_args):
    pufferl.report_eval_to_wandb(eval_args, {"carla": {"episodes": [], "summary": None}}, RUN_IDENTITY, "eval")

    assert stub_wandb.init_kwargs is None
    assert stub_wandb.logged == []


@pytest.mark.parametrize("missing_key", sorted(drive_benchmark.CHECKPOINT_RUN_IDENTITY_KEYS))
def test_incomplete_checkpoint_identity_is_rejected(tmp_path, missing_key):
    config = {key: value for key, value in RUN_IDENTITY.items() if key != missing_key}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(pufferlib.APIUsageError, match=missing_key):
        drive_benchmark.load_checkpoint_run_identity(str(config_path))


def test_blank_checkpoint_identity_is_rejected(tmp_path):
    config_path = _write_checkpoint_config(tmp_path, wandb_group="   ")

    with pytest.raises(pufferlib.APIUsageError, match="wandb_group"):
        drive_benchmark.load_checkpoint_run_identity(config_path)


def test_summarize_benchmark_metrics_skips_benchmarks_without_a_summary():
    results = {
        "carla": _benchmark_result(collision_rate=0.11),
        "nuplan_single": {"episodes": [], "summary": None},
    }

    metrics = drive_benchmark.summarize_benchmark_metrics(results, "eval_")

    assert set(metrics) == {"eval_carla/num_scenarios", "eval_carla/num_episodes", "eval_carla/collision_rate"}
    assert drive_benchmark.summarize_benchmark_metrics({}, "eval_") == {}
