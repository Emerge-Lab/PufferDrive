"""PPO update guards: a one-transition minibatch, a non-finite loss, and best-checkpoint scoring across log intervals."""

import os
import sys
from collections import defaultdict

import pytest
import torch

from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_PATH = os.path.join(
    REPO_ROOT, "pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin"
)
HORIZON = 16

ENV_OVERRIDES = dict(
    simulation_mode="replay",
    goal_source="gt",
    scenario_length=40,
    num_agents=16,
    min_agents_per_env=1,
    max_agents_per_env=16,
    action_type="discrete",
    init_step=0,
    init_step_spread=False,
    resample_frequency=0,
    num_maps=1,
    use_map_cache=0,
    reward_randomization=False,
)
POLICY_OVERRIDES = dict(
    ego_input_size=16,
    partner_input_size=16,
    lane_input_size=16,
    boundary_input_size=16,
    traffic_control_input_size=8,
    context_input_size=8,
    backbone_hidden_size=32,
    backbone_num_layers=1,
    actor_hidden_size=16,
    critic_hidden_size=16,
    action_type="discrete",
)


class NullLogger:
    run_id = "ppo_update_guards"

    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


@pytest.fixture
def make_trainer(tmp_path):
    built = []

    def build(**train_overrides):
        map_dir = tmp_path / "maps"
        map_dir.mkdir(exist_ok=True)
        if not (map_dir / os.path.basename(MAP_PATH)).exists():
            os.symlink(MAP_PATH, map_dir / os.path.basename(MAP_PATH))
        saved_argv = sys.argv
        sys.argv = [saved_argv[0]]
        try:
            args = load_config("puffer_drive")
        finally:
            sys.argv = saved_argv
        args["env"].update(ENV_OVERRIDES, map_dir=str(map_dir))
        args["policy"].update(POLICY_OVERRIDES)
        args["vec"].update(backend="Serial", num_envs=1, num_workers=1, batch_size=1, seed=3)
        args["train"]["device"] = "cpu"
        args["rnn_name"] = None
        torch.manual_seed(0)
        vecenv = load_env("puffer_drive", args)
        policy = load_policy(args, vecenv, "puffer_drive")
        args["train"].update(
            compile=False,
            use_rnn=False,
            anneal_lr=False,
            update_epochs=1,
            bptt_horizon=HORIZON,
            batch_size="auto",
            minibatch_size=vecenv.num_agents * HORIZON,
            max_minibatch_size=vecenv.num_agents * HORIZON,
            min_batch_size=None,
            total_timesteps=10_000_000,
            checkpoint_interval=10_000_000,
            data_dir=str(tmp_path / "run"),
            adv_filter_enabled=True,
            render=False,
        )
        args["train"].update(train_overrides)
        train_config = dict(**args["train"], env="puffer_drive", eval={}, run_name="ppo_update_guards")
        trainer = PuffeRL(train_config, vecenv, policy, NullLogger())
        built.append((trainer, vecenv))
        return trainer

    yield build
    for trainer, vecenv in built:
        vecenv.close()
        trainer.utilization.stop()


def test_lone_kept_transition_is_skipped(make_trainer):
    # On the first update ema_max is the batch max, so a threshold scale of 1 keeps only the max-|advantage| transition
    trainer = make_trainer(adv_filter_threshold_scale=1.0)
    weights_before = {key: value.clone() for key, value in trainer.uncompiled_policy.state_dict().items()}
    trainer.evaluate()
    trainer.last_log_time = 0.0
    trainer.train()

    assert round(trainer.losses["kept_fraction"] * trainer.segments * HORIZON) == 1
    for key, value in trainer.uncompiled_policy.state_dict().items():
        assert torch.equal(value, weights_before[key]), f"{key} changed without a usable minibatch"


def test_non_finite_loss_raises_before_backward(make_trainer):
    trainer = make_trainer(adv_filter_enabled=False)
    trainer.evaluate()
    trainer.rewards[0, HORIZON // 2] = float("nan")
    weights_before = {key: value.clone() for key, value in trainer.uncompiled_policy.state_dict().items()}

    with pytest.raises(FloatingPointError, match="Non-finite PPO loss"):
        trainer.train()
    for key, value in trainer.uncompiled_policy.state_dict().items():
        assert torch.equal(value, weights_before[key])


def test_best_checkpoint_uses_last_logged_score(make_trainer, tmp_path):
    trainer = make_trainer()
    trainer.evaluate()
    trainer.last_log_time = 0.0
    trainer.train()
    trainer.stats = defaultdict(list, {"score": [0.25, 0.75]})
    trainer.mean_and_log()
    trainer.stats = defaultdict(list, {"obs/max": [1.0]})
    trainer.mean_and_log()

    trainer.save_checkpoint()
    assert trainer.best_score == 0.5
    assert os.listdir(tmp_path / "run" / "best_models") == [f"best_trainer_state_{trainer.epoch:06d}.pt"]
