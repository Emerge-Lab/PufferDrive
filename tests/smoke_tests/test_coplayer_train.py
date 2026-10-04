"""PuffeRL training of the SDC against a frozen, reward-conditioned co-player (Multiprocessing, 2 workers).

Each case checks that the co-player never changes, the learner does, only learner rows reach the PPO
batch, metrics count only SDCs, and the saved checkpoint is the learner. SDC and co-player rows get
distinct reward-coef overrides, so every recv() also proves the learner/co-player row split is right
across workers and map resamples.
"""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

import pufferlib.ocean.torch as drive_torch
from pufferlib.ocean.drive.drive import REWARD_COEF_NAMES, Drive, reward_coef_to_normalized
from pufferlib.pufferl import COPLAYER_ENV_SETUP, PuffeRL, load_config, load_env, load_policy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_PATHS = [
    os.path.join(
        REPO_ROOT, "pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin"
    ),
    os.path.join(
        REPO_ROOT, "pufferlib/resources/drive/binaries/sdc_replay_test/nuplan__c2a99be0-5b3e-5e67-88e6-6fd0f22e5419.bin"
    ),
]
HORIZON = 16
EPOCHS = 4
SDC_COLLISION = 5.0
COPLAYER_COLLISION = -2.0

ENV_OVERRIDES = dict(
    simulation_mode="replay",
    goal_source="gt",
    scenario_length=40,
    num_agents=64,
    min_agents_per_env=1,
    max_agents_per_env=64,
    action_type="discrete",
    dynamics_model="jerk",
    init_step=0,
    init_step_spread=False,
    terminate_on_goal=0,
    termination_mode=0,
    collision_behavior="ignore",
    offroad_behavior="ignore",
    traffic_light_behavior="ignore",
    goal_reached_behavior="ignore",
    obs_slots_lane_n=16,
    obs_slots_boundary_n=16,
    obs_slots_partners_n=8,
    obs_slots_traffic_controls_n=4,
    obs_dropout_lane=0.0,
    obs_dropout_boundary=0.0,
    use_map_cache=0,
    reward_randomization=False,
    **COPLAYER_ENV_SETUP,
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
RNN_OVERRIDES = dict(input_size=32, hidden_size=32)


class NullLogger:
    run_id = "coplayer_test"

    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


def base_args(map_dir, resample_frequency):
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    args["env"].update(ENV_OVERRIDES, map_dir=map_dir, num_maps=len(MAP_PATHS), resample_frequency=resample_frequency)
    args["policy"].update(POLICY_OVERRIDES)
    args["rnn"].update(RNN_OVERRIDES)
    args["vec"].update(backend="Multiprocessing", num_envs=2, num_workers=2, batch_size=2, seed=3, overwork=True)
    args["train"]["device"] = "cpu"
    args["wandb"] = False
    args["eval"] = {}
    return args


def write_coplayer_checkpoint(run_dir, args, reward_conditioning, recurrent):
    """A random-init co-player saved the way pufferl does: <run>/config.yaml + <run>/models/model.pt."""
    env_config = {**args["env"], "reward_conditioning": reward_conditioning, "reward_coef_overrides": None}
    env = Drive(**{**env_config, "num_agents": 16})
    torch.manual_seed(11)
    policy = drive_torch.Drive(env, **args["policy"])
    if recurrent:
        policy = drive_torch.Recurrent(env, policy, **args["rnn"])
    env.close()
    os.makedirs(os.path.join(run_dir, "models"))
    model_path = os.path.join(run_dir, "models", "model.pt")
    torch.save(policy.state_dict(), model_path)
    config = {
        "package": "ocean",
        "policy_name": "Drive",
        "rnn_name": "Recurrent" if recurrent else None,
        "env": env_config,
        "policy": dict(args["policy"]),
        "rnn": dict(args["rnn"]),
    }
    with open(os.path.join(run_dir, "config.yaml"), "w") as config_file:
        yaml.safe_dump(config, config_file)
    return model_path, policy


def check_rows_match_learner_split(vecenv, env_id, observations, column, seen_layouts, sdc_rows_per_worker):
    rows = slice(env_id[0], env_id[-1] + 1)
    learner_rows = np.array(vecenv.learner_masks[rows])
    assert learner_rows.any() and not learner_rows.all()
    np.testing.assert_allclose(
        observations[learner_rows, column], reward_coef_to_normalized("collision", SDC_COLLISION)
    )
    np.testing.assert_allclose(
        observations[~learner_rows, column], reward_coef_to_normalized("collision", COPLAYER_COLLISION)
    )
    seen_layouts.add((env_id[0], tuple(np.flatnonzero(learner_rows))))
    sdc_rows_per_worker.update(learner_rows.reshape(-1, vecenv.num_agents // vecenv.num_workers).sum(axis=1))


@pytest.mark.parametrize(
    "learner_conditioned, coplayer_conditioned, learner_recurrent, coplayer_recurrent, coplayer_by_path, resample",
    [
        (True, False, False, False, True, 0),
        (False, True, False, True, False, 0),
        (True, True, True, False, True, 0),
        (True, False, False, True, True, 45),
    ],
    ids=["learner-cond", "coplayer-cond-rnn", "both-cond-learner-rnn", "resample-maps"],
)
def test_ppo_against_frozen_coplayer(
    tmp_path,
    learner_conditioned,
    coplayer_conditioned,
    learner_recurrent,
    coplayer_recurrent,
    coplayer_by_path,
    resample,
):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    for map_path in MAP_PATHS:
        os.symlink(map_path, map_dir / os.path.basename(map_path))
    args = base_args(str(map_dir), resample)
    args["env"]["reward_conditioning"] = learner_conditioned or coplayer_conditioned
    args["env"]["reward_coef_overrides"] = {
        "sdc": {"collision": SDC_COLLISION},
        "others": {"collision": COPLAYER_COLLISION},
    }
    coplayer_path, coplayer_policy = write_coplayer_checkpoint(
        str(tmp_path / "coplayer_run"), args, coplayer_conditioned, coplayer_recurrent
    )

    torch.manual_seed(0)
    vecenv = load_env("puffer_drive", args)
    args["rnn_name"] = "Recurrent" if learner_recurrent else None
    args["train"]["use_rnn"] = learner_recurrent
    learner = load_policy(args, vecenv, "puffer_drive", reward_conditioning=learner_conditioned)
    args["train"].update(
        compile=False,
        torch_deterministic=True,
        anneal_lr=False,
        learning_rate=3e-3,
        update_epochs=1,
        bptt_horizon=HORIZON,
        batch_size="auto",
        minibatch_size=vecenv.num_agents * HORIZON // 2,
        max_minibatch_size=vecenv.num_agents * HORIZON // 2,
        total_timesteps=10_000_000,
        checkpoint_interval=10_000_000,
        data_dir=str(tmp_path / "learner_run"),
        adv_filter_enabled=False,
        render=False,
        coplayer_policy_path=coplayer_path if coplayer_by_path else None,
        coplayer_action_selection="sample",
    )
    train_config = dict(**args["train"], env="puffer_drive", eval={}, run_name="coplayer_test")

    trainer = PuffeRL(
        train_config, vecenv, learner, NullLogger(), coplayer_policy=None if coplayer_by_path else coplayer_policy
    )
    try:
        column = vecenv.driver_env.ego_features + REWARD_COEF_NAMES.index("collision")
        seen_layouts = set()
        sdc_rows_per_worker = set()
        original_recv = vecenv.recv

        def checked_recv():
            batch = original_recv()
            check_rows_match_learner_split(vecenv, batch[5], batch[0], column, seen_layouts, sdc_rows_per_worker)
            return batch

        vecenv.recv = checked_recv
        coplayer_before = {k: v.clone() for k, v in trainer.coplayer.policy.state_dict().items()}
        learner_before = {k: v.clone() for k, v in trainer.uncompiled_policy.state_dict().items()}
        logged_counts = []
        logged_keys = set()
        for _ in range(EPOCHS):
            trainer.evaluate()
            logged_counts.extend(trainer.stats.get("n", []))
            logged_keys.update(trainer.stats)
            trainer.last_log_time = 0.0
            trainer.train()
            if resample == 0:
                learner_fraction = float(np.mean(vecenv.learner_masks))
                assert trainer.losses["masked_fraction"] == pytest.approx(1.0 - learner_fraction, abs=1e-6)

        for key, value in trainer.coplayer.policy.state_dict().items():
            assert torch.equal(value, coplayer_before[key]), f"co-player parameter {key} changed"
        assert any(
            not torch.equal(value, learner_before[key]) for key, value in trainer.uncompiled_policy.state_dict().items()
        ), "learner did not train"
        assert not any(p.requires_grad for p in trainer.coplayer.policy.parameters())

        assert logged_counts, "no episode metrics were logged"
        assert max(logged_counts) <= max(sdc_rows_per_worker), "metrics counted co-player agents"
        assert {"score", "collision_rate", "offroad_rate"} <= logged_keys
        workers_seen = {first_row for first_row, _ in seen_layouts}
        assert len(seen_layouts) > len(workers_seen) if resample else len(seen_layouts) == len(workers_seen)

        model_path = trainer.save_checkpoint()
        saved = torch.load(model_path, map_location="cpu")
        learner_state = trainer.uncompiled_policy.state_dict()
        assert saved.keys() == learner_state.keys()
        assert all(torch.equal(saved[key], learner_state[key]) for key in saved)
    finally:
        vecenv.recv = original_recv
        vecenv.close()
        trainer.utilization.stop()
