"""PuffeRL training of the SDC against a frozen, reward-conditioned co-player (Multiprocessing).

Each case checks that the co-player never changes, the learner does, the buffer holds exactly the learner
rows (each segment is one learner row's consecutive steps), metrics count only SDCs, and the saved
checkpoint is the learner. SDC and co-player rows get distinct reward-coef overrides, so every recv() also
proves the learner/co-player row split is right across workers, vecenv batches and map resamples.
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


def base_args(map_dir, resample_frequency, workers, workers_per_batch):
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    args["env"].update(ENV_OVERRIDES, map_dir=map_dir, num_maps=len(MAP_PATHS), resample_frequency=resample_frequency)
    args["policy"].update(POLICY_OVERRIDES)
    args["rnn"].update(RNN_OVERRIDES)
    args["vec"].update(
        backend="Multiprocessing",
        num_envs=workers,
        num_workers=workers,
        batch_size=workers_per_batch,
        seed=3,
        overwork=True,
    )
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


class RolloutRecorder:
    """Records, per vecenv batch and in recv order, what each learner row observed and the action sent for it."""

    def __init__(self, vecenv, trainer, column):
        self.vecenv = vecenv
        self.trainer = trainer
        self.column = column
        self.recv = vecenv.recv
        self.send = vecenv.send
        self.seen_layouts = set()
        self.sdc_rows_per_worker = set()
        self.batches = {}
        self.last_learner_rows = None
        self.last_batch = None
        vecenv.recv = self.checked_recv
        vecenv.send = self.recording_send

    def restore(self):
        self.vecenv.recv = self.recv
        self.vecenv.send = self.send

    def start_rollout(self):
        self.batches = {}

    def checked_recv(self):
        batch = self.recv()
        observations, terminals, truncations, env_id = batch[0], batch[2], batch[3], batch[5]
        check_rows_match_learner_split(
            self.vecenv, env_id, observations, self.column, self.seen_layouts, self.sdc_rows_per_worker
        )
        learner_rows = np.array(self.vecenv.learner_masks[env_id[0] : env_id[-1] + 1])
        columns = self.trainer.learner_obs_columns
        learner_obs = observations[learner_rows]
        if columns is not None:
            learner_obs = learner_obs[:, columns.cpu().numpy()]
        steps = self.batches.setdefault(int(env_id[0]), [])
        steps.append({"obs": learner_obs.copy(), "done": (terminals | truncations)[learner_rows].copy()})
        self.last_learner_rows = learner_rows
        self.last_batch = int(env_id[0])
        return batch

    def recording_send(self, actions):
        self.batches[self.last_batch][-1]["action"] = np.asarray(actions)[self.last_learner_rows].copy()
        return self.send(actions)

    def expected_buffer(self):
        """(obs, action, done) per segment and step: a batch's learner rows take the next segments, in order."""
        segment_obs, segment_actions, segment_dones = [], [], []
        for steps in self.batches.values():
            for row in range(len(steps[0]["obs"])):
                segment_obs.append(np.stack([step["obs"][row] for step in steps[:HORIZON]]))
                segment_actions.append(np.stack([step["action"][row] for step in steps[:HORIZON]]))
                segment_dones.append(np.stack([step["done"][row] for step in steps[:HORIZON]]))
        return np.stack(segment_obs), np.stack(segment_actions), np.stack(segment_dones)


@pytest.mark.parametrize(
    "learner_conditioned, coplayer_conditioned, learner_recurrent, coplayer_recurrent, coplayer_by_path, resample, "
    "workers, workers_per_batch, device",
    [
        (True, False, False, False, True, 0, 2, 2, "cpu"),
        (False, True, False, True, False, 0, 2, 2, "cpu"),
        (True, True, True, False, True, 0, 2, 2, "cpu"),
        (True, False, False, True, True, 45, 2, 2, "cpu"),
        (False, True, True, False, True, 0, 4, 2, "cpu"),
        pytest.param(
            False,
            True,
            False,
            False,
            True,
            0,
            4,
            2,
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        ),
    ],
    ids=[
        "learner-cond",
        "coplayer-cond-rnn",
        "both-cond-learner-rnn",
        "resample-maps",
        "two-vec-batches-rnn",
        "cuda-cpu-offload",
    ],
)
def test_ppo_against_frozen_coplayer(
    tmp_path,
    learner_conditioned,
    coplayer_conditioned,
    learner_recurrent,
    coplayer_recurrent,
    coplayer_by_path,
    resample,
    workers,
    workers_per_batch,
    device,
):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    for map_path in MAP_PATHS:
        os.symlink(map_path, map_dir / os.path.basename(map_path))
    args = base_args(str(map_dir), resample, workers, workers_per_batch)
    args["env"]["reward_conditioning"] = learner_conditioned or coplayer_conditioned
    args["env"]["reward_coef_overrides"] = {
        "sdc": {"collision": SDC_COLLISION},
        "others": {"collision": COPLAYER_COLLISION},
    }
    coplayer_path, coplayer_policy = write_coplayer_checkpoint(
        str(tmp_path / "coplayer_run"), args, coplayer_conditioned, coplayer_recurrent
    )

    torch.manual_seed(0)
    args["train"]["device"] = device
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
        minibatch_size=vecenv.num_agents * HORIZON,
        max_minibatch_size=vecenv.num_agents * HORIZON,
        total_timesteps=10_000_000,
        checkpoint_interval=10_000_000,
        data_dir=str(tmp_path / "learner_run"),
        adv_filter_enabled=False,
        render=False,
        cpu_offload=device == "cuda",
        coplayer_policy_path=coplayer_path if coplayer_by_path else None,
        coplayer_action_selection="sample",
    )
    train_config = dict(**args["train"], env="puffer_drive", eval={}, run_name="coplayer_test")

    trainer = PuffeRL(
        train_config, vecenv, learner, NullLogger(), coplayer_policy=None if coplayer_by_path else coplayer_policy
    )
    column = vecenv.driver_env.ego_features + REWARD_COEF_NAMES.index("collision")
    recorder = RolloutRecorder(vecenv, trainer, column)
    try:
        initial_learner_rows = int(np.count_nonzero(vecenv.learner_masks))
        assert trainer.segments == initial_learner_rows, "the buffer is sized from learner rows only"
        assert trainer.observations.shape[0] == initial_learner_rows
        assert trainer.observations.device.type == "cpu" if device == "cuda" else True
        coplayer_before = {k: v.clone() for k, v in trainer.coplayer.policy.state_dict().items()}
        learner_before = {k: v.clone() for k, v in trainer.uncompiled_policy.state_dict().items()}
        logged_counts = []
        logged_keys = set()
        closed_early = False
        for epoch in range(EPOCHS):
            recorder.start_rollout()
            learner_steps_before = trainer.learner_steps
            trainer.evaluate()
            masks = trainer.masks.cpu().numpy()
            assert trainer.learner_steps - learner_steps_before == masks.sum(), "every written learner step counts"
            # A segment ended early by a resample is unmasked up to that step, which is terminal, then masked.
            first_masked = np.where(masks.all(axis=1), HORIZON, np.argmin(masks, axis=1))
            for segment, step in enumerate(first_masked):
                assert not masks[segment, step:].any()
                assert step == HORIZON or trainer.terminals[segment, step] == 1.0
            closed_early |= bool((first_masked < HORIZON).any())
            logged_counts.extend(trainer.stats.get("n", []))
            logged_keys.update(trainer.stats)
            if resample == 0:
                expected_obs, expected_actions, expected_dones = recorder.expected_buffer()
                np.testing.assert_array_equal(trainer.observations.cpu().numpy(), expected_obs)
                np.testing.assert_array_equal(
                    trainer.actions.cpu().numpy().reshape(expected_actions.shape), expected_actions
                )
                np.testing.assert_array_equal(trainer.terminals.cpu().numpy(), expected_dones.astype(np.float32))
                assert trainer.masks.all(), "no learner step is masked without infractions"
                assert trainer.learner_steps == (epoch + 1) * trainer.segments * HORIZON
            trainer.last_log_time = 0.0
            trainer.train()
            if resample == 0:
                assert trainer.losses["masked_fraction"] == 0.0
        assert trainer.global_step >= EPOCHS * HORIZON * vecenv.num_agents

        for key, value in trainer.coplayer.policy.state_dict().items():
            assert torch.equal(value, coplayer_before[key]), f"co-player parameter {key} changed"
        assert any(
            not torch.equal(value, learner_before[key]) for key, value in trainer.uncompiled_policy.state_dict().items()
        ), "learner did not train"
        assert not any(p.requires_grad for p in trainer.coplayer.policy.parameters())

        assert logged_counts, "no episode metrics were logged"
        assert max(logged_counts) <= max(recorder.sdc_rows_per_worker), "metrics counted co-player agents"
        assert {"score", "collision_rate", "offroad_rate"} <= logged_keys
        assert closed_early == bool(resample), "only a map resample ends segments early"
        workers_seen = {first_row for first_row, _ in recorder.seen_layouts}
        if resample:
            assert len(recorder.seen_layouts) > len(workers_seen)
        else:
            assert len(recorder.seen_layouts) == len(workers_seen)

        model_path = trainer.save_checkpoint()
        saved = torch.load(model_path, map_location="cpu")
        learner_state = trainer.uncompiled_policy.state_dict()
        assert saved.keys() == learner_state.keys()
        assert all(torch.equal(saved[key], learner_state[key].cpu()) for key in saved)
    finally:
        recorder.restore()
        vecenv.close()
        trainer.utilization.stop()


def test_rollout_longer_than_one_round_of_learner_rows(tmp_path):
    """With a buffer of 1.5 rounds of learner rows, the second round only has room for half of them: those
    rows' next HORIZON steps fill the remaining segments and the rest act without being stored."""
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    for map_path in MAP_PATHS:
        os.symlink(map_path, map_dir / os.path.basename(map_path))
    args = base_args(str(map_dir), 0, 2, 2)
    args["env"]["reward_conditioning"] = True
    args["env"]["reward_coef_overrides"] = {
        "sdc": {"collision": SDC_COLLISION},
        "others": {"collision": COPLAYER_COLLISION},
    }
    coplayer_path, _ = write_coplayer_checkpoint(str(tmp_path / "coplayer_run"), args, True, False)
    torch.manual_seed(0)
    vecenv = load_env("puffer_drive", args)
    learner = load_policy(args, vecenv, "puffer_drive", reward_conditioning=True)
    learner_rows = vecenv.learner_row_count()
    segments = learner_rows + learner_rows // 2
    args["train"].update(
        compile=False,
        update_epochs=1,
        bptt_horizon=HORIZON,
        batch_size=segments * HORIZON,
        minibatch_size=segments * HORIZON,
        max_minibatch_size=segments * HORIZON,
        total_timesteps=10_000_000,
        checkpoint_interval=10_000_000,
        data_dir=str(tmp_path / "learner_run"),
        render=False,
        coplayer_policy_path=coplayer_path,
    )
    trainer = PuffeRL(
        dict(**args["train"], env="puffer_drive", eval={}, run_name="rounds"), vecenv, learner, NullLogger()
    )
    column = vecenv.driver_env.ego_features + REWARD_COEF_NAMES.index("collision")
    recorder = RolloutRecorder(vecenv, trainer, column)
    try:
        assert trainer.segments == segments
        recorder.start_rollout()
        trainer.evaluate()
        (steps,) = recorder.batches.values()
        assert len(steps) == 2 * HORIZON
        observations = trainer.observations.numpy()
        for segment in range(segments):
            row = segment if segment < learner_rows else segment - learner_rows
            first_step = 0 if segment < learner_rows else HORIZON
            expected = np.stack([steps[first_step + step]["obs"][row] for step in range(HORIZON)])
            np.testing.assert_array_equal(observations[segment], expected)
        assert trainer.learner_steps == segments * HORIZON
        assert trainer.stats["learner_rows_unbuffered"] == [learner_rows - learner_rows // 2] * HORIZON
        trainer.last_log_time = 0.0
        trainer.train()
    finally:
        recorder.restore()
        vecenv.close()
        trainer.utilization.stop()
