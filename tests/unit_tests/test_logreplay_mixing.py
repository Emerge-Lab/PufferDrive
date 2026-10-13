"""Log-replay mixing for co-player training (vec.logreplay_fraction).

A log-replay env packs logreplay_scene_count one-row SDC scenes into its first rows and leaves the rest idle;
its episode metrics log under logreplay/. load_env's plan turns the last co-player envs of a vecenv into
log-replay envs so that log-replay scenes hold the requested share of learner rows.
"""

import os

import numpy as np
import pytest

import pufferlib
from pufferlib.ocean.drive.drive import ROW_KIND_IDLE, ROW_KIND_LOGREPLAY, ROW_KIND_SCENE, Drive
from pufferlib.pufferl import COPLAYER_ENV_SETUP, LOGREPLAY_ENV_SETUP, logreplay_mix_env_kwargs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUPLAN_MAP = os.path.join(
    REPO_ROOT, "pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin"
)
SDC_REPLAY_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib/resources/drive/binaries/sdc_replay_test")
SCENARIO_LENGTH = 40
ROWS_PER_ENV = 64
COPLAYER_KWARGS = dict(
    map_dir=NUPLAN_MAP,
    num_maps=1,
    simulation_mode="replay",
    goal_source="gt",
    scenario_length=SCENARIO_LENGTH,
    resample_frequency=0,
    num_agents=ROWS_PER_ENV,
    max_agents_per_env=64,
    min_agents_per_env=1,
    obs_slots_lane_n=8,
    obs_slots_boundary_n=8,
    obs_slots_partners_n=4,
    dynamics_model="jerk",
    action_type="discrete",
    reward_conditioning=True,
    termination_mode=0,
    seed=0,
    **COPLAYER_ENV_SETUP,
)
MULTIPROCESSING_VEC = dict(backend="Multiprocessing", num_envs=16, num_workers=8, batch_size=8)
LOGREPLAY_SCENES = 5


def test_logreplay_env_packs_sdc_scenes_and_idles_the_rest():
    coplayer_env = Drive(**COPLAYER_KWARGS)
    assert (coplayer_env.row_kinds() == ROW_KIND_SCENE).all()
    coplayer_env.close()

    env = Drive(**{**COPLAYER_KWARGS, **LOGREPLAY_ENV_SETUP, "logreplay_scene_count": LOGREPLAY_SCENES})
    kinds = env.row_kinds()
    assert (kinds[:LOGREPLAY_SCENES] == ROW_KIND_LOGREPLAY).all()
    assert (kinds[LOGREPLAY_SCENES:] == ROW_KIND_IDLE).all()
    assert env.learner_masks[:LOGREPLAY_SCENES].all() and not env.learner_masks[LOGREPLAY_SCENES:].any()
    assert env.log_agent_count == LOGREPLAY_SCENES
    env.action_space.seed(0)
    env.reset(seed=0)
    logs = []
    try:
        for _ in range(2 * SCENARIO_LENGTH):
            _, rewards, terminals, truncations, info = env.step(env.action_space.sample())
            assert not rewards[LOGREPLAY_SCENES:].any()
            assert not terminals[LOGREPLAY_SCENES:].any() and not truncations[LOGREPLAY_SCENES:].any()
            logs.extend(info)
    finally:
        env.close()
    assert logs and all(set(log) == {"logreplay"} for log in logs)
    assert {"score", "collision_rate", "n"} <= set(logs[0]["logreplay"])
    assert logs[0]["logreplay"]["n"] >= LOGREPLAY_SCENES


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"eval_mode": 1}, "only in training"),
        (COPLAYER_ENV_SETUP, "control_sdc_only"),
        ({"failure_branching": True}, "failure_branching"),
        ({"logreplay_scene_count": ROWS_PER_ENV + 1}, r"\[1, num_agents=64\]"),
        ({"logreplay_scene_count": -1}, r"\[1, num_agents=64\]"),
    ],
    ids=["eval", "coplayer-setup", "failure-branching", "too-many-scenes", "negative"],
)
def test_logreplay_env_rejects(overrides, message):
    kwargs = {**COPLAYER_KWARGS, **LOGREPLAY_ENV_SETUP, "logreplay_scene_count": LOGREPLAY_SCENES, **overrides}
    with pytest.raises(ValueError, match=message):
        Drive(**kwargs)


@pytest.mark.parametrize("logreplay_fraction", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_mix_plan_gives_log_replay_the_target_share(logreplay_fraction):
    probe = Drive(**COPLAYER_KWARGS)
    coplayer_learner_rows = probe.learner_row_count()
    probe.close()
    env_kwargs = logreplay_mix_env_kwargs(Drive, COPLAYER_KWARGS, MULTIPROCESSING_VEC, logreplay_fraction, None)
    scene_counts = [kwargs.get("logreplay_scene_count", 0) for kwargs in env_kwargs]
    logreplay_env_count = int(np.count_nonzero(scene_counts))
    num_envs = MULTIPROCESSING_VEC["num_envs"]
    assert len(env_kwargs) == num_envs
    assert 1 <= logreplay_env_count < num_envs
    assert all(scene_counts[-logreplay_env_count:]), "log-replay envs come last, after the driver env"
    assert env_kwargs[0] == COPLAYER_KWARGS
    assert all(
        kwargs[key] == value
        for kwargs in env_kwargs[-logreplay_env_count:]
        for key, value in LOGREPLAY_ENV_SETUP.items()
    )
    logreplay_rows = sum(scene_counts)
    realized = logreplay_rows / (logreplay_rows + (num_envs - logreplay_env_count) * coplayer_learner_rows)
    assert realized == pytest.approx(logreplay_fraction, abs=0.02)
    assert max(scene_counts) <= ROWS_PER_ENV
    if logreplay_env_count > 1:
        fewer = logreplay_env_count - 1
        fullest_share = fewer * ROWS_PER_ENV / (fewer * ROWS_PER_ENV + (num_envs - fewer) * coplayer_learner_rows)
        assert fullest_share < logreplay_fraction, "the plan must keep the most co-player envs"


def test_mix_plan_uses_the_log_replay_map_dir():
    env_kwargs = logreplay_mix_env_kwargs(Drive, COPLAYER_KWARGS, MULTIPROCESSING_VEC, 0.5, SDC_REPLAY_MAP_DIR)
    map_count = len([name for name in os.listdir(SDC_REPLAY_MAP_DIR) if name.endswith(".bin")])
    assert env_kwargs[-1]["map_dir"] == SDC_REPLAY_MAP_DIR and env_kwargs[-1]["num_maps"] == map_count
    assert env_kwargs[0]["map_dir"] == NUPLAN_MAP


@pytest.mark.parametrize(
    "env_overrides, vec_overrides, logreplay_fraction, error, message",
    [
        ({"failure_branching": True}, {}, 0.5, pufferlib.APIUsageError, "failure_branching"),
        ({"eval_mode": 1}, {}, 0.5, pufferlib.APIUsageError, "only in training"),
        (LOGREPLAY_ENV_SETUP, {}, 0.5, pufferlib.APIUsageError, "co-player training"),
        ({}, {"backend": "PufferEnv", "num_envs": 1}, 0.5, pufferlib.APIUsageError, "num_envs >= 2"),
        ({}, {"backend": "Serial", "batch_size": 8}, 0.5, pufferlib.APIUsageError, "Serial builds only"),
        ({}, {}, 1.0, pufferlib.APIUsageError, r"\[0, 1\)"),
        ({}, {}, -0.25, pufferlib.APIUsageError, r"\[0, 1\)"),
        ({}, {}, float("nan"), pufferlib.APIUsageError, r"\[0, 1\)"),
        ({}, {}, "0.25", pufferlib.APIUsageError, r"\[0, 1\)"),
    ],
    ids=["failure-branching", "eval", "not-coplayer", "single-env", "serial-batch", "one", "negative", "nan", "string"],
)
def test_mix_plan_rejects(env_overrides, vec_overrides, logreplay_fraction, error, message):
    with pytest.raises(error, match=message):
        logreplay_mix_env_kwargs(
            Drive,
            {**COPLAYER_KWARGS, **env_overrides},
            {**MULTIPROCESSING_VEC, **vec_overrides},
            logreplay_fraction,
            None,
        )


def test_mix_plan_rejects_a_missing_log_replay_map_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        logreplay_mix_env_kwargs(Drive, COPLAYER_KWARGS, MULTIPROCESSING_VEC, 0.5, str(tmp_path / "missing"))
