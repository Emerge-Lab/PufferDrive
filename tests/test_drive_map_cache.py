import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive


CARLA_MAP_DIR = "pufferlib/resources/drive/binaries/carla"
REPLAY_MAP_DIR = "pufferlib/resources/drive"


def test_gigaflow_map_cache_shares_immutable_map_data():
    try:
        env = Drive(
            num_agents=8,
            min_agents_per_env=4,
            max_agents_per_env=4,
            num_maps=1,
            maps=2,
            map_dir=CARLA_MAP_DIR,
            simulation_mode="gigaflow",
            scenario_length=8,
            resample_frequency=0,
            render_mode=None,
        )
    except FileNotFoundError:
        pytest.skip("CARLA Drive map binaries are not available in this checkout")

    try:
        stats = binding.map_cache_stats(env._map_cache)
        debug = binding.vec_map_debug(env.c_envs)

        assert len(debug) == 2
        assert stats["count"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 1

        assert len({item["shared_map_ptr"] for item in debug}) == 1
        assert len({item["road_elements_ptr"] for item in debug}) == 1
        assert len({item["grid_map_ptr"] for item in debug}) == 1
        assert len({item["traffic_elements_ptr"] for item in debug}) == len(debug)
        assert all(item["owns_map_data"] == 0 for item in debug)
        assert all(item["owns_traffic_data"] == 1 for item in debug)

        obs, _ = env.reset(seed=123)
        assert np.isfinite(obs).all()
        for _ in range(2):
            obs, rewards, _, _, _ = env.step(np.zeros_like(env.actions))
            assert np.isfinite(obs).all()
            assert np.isfinite(rewards).all()
    finally:
        env.close()


def test_replay_does_not_use_shared_map_cache():
    try:
        env = Drive(
            num_agents=4,
            min_agents_per_env=1,
            max_agents_per_env=4,
            num_maps=1,
            map_dir=REPLAY_MAP_DIR,
            simulation_mode="replay",
            control_mode="control_sdc_only",
            scenario_length=8,
            resample_frequency=0,
            render_mode=None,
        )
    except (FileNotFoundError, ValueError):
        pytest.skip("Replay Drive map binary is not available in this checkout")

    try:
        debug = binding.vec_map_debug(env.c_envs)

        assert debug
        assert all(item["shared_map_ptr"] == 0 for item in debug)
        assert all(item["owns_map_data"] == 1 for item in debug)
    finally:
        env.close()
