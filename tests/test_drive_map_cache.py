import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive


CARLA_MAP_DIR = "pufferlib/resources/drive/binaries/carla"


def test_gigaflow_map_cache_reuses_loaded_map():
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

        assert stats["count"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 1

        obs, _ = env.reset(seed=123)
        assert np.isfinite(obs).all()
        for _ in range(2):
            obs, rewards, _, _, _ = env.step(np.zeros_like(env.actions))
            assert np.isfinite(obs).all()
            assert np.isfinite(rewards).all()
    finally:
        env.close()
