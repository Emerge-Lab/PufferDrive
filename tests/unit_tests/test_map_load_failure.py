"""A map bin that cannot be loaded fails env creation with an exception instead of crashing on first use."""

import pytest

from pufferlib.ocean.drive.drive import Drive

SOURCE_BIN = "pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin"
AGENTS_PER_ENV = 8


def _kwargs(map_dir, simulation_mode):
    return {
        "num_agents": AGENTS_PER_ENV,
        "min_agents_per_env": AGENTS_PER_ENV,
        "max_agents_per_env": AGENTS_PER_ENV,
        "num_maps": 1,
        "allow_map_subset": True,
        "map_dir": str(map_dir),
        "simulation_mode": simulation_mode,
        "scenario_length": 50,
        "resample_frequency": 0,
        "termination_mode": 0,
        "stagger_first_episode": False,
        "seed": 3,
    }


@pytest.mark.parametrize("simulation_mode", ["gigaflow", "replay"])
def test_truncated_map_bin_raises(tmp_path, simulation_mode):
    with open(SOURCE_BIN, "rb") as source:
        map_bytes = source.read()
    (tmp_path / "truncated.bin").write_bytes(map_bytes[: len(map_bytes) // 2])
    with pytest.raises(RuntimeError):
        Drive(**_kwargs(tmp_path, simulation_mode))
