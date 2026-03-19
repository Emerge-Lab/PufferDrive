import configparser
import os
import sys
import tempfile

import gym

import pufferlib
import pufferlib.utils


def test_suppress():
    with pufferlib.utils.Suppress():
        gym.make("Breakout-v4")
        print("stdout (you should not see this)", file=sys.stdout)
        print("stderr (you should not see this)", file=sys.stderr)


def _make_base_ini():
    """Create a minimal base INI file for testing."""
    config = configparser.ConfigParser()
    config.add_section("env")
    config.set("env", "reward_randomization", "0")
    config.set("env", "reward_conditioning", "0")
    config.set("env", "resample_frequency", "300")
    config.set("env", "episode_length", "300")
    config.set("env", "num_agents", "1024")
    config.set("env", "min_goal_distance", "0.5")
    config.set("env", "max_goal_distance", "60.0")
    # Add some reward bounds
    config.set("env", "reward_bound_collision_min", "-0.1")
    config.set("env", "reward_bound_collision_max", "-0.1")
    config.set("env", "reward_bound_offroad_min", "-0.1")
    config.set("env", "reward_bound_offroad_max", "-0.1")
    config.set("env", "reward_bound_lane_align_min", "0.002")
    config.set("env", "reward_bound_lane_align_max", "0.0025")
    config.set("env", "reward_bound_velocity_min", "0.0")
    config.set("env", "reward_bound_velocity_max", "0.005")

    fd, path = tempfile.mkstemp(suffix=".ini")
    with os.fdopen(fd, "w") as f:
        config.write(f)
    return path


def test_generate_safe_eval_ini_pins_reward_bounds():
    """Reward bounds should be pinned: min == max == safe value."""
    base_ini = _make_base_ini()
    try:
        safe_cfg = {
            "collision": -3.0,
            "offroad": -3.0,
            "lane_align": 0.025,
            "velocity": 0.005,
        }
        out_path = pufferlib.utils.generate_safe_eval_ini(safe_cfg, base_ini_path=base_ini)
        try:
            result = configparser.ConfigParser()
            result.read(out_path)

            for key, val in safe_cfg.items():
                assert result.get("env", f"reward_bound_{key}_min") == str(val), f"reward_bound_{key}_min not pinned"
                assert result.get("env", f"reward_bound_{key}_max") == str(val), f"reward_bound_{key}_max not pinned"
        finally:
            os.remove(out_path)
    finally:
        os.remove(base_ini)


def test_generate_safe_eval_ini_sets_flags():
    """reward_randomization=1, reward_conditioning=1, resample_frequency=0."""
    base_ini = _make_base_ini()
    try:
        out_path = pufferlib.utils.generate_safe_eval_ini({}, base_ini_path=base_ini)
        try:
            result = configparser.ConfigParser()
            result.read(out_path)

            assert result.get("env", "reward_randomization") == "1"
            assert result.get("env", "reward_conditioning") == "1"
            assert result.get("env", "resample_frequency") == "0"
        finally:
            os.remove(out_path)
    finally:
        os.remove(base_ini)


def test_generate_safe_eval_ini_env_overrides():
    """episode_length, num_agents, goal distances should be overridden."""
    base_ini = _make_base_ini()
    try:
        safe_cfg = {
            "episode_length": 1000,
            "num_agents": 64,
            "min_goal_distance": 0.5,
            "max_goal_distance": 1000.0,
            "map_dir": "resources/drive/binaries/carla_3D",
            "num_maps": 8,
        }
        out_path = pufferlib.utils.generate_safe_eval_ini(safe_cfg, base_ini_path=base_ini)
        try:
            result = configparser.ConfigParser()
            result.read(out_path)

            assert result.get("env", "episode_length") == "1000"
            assert result.get("env", "num_agents") == "64"
            assert result.get("env", "min_goal_distance") == "0.5"
            assert result.get("env", "max_goal_distance") == "1000.0"
            assert result.get("env", "map_dir") == "resources/drive/binaries/carla_3D"
            assert result.get("env", "num_maps") == "8"
        finally:
            os.remove(out_path)
    finally:
        os.remove(base_ini)


def test_generate_safe_eval_ini_missing_file():
    """Should raise FileNotFoundError for nonexistent base INI."""
    import pytest

    with pytest.raises(FileNotFoundError):
        pufferlib.utils.generate_safe_eval_ini({}, base_ini_path="/nonexistent/path.ini")


def test_generate_safe_eval_ini_missing_env_section():
    """Should raise ValueError if base INI has no [env] section."""
    import pytest

    config = configparser.ConfigParser()
    config.add_section("train")
    config.set("train", "seed", "42")
    fd, path = tempfile.mkstemp(suffix=".ini")
    with os.fdopen(fd, "w") as f:
        config.write(f)

    try:
        with pytest.raises(ValueError, match="missing required .env. section"):
            pufferlib.utils.generate_safe_eval_ini({}, base_ini_path=path)
    finally:
        os.remove(path)


if __name__ == "__main__":
    test_suppress()
    test_generate_safe_eval_ini_pins_reward_bounds()
    test_generate_safe_eval_ini_sets_flags()
    test_generate_safe_eval_ini_env_overrides()
    test_generate_safe_eval_ini_missing_file()
    test_generate_safe_eval_ini_missing_env_section()
    print("All tests passed!")
