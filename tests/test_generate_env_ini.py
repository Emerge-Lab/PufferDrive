"""Tests for generate_env_ini and generate_safe_eval_ini."""

import configparser
import os

from pufferlib.utils import generate_env_ini, generate_safe_eval_ini

BASE_INI = "pufferlib/config/ocean/drive.ini"


def _read_ini(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def test_generate_env_ini_overrides_existing_key():
    """Overriding a key that exists in the base INI works."""
    tmp = generate_env_ini({"max_agents_per_env": 1}, BASE_INI)
    try:
        config = _read_ini(tmp)
        assert config.get("env", "max_agents_per_env") == "1"
    finally:
        os.unlink(tmp)


def test_generate_env_ini_preserves_unmodified_keys():
    """Keys not in env_config remain at their base INI values."""
    base = _read_ini(BASE_INI)
    original_episode_length = base.get("env", "episode_length")

    tmp = generate_env_ini({"max_agents_per_env": 1}, BASE_INI)
    try:
        config = _read_ini(tmp)
        assert config.get("env", "episode_length") == original_episode_length
    finally:
        os.unlink(tmp)


def test_generate_env_ini_output_is_readable():
    """The returned temp file can be parsed by ConfigParser."""
    tmp = generate_env_ini({"max_agents_per_env": 5}, BASE_INI)
    try:
        config = _read_ini(tmp)
        assert config.has_section("env")
        assert config.get("env", "max_agents_per_env") == "5"
    finally:
        os.unlink(tmp)


def test_generate_safe_eval_ini_pins_reward_bounds():
    """generate_safe_eval_ini pins reward bounds min=max."""
    safe_config = {"collision": -1.5, "episode_length": 100}
    tmp = generate_safe_eval_ini(safe_config, BASE_INI)
    try:
        config = _read_ini(tmp)
        assert config.get("env", "reward_bound_collision_min") == "-1.5"
        assert config.get("env", "reward_bound_collision_max") == "-1.5"
    finally:
        os.unlink(tmp)


def test_generate_safe_eval_ini_overrides_env_keys():
    """generate_safe_eval_ini passes through env override keys."""
    safe_config = {"episode_length": 500, "num_maps": 10}
    tmp = generate_safe_eval_ini(safe_config, BASE_INI)
    try:
        config = _read_ini(tmp)
        assert config.get("env", "episode_length") == "500"
        assert config.get("env", "num_maps") == "10"
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    test_generate_env_ini_overrides_existing_key()
    test_generate_env_ini_preserves_unmodified_keys()
    test_generate_env_ini_output_is_readable()
    test_generate_safe_eval_ini_pins_reward_bounds()
    test_generate_safe_eval_ini_overrides_env_keys()
    print("All tests passed!")
