"""The shadow env's eval settings default to COSIM_EVAL_DEFAULTS; the environment variable of the same name overrides."""

import pytest

from opengigaflow.ocean.cosim.carla import shadow_ego


CHECKPOINT_ENV_CONFIG = {"base_max_speed_mps": 20.0, "conditioning_speed_scale": 2.0}
NARROW_SPEED_RANGE_ENV_CONFIG = {"base_max_speed_mps": 20.0, "conditioning_speed_scale": 1.2}


@pytest.fixture(autouse=True)
def _no_inherited_settings(monkeypatch):
    for name in shadow_ego.COSIM_EVAL_DEFAULTS:
        monkeypatch.delenv(name, raising=False)


def test_unset_settings_take_the_defaults():
    defaults = shadow_ego.COSIM_EVAL_DEFAULTS
    assert shadow_ego.cosim_max_speed_mps(CHECKPOINT_ENV_CONFIG) == float(defaults["COSIM_MAX_SPEED_MPS"])
    assert shadow_ego.cosim_obs_slots_partners_n() == int(defaults["COSIM_OBS_SLOTS_PARTNERS_N"])


def test_empty_settings_keep_the_checkpoint_values(monkeypatch):
    monkeypatch.setenv("COSIM_MAX_SPEED_MPS", "")
    monkeypatch.setenv("COSIM_OBS_SLOTS_PARTNERS_N", "")
    assert shadow_ego.cosim_max_speed_mps(CHECKPOINT_ENV_CONFIG) is None
    assert shadow_ego.cosim_obs_slots_partners_n() is None


def test_the_environment_overrides_a_default(monkeypatch):
    monkeypatch.setenv("COSIM_MAX_SPEED_MPS", "15")
    monkeypatch.setenv("COSIM_OBS_SLOTS_PARTNERS_N", "20")
    assert shadow_ego.cosim_max_speed_mps(CHECKPOINT_ENV_CONFIG) == 15.0
    assert shadow_ego.cosim_obs_slots_partners_n() == 20


def test_a_checkpoint_that_never_trained_the_default_speed_cap_is_refused():
    with pytest.raises(ValueError, match="trained C_vel range"):
        shadow_ego.cosim_max_speed_mps(NARROW_SPEED_RANGE_ENV_CONFIG)


def test_the_town_maps_default_to_the_tracked_towns():
    assert shadow_ego.cosim_town_bin_dir() == shadow_ego.COSIM_EVAL_DEFAULTS["COSIM_TOWN_BIN_DIR"]


def test_the_environment_selects_another_town_map_folder(monkeypatch, tmp_path):
    (tmp_path / "opendrive__Town01.bin").touch()
    monkeypatch.setenv("COSIM_TOWN_BIN_DIR", str(tmp_path))
    assert shadow_ego.cosim_town_bin_dir() == str(tmp_path.resolve())


@pytest.mark.parametrize("town_bin_dir", ["", "missing_folder", "folder_without_towns"])
def test_a_town_bin_dir_without_town_maps_is_refused(monkeypatch, tmp_path, town_bin_dir):
    (tmp_path / "folder_without_towns").mkdir()
    monkeypatch.setenv("COSIM_TOWN_BIN_DIR", town_bin_dir and str(tmp_path / town_bin_dir))
    with pytest.raises(FileNotFoundError, match="town map"):
        shadow_ego.cosim_town_bin_dir()
