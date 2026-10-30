import os

import pytest

from pufferlib.wheel_demo.maps import (
    CARLA_MAP_PREFIX,
    CARLA_TOWNS,
    DEFAULT_MAP_DIR,
    MAP_SUFFIX,
    SPEED_ZONE_SECTION_TAG,
    TRAFFIC_PHASE_SECTION_TAG,
    map_sections,
    resolve_carla_map_files,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIPPED_MAP_DIR = os.path.join(REPO_ROOT, DEFAULT_MAP_DIR)
FILLER_BYTES = b"\x00" * 32


def write_map(map_dir, town, speed_zones=True, traffic_phases=True):
    blob = FILLER_BYTES
    if speed_zones:
        blob += SPEED_ZONE_SECTION_TAG + FILLER_BYTES
    if traffic_phases:
        blob += TRAFFIC_PHASE_SECTION_TAG + FILLER_BYTES
    path = os.path.join(map_dir, f"{CARLA_MAP_PREFIX}{town}{MAP_SUFFIX}")
    with open(path, "wb") as handle:
        handle.write(blob)
    return path


def write_full_map_set(map_dir, towns=CARLA_TOWNS, **kwargs):
    return [write_map(map_dir, town, **kwargs) for town in towns]


def test_shipped_carla_maps_resolve():
    paths = resolve_carla_map_files(map_dir=SHIPPED_MAP_DIR)
    assert len(paths) == len(CARLA_TOWNS)
    for town, path in zip(CARLA_TOWNS, paths):
        assert os.path.isabs(path)
        assert os.path.isfile(path)
        assert os.path.basename(path) == f"{CARLA_MAP_PREFIX}{town}{MAP_SUFFIX}"


def test_shipped_carla_maps_carry_both_sections():
    for town in CARLA_TOWNS:
        path = os.path.join(SHIPPED_MAP_DIR, f"{CARLA_MAP_PREFIX}{town}{MAP_SUFFIX}")
        assert map_sections(path) == {"speed_zones": True, "traffic_phases": True}


def test_map_sections_reports_each_tag_independently(tmp_path):
    map_dir = str(tmp_path)
    assert map_sections(write_map(map_dir, "Town01")) == {"speed_zones": True, "traffic_phases": True}
    assert map_sections(write_map(map_dir, "Town02", traffic_phases=False)) == {
        "speed_zones": True,
        "traffic_phases": False,
    }
    assert map_sections(write_map(map_dir, "Town03", speed_zones=False)) == {
        "speed_zones": False,
        "traffic_phases": True,
    }
    assert map_sections(write_map(map_dir, "Town04", speed_zones=False, traffic_phases=False)) == {
        "speed_zones": False,
        "traffic_phases": False,
    }


def test_missing_map_dir_raises():
    with pytest.raises(FileNotFoundError, match="map_dir does not exist"):
        resolve_carla_map_files(map_dir=os.path.join(REPO_ROOT, "definitely_not_a_map_dir"))


def test_missing_town_raises_and_names_it(tmp_path):
    map_dir = str(tmp_path)
    write_full_map_set(map_dir, towns=[town for town in CARLA_TOWNS if town != "Town05"])
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_carla_map_files(map_dir=map_dir)
    message = str(excinfo.value)
    assert "Town05" in message
    assert f"{CARLA_MAP_PREFIX}<Town>{MAP_SUFFIX}" in message


def test_directory_named_like_a_map_counts_as_missing(tmp_path):
    map_dir = str(tmp_path)
    write_full_map_set(map_dir, towns=[town for town in CARLA_TOWNS if town != "Town07"])
    os.mkdir(os.path.join(map_dir, f"{CARLA_MAP_PREFIX}Town07{MAP_SUFFIX}"))
    with pytest.raises(FileNotFoundError, match="Town07"):
        resolve_carla_map_files(map_dir=map_dir)


@pytest.mark.parametrize("speed_zones,traffic_phases", [(False, True), (True, False), (False, False)])
def test_old_build_without_sections_raises(tmp_path, speed_zones, traffic_phases):
    map_dir = str(tmp_path)
    write_full_map_set(map_dir)
    write_map(map_dir, "Town03", speed_zones=speed_zones, traffic_phases=traffic_phases)
    with pytest.raises(ValueError) as excinfo:
        resolve_carla_map_files(map_dir=map_dir)
    message = str(excinfo.value)
    assert "Town03" in message
    assert SPEED_ZONE_SECTION_TAG.decode() in message
    assert TRAFFIC_PHASE_SECTION_TAG.decode() in message


def test_section_check_can_be_waived(tmp_path):
    map_dir = str(tmp_path)
    write_full_map_set(map_dir, speed_zones=False, traffic_phases=False)
    paths = resolve_carla_map_files(map_dir=map_dir, require_sections=False)
    assert len(paths) == len(CARLA_TOWNS)


def test_missing_town_outranks_degraded_build(tmp_path):
    map_dir = str(tmp_path)
    write_full_map_set(map_dir, towns=[town for town in CARLA_TOWNS if town != "Town02"], speed_zones=False)
    with pytest.raises(FileNotFoundError):
        resolve_carla_map_files(map_dir=map_dir)


def test_town_order_is_preserved(tmp_path):
    map_dir = str(tmp_path)
    requested_towns = ("Town05", "Town01", "Town10HD")
    write_full_map_set(map_dir, towns=requested_towns)
    paths = resolve_carla_map_files(map_dir=map_dir, towns=requested_towns)
    assert [os.path.basename(path) for path in paths] == [
        f"{CARLA_MAP_PREFIX}{town}{MAP_SUFFIX}" for town in requested_towns
    ]


def test_relative_map_dir_returns_absolute_paths(tmp_path, monkeypatch):
    map_dir = str(tmp_path / "maps")
    os.mkdir(map_dir)
    write_full_map_set(map_dir, towns=("Town01",))
    monkeypatch.chdir(str(tmp_path))
    paths = resolve_carla_map_files(map_dir="maps", towns=("Town01",))
    assert len(paths) == 1
    assert os.path.isabs(paths[0])
    assert os.path.isfile(paths[0])


def test_empty_town_tuple_returns_nothing(tmp_path):
    assert resolve_carla_map_files(map_dir=str(tmp_path), towns=()) == []


def test_default_map_dir_points_at_the_carla_binaries():
    assert DEFAULT_MAP_DIR.endswith(os.path.join("drive", "binaries", "carla"))
    assert os.path.isdir(SHIPPED_MAP_DIR)
