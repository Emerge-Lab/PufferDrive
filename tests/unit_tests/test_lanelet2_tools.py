"""Lanelet2 conversion and preview tests.

- Verify projection, geometry, connectivity, crop, and validation.
- Exercise the documented conversion and PNG commands end to end.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data_utils.lanelet2_conversion import (
    ROAD_EDGE_BOUNDARY,
    ROAD_LINE_SOLID_SINGLE_WHITE,
    _connect_nearby_endpoints,
    _speed_limit_mps,
    build_map,
    convert_map,
)
from data_utils.lanelet2_geometry import clip_polyline, detect_utm_epsg, project_epsg_array
from data_utils.lanelet2_validation import validate_bin
from data_utils.mirror_map_bin import read_bin, write_bin


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lanelet2_tiny.osm"


def test_projection_is_detected_from_map_coordinates():
    """Detect the expected UTM zone and projected Tokyo coordinate."""

    epsg = detect_utm_epsg([35.6], [139.7])
    eastings, northings = project_epsg_array([35.6], [139.7], epsg)

    assert epsg == 32654
    assert eastings[0] == pytest.approx(382241.991, abs=0.02)
    assert northings[0] == pytest.approx(3940361.836, abs=0.02)


def test_projection_detects_southern_hemisphere():
    """Select a southern-hemisphere UTM CRS when required."""

    assert detect_utm_epsg([-33.9], [151.2]) == 32756


def test_projection_rejects_missing_coordinates():
    """Reject projection requests without coordinate pairs."""

    with pytest.raises(ValueError, match="latitude/longitude"):
        detect_utm_epsg([], [])


def test_conversion_uses_an_automatic_local_origin():
    """Shift converted geometry to an automatically selected origin."""

    map_data, projection = convert_map(FIXTURE)
    xs = [x for road in map_data["roads"] for x in road["x"]]
    ys = [y for road in map_data["roads"] for y in road["y"]]

    assert projection.epsg == 32654
    assert min(xs) == pytest.approx(0.0, abs=1e-6)
    assert min(ys) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "raw, expected_mps",
    [("36", 10.0), ("36 km/h", 10.0), ("10 m/s", 10.0), ("22.3694 mph", 10.0)],
)
def test_speed_limit_units_are_normalized(raw, expected_mps):
    """Normalize supported speed-limit units to metres per second."""

    assert _speed_limit_mps({"speed_limit": raw}) == pytest.approx(expected_mps, abs=1e-4)


def test_missing_speed_limit_uses_standard_road_default():
    """Use 50 km/h when a lane has no speed-limit tag."""

    assert _speed_limit_mps({}) == pytest.approx(50.0 / 3.6)


def test_conversion_reindexes_roads_and_links_lanelets(tmp_path):
    """Reindex emitted roads and preserve directed lane connectivity."""

    map_data = build_map(FIXTURE)
    output = tmp_path / "map.bin"
    write_bin(map_data, output)

    decoded = read_bin(output)
    lanes = [road for road in decoded["roads"] if road["type"] == 2]

    assert [road["id"] for road in decoded["roads"]] == list(range(len(decoded["roads"])))
    assert len(lanes) == 2
    assert lanes[0]["exit_lanes"] == (1,)
    assert lanes[1]["entry_lanes"] == (0,)
    assert lanes[1]["exit_lanes"] == ()
    assert lanes[0]["speed_limit"] == pytest.approx(40.0 / 3.6)
    assert validate_bin(output)["id_equals_index"] is True


def test_centerline_is_between_boundaries_and_has_finite_headings():
    """Create a forward centerline with finite headings."""

    map_data = build_map(FIXTURE)
    first_lane = map_data["roads"][0]

    assert first_lane["y"][-1] > first_lane["y"][0]
    assert all(math.isfinite(value) for value in first_lane["headings"])


def test_fixture_is_a_short_straight_connected_segment():
    """Keep the public fixture small, straight, and connected."""

    roads = build_map(FIXTURE)["roads"]
    xs = [value for road in roads for value in road["x"]]
    ys = [value for road in roads for value in road["y"]]
    lanes = [road for road in roads if road["type"] == 2]

    assert max(xs) - min(xs) < 10.0
    assert max(ys) - min(ys) < 26.0
    assert sum(lane["length"] for lane in lanes) == pytest.approx(25.09, abs=0.05)
    assert max(abs(heading - math.radians(75)) for lane in lanes for heading in lane["headings"]) < 0.01


def test_virtual_boundaries_can_be_rendered_as_road_edges(tmp_path):
    """Allow virtual boundaries to be emitted as road edges."""

    virtual_fixture = tmp_path / "virtual.osm"
    virtual_fixture.write_text(
        FIXTURE.read_text().replace('v="line_thin"', 'v="virtual"', 1),
        encoding="utf-8",
    )
    default_map = build_map(virtual_fixture)
    edge_map = build_map(virtual_fixture, virtual_as_edge=True)
    changed = [
        (default["type"], edge["type"])
        for default, edge in zip(default_map["roads"], edge_map["roads"])
        if default["type"] != edge["type"]
    ]

    assert changed == [(ROAD_LINE_SOLID_SINGLE_WHITE, ROAD_EDGE_BOUNDARY)]


def test_link_tolerance_connects_nearby_lane_endpoints():
    """Connect close endpoints when a positive tolerance is configured."""

    roads = [
        {"x": (0.0, 1.0), "y": (0.0, 0.0), "entry_lanes": [], "exit_lanes": []},
        {"x": (1.05, 2.0), "y": (0.0, 0.0), "entry_lanes": [], "exit_lanes": []},
    ]

    _connect_nearby_endpoints(roads, tolerance_m=0.1)

    assert roads[0]["exit_lanes"] == [1]
    assert roads[1]["entry_lanes"] == [0]


def test_crop_rejects_an_empty_selection():
    """Reject crop windows containing no drivable lanelet."""

    with pytest.raises(ValueError, match="no drivable Lanelet2"):
        build_map(FIXTURE, crop=(100.0, 100.0, 101.0, 101.0))


def test_crop_clips_geometry_at_the_requested_bounds():
    """Clip polyline geometry exactly at the crop boundaries."""

    points = [(-2.0, 0.5), (0.5, 0.5), (2.0, 0.5)]

    clipped = clip_polyline(points, (0.0, 0.0, 1.0, 1.0))

    assert clipped == [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]


def test_validator_rejects_lane_links_outside_the_lane_table(tmp_path):
    """Reject directed links that reference a non-lane record."""

    map_data = build_map(FIXTURE)
    map_data["roads"][0]["exit_lanes"] = [999]
    output = tmp_path / "invalid-link.bin"
    write_bin(map_data, output)

    with pytest.raises(ValueError, match="references non-lane road"):
        validate_bin(output)


def test_renderer_writes_png_from_the_binary(tmp_path):
    """Render a PNG and report expected element counts."""

    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")
    from data_utils.visualize_map_bin import _element_summary, write_png

    binary = tmp_path / "map.bin"
    png = tmp_path / "map.png"
    write_bin(build_map(FIXTURE), binary)

    result = write_png(binary, png, width_px=320, height_px=240)

    assert result["width"] == 320
    assert result["height"] == 240
    assert _element_summary(read_bin(binary)["roads"]) == {
        "Road lane": {"segments": 2, "groups": 1},
        "Road line": {"segments": 4, "groups": 2},
        "Road edge": {"segments": 0, "groups": 0},
    }
    assert png.stat().st_size > 1_000


def test_documented_cli_converts_and_renders_the_fixture(tmp_path):
    """Run the documented crop, validation, and preview workflow."""

    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")
    converter = REPO_ROOT / "data_utils" / "lanelet2_to_bin.py"
    visualizer = REPO_ROOT / "data_utils" / "visualize_map_bin.py"
    binary = tmp_path / "map.bin"
    report = tmp_path / "validation.json"
    png = tmp_path / "map.png"

    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(FIXTURE),
            str(binary),
            "--crop",
            "0",
            "0",
            "9",
            "20",
            "--validation-report",
            str(report),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(visualizer), str(binary), "--png", str(png), "--width", "320", "--height", "240"],
        check=True,
    )

    validation = validate_bin(binary)
    assert validation["id_equals_index"] is True
    assert validation["lanes"] == 2
    decoded = read_bin(binary)
    assert all(0.0 <= x <= 9.0 for road in decoded["roads"] for x in road["x"])
    assert all(0.0 <= y <= 20.0 for road in decoded["roads"] for y in road["y"])
    report_data = json.loads(report.read_text())
    assert report_data["projection"]["crs"] == "EPSG:32654"
    assert png.stat().st_size > 1_000
