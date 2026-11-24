"""Write simulator binaries for the flat CARLA towns.

A town is 2D when the road-geometry z span is at most 0.1 m. Towns
with elevation or overpasses are skipped. Town06 is flat.

  data/carla/json/{Town}.json
  data/carla/roads/{Town}.json
  data/carla/binaries/map_%03d.bin

Sorted town names are the map index: Town01, Town02, Town06, Town10HD.
The binary is save_map_binary (91 states). This does not read XODR and
does not regenerate agents.

  python utils/carla.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from utils.common import write_log_binary

FLAT_Z_SPAN_M = 0.1
REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = REPO_ROOT / "data" / "carla" / "json"
BINARY_DIR = REPO_ROOT / "data" / "carla" / "binaries"


def road_z_span(scenario: dict) -> tuple[float, float]:
    """Min and max z across every road polyline point."""
    zs = [
        float(point.get("z", 0.0))
        for road in scenario.get("roads") or []
        for point in road.get("geometry") or []
    ]
    if not zs:
        raise ValueError("Scenario has no road geometry")
    return min(zs), max(zs)


def is_flat(scenario: dict, max_span: float = FLAT_Z_SPAN_M) -> bool:
    low, high = road_z_span(scenario)
    return (high - low) <= max_span


def selected_maps(json_dir: Path = JSON_DIR, max_span: float = FLAT_Z_SPAN_M) -> list[tuple[Path, float]]:
    """Sorted JSON paths whose road z span is within max_span, with that span."""
    chosen = []
    for path in sorted(json_dir.glob("*.json")):
        scenario = json.loads(path.read_text())
        low, high = road_z_span(scenario)
        span = high - low
        if span <= max_span:
            chosen.append((path, span))
    return chosen


def write_binaries(
    json_dir: Path = JSON_DIR,
    binary_dir: Path = BINARY_DIR,
    max_span: float = FLAT_Z_SPAN_M,
) -> list[dict]:
    """Pack each flat town. Returns one manifest row per written map."""
    rows = []
    binary_dir.mkdir(parents=True, exist_ok=True)
    for index, (path, span) in enumerate(selected_maps(json_dir, max_span)):
        scenario = json.loads(path.read_text())
        binary_path = binary_dir / f"map_{index:03d}.bin"
        write_log_binary(scenario, binary_path, unique_map_id=index)
        rows.append(
            {
                "map_index": index,
                "town": path.stem,
                "json": str(path.relative_to(REPO_ROOT)),
                "binary": str(binary_path.relative_to(REPO_ROOT)),
                "road_z_span_m": span,
                "num_objects": len(scenario.get("objects") or []),
                "num_roads": len(scenario.get("roads") or []),
            }
        )
    manifest_path = json_dir.parent / "manifest.json"
    manifest_path.write_text(json.dumps({"max_road_z_span_m": max_span, "maps": rows}, indent=2) + "\n")
    return rows


def check_binary(path: Path) -> tuple[int, int, int]:
    """Read scenario header counts: sdc index, object count, road count."""
    data = path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"{path} is too small to be a map binary")
    _sdc, n_tracks = struct.unpack_from("<ii", data, 16)
    offset = 24 + 4 * n_tracks
    n_objects, n_roads = struct.unpack_from("<ii", data, offset)
    if n_objects < 1 or n_roads < 1:
        raise ValueError(f"{path.name} has {n_objects} objects and {n_roads} roads")
    return _sdc, n_objects, n_roads


def main() -> None:
    rows = write_binaries()
    if not rows:
        raise SystemExit(f"No flat CARLA JSON under {JSON_DIR}")
    for row in rows:
        binary = REPO_ROOT / row["binary"]
        sdc, n_objects, n_roads = check_binary(binary)
        if n_objects != row["num_objects"] or n_roads != row["num_roads"]:
            raise SystemExit(f"{binary.name} header does not match {row['town']}")
        print(
            f"{row['town']} -> {binary.name} "
            f"objects={n_objects} roads={n_roads} sdc={sdc} "
            f"z_span={row['road_z_span_m']:.4f} bytes={binary.stat().st_size}"
        )


if __name__ == "__main__":
    main()
