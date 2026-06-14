"""Convert exported Waymo Motion JSON scenarios into PufferDrive binaries."""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from pufferlib.ocean.drive.drive import load_map


def prepare_scenario(source_path, prepared_path):
    with source_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    sdc_track_index = data.get("metadata", {}).get("sdc_track_index")
    for index, obj in enumerate(data.get("objects", [])):
        obj["mark_as_expert"] = index != sdc_track_index

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.write_text(json.dumps(data), encoding="utf-8")
    return str(data.get("scenario_id") or source_path.stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resources/drive/binaries/training"),
    )
    parser.add_argument(
        "--prepared-json-dir",
        type=Path,
        default=Path("resources/drive/waymo_training_json"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.prepared_json_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for map_index, source_path in enumerate(args.json_files):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        prepared_path = args.prepared_json_dir / f"map_{map_index:03d}.json"
        binary_path = args.output_dir / f"map_{map_index:03d}.bin"
        scenario_id = prepare_scenario(source_path, prepared_path)
        load_map(
            str(prepared_path),
            unique_map_id=map_index,
            binary_output=str(binary_path),
        )
        manifest.append(
            {
                "map_index": map_index,
                "scenario_id": scenario_id,
                "source": str(source_path),
                "binary": str(binary_path),
            }
        )
        print(f"[{map_index + 1}/{len(args.json_files)}] {scenario_id} -> {binary_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} maps")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
