#!/usr/bin/env python3
"""Batch-evaluate imported SMART trajectory exports and collate summary CSVs."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pufferlib.pufferl as pufferl
from pufferlib.ocean.benchmark.evaluate_imported_trajectories import (
    evaluate_trajectories,
    evaluate_trajectories_chunked,
)


EXPORT_SUFFIXES = OrderedDict(
    [
        ("_planning_sdc_only", {"eval_scope": "sdc_only", "planning_filter": "sdc", "run_wosac": False}),
        ("_wosac_all_agents", {"eval_scope": "all_agents", "planning_filter": "all", "run_wosac": True}),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing exported .pkl trajectory files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write collated CSVs. Defaults to eval_results_<timestamp> under --input-dir.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Map chunk size for both planning and WOSAC evaluation. Use 0 to disable chunking.",
    )
    parser.add_argument(
        "--planning-map-agent-chunk-size",
        type=int,
        default=32,
        help="Optional per-scenario agent chunk size for planning map metrics. Use 0 to disable chunking.",
    )
    parser.add_argument(
        "--planning-interaction-eval-chunk-size",
        type=int,
        default=16,
        help="Optional per-scenario evaluated-agent chunk size for planning interaction metrics. Use 0 to disable.",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=None,
        help="Override config eval.map_dir.",
    )
    parser.add_argument(
        "--num-maps",
        type=int,
        default=None,
        help="Override config eval.wosac_num_maps / env.num_maps.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override config train.device.",
    )
    parser.add_argument(
        "--skip-wosac",
        action="store_true",
        help="Skip WOSAC realism evaluation even for *_wosac_all_agents.pkl files.",
    )
    return parser.parse_args()


def infer_export_metadata(path: Path):
    stem = path.stem
    export_variant = ""
    rollout_match = re.search(r"_r\d+$", stem)
    if rollout_match:
        export_variant = rollout_match.group(0).lstrip("_")
        stem = stem[: rollout_match.start()]

    for suffix, metadata in EXPORT_SUFFIXES.items():
        if stem.endswith(suffix):
            model = stem[: -len(suffix)]
            model = re.sub(r"_val\d+k?$", "", model)
            return model, metadata, export_variant
    raise ValueError(
        f"Could not infer eval mode from {path.name}. Expected one of: "
        + ", ".join(f"*{suffix}[_rN].pkl" for suffix in EXPORT_SUFFIXES)
    )


def list_exports(input_dir: Path):
    exports = sorted(path for path in input_dir.glob("*.pkl") if path.is_file())
    if not exports:
        raise FileNotFoundError(f"No .pkl exports found in {input_dir}")
    return exports


def make_config(args, *, wosac_enabled: bool, planning_filter: str | None = None, wosac_filter: str | None = None):
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        config = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv

    config["eval"]["wosac_aggregate_results"] = True
    config["eval"]["wosac_realism_eval"] = wosac_enabled

    if args.map_dir is not None:
        config["eval"]["map_dir"] = str(args.map_dir)
    if args.num_maps is not None:
        config["env"]["num_maps"] = int(args.num_maps)
        config["eval"]["wosac_num_maps"] = int(args.num_maps)
    if args.device is not None:
        config["train"]["device"] = args.device

    if planning_filter is not None:
        config["eval"]["planning_eval_agent_filter"] = planning_filter

    if wosac_filter is not None:
        config["eval"]["wosac_eval_agent_filter"] = wosac_filter

    if args.planning_map_agent_chunk_size and args.planning_map_agent_chunk_size > 0:
        config["eval"]["planning_map_agent_chunk_size"] = int(args.planning_map_agent_chunk_size)
    if args.planning_interaction_eval_chunk_size and args.planning_interaction_eval_chunk_size > 0:
        config["eval"]["planning_interaction_eval_chunk_size"] = int(args.planning_interaction_eval_chunk_size)

    return config


def run_eval(path: Path, config, chunk_size: int):
    if chunk_size and chunk_size > 0:
        return evaluate_trajectories_chunked(str(path), args=copy.deepcopy(config), chunk_size=chunk_size)
    return evaluate_trajectories(str(path), args=copy.deepcopy(config))


def prefixed(metrics: dict, prefix: str):
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def build_wide_rows(long_rows):
    result = OrderedDict()
    for row in long_rows:
        model = row["model"]
        eval_scope = row["eval_scope"]
        wide_row = result.setdefault(model, {"model": model})

        for key, value in row.items():
            if key in {"model", "eval_scope", "source_file"}:
                continue
            if key.startswith("planning_"):
                if eval_scope == "sdc_only":
                    wide_key = key.replace("planning_", "sdc_planning_", 1)
                elif eval_scope == "all_agents":
                    wide_key = key.replace("planning_", "all_agents_planning_", 1)
                else:
                    raise ValueError(f"Unknown eval_scope={eval_scope!r}")
                wide_row[wide_key] = value
            elif key.startswith("wosac_"):
                wide_row[key] = value

    return list(result.values())


def write_outputs(output_dir: Path, long_rows, raw_metrics):
    long_df = pd.DataFrame(long_rows)
    long_csv = output_dir / "summary_full_with_wosac.csv"
    long_df.to_csv(long_csv, index=False)

    wide_df = pd.DataFrame(build_wide_rows(long_rows))
    wide_csv = output_dir / "summary_model_wide.csv"
    wide_df.to_csv(wide_csv, index=False)

    raw_json = output_dir / "summary_raw_metrics.json"
    with raw_json.open("w") as f:
        json.dump(raw_metrics, f, indent=2)

    return long_csv, wide_csv, raw_json


def main():
    args = parse_args()
    exports = list_exports(args.input_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (args.input_dir / f"eval_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []
    raw_metrics = OrderedDict()
    long_csv = output_dir / "summary_full_with_wosac.csv"
    raw_json = output_dir / "summary_raw_metrics.json"
    if long_csv.exists():
        existing_df = pd.read_csv(long_csv)
        long_rows = existing_df.to_dict("records")
        print(f"Resuming from {long_csv}; found {len(long_rows)} completed rows.")
    if raw_json.exists():
        with raw_json.open() as f:
            raw_metrics = OrderedDict(json.load(f))

    completed_files = {row["source_file"] for row in long_rows if "source_file" in row}

    for export_path in exports:
        if export_path.name in completed_files:
            print(f"Skipping completed export {export_path.name}")
            continue

        model, metadata, export_variant = infer_export_metadata(export_path)
        eval_scope = metadata["eval_scope"]
        planning_filter = metadata["planning_filter"]

        print(f"\n=== {export_path.name} ===")
        print(f"model={model} eval_scope={eval_scope} export_variant={export_variant or 'default'}")

        planning_config = make_config(args, wosac_enabled=False, planning_filter=planning_filter)
        planning_metrics = run_eval(export_path, planning_config, args.chunk_size)

        row = {
            "model": model,
            "eval_scope": eval_scope,
            "export_variant": export_variant,
            "source_file": export_path.name,
            **prefixed(planning_metrics, "planning_"),
        }

        raw_metrics[export_path.name] = {"planning": planning_metrics}

        if metadata["run_wosac"] and not args.skip_wosac:
            wosac_config = make_config(args, wosac_enabled=True, wosac_filter="tracks_to_predict")
            wosac_metrics = run_eval(export_path, wosac_config, args.chunk_size)
            row.update(prefixed(wosac_metrics, "wosac_"))
            raw_metrics[export_path.name]["wosac"] = wosac_metrics

        long_rows.append(row)
        long_csv, wide_csv, raw_json = write_outputs(output_dir, long_rows, raw_metrics)
        print(f"Wrote incremental summaries to {output_dir}")

    print(f"\nWrote long summary to {long_csv}")
    print(f"Wrote wide summary to {wide_csv}")
    print(f"Wrote raw metrics to {raw_json}")


if __name__ == "__main__":
    main()
