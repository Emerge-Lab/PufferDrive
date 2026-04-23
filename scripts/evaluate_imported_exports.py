#!/usr/bin/env python3
"""Batch-evaluate imported SMART trajectory exports and collate summary CSVs."""

from __future__ import annotations

import argparse
import copy
import json
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
    _load_simulated_trajectories,
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
    for suffix, metadata in EXPORT_SUFFIXES.items():
        if stem.endswith(suffix):
            model = stem[: -len(suffix)]
            return model, metadata
    raise ValueError(
        f"Could not infer eval mode from {path.name}. Expected one of: "
        + ", ".join(f"*{suffix}.pkl" for suffix in EXPORT_SUFFIXES)
    )


def list_exports(input_dir: Path):
    exports = sorted(path for path in input_dir.glob("*.pkl") if path.is_file())
    if not exports:
        raise FileNotFoundError(f"No .pkl exports found in {input_dir}")
    return exports


def make_config(args, *, wosac_enabled: bool, planning_filter: str | None = None, wosac_filter: str | None = None):
    config = pufferl.load_config("puffer_drive")
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


def main():
    args = parse_args()
    exports = list_exports(args.input_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (args.input_dir / f"eval_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []
    raw_metrics = OrderedDict()

    for export_path in exports:
        model, metadata = infer_export_metadata(export_path)
        eval_scope = metadata["eval_scope"]
        planning_filter = metadata["planning_filter"]

        simulated = _load_simulated_trajectories(str(export_path))
        num_rollouts = int(simulated["x"].shape[1])
        num_agents = int(simulated["x"].shape[0])
        print(f"\n=== {export_path.name} ===")
        print(f"model={model} eval_scope={eval_scope} num_agents={num_agents} num_rollouts={num_rollouts}")

        planning_config = make_config(args, wosac_enabled=False, planning_filter=planning_filter)
        planning_metrics = run_eval(export_path, planning_config, args.chunk_size)

        row = {
            "model": model,
            "eval_scope": eval_scope,
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

    long_df = pd.DataFrame(long_rows)
    long_csv = output_dir / "summary_full_with_wosac.csv"
    long_df.to_csv(long_csv, index=False)

    wide_df = pd.DataFrame(build_wide_rows(long_rows))
    wide_csv = output_dir / "summary_model_wide.csv"
    wide_df.to_csv(wide_csv, index=False)

    with (output_dir / "summary_raw_metrics.json").open("w") as f:
        json.dump(raw_metrics, f, indent=2)

    print(f"\nWrote long summary to {long_csv}")
    print(f"Wrote wide summary to {wide_csv}")
    print(f"Wrote raw metrics to {output_dir / 'summary_raw_metrics.json'}")


if __name__ == "__main__":
    main()
