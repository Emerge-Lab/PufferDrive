#!/usr/bin/env python3
"""Plot sampled imported trajectory rollouts from SMART export pickles."""

from __future__ import annotations

import argparse
import gc
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-agents", type=int, default=40)
    parser.add_argument("--max-rollouts", type=int, default=8)
    parser.add_argument(
        "--glob",
        default="*_wosac_all_agents*.pkl",
        help="Which exports to visualize. Defaults to all-agent WOSAC exports.",
    )
    return parser.parse_args()


def extract_scalar_metadata(values):
    values = np.asarray(values)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[:, 0]
    return values[:, 0, 0]


def model_name(path: Path):
    stem = re.sub(r"_r\d+$", "", path.stem)
    stem = stem.replace("_val10k_wosac_all_agents", "")
    stem = stem.replace("_val10k_planning_sdc_only", "")
    return re.sub(r"_val\d+k?$", "", stem)


def normalize_traj_array(values):
    values = np.asarray(values)
    if values.ndim == 2:
        return values[:, None, :]
    return values


def load_export(path: Path):
    print(f"Loading {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    for key in ("x", "y", "heading", "id", "scenario_id"):
        if key in data:
            data[key] = normalize_traj_array(data[key])
    return data


def choose_scenarios(first_file: Path, num_scenes: int, seed: int):
    data = load_export(first_file)
    scenario_ids = np.unique(extract_scalar_metadata(data["scenario_id"]))
    rng = np.random.default_rng(seed)
    count = min(num_scenes, len(scenario_ids))
    chosen = rng.choice(scenario_ids, size=count, replace=False)
    del data
    gc.collect()
    return chosen.tolist()


def rank_agents(x, y, valid):
    if valid is None:
        valid = np.isfinite(x[:, 0, :]) & np.isfinite(y[:, 0, :])
    first_rollout_valid = valid[:, 0, :]
    start_idx = np.argmax(first_rollout_valid, axis=1)
    end_idx = first_rollout_valid.shape[1] - 1 - np.argmax(first_rollout_valid[:, ::-1], axis=1)
    agent_idx = np.arange(x.shape[0])
    distance = np.hypot(x[agent_idx, 0, end_idx] - x[agent_idx, 0, start_idx], y[agent_idx, 0, end_idx] - y[agent_idx, 0, start_idx])
    distance[~first_rollout_valid.any(axis=1)] = -1.0
    return np.argsort(distance)[::-1]


def plot_scene(data, scenario_id, title, output_path: Path, max_agents: int, max_rollouts: int):
    scenario_ids = extract_scalar_metadata(data["scenario_id"])
    mask = scenario_ids == scenario_id
    if not np.any(mask):
        print(f"Scenario {scenario_id} not found in {title}")
        return False

    x = data["x"][mask]
    y = data["y"][mask]
    valid = data.get("valid")
    if valid is not None:
        valid = normalize_traj_array(valid)[mask].astype(bool)
    else:
        valid = np.isfinite(x) & np.isfinite(y)

    ids = extract_scalar_metadata(data["id"][mask])
    ranked = rank_agents(x, y, valid)
    selected = ranked[: min(max_agents, len(ranked))]
    sdc_candidates = np.where(ids <= -2)[0]
    if len(sdc_candidates) > 0 and sdc_candidates[0] not in selected:
        selected = np.r_[sdc_candidates[0], selected[:-1]]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    rollout_count = min(max_rollouts, x.shape[1])

    for local_idx in selected:
        is_sdc = ids[local_idx] <= -2
        color = "#b21f35" if is_sdc else "#2a2f36"
        linewidth = 1.8 if is_sdc else 0.8
        alpha = 0.7 if is_sdc else 0.22
        for rollout_idx in range(rollout_count):
            step_mask = valid[local_idx, rollout_idx] & np.isfinite(x[local_idx, rollout_idx]) & np.isfinite(y[local_idx, rollout_idx])
            if step_mask.sum() < 2:
                continue
            ax.plot(
                x[local_idx, rollout_idx, step_mask],
                y[local_idx, rollout_idx, step_mask],
                color=color,
                linewidth=linewidth,
                alpha=alpha if rollout_idx == 0 else alpha * 0.35,
            )
            if rollout_idx == 0:
                ax.scatter(x[local_idx, rollout_idx, step_mask][0], y[local_idx, rollout_idx, step_mask][0], s=8, color=color, alpha=0.8)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} in {args.input_dir}")

    scenarios = choose_scenarios(files[0], args.num_scenes, args.seed)
    rows = []
    for path in files:
        model = model_name(path)
        data = load_export(path)
        for scenario_id in scenarios:
            output_path = args.output_dir / model / f"{model}_scenario_{scenario_id}.png"
            title = f"{model} | scenario {scenario_id}"
            if plot_scene(data, scenario_id, title, output_path, args.max_agents, args.max_rollouts):
                rows.append({"model": model, "scenario_id": scenario_id, "source_file": path.name, "plot_path": str(output_path)})
        del data
        gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "visualization_index.csv", index=False)
    print(f"Wrote {len(rows)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
