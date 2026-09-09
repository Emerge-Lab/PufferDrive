"""Replay a saved ReGentS artifact set in C under a chosen ego controller.

Generation targets one ego; an artifact set is therefore not controller-neutral. This
replays a fixed adversary plan against any ego so IDM, log replay, and a policy can be
compared on identical scenarios. No Torch optimization runs here: C is the only oracle.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.artifacts import load_generation_artifact, source_configuration_hash
from pufferlib.ocean.regents.policy_ego import PolicyEgoActor, ReGentSPolicyEgoConfig
from pufferlib.ocean.regents.rollout import _capture_c_rollout, baseline_relative_events

EVALUATION_FILE_NAME = "evaluation_metrics.csv"
EVALUATION_EGO_CONTROLLERS = ("replay", "idm", "corridor_idm", "pdm", "policy")
EVALUATION_FIELD_NAMES = (
    "scenario_index",
    "scenario_id",
    "ego_controller",
    "generated_against",
    "seed",
    "candidate_count",
    "selected_adversary_idx",
    "ego_collision",
    "actionable_collision",
    "background_collision",
    "baseline_ego_collision",
    "offroad",
    "collision_timestep",
    "ego_collision_timestep",
    "recorded_collision_timestep",
    "reproduces_recorded_collision",
)


@dataclass(frozen=True)
class EvaluationReport:
    ego_controller: str
    rows: tuple
    metrics_path: Path


def _evaluation_env(metadata, ego_controller, ego_policy):
    """Rebuild the generation env, swapping only the ego controller it targeted."""
    source_configuration = metadata["source_configuration"]
    environment = dict(source_configuration["env"])
    # The artifact's deterministic seed is authoritative, so an env-level seed cannot
    # silently replay a different scenario than the one that was generated.
    environment.pop("seed", None)
    environment.pop("eval_map_indices", None)
    environment.pop("eval_scenario_seeds", None)
    environment["sdc_controller"] = ego_controller
    # The jerk action head a policy checkpoint was trained on governs the ego only;
    # injected adversaries always integrate the classic bicycle model.
    environment["dynamics_model"] = "jerk" if ego_controller == "policy" else "classic"
    if ego_controller == "policy":
        if ego_policy is None:
            raise ValueError("Evaluating a policy ego requires an ego policy configuration")
        for key in (
            "obs_dropout_lane",
            "obs_dropout_boundary",
            "partner_blindness_prob",
            "partner_blindness_trigger_prob",
            "phantom_braking_prob",
            "phantom_braking_trigger_prob",
        ):
            environment[key] = 0.0
    return environment


def _evaluate_one_artifact(artifact_path, scenario_idx, ego_controller, ego_policy):
    metadata, arrays = load_generation_artifact(artifact_path)
    map_path = Path(metadata["source_map"])
    if not map_path.is_file():
        raise ValueError(f"Artifact {artifact_path} references a missing source map: {map_path}")
    # The recorded hash covers the canonical config and the exact map bytes, so this
    # rejects both a mutated map file and a config that no longer describes the artifact.
    recomputed = source_configuration_hash(metadata["source_configuration"], map_path)
    if recomputed != metadata["source_configuration_hash"]:
        raise ValueError(f"Artifact {artifact_path} no longer matches its source map and configuration")

    seed = int(metadata["deterministic_seed"])
    environment = _evaluation_env(metadata, ego_controller, ego_policy)
    action_mask = np.ascontiguousarray(arrays["optimized_action_mask"][0], dtype=np.bool_)
    baseline_plan = np.ascontiguousarray(arrays["initial_actions"][0], dtype=np.float32)
    optimized_plan = np.ascontiguousarray(arrays["optimized_actions"][0], dtype=np.float32)
    transition_count = optimized_plan.shape[1]
    agent_count = optimized_plan.shape[0]
    scenario_id = metadata["scenario_id"]

    drive = Drive(**environment, eval_map_indices=[scenario_idx], eval_scenario_seeds=[seed], seed=seed)
    try:
        ego_action_fn = PolicyEgoActor(ego_policy, drive) if ego_controller == "policy" else None
        try:
            binding.regents_set_action_plan(drive.c_envs, baseline_plan, action_mask)
            baseline = _capture_c_rollout(
                drive, transition_count, scenario_id, agent_count, seed, ego_action_fn=ego_action_fn
            )
            binding.regents_set_action_plan(drive.c_envs, optimized_plan, action_mask)
            adversarial = _capture_c_rollout(
                drive, transition_count, scenario_id, agent_count, seed, ego_action_fn=ego_action_fn
            )
        finally:
            binding.regents_set_action_plan(drive.c_envs, baseline_plan, np.zeros_like(action_mask))
    finally:
        drive.close()

    selected_adversary_idx = metadata["optimization"]["selected_adversary_idx"]
    injected_agent_mask = action_mask.any(axis=1)
    (
        ego_collision,
        actionable_collision,
        background_collision,
        offroad,
        first_collision_timestep,
        _,
        first_ego_collision_timestep,
        baseline_ego_collision,
        _,
    ) = baseline_relative_events(baseline, adversarial, selected_adversary_idx, injected_agent_mask)

    recorded_collision_timestep = metadata["c_replay"]["first_collision_timestep"]
    return {
        "scenario_index": scenario_idx,
        "scenario_id": scenario_id,
        "ego_controller": ego_controller,
        "generated_against": metadata.get("ego_controller", "unknown"),
        "seed": seed,
        "candidate_count": int(arrays["candidate_mask"][0].sum()),
        "selected_adversary_idx": selected_adversary_idx,
        "ego_collision": int(ego_collision),
        "actionable_collision": int(actionable_collision),
        "background_collision": int(background_collision),
        "baseline_ego_collision": int(baseline_ego_collision),
        "offroad": int(offroad),
        "collision_timestep": first_collision_timestep,
        "ego_collision_timestep": first_ego_collision_timestep,
        "recorded_collision_timestep": recorded_collision_timestep,
        "reproduces_recorded_collision": int(first_collision_timestep == recorded_collision_timestep),
    }


def evaluate_artifact_set(artifact_dir, ego_controller, *, ego_policy=None, output_dir=None):
    """Replay every artifact in a generated set under one ego controller."""
    if ego_controller not in EVALUATION_EGO_CONTROLLERS:
        raise ValueError(f"ego_controller must be one of {EVALUATION_EGO_CONTROLLERS}")
    if ego_policy is not None and not isinstance(ego_policy, ReGentSPolicyEgoConfig):
        raise TypeError("ego_policy must be a ReGentSPolicyEgoConfig")
    source = Path(artifact_dir)
    artifact_paths = sorted((source / "npz").glob("scenario_*.npz"))
    if not artifact_paths:
        raise ValueError(f"No ReGentS artifacts found under {source / 'npz'}")

    rows = []
    for artifact_path in artifact_paths:
        scenario_idx = int(artifact_path.stem.split("_")[-1])
        rows.append(_evaluate_one_artifact(artifact_path, scenario_idx, ego_controller, ego_policy))

    destination = Path(output_dir) if output_dir is not None else source
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / f"{ego_controller}_{EVALUATION_FILE_NAME}"
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=EVALUATION_FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)
    return EvaluationReport(ego_controller=ego_controller, rows=tuple(rows), metrics_path=metrics_path)


def summarize_evaluations(reports):
    """One row per ego controller: how the fixed adversary set fares against each."""
    summary = []
    for report in reports:
        scenario_count = len(report.rows)
        if scenario_count == 0:
            raise ValueError(f"Evaluation of {report.ego_controller} produced no rows")
        summary.append(
            {
                "ego_controller": report.ego_controller,
                "scenario_count": scenario_count,
                "ego_collision_rate": sum(row["ego_collision"] for row in report.rows) / scenario_count,
                "actionable_collision_rate": sum(row["actionable_collision"] for row in report.rows) / scenario_count,
                "background_collision_rate": sum(row["background_collision"] for row in report.rows) / scenario_count,
                "offroad_rate": sum(row["offroad"] for row in report.rows) / scenario_count,
                "reproduces_recorded_collision": sum(row["reproduces_recorded_collision"] for row in report.rows)
                / scenario_count,
            }
        )
    return tuple(summary)


def format_summary_table(summary):
    """Render the controller comparison as a fixed-width table."""
    columns = (
        ("ego_controller", "ego", "{}"),
        ("scenario_count", "scenarios", "{}"),
        ("ego_collision_rate", "ego col", "{:.3f}"),
        ("actionable_collision_rate", "actionable", "{:.3f}"),
        ("background_collision_rate", "bg col", "{:.3f}"),
        ("offroad_rate", "offroad", "{:.3f}"),
        ("reproduces_recorded_collision", "reproduces", "{:.3f}"),
    )
    header = [label for _, label, _ in columns]
    rows = [[template.format(row[key]) for key, _, template in columns] for row in summary]
    widths = [
        max(len(header[idx]), *(len(row[idx]) for row in rows)) if rows else len(header[idx])
        for idx in range(len(columns))
    ]
    lines = ["  ".join(label.ljust(widths[idx]) for idx, label in enumerate(header))]
    lines.append("  ".join("-" * widths[idx] for idx in range(len(columns))))
    for row in rows:
        lines.append("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)
