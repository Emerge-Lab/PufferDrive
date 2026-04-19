import copy
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pufferlib.pufferl as pufferl
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator


def _extract_scalar_metadata(metadata):
    metadata = np.asarray(metadata)
    if metadata.ndim == 1:
        return metadata
    if metadata.ndim == 2:
        return metadata[:, 0]
    return metadata[:, 0, 0]


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_simulated_trajectories(simulated_trajectory_file):
    print(f"Loading simulated trajectories from {simulated_trajectory_file}...")
    with open(simulated_trajectory_file, "rb") as f:
        simulated = pickle.load(f)

    for key in ("x", "y", "z", "heading"):
        if simulated[key].ndim == 2:
            simulated[key] = simulated[key][:, np.newaxis, :]

    if "dones" not in simulated:
        simulated["dones"] = np.zeros_like(simulated["x"], dtype=bool)
    elif simulated["dones"].ndim == 2:
        simulated["dones"] = simulated["dones"][:, np.newaxis, :]

    return simulated


def _build_scenario_index(simulated):
    scenario_index = {}
    for idx, scenario_id in enumerate(_extract_scalar_metadata(simulated["scenario_id"]).tolist()):
        scenario_index.setdefault(scenario_id, []).append(idx)
    return scenario_index


def _filter_trajectories_by_scenario_ids(simulated, scenario_index, scenario_ids):
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenario_index]
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise KeyError(f"Missing simulated trajectories for {len(missing)} scenarios. First missing ids: {preview}")

    indices = np.array([idx for scenario_id in scenario_ids for idx in scenario_index[scenario_id]], dtype=int)
    return {key: value[indices] for key, value in simulated.items()}


def _sorted_map_files(map_dir):
    def _sort_key(path):
        try:
            return (0, int(path.stem.split("_")[-1]))
        except ValueError:
            return (1, path.name)

    return sorted(Path(map_dir).glob("*.bin"), key=_sort_key)


def _aggregate_scene_level_results(scene_level_results, wosac_enabled):
    aggregate_metrics = scene_level_results.mean().to_dict()

    if wosac_enabled:
        aggregate_metrics["total_num_agents"] = int(scene_level_results["num_agents_per_scene"].sum())
        aggregate_metrics["realism_score_std"] = scene_level_results["realism_meta_score"].std()
    else:
        aggregate_metrics["num_scenarios"] = int(scene_level_results.shape[0])

    return aggregate_metrics


def _print_results(results, wosac_enabled):
    if wosac_enabled:
        print("\n--- WOSAC METRICS START ---")
        print(json.dumps(results, indent=4, default=_json_default))
        print("--- WOSAC METRICS END ---")
        return

    print("\nPLANNING_METRICS_START")
    print(json.dumps(results, indent=4, default=_json_default))
    print("PLANNING_METRICS_END")


def _validate_agent_counts(simulated_trajectories, ground_truth_trajectories, planning_eval):
    num_agents_gt = ground_truth_trajectories["x"].shape[0]
    num_agents_sim = simulated_trajectories["x"].shape[0]

    print(f"Number of scenarios: {len(np.unique(ground_truth_trajectories['scenario_id']))}")
    print(f"Number of controlled agents: {num_agents_gt}")
    print(f"Number of evaluated agents: {ground_truth_trajectories['is_track_to_predict'].sum()}")

    assert num_agents_sim >= num_agents_gt, (
        "There is less agents in your simulation than in the GT, so the computation won't be valid"
    )

    if num_agents_sim > num_agents_gt:
        if planning_eval:
            print("There are more agents in your sim than in the GT")
            print("If you are evaluating on a subset of your trajectories it is fine.")
            print("Else, you should consider changing the value of MAX_AGENTS in drive.h and compile")
        else:
            print("If you are evaluating on a subset of your trajectories it is fine.")
            print("\n Else, you should consider changing the value of MAX_AGENTS in drive.h and compile")


def align_trajectories(simulated, ground_truth):
    # Idea is to use the (scenario_id, id) pair to reindex simulated_trajectories in order to align it with GT
    gt_scenario_ids = _extract_scalar_metadata(ground_truth["scenario_id"])
    sim_scenario_ids = _extract_scalar_metadata(simulated["scenario_id"])

    gt_ids = _extract_scalar_metadata(ground_truth["id"])
    sim_ids = _extract_scalar_metadata(simulated["id"])

    lookup = {(s_id, a_id): idx for idx, (s_id, a_id) in enumerate(zip(sim_scenario_ids, sim_ids))}

    try:
        indices = [lookup[(s, i)] for (s, i) in zip(gt_scenario_ids, gt_ids)]
        indices = np.array(indices, dtype=int)
    except KeyError:
        print("An agent present in the GT is missing in your simulation")
        raise

    sim_traj = {k: v[indices] for k, v in simulated.items()}
    sim_traj["valid"] = ground_truth["valid"].copy()
    if "is_sdc" in ground_truth:
        sim_traj["is_sdc"] = ground_truth["is_sdc"].copy()

    return sim_traj


def check_alignment(simulated, ground_truth, tolerance=1e-4):
    # Imported rollouts may use placeholder coordinates on invalid timesteps, so
    # compare the first valid timestep per agent instead of always comparing t=0.
    gt_valid = np.asarray(ground_truth["valid"])[:, 0, :]
    valid_agents = gt_valid.any(axis=1)
    if not np.any(valid_agents):
        return True

    first_valid_idx = np.argmax(gt_valid, axis=1)
    agent_indices = np.where(valid_agents)[0]
    timestep_indices = first_valid_idx[valid_agents]

    gt_x = ground_truth["x"][agent_indices, 0, timestep_indices]
    gt_y = ground_truth["y"][agent_indices, 0, timestep_indices]
    gt_z = ground_truth["z"][agent_indices, 0, timestep_indices]

    sim_x = simulated["x"][agent_indices, 0, timestep_indices]
    sim_y = simulated["y"][agent_indices, 0, timestep_indices]
    sim_z = simulated["z"][agent_indices, 0, timestep_indices]

    diffs = np.maximum(np.maximum(np.abs(gt_x - sim_x), np.abs(gt_y - sim_y)), np.abs(gt_z - sim_z))

    if np.any(diffs > tolerance):
        print("Tolerance broken by this distance: ", np.max(diffs))
        return False
    return True


def _configure_eval_args(args, map_dir, num_maps):
    run_args = copy.deepcopy(args)
    run_args["env"]["map_dir"] = map_dir
    run_args["env"]["num_maps"] = num_maps
    run_args["env"]["episode_length"] = 91
    run_args["env"]["sequential_map_sampling"] = True
    run_args["vec"] = dict(backend=run_args["eval"]["backend"], num_envs=1)
    return run_args


def _collect_wosac_context(args):
    env_name = "puffer_drive"
    args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
    args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
    args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]
    args["env"]["goal_behavior"] = args["eval"]["wosac_goal_behavior"]
    args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

    vecenv = pufferl.load_env(env_name, args)
    evaluator = WOSACEvaluator(args)
    ground_truth_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
    return evaluator, ground_truth_trajectories, agent_state, road_edge_polylines, vecenv


def _collect_planning_context(args):
    env_name = "puffer_drive"
    from pufferlib.ocean.benchmark.evaluator import PlanningEvaluator

    args["eval"]["wosac_num_rollouts"] = 1
    args["env"]["control_mode"] = "control_sdc_only"
    args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]
    args["env"]["goal_behavior"] = args["eval"]["wosac_goal_behavior"]
    args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

    gt_args = copy.deepcopy(args)
    gt_args["env"]["control_mode"] = "control_wosac"
    gt_vecenv = pufferl.load_env(env_name, gt_args)

    ground_truth_trajectories = WOSACEvaluator(gt_args).collect_ground_truth_trajectories(gt_vecenv)
    agent_state = gt_vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = gt_vecenv.driver_env.get_road_edge_polylines()
    evaluator = PlanningEvaluator(args)
    return evaluator, ground_truth_trajectories, agent_state, road_edge_polylines, gt_vecenv


def _evaluate_scene_level(simulated_trajectories, scenario_index, args, wosac_enabled, drop_last_scenario=True):
    backend = args["eval"]["backend"]
    mode_name = "WOSAC" if wosac_enabled else "Planning"
    assert backend == "PufferEnv", f"{mode_name} evaluation only supports PufferEnv backend."

    if wosac_enabled:
        evaluator, ground_truth_trajectories, agent_state, road_edge_polylines, vecenv = _collect_wosac_context(args)
    else:
        evaluator, ground_truth_trajectories, agent_state, road_edge_polylines, vecenv = _collect_planning_context(args)

    try:
        if scenario_index is not None:
            scenario_ids = np.unique(_extract_scalar_metadata(ground_truth_trajectories["scenario_id"])).tolist()
            simulated_trajectories = _filter_trajectories_by_scenario_ids(
                simulated_trajectories, scenario_index, scenario_ids
            )

        _validate_agent_counts(simulated_trajectories, ground_truth_trajectories, planning_eval=not wosac_enabled)

        simulated_trajectories = align_trajectories(simulated_trajectories, ground_truth_trajectories)

        assert check_alignment(simulated_trajectories, ground_truth_trajectories), (
            "There might be an issue with the way you generated your data."
        )

        if wosac_enabled:
            print("\n--- Computing WOSAC Metrics ---")
            return evaluator.compute_metrics(
                ground_truth_trajectories,
                simulated_trajectories,
                agent_state,
                road_edge_polylines,
                aggregate_results=False,
                drop_last_scenario=drop_last_scenario,
            )

        return evaluator.compute_metrics(
            simulated_trajectories,
            agent_state,
            road_edge_polylines,
            aggregate_results=False,
            ground_truth_trajectories=ground_truth_trajectories,
            goal_radius=args["env"]["goal_radius"],
            goal_speed=args["env"].get("goal_speed", 100.0),
        )
    finally:
        vecenv.close()


def evaluate_trajectories_chunked(simulated_trajectory_file, args, chunk_size):
    map_dir = args["eval"]["map_dir"]
    target_num_maps = args["eval"].get("wosac_num_maps", args["env"]["num_maps"])
    map_files = _sorted_map_files(map_dir)[:target_num_maps]
    if not map_files:
        raise FileNotFoundError(f"No map files found in {map_dir}")

    dataset_name = Path(map_dir).name
    wosac_enabled = args["eval"]["wosac_realism_eval"]
    mode_name = "WOSAC realism" if wosac_enabled else "Planning"

    print(f"Running chunked {mode_name} evaluation with {dataset_name} dataset.")
    print(f"Processing {len(map_files)} maps in chunks of {chunk_size}.")

    simulated_trajectories = _load_simulated_trajectories(simulated_trajectory_file)
    scenario_index = _build_scenario_index(simulated_trajectories)
    scene_level_results = []

    for chunk_idx, start in enumerate(range(0, len(map_files), chunk_size), start=1):
        chunk_files = map_files[start : start + chunk_size]
        print(f"\n=== Chunk {chunk_idx}: {len(chunk_files)} maps ===")

        with tempfile.TemporaryDirectory(prefix="pufferdrive_eval_chunk_") as temp_dir:
            chunk_dir = Path(temp_dir)
            for local_idx, map_file in enumerate(chunk_files):
                (chunk_dir / f"map_{local_idx:03d}.bin").symlink_to(map_file)

            chunk_args = _configure_eval_args(args, temp_dir, len(chunk_files))
            chunk_results = _evaluate_scene_level(
                simulated_trajectories,
                scenario_index,
                chunk_args,
                wosac_enabled=wosac_enabled,
                drop_last_scenario=False,
            )
            scene_level_results.append(chunk_results)

    combined_results = pd.concat(scene_level_results)
    combined_results = combined_results[~combined_results.index.duplicated(keep="first")]

    if args["eval"]["wosac_aggregate_results"]:
        results = _aggregate_scene_level_results(combined_results, wosac_enabled)
        _print_results(results, wosac_enabled)
        return results

    print("\n Scene-level results:\n")
    print(combined_results)
    return combined_results


def evaluate_trajectories(simulated_trajectory_file, args):
    """
    Evaluates pre-computed simulated trajectories against live ground truth from the environment.
    """
    map_dir = args["eval"]["map_dir"]
    target_num_maps = args["eval"].get("wosac_num_maps", args["env"]["num_maps"])
    dataset_name = Path(map_dir).name
    wosac_enabled = args["eval"]["wosac_realism_eval"]

    if wosac_enabled:
        print(f"Running WOSAC realism evaluation with {dataset_name} dataset. \n")
    else:
        print(f"Running Planning evaluation with {dataset_name} dataset. \n")

    simulated_trajectories = _load_simulated_trajectories(simulated_trajectory_file)
    run_args = _configure_eval_args(args, map_dir, target_num_maps)
    scene_level_results = _evaluate_scene_level(
        simulated_trajectories,
        scenario_index=None,
        args=run_args,
        wosac_enabled=wosac_enabled,
        drop_last_scenario=True,
    )

    if args["eval"]["wosac_aggregate_results"]:
        results = _aggregate_scene_level_results(scene_level_results, wosac_enabled)
        _print_results(results, wosac_enabled)
        return results

    print("\n Scene-level results:\n")
    print(scene_level_results)
    return scene_level_results


if __name__ == "__main__":
    simulated_file = None
    chunk_size = None
    if "--simulated-file" in sys.argv:
        try:
            idx = sys.argv.index("--simulated-file")
            simulated_file = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)
        except (ValueError, IndexError):
            print("ERROR: --simulated-file argument found but no value was provided.")
            sys.exit(1)

    if simulated_file is None:
        print("ERROR: --simulated-file argument is required.")
        sys.exit(1)

    if "--chunk-size" in sys.argv:
        try:
            idx = sys.argv.index("--chunk-size")
            chunk_size = int(sys.argv[idx + 1])
            sys.argv.pop(idx)
            sys.argv.pop(idx)
        except (ValueError, IndexError):
            print("ERROR: --chunk-size argument found but no valid integer value was provided.")
            sys.exit(1)

    config = pufferl.load_config("puffer_drive")

    if chunk_size is not None and chunk_size > 0:
        evaluate_trajectories_chunked(simulated_file, args=config, chunk_size=chunk_size)
    else:
        evaluate_trajectories(simulated_file, args=config)
