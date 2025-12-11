import pickle
import numpy as np
import pufferlib.pufferl as pufferl
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator


def check_alignment(simulated, ground_truth, tolerance=1e-4):
    # Check that initial positions match within a tolerance
    sim_x = simulated["x"][:, 0, 0]
    sim_y = simulated["y"][:, 0, 0]
    sim_z = simulated["z"][:, 0, 0]

    gt_x = ground_truth["x"][:, 0, 0]
    gt_y = ground_truth["y"][:, 0, 0]
    gt_z = ground_truth["z"][:, 0, 0]

    diffs = np.abs(sim_x - gt_x) + np.abs(sim_y - gt_y) + np.abs(sim_z - gt_z)

    if np.any(diffs > tolerance):
        return False
    return True


def align_trajectories_by_initial_position(simulated, ground_truth):
    """
    Idea: Even if simulated trajectories was obtained with a different codebase,
          the data is the same so we can realign it based on initial positions.

          Once it is aligned, we can use our codebase to score the trajectories.
          We have have consistent agent ids across both datasets, but since we do not have
          the scenario_ids, we cannot simply align this way.

          Plus it is nice to know that the things are really aligned by position.
    """
    print("\n--- Aligning the trajectories using initial positions ---")

    sim_pos_map = {}
    sim_x = np.round(simulated["x"][:, 0, 0], 5)  # Avoid floating point issues
    sim_y = np.round(simulated["y"][:, 0, 0], 5)
    sim_z = np.round(simulated["z"][:, 0, 0], 5)

    for i in range(len(sim_x)):
        key = (sim_x[i], sim_y[i], sim_z[i])
        assert key not in sim_pos_map, "Duplicate positions found in simulated trajectories, so cannot align."
        sim_pos_map[key] = i

    gt_x = np.round(ground_truth["x"][:, 0, 0], 5)
    gt_y = np.round(ground_truth["y"][:, 0, 0], 5)
    gt_z = np.round(ground_truth["z"][:, 0, 0], 5)

    # Need this one to debug if we have unmatched agents
    gt_ids = ground_truth["id"][:, 0]

    num_gt_agents = len(gt_x)
    reorder_indices = [-1] * num_gt_agents
    unmatched_count = 0

    for i in range(num_gt_agents):
        key = (gt_x[i], gt_y[i], gt_z[i])
        if key in sim_pos_map:
            reorder_indices[i] = sim_pos_map[key]

            # Additional safety to be sure the rounding did not create issues
            assert simulated["id"][reorder_indices[i], 0, 0] == ground_truth["id"][i, 0], (
                "The matching by position failed, is surely wrong"
            )
        else:
            unmatched_count += 1
            print(f"  ERROR: No match found for GT agent {i} (id={gt_ids[i]}) at position {key}")

    if unmatched_count > 0:
        raise ValueError(f"Failed to align {unmatched_count} agents by position.")

    # This might not be necessary, but to be sure let's also check reorder_indices has no -1 or duplicates
    assert -1 not in reorder_indices, "Some agents could not be matched."
    assert len(set(reorder_indices)) == len(reorder_indices), "Duplicate indices found in reorder_indices."

    reordered_sim = {}
    for key, val in simulated.items():
        reordered_sim[key] = val[reorder_indices]

    num_agents, num_rollouts, num_steps = reordered_sim["id"].shape
    gt_ids_reshaped = np.array(gt_ids, dtype=np.int32).reshape(num_agents, 1, 1)
    reordered_sim["id"] = np.broadcast_to(gt_ids_reshaped, (num_agents, num_rollouts, num_steps)).copy()

    print(f"Successfully aligned {num_gt_agents} trajectories")
    return reordered_sim


def combine_trajectories(ground_truth_trajectories, planning_trajectories):
    # In this case, you just identify the sdc and replace it, but I also want to
    # Add a safeguard

    sdc_mask = ground_truth_trajectories["track_id"][:, 0] <= -2

    from copy import deepcopy

    combine_trajectories = deepcopy(ground_truth_trajectories)

    for key in ["x", "y", "z", "heading"]:
        combine_trajectories[key][sdc_mask, :, :] = planning_trajectories[key][sdc_mask, :, :]

    # Mark the sdc id as -1 to identify them during evaluation
    combine_trajectories["id"][sdc_mask, 0] = -1
    ground_truth_trajectories["id"][sdc_mask, 0] = -1

    return combine_trajectories, ground_truth_trajectories


def evaluate_trajectories(simulated_trajectory_file, args):
    """
    Evaluates pre-computed simulated trajectories against live ground truth from the environment.
    """
    env_name = "puffer_drive"
    args["env"]["map_dir"] = args["eval"]["map_dir"]
    args["env"]["num_maps"] = args["eval"]["num_maps"]
    args["env"]["use_all_maps"] = True
    dataset_name = args["env"]["map_dir"].split("/")[-1]

    wosac_enabled = args["eval"]["wosac_realism_eval"]
    planning_enabled = args["eval"]["planning_eval"]

    if wosac_enabled:
        print(f"Running WOSAC realism evaluation with {dataset_name} dataset. \n")
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

        backend = args["eval"]["backend"]
        assert backend == "PufferEnv" or not wosac_enabled, "WOSAC evaluation only supports PufferEnv backend."
        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["num_agents"] = args["eval"]["wosac_num_agents"]
        args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
        args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
        args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]
        args["env"]["goal_behavior"] = args["eval"]["wosac_goal_behavior"]
        args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

        vecenv = pufferl.load_env(env_name, args)
        evaluator = WOSACEvaluator(args)

        # Collect ground truth trajectories from the dataset
        gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

        print(f"Number of scenarios: {len(np.unique(gt_trajectories['scenario_id']))}")
        print(f"Number of controlled agents: {gt_trajectories['x'].shape[0]}")
        print(f"Number of evaluated agents: {np.sum(gt_trajectories['track_id'] >= 0)}")

        print(f"Loading simulated trajectories from {simulated_trajectory_file}...")
        with open(simulated_trajectory_file, "rb") as f:
            simulated_trajectories_original = pickle.load(f)

        num_agents = gt_trajectories["x"].shape[0]
        sim_trajectories = {k: v[:num_agents] for k, v in simulated_trajectories_original.items()}

        assert check_alignment(sim_trajectories, gt_trajectories), "Code is broken lol"

        # simulated_trajectories_reordered = align_trajectories_by_initial_position(sim_trajectories, gt_trajectories)

        agent_state = vecenv.driver_env.get_global_agent_state()
        road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()

        print("\n--- Computing WOSAC Metrics ---")
        results = evaluator.compute_metrics(
            gt_trajectories,
            sim_trajectories,
            agent_state,
            road_edge_polylines,
            args["eval"]["wosac_aggregate_results"],
        )

        if args["eval"]["wosac_aggregate_results"]:
            import json

            print("\n--- WOSAC METRICS START ---")
            print(json.dumps(results, indent=4))
            print("--- WOSAC METRICS END ---")

        vecenv.close()
        return results

    elif planning_enabled:
        print(f"Running Planning evaluation with {dataset_name} dataset. \n")
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator, PlanningEvaluator

        backend = args["eval"]["backend"]
        assert backend == "PufferEnv", "Planning evaluation only supports PufferEnv backend."

        args["vec"] = dict(backend=backend, num_envs=1)
        args["eval"]["wosac_num_rollouts"] = 1
        args["env"]["control_mode"] = "control_sdc_only"
        args["env"]["init_steps"] = args["eval"]["planning_init_steps"]
        args["env"]["goal_behavior"] = args["eval"]["planning_goal_behavior"]
        args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

        vecenv = pufferl.load_env(env_name, args)

        # Step 0: collect gt trajectories
        wosac_evaluator = WOSACEvaluator(args)
        gt_args = args.copy()
        gt_args["env"]["control_mode"] = "control_wosac"
        gt_vecenv = pufferl.load_env(env_name, gt_args)
        gt_trajectories = wosac_evaluator.collect_ground_truth_trajectories(gt_vecenv)
        agent_state = gt_vecenv.driver_env.get_global_agent_state()
        road_edge_polylines = gt_vecenv.driver_env.get_road_edge_polylines()

        evaluator = PlanningEvaluator(args)

        with open(simulated_trajectory_file, "rb") as f:
            simulated_trajectories_original = pickle.load(f)

        num_agents = gt_trajectories["x"].shape[0]
        planning_trajectories = {k: v[:num_agents] for k, v in simulated_trajectories_original.items()}

        assert check_alignment(planning_trajectories, gt_trajectories), "Code is broken lol"
        # planning_trajectories_reordered = align_trajectories_by_initial_position(simulated_trajectories_original, gt_trajectories)

        # Combine them to direclty call the metrics computation
        combined_trajectories, gt_trajectories = combine_trajectories(gt_trajectories, planning_trajectories)

        results = evaluator.compute_metrics(
            combined_trajectories, agent_state, road_edge_polylines, args["eval"]["planning_aggregate_results"]
        )

        gt_results = evaluator.compute_metrics(
            gt_trajectories, agent_state, road_edge_polylines, args["eval"]["planning_aggregate_results"]
        )

        if args["eval"]["planning_aggregate_results"]:
            import json

            print("\nPLANNING_METRICS_START")
            print(json.dumps(results, indent=4))
            print("PLANNING_METRICS_END")

            print("\nPLANNING_GT_METRICS_START")
            print(json.dumps(gt_results, indent=4))
            print("PLANNING_GT_METRICS_END")


if __name__ == "__main__":
    import sys

    simulated_file = None
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

    config = pufferl.load_config("puffer_drive")

    evaluate_trajectories(simulated_file, args=config)
