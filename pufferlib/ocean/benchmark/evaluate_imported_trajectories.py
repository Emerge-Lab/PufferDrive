import pickle
import numpy as np
import pufferlib.pufferl as pufferl
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator


def align_trajectories_by_position(simulated, ground_truth):
    """
    Idea: Even if simulated trajectories was obtained with a different codebase,
          the data is the same so we can realign it based on initial positions.

          Once it is aligned, we can use our codebase to score the trajectories.
          We could use agent IDs and scenario IDs, but they are discarded in preprocessing,
          if this method doesn't work (ie. unmatched agents), we should consider modifying the code to keep this info.
    """
    print("\n--- Aligning the trajectories using initial positions ---")

    sim_pos_map = {}
    sim_x = simulated["x"][:, 0, 0]
    sim_y = simulated["y"][:, 0, 0]
    sim_z = simulated["z"][:, 0, 0]

    for i in range(len(sim_x)):
        key = (sim_x[i], sim_y[i], sim_z[i])
        assert key not in sim_pos_map, "Duplicate positions found in simulated trajectories, so cannot align."
        sim_pos_map[key] = i

    gt_x = ground_truth["x"][:, 0, 0]
    gt_y = ground_truth["y"][:, 0, 0]
    gt_z = ground_truth["z"][:, 0, 0]

    # Need this one to debug if we have unmatched agents
    gt_ids = ground_truth["id"][:, 0]

    num_gt_agents = len(gt_x)
    reorder_indices = [-1] * num_gt_agents
    unmatched_count = 0

    for i in range(num_gt_agents):
        key = (gt_x[i], gt_y[i], gt_z[i])
        if key in sim_pos_map:
            reorder_indices[i] = sim_pos_map[key]
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


def evaluate_trajectories(simulated_trajectory_file, args):
    """
    Evaluates pre-computed simulated trajectories against live ground truth from the environment.
    """
    env_name = "puffer_drive"
    args["env"]["map_dir"] = args["eval"]["map_dir"]
    args["env"]["num_maps"] = args["eval"]["num_maps"]
    args["env"]["use_all_maps"] = True
    dataset_name = args["env"]["map_dir"].split("/")[-1]
    backend = args["eval"]["backend"]
    assert backend == "PufferEnv", "WOSAC evaluation only supports PufferEnv backend."
    args["vec"] = dict(backend=backend, num_envs=1)
    args["env"]["num_agents"] = args["eval"]["wosac_num_agents"]
    args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
    args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
    args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]
    args["env"]["goal_behavior"] = args["eval"]["wosac_goal_behavior"]
    args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

    print(f"Running WOSAC realism evaluation with {dataset_name} dataset. \n")
    vecenv = pufferl.load_env(env_name, args)

    evaluator = WOSACEvaluator(args)

    # Collect ground truth trajectories from the dataset
    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    print(f"Number of scenarios: {len(np.unique(gt_trajectories['scenario_id']))}")
    print(f"Number of controlled agents: {gt_trajectories['x'].shape[0]}")
    print(f"Number of evaluated agents: {np.sum(gt_trajectories['id'] >= 0)}")

    print(f"Loading simulated trajectories from {simulated_trajectory_file}...")
    with open(simulated_trajectory_file, "rb") as f:
        simulated_trajectories_original = pickle.load(f)

    simulated_trajectories_reordered = align_trajectories_by_position(simulated_trajectories_original, gt_trajectories)

    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()

    print("\n--- Computing WOSAC Metrics ---")
    results = evaluator.compute_metrics(
        gt_trajectories,
        simulated_trajectories_reordered,
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
