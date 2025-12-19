import sys
import pickle
import numpy as np
from scipy.spatial import cKDTree
import pufferlib.pufferl as pufferl
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator


def align_trajectories_by_initial_position(simulated, ground_truth, tolerance=1e-4):
    """
    If the trajectories where generated using the same dataset, then regardless of the algorithm the initial positions should be the same.
    We use this information to align the trajectories for WOSAC evaluation.

    Ideally we would not have to use a tolerance, but the preprocessing in SMART shifts some values by around 2-e5 for some agents.

    Also, the preprocessing in SMART messes up some heading values, so I decided not to include heading.

    Idea of this script, use a nearest neighbor algorithm to associate all initial positions in gt to positions in simulated,
    and check that everyone matching respects the tolerance and there are no duplicates.
    """

    sim_pos = np.stack([simulated["x"][:, 0, 0], simulated["y"][:, 0, 0], simulated["z"][:, 0, 0]], axis=1).astype(
        np.float64
    )

    gt_pos = np.stack(
        [ground_truth["x"][:, 0, 0], ground_truth["y"][:, 0, 0], ground_truth["z"][:, 0, 0]], axis=1
    ).astype(np.float64)

    tree = cKDTree(sim_pos)

    dists, indices = tree.query(gt_pos, k=1)

    tol_check = dists <= tolerance

    if not np.all(tol_check):
        max_dist = np.max(dists)
        raise ValueError(f"Didn't find a match for {np.sum(~tol_check)} agents, tolerance broken by {max_dist}m.")

    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate matching found, I am sorry but this likely indicates that your data is wrong")

    reordered_sim = {}
    for key, val in simulated.items():
        reordered_sim[key] = val[indices]
    return reordered_sim


def check_alignment(simulated, ground_truth, tolerance=1e-4):
    # Check that initial positions match within a tolerance
    gt_x = ground_truth["x"][:, 0, 0]
    gt_y = ground_truth["y"][:, 0, 0]
    gt_z = ground_truth["z"][:, 0, 0]

    num_agents_gt = gt_x.shape[0]

    sim_x = simulated["x"][:num_agents_gt, 0, 0]
    sim_y = simulated["y"][:num_agents_gt, 0, 0]
    sim_z = simulated["z"][:num_agents_gt, 0, 0]

    diffs = np.maximum(np.maximum(np.abs(gt_x - sim_x), np.abs(gt_y - sim_y)), np.abs(gt_z - sim_z))

    if np.any(diffs > tolerance):
        print("Tolerance broken by this distance: ", np.max(diffs))
        return False
    return True


def check_consistent_alignment(simulated, ground_truth, tolerance=1e-4):
    # For planning evaluation, for agents other than the sdc, trajectories should be the same across timesteps.

    gt_x = ground_truth["x"].squeeze()
    gt_y = ground_truth["y"].squeeze()
    gt_z = ground_truth["z"].squeeze()

    is_ego = (ground_truth["id"] <= -2).squeeze()
    is_valid = ground_truth["valid"].squeeze()

    sim_x = simulated["x"].squeeze()
    sim_y = simulated["y"].squeeze()
    sim_z = simulated["z"].squeeze()

    diffs = np.maximum(np.maximum(np.abs(gt_x - sim_x), np.abs(gt_y - sim_y)), np.abs(gt_z - sim_z))

    diffs = np.where(is_ego[:, None] | ~is_valid, 0.0, diffs)
    max_diff = np.max(diffs)

    if max_diff < tolerance:
        return True
    else:
        print("There is a shift of at least: ", max_diff)
        return False


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

        backend = args["eval"]["backend"]
        assert backend == "PufferEnv", "WOSAC evaluation only supports PufferEnv backend."
        args["vec"] = dict(backend=backend, num_envs=1)

        args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
        args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
        args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]
        args["env"]["goal_behavior"] = args["eval"]["wosac_goal_behavior"]
        args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

        vecenv = pufferl.load_env(env_name, args)
        evaluator = WOSACEvaluator(args)

        # Collect ground truth trajectories from the dataset
        gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)
        num_agents_gt = gt_trajectories["x"].shape[0]

        print(f"Number of scenarios: {len(np.unique(gt_trajectories['scenario_id']))}")
        print(f"Number of controlled agents: {num_agents_gt}")
        print(f"Number of evaluated agents: {np.sum(gt_trajectories['id'] >= 0)}")

        print(f"Loading simulated trajectories from {simulated_trajectory_file}...")
        with open(simulated_trajectory_file, "rb") as f:
            sim_trajectories = pickle.load(f)

        if sim_trajectories["x"].shape[0] != gt_trajectories["x"].shape[0]:
            print("\nThe number of agents in simulated and ground truth trajectories do not match.")
            print("This is okay if you are running this script on a subset of the val dataset")
            print("But please also check that in drive.h MAX_AGENTS is set to 256 and recompile")

        if not check_alignment(sim_trajectories, gt_trajectories):
            print("\nTrajectories are not aligned, trying to align them, if it fails consider changing the tolerance.")
            sim_trajectories = align_trajectories_by_initial_position(sim_trajectories, gt_trajectories)
            assert check_alignment(sim_trajectories, gt_trajectories), (
                "There might be an issue with the way you generated your data."
            )
            print("Alignment successful")
        else:
            sim_trajectories = {k: v[:num_agents_gt] for k, v in sim_trajectories.items()}

        # Evaluator code expects to have matching ids between gt and sim trajectories
        # Since alignment is checked it is safe to do that
        sim_trajectories["id"][:] = gt_trajectories["id"][..., None]

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

            print("\n")
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

        wosac_evaluator = WOSACEvaluator(args)

        gt_args = args.copy()
        gt_args["env"]["control_mode"] = "control_wosac"
        gt_vecenv = pufferl.load_env(env_name, gt_args)
        gt_trajectories = wosac_evaluator.collect_ground_truth_trajectories(gt_vecenv)
        num_agents_gt = gt_trajectories["x"].shape[0]

        agent_state = gt_vecenv.driver_env.get_global_agent_state()
        road_edge_polylines = gt_vecenv.driver_env.get_road_edge_polylines()

        evaluator = PlanningEvaluator(args)

        with open(simulated_trajectory_file, "rb") as f:
            sim_trajectories = pickle.load(f)

        if sim_trajectories["x"].shape[0] != gt_trajectories["x"].shape[0]:
            print("\nThe number of agents in simulated and ground truth trajectories do not match.")
            print("This is okay if you are running this script on a subset of the val dataset")
            print("But please also check that in drive.h MAX_AGENTS is set to 256 and recompile")

        if not check_alignment(sim_trajectories, gt_trajectories):
            print("\nTrajectories are not aligned, trying to align them, if it fails consider changing the tolerance.")
            sim_trajectories = align_trajectories_by_initial_position(sim_trajectories, gt_trajectories)
            assert check_alignment(sim_trajectories, gt_trajectories), (
                "There might be an issue with the way you generated your data."
            )
            print("Alignment successful")
        else:
            sim_trajectories = {k: v[:num_agents_gt] for k, v in sim_trajectories.items()}

        assert check_consistent_alignment(sim_trajectories, gt_trajectories), (
            "You should check your data, you can increase tolerance or you should check that you used the same dataset"
        )

        # Safe to do that because we checked the alignment
        sim_trajectories["id"] = gt_trajectories["id"]
        sim_trajectories["valid"] = gt_trajectories["valid"]
        sim_trajectories["scenario_id"] = gt_trajectories["scenario_id"]

        results = evaluator.compute_metrics(
            sim_trajectories, agent_state, road_edge_polylines, args["eval"]["planning_aggregate_results"]
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
