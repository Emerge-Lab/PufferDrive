import sys
import pickle
import numpy as np
import pufferlib.pufferl as pufferl
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator


def align_trajectories(simulated, ground_truth):
    # Idea is to use the (scenario_id, id) pair to reindex simulated_trajectories in order to align it with GT
    gt_scenario_ids = ground_truth["scenario_id"][:, 0]
    sim_scenario_ids = simulated["scenario_id"][:, 0, 0]

    gt_ids = ground_truth["id"][:, 0]
    sim_ids = simulated["id"][:, 0, 0]

    lookup = {(s_id, a_id): idx for idx, (s_id, a_id) in enumerate(zip(sim_scenario_ids, sim_ids))}

    try:
        indices = [lookup[(s, i)] for (s, i) in zip(gt_scenario_ids, gt_ids)]
        indices = np.array(indices, dtype=int)
    except KeyError:
        print("An agent present in the GT is missing in your simulation")
        raise

    sim_traj = {k: v[indices] for k, v in simulated.items()}

    return sim_traj


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


def evaluate_trajectories(simulated_trajectory_file, args):
    """
    Evaluates pre-computed simulated trajectories against live ground truth from the environment.
    """
    env_name = "puffer_drive"
    args["env"]["map_dir"] = args["eval"]["map_dir"]
    args["env"]["num_maps"] = args["eval"]["wosac_num_maps"]
    args["env"]["sequential_map_sampling"] = True
    dataset_name = args["env"]["map_dir"].split("/")[-1]

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
    print(f"Number of evaluated agents: {gt_trajectories['is_track_to_predict'].sum()}")

    print(f"Loading simulated trajectories from {simulated_trajectory_file}...")
    with open(simulated_trajectory_file, "rb") as f:
        sim_trajectories = pickle.load(f)

    num_agents_sim = sim_trajectories["x"].shape[0]
    assert num_agents_sim >= num_agents_gt, (
        "There is less agents in your simulation than in the GT, so the computation won't be valid"
    )

    if num_agents_sim > num_agents_gt:
        print("If you are evaluating on a subset of your trajectories it is fine.")
        print("\n Else, you should consider changing the value of MAX_AGENTS in drive.h and compile")

    sim_trajectories = align_trajectories(sim_trajectories, gt_trajectories)

    assert check_alignment(sim_trajectories, gt_trajectories), (
        "There might be an issue with the way you generated your data."
    )

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
