import pickle
import numpy as np
import pufferlib.pufferl as pufferl
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator


def align_trajectories_by_position(simulated, ground_truth):
    """
    Aligns simulated trajectories to match the agent order in ground_truth based on initial positions.
    """
    print("\n--- Aligning Simulated Trajectories to Ground Truth by Position ---")

    def get_initial_values(array):
        if array.ndim == 3:
            return array[:, 0, 0]
        if array.ndim == 2:
            return array[:, 0]
        return array

    sim_pos_map = {}
    sim_x = get_initial_values(simulated["x"])
    sim_y = get_initial_values(simulated["y"])
    sim_z = get_initial_values(simulated["z"])

    for i in range(len(sim_x)):
        key = (round(sim_x[i], 3), round(sim_y[i], 3), round(sim_z[i], 3))
        if key not in sim_pos_map:
            sim_pos_map[key] = []
        sim_pos_map[key].append(i)

    gt_x = get_initial_values(ground_truth["x"])
    gt_y = get_initial_values(ground_truth["y"])
    gt_z = get_initial_values(ground_truth["z"])
    gt_ids = get_initial_values(ground_truth["id"])

    num_gt_agents = len(gt_x)
    reorder_indices = [-1] * num_gt_agents
    src_indices_used = set()
    unmatched_count = 0

    for i in range(num_gt_agents):
        key = (round(gt_x[i], 3), round(gt_y[i], 3), round(gt_z[i], 3))
        if key in sim_pos_map:
            potential_matches = [idx for idx in sim_pos_map[key] if idx not in src_indices_used]
            if potential_matches:
                match_idx = potential_matches[0]
                reorder_indices[i] = match_idx
                src_indices_used.add(match_idx)
            else:
                unmatched_count += 1
                print(f"  ERROR: No unused match for GT agent {i} (id={gt_ids[i]}) at position {key}")
        else:
            unmatched_count += 1
            print(f"  ERROR: No match found for GT agent {i} (id={gt_ids[i]}) at position {key}")

    if unmatched_count > 0:
        raise ValueError(f"Failed to align {unmatched_count} agents by position.")

    reordered_sim = {}
    for key, val in simulated.items():
        reordered_val = val[reorder_indices]
        reordered_sim[key] = reordered_val

    num_agents, num_rollouts, num_steps = reordered_sim["id"].shape
    gt_ids_reshaped = np.array(gt_ids, dtype=np.int32).reshape(num_agents, 1, 1)
    reordered_sim["id"] = np.broadcast_to(gt_ids_reshaped, (num_agents, num_rollouts, num_steps)).copy()

    print(f"✓ Successfully aligned {num_gt_agents} trajectories by position")
    return reordered_sim


def evaluate_trajectories(simulated_trajectory_file, args):
    """
    Evaluates pre-computed simulated trajectories against live ground truth from the environment.
    """
    env_name = "puffer_drive"

    wosac_enabled = args["eval"]["wosac_realism_eval"]
    if not wosac_enabled:
        print("wosac_realism_eval is not enabled in the config. Aborting.")
        return

    print("Running WOSAC realism evaluation with offline simulated trajectories...")

    backend = args["eval"]["backend"]
    assert backend == "PufferEnv", "WOSAC evaluation only supports PufferEnv backend."
    args["vec"] = dict(backend=backend, num_envs=1)
    args["env"]["num_agents"] = args["eval"]["wosac_num_agents"]
    args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
    args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
    args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]

    if args["eval"].get("wosac_num_scenarios") is not None:
        args["env"]["num_scenarios"] = args["eval"]["wosac_num_scenarios"]
    if args["eval"].get("wosac_map_seed") is not None:
        args["env"]["scenario_seed"] = args["eval"]["wosac_map_seed"]

    print("Instantiating environment...")
    vecenv = pufferl.load_env(env_name, args)

    evaluator = WOSACEvaluator(args)

    print("Collecting ground truth trajectories from the environment...")
    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    print(f"Loading simulated trajectories from {simulated_trajectory_file}...")
    with open(simulated_trajectory_file, "rb") as f:
        simulated_trajectories_original = pickle.load(f)

    simulated_trajectories_reordered = align_trajectories_by_position(simulated_trajectories_original, gt_trajectories)

    print(f"Collected trajectories on {len(np.unique(gt_trajectories['scenario_id']))} scenarios.")

    print("Getting agent state and road polylines from the environment...")
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

    env_name = "puffer_drive"

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

    print(f"env_name: {env_name}")
    print(f"simulated_file: {simulated_file}")

    config = pufferl.load_config(env_name)

    evaluate_trajectories(simulated_file, args=config)
