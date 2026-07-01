from __future__ import annotations

import glob
import os
import numbers
import numpy as np
import pandas as pd
import time
import torch
from tqdm import tqdm
import yaml

import pufferlib
import pufferlib.viz
from pufferlib.ocean.drive import binding


def _reduce_environment_metrics(metric_lists, sum_keys=None):
    reduced = {}
    total_distance_travelled_sum = None
    total_infraction_count = None
    sum_keys = set() if sum_keys is None else set(sum_keys)

    for key, values in metric_lists.items():
        if not values or not isinstance(values[0], numbers.Number):
            continue
        if key == "total_distance_travelled_sum":
            total_distance_travelled_sum = float(np.sum(values))
        elif key == "total_infraction_count":
            total_infraction_count = float(np.sum(values))
        elif key in sum_keys:
            reduced[key] = float(np.sum(values))
        else:
            reduced[key] = float(np.mean(values))

    if total_distance_travelled_sum is not None and total_infraction_count is not None:
        reduced["total_infractions"] = total_infraction_count
        reduced["avg_distance_per_infraction"] = total_distance_travelled_sum / max(total_infraction_count, 1.0)

    return reduced


def _is_cloud():
    return bool(os.environ.get("CLOUD", False))


def _get_latest_checkpoint(exp_dir):
    checkpoints = glob.glob(os.path.join(exp_dir, "models", "*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
    return checkpoints[-1]


def _reset_rnn_state(args, policy, num_agents):
    """Initialize the LSTM state if necessary."""
    if not args["train"]["use_rnn"]:
        return {}
    device = args["train"]["device"]
    return {
        "lstm_h": torch.zeros(num_agents, policy.hidden_size, device=device),
        "lstm_c": torch.zeros(num_agents, policy.hidden_size, device=device),
    }


def build_eval_overrides(
    mode, num_agents, num_scenarios, map_dir, num_maps, scenario_length, max_agents, control_mode, reward_overrides=None
):
    """Build evaluation overrides - only for default reward gigaflow, terminations modes.

    reward_overrides: optional {env_param: value} dict applied last, e.g. {"reward_collision": 0.01}.
    With reward_conditioning=True these set the conditioning vector the policy observes at inference,
    so distinct presets exercise distinct conditioned behaviors without retraining.
    """
    env = {
        "eval_mode": 1,
        "compute_eval_metrics": 1,
        "collision_behavior": 1,
        "offroad_behavior": 1,
        "traffic_light_behavior": 0,
        "dt": 0.1,
        "use_neighbor_cache": 0,
        "obs_slots_partners_n": 20,
        "obs_slots_lane_n": 80,
        "obs_slots_boundary_n": 80,
        "reward_randomization": False,
        "reward_collision": 3.0,
        "reward_offroad": 3.0,
        "reward_stop_line": 1.0,
        "reward_ade": 0.0,
        "reward_goal": 1.0,
        "reward_overspeed": 0.05,
        "reward_comfort": 0.05,
        "reward_velocity": 0.0025,
        "reward_vel_align": 1.0,
        "reward_lane_align": 0.025,
        "reward_lane_center": 0.0038,
        "reward_timestep": 0.000025,
        "reward_reverse": 0.005,
        "goal_speed": 20.0,
        "goal_radius": 2.0,
        "obs_dropout_lane": 0.0,
        "obs_dropout_boundary": 0.0,
        "num_goals": 3,
        "goal_regen_mode": "rolling",
        "min_goal_spacing": 30.0,
        "max_goal_spacing": 30.0,
        "termination_mode": 0.0,
        "num_agents": num_agents,
        "control_mode": control_mode,
    }

    if mode == "gigaflow":
        if os.path.isdir(map_dir):
            num_maps = min(num_maps, len([n for n in os.listdir(map_dir) if n.endswith(".bin")]))
        env.update(
            {
                "simulation_mode": mode,
                "resample_frequency": scenario_length,
                "scenario_length": scenario_length,
                "max_agents_per_env": max_agents,
                "map_dir": map_dir,
                "num_maps": num_maps,
                "goal_source": "route",
            }
        )
    elif mode == "replay":
        env.update(
            {
                "simulation_mode": mode,
                "resample_frequency": scenario_length,
                "scenario_length": scenario_length,
                "max_agents_per_env": 200,
                "map_dir": map_dir,
                "num_maps": num_scenarios,
                "goal_source": "route",
                "use_neighbor_cache": 0,
            }
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if reward_overrides:
        env.update(reward_overrides)

    return {"env": env}


def _resolve_benchmark_context(env_name, args, logger, policy=None, create_dir=True):
    """Resolve model paths and evaluation directories."""
    data_dir = args["train"]["data_dir"]

    if logger:
        exp_dir = os.path.join(data_dir, f"{env_name}_{logger.run_id}")
        model_path = _get_latest_checkpoint(exp_dir)
    else:
        requested_model_path = args["load_model_path"]
        model_path = requested_model_path
        if not model_path:
            exp_dir = os.path.join(data_dir, env_name)
        else:
            exp_dir = os.path.dirname(os.path.dirname(model_path)) if model_path.endswith(".pt") else model_path
            model_path = model_path if model_path.endswith(".pt") else _get_latest_checkpoint(exp_dir)
            if not model_path:
                raise pufferlib.APIUsageError(f"Could not resolve a checkpoint from {requested_model_path}.")

    # When evaluating a shared, read-only checkpoint (load_model_path points at someone else's run),
    # all jobs would otherwise resolve the SAME bench_dir under that model's dir and clobber each other.
    # eval.output_dir redirects results to a per-job location (the synced cloud OUTPUT_DIR on GCP).
    output_base = args["eval"].get("output_dir") or exp_dir

    if not model_path and not policy:
        bench_dir = os.path.join(output_base, "final_evaluation", "expert")
    elif model_path:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        bench_dir = os.path.join(output_base, "final_evaluation", model_name)
    else:
        bench_dir = os.path.join(output_base, "final_evaluation", "final_policy")

    if create_dir:
        os.makedirs(bench_dir, exist_ok=True)
    return exp_dir, bench_dir, model_path


def _build_final_master_eval_suites(args):
    """Parse YAML catalog and filter by selected datasets."""
    catalog_path = args["eval"]["benchmark_config"]
    with open(catalog_path, "r") as f:
        catalog = yaml.safe_load(f) or {}

    # Get strict datasets (expecting list or string from CLI)
    selected = args["eval"]["benchmark_datasets"]
    if isinstance(selected, str):
        selected = [s.strip() for s in selected.split(",") if s.strip()]

    target = "gcp" if _is_cloud() else "local"
    suites = []

    for b in catalog["benchmarks"]:
        suite_id = str(b["name"]).strip()

        if selected and suite_id not in selected and b["name"] not in selected:
            continue

        map_dir = b["paths"][target]
        num_maps = b["num_maps"] if b["mode"] == "gigaflow" else b["num_scenarios"]
        max_agents = b["max_agents_per_env"] if b["mode"] == "gigaflow" else 64
        suites.append(
            {
                "suite_id": suite_id,
                "name": str(b["name"]).strip(),
                "simulation_mode": b["mode"],
                "map_dir": map_dir,
                "num_scenarios": b["num_scenarios"],
                "num_scenarios_to_render": b["num_scenarios_to_render"],
                "num_maps": num_maps,
                "scenario_length": b["scenario_length"],
                "max_agents_per_env": max_agents,
                "control_mode": b["control_mode"],
            }
        )
    return suites


def _merge_master_benchmark_summary(csv_path, new_entries):
    """Merge new summaries into the master CSV cleanly using Pandas."""
    df_new = pd.DataFrame(new_entries)
    if not df_new.empty and os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            df_new = pd.concat([df_old, df_new]).drop_duplicates(subset=["suite_id"], keep="last")
        except Exception:
            pass
    return df_new.sort_values("suite_id")


def _export_metrics(global_infos, eval_folder, num_scenarios, quiet, verify_coverage=False, simulation_mode="replay"):
    """Export metrics, maintaining expected row counts and generating averages."""
    df_ep = pd.DataFrame(global_infos)
    if not df_ep.empty:
        cols = ["episode_id", "map_name"] + [c for c in df_ep.columns if c not in ["episode_id", "map_name"]]
        df_ep = df_ep[cols].sort_values(["map_name", "episode_id"]) if verify_coverage else df_ep[cols]
        ep_csv = os.path.join(eval_folder, "episode_metrics.csv")
        df_ep.to_csv(ep_csv, index=False)

        if not quiet and verify_coverage:
            count = len(df_ep) if simulation_mode == "gigaflow" else df_ep["map_name"].nunique()
            unit = "episodes" if simulation_mode == "gigaflow" else "unique scenarios"
            print(f"✅ Exported {count}/{num_scenarios} {unit} to {ep_csv}")

    avg_infos = _reduce_environment_metrics(global_infos, sum_keys={"num_scenarios"})
    if avg_infos:
        df_sum = pd.DataFrame(list(avg_infos.items()), columns=["Metric", "Average"])
        sum_csv = os.path.join(eval_folder, "evaluation_summary.csv")
        df_sum.to_csv(sum_csv, index=False)
        if not quiet:
            print(f"✅ Averages exported to {sum_csv}\n", df_sum.to_string(index=False))

    return avg_infos


# ==============================================================================
# 2. CORE EVALUATION FUNCTIONS
# ==============================================================================
def evaluation_metrics(args, vecenv, policy, quiet=False):
    """Compute evaluation metrics (multi-worker)."""
    t0 = time.time()

    seed = args["train"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_scenarios = args["num_scenarios"]
    device = args["train"]["device"]
    eval_folder = args["eval_results_dir"]
    replay_expert_actions = args["env"].get("replay_expert_actions", False)

    if policy is not None:
        policy.eval()
    num_agents = vecenv.observation_space.shape[0]
    global_infos = {}
    scenarios_processed = 0

    vecenv.async_reset(seed)
    ob, _, _, _, infos, _, _ = vecenv.recv()

    with tqdm(total=num_scenarios, desc="Processing metrics", disable=quiet) as pbar:
        while scenarios_processed < num_scenarios:
            state = {} if replay_expert_actions else _reset_rnn_state(args, policy, num_agents)

            for _ in range(args["env"]["scenario_length"]):
                if replay_expert_actions:
                    action = np.zeros(vecenv.action_space.shape, dtype=getattr(vecenv.action_space, "dtype", np.int32))
                else:
                    with torch.no_grad():
                        ob_tensor = torch.as_tensor(ob).to(device)
                        logits, _ = policy.forward_eval(ob_tensor, state)
                        action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                        action = action.cpu().numpy().reshape(vecenv.action_space.shape)

                    if isinstance(logits, torch.distributions.Normal):
                        action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                ob, _, _, _, infos = vecenv.step(action)

                if infos and infos[0]:
                    for sub_env in infos:
                        for env_idx, summary in enumerate(sub_env):
                            summary.update(
                                {"episode_id": env_idx, "map_name": summary["map_name"].split("/")[-1].split(".")[0]}
                            )
                            scenarios_processed += 1
                            pbar.update(1)
                            for k, v in summary.items():
                                global_infos.setdefault(k, []).append(v)

    avg_infos = _export_metrics(
        global_infos,
        eval_folder,
        num_scenarios,
        quiet,
        verify_coverage=True,
        simulation_mode=args["env"]["simulation_mode"],
    )

    if not quiet:
        print(f"\nTotal metric eval time: {time.time() - t0:.2f} s for {num_scenarios} scenarios.")

    return avg_infos


def evaluation_render(args, vecenv, policy, quiet=False, dump_metrics=False, only_failures=False):
    """Generate interactive HTML replays (serial) via the compact C frame grabber.

    only_failures: write a replay only for scenarios that reported an infraction
    (collision/offroad/red-light). The rendered scenario is exactly the detected one — detection
    and rendering share this single serial pass, so the RNG-derived scenario matches.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    replay_expert_actions = args["env"].get("replay_expert_actions", False)
    if policy is not None:
        policy.eval()
    num_agents = vecenv.observation_space.shape[0]
    device = args["train"]["device"]
    env_args = args["env"]

    eval_folder = args["eval_results_dir"]
    gif_folder = os.path.join(eval_folder, "failures" if only_failures else "gif")
    os.makedirs(gif_folder, exist_ok=True)

    drive = vecenv.envs[0]
    af, ai, mf, sf, tf = (
        binding.AGENT_F32_FIELDS,
        binding.AGENT_I32_FIELDS,
        binding.METRICS_F32_FIELDS,
        binding.SCORE_F32_FIELDS,
        binding.TRAFFIC_I16_FIELDS,
    )
    chunk_keys = (
        "agent_f32",
        "agent_i32",
        "metrics_f32",
        "puffer_f32",
        "traffic_i16",
        "obs",
        "raw_action",
        "clipped_action",
        "value",
        "entropy",
    )
    pool_keys = ("pool_lane", "pool_boundary", "pool_partner", "pool_traffic")
    policy_keys = ("policy_probs", "policy_mean", "policy_std", "policy_log_prob")

    global_infos = {}
    num_scenarios = args["num_scenarios"]
    scenarios_processed = 0

    with tqdm(total=num_scenarios, desc="Rendering scenarios", disable=quiet) as pbar:
        while scenarios_processed < num_scenarios:
            ob, _ = vecenv.reset()
            scenarios = vecenv.get_state()
            num_envs = len(scenarios)
            batch_start = scenarios_processed

            vecenv.envs[0].batch_size_eval = max(1, num_scenarios - scenarios_processed - num_envs)
            map_names = [s["map_name"].split("/")[-1].split(".")[0] for s in scenarios]
            active_counts = [s["active_agent_count"] for s in scenarios]
            agent_cap = max((s["num_total_agents"] for s in scenarios), default=1)
            traffic_cap = max((s["num_traffic_elements"] for s in scenarios), default=1)
            state = {} if replay_expert_actions else _reset_rnn_state(args, policy, num_agents)

            hist = {k: [[] for _ in range(num_envs)] for k in chunk_keys + pool_keys + policy_keys}
            failed_envs = {}

            for t in range(env_args["scenario_length"]):
                agent_f32 = np.zeros((num_envs, agent_cap, af), dtype=np.float32)
                agent_i32 = np.zeros((num_envs, agent_cap, ai), dtype=np.int32)
                metrics_f32 = np.zeros((num_envs, agent_cap, mf), dtype=np.float32)
                puffer_f32 = np.zeros((num_envs, agent_cap, sf), dtype=np.float32)
                traffic_i16 = np.zeros((num_envs, traffic_cap, tf), dtype=np.int16)
                binding.vec_get_obs_html_frame(drive.c_envs, agent_f32, agent_i32, metrics_f32, puffer_f32, traffic_i16)

                pools, policy_extras = {}, {}
                if replay_expert_actions:
                    action = np.zeros(vecenv.action_space.shape, dtype=getattr(vecenv.action_space, "dtype", np.int32))
                    raw_action = action
                    value = np.zeros(num_agents, dtype=np.float32)
                    entropy = np.zeros(num_agents, dtype=np.float32)
                else:
                    with torch.no_grad():
                        ob_tensor = torch.as_tensor(ob).to(device)
                        logits, value_t = policy.forward_eval(ob_tensor, state)
                        action, logprob_t, entropy_t = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                        action = action.cpu().numpy().reshape(vecenv.action_space.shape)
                        # Discrete heads come back as a tuple of tensors (torch.split in forward_eval).
                        discrete_logits = logits if isinstance(logits, torch.Tensor) else None
                        if discrete_logits is None and isinstance(logits, (list, tuple)) and len(logits) == 1:
                            discrete_logits = logits[0]
                        if discrete_logits is not None:
                            policy_extras["policy_probs"] = (
                                torch.softmax(discrete_logits, dim=-1).cpu().numpy().astype(np.float32)
                            )
                        elif isinstance(logits, torch.distributions.Normal):
                            policy_extras["policy_mean"] = logits.loc.cpu().numpy().astype(np.float32)
                            policy_extras["policy_std"] = logits.scale.cpu().numpy().astype(np.float32)
                            policy_extras["policy_log_prob"] = logprob_t.detach().cpu().numpy().astype(np.float32)
                        pools = {
                            k: v.cpu().numpy().astype(np.int16) for k, v in policy.pool_slot_counts(ob_tensor).items()
                        }

                    raw_action = np.array(action, copy=True)
                    if isinstance(logits, torch.distributions.Normal):
                        action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)
                    value = value_t.detach().flatten().cpu().numpy()
                    entropy = entropy_t.detach().flatten().cpu().numpy()

                cursor = 0
                for idx in range(num_envs):
                    end = cursor + active_counts[idx]
                    hist["agent_f32"][idx].append(agent_f32[idx])
                    hist["agent_i32"][idx].append(agent_i32[idx])
                    hist["metrics_f32"][idx].append(metrics_f32[idx])
                    hist["puffer_f32"][idx].append(puffer_f32[idx])
                    hist["traffic_i16"][idx].append(traffic_i16[idx])
                    hist["obs"][idx].append(np.array(ob[cursor:end], dtype=np.float32, copy=True))
                    hist["raw_action"][idx].append(
                        np.asarray(raw_action[cursor:end], dtype=np.float32).reshape(active_counts[idx], -1)
                    )
                    hist["clipped_action"][idx].append(
                        np.asarray(action[cursor:end], dtype=np.float32).reshape(active_counts[idx], -1)
                    )
                    hist["value"][idx].append(np.asarray(value[cursor:end], dtype=np.float32))
                    hist["entropy"][idx].append(np.asarray(entropy[cursor:end], dtype=np.float32))
                    for k in pool_keys:
                        if k in pools:
                            hist[k][idx].append(pools[k][cursor:end])
                    for k in policy_keys:
                        if k in policy_extras:
                            hist[k][idx].append(policy_extras[k][cursor:end])
                    cursor = end

                ob, _, _, _, infos = vecenv.step(action)

                if infos and infos[0]:
                    for env_idx, summary in enumerate(infos[0]):
                        summary.update(
                            {
                                "episode_id": batch_start + env_idx,
                                "env_id": env_idx,
                                "map_name": summary["map_name"].split("/")[-1].split(".")[0],
                            }
                        )
                        failed_envs[env_idx] = (
                            failed_envs.get(env_idx, False) or summary.get("total_infraction_count", 0) > 0
                        )
                        for k, v in summary.items():
                            global_infos.setdefault(k, []).append(v)

            # Generate HTML Replays
            for env_idx in range(num_envs):
                global_ep_id = batch_start + env_idx
                if global_ep_id >= num_scenarios:
                    break

                if only_failures and not failed_envs.get(env_idx):
                    continue

                replay = {"env": env_args, "eval_overrides": args.get("eval_env_overrides", {})}
                replay.update({k: np.stack(hist[k][env_idx]) for k in chunk_keys})
                replay.update({k: np.stack(hist[k][env_idx]) for k in pool_keys + policy_keys if hist[k][env_idx]})
                pufferlib.viz.generate_interactive_replay(
                    scenarios[env_idx],
                    replay,
                    os.path.join(gif_folder, f"{map_names[env_idx]}_{global_ep_id:03d}.html"),
                )

            scenarios_processed += num_envs
            pbar.update(num_envs)

    pufferlib.viz.build_gallery_index(gif_folder)

    if dump_metrics:
        return _export_metrics(global_infos, eval_folder, num_scenarios, quiet, verify_coverage=False)
