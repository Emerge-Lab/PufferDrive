import os
import sys
import glob
import shutil
import subprocess
import json
import configparser
import tempfile


def _normalize_device(device):
    """Convert device to a string suitable for torch.load(map_location=...)."""
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def _get_env_reward_bound_names(ini_path="pufferlib/config/ocean/drive.ini"):
    """Discover valid reward bound names from the env config section."""
    import re

    config = configparser.ConfigParser()
    config.read(ini_path)
    bounds = set()
    for key in config["env"]:
        m = re.match(r"reward_bound_(.+)_min$", key)
        if m:
            bounds.add(m.group(1))
    return bounds


def _run_eval_subprocess(
    config, logger, global_step, mode, extra_args, marker_name, wandb_keys=None, results_queue=None
):
    """Run an evaluation subprocess and log metrics to wandb.

    Args:
        config: Training config dict (must have data_dir, env)
        logger: Logger with run_id and optional wandb attribute
        global_step: Current global training step
        mode: pufferl mode to run (e.g. "eval", "safe_eval")
        extra_args: List of extra CLI args appended to the base command
        marker_name: Marker prefix for JSON extraction (e.g. "WOSAC" looks for WOSAC_METRICS_START/END)
        wandb_keys: If dict, maps metric keys to wandb keys. If None, logs all as eval/<key>.
        results_queue: If provided, put results on this queue instead of logging directly.
    """
    eval_name = marker_name.lower().replace("_", " ")
    run_id = logger.run_id
    model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")
    model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

    if not model_files:
        print(f"No model files found for {eval_name} evaluation")
        return

    latest_cpt = max(model_files)

    cmd = [
        sys.executable,
        "-m",
        "pufferlib.pufferl",
        mode,
        config["env"],
        "--load-model-path",
        latest_cpt,
        "--train.device",
        _normalize_device(config.get("device", "cuda")),
    ]

    cmd += extra_args

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

    start_marker = f"{marker_name}_METRICS_START"
    end_marker = f"{marker_name}_METRICS_END"

    if result.returncode == 0:
        stdout = result.stdout
        has_markers = start_marker in stdout and end_marker in stdout
        if has_markers:
            start = stdout.find(start_marker) + len(start_marker)
            end = stdout.find(end_marker)
            metrics = json.loads(stdout[start:end].strip())

            if hasattr(logger, "wandb") and logger.wandb:
                if wandb_keys is not None:
                    payload = {wandb_keys[k]: metrics[k] for k in wandb_keys if k in metrics}
                else:
                    payload = {f"eval/{k}": v for k, v in metrics.items()}
                if payload:
                    payload["train_step"] = global_step
                    if results_queue is not None:
                        results_queue.put(payload)
                    else:
                        logger.wandb.log(payload)
    else:
        print(f"{eval_name} evaluation failed with exit code {result.returncode}: {result.stderr[-1000:]}")


def run_human_replay_eval_in_subprocess(config, logger, global_step, results_queue=None):
    eval_config = config.get("eval", {})
    _run_eval_subprocess(
        config,
        logger,
        global_step,
        mode="eval",
        extra_args=[
            "--eval.wosac-realism-eval",
            "False",
            "--eval.human-replay-eval",
            "True",
            "--eval.human-replay-num-agents",
            str(eval_config.get("human_replay_num_agents", 16)),
            "--eval.human-replay-control-mode",
            str(eval_config.get("human_replay_control_mode", "control_sdc_only")),
            "--eval.map-dir",
            str(eval_config.get("map_dir", "resources/drive/binaries/training")),
            "--env.num-maps",
            str(eval_config.get("num_maps", 20)),
        ],
        marker_name="HUMAN_REPLAY",
        wandb_keys={
            "collision_rate": "human_replay/collision_rate",
            "offroad_rate": "human_replay/offroad_rate",
            "completion_rate": "human_replay/completion_rate",
        },
        results_queue=results_queue,
    )


def run_wosac_eval_in_subprocess(config, logger, global_step, results_queue=None):
    eval_config = config.get("eval", {})
    _run_eval_subprocess(
        config,
        logger,
        global_step,
        mode="eval",
        extra_args=[
            "--eval.wosac-realism-eval",
            "True",
            "--eval.wosac-batch-size",
            str(eval_config.get("wosac_batch_size", 32)),
            "--eval.wosac-target-scenarios",
            str(eval_config.get("wosac_target_scenarios", 64)),
            "--eval.wosac-scenario-pool-size",
            str(eval_config.get("wosac_scenario_pool_size", 10_000)),
            "--eval.wosac-init-mode",
            str(eval_config.get("wosac_init_mode", "create_all_valid")),
            "--eval.wosac-control-mode",
            str(eval_config.get("wosac_control_mode", "control_wosac")),
            "--eval.wosac-init-steps",
            str(eval_config.get("wosac_init_steps", 10)),
            "--eval.wosac-goal-behavior",
            str(eval_config.get("wosac_goal_behavior", 2)),
            "--eval.wosac-goal-radius",
            str(eval_config.get("wosac_goal_radius", 2.0)),
            "--eval.wosac-sanity-check",
            str(eval_config.get("wosac_sanity_check", False)),
            "--eval.wosac-aggregate-results",
            str(eval_config.get("wosac_aggregate_results", True)),
            "--eval.wosac-eval-mode",
            str(eval_config.get("wosac_eval_mode", "policy")),
            "--env.episode-length",
            str(eval_config.get("wosac_episode_length", 91)),
            "--eval.map-dir",
            str(eval_config.get("map_dir", "resources/drive/binaries/training")),
        ],
        marker_name="WOSAC",
        wandb_keys={
            "realism_meta_score": "wosac/realism_meta_score",
            "realism_meta_score_std": "wosac/realism_meta_score_std",
            "kinematic_metrics": "wosac/kinematic_metrics",
            "interactive_metrics": "wosac/interactive_metrics",
            "map_based_metrics": "wosac/map_based_metrics",
            "ade": "wosac/ade",
            "min_ade": "wosac/min_ade",
            "total_num_agents": "wosac/total_num_agents",
        },
        results_queue=results_queue,
    )


def render_videos(
    config,
    env_cfg,
    run_id,
    wandb_log,
    epoch,
    global_step,
    bin_path,
    render_async,
    render_queue=None,
    wandb_run=None,
    config_path=None,
    wandb_prefix="render",
):
    """
    Generate and log training videos using C-based rendering.

    Args:
        config: Configuration dictionary containing data_dir, env, and render settings
        env_cfg: Environment config object (driver_env) with map_dir, num_maps, etc.
        run_id: Wandb/Neptune run identifier
        wandb_log: Whether to log videos to wandb
        epoch: Current training epoch
        global_step: Current global training step
        bin_path: Path to the exported .bin model weights file
        render_async: Whether rendering is async (uses render_queue)
        render_queue: Queue for async render results
        wandb_run: Wandb run object for sync logging
        config_path: Optional path to alternative INI config file for the visualize binary
        wandb_prefix: Prefix for wandb keys (e.g. "render" or "eval")
    """
    if not os.path.exists(bin_path):
        print(f"Binary weights file does not exist: {bin_path}")
        return

    model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")

    video_output_dir = os.path.join(model_dir, "videos")
    os.makedirs(video_output_dir, exist_ok=True)

    # TODO: Fix memory leaks so that this is not needed
    env_vars = os.environ.copy()
    env_vars["ASAN_OPTIONS"] = "exitcode=0"

    base_cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", "./visualize"]

    if config_path:
        base_cmd.extend(["--config", config_path])

    if config.get("show_grid", False):
        base_cmd.append("--show-grid")
    if config.get("obs_only", False):
        base_cmd.append("--obs-only")
    if config.get("show_lasers", False):
        base_cmd.append("--lasers")
    if config.get("show_human_logs", False):
        base_cmd.append("--show-human-logs")
    if config.get("zoom_in", False):
        base_cmd.append("--zoom-in")

    frame_skip = config.get("frame_skip", 1)
    if frame_skip > 1:
        base_cmd.extend(["--frame-skip", str(frame_skip)])

    view_mode = config.get("view_mode", "both")
    base_cmd.extend(["--view", view_mode])

    if env_cfg is not None and getattr(env_cfg, "num_maps", None):
        base_cmd.extend(["--num-maps", str(env_cfg.num_maps)])

    base_cmd.extend(["--policy-name", bin_path])

    # Handle single or multiple map rendering
    render_maps = config.get("render_map", None)
    if render_maps is None or render_maps == "none":
        map_dir = None
        if env_cfg is not None and hasattr(env_cfg, "map_dir"):
            map_dir = env_cfg.map_dir
        if map_dir and os.path.isdir(map_dir):
            import random

            bin_files = [f for f in os.listdir(map_dir) if f.endswith(".bin")]
            if bin_files:
                render_maps = [os.path.join(map_dir, random.choice(bin_files))]
            else:
                print(f"Warning: No .bin files found in {map_dir}, skipping render")
                return
        else:
            print(f"Warning: map_dir not found or invalid ({map_dir}), skipping render")
            return
    elif isinstance(render_maps, (str, os.PathLike)):
        render_maps = [render_maps]
    else:
        render_maps = list(render_maps)

    file_prefix = f"{wandb_prefix}_" if wandb_prefix != "render" else ""
    videos_to_log_world = []
    videos_to_log_agent = []
    generated_videos = {"output_topdown": [], "output_agent": []}
    output_topdown = f"resources/drive/{file_prefix}output_topdown_{epoch}"
    output_agent = f"resources/drive/{file_prefix}output_agent_{epoch}"

    for i, map_path in enumerate(render_maps):
        cmd = list(base_cmd)
        if map_path is not None and os.path.exists(map_path):
            cmd.extend(["--map-name", str(map_path)])

        output_topdown_map = output_topdown + (f"_map{i:02d}.mp4" if len(render_maps) > 1 else ".mp4")
        output_agent_map = output_agent + (f"_map{i:02d}.mp4" if len(render_maps) > 1 else ".mp4")

        cmd.extend(["--output-topdown", output_topdown_map])
        cmd.extend(["--output-agent", output_agent_map])

        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=1200, env=env_vars)

        vids_exist = os.path.exists(output_topdown_map) and os.path.exists(output_agent_map)

        if result.returncode == 0 or (result.returncode == 1 and vids_exist):
            videos = [
                (
                    "output_topdown",
                    output_topdown_map,
                    f"{file_prefix}epoch_{epoch:06d}_map{i:02d}_topdown.mp4"
                    if map_path
                    else f"{file_prefix}epoch_{epoch:06d}_topdown.mp4",
                ),
                (
                    "output_agent",
                    output_agent_map,
                    f"{file_prefix}epoch_{epoch:06d}_map{i:02d}_agent.mp4"
                    if map_path
                    else f"{file_prefix}epoch_{epoch:06d}_agent.mp4",
                ),
            ]

            for vid_type, source_vid, target_filename in videos:
                if os.path.exists(source_vid):
                    target_path = os.path.join(video_output_dir, target_filename)
                    shutil.move(source_vid, target_path)
                    generated_videos[vid_type].append(target_path)
                    if render_async:
                        continue
                    if wandb_log:
                        import wandb

                        if "topdown" in target_filename:
                            videos_to_log_world.append(wandb.Video(target_path, format="mp4"))
                        else:
                            videos_to_log_agent.append(wandb.Video(target_path, format="mp4"))
                else:
                    print(f"Video generation completed but {source_vid} not found")
                    if result.stdout:
                        print(f"StdOUT: {result.stdout}")
                    if result.stderr:
                        print(f"StdERR: {result.stderr}")
        else:
            print(f"C rendering failed (map index {i}) with exit code {result.returncode}: {result.stdout}")

    if render_async:
        render_queue.put(
            {
                "videos": generated_videos,
                "step": global_step,
                "wandb_prefix": wandb_prefix,
                "bin_path": bin_path,
                "config_path": config_path,
            }
        )

    if wandb_log and (videos_to_log_world or videos_to_log_agent) and not render_async:
        payload = {}
        if videos_to_log_world:
            payload[f"{wandb_prefix}/world_state"] = videos_to_log_world
        if videos_to_log_agent:
            payload[f"{wandb_prefix}/agent_view"] = videos_to_log_agent
        payload["train_step"] = global_step
        wandb_run.log(payload)


def generate_safe_eval_ini(safe_eval_config, base_ini_path="pufferlib/config/ocean/drive.ini"):
    """Generate a temporary ini file with safe/law-abiding reward conditioning values.

    Sets reward_randomization=1 with min=max bounds so the conditioning values
    are deterministically set to the safe values the policy sees in its observation.
    """
    config = configparser.ConfigParser()
    config.read(base_ini_path)

    valid_bounds = _get_env_reward_bound_names(base_ini_path)
    for key, val in safe_eval_config.items():
        if key not in valid_bounds:
            continue
        val = str(val)
        config.set("env", f"reward_bound_{key}_min", val)
        config.set("env", f"reward_bound_{key}_max", val)

    config.set("env", "reward_randomization", "1")
    config.set("env", "reward_conditioning", "1")

    # Match the metrics subprocess setup so the render shows the same behavior
    config.set("env", "episode_length", str(safe_eval_config.get("episode_length", 1000)))
    config.set("env", "resample_frequency", "0")
    config.set("env", "num_agents", str(safe_eval_config.get("num_agents", 64)))
    config.set("env", "min_goal_distance", str(safe_eval_config.get("min_goal_distance", 0.5)))
    config.set("env", "max_goal_distance", str(safe_eval_config.get("max_goal_distance", 1000.0)))

    fd, tmp_path = tempfile.mkstemp(suffix=".ini", prefix="safe_eval_")
    with os.fdopen(fd, "w") as f:
        config.write(f)

    return tmp_path


def generate_human_replay_ini(eval_config, base_ini_path="pufferlib/config/ocean/drive.ini"):
    """Generate a temporary ini file for human replay rendering.

    Sets control_mode to control_sdc_only so only the SDC is policy-controlled,
    with all other agents replaying logged trajectories.
    """
    config = configparser.ConfigParser()
    config.read(base_ini_path)

    config.set("env", "control_mode", '"control_sdc_only"')
    config.set("env", "init_mode", '"create_all_valid"')
    config.set("env", "init_steps", "10")
    # Use eval map_dir (waymo maps), not training map_dir
    map_dir = eval_config.get("map_dir", "resources/drive/binaries/training")
    config.set("env", "map_dir", f'"{map_dir}"')

    fd, tmp_path = tempfile.mkstemp(suffix=".ini", prefix="human_replay_")
    with os.fdopen(fd, "w") as f:
        config.write(f)

    return tmp_path


def run_safe_eval_metrics_in_subprocess(config, logger, global_step, safe_eval_config, results_queue=None):
    """Run policy evaluation with safe reward conditioning in a subprocess and log metrics."""
    num_episodes = safe_eval_config.get("num_episodes", 300)

    # Forward training env's map_dir and num_maps so the subprocess uses the
    # same maps as training (the default INI may point elsewhere).
    env_config = config.get("env_config", {})
    extra_args = [
        "--env.reward-randomization",
        "1",
        "--env.reward-conditioning",
        "1",
        "--safe-eval.num-episodes",
        str(num_episodes),
        "--safe-eval.num-agents",
        str(safe_eval_config.get("num_agents", 64)),
        f"--env.map-dir={env_config.get('map_dir', 'resources/drive/binaries/training')}",
        f"--env.num-maps={env_config.get('num_maps', 100)}",
    ]

    # Pass safe_eval overrides that safe_eval() applies to env config
    safe_eval_overrides = ["episode_length", "min_goal_distance", "max_goal_distance"]
    for key in safe_eval_overrides:
        if key in safe_eval_config:
            cli_key = key.replace("_", "-")
            extra_args.extend([f"--safe-eval.{cli_key}", str(safe_eval_config[key])])

    valid_bounds = _get_env_reward_bound_names()
    for key, val in safe_eval_config.items():
        if key not in valid_bounds:
            continue
        val = str(val)
        cli_name = key.replace("_", "-")
        # Use = syntax to avoid argparse interpreting negative values as flags
        extra_args.extend([f"--env.reward-bound-{cli_name}-min={val}", f"--env.reward-bound-{cli_name}-max={val}"])

    _run_eval_subprocess(
        config,
        logger,
        global_step,
        mode="safe_eval",
        extra_args=extra_args,
        marker_name="SAFE_EVAL",
        results_queue=results_queue,
    )
