import configparser
import os
import sys
import glob
import shutil
import subprocess
import json
import random
import tempfile


def generate_env_ini(env_config, base_ini_path="pufferlib/config/ocean/drive.ini", prefix="env_"):
    """Generate a temporary INI file with env config overrides applied.

    The visualize binary and other tools read config from an INI file,
    not CLI args. This ensures they use the same settings as training.
    """
    config = configparser.ConfigParser()
    read_files = config.read(base_ini_path)
    if not read_files:
        raise FileNotFoundError(f"Could not read base INI file: {base_ini_path}")
    if not config.has_section("env"):
        raise ValueError(f"INI file {base_ini_path} missing required [env] section")

    for key, val in env_config.items():
        config.set("env", key, str(val))

    fd, tmp_path = tempfile.mkstemp(suffix=".ini", prefix=prefix)
    with os.fdopen(fd, "w") as f:
        config.write(f)

    return tmp_path


def generate_safe_eval_ini(safe_eval_config, base_ini_path="pufferlib/config/ocean/drive.ini"):
    """Generate a temporary INI file with safe reward conditioning values.

    Builds on generate_env_ini, then pins reward bounds min=max.
    """
    env_overrides = {
        "reward_randomization": 1,
        "reward_conditioning": 1,
        "resample_frequency": 0,
    }
    for key in ["episode_length", "num_agents", "min_goal_distance", "max_goal_distance", "map_dir", "num_maps"]:
        if key in safe_eval_config:
            env_overrides[key] = safe_eval_config[key]

    tmp_path = generate_env_ini(env_overrides, base_ini_path, prefix="safe_eval_")

    # Re-read to pin reward bounds
    config = configparser.ConfigParser()
    config.read(tmp_path)
    for key, val in safe_eval_config.items():
        if config.has_option("env", f"reward_bound_{key}_min"):
            config.set("env", f"reward_bound_{key}_min", str(val))
            config.set("env", f"reward_bound_{key}_max", str(val))

    with open(tmp_path, "w") as f:
        config.write(f)

    return tmp_path


def run_human_replay_eval_in_subprocess(config, logger, global_step):
    """
    Run human replay evaluation in a subprocess and log metrics to wandb.

    """
    try:
        run_id = logger.run_id
        model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")
        model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

        if not model_files:
            print("No model files found for human replay evaluation")
            return

        latest_cpt = max(model_files, key=os.path.getctime)

        # Prepare evaluation command
        eval_config = config["eval"]
        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            config["env"],
            "--load-model-path",
            latest_cpt,
            "--eval.wosac-realism-eval",
            "False",
            "--eval.human-replay-eval",
            "True",
            "--eval.human-replay-num-agents",
            str(eval_config["human_replay_num_agents"]),
            "--eval.human-replay-control-mode",
            str(eval_config["human_replay_control_mode"]),
        ]

        # Run human replay evaluation in subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

        if result.returncode == 0:
            # Extract JSON from stdout between markers
            stdout = result.stdout
            if "HUMAN_REPLAY_METRICS_START" in stdout and "HUMAN_REPLAY_METRICS_END" in stdout:
                start = stdout.find("HUMAN_REPLAY_METRICS_START") + len("HUMAN_REPLAY_METRICS_START")
                end = stdout.find("HUMAN_REPLAY_METRICS_END")
                json_str = stdout[start:end].strip()
                human_replay_metrics = json.loads(json_str)

                # Log to wandb if available
                if hasattr(logger, "wandb") and logger.wandb:
                    logger.wandb.log(
                        {
                            "eval/human_replay_collision_rate": human_replay_metrics["collision_rate"],
                            "eval/human_replay_offroad_rate": human_replay_metrics["offroad_rate"],
                            "eval/human_replay_completion_rate": human_replay_metrics["completion_rate"],
                        },
                        step=global_step,
                    )
        else:
            print(f"Human replay evaluation failed with exit code {result.returncode}: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("Human replay evaluation timed out")
    except Exception as e:
        print(f"Failed to run human replay evaluation: {e}")


def run_driving_behaviour_class_eval_in_subprocess(config, class_name, class_cfg, reward_config, logger, global_step):
    """
    Run a single driving behaviour class eval in a subprocess via human replay.
    Uses the latest checkpoint and passes class-specific map_dir and reward bounds as CLI args.
    Logs results to wandb under driving_behaviours/<short>/<metric>.
    """
    EVAL_SECTIONS_PREFIX = "eval_"
    try:
        run_id = logger.run_id
        model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")
        model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

        if not model_files:
            print(f"[DrivingBehavioursEval] No model files found, skipping {class_name}")
            return {}

        latest_cpt = max(model_files, key=os.path.getctime)

        map_dir = class_cfg.get("map_dir", "")
        if isinstance(map_dir, str):
            map_dir = map_dir.strip('"')

        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            config["env"],
            "--load-model-path",
            latest_cpt,
            "--eval.wosac-realism-eval",
            "False",
            "--eval.human-replay-eval",
            "True",
            "--eval.human-replay-control-mode",
            "control_sdc_only",
            "--env.map-dir",
            map_dir,
            "--env.init-mode",
            "create_all_valid",
            "--env.episode-length",
            "91",
            "--env.resample-frequency",
            "0",
        ]

        # Fix reward conditioning: set both min and max to the eval value
        for key, val in reward_config.items():
            cmd += [f"--env.reward-bound-{key.replace('_', '-')}-min", str(val)]
            cmd += [f"--env.reward-bound-{key.replace('_', '-')}-max", str(val)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

        if result.returncode != 0:
            print(
                f"[DrivingBehavioursEval] Subprocess failed for {class_name} "
                f"(exit {result.returncode}):\n{result.stderr}"
            )
            return {}

        stdout = result.stdout
        if "HUMAN_REPLAY_METRICS_START" not in stdout or "HUMAN_REPLAY_METRICS_END" not in stdout:
            print(f"[DrivingBehavioursEval] No metrics found in subprocess output for {class_name}")
            return {}

        start = stdout.find("HUMAN_REPLAY_METRICS_START") + len("HUMAN_REPLAY_METRICS_START")
        end = stdout.find("HUMAN_REPLAY_METRICS_END")
        metrics = json.loads(stdout[start:end].strip())

        short = class_name[len(EVAL_SECTIONS_PREFIX) :]
        print(f"[DrivingBehavioursEval] {short}: {metrics}")

        if hasattr(logger, "wandb") and logger.wandb:
            payload = {f"driving_behaviours/{short}/{k}": float(v) for k, v in metrics.items()}
            if global_step is not None:
                payload["train_step"] = global_step
            logger.wandb.log(payload)

        return metrics

    except subprocess.TimeoutExpired:
        print(f"[DrivingBehavioursEval] Subprocess timed out for {class_name}")
        return {}
    except Exception as e:
        print(f"[DrivingBehavioursEval] Failed for {class_name}: {e}")
        return {}


def run_wosac_eval_in_subprocess(config, logger, global_step):
    """
    Run WOSAC evaluation in a subprocess and log metrics to wandb.

    Args:
        config: Configuration dictionary containing data_dir, env, and wosac settings
        logger: Logger object with run_id and optional wandb attribute
        epoch: Current training epoch
        global_step: Current global training step

    Returns:
        None. Prints error messages if evaluation fails.
    """
    try:
        run_id = logger.run_id
        model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")
        model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

        # Prepare evaluation command
        eval_config = config.get("eval", {})
        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            config["env"],
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
        ]

        if not model_files:
            print("No model files found for WOSAC evaluation. Running WOSAC with random policy.")
        else:
            latest_cpt = max(model_files, key=os.path.getctime)
            cmd.extend(["--load-model-path", latest_cpt])

        # Run WOSAC evaluation in subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

        if result.returncode == 0:
            # Extract JSON from stdout between markers
            stdout = result.stdout
            if "WOSAC_METRICS_START" in stdout and "WOSAC_METRICS_END" in stdout:
                start = stdout.find("WOSAC_METRICS_START") + len("WOSAC_METRICS_START")
                end = stdout.find("WOSAC_METRICS_END")
                json_str = stdout[start:end].strip()
                wosac_metrics = json.loads(json_str)

                # Log to wandb if available
                if hasattr(logger, "wandb") and logger.wandb:
                    logger.wandb.log(
                        {
                            "eval/wosac_realism_meta_score": wosac_metrics["realism_meta_score"],
                            "eval/realism_meta_score_std": wosac_metrics["realism_meta_score_std"],
                            "eval/wosac_kinematic_metrics": wosac_metrics["kinematic_metrics"],
                            "eval/wosac_interactive_metrics": wosac_metrics["interactive_metrics"],
                            "eval/wosac_map_based_metrics": wosac_metrics["map_based_metrics"],
                            "eval/wosac_ade": wosac_metrics["ade"],
                            "eval/wosac_min_ade": wosac_metrics["min_ade"],
                            "eval/wosac_total_num_agents": wosac_metrics["total_num_agents"],
                        },
                        step=global_step,
                    )
        else:
            print(f"WOSAC evaluation failed with exit code {result.returncode}")
            print(f"Error: {result.stderr}")

            # Check for memory issues
            stderr_lower = result.stderr.lower()
            if "out of memory" in stderr_lower or "cuda out of memory" in stderr_lower:
                print("GPU out of memory. Skipping this WOSAC evaluation.")

    except subprocess.TimeoutExpired:
        print("WOSAC evaluation timed out after 600 seconds")
    except MemoryError as e:
        print(f"WOSAC evaluation ran out of memory. Skipping this evaluation: {e}")
    except Exception as e:
        print(f"Failed to run WOSAC evaluation: {type(e).__name__}: {e}")


def render_videos(
    config,
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
    num_maps=None,
    map_dir=None,
):
    """Generate and log training videos using C-based rendering."""
    if not os.path.exists(bin_path):
        print(f"Binary weights file does not exist: {bin_path}")
        return

    model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")

    # Now call the C rendering function
    try:
        # Create output directory for videos
        video_output_dir = os.path.join(model_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        # TODO: Fix memory leaks so that this is not needed
        # Suppress AddressSanitizer exit code (temp)
        env_vars = os.environ.copy()
        env_vars["ASAN_OPTIONS"] = "exitcode=0"

        # Base command with only visualization flags (env config comes from INI)
        base_cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", "./visualize"]

        if config_path:
            base_cmd.extend(["--config", config_path])

        # Visualization config flags only
        if config.get("show_grid", False):
            base_cmd.append("--show-grid")
        if config.get("obs_only", False):
            base_cmd.append("--obs-only")
        if config.get("show_lasers", False):
            base_cmd.append("--lasers")
        if config.get("show_human_logs", False):
            base_cmd.append("--log-trajectories")
        if config.get("zoom_in", False):
            base_cmd.append("--zoom-in")

        # Frame skip for rendering performance
        frame_skip = config.get("frame_skip", 1)
        if frame_skip > 1:
            base_cmd.extend(["--frame-skip", str(frame_skip)])

        # View mode
        view_mode = config.get("view_mode", "both")
        base_cmd.extend(["--view", view_mode])

        if num_maps:
            base_cmd.extend(["--num-maps", str(num_maps)])

        base_cmd.extend(["--policy-name", bin_path])

        # Handle single or multiple map rendering
        render_maps = config.get("render_map", None)
        if render_maps is None or render_maps == "none":
            pass  # use map_dir passed as parameter
            if map_dir and os.path.isdir(map_dir):
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

        generated_videos = {"output_topdown": [], "output_agent": []}
        output_topdown = f"resources/drive/{wandb_prefix}_output_topdown_{epoch}"
        output_agent = f"resources/drive/{wandb_prefix}_output_agent_{epoch}"

        for i, map_path in enumerate(render_maps):
            cmd = list(base_cmd)  # copy
            if os.path.exists(map_path):
                cmd.extend(["--map-name", str(map_path)])

            output_topdown_map = output_topdown + (f"_map{i:02d}.mp4" if len(render_maps) > 1 else ".mp4")
            output_agent_map = output_agent + (f"_map{i:02d}.mp4" if len(render_maps) > 1 else ".mp4")

            cmd.extend(["--output-topdown", output_topdown_map])
            cmd.extend(["--output-agent", output_agent_map])

            print(f"Running render: {' '.join(cmd[:6])}...")
            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=1200, env=env_vars)

            vids_exist = os.path.exists(output_topdown_map) and os.path.exists(output_agent_map)
            print(f"Render exit code: {result.returncode}, vids_exist: {vids_exist}")
            if result.returncode != 0 and result.stderr:
                print(f"Render stderr: {result.stderr[-500:]}")

            if result.returncode == 0 or (result.returncode == 1 and vids_exist):
                videos = [
                    ("output_topdown", output_topdown_map, f"epoch_{epoch:06d}_map{i:02d}_topdown.mp4"),
                    ("output_agent", output_agent_map, f"epoch_{epoch:06d}_map{i:02d}_agent.mp4"),
                ]

                for vid_type, source_vid, target_filename in videos:
                    if os.path.exists(source_vid):
                        target_path = os.path.join(video_output_dir, target_filename)
                        shutil.move(source_vid, target_path)
                        generated_videos[vid_type].append(target_path)
                    else:
                        print(f"Video generation completed but {source_vid} not found")
                        if result.stdout:
                            print(f"StdOUT: {result.stdout}")
                        if result.stderr:
                            print(f"StdERR: {result.stderr}")
            else:
                print(f"C rendering failed (map index {i}) with exit code {result.returncode}: {result.stderr}")

        if render_async:
            render_queue.put(
                {
                    "videos": generated_videos,
                    "step": global_step,
                    "prefix": wandb_prefix,
                }
            )
        elif wandb_log and wandb_run:
            import wandb

            payload = {}
            if generated_videos["output_topdown"]:
                payload[f"{wandb_prefix}/world_state"] = [
                    wandb.Video(p, format="mp4") for p in generated_videos["output_topdown"]
                ]
            if generated_videos["output_agent"]:
                payload[f"{wandb_prefix}/agent_view"] = [
                    wandb.Video(p, format="mp4") for p in generated_videos["output_agent"]
                ]
            if payload:
                print(f"Logging {len(payload)} video keys to wandb: {list(payload.keys())}")
                payload["train_step"] = global_step
                wandb_run.log(payload)

    except subprocess.TimeoutExpired:
        print("C rendering timed out")
    except Exception as e:
        print(f"Failed to render videos: {e}")


def render_videos_and_cleanup(cleanup_files=None, **render_kwargs):
    """Wrapper that runs render_videos then cleans up temp files.

    Intended as the target for multiprocessing.Process so that temp files
    (bin weights, generated INI) are cleaned up inside the spawned process.
    """
    try:
        render_videos(**render_kwargs)
    finally:
        for f in cleanup_files or []:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
