import os
import sys
import glob
import shutil
import subprocess
import json


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

        if not model_files:
            print("No model files found for WOSAC evaluation")
            return

        latest_cpt = max(model_files, key=os.path.getctime)

        # Prepare evaluation command
        eval_config = config.get("eval", {})
        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            config["env"],
            "--load-model-path",
            latest_cpt,
            "--eval.wosac-realism-eval",
            "True",
            "--eval.wosac-num-agents",
            str(eval_config.get("wosac_num_agents", 256)),
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
        ]

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
    config, env_cfg, run_id, wandb_log, epoch, global_step, bin_path, render_async, render_queue=None, wandb_run=None
):
    """
    Generate and log training videos using C-based rendering.

    Args:
        config: Configuration dictionary containing data_dir, env, and render settings
        vecenv: Vectorized environment with driver_env attribute
        logger: Logger object with run_id and optional wandb attribute
        epoch: Current training epoch
        global_step: Current global training step
        bin_path: Path to the exported .bin model weights file

    Returns:
        None. Prints error messages if rendering fails.
    """
    if not os.path.exists(bin_path):
        print(f"Binary weights file does not exist: {bin_path}")
        return

    model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")

    # Now call the C rendering function
    try:
        # Create output directory for videos
        video_output_dir = os.path.join(model_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        # Record the path of the latest weights file
        latest_weights_path = "resources/drive/puffer_drive_weights.bin"
        os.makedirs(os.path.dirname(latest_weights_path), exist_ok=True)

        # TODO: Fix memory leaks so that this is not needed
        # Suppress AddressSanitizer exit code (temp)
        env_vars = os.environ.copy()
        env_vars["ASAN_OPTIONS"] = "exitcode=0"

        # Base command with only visualization flags (env config comes from INI)
        base_cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", "./visualize"]

        # Visualization config flags only
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

        # Frame skip for rendering performance
        frame_skip = config.get("frame_skip", 1)
        if frame_skip > 1:
            base_cmd.extend(["--frame-skip", str(frame_skip)])

        # View mode
        view_mode = config.get("view_mode", "both")
        base_cmd.extend(["--view", view_mode])

        # Get num_maps if available
        if env_cfg is not None and getattr(env_cfg, "num_maps", None):
            base_cmd.extend(["--num-maps", str(env_cfg.num_maps)])

        base_cmd.extend(["--policy-name", bin_path])

        # Handle single or multiple map rendering
        render_maps = config.get("render_map", None)
        if render_maps is None or render_maps == "none":
            # Pick a random map from the training map_dir
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
            # Ensure list-like
            render_maps = list(render_maps)

        # Collect videos to log as lists so W&B shows all in the same step
        videos_to_log_world = []
        videos_to_log_agent = []
        generated_videos = {"output_topdown": [], "output_agent": []}
        output_topdown = f"resources/drive/output_topdown_{epoch}.mp4"
        output_agent = f"resources/drive/output_agent_{epoch}.mp4"

        for i, map_path in enumerate(render_maps):
            cmd = list(base_cmd)  # copy
            if map_path is not None and os.path.exists(map_path):
                cmd.extend(["--map-name", str(map_path)])

            # Output paths (overwrite each iteration; then moved/renamed)
            cmd.extend(["--output-topdown", output_topdown])
            cmd.extend(["--output-agent", output_agent])

            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=600, env=env_vars)

            vids_exist = os.path.exists(output_topdown) and os.path.exists(output_agent)

            if result.returncode == 0 or (result.returncode == 1 and vids_exist):
                videos = [
                    (
                        "output_topdown",
                        output_topdown,
                        f"epoch_{epoch:06d}_map{i:02d}_topdown.mp4" if map_path else f"epoch_{epoch:06d}_topdown.mp4",
                    ),
                    (
                        "output_agent",
                        output_agent,
                        f"epoch_{epoch:06d}_map{i:02d}_agent.mp4" if map_path else f"epoch_{epoch:06d}_agent.mp4",
                    ),
                ]

                for vid_type, source_vid, target_filename in videos:
                    if os.path.exists(source_vid):
                        target_path = os.path.join(video_output_dir, target_filename)
                        shutil.move(source_vid, target_path)
                        generated_videos[vid_type].append(target_path)
                        if render_async:
                            continue
                        # Accumulate for a single wandb.log call
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
                }
            )

        # Log all videos at once so W&B keeps all of them under the same step
        if wandb_log and (videos_to_log_world or videos_to_log_agent) and not render_async:
            payload = {}
            if videos_to_log_world:
                payload["render/world_state"] = videos_to_log_world
            if videos_to_log_agent:
                payload["render/agent_view"] = videos_to_log_agent
            wandb_run.log(payload, step=global_step)

    except subprocess.TimeoutExpired:
        print("C rendering timed out")
    except Exception as e:
        print(f"Failed to generate GIF: {e}")

    finally:
        # Clean up bin weights file
        if os.path.exists(latest_weights_path):
            os.remove(latest_weights_path)
        if os.path.exists(bin_path):
            os.remove(bin_path)
