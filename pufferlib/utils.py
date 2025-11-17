import os
import sys
import glob
import shutil
import subprocess
import json


def run_wosac_eval(config, logger, epoch, global_step):
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
        wosac_config = config.get("wosac", {})
        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            config["env"],
            "--load-model-path",
            latest_cpt,
            "--wosac.enabled",
            "True",
            "--wosac.num-total-wosac-agents",
            str(wosac_config.get("num_total_wosac_agents", 256)),
            "--wosac.init-mode",
            str(wosac_config.get("init_mode", "create_all_valid")),
            "--wosac.control-mode",
            str(wosac_config.get("control_mode", "control_tracks_to_predict")),
            "--wosac.init-steps",
            str(wosac_config.get("init_steps", 10)),
            "--wosac.goal-behavior",
            str(wosac_config.get("goal_behavior", 2)),
            "--wosac.goal-radius",
            str(wosac_config.get("goal_radius", 2.0)),
            "--wosac.aggregate-results",
            str(wosac_config.get("aggregate_results", True)),
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
                            "realism/realism_meta_score": wosac_metrics["realism_meta_score"],
                            "realism/ade": wosac_metrics["ade"],
                            "realism/min_ade": wosac_metrics["min_ade"],
                            "realism/total_num_agents": wosac_metrics["total_num_agents"],
                            # Optionally log all metrics:
                            # **{f"realism/{k}": v for k, v in wosac_metrics.items()}
                        },
                        step=global_step,
                    )
        else:
            print(f"WOSAC evaluation failed with exit code {result.returncode}: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("WOSAC evaluation timed out")
    except Exception as e:
        print(f"Failed to run WOSAC evaluation: {e}")


def render_videos(config, vecenv, logger, epoch, global_step, bin_path):
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

    run_id = logger.run_id
    model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")

    # Now call the C rendering function
    try:
        # Create output directory for videos
        video_output_dir = os.path.join(model_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        # Copy the binary weights to the expected location
        expected_weights_path = "resources/drive/puffer_drive_weights.bin"
        os.makedirs(os.path.dirname(expected_weights_path), exist_ok=True)
        shutil.copy2(bin_path, expected_weights_path)

        # TODO: Fix memory leaks so that this is not needed
        # Suppress AddressSanitizer exit code (temp)
        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "exitcode=0"

        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", "./visualize"]

        # Add render configurations
        if config["show_grid"]:
            cmd.append("--show-grid")
        if config["obs_only"]:
            cmd.append("--obs-only")
        if config["show_lasers"]:
            cmd.append("--lasers")
        if config["show_human_logs"]:
            cmd.append("--log-trajectories")
        if vecenv.driver_env.goal_radius is not None:
            cmd.extend(["--goal-radius", str(vecenv.driver_env.goal_radius)])
        if vecenv.driver_env.init_steps > 0:
            cmd.extend(["--init-steps", str(vecenv.driver_env.init_steps)])
        if config["render_map"] is not None:
            map_path = config["render_map"]
            if os.path.exists(map_path):
                cmd.extend(["--map-name", map_path])
        if vecenv.driver_env.init_mode is not None:
            cmd.extend(["--init-mode", str(vecenv.driver_env.init_mode)])
        if vecenv.driver_env.control_mode is not None:
            cmd.extend(["--control-mode", str(vecenv.driver_env.control_mode)])

        # Specify output paths for videos
        cmd.extend(["--output-topdown", "resources/drive/output_topdown.mp4"])
        cmd.extend(["--output-agent", "resources/drive/output_agent.mp4"])

        # Add environment configuration
        env_cfg = getattr(vecenv, "driver_env", None)
        if env_cfg is not None:
            n_policy = getattr(env_cfg, "max_controlled_agents", -1)
            try:
                n_policy = int(n_policy)
            except (TypeError, ValueError):
                n_policy = -1
            if n_policy > 0:
                cmd += ["--num-policy-controlled-agents", str(n_policy)]
            if getattr(env_cfg, "num_maps", False):
                cmd.extend(["--num-maps", str(env_cfg.num_maps)])
            if getattr(env_cfg, "scenario_length", None):
                cmd.extend(["--scenario-length", str(env_cfg.scenario_length)])

        # Call C code that runs eval_gif() in subprocess
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=120, env=env)

        vids_exist = os.path.exists("resources/drive/output_topdown.mp4") and os.path.exists(
            "resources/drive/output_agent.mp4"
        )

        if result.returncode == 0 or (result.returncode == 1 and vids_exist):
            # Move both generated videos to the model directory
            videos = [
                ("resources/drive/output_topdown.mp4", f"epoch_{epoch:06d}_topdown.mp4"),
                ("resources/drive/output_agent.mp4", f"epoch_{epoch:06d}_agent.mp4"),
            ]

            for source_vid, target_filename in videos:
                if os.path.exists(source_vid):
                    target_gif = os.path.join(video_output_dir, target_filename)
                    shutil.move(source_vid, target_gif)

                    # Log to wandb if available
                    if hasattr(logger, "wandb") and logger.wandb:
                        import wandb

                        view_type = "world_state" if "topdown" in target_filename else "agent_view"
                        logger.wandb.log(
                            {f"render/{view_type}": wandb.Video(target_gif, format="mp4")},
                            step=global_step,
                        )
                else:
                    print(f"Video generation completed but {source_vid} not found")
        else:
            print(f"C rendering failed with exit code {result.returncode}: {result.stdout}")

    except subprocess.TimeoutExpired:
        print("C rendering timed out")
    except Exception as e:
        print(f"Failed to generate GIF: {e}")

    finally:
        # Clean up bin weights file
        if os.path.exists(expected_weights_path):
            os.remove(expected_weights_path)
