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
