"""
Submitit script for launching PufferDrive (puffer-4) training jobs on SLURM.

Example usage:
    # Single job with config file
    python scripts/submit_cluster.py \
        --save_dir /path/to/experiments \
        --compute_config scripts/cluster_configs/nyu_greene.yaml \
        --program_config scripts/cluster_configs/train_base.yaml

    # Sweep over learning rates
    python scripts/submit_cluster.py \
        --save_dir /path/to/experiments \
        --compute_config scripts/cluster_configs/nyu_greene.yaml \
        --args train.learning_rate=1e-4:3e-4:1e-3

    # Override compute settings
    python scripts/submit_cluster.py \
        --save_dir /path/to/experiments \
        --compute_config scripts/cluster_configs/nyu_greene.yaml \
        --gpus 4 --time 120

    # Dry run (preview commands without submitting)
    python scripts/submit_cluster.py \
        --save_dir /path/to/experiments \
        --compute_config scripts/cluster_configs/nyu_greene.yaml \
        --dry

    # Run inside Singularity container (for glibc compatibility)
    python scripts/submit_cluster.py \
        --save_dir /path/to/experiments \
        --compute_config scripts/cluster_configs/nyu_greene.yaml \
        --program_config scripts/cluster_configs/train_base.yaml \
        --container

Notes on puffer-4:
    * Entry point is `puffer train drive` (installed via pyproject.toml
      [project.scripts]). We use `python -m pufferlib.pufferl train drive`
      here for robustness against PATH.
    * The `puffer-4` config loader reads nested keys as `--section.key=value`
      (e.g. `--train.learning_rate=1e-4`). Top-level [base] keys have no
      section prefix (e.g. `--checkpoint-dir`, `--wandb-project`).
    * Code isolation: hard-copies `pufferlib/` + `config/` into the job's
      work dir so branch switches or rebuilds don't affect running jobs.
      Everything else in the project root is symlinked.
"""

import argparse
import hashlib
import json
import os
import pprint
import time
from typing import Dict, List, Optional, Tuple

import submitit
import yaml


# Keys in the flat program config that live under [base] (no section prefix in CLI).
# Everything else is assumed to be under a section (train.*, vec.*, env.*, policy.*).
BASE_LEVEL_KEYS = {
    "env_name",
    "rank",
    "world_size",
    "gpu_id",
    "nccl_id",
    "profile",
    "checkpoint_dir",
    "log_dir",
    "checkpoint_interval",
    "eval_episodes",
    "cudagraphs",
    "seed",
    "reset_state",
    "wandb",
    "wandb_project",
    "wandb_group",
    "wandb_name",
    "tag",
    "slowly",
    "load_model_path",
    "load_id",
    "render_mode",
    "save_frames",
    "gif_path",
    "fps",
}

# Boolean [base] flags that are store_true on the parser side (no value).
BOOLEAN_FLAGS = {"wandb", "slowly"}


def parse_args():
    parser = argparse.ArgumentParser(description="Submit PufferDrive (puffer-4) training jobs to SLURM cluster")

    # Job management
    parser.add_argument("--save_dir", type=str, required=True, help="Base directory for experiment outputs")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for job names and wandb run name")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name (defaults to --prefix)")
    parser.add_argument("--wandb-group", type=str, default=None, help="Wandb group name (overrides program config)")
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="Wandb project name (overrides program config)"
    )
    parser.add_argument("--dry", action="store_true", help="Dry run (don't submit, just print commands)")

    # Config files
    parser.add_argument("--compute_config", type=str, default=None, help="YAML file with SLURM settings")
    parser.add_argument("--program_config", type=str, default=None, help="YAML file with training args")

    # SLURM settings (override compute_config)
    parser.add_argument("--account", type=str, help="SLURM account")
    parser.add_argument("--partition", type=str, help="SLURM partition")
    parser.add_argument("--cpus", type=int, default=None, help="CPUs per task")
    parser.add_argument("--gpus", type=int, default=None, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=None, help="Number of nodes")
    parser.add_argument("--gpu_type", type=str, default=None, help="GPU type (a100/l40s/h100/h200)")
    parser.add_argument("--nodelist", type=str, default=None, help="Specific nodes to use")
    parser.add_argument("--mem", type=str, default=None, help="Memory per node (e.g., 32gb)")
    parser.add_argument("--exclude", type=str, default="", help="Nodes to exclude")
    parser.add_argument("--time", type=int, default=None, help="Time limit in minutes")
    parser.add_argument("--task_per_node", type=int, default=1, help="Tasks per node")
    parser.add_argument("--max_pjob", type=int, default=None, help="Max parallel jobs")

    # Program settings
    parser.add_argument(
        "--main",
        type=str,
        default="-m pufferlib.pufferl train drive",
        help="Main command to run",
    )
    parser.add_argument(
        "--args",
        type=str,
        nargs="+",
        default=None,
        help="Args to override/sweep (e.g., train.learning_rate=1e-4:3e-4)",
    )

    # GPU heartbeat: keeps utilization above threshold to prevent job reclamation on NYU cluster
    parser.add_argument(
        "--heartbeat",
        action="store_true",
        help="Run scripts/gpu_heartbeat.py in background alongside training",
    )

    # Container settings
    parser.add_argument("--container", action="store_true", help="Run inside Singularity container")
    parser.add_argument(
        "--container_image",
        type=str,
        default="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif",
        help="Singularity image path",
    )
    parser.add_argument(
        "--container_overlay",
        type=str,
        default=f"/scratch/{os.environ.get('USER', '')}/images/PufferDrive/overlay-15GB-500K.ext3",
        help="Singularity overlay path",
    )

    args = parser.parse_args()
    return args


def process_main_args(main_args: Optional[List[str]], program_config: Optional[str]) -> Tuple[List[Dict], List[str]]:
    """Process arguments and expand sweep syntax (colon-separated values)."""
    from_config: Dict = {}
    if program_config is not None:
        from_config = yaml.safe_load(open(program_config, "r"))
        print("Loaded base config:")
        pprint.pprint(from_config)

    full_args: List[Dict] = [from_config]
    if main_args is None:
        return full_args, []

    override_keys: List[str] = []
    for arg in main_args:
        new_full_args = []
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: '{arg}'. Expected 'key=value' or 'key=val1:val2'")
        key, vals = arg.split("=", 1)
        override_keys.append(key)
        vals = vals.split(":")
        for val in vals:
            for a in full_args:
                new_a = a.copy()
                new_a[key] = val
                new_full_args.append(new_a)
        full_args = new_full_args

    return full_args, override_keys


def generate_dict_hash(params_dict: Dict, hash_len: int = 7) -> str:
    """Generate a short hash of the params dict for unique job naming."""
    hash_obj = hashlib.sha1(json.dumps(params_dict, sort_keys=True).encode())
    return hash_obj.hexdigest()[:hash_len]


def _key_to_cli(key: str) -> str:
    """Translate a flat/nested config key into a CLI flag for puffer-4's parser.

    Examples:
        "train.learning_rate"  -> "--train.learning-rate"
        "vec.total_agents"     -> "--vec.total-agents"
        "checkpoint_interval"  -> "--checkpoint-interval"          # [base] level
        "wandb_project"        -> "--wandb-project"                # [base] level
    """
    if "." in key:
        section, subkey = key.split(".", 1)
        return f"--{section}.{subkey.replace('_', '-')}"
    return f"--{key.replace('_', '-')}"


def get_all_commands(args) -> Dict[str, Tuple[List[str], str]]:
    """Generate all commands to run (expanding sweeps)."""
    all_main_args, overrides = process_main_args(args.args, args.program_config)
    name2commands: Dict[str, Tuple[List[str], str]] = {}

    # Keys to exclude from auto-generated job name (paths, wandb config, common overrides).
    name_skip_keys = {
        "config",
        "config_path",
        "total_timesteps",
        "train.total_timesteps",
        "wandb",
        "wandb_project",
        "wandb_group",
        "wandb_name",
        "checkpoint_dir",
        "log_dir",
    }

    for main_args_inst in all_main_args:
        cmd: List[str] = []
        name_entries: List[str] = []

        if args.program_config is not None:
            name_entries.append(args.program_config.split("/")[-1].rsplit(".", 1)[0])

        for key, val in main_args_inst.items():
            cli_key = _key_to_cli(key)
            flat_key = key.split(".")[-1] if "." in key else key

            # Boolean [base] flags that don't take values (store_true).
            if flat_key in BOOLEAN_FLAGS:
                if val in (True, "True", "true", 1, "1"):
                    cmd.append(cli_key)
                # Skip if False - don't add the flag at all.
            else:
                cmd.append(cli_key)
                cmd.append(str(val))

            if key in overrides and key not in name_skip_keys:
                display_key = flat_key
                name_entries.append(f"{display_key}{val}")

        job_name = "_".join(name_entries) if name_entries else "puffer4_drive"
        # Sanitize job name
        job_name = (
            job_name[:128]
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .replace('"', "")
            .replace(":", "")
            .replace("/", "")
        )
        job_name += "_" + generate_dict_hash(main_args_inst)

        if args.prefix is not None:
            job_name = f"{args.prefix}_{job_name}"

        # Wandb overrides: explicit flags take priority, then prefix for name
        wandb_name = args.wandb_name or args.prefix
        if wandb_name is not None:
            cmd.extend(["--tag", wandb_name])  # puffer-4 uses --tag for the run name suffix
        if args.wandb_group is not None:
            cmd.extend(["--wandb-group", args.wandb_group])
        if args.wandb_project is not None:
            cmd.extend(["--wandb-project", args.wandb_project])

        save_dir = os.path.join(args.save_dir, job_name)
        name2commands[job_name] = (cmd, save_dir)

    return name2commands


def submit(args, job_name: str, command: List[str], save_dir: str, dry: bool):
    """Submit a single job to SLURM via submitit."""
    # Load compute config
    from_config: Dict = {}
    if args.compute_config is not None:
        from_config = yaml.safe_load(open(args.compute_config, "r"))
        from_config = {k: v for k, v in from_config.items() if v is not None}

    # Override with CLI args
    for key in ["account", "partition", "cpus", "gpus", "gpu_type", "mem", "nodes", "time", "nodelist", "exclude"]:
        if vars(args)[key] is not None:
            from_config[key] = vars(args)[key]

    print(">>> Compute config:")
    pprint.pprint(from_config)

    # Set up executor
    executor = submitit.AutoExecutor(folder=os.path.join(save_dir, "submitit"))

    # Build GRES string for GPUs
    if from_config.get("gpu_type") is not None:
        gres = f"gpu:{from_config['gpu_type']}:{from_config['gpus']}"
    elif from_config.get("gpus") is not None:
        gres = f"gpu:{from_config['gpus']}"
    else:
        gres = None

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    additional_parameters: Dict = {}
    if from_config.get("nodelist") is not None:
        additional_parameters["nodelist"] = from_config["nodelist"]

    executor.update_parameters(
        slurm_account=from_config.get("account"),
        slurm_partition=from_config.get("partition"),
        cpus_per_task=from_config.get("cpus", 8) // args.task_per_node,
        tasks_per_node=args.task_per_node,
        nodes=from_config.get("nodes", 1),
        slurm_gres=gres,
        slurm_exclude=from_config.get("exclude") or None,
        slurm_mem=from_config.get("mem"),
        slurm_time=from_config.get("time", 60),
        slurm_job_name=job_name,
        slurm_additional_parameters=additional_parameters,
    )

    def launch_training(args, from_config, cmd, save_dir, project_root, container_config=None):
        """Runs inside the SLURM allocation."""
        import os
        import shutil
        import subprocess

        import submitit

        # --- Code isolation ---
        # puffer-4 layout:
        #   pufferlib/   — Python package + compiled _C.so  (HARD COPY)
        #   config/      — .ini files read at runtime         (HARD COPY)
        #   sim/         — C source                           (symlink)
        #   src/         — CUDA source                        (symlink)
        #   vendor/      — pinned external deps               (symlink)
        #   resources/   — maps + other static data           (symlink)
        #   tests/, constellation/, docs, etc.                (symlink)
        #
        # We hard-copy pufferlib/ and config/ so that switching branches or
        # rebuilding on the login node doesn't change the code running jobs.
        # Everything else is symlinked for speed.
        isolated_root = os.path.join(save_dir, "code")
        if os.path.exists(isolated_root):
            version = 1
            while os.path.exists(f"{isolated_root}_v{version}"):
                version += 1
            isolated_root = f"{isolated_root}_v{version}"
        os.makedirs(isolated_root, exist_ok=True)

        hard_copy_dirs = {"pufferlib", "config"}

        # Symlink every top-level entry that isn't being hard-copied.
        for entry in os.listdir(project_root):
            src = os.path.join(project_root, entry)
            dst = os.path.join(isolated_root, entry)
            if entry in hard_copy_dirs:
                continue
            if os.path.exists(dst) or os.path.islink(dst):
                if os.path.isdir(dst) and not os.path.islink(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            os.symlink(src, dst)

        # Hard copy the directories that must be pinned to submit-time state.
        for entry in hard_copy_dirs:
            src = os.path.join(project_root, entry)
            dst = os.path.join(isolated_root, entry)
            if not os.path.exists(src):
                continue
            if os.path.islink(dst):
                os.remove(dst)
            elif os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=False)

        project_root = isolated_root

        # Change to project directory and set up environment
        os.chdir(project_root)
        os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")

        nodes = from_config.get("nodes", 1)
        gpus = from_config.get("gpus", 1)

        # Parse the main command
        main_parts = args.main.split()

        if nodes == 1:
            base_cmd = [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(gpus),
            ] + main_parts
        else:
            env = submitit.JobEnvironment()
            master_addr = env.hostnames[0]
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = "29500"

            base_cmd = [
                "torchrun",
                "--nnodes",
                str(nodes),
                "--nproc_per_node",
                str(gpus),
                "--rdzv-backend",
                "c10d",
                "--rdzv-id",
                str(env.job_id),
                "--rdzv-endpoint",
                f"{master_addr}:29500",
            ] + main_parts

        # Redirect checkpoint and log dirs to save_dir. On puffer-4 these are
        # [base]-level flags (no section prefix).
        full_cmd = base_cmd + cmd + [
            "--checkpoint-dir",
            os.path.join(save_dir, "checkpoints"),
            "--log-dir",
            os.path.join(save_dir, "logs"),
        ]

        # If heartbeat is enabled, wrap training in a brace group that backgrounds
        # the heartbeat and kills it on training exit, preserving training's exit code.
        def wrap_with_heartbeat(train_cmd_str):
            hb = "python scripts/gpu_heartbeat.py > /tmp/gpu_heartbeat.log 2>&1 & HEARTBEAT_PID=$!"
            return f"{{ {hb}; {train_cmd_str}; TRAIN_EXIT=$?; kill $HEARTBEAT_PID 2>/dev/null; exit $TRAIN_EXIT; }}"

        # Wrap with singularity if container mode is enabled
        if container_config is not None:
            env_setup = "source /ext3/env.sh"
            scratch_dir = os.environ.get("SCRATCH_DIR", "/scratch/" + os.environ.get("USER", ""))
            cache_exports = (
                f"export XDG_CACHE_HOME={scratch_dir}/cache && "
                f"export WANDB_CACHE_DIR={scratch_dir}/wandb_cache && "
                f"export WANDB_CONFIG_DIR={scratch_dir}/wandb_config && "
                f"export WANDB_DATA_DIR={scratch_dir}/wandb_data && "
                f"export WANDB_DIR={scratch_dir}/wandb_data && "
                f"mkdir -p {scratch_dir}/cache"
            )
            train_str = " ".join(full_cmd)
            if args.heartbeat:
                train_str = wrap_with_heartbeat(train_str)
            inner_cmd = f"{env_setup} && {cache_exports} && cd {project_root} && {train_str}"
            full_cmd = [
                "singularity",
                "exec",
                "--nv",
                "--overlay",
                container_config["overlay"] + ":ro",
            ]
            for cert_path in ["/etc/ssl/certs", "/etc/pki"]:
                if os.path.exists(cert_path):
                    full_cmd.extend(["--bind", f"{cert_path}:{cert_path}:ro"])
            full_cmd.extend(
                [
                    container_config["image"],
                    "bash",
                    "-c",
                    inner_cmd,
                ]
            )
        elif args.heartbeat:
            train_str = " ".join(full_cmd)
            full_cmd = ["bash", "-c", wrap_with_heartbeat(train_str)]

        print(f">>> Job: {job_name}")
        print(f">>> Working directory: {project_root}")
        print(f">>> Container: {container_config is not None}")
        print(f">>> Command: {' '.join(full_cmd)}")
        subprocess.run(full_cmd, check=True)

    # Get project root (directory containing this script's parent)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build container config if enabled
    container_config = None
    if args.container:
        container_config = {
            "image": args.container_image,
            "overlay": args.container_overlay,
        }
        print(f">>> Container mode enabled: {container_config['image']}")

    if not dry:
        job = executor.submit(launch_training, args, from_config, command, save_dir, project_root, container_config)
        print(f"Submitted job {job.job_id}: {job_name}")
        return job
    else:
        print(f"[DRY RUN] Would submit: {job_name}")
        return None


def wait_if_full(jobs: List, max_pjob: Optional[int]):
    """Wait if we've hit the max parallel job limit."""

    def remove_done_jobs(jobs):
        for i in range(len(jobs) - 1, -1, -1):
            if jobs[i].done():
                jobs.pop(i)
        return len(jobs)

    if max_pjob is None:
        return

    while remove_done_jobs(jobs) >= max_pjob and len(jobs) > 0:
        print(f"Reached max jobs ({len(jobs)}), waiting...")
        time.sleep(120)

    print(f"{len(jobs)} jobs remaining, launching new job")


if __name__ == "__main__":
    args = parse_args()
    name2commands = get_all_commands(args)

    print(f">>> Will submit {len(name2commands)} job(s)")
    jobs = []
    for name, (cmd, save_dir) in name2commands.items():
        job = submit(args, name, cmd, save_dir, args.dry)
        if job is not None:
            jobs.append(job)
        wait_if_full(jobs, args.max_pjob)

    print("All jobs launched!")
