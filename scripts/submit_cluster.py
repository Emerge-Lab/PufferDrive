"""
Submitit script for launching PufferDrive training jobs on SLURM clusters.

Example usage:
    # Single job with config file
    python scripts/submit_cluster.py \
        --save_dir /path/to/experiments \
        --compute_config scripts/cluster_configs/nyu_greene.yaml \
        --program_config scripts/cluster_configs/train_base.yaml \
        --dry 0

    # Sweep over learning rates
    python scripts/submit_cluster.py \
        --save_dir /path/to/experiments \
        --compute_config scripts/cluster_configs/nyu_greene.yaml \
        --args learning_rate=1e-4:3e-4:1e-3 \
        --dry 0

    # Override compute settings
    python scripts/submit_cluster.py \
        --save_dir /path/to/experiments \
        --compute_config scripts/cluster_configs/nyu_greene.yaml \
        --gpus 4 --time 120 \
        --dry 0
"""

import argparse
import hashlib
import json
import os
import pprint
import time
from typing import List, Optional, Tuple, Dict

import yaml
import submitit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit PufferDrive training jobs to SLURM cluster"
    )

    # Job management
    parser.add_argument("--save_dir", type=str, required=True, help="Base directory for experiment outputs")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for job names")
    parser.add_argument("--dry", type=int, default=1, help="Dry run (1) or submit (0)")

    # Config files
    parser.add_argument("--compute_config", type=str, default=None, help="YAML file with SLURM settings")
    parser.add_argument("--program_config", type=str, default=None, help="YAML file with training args")

    # SLURM settings (override compute_config)
    parser.add_argument("--account", type=str, help="SLURM account")
    parser.add_argument("--partition", type=str, help="SLURM partition")
    parser.add_argument("--cpus", type=int, default=None, help="CPUs per task")
    parser.add_argument("--gpus", type=int, default=None, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=None, help="Number of nodes")
    parser.add_argument("--gpu_type", type=str, default=None, help="GPU type (a100/v100/etc)")
    parser.add_argument("--nodelist", type=str, default=None, help="Specific nodes to use")
    parser.add_argument("--mem", type=str, default=None, help="Memory per node (e.g., 32gb)")
    parser.add_argument("--exclude", type=str, default="", help="Nodes to exclude")
    parser.add_argument("--time", type=int, default=None, help="Time limit in minutes")
    parser.add_argument("--task_per_node", type=int, default=1, help="Tasks per node")
    parser.add_argument("--max_pjob", type=int, default=None, help="Max parallel jobs")

    # Program settings
    parser.add_argument("--main", type=str, default="-m pufferlib.pufferl train puffer_drive", help="Main command to run")
    parser.add_argument("--args", type=str, nargs="+", default=None, help="Args to override/sweep (e.g., learning_rate=1e-4:3e-4)")

    args = parser.parse_args()
    return args


def process_main_args(main_args: Optional[List[str]], program_config: Optional[str]) -> Tuple[List[Dict], List[str]]:
    """Process arguments and expand sweep syntax (colon-separated values)."""
    from_config = {}
    if program_config is not None:
        from_config = yaml.safe_load(open(program_config, "r"))
        print("Loaded base config:")
        pprint.pprint(from_config)

    full_args = [from_config]
    if main_args is None:
        return full_args, []

    override_keys = []
    for arg in main_args:
        new_full_args = []
        key, vals = arg.split("=")
        override_keys.append(key)
        vals = vals.split(":")
        for val in vals:
            for args in full_args:
                new_args = args.copy()
                new_args[key] = val
                new_full_args.append(new_args)
        full_args = new_full_args

    return full_args, override_keys


def generate_dict_hash(params_dict: Dict, hash_len: int = 7) -> str:
    """Generate a short hash of the params dict for unique job naming."""
    hash_obj = hashlib.sha1(json.dumps(params_dict, sort_keys=True).encode())
    return hash_obj.hexdigest()[:hash_len]


def get_all_commands(args) -> Dict[str, Tuple[List[str], str]]:
    """Generate all commands to run (expanding sweeps)."""
    all_main_args, overrides = process_main_args(args.args, args.program_config)
    name2commands = {}

    for main_args in all_main_args:
        cmd = []
        name_entries = []

        if args.program_config is not None:
            name_entries.append(args.program_config.split("/")[-1].rsplit(".", 1)[0])

        # Boolean flags that don't take values (store_true)
        boolean_flags = {"wandb", "neptune"}

        for key, val in main_args.items():
            # Convert underscores to dashes for CLI compatibility
            cli_key = key.replace("_", "-")

            # Handle boolean flags that don't take values
            if key in boolean_flags:
                if val in (True, "True", "true", "1"):
                    cmd.append(f"--{cli_key}")
                # Skip if False - don't add the flag at all
            else:
                cmd.append(f"--{cli_key}")
                cmd.append(str(val))

            if key in overrides and key not in ["config", "config_path"]:
                display_key = key.split(".")[-1] if "." in key else key
                name_entries.append(f"{display_key}{val}")

        job_name = "_".join(name_entries) if name_entries else "pufferdrive"
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
        job_name += "_" + generate_dict_hash(main_args)

        if args.prefix is not None:
            job_name = f"{args.prefix}_{job_name}"

        save_dir = os.path.join(args.save_dir, job_name)
        name2commands[job_name] = (cmd, save_dir)

    return name2commands


def submit(args, job_name: str, command: List[str], save_dir: str, dry: bool):
    """Submit a single job to SLURM via submitit."""
    # Load compute config
    from_config = {}
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

    additional_parameters = {}
    if from_config.get("nodelist") is not None:
        additional_parameters["nodelist"] = from_config["nodelist"]

    executor.update_parameters(
        slurm_account=from_config.get("account"),
        slurm_partition=from_config.get("partition"),
        cpus_per_task=from_config.get("cpus", 8) // args.task_per_node,
        tasks_per_node=args.task_per_node,
        nodes=from_config.get("nodes", 1),
        slurm_gres=gres,
        slurm_exclude=from_config.get("exclude", ""),
        slurm_mem=from_config.get("mem"),
        slurm_time=from_config.get("time", 60),
        slurm_job_name=job_name,
        slurm_additional_parameters=additional_parameters,
    )

    def launch_training(args, from_config, cmd, save_dir, project_root):
        """Runs inside the SLURM allocation."""
        import os
        import subprocess
        import sys
        import submitit

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
                "--nproc_per_node", str(gpus),
            ] + main_parts
        else:
            env = submitit.JobEnvironment()
            master_addr = env.hostnames[0]
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = "29500"

            base_cmd = [
                "torchrun",
                "--nnodes", str(nodes),
                "--nproc_per_node", str(gpus),
                "--rdzv-backend", "c10d",
                "--rdzv-id", str(env.job_id),
                "--rdzv-endpoint", f"{master_addr}:29500",
            ] + main_parts

        # Add save_dir to command
        full_cmd = base_cmd + cmd + ["--train.data-dir", save_dir]

        print(f">>> Job: {job_name}")
        print(f">>> Working directory: {project_root}")
        print(f">>> Command: {' '.join(full_cmd)}")
        subprocess.run(full_cmd, check=True)

    # Get project root (directory containing this script's parent)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not dry:
        job = executor.submit(launch_training, args, from_config, command, save_dir, project_root)
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
