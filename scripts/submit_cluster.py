"""Submit OpenGigaFlow training jobs to a SLURM cluster via submitit, one per point of the --args colon sweeps."""

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
    parser = argparse.ArgumentParser(description="Submit OpenGigaFlow training jobs to SLURM cluster")

    parser.add_argument("--save_dir", type=str, required=True, help="Base directory for experiment outputs")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for job names and wandb run name")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name (defaults to --prefix)")
    parser.add_argument("--wandb-group", type=str, default=None, help="Wandb group name (overrides program config)")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name (overrides program config)")
    # env vars of the container command, not --args: a URL's ':' would read as a sweep separator
    parser.add_argument("--wandb-base-url", type=str, default=None, help="WANDB_BASE_URL, e.g. https://api.wandb.ai")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WANDB_ENTITY, e.g. emerge_")
    parser.add_argument("--dry", action="store_true", help="Dry run (don't submit, just print commands)")

    parser.add_argument("--compute_config", type=str, default=None, help="YAML file with SLURM settings")
    parser.add_argument("--program_config", type=str, default=None, help="YAML file with training args")

    parser.add_argument("--account", type=str, help="SLURM account")
    parser.add_argument("--partition", type=str, help="SLURM partition")
    parser.add_argument("--cpus", type=int, default=None, help="CPUs per task")
    parser.add_argument("--gpus", type=int, default=None, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=None, help="Number of nodes")
    parser.add_argument("--gpu_type", type=str, default=None, help="GPU type (a100/v100/etc)")
    parser.add_argument("--nodelist", type=str, default=None, help="Specific nodes to use")
    parser.add_argument(
        "--mem",
        type=str,
        default=None,
        help="Memory per node (e.g., 96gb). Falls back to compute_config['mem'], then to 96gb.",
    )
    parser.add_argument("--exclude", type=str, default="", help="Nodes to exclude")
    parser.add_argument("--time", type=int, default=None, help="Time limit in minutes")
    parser.add_argument("--task_per_node", type=int, default=1, help="Tasks per node")
    parser.add_argument("--max_pjob", type=int, default=None, help="Max parallel jobs")

    parser.add_argument("--main", type=str, default="-m opengigaflow.rl train", help="Main command to run")
    parser.add_argument(
        "--args", type=str, nargs="+", default=None, help="Args to override/sweep (e.g., learning_rate=1e-4:3e-4)"
    )

    # keeps the GPU utilization up so the NYU cluster does not reclaim the job
    parser.add_argument(
        "--heartbeat",
        action="store_true",
        help="Run scripts/gpu_heartbeat.py in background alongside training",
    )

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
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: '{arg}'. Expected 'key=value' or 'key=val1:val2'")
        key, vals = arg.split("=", 1)
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

    # Keys to exclude from auto-generated job name (paths, wandb config, common overrides)
    name_skip_keys = {
        "config",
        "config_path",
        "map_dir",
        "env.map_dir",
        "init_mode",
        "env.init_mode",
        "total_timesteps",
        "train.total_timesteps",
        "wandb_project",
        "wandb_group",
        "wandb_name",
    }
    for main_args in all_main_args:
        cmd = []
        name_entries = []

        if args.program_config is not None:
            name_entries.append(args.program_config.split("/")[-1].rsplit(".", 1)[0])

        for key, val in main_args.items():
            # Hydra override syntax: lowercase booleans, YAML null as Hydra's null
            if isinstance(val, bool):
                val = str(val).lower()
            elif val is None:
                val = "null"
            cmd.append(f"{key}={val}")

            if key in overrides and key not in name_skip_keys:
                display_key = key.split(".")[-1] if "." in key else key
                name_entries.append(f"{display_key}{val}")

        job_name = "_".join(name_entries) if name_entries else "opengigaflow"
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

        # Wandb overrides: explicit flags take priority, then prefix for name
        wandb_name = args.wandb_name or args.prefix
        if wandb_name is not None:
            cmd.append(f"tag={wandb_name}")
        if args.wandb_group is not None:
            cmd.append(f"wandb_group={args.wandb_group}")
        if args.wandb_project is not None:
            cmd.append(f"wandb_project={args.wandb_project}")

        save_dir = os.path.join(args.save_dir, job_name)
        name2commands[job_name] = (cmd, save_dir)

    return name2commands


def isolate_code(project_root: str, save_dir: str) -> str:
    """Snapshot the repo into save_dir/code[_vN] (in-repo save_dirs: <repo>_code_snapshots/). -> the snapshot root."""
    import os
    import shutil
    from pathlib import Path

    # setuptools file discovery follows symlinks through the whole source tree: snapshots must live outside it
    project_root_resolved = Path(project_root).resolve()
    save_dir_resolved = Path(save_dir).resolve()
    save_root_entry = None
    if save_dir_resolved.is_relative_to(project_root_resolved):
        snapshots_root = project_root_resolved.parent / f"{project_root_resolved.name}_code_snapshots"
        save_dir_rel = save_dir_resolved.relative_to(project_root_resolved)
        # the in-repo save root holds run outputs: never link it into the snapshot
        save_root_entry = save_dir_rel.parts[0]
        isolated_root = str(snapshots_root / save_dir_rel / "code")
    else:
        isolated_root = os.path.join(save_dir, "code")
    if os.path.exists(isolated_root):
        version = 1
        while os.path.exists(f"{isolated_root}_v{version}"):
            version += 1
        isolated_root = f"{isolated_root}_v{version}"
    os.makedirs(isolated_root, exist_ok=True)
    isolated_root_real = Path(isolated_root).resolve()
    for entry in os.listdir(project_root):
        src = os.path.join(project_root, entry)
        dst = os.path.join(isolated_root, entry)
        # Do not symlink ancestors to avoid infinite recursion
        if isolated_root_real.is_relative_to(Path(src).resolve()):
            continue
        if entry == save_root_entry:
            continue
        if os.path.exists(dst) or os.path.islink(dst):
            if os.path.isdir(dst) and not os.path.islink(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        os.symlink(src, dst)
    opengigaflow_dst = os.path.join(isolated_root, "opengigaflow")
    if os.path.islink(opengigaflow_dst):
        os.remove(opengigaflow_dst)
    elif os.path.isdir(opengigaflow_dst):
        shutil.rmtree(opengigaflow_dst)
    opengigaflow_src = os.path.join(project_root, "opengigaflow")
    shutil.copytree(
        opengigaflow_src,
        opengigaflow_dst,
        symlinks=False,
        ignore=shutil.ignore_patterns("resources"),
    )
    resources_src = os.path.join(opengigaflow_src, "resources")
    resources_dst = os.path.join(opengigaflow_dst, "resources")
    if os.path.isdir(resources_src):
        os.symlink(resources_src, resources_dst)
    return isolated_root


def submit(args, job_name: str, command: List[str], save_dir: str, dry: bool):
    """Submit a single job to SLURM via submitit."""
    from_config = {}
    if args.compute_config is not None:
        from_config = yaml.safe_load(open(args.compute_config, "r"))
        from_config = {k: v for k, v in from_config.items() if v is not None}

    for key in ["account", "partition", "cpus", "gpus", "gpu_type", "mem", "nodes", "time", "nodelist", "exclude"]:
        if vars(args)[key] is not None:
            from_config[key] = vars(args)[key]

    print(">>> Compute config:")
    pprint.pprint(from_config)

    executor = submitit.AutoExecutor(folder=os.path.join(save_dir, "submitit"))

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
        slurm_exclude=from_config.get("exclude") or None,
        slurm_mem=from_config.get("mem", "96gb") or "96gb",
        slurm_time=from_config.get("time", 60),
        slurm_job_name=job_name,
        slurm_additional_parameters=additional_parameters,
    )

    def launch_training(args, from_config, cmd, save_dir, project_root, container_config=None):
        """Runs inside the SLURM allocation."""
        import os
        import subprocess
        import submitit

        # project_root is the code snapshot isolate_code took at submission time
        os.chdir(project_root)
        os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")

        nodes = from_config.get("nodes", 1)
        gpus = from_config.get("gpus", 1)

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

        full_cmd = base_cmd + cmd + [f"train.data_dir={save_dir}"]

        # a brace group runs in this shell (cd and exports apply): heartbeat backgrounded, training's exit code kept
        def wrap_with_heartbeat(train_cmd_str):
            hb = "python scripts/gpu_heartbeat.py > /tmp/gpu_heartbeat.log 2>&1 & HEARTBEAT_PID=$!"
            return f"{{ {hb}; {train_cmd_str}; TRAIN_EXIT=$?; kill $HEARTBEAT_PID 2>/dev/null; exit $TRAIN_EXIT; }}"

        if container_config is not None:
            # the /scratch venv rather than /ext3/env.sh: every package read stays off fuse2fs
            scratch_dir = os.environ.get("SCRATCH_DIR", "/scratch/" + os.environ.get("USER", ""))
            venv_path = os.environ.get("VENV_PATH", f"{scratch_dir}/venvs/pufferdrive")
            env_setup = f"source {venv_path}/bin/activate && export PYTHONNOUSERSITE=1"
            # Redirect caches to scratch to avoid home quota issues
            cache_exports = (
                f"export XDG_CACHE_HOME={scratch_dir}/cache && "
                f"export WANDB_CACHE_DIR={scratch_dir}/wandb_cache && "
                f"export WANDB_CONFIG_DIR={scratch_dir}/wandb_config && "
                f"export WANDB_DATA_DIR={scratch_dir}/wandb_data && "
                f"export WANDB_DIR={scratch_dir}/wandb_data && "
                f"mkdir -p {scratch_dir}/cache"
            )
            # pins the wandb server and org over the container's default login
            if args.wandb_base_url:
                cache_exports += f" && export WANDB_BASE_URL={args.wandb_base_url}"
            if args.wandb_entity:
                cache_exports += f" && export WANDB_ENTITY={args.wandb_entity}"
            train_str = " ".join(full_cmd)
            if args.heartbeat:
                train_str = wrap_with_heartbeat(train_str)
            inner_cmd = f"{env_setup} && {cache_exports} && cd {project_root} && {train_str}"
            full_cmd = [
                "singularity",
                "exec",
                "--nv",
                "--overlay",
                container_config["overlay"] + ":ro",  # Read-only overlay for running
            ]
            # Bind mount SSL certificates for TLS verification (wandb, etc.)
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
            # No container: still need to wrap in bash -c so the brace group parses.
            train_str = " ".join(full_cmd)
            full_cmd = ["bash", "-c", wrap_with_heartbeat(train_str)]

        print(f">>> Job: {job_name}")
        print(f">>> Working directory: {project_root}")
        print(f">>> Container: {container_config is not None}")
        print(f">>> Command: {' '.join(full_cmd)}")
        subprocess.run(full_cmd, check=True)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    container_config = None
    if args.container:
        container_config = {
            "image": args.container_image,
            "overlay": args.container_overlay,
        }
        print(f">>> Container mode enabled: {container_config['image']}")

    if not dry:
        # snapshot now, so edits made while the job queues do not leak into the run
        isolated_root = isolate_code(project_root, save_dir)
        print(f">>> Code snapshot: {isolated_root}")
        job = executor.submit(launch_training, args, from_config, command, save_dir, isolated_root, container_config)
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
