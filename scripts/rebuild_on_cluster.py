"""Submit a SLURM job to rebuild the PufferDrive C extension inside the Singularity container.

Avoids the nested quoting hell of sbatch --wrap by writing a standalone bash script
to a temp location and sbatch-ing that file. The script runs `setup.py build_ext`
inside the container overlay where torch is installed.

Example:
    python scripts/rebuild_on_cluster.py
    python scripts/rebuild_on_cluster.py --account torch_pr_924_general
    python scripts/rebuild_on_cluster.py --project-root /scratch/$USER/code/PufferDrive --wait
"""

import argparse
import os
import subprocess
import sys
import time


DEFAULT_IMAGE = "/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"


def parse_args():
    user = os.environ.get("USER", "")
    parser = argparse.ArgumentParser(description="Rebuild PufferDrive C extension on SLURM cluster")
    parser.add_argument("--account", default="torch_pr_924_general", help="SLURM account")
    parser.add_argument("--user", default=user, help="Cluster username (default: $USER)")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Path to PufferDrive on the cluster (default: /scratch/<user>/code/PufferDrive)",
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Singularity overlay path (default: /scratch/<user>/images/PufferDrive/overlay-15GB-500K.ext3)",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Singularity image path")
    parser.add_argument("--time", default="15", help="SLURM time limit in minutes")
    parser.add_argument("--mem", default="16gb", help="SLURM memory")
    parser.add_argument("--cpus", default="8", help="SLURM cpus-per-task")
    parser.add_argument("--wait", action="store_true", help="Poll until the job finishes and print its log")
    parser.add_argument("--dry", action="store_true", help="Print the script and sbatch command without submitting")
    return parser.parse_args()


def build_rebuild_script(project_root: str, overlay: str, image: str) -> str:
    """Return a bash script that runs the rebuild inside the container.

    Matches the training launcher's invocation exactly: read-only overlay mount,
    no fakeroot, sources /ext3/env.sh which activates the venv/conda env with torch
    and other deps installed.
    """
    inner = (
        "source /ext3/env.sh && "
        f"cd {project_root} && "
        "which python3 && "
        'python3 -c "import torch; print(\\"torch:\\", torch.__version__)" && '
        "python3 setup.py build_ext --inplace --force && "
        'python3 -c "from pufferlib.ocean.drive import binding; print(\\"C binding loaded OK\\")"'
    )
    return (
        "#!/bin/bash\n"
        "set -e\n"
        f"cd {project_root}\n"
        f"singularity exec --nv \\\n"
        f"    --overlay {overlay}:ro \\\n"
        f"    {image} \\\n"
        f"    bash -c '{inner}'\n"
    )


def run_ssh(cmd: str, check: bool = True) -> str:
    """Run a command on the cluster via ssh and return stdout."""
    result = subprocess.run(["ssh", "torch", cmd], capture_output=True, text=True)
    if check and result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"ssh command failed: {cmd}")
    return result.stdout


def main():
    args = parse_args()
    project_root = args.project_root or f"/scratch/{args.user}/code/PufferDrive"
    overlay = args.overlay or f"/scratch/{args.user}/images/PufferDrive/overlay-15GB-500K.ext3"

    script = build_rebuild_script(project_root, overlay, args.image)
    # Use a scratch location for script and log so they survive the compute node.
    log_dir = f"/scratch/{args.user}/rebuild_logs"
    script_path = f"{log_dir}/rebuild_pufferdrive.sh"
    log_path = f"{log_dir}/rebuild_pufferdrive_%j.log"
    run_ssh(f"mkdir -p {log_dir}")

    if args.dry:
        print("=== rebuild script ===")
        print(script)
        print(f"=== sbatch destination: {script_path} ===")
        print(f"=== log path: {log_path} ===")
        return 0

    # Write script to cluster via ssh
    subprocess.run(
        ["ssh", "torch", f"cat > {script_path} && chmod +x {script_path}"],
        input=script,
        text=True,
        check=True,
    )

    # Submit
    sbatch_cmd = (
        f"sbatch --account={args.account} --gres=gpu:1 "
        f"--cpus-per-task={args.cpus} --mem={args.mem} --time={args.time} "
        f"-o {log_path} {script_path}"
    )
    stdout = run_ssh(sbatch_cmd)
    print(stdout.strip())

    # Parse job id from "Submitted batch job 12345"
    parts = stdout.strip().split()
    if len(parts) < 4 or not parts[-1].isdigit():
        print("Could not parse job id from sbatch output", file=sys.stderr)
        return 1
    job_id = parts[-1]
    resolved_log = log_path.replace("%j", job_id)
    print(f"Job ID: {job_id}")
    print(f"Log: {resolved_log}")

    if not args.wait:
        return 0

    # Poll for completion
    print("Waiting for job to finish...")
    while True:
        time.sleep(20)
        state = run_ssh(
            f"sacct -j {job_id} --format=State -n -P 2>/dev/null | head -1",
            check=False,
        ).strip()
        if not state:
            print("  (job not yet registered in sacct)")
            continue
        print(f"  state: {state}")
        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            break

    print()
    print("=== log ===")
    log_content = run_ssh(f"cat {resolved_log} 2>/dev/null || echo '(no log)'", check=False)
    print(log_content)
    return 0 if state == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
