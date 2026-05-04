"""Submit a SLURM job to rebuild the PufferDrive C extension inside the Singularity container.

Run this on the cluster login node (where sbatch is available). It writes a standalone
bash script to /scratch/$USER/rebuild_logs/ and submits it. The script runs
`setup.py build_ext` inside the container overlay where torch is installed.

Avoids the nested quoting hell of `sbatch --wrap` by writing the script to a file first.

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

    Matches submit_cluster.py's container invocation: read-only overlay mount,
    no fakeroot, sources /ext3/env.sh which activates the venv with torch installed.
    """
    # TORCH_CUDA_ARCH_LIST must cover every GPU type the cluster might schedule jobs on:
    #   8.0 = A100, 8.9 = L40S, 9.0 = H100/H200
    # Without this, the torch CUDA extension is built only for the build node's GPU
    # arch and jobs that land on other GPU types crash with
    # "no kernel image is available for execution on the device".
    # NCCL fix: prepend torch's bundled NCCL dir to LD_LIBRARY_PATH so torch >= 2.10
    # finds the right libnccl (with ncclCommShrink) instead of the sif's older
    # /usr/lib/libnccl.so.2.25.1. See submit_cluster.py for the full story. The
    # brace group + ; true makes a missing libnccl non-fatal so we still attempt
    # the build (and fail with a clearer error if torch genuinely can't import).
    nccl_fix = (
        "{ NCCL_DIR=$(compgen -G '/ext3/miniforge3/lib/python3.*/site-packages/nvidia/nccl/lib' | head -1); "
        '[ -n "$NCCL_DIR" ] && [ -d "$NCCL_DIR" ] && export LD_LIBRARY_PATH="$NCCL_DIR:${LD_LIBRARY_PATH:-}"; '
        "true; }"
    )
    inner = (
        f"source /ext3/env.sh && {nccl_fix} && "
        'export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" && '
        f"cd {project_root} && "
        "which python3 && "
        'python3 -c "import torch; print(\\"torch:\\", torch.__version__)" && '
        "python3 setup.py build_ext --inplace --force && "
        'python3 -c "from pufferlib import _C; print(\\"_C loaded, gpu=\\" + str(_C.gpu))"'
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


def run(cmd: str, check: bool = True, capture: bool = True) -> str:
    """Run a shell command on this host. Returns stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and result.returncode != 0:
        if capture:
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed: {cmd}")
    return result.stdout if capture else ""


def main():
    args = parse_args()
    project_root = args.project_root or f"/scratch/{args.user}/code/PufferDrive"
    overlay = args.overlay or f"/scratch/{args.user}/images/PufferDrive/overlay-15GB-500K.ext3"

    script = build_rebuild_script(project_root, overlay, args.image)
    log_dir = f"/scratch/{args.user}/rebuild_logs"
    script_path = f"{log_dir}/rebuild_pufferdrive.sh"
    log_path = f"{log_dir}/rebuild_pufferdrive_%j.log"
    os.makedirs(log_dir, exist_ok=True)

    if args.dry:
        print("=== rebuild script ===")
        print(script)
        print(f"=== sbatch destination: {script_path} ===")
        print(f"=== log path: {log_path} ===")
        return 0

    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    sbatch_cmd = (
        f"sbatch --account={args.account} --gres=gpu:1 "
        f"--cpus-per-task={args.cpus} --mem={args.mem} --time={args.time} "
        f"-o {log_path} {script_path}"
    )
    stdout = run(sbatch_cmd)
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

    print("Waiting for job to finish...")
    state = ""
    while True:
        time.sleep(20)
        state = run(
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
    log_content = run(f"cat {resolved_log} 2>/dev/null || echo '(no log)'", check=False)
    print(log_content)
    return 0 if state == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
