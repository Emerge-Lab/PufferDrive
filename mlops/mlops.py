#!/usr/bin/env python3
"""
Unified MLOps CLI for PufferDrive training on GCP Vertex AI.

Usage:
    # Launch a single run within an experiment
    python mlops/mlops.py launch my_experiment my_run --machine T4

    # Launch multiple runs from config file
    python mlops/mlops.py launch-batch mlops/configs/experiments.yaml

    # Build and push the base image
    python mlops/mlops.py build-image

    # Sync data to/from GCS
    python mlops/mlops.py sync-data gs://bucket/path local/path --download
"""

import argparse
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from google.cloud import aiplatform
from google.cloud.aiplatform.compat.types import reservation_affinity_v1 as reservation_affinity
from pathlib import Path
import yaml

# Suppress noisy Google Cloud logs
os.environ["GRPC_VERBOSITY"] = "ERROR"
import logging

logging.getLogger("google.api_core").setLevel(logging.ERROR)
logging.getLogger("google.auth").setLevel(logging.ERROR)
logging.getLogger("google.cloud").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GCPConfig:
    """GCP project configuration."""

    project_id: str
    staging_bucket: str
    service_account: str
    default_dataset_path: str
    training_container_registry: str
    use_reservation: bool = False
    location: str = "europe-west4"
    builder_image: str = "europe-west4-docker.pkg.dev/valeo-cp2386-dev/build-images/pufferdrive_cuda12.8_py3.12:latest"
    tensorboard_resource: str | None = None


GCP_CONFIGS = {
    "DRILAX": GCPConfig(
        project_id="valeo-cp2386-dev",
        staging_bucket="gs://valeo-cp2386-runs",
        service_account="valeo-cp2386-dev@appspot.gserviceaccount.com",
        default_dataset_path="/gcs/valeo-cp2386-datasets/pufferdrive/v1.0/womd/training/",
        training_container_registry="europe-west4-docker.pkg.dev/valeo-cp2386-dev/training-images",
        tensorboard_resource="projects/858441583698/locations/europe-west4/tensorboards/6648324600696406016",
    ),
    "NOA": GCPConfig(
        project_id="valeo-cp2879-dev",
        staging_bucket="gs://valeo-cp2879-driving-policy",
        service_account="valeo-cp2879-dev-job@valeo-cp2879-dev.iam.gserviceaccount.com",
        default_dataset_path="/gcs/valeo-cp2879-driving-policy/datasets/carla/",
        training_container_registry="europe-west4-docker.pkg.dev/valeo-cp2879-dev/driving-policy/training-images",
        use_reservation=False,
        tensorboard_resource=None,
    ),
}


def get_config(project):
    if project not in GCP_CONFIGS:
        raise ValueError(f"Unknown project: {project}. Available: {list(GCP_CONFIGS.keys())}")
    return GCP_CONFIGS[project]


@dataclass
class MachineSpec:
    """Machine type specification."""

    machine_type: str
    accelerator_type: str
    accelerator_count: int
    cuda_arch: str
    boot_disk_type: str = "pd-ssd"


# Machine type presets
MACHINE_PRESETS: dict[str, MachineSpec] = {
    "T4": MachineSpec("n1-standard-16", "NVIDIA_TESLA_T4", 1, "7.5"),
    "T4_2": MachineSpec("n1-standard-32", "NVIDIA_TESLA_T4", 2, "7.5"),
    "T4_4": MachineSpec("n1-standard-32", "NVIDIA_TESLA_T4", 4, "7.5"),
    "T4_8": MachineSpec("n1-standard-64", "NVIDIA_TESLA_T4", 8, "7.5"),
    "L4": MachineSpec("g2-standard-16", "NVIDIA_L4", 1, "8.9"),
    "L4_2": MachineSpec("g2-standard-32", "NVIDIA_L4", 2, "8.9"),
    "L4_4": MachineSpec("g2-standard-64", "NVIDIA_L4", 4, "8.9"),
    "L4_8": MachineSpec("g2-standard-96", "NVIDIA_L4", 8, "8.9"),
    "A100": MachineSpec("a2-highgpu-1g", "NVIDIA_A100", 1, "8.0"),
    "A100_2": MachineSpec("a2-highgpu-2g", "NVIDIA_A100", 2, "8.0"),
    "A100_4": MachineSpec("a2-megagpu-16g", "NVIDIA_A100", 4, "8.0"),
    "A100_8": MachineSpec("a2-megagpu-16g", "NVIDIA_A100", 8, "8.0"),
    "RTX6000": MachineSpec("g4-standard-48", "NVIDIA_RTX_PRO_6000", 1, "12.0", "hyperdisk-balanced"),
    "RTX6000_2": MachineSpec("g4-standard-96", "NVIDIA_RTX_PRO_6000", 2, "12.0", "hyperdisk-balanced"),
    "RTX6000_4": MachineSpec("g4-standard-192", "NVIDIA_RTX_PRO_6000", 4, "12.0", "hyperdisk-balanced"),
    "RTX6000_8": MachineSpec("g4-standard-384", "NVIDIA_RTX_PRO_6000", 8, "12.0", "hyperdisk-balanced"),
}


@dataclass
class Run:
    """Run configuration within an experiment."""

    experiment: str
    name: str
    machine: str = "T4"
    dataset_path: str = ""
    eval_dataset_path: str = ""
    project: str = "DRILAX"
    params: dict = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)


# =============================================================================
# Utility Functions
# =============================================================================


def run_command(command: list[str], description: str, verbose: bool = False) -> bool:
    """Run a shell command with nice output."""
    print(f"🔧 {description}")
    print(f"   $ {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=not verbose, text=True)
        print("   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print("   ❌ Failed")
        if not verbose and e.stdout:
            print(f"   STDOUT: {e.stdout}")
        if not verbose and e.stderr:
            print(f"   STDERR: {e.stderr}")
        return False


@contextmanager
def setup_dockerignore():
    """Context manager to copy .dockerignore to root for build and clean up after."""
    dockerignore_src = "mlops/dockerfile/.dockerignore"
    dockerignore_dst = ".dockerignore"

    if os.path.exists(dockerignore_src):
        shutil.copy(dockerignore_src, dockerignore_dst)

    try:
        yield
    finally:
        if os.path.exists(dockerignore_dst):
            os.remove(dockerignore_dst)


@contextmanager
def stage_model_artifacts(extra_args: list[str]):
    """
    Checks if --load-model-path is present. If so:
    1. Copies the model and config.yaml to mlops/staging_models/
    2. Updates the arg path to point to the container location.
    3. Cleans up after the build.
    """
    staging_dir = Path("mlops/staging_models")

    # 1. Clean start: Create staging dir (even if empty, to satisfy Docker COPY)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    (staging_dir / ".keep").touch()  # Prevent Docker COPY failure on empty dir

    # Check for argument
    model_arg_idx = -1
    if "--load-model-path" in extra_args:
        print("🔍 Detected --load-model-path argument, preparing model artifacts...")
        model_arg_idx = extra_args.index("--load-model-path")

    try:
        # If model arg exists and has a value
        if model_arg_idx != -1 and model_arg_idx + 1 < len(extra_args):
            local_path_str = extra_args[model_arg_idx + 1]
            local_pt = Path(local_path_str)

            if not local_pt.exists():
                print(f"❌ Error: Model file not found: {local_pt}")
                sys.exit(1)

            # Copy .pt file
            print(f"📦 Staging model for Docker build: {local_pt.name}")
            shutil.copy(local_pt, staging_dir / local_pt.name)

            # Copy config.yaml (check parent dir)
            local_config = local_pt.parent.parent / "config.yaml"
            if local_config.exists():
                shutil.copy(local_config, staging_dir / "config.yaml")
                print(f"📦 Staged config: {local_config.name}")
            else:
                print("❌ Error: Config file not found: config.yaml")

            # The file will be at /pufferdrive/preload_model/<filename>
            container_path = f"/pufferdrive/preload_model/{local_pt.name}"
            extra_args[model_arg_idx + 1] = container_path
            print(f"🔄 Remapped --load-model-path to container path: {container_path}")

        yield

    finally:
        # Cleanup staging area after build/push is done
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


# =============================================================================
# Vertex AI Job Submission
# =============================================================================


def submit_vertex_job(
    config: GCPConfig,
    display_name: str,
    container_uri: str,
    machine: MachineSpec,
    experiment_name: str,
    run_name: str,
    project: str,
    dataset_path: str,
    eval_dataset_path: str = "",
    extra_args: list[str] | None = None,
    mode: str = "train",
) -> None:
    """Submit a training job to Vertex AI."""

    experiment_name = experiment_name.replace("_", "-")
    run_name = run_name.replace("_", "-")
    if config.project_id == "valeo-cp2879-dev":
        cloud_path = f"{config.staging_bucket}/runs/{experiment_name}/{run_name}"
    else:
        cloud_path = f"{config.staging_bucket}/{experiment_name}/{run_name}"

    aiplatform.init(
        project=config.project_id,
        location=config.location,
        staging_bucket=config.staging_bucket,
        # experiment=experiment_name,
        # experiment_tensorboard=config.tensorboard_resource,
    )

    # # Explicitly register the run in the Vertex ML Metadata store
    # try:
    #     # First, try to fetch it if it already exists (e.g., re-running a failed job)
    #     aiplatform.start_run(run=run_name, resume=True)
    # except Exception:
    #     # If it throws a 404 (doesn't exist yet), create it fresh
    #     aiplatform.start_run(run=run_name)
    # finally:
    #     # Always end the local session so the CustomJob can take over
    #     aiplatform.end_run()

    # Build container arguments
    container_args = ["--mode", mode, "--accelerator-count", str(machine.accelerator_count)]
    if not dataset_path:
        dataset_path = config.default_dataset_path
    job_extra_args = list(extra_args) if extra_args else []
    job_extra_args.extend(["--env.map-dir", dataset_path])

    # Add eval dataset path if provided
    if eval_dataset_path:
        job_extra_args.extend(["--eval.map-dir", eval_dataset_path])

    container_args.extend(job_extra_args)

    # Machine spec with optional reservation affinity
    machine_spec = {
        "machine_type": machine.machine_type,
        "accelerator_type": machine.accelerator_type,
        "accelerator_count": machine.accelerator_count,
    }
    if config.use_reservation:
        machine_spec["reservation_affinity"] = reservation_affinity.ReservationAffinity(
            reservation_affinity_type=reservation_affinity.ReservationAffinity.Type.ANY_RESERVATION
        )

    worker_pool_specs = [
        {
            "machine_spec": machine_spec,
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": machine.boot_disk_type,
                "boot_disk_size_gb": 100,
            },
            "container_spec": {
                "image_uri": container_uri,
                "args": container_args,
                "env": [
                    {
                        "name": "TORCH_CUDA_ARCH_LIST",
                        "value": machine.cuda_arch,
                    },
                    {
                        "name": "CLOUD",
                        "value": "1",
                    },
                    {
                        "name": "CLOUD_PATH",
                        "value": cloud_path,
                    },
                ],
            },
        }
    ]

    if sha := os.environ.get("GITHUB_SHA"):
        labels = {"commit_sha": sha}
    else:
        labels = {"username": os.getlogin()}

    # Create and submit job
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=config.staging_bucket,
        base_output_dir=cloud_path,
        labels=labels,
        project=config.project_id,
        location=config.location,
    )

    print(f"📤 Submitting job: {display_name}")
    job.submit(
        service_account=config.service_account,
        enable_web_access=False,
        scheduling_strategy="STANDARD",
        tensorboard=config.tensorboard_resource,
        disable_retries=True,
        # experiment=experiment_name,
        # experiment_run=run_name,
    )
    print("   ✅ Job submitted successfully")


# =============================================================================
# Docker Operations
# =============================================================================


def build_and_push_image(config: GCPConfig, tag: str, verbose: bool = False, gpu_arch: str = "") -> str:
    """Build and push the training Docker image."""
    if not gpu_arch.strip():
        print("❌ GPU_ARCH is empty. Set machine.cuda_arch explicitly for this build.")
        sys.exit(1)

    if os.environ.get("GITHUB_SHA"):
        username = "pr-test"
    else:
        username = os.getlogin()
    container_uri = f"{config.training_container_registry}/{username}:{tag}"

    with setup_dockerignore():
        build_cmd = [
            "docker",
            "build",
            "--tag",
            container_uri,
            "--build-arg",
            f"GPU_ARCH={gpu_arch}",
            "--file",
            "mlops/dockerfile/launch.dockerfile",
            ".",
        ]
        if not run_command(build_cmd, "Building Docker image", verbose):
            sys.exit(1)

    # Push image
    push_cmd = ["docker", "push", container_uri]
    if not run_command(push_cmd, "Pushing Docker image", verbose):
        print("💡 Tip: Run `gcloud auth configure-docker europe-west4-docker.pkg.dev`")
        sys.exit(1)

    return container_uri


def build_base_image(config: GCPConfig, verbose: bool = False) -> None:
    """Build and push the base builder image."""

    with setup_dockerignore():
        build_cmd = [
            "docker",
            "build",
            "--tag",
            config.builder_image,
            "--file",
            "mlops/dockerfile/build.dockerfile",
            ".",
        ]
        if not run_command(build_cmd, "Building base image", verbose):
            sys.exit(1)

    push_cmd = ["docker", "push", config.builder_image]
    if not run_command(push_cmd, "Pushing base image", verbose):
        sys.exit(1)

    print(f"✅ Base image ready: {config.builder_image}")


# =============================================================================
# Experiment Loading
# =============================================================================


def load_runs_from_yaml(path: str, default_project: str = "DRILAX") -> list[Run]:
    """Load runs from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    runs = []
    for exp_data in data.get("experiments", []):
        experiment_name = exp_data["name"]
        experiment_machine = exp_data.get("machine", "T4")
        experiment_dataset = exp_data.get("dataset_path", "")
        experiment_eval_dataset = exp_data.get("eval_dataset_path", "")
        experiment_project = exp_data.get("project", default_project)
        experiment_params = exp_data.get("params", {})
        experiment_extra_args = exp_data.get("extra_args", [])

        # If runs are defined, create a Run for each
        if "runs" in exp_data:
            for run_data in exp_data["runs"]:
                # Merge experiment-level params with run-level params (run takes precedence)
                merged_params = {**experiment_params, **run_data.get("params", {})}
                merged_extra_args = experiment_extra_args + run_data.get("extra_args", [])
                runs.append(
                    Run(
                        experiment=experiment_name,
                        name=run_data["name"],
                        machine=run_data.get("machine", experiment_machine),
                        dataset_path=run_data.get("dataset_path", experiment_dataset),
                        eval_dataset_path=run_data.get("eval_dataset_path", experiment_eval_dataset),
                        project=run_data.get("project", experiment_project),
                        params=merged_params,
                        extra_args=merged_extra_args,
                    )
                )
        else:
            # No runs defined, treat experiment as a single run with same name
            runs.append(
                Run(
                    experiment=experiment_name,
                    name=experiment_name,
                    machine=experiment_machine,
                    dataset_path=experiment_dataset,
                    eval_dataset_path=experiment_eval_dataset,
                    project=experiment_project,
                    params=experiment_params,
                    extra_args=experiment_extra_args,
                )
            )

    return runs


def params_to_args(params: dict) -> list[str]:
    """Convert parameter dictionary to command-line arguments."""
    args = []
    for key, value in params.items():
        # Convert param.subparam to --param.subparam format
        args.extend([f"--{key}", str(value)])
    return args


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_launch(args) -> None:
    """Launch a single run within an experiment."""
    config = get_config(args.project)

    # Validate machine type
    if args.machine not in MACHINE_PRESETS:
        print(f"❌ Unknown machine type: {args.machine}")
        print(f"   Available: {', '.join(MACHINE_PRESETS.keys())}")
        sys.exit(1)

    machine = MACHINE_PRESETS[args.machine]

    with stage_model_artifacts(args.extra_args):
        # Generate tag and build image
        tag = datetime.now().strftime("%d%m%H%M%S")
        container_uri = build_and_push_image(config, tag, args.verbose, machine.cuda_arch)

        submit_vertex_job(
            config=config,
            display_name=f"pufferdrive/{args.experiment_name}/{args.run_name}",
            container_uri=container_uri,
            machine=machine,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            project=args.project,
            dataset_path=args.dataset_path or "",
            eval_dataset_path=args.eval_dataset_path or "",
            extra_args=args.extra_args,
            mode=args.mode,
        )

    print(
        f"\n🎉 Run '{args.run_name}' in experiment '{args.experiment_name}' launched on {args.machine} ({args.project})"
    )


def cmd_launch_batch(args) -> None:
    """Launch multiple runs from a YAML config file."""
    # Load runs with CLI project as default
    runs = load_runs_from_yaml(args.config_file, default_project=args.project)
    if not runs:
        print("❌ No runs found in config file")
        sys.exit(1)

    print(f"📋 Found {len(runs)} runs")

    # Validate all machine types and projects upfront
    valid_runs = []
    for run in runs:
        if run.machine not in MACHINE_PRESETS:
            print(f"⚠️  Unknown machine type '{run.machine}' for '{run.experiment}/{run.name}', skipping...")
        elif run.project not in GCP_CONFIGS:
            print(f"⚠️  Unknown project '{run.project}' for '{run.experiment}/{run.name}', skipping...")
        else:
            valid_runs.append(run)

    if not valid_runs:
        print("❌ No valid runs to launch")
        sys.exit(1)

    # Group runs by project to build one image per project
    runs_by_project = {}
    for run in valid_runs:
        runs_by_project.setdefault(run.project, []).append(run)

    # Build and push images for each project
    tag = datetime.now().strftime("%d%m%H%M%S")
    container_uris = {}

    with stage_model_artifacts([]):
        for project in runs_by_project:
            config = get_config(project)
            gpu_arch = " ".join(sorted({MACHINE_PRESETS[run.machine].cuda_arch for run in runs_by_project[project]}))
            print(f"\n📦 Building image for {project}...")
            container_uris[project] = build_and_push_image(config, tag, args.verbose, gpu_arch)

    print(f"\n🚀 Submitting {len(valid_runs)} runs sequentially...")

    for i, run in enumerate(valid_runs, 1):
        print(f"\n[{i}/{len(valid_runs)}] Processing: {run.experiment}/{run.name} ...")
        try:
            config = get_config(run.project)
            machine = MACHINE_PRESETS[run.machine]
            # Construct args specifically for this run
            extra_args = params_to_args(run.params) + run.extra_args
            display_name = f"pufferdrive/{run.experiment}/{run.name}"

            submit_vertex_job(
                config=config,
                display_name=display_name,
                container_uri=container_uris[run.project],
                machine=machine,
                experiment_name=run.experiment,
                run_name=run.name,
                project=run.project,
                dataset_path=run.dataset_path,
                eval_dataset_path=run.eval_dataset_path,
                extra_args=extra_args,
            )
        except Exception as e:
            print(f"   ❌ {run.experiment}/{run.name} failed: {e}")
            continue

    print(f"\n🎉 All {len(valid_runs)} runs submitted!")


def cmd_train_test(args) -> None:
    """Launch a training test for CI."""
    config_file = "mlops/configs/smoke-tests.yaml"
    print(f"🚀 Launching CI training test from {config_file}")

    # Load runs with CLI project as default
    runs = load_runs_from_yaml(config_file)

    # Group runs by project to build one image per project
    runs_by_project = {}
    for run in runs:
        runs_by_project.setdefault(run.project, []).append(run)

    tag = os.environ.get("GITHUB_SHA") or datetime.now().strftime("%d%m%H%M%S")
    short_sha = (
        os.environ.get("GITHUB_SHA") or subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )[:6]
    container_uris = {}

    with stage_model_artifacts([]):  # No extra args needed here
        for project in runs_by_project:
            config = get_config(project)
            gpu_arch = " ".join(sorted({MACHINE_PRESETS[run.machine].cuda_arch for run in runs_by_project[project]}))
            print(f"\n📦 Building image for {project}...")
            container_uris[project] = build_and_push_image(config, tag, args.verbose, gpu_arch)

    print(f"\n🚀 Submitting {len(runs)} CI test runs sequentially...")

    for i, run in enumerate(runs, 1):
        run_name = f"{short_sha}-{run.name}"
        print(f"\n[{i}/{len(runs)}] Processing: {run.experiment}/{run_name} ...")
        config = get_config(run.project)
        machine = MACHINE_PRESETS[run.machine]
        extra_args = params_to_args(run.params) + run.extra_args
        display_name = f"pufferdrive-ci/{run.experiment}/{run_name}"
        submit_vertex_job(
            config=config,
            display_name=display_name,
            container_uri=container_uris[run.project],
            machine=machine,
            experiment_name=run.experiment,
            run_name=run_name,
            project=run.project,
            dataset_path=run.dataset_path,
            eval_dataset_path=run.eval_dataset_path,
            extra_args=extra_args,
        )

    print(f"\n🎉 All {len(runs)} CI test runs submitted!")


def cmd_build_image(args) -> None:
    """Build and push the base builder image."""
    config = get_config(args.project)
    build_base_image(config, args.verbose)


# =============================================================================
# Main
# =============================================================================


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--project",
        "-p",
        default="DRILAX",
        choices=list(GCP_CONFIGS.keys()),
        help="GCP project to use (default: DRILAX)",
    )

    parser = argparse.ArgumentParser(
        description="PufferDrive MLOps CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Launch single run
    launch_parser = subparsers.add_parser("launch", parents=[common], help="Launch a single run within an experiment")
    launch_parser.add_argument("experiment_name", help="Name for the experiment (groups related runs)")
    launch_parser.add_argument("run_name", help="Name for this specific run")
    launch_parser.add_argument(
        "--machine", "-m", default="T4", choices=list(MACHINE_PRESETS.keys()), help="Machine type preset"
    )
    launch_parser.add_argument("--dataset-path", "-d", help="GCS path to training dataset")
    launch_parser.add_argument("--eval-dataset-path", help="GCS path to eval dataset")
    launch_parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    launch_parser.add_argument("--mode", choices=["train", "sweep"], default="train", help="Run mode: train or sweep")
    launch_parser.set_defaults(func=cmd_launch)

    # Launch batch runs
    batch_parser = subparsers.add_parser("launch-batch", parents=[common], help="Launch multiple runs from config")
    batch_parser.add_argument("config_file", help="YAML config file with experiments")
    batch_parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    batch_parser.set_defaults(func=cmd_launch_batch)

    # Train test
    train_test_parser = subparsers.add_parser("train-test", help="Launch a training test for CI")
    train_test_parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    train_test_parser.set_defaults(func=cmd_train_test)

    # Build base image
    build_parser = subparsers.add_parser("build-image", parents=[common], help="Build and push base Docker image")
    build_parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    build_parser.set_defaults(func=cmd_build_image)

    args, extra_args = parser.parse_known_args()

    # Store extra args for passing to the container
    args.extra_args = extra_args

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
