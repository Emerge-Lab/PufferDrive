import argparse
import subprocess
import os
from datetime import datetime
from typing import Optional

from google.cloud import aiplatform
from google.api_core import exceptions

def run_command(command: list[str], error_message: str, verbose: bool = False):
    """Runs a command, checks for errors, and prints helpful output."""
    print(f"🚀 Running command: {' '.join(command)}")
    try:
        # If verbose, stream output directly. Otherwise, capture it for display on error.
        subprocess.run(command, check=True, capture_output=not verbose, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ {error_message}")
        # If output was captured, print it. Otherwise, it was already streamed.
        if not verbose:
            print(f"   STDOUT: {e.stdout}")
            print(f"   STDERR: {e.stderr}")
        else:
            # When not capturing, output is already on the console.
            print("   See the output above for details.")
        exit(1)
    print("✅ Command successful.")

def create_custom_job(
    project: str,
    location: str,
    staging_bucket: str,
    display_name: str,
    container_uri: str,
    service_account: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    dataset_path: Optional[str] = None,
    boot_disk_size_gb: int = 100,
    timeout_seconds: int = 604800,  # 7 days
    experiment: Optional[str] = None,
    experiment_run: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
) -> None:
    """Creates and runs a Vertex AI Custom Job using a pre-built container."""
    # If an experiment is specified, ensure it exists before initializing.
    if experiment:
        try:
            # This will check if the experiment exists by attempting to get it.
            aiplatform.Experiment(
                experiment_name=experiment, project=project, location=location
            )
        except exceptions.NotFound:
            print(f"🧪 Experiment '{experiment}' not found. Creating it now.")
            aiplatform.Experiment.create(
                experiment_name=experiment, project=project, location=location
            )
    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)

    # Define the container spec, passing the dataset path as an argument if provided.
    container_spec = {"image_uri": container_uri}
    if dataset_path:
        container_spec["args"] = [dataset_path]

    # Define the hardware resources for the job, mirroring the YAML configuration.
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": boot_disk_size_gb,
            },
            "container_spec": container_spec,
        }
    ]

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        labels=labels,
    )

    job.run(
        service_account=service_account,
        timeout=timeout_seconds,
        experiment=experiment,
        experiment_run=experiment_run,
        # Set sync=True to run the job synchronously.
        sync=False,
    )

    print(f"✅ Custom job '{display_name}' submitted successfully.")
    print(f"   View job: https://console.cloud.google.com/ai/platform/jobs/{job.resource_name.split('/')[-1]}?project={project}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build, push, and run a PufferDrive training job on Vertex AI.")
    parser.add_argument("--project", required=True, help="Your Google Cloud project ID.")
    parser.add_argument("--location", required=True, help="The GCP region for the job (e.g., 'us-central1').")
    parser.add_argument("--staging-bucket", required=True, help="GCS bucket for staging artifacts (e.g., 'gs://my-bucket').")
    parser.add_argument("--display-name", required=True, help="Display name for the custom job.")
    parser.add_argument("--container-uri", required=True, help="URI for the Docker image (e.g., 'us-central1-docker.pkg.dev/my-project/repo/image:$USER:tag').")
    parser.add_argument("--service-account", required=True, help="Service account for the job.")
    parser.add_argument("--machine-type", default="g2-standard-16", help="The type of machine to use for training.")
    parser.add_argument("--accelerator-type", default="NVIDIA_L4", help="The type of accelerator to use.")
    parser.add_argument("--accelerator-count", type=int, default=1, help="The number of accelerators.")
    parser.add_argument("--dataset-path", help="Optional: GCS path to the dataset (e.g., gs://bucket/data).")
    parser.add_argument("--experiment", help="Optional: Name of the Vertex AI experiment.")
    parser.add_argument("--experiment_run", type=str, default=None, help="Optional: Name of the experiment run.")
    parser.add_argument("--labels", nargs='*', help="Labels for the job in key=value format (e.g., user=name env=prod).")
    parser.add_argument("--verbose-subprocess", action="store_true", help="If set, shows the live output of subprocesses like docker build/push.")
    args = parser.parse_args()

    # 1. Argument parsing
    # 1.1 Expand environment variables in container URI (e.g., $DOMAIN)
    container_uri = os.path.expandvars(args.container_uri)
    
    # 1.1.5 If an experiment is specified but no run name, create a default one
    experiment_run = args.experiment_run
    if args.experiment and not experiment_run:
        experiment_run = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 1.2 Parse labels from key=value format
    labels = {}
    if args.labels:
        for label in args.labels:
            try:
                key, value = label.split('=', 1)
                labels[key] = value
            except ValueError:
                print(f"❌ Invalid label format: {label}. Must be key=value.")
                exit(1)

    # 2. Build the Docker image
    build_command = ["docker", "build", "--tag", container_uri, "--file", "gcp.dockerfile", "."]
    run_command(build_command, "Docker build failed.", verbose=args.verbose_subprocess)

    # 3. Push the image to Artifact Registry
    push_command = ["docker", "push", container_uri]
    run_command(push_command, "Docker push failed. Ensure you are authenticated with `gcloud auth configure-docker`.", verbose=args.verbose_subprocess)

    # 4. Submit the training job to Vertex AI
    create_custom_job(
        project=args.project,
        location=args.location,
        staging_bucket=args.staging_bucket,
        display_name=args.display_name,
        container_uri=container_uri,
        service_account=args.service_account,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        dataset_path=args.dataset_path,
        experiment=args.experiment,
        experiment_run=experiment_run,
        labels=labels,
    )
