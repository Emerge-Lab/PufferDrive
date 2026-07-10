"""Fetch shared PufferDrive datasets from the lab S3 buckets.

Datasets are declared in data_utils/datasets.yaml and sync into
$PUFFERDRIVE_DATA_ROOT/<name>/ (default: <repo>/data/<name>/, gitignored).

    python data_utils/fetch_data.py --list
    python data_utils/fetch_data.py nuplan_dev_maps
    python data_utils/fetch_data.py nuplan_dev_maps --data-root /scratch/$USER/data
    python data_utils/fetch_data.py nuplan_dev_maps --dry-run

Requires the AWS CLI with lab IAM credentials (see docs/data_storage.md).
Syncs are incremental: re-running downloads only new or changed files.
"""

import argparse
import os
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(REPO_ROOT, "data_utils", "datasets.yaml")
DEFAULT_DATA_ROOT = os.environ.get("PUFFERDRIVE_DATA_ROOT", os.path.join(REPO_ROOT, "data"))


def load_manifest():
    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict) or not manifest:
        sys.exit(f"error: {MANIFEST_PATH} is empty or not a mapping of dataset entries")
    for name, entry in manifest.items():
        for field in ("s3_uri", "description", "license"):
            if field not in entry:
                sys.exit(f"error: dataset '{name}' in {MANIFEST_PATH} is missing the '{field}' field")
    return manifest


def list_datasets(manifest, data_root):
    for name, entry in manifest.items():
        local_dir = os.path.join(data_root, name)
        # aws s3 sync creates the destination dir even when it fails, so an
        # empty dir does not count as fetched.
        status = "present" if os.path.isdir(local_dir) and os.listdir(local_dir) else "not fetched"
        print(f"{name}  [{status}]")
        print(f"    source:  {entry['s3_uri']}")
        print(f"    local:   {local_dir}")
        print(f"    about:   {entry['description']}")
        print(f"    license: {entry['license']}")


def fetch_dataset(manifest, name, data_root, dry_run):
    if name not in manifest:
        sys.exit(f"error: unknown dataset '{name}'. Available: {', '.join(sorted(manifest))}")
    if shutil.which("aws") is None:
        sys.exit(
            "error: the AWS CLI is not installed. Install it (https://aws.amazon.com/cli/)\n"
            "and configure lab IAM credentials — see docs/data_storage.md."
        )
    entry = manifest[name]
    destination_dir = os.path.join(data_root, name)
    sync_command = ["aws", "s3", "sync", entry["s3_uri"], destination_dir]
    if dry_run:
        sync_command.append("--dryrun")
    print(f"license: {entry['license']}")
    print(f"syncing {entry['s3_uri']} -> {destination_dir}")
    completed = subprocess.run(sync_command)
    if completed.returncode != 0:
        sys.exit(
            f"error: aws s3 sync exited with {completed.returncode}. If this is a credential\n"
            "error, configure lab IAM access first — see docs/data_storage.md."
        )
    if not dry_run:
        file_count = sum(len(files) for _, _, files in os.walk(destination_dir))
        print(f"done: {file_count} files in {destination_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", help="dataset name from data_utils/datasets.yaml")
    parser.add_argument("--list", action="store_true", help="list datasets and their local status")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="destination root (default: $PUFFERDRIVE_DATA_ROOT or <repo>/data)",
    )
    parser.add_argument("--dry-run", action="store_true", help="show what would be downloaded without writing")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.list:
        list_datasets(manifest, args.data_root)
        return
    if args.dataset is None:
        parser.error("provide a dataset name, or --list to see what is available")
    fetch_dataset(manifest, args.dataset, args.data_root, args.dry_run)


if __name__ == "__main__":
    main()
