"""Fetch shared PufferDrive datasets from S3.

Datasets are declared in data_utils/datasets.yaml and sync into
$PUFFERDRIVE_DATA_ROOT/<name>/ (default: <repo>/data/<name>/, gitignored).

    python data_utils/fetch_data.py                     # the default ~10 GB mini sets
    python data_utils/fetch_data.py --list
    python data_utils/fetch_data.py nuplan_val
    python data_utils/fetch_data.py nuplan_train --data-root /scratch/$USER/data
    python data_utils/fetch_data.py nuplan_mini_val --dry-run

Requires the AWS CLI. Datasets marked public in the manifest need no AWS
account; private ones need lab IAM credentials (see docs/data_storage.md).
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
        for field in ("s3_uri", "description", "license", "size"):
            if field not in entry:
                sys.exit(f"error: dataset '{name}' in {MANIFEST_PATH} is missing the '{field}' field")
    return manifest


def list_datasets(manifest, data_root):
    for name, entry in manifest.items():
        local_dir = os.path.join(data_root, name)
        # aws s3 sync creates the destination dir even when it fails, so an
        # empty dir does not count as fetched.
        status = "present" if os.path.isdir(local_dir) and os.listdir(local_dir) else "not fetched"
        default_marker = "  (default)" if entry.get("default") else ""
        print(f"{name}  [{status}]{default_marker}")
        print(f"    source:  {entry['s3_uri']}")
        print(f"    size:    {entry['size']}")
        print(f"    local:   {local_dir}")
        print(f"    about:   {entry['description']}")
        print(f"    license: {entry['license']}")


def fetch_dataset(manifest, name, data_root, dry_run):
    if name not in manifest:
        sys.exit(f"error: unknown dataset '{name}'. Available: {', '.join(sorted(manifest))}")
    if shutil.which("aws") is None:
        sys.exit(
            "error: the AWS CLI is not installed. Install it (https://aws.amazon.com/cli/)\n— see docs/data_storage.md."
        )
    entry = manifest[name]
    destination_dir = os.path.join(data_root, name)
    sync_command = ["aws", "s3", "sync", entry["s3_uri"], destination_dir]
    # The AWS CLI refuses even public buckets without credentials unless the
    # request is explicitly unsigned.
    if entry.get("public"):
        sync_command.append("--no-sign-request")
    if dry_run:
        sync_command.append("--dryrun")
    print(f"license: {entry['license']}")
    print(f"size:    {entry['size']}")
    print(f"syncing {entry['s3_uri']} -> {destination_dir}")
    completed = subprocess.run(sync_command)
    if completed.returncode != 0:
        credential_hint = (
            ""
            if entry.get("public")
            else (" If this is a credential\nerror, configure lab IAM access first — see docs/data_storage.md.")
        )
        sys.exit(f"error: aws s3 sync exited with {completed.returncode}.{credential_hint}")
    if not dry_run:
        file_count = sum(len(files) for _, _, files in os.walk(destination_dir))
        print(f"done: {file_count} files in {destination_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "datasets",
        nargs="*",
        help="dataset names from data_utils/datasets.yaml; none = the entries marked default",
    )
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
    dataset_names = args.datasets
    if not dataset_names:
        dataset_names = [name for name, entry in manifest.items() if entry.get("default")]
        if not dataset_names:
            sys.exit(f"error: no dataset given and no entry in {MANIFEST_PATH} is marked default: true")
        print(f"no dataset given — fetching the defaults: {', '.join(dataset_names)}")
    for name in dataset_names:
        fetch_dataset(manifest, name, args.data_root, args.dry_run)


if __name__ == "__main__":
    main()
