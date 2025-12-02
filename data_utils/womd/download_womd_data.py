import os
from huggingface_hub import snapshot_download, login
import threading
import time


def count_files_in_dir(directory):
    """Count files in a directory."""
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count


def monitor_and_alert(local_dir, split_name, max_files):
    """Monitor file count and alert when limit is reached."""
    if max_files is None:
        return

    while True:
        count = count_files_in_dir(os.path.join(local_dir, split_name))
        print(f"\r{split_name}: {count}/{max_files} files", end="", flush=True)

        if count >= max_files:
            print(f"\n⚠ {split_name} LIMIT REACHED! Press Ctrl+C to stop download.")
            break

        time.sleep(2)


def download_by_groups(
    repo_id,
    local_dir,
    max_training_files=20000,
    max_validation_files=None,
    max_testing_files=None,
    files_per_group=10000,
    max_workers=4,  # Reduced to avoid rate limiting
    use_auth=True,
):
    """
    Download dataset efficiently with rate limit handling.

    Args:
        files_per_group: Number of files per training group (default 10000)
        max_workers: Parallel download workers (keep low to avoid rate limits)
        use_auth: Whether to use HuggingFace authentication (recommended)
    """

    # Check for authentication
    if use_auth:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
            if token:
                print("✓ Using existing HuggingFace authentication")
            else:
                print("\n⚠ No HuggingFace token found!")
                print("Please run: huggingface-cli login")
                print("This will increase your rate limits significantly.\n")
                response = input("Continue without authentication? (y/n): ")
                if response.lower() != "y":
                    return
        except:
            print("⚠ Could not check authentication status")

    print("=" * 60)
    print("Download Configuration")
    print("=" * 60)
    print(f"Training files: {max_training_files or 'All'}")
    print(f"Validation files: {max_validation_files or 'All'}")
    print(f"Testing files: {max_testing_files or 'All'}")
    print(f"Files per training group: {files_per_group}")
    print(f"Parallel workers: {max_workers} (low to avoid rate limits)")
    print("=" * 60)

    total_downloaded = {"training": 0, "validation": 0, "testing": 0}

    # ==================== TRAINING (GROUPED) ====================
    if max_training_files and max_training_files > 0:
        print("\n" + "=" * 60)
        print("DOWNLOADING TRAINING GROUPS")
        print("=" * 60)

        num_groups_needed = (max_training_files + files_per_group - 1) // files_per_group
        print(f"Need approximately {num_groups_needed} groups for {max_training_files} files\n")

        for group_idx in range(num_groups_needed + 1):
            current_count = count_files_in_dir(os.path.join(local_dir, "training"))

            if current_count >= max_training_files:
                print(f"\n✓ Training limit reached: {current_count}/{max_training_files}")
                break

            group_name = f"training/group_{group_idx}"
            print(f"\n[Group {group_idx}] Downloading {group_name}...")
            print(f"Current total: {current_count}/{max_training_files} files")

            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=local_dir,
                    max_workers=max_workers,
                    allow_patterns=[f"{group_name}/*"],
                )

                new_count = count_files_in_dir(os.path.join(local_dir, "training"))
                downloaded_this_group = new_count - current_count
                print(f"✓ Downloaded {downloaded_this_group} files from group_{group_idx}")
                print(f"Total training files: {new_count}/{max_training_files}")

                if new_count >= max_training_files:
                    print(f"\n✓ Training limit reached!")
                    break

            except KeyboardInterrupt:
                print("\n⚠ Download interrupted by user")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"⚠ Error downloading group_{group_idx}: {e}")

                if "429" in error_msg:
                    print("\n⚠ Rate limit hit! Suggestions:")
                    print("  1. Login with: huggingface-cli login")
                    print("  2. Wait a few minutes before retrying")
                    print("  3. Reduce max_workers further (currently {})".format(max_workers))
                    break
                elif "404" in error_msg or "not found" in error_msg.lower():
                    print(f"No more training groups found. Stopping at group_{group_idx - 1}")
                    break
                continue

        total_downloaded["training"] = count_files_in_dir(os.path.join(local_dir, "training"))

    # ==================== VALIDATION (NOT GROUPED) ====================
    if max_validation_files is None or max_validation_files > 0:
        print("\n" + "=" * 60)
        print("DOWNLOADING VALIDATION")
        print("=" * 60)

        current_val_count = count_files_in_dir(os.path.join(local_dir, "validation"))

        if max_validation_files and current_val_count >= max_validation_files:
            print(f"✓ Already have {current_val_count} validation files (limit: {max_validation_files})")
        else:
            if max_validation_files:
                print(f"⚠ Validation is NOT grouped. Downloading all validation files.")
                print(f"⚠ Press Ctrl+C when you reach {max_validation_files} files.")
                print(f"Current: {current_val_count} files\n")

                monitor_thread = threading.Thread(
                    target=monitor_and_alert, args=(local_dir, "validation", max_validation_files), daemon=True
                )
                monitor_thread.start()

            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=local_dir,
                    max_workers=max_workers,
                    allow_patterns=["validation/*"],
                )
                total_downloaded["validation"] = count_files_in_dir(os.path.join(local_dir, "validation"))
                print(f"\n✓ Validation complete: {total_downloaded['validation']} files")
            except KeyboardInterrupt:
                total_downloaded["validation"] = count_files_in_dir(os.path.join(local_dir, "validation"))
                print(f"\n⚠ Validation download stopped by user: {total_downloaded['validation']} files")
            except Exception as e:
                print(f"⚠ Error downloading validation: {e}")
                total_downloaded["validation"] = count_files_in_dir(os.path.join(local_dir, "validation"))

    # ==================== TESTING (NOT GROUPED) ====================
    if max_testing_files is None or max_testing_files > 0:
        print("\n" + "=" * 60)
        print("DOWNLOADING TESTING")
        print("=" * 60)

        current_test_count = count_files_in_dir(os.path.join(local_dir, "testing"))

        if max_testing_files and current_test_count >= max_testing_files:
            print(f"✓ Already have {current_test_count} testing files (limit: {max_testing_files})")
        else:
            if max_testing_files:
                print(f"⚠ Testing is NOT grouped. Downloading all testing files.")
                print(f"⚠ Press Ctrl+C when you reach {max_testing_files} files.")
                print(f"Current: {current_test_count} files\n")

                monitor_thread = threading.Thread(
                    target=monitor_and_alert, args=(local_dir, "testing", max_testing_files), daemon=True
                )
                monitor_thread.start()

            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=local_dir,
                    max_workers=max_workers,
                    allow_patterns=["testing/*"],
                )
                total_downloaded["testing"] = count_files_in_dir(os.path.join(local_dir, "testing"))
                print(f"\n✓ Testing complete: {total_downloaded['testing']} files")
            except KeyboardInterrupt:
                total_downloaded["testing"] = count_files_in_dir(os.path.join(local_dir, "testing"))
                print(f"\n⚠ Testing download stopped by user: {total_downloaded['testing']} files")
            except Exception as e:
                print(f"⚠ Error downloading testing: {e}")
                total_downloaded["testing"] = count_files_in_dir(os.path.join(local_dir, "testing"))

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Training: {total_downloaded['training']}" + (f" / {max_training_files}" if max_training_files else ""))
    print(
        f"Validation: {total_downloaded['validation']}" + (f" / {max_validation_files}" if max_validation_files else "")
    )
    print(f"Testing: {total_downloaded['testing']}" + (f" / {max_testing_files}" if max_testing_files else ""))
    print(f"Total: {sum(total_downloaded.values())}")
    print("=" * 60)


if __name__ == "__main__":
    download_by_groups(
        repo_id="EMERGE-lab/GPUDrive",
        local_dir="data/processed",
        max_training_files=10000,
        max_validation_files=None,
        max_testing_files=None,
        files_per_group=10000,
        max_workers=1,  # Reduced from 16 to avoid rate limits
        use_auth=True,
    )
