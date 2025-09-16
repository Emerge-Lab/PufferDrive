import sys
import gcsfs

def sync_to_gcs(local_path: str, gcs_path: str):
    """Recursively copies a local directory to a GCS path using gcsfs."""
    print(f"Syncing local directory '{local_path}' to GCS path '{gcs_path}'...")
    try:
        fs = gcsfs.GCSFileSystem()
        # The put method recursively copies the directory
        fs.put(local_path, gcs_path, recursive=True)
        print("✅ Sync successful.")
    except Exception as e:
        print(f"❌ Failed to sync to GCS: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gcs_sync.py <local_directory> <gcs_destination_path>")
        sys.exit(1)

    local_dir_path = sys.argv[1]
    gcs_dest_path = sys.argv[2]
    sync_to_gcs(local_dir_path, gcs_dest_path)