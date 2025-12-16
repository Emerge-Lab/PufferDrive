import os
import shutil
import math
import glob


def split_and_rename_files(source_dir, num_folders=100):
    """
    Split files from source_dir into num_folders subfolders and rename them sequentially.
    """
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist.")
        return

    # Get all .bin files
    files = sorted(
        [f for f in os.listdir(source_dir) if f.endswith(".bin") and os.path.isfile(os.path.join(source_dir, f))]
    )
    total_files = len(files)
    print(f"Found {total_files} files in {source_dir}")

    if total_files == 0:
        return

    files_per_folder = math.ceil(total_files / num_folders)
    print(f"Splitting into {num_folders} folders with approx {files_per_folder} files each.")

    for i in range(num_folders):
        folder_name = f"part_{i:03d}"
        folder_path = os.path.join(source_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        start_idx = i * files_per_folder
        end_idx = min((i + 1) * files_per_folder, total_files)

        batch_files = files[start_idx:end_idx]

        # Move files to folder
        for src_file in batch_files:
            src_path = os.path.join(source_dir, src_file)
            dst_path = os.path.join(folder_path, src_file)
            shutil.move(src_path, dst_path)

        # Rename files sequentially
        for j, _ in enumerate(batch_files):
            old_name = os.path.join(folder_path, batch_files[j])
            new_name = os.path.join(folder_path, f"map_{j:03d}.bin")
            os.rename(old_name, new_name)

        print(f"Processed {folder_name}: moved and renamed {len(batch_files)} files")


if __name__ == "__main__":
    source_directory = "resources/drive/binaries/validation_10k"
    split_and_rename_files(source_directory)
    print("Done!")
