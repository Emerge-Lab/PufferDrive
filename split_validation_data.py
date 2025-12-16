import os
import shutil
import math


def split_files(source_dir, num_folders=100):
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist.")
        return

    files = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])
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

        for file_name in batch_files:
            src_path = os.path.join(source_dir, file_name)
            dst_path = os.path.join(folder_path, file_name)
            shutil.move(src_path, dst_path)

        print(f"Moved {len(batch_files)} files to {folder_name}")


if __name__ == "__main__":
    source_directory = "/scratch/wbd2016/PufferDrive/resources/drive/binaries/validation_10k"
    split_files(source_directory)
