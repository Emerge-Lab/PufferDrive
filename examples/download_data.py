import os
import zipfile
import urllib.request

BASE_URL = "https://huggingface.co/datasets/julianh65/pufferdrive_womd_subsets/resolve/main"

FILES = [
    "pufferdrive_womd_training_10k.zip",
    "pufferdrive_womd_training_20k.zip",
    "pufferdrive_womd_training_50k.zip",
]
OUT_DIR = "data/processed"

os.makedirs(OUT_DIR, exist_ok=True)


def make_progress_hook(filename):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            print(f"\r  {filename}: {pct:.1f}% ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)
        else:
            print(f"\r  {filename}: {downloaded / 1e6:.1f} MB downloaded", end="", flush=True)

    return hook


for filename in FILES:
    url = f"{BASE_URL}/{filename}"
    zip_path = os.path.join(OUT_DIR, filename)

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, zip_path, reporthook=make_progress_hook(filename))
    print()  # newline after progress line

    print(f"Unzipping {filename}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for i, member in enumerate(members, 1):
            zf.extract(member, OUT_DIR)
            print(f"\r  {i}/{len(members)} files extracted", end="", flush=True)
    print()

    os.remove(zip_path)
    print(f"Done: {filename}\n")

print("All files downloaded and extracted.")
