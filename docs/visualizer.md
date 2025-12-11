# Visualizer

PufferDrive ships a Raylib-based visualizer for replaying scenes, exporting videos, and debugging policies.

## Dependencies
Install the minimal system packages for headless render/export:

```bash
sudo apt update
sudo apt install ffmpeg xvfb
```

On environments without sudo, install them into your conda/venv:

```bash
conda install -c conda-forge xorg-x11-server-xvfb-cos6-x86_64 ffmpeg
```

## Build
Compile the visualizer binary from the repo root:

```bash
bash scripts/build_ocean.sh visualize local
```

If you need to force a rebuild, remove the cached binary first (`rm ./visualize`).

## Run headless
Launch the visualizer with a virtual display and export an `.mp4`:

```bash
xvfb-run -s "-screen 0 1280x720x24" ./visualize
```

Adjust the screen size and color depth as needed. The `xvfb-run` wrapper allows Raylib to render without an attached display, which is convenient for servers and CI jobs.
