# PufferDrive Minimal PPO on Windows and WSL

This branch adds a small reinforcement learning workflow on top of the
original [PufferDrive](https://github.com/Emerge-Lab/PufferDrive) project.

The goal is simple:

1. Export Waymo Motion Dataset scenarios as JSON.
2. Convert the JSON files to PufferDrive map binaries.
3. Train a small continuous-action PPO policy.
4. Render the trained policy as an MP4 video.

This is a working training example, not a finished autonomous driving model.
The default short run is mainly useful for checking that the complete pipeline
works.

## Why WSL is used

The native PufferDrive renderer uses Linux libraries and Raylib. On Windows,
the easiest setup is:

- Keep the Git repository on the Windows drive.
- Run compilation, training, and native 3D rendering inside WSL.
- Copy checkpoints and videos back to the Windows repository.

The setup script creates a Linux copy at:

```text
~/PufferDrive-native
```

Your original Windows files remain under:

```text
/mnt/c/Users/<username>/Desktop/PufferDrive
```

## 1. Clone this branch

From PowerShell:

```powershell
git clone --branch codex/minimal-ppo-wsl https://github.com/HC-Seaple/PufferDrive.git
cd PufferDrive
```

If Ubuntu is not installed in WSL, open PowerShell as Administrator:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_wsl_admin.ps1
```

Restart Windows if requested, then open Ubuntu or run `wsl`.

## 2. Build the native environment

From WSL:

```bash
cd /mnt/c/Users/<username>/Desktop/PufferDrive
bash scripts/wsl_native_3d_setup.sh
```

This installs the Linux dependencies, creates a Python environment, copies the
repository to the Linux filesystem, and builds the native extension.

The warning messages from the C compiler about ignored return values are not
fatal. The setup is successful when it prints:

```text
Done. Native build is ready
```

## 3. Prepare Waymo scenarios

Put one or more exported scenario JSON files in the Windows repository. Then
run from WSL:

```bash
cd /mnt/c/Users/<username>/Desktop/PufferDrive
bash scripts/prepare_waymo_maps_wsl.sh \
  ./scenario_a.json \
  ./scenario_b.json
```

The binary maps are written to the Linux-native repository:

```text
~/PufferDrive-native/resources/drive/binaries/training
```

The files must be contiguous:

```text
map_000.bin
map_001.bin
map_002.bin
```

## 4. Run a small training test

From the Windows-mounted repository in WSL:

```bash
bash scripts/run_minimal_ppo_wsl.sh \
  --map-dir resources/drive/binaries/training \
  --num-maps 2 \
  --num-envs 1 \
  --total-timesteps 10000 \
  --rollout-steps 128 \
  --minibatch-size 128
```

Change `--num-maps` to match the number of prepared map files.

Checkpoints are saved under:

```text
~/PufferDrive-native/checkpoints/minimal_ppo
```

The training is running correctly when the step count increases, the losses
remain finite, and `.pt` checkpoint files are created.

## 5. Render the trained policy

```bash
bash scripts/visualize_minimal_ppo_wsl.sh \
  --map-dir resources/drive/binaries/training \
  --num-maps 2 \
  --episode-length 91 \
  --draw-traces
```

The script renders with the native Raylib 3D renderer and copies the MP4 and
JSON metrics to the Windows folder:

```text
training_visualizations/
```

## Visualize a Waymo JSON without training

The following commands use the Windows virtual environment.

Create a top-down replay:

```powershell
.\.venv\Scripts\python.exe scripts\visualize_waymo_json.py scenario.json
```

Create a simple 3D chase-camera replay for a selected Waymo track:

```powershell
.\.venv\Scripts\python.exe scripts\render_waymo_follow_3d.py scenario.json `
  --track-index 90 `
  --start-frame 0 `
  --end-frame 60
```

The lightweight chase-camera renderer uses boxes and map lines. It is useful
for quickly checking a recorded trajectory without compiling Raylib. The
native checkpoint renderer uses the PufferDrive car models and full native
rendering.

Outputs are written to:

```text
visualizations/
```

## Main files

| File | Purpose |
| --- | --- |
| `scripts/minimal_ppo_train.py` | Small PPO actor-critic training loop |
| `scripts/parallel_data_collect.py` | Original rollout pattern used by the trainer |
| `scripts/prepare_waymo_maps.py` | Converts Waymo JSON to PufferDrive maps |
| `scripts/run_minimal_ppo_wsl.sh` | Starts training in the Linux-native copy |
| `scripts/visualize_minimal_ppo.py` | Runs a checkpoint and records native 3D video |
| `scripts/visualize_waymo_json.py` | Creates a top-down JSON replay |
| `scripts/render_waymo_follow_3d.py` | Creates a lightweight 3D chase replay |
| `docs/src/minimal-ppo.md` | More detail about PPO and command options |

## Current limitation

A 10,000-step run only verifies the architecture. It is usually not enough to
learn good driving. Early policies may reverse, steer poorly, or fail to reach
the goal.

For a useful model, use more scenarios and training steps, then evaluate on
scenarios that were not used for training. Useful measurements include:

- goal completion rate
- collision rate
- off-road rate
- reverse-motion frequency
- average episode return

## Upstream project

PufferDrive is developed by Emerge Lab. The original documentation is
available at <https://emerge-lab.github.io/PufferDrive>.
