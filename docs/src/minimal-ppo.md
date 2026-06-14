# Minimal PPO training

`scripts/minimal_ppo_train.py` is a small continuous-action PPO implementation
built directly on the rollout pattern in `scripts/parallel_data_collect.py`.
It is intended to make the training architecture explicit before moving to the
full PuffeRL trainer.

## Architecture

Each controlled agent produces one row of the vectorized batch:

```text
Drive observations
      |
      v
2-layer MLP encoder
      |
      +----> Gaussian actor ----> tanh ----> [acceleration, steering]
      |
      +----> critic ------------> V(observation)
```

The trainer repeatedly:

1. Collects a vectorized rollout from `Drive`.
2. Stores observations, raw policy actions, log probabilities, rewards,
   terminal flags, and value predictions.
3. Computes generalized advantage estimates (GAE).
4. Updates actor and critic with the PPO clipped objective.
5. Saves checkpoints and runs a short deterministic evaluation.

## What is included

- `scripts/wsl_native_3d_setup.sh`: installs the WSL/Linux dependencies and
  builds the native Raylib binding on the Linux filesystem.
- `scripts/prepare_waymo_maps_wsl.sh`: converts one or more exported WOMD JSON
  scenarios into contiguous `map_000.bin`, `map_001.bin`, ... files.
- `scripts/minimal_ppo_train.py`: self-contained continuous-action PPO.
- `scripts/run_minimal_ppo_wsl.sh`: launches training in the Linux-native copy.
- `scripts/visualize_minimal_ppo.py`: deterministic evaluation with native 3D
  rendering.
- `scripts/visualize_minimal_ppo_wsl.sh`: launches evaluation and copies the
  video and JSON metrics back to Windows.

Waymo scenarios, checkpoints, and rendered videos are deliberately not stored
in Git. Users must comply with the Waymo Open Dataset license and provide their
own exported scenario JSON files.

## Clone and set up

From PowerShell, clone the repository and enter WSL:

```powershell
git clone https://github.com/<your-account>/PufferDrive.git
cd PufferDrive
wsl
```

If WSL/Ubuntu is not installed, open an elevated PowerShell and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_wsl_admin.ps1
```

From the WSL shell:

```bash
bash scripts/wsl_native_3d_setup.sh
```

The setup script mirrors a Windows-mounted checkout to
`~/PufferDrive-native`, creates `~/.venvs/pufferdrive-wsl`, installs the
dependencies, and builds the native extension there.

## Prepare Waymo scenarios

Pass one or more exported Waymo Motion Dataset scenario JSON files:

```bash
bash scripts/prepare_waymo_maps_wsl.sh \
  ./scenario_a.json \
  ./scenario_b.json
```

The generated maps live in the Linux-native repository under
`resources/drive/binaries/training`. The order of the JSON arguments determines
the contiguous map indices.

## WSL smoke test

Run this from the Windows-mounted repository:

```bash
bash scripts/run_minimal_ppo_wsl.sh \
  --map-dir resources/drive/binaries/training \
  --num-maps 2 \
  --num-envs 1 \
  --total-timesteps 10000 \
  --rollout-steps 128 \
  --minibatch-size 128 \
  --checkpoint-interval 10
```

This verifies the training loop but is not enough data or experience to learn
a useful driving policy.

## Visualize a checkpoint

```bash
bash scripts/visualize_minimal_ppo_wsl.sh \
  --map-dir resources/drive/binaries/training \
  --num-maps 2 \
  --episode-length 91 \
  --draw-traces
```

The command writes an MP4 and a JSON metrics file to
`training_visualizations/` in the Windows checkout.

## Small experiment

After preparing a directory with multiple contiguous maps named
`map_000.bin`, `map_001.bin`, and so on:

```bash
bash scripts/run_minimal_ppo_wsl.sh \
  --map-dir resources/drive/binaries/training \
  --num-maps 100 \
  --num-envs 16 \
  --total-timesteps 1000000 \
  --rollout-steps 128 \
  --minibatch-size 512
```

The default `goal_behavior=1` samples new lane-based goals after reaching the
current goal. Use `--goal-behavior 2` to train only against each JSON scenario's
fixed `goalPosition`.

## Reading the logs

- `reward/step`: mean immediate reward in the latest rollout.
- `episode_return`: mean completed-episode return over the latest 100 episodes.
- `policy_loss`: PPO actor loss.
- `value_loss`: critic regression loss.
- `entropy`: policy exploration; it normally decreases gradually.
- `kl`: approximate policy change per update.
- `clipfrac`: fraction of samples clipped by PPO.

The smoke test succeeds when steps advance, losses remain finite, and
checkpoints are written. A driving model should instead be judged on held-out
maps using goal completion, collision rate, and off-road rate.

## Current limitation

The minimal trainer validates the architecture, but its current reward is not
yet sufficient for high-quality driving. The first smoke-test policy may learn
undesired motion such as reversing. The next iteration should add explicit
route-progress reward, reverse-motion penalty, and stronger collision/off-road
costs before scaling training.
