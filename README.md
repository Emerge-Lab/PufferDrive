# PufferDrive

MARL autonomous driving RL environment. C simulation engine (GigaFlow/Waymo replay) + Python/PyTorch PPO training loop with V-trace and priority sampling.

## Install (local)

```bash
# Python env
uv venv && source .venv/bin/activate
uv pip install -e .

# Build C extensions (required after any .h/.c change)
python setup.py build_ext --inplace --force
```

## Install (HPC cluster)

For the NYU cluster, PufferDrive recommends a **mixed Singularity + venv** layout:

- **Singularity image** (read-only, system-wide): supplies CUDA + cuDNN.
- **ext3 overlay** (writable via `--fakeroot`, host the miniforge3 base interpreter at `/ext3/miniforge3` only).
- **Venv on `/scratch`** (regular ext4, fast): everything else — `torch`, `pufferlib`, the compiled `_C.so`.

The venv lives outside the overlay because fuse2fs is single-threaded (~10 MB/s); putting torch/pufferlib in `/scratch` makes installs and rebuilds ~50× faster.

`scripts/setup_container.sh` is the entrypoint. It auto-detects whether it's running inside the container and re-`singularity exec`s itself accordingly, so you can call it from the login node directly.

**Defaults** (all env-var overridable):

| Variable | Default |
|---|---|
| `OVERLAY_PATH` | `/scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3` |
| `IMAGE_PATH` | `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif` |
| `OVERLAY_TEMPLATE` | `/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz` |
| `VENV_PATH` | `/scratch/$USER/venvs/pufferdrive` |
| `CONTAINER_PYTHON` | `/ext3/miniforge3/bin/python3` |

The defaults match NYU Greene's filesystem layout. Override the env vars before invoking `setup_container.sh` if your cluster differs.

**One-time setup:**

```bash
# 1. Create the overlay (login node, ~2 min, no GPU needed)
./scripts/setup_container.sh create-overlay

# 2. Install dependencies into the venv (writable mount; submit as a GPU job)
sbatch --account=$ACCOUNT --gres=gpu:1 --cpus-per-task=8 --mem=32gb --time=60 \
    --wrap "./scripts/setup_container.sh install"
```

The `install` step bootstraps `uv` if it's missing, creates the venv against the overlay's miniforge3, installs `torch` (cu121 wheels), then `pip install -e .` (which also builds the C extension). It also patches `bin/activate` so torch's bundled NCCL wins over the sif's older `libnccl.so.2` — without that, `torchrun`-spawned children crash on `undefined symbol: ncclCommShrink`. `TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"` is set during build so the resulting `_C.so` runs on A100 / L40S / H100 / H200 without "no kernel image" errors.

**Per-code-change rebuild:**

```bash
sbatch --account=$ACCOUNT --gres=gpu:1 --cpus-per-task=4 --mem=8gb --time=15 \
    --wrap "./scripts/setup_container.sh rebuild"
```

`rebuild` mounts the overlay read-only — safe to run while other jobs hold the same overlay.

**Submitting training jobs:**

```bash
python scripts/submit_cluster.py \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --save_dir experiments \
    --container
```

`scripts/cluster_configs/nyu_greene.yaml` defines `account`, `gpus`, `cpus`, `mem`, `time` — edit `account` to your allocation before first submit. `--container` makes `submit_cluster.py` wrap the job command in `singularity exec --nv --overlay $OVERLAY_PATH:ro $IMAGE_PATH ...`.

## Data

Place binaries under `pufferlib/resources/drive/binaries/`.

## Train

```bash
# Single node
puffer train puffer_drive

# Override config on the fly
puffer train puffer_drive --train.learning_rate 0.001 --env.num_agents 512

# Multi-GPU
torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_drive
```

## Eval

```bash
# Multi-scenario eval (replay mode)
puffer eval_multi_scenarios puffer_drive \
  --load-model-path experiments/puffer_drive_177193887946/models/model_puffer_drive_000001.pt \
  --num_scenarios 250 --eval_simulation replay

# Multi-scenario eval (gigaflow mode)
puffer eval_multi_scenarios puffer_drive \
  --load-model-path experiments/puffer_drive_177193887946/models/model_puffer_drive_000001.pt \
  --num_scenarios 10 --eval_simulation gigaflow

# Multi-scenario eval with rendering
puffer eval_multi_scenarios_render puffer_drive \
  --load-model-path experiments/puffer_drive_177193887946/models/model_puffer_drive_000001.pt \
  --num_scenarios 10 --eval_simulation gigaflow --render 1 --render_obs 0

# Save eval as GIF
puffer eval_multi_scenarios_render puffer_drive \
  --load-model-path experiments/puffer_drive_177193887946/models/model_puffer_drive_000001.pt \
  --num_scenarios 5 --eval_simulation gigaflow --save-frames 1 --gif-path eval.gif --fps 15
```

## Failure mining

Roll a trained policy out against a scenario suite, rank episodes by `avg_distance_per_infraction` (lower = worse), keep the bottom K as "failures", render each one as an interactive HTML page, and produce a sortable cross-episode index. Useful for triaging what a policy fails at after a long training run.

```bash
puffer mine_failures puffer_drive \
    --load-model-path experiments/puffer_drive_xxxx/models/model_puffer_drive_000123.pt \
    --mine.output_dir ./failure_mining/puffer_drive_xxxx \
    --mine.num_episodes 200 \
    --mine.num_failures 20
```

Config keys (under `[mine]` in `drive.ini` or `--mine.<key>` on the CLI):

| Key | Default | Notes |
|---|---|---|
| `output_dir` | `./failure_mining/<env_name>` | Where replays, CSV, and HTML output go |
| `num_episodes` | `100` | Total episodes to roll out |
| `num_failures` | `20` | Bottom-K episodes by `avg_distance_per_infraction` (ascending) are flagged as failures and have replays persisted to disk. |
| `render` | `True` | Render each captured replay to HTML + write `index.html` via `mining_viz` |
| `observe_agent` | `-1` | Active-slot index of an agent whose FOV-pass set is captured per step. When `>= 0`, each rendered HTML draws the policy's road FOV rectangle + partner FOV circle around that agent and highlights the road segments, partner agents, and traffic controls that passed the gate. `0` picks the first active agent (the SDC under `control_sdc_only`). `-1` disables the overlay. |

`env.*` overrides apply (e.g. `--env.simulation_mode gigaflow` to mine on procedural scenarios). Single vec env, sequential rollout — no per-worker map pinning yet.

### Observation overlay

When `--mine.observe-agent N` is set (where `N` is an active-agent slot index), each replay HTML gains a togglable layer showing what the policy could observe at every step:

- A blue, agent-heading-aligned rectangle for the road FOV (`road_obs_front_dist` ahead, `road_obs_behind_dist` behind, `road_obs_side_dist` to each side).
- A red circle for the partner FOV (`agent_obs_max_dist`).
- Thicker blue strokes on road segments whose midpoint fell in the road FOV.
- Red outlines on partner agents within the partner FOV.
- Yellow rings on traffic controls whose stop-line midpoint fell in range.
- Dashed green polylines for the observed agent's goal route + a green diamond at the goal point.
- A green outline on the observed agent itself.

The "Hide Observations" button in the viewer toggles the layer. The visible-segment set is captured inside the C engine's `compute_observations` so the overlay can't drift from the policy's actual FOV gates (3D partner distance, `is_blind_partner` zero-out, etc.).

**Output structure** (under `output_dir`):

```
episodes.csv                 # one row per episode, all summary metrics
replays/episode_NNNNNN.replay.zlib   # only for the bottom-K failures
renders/episode_NNNNNN.html  # one viewer page per failure
renders/index.html           # sortable index of all episodes
```

Open `renders/index.html` in a browser to triage. The index page filters by "failures only" / "replays only" and sorts by any metric column. Each row links to the per-episode viewer with the scene's full 2D animation.

### Example: nuplan with the obs overlay

```bash
puffer mine_failures puffer_drive \
    --load-model-path experiments/puffer_drive_xxxx/models/model_puffer_drive_000123.pt \
    --mine.output-dir ./failure_mining/nuplan_xxxx \
    --mine.num-episodes 50 \
    --mine.num-failures 15 \
    --mine.observe-agent 0 \
    --env.map-dir /path/to/nuplan_bins \
    --env.num-maps 50 \
    --env.simulation-mode replay \
    --env.control-mode control_sdc_only \
    --env.init-mode create_all_valid \
    --env.scenario-length 201
```

Under `control_sdc_only` the SDC is the only controlled agent per scenario, so slot `0` is the ego car everywhere. Ranking by `avg_distance_per_infraction` then surfaces episodes where the SDC crashed early relative to how far it drove.

## Key Configuration (`pufferlib/config/ocean/drive.ini`)

### `[env]` — Simulation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `simulation_mode` | `"gigaflow"` | `"gigaflow"` (procedural) or `"replay"` (Data-driven) |
| `num_agents` | `1024` | Total agents across all workers |
| `min/max_agents_per_env` | `10 / 80` | Per-env agent count range |
| `action_type` | `"discrete"` | `"discrete"`, `"continuous"`, `"trajectory"`, `"trajectory_frenet"`, `"trajectory_jerk"` |
| `dynamics_model` | `"jerk"` | `"classic"` or `"jerk"` |
| `scenario_length` | `128` | Steps per episode (128 GF, 91 replay) |
| `collision_behavior` | `1` | `0` ignore, `1` stop, `2` remove |
| `offroad_behavior` | `1` | Same options |
| `traffic_light_behavior` | `1` | Same options |
| `control_mode` | `"control_vehicles"` | `"control_vehicles"`, `"control_agents"`, `"control_sdc_only"` |
| `reward_conditioning` | `True` | Condition policy on reward weights |
| `reward_randomization` | `True` | Randomize reward weights each episode |

### `[env]` — Reward Shaping

| Parameter | Default | Effect |
|-----------|---------|--------|
| `reward_goal` | `1.0` | Goal reaching (set 0 without reward conditioning) |
| `reward_vehicle_collision` | `1.0` | Collision penalty |
| `reward_comfort` | `0.05` | Smooth driving |
| `reward_lane_align` | `0.025` | Lane heading alignment |
| `reward_vel_align` | `1.0` | Speed matching road limit |
| `reward_lane_center` | `0.0038` | Lane centering |
| `reward_stop_line` | `1.0` | Stop line penalty |
| `reward_overspeed` | `0.05` | Speeding penalty |

### `[train]` — PPO

| Parameter | Default | Notes |
|-----------|---------|-------|
| `learning_rate` | `0.001` | Adam LR |
| `bptt_horizon` | `64` | BPTT window (64 GF, 91 replay) |
| `minibatch_size` | `4096` | |
| `gamma` | `0.98` | Discount |
| `gae_lambda` | `0.95` | GAE lambda |
| `vf_coef` | `2` | Value loss weight |
| `ent_coef` | `0.001` | Entropy bonus |
| `vtrace_rho_clip` / `vtrace_c_clip` | `1` | V-trace IS ratio clipping |
| `adv_sampling_prio_alpha` / `adv_sampling_prio_beta0` | `0.85` | Priority sampling exponents |

### `[policy]` — Network

| Parameter | Default | Notes |
|-----------|---------|-------|
| `backbone_hidden_size` | `512` | |
| `split_network` | `False` | GigaFlow network vs LSTM |
| `encoder_gigaflow` | `False` | |
| `dropout` | `0.0` | |

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_observations.ipynb` | Verify obs vector packing, normalization, interpretability |
| `02_rewards.ipynb` | Reward magnitudes, component breakdown, correlation with behavior |
| `03_metrics.ipynb` | Episode metrics, `vec_log` aggregation, episode boundary handling |
| `04_training.ipynb` | End-to-end data flow: env → policy → loss; encoding, sampling, advantages, gradients |
| `05_inference.ipynb` | Config loading, policy forward pass, rollouts (det. vs stochastic), value accuracy, LSTM state |
| `06_architecture.ipynb` | Model summary, per-encoder breakdown, forward pass shape tracing, weight distributions |


## Debug

**C build issues**
```bash
# Verbose build with AddressSanitizer
DEBUG=1 python setup.py build_ext --inplace --force

# Skip unrelated extensions
NO_OCEAN=1 python setup.py build_ext --inplace --force
NO_TRAIN=1 python setup.py build_ext --inplace --force

# Debug train
CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) python -m pufferlib.pufferl train puffer_drive --train.device cpu --vec.backend Serial
# or
gdb --args python -m pufferlib.pufferl train puffer_drive --train.device cpu --vec.backend Serial
```
