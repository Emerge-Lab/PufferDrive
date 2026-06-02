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

**For a full guide on how to use this see [`docs/cluster_training.md`](docs/cluster_training.md).**

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

All evaluation runs through the unified `Evaluator`/`EvalManager` pipeline.
`[eval.<name>]` sections in `drive.ini` define each evaluator; the same ones
run inline during training and standalone here.

```bash
# Run a named evaluator on a checkpoint (config from [eval.<name>])
puffer eval puffer_drive --evaluator validation_gigaflow \
  --load-model-path experiments/puffer_drive_xxxx/models/model_puffer_drive_000500.pt

# Ad-hoc: pick by simulation + override scale from the CLI
puffer eval puffer_drive --eval_simulation replay \
  --load-model-path experiments/puffer_drive_xxxx/models/model_puffer_drive_000500.pt \
  --num_scenarios 250 --render 1

# Render the agent's observations (interactive HTML)
puffer eval puffer_drive --eval_simulation gigaflow \
  --load-model-path experiments/puffer_drive_xxxx/models/model_puffer_drive_000500.pt \
  --num_scenarios 10 --render 1 --render-backend obs_html
```

**For the full guide see [`docs/evaluation.md`](docs/evaluation.md).**

## Failure mining

Roll a trained policy out against a scenario suite, capture per-episode compact replays for episodes whose `episode_return` falls below a threshold, render each one as an interactive HTML page, and produce a sortable cross-episode index. Useful for triaging what a policy fails at after a long training run.

```bash
puffer mine_failures puffer_drive \
    --load-model-path experiments/puffer_drive_xxxx/models/model_puffer_drive_000123.pt \
    --mine.output_dir ./failure_mining/puffer_drive_xxxx \
    --mine.num_episodes 200 \
    --mine.score_threshold -10.0
```

Config keys (under `[mine]` in `drive.ini` or `--mine.<key>` on the CLI):

| Key | Default | Notes |
|---|---|---|
| `output_dir` | `./failure_mining/<env_name>` | Where replays, CSV, and HTML output go |
| `num_episodes` | `100` | Total episodes to roll out |
| `score_threshold` | `-inf` | `episode_return < threshold` → flagged as failure; replay written to disk. With `-inf`, no failures are flagged and no replays are persisted. |
| `render` | `True` | Render each captured replay to HTML + write `index.html` via `mining_viz` |

`env.*` overrides apply (e.g. `--env.simulation_mode gigaflow` to mine on procedural scenarios). Single vec env, sequential rollout — no per-worker map pinning yet.

**Output structure** (under `output_dir`):

```
episodes.csv                 # one row per episode, all summary metrics
replays/episode_NNNNNN.replay.zlib   # only for failures (above-threshold episodes are not persisted)
renders/episode_NNNNNN.html  # one viewer page per failure
renders/index.html           # sortable index of all episodes
```

Open `renders/index.html` in a browser to triage. The index page filters by "failures only" / "replays only" and sorts by any metric column. Each row links to the per-episode viewer with the scene's full 2D animation.

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
| `reward_collision` | `1.0` | Collision penalty |
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
