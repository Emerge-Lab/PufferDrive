# PufferDrive

MARL autonomous driving RL environment. C simulation engine (GigaFlow/Waymo replay) + Python/PyTorch PPO training loop with V-trace and priority sampling.

## Install

```bash
# Python env
uv venv && source .venv/bin/activate
uv pip install -e .

# Build C extensions (required after any .h/.c change)
python setup.py build_ext --inplace --force
```

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

Daily workflow commands now live in:

- [competition README](pufferlib/ocean/competition/README.md)

Benchmark-specific commands and catalog configuration live in:

- `pufferlib/ocean/competition/README.md`
- `pufferlib/ocean/competition/benchmark_catalog.yaml`

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
| `control_mode` | `"control_vehicles"` | `"control_vehicles"`, `"control_sdc_only"` |
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
| `shared_network` | `True` | |
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


## Profiling

```bash
# 1. Build
./scripts/build_ocean.sh drive profile-debug

# 2. Record with perf (discrete mode)
sudo perf record --call-graph fp -F 99 -m 64M --delay=200 ./drive

# 3. Export to text
sudo perf script -i perf.data > profile.linux-perf.txt

# 4. Visualize
#    Open https://www.speedscope.app and drag-drop profile.linux-perf.txt
