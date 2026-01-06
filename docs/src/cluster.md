# Cluster Training

This guide covers launching PufferDrive training jobs on SLURM clusters using the `submit_cluster.py` script.

## Installation

Install the cluster dependencies:

```bash
pip install pufferlib[cluster]
```

Or manually:

```bash
pip install submitit pyyaml
```

## Quick Start

```bash
python scripts/submit_cluster.py \
    --save_dir /path/to/experiments \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --dry 0
```

Set `--dry 1` (default) to preview commands without submitting.

## Configuration Files

The launcher uses two YAML configuration files:

### Compute Config

Defines SLURM resource allocation. Example (`scripts/cluster_configs/nyu_greene.yaml`):

```yaml
account: your_slurm_account
nodes: 1
gpus: 1
cpus: 16
mem: 32gb
time: 360  # minutes
gpu_type: null  # rtx8000, a100, v100 (optional)
exclude: ""
nodelist: null
```

### Program Config

Defines training arguments passed to `puffer train puffer_drive`. Example (`scripts/cluster_configs/train_base.yaml`):

```yaml
# Vectorization
vec.num_workers: 16
vec.num_envs: 16

# Environment
env.num_agents: 1024
env.map_dir: "resources/drive/binaries/training"
env.num_maps: 10000

# Training
train.total_timesteps: 2_000_000_000
train.checkpoint_interval: 1000
train.render: False

# W&B logging
wandb: True
wandb_project: pufferdrive
wandb_group: cluster
```

## Passing Training Arguments

### Via Program Config

The recommended way to pass training arguments is through a program config YAML file:

```yaml
# my_config.yaml
vec.num_workers: 16
env.num_agents: 1024
train.total_timesteps: 1_000_000_000
wandb: True
```

Then reference it:

```bash
python scripts/submit_cluster.py \
    --program_config my_config.yaml \
    ...
```

### Via CLI with --args

Pass or override individual arguments using `--args`:

```bash
python scripts/submit_cluster.py \
    --save_dir /path/to/experiments \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --args env.num_agents=2048 train.learning_rate=3e-4 \
    --dry 0
```

Arguments use dot notation matching the training CLI:
- `env.num_agents` → `--env.num-agents`
- `train.learning_rate` → `--train.learning-rate`
- `vec.num_workers` → `--vec.num-workers`

Underscores are automatically converted to dashes for CLI compatibility.

### Combining Config and CLI

CLI args override config file values:

```bash
python scripts/submit_cluster.py \
    --program_config scripts/cluster_configs/train_base.yaml \
    --args train.total_timesteps=100_000_000 \
    --dry 0
```

This uses all settings from `train_base.yaml` but overrides `total_timesteps`.

### Boolean Flags

Boolean flags like `wandb` and `neptune` are handled specially:

```bash
# Enable wandb
--args wandb=True

# Disable wandb (omit from command)
--args wandb=False
```

## Parameter Sweeps

Sweep over parameter values using colon-separated syntax:

```bash
# Sweep learning rates
python scripts/submit_cluster.py \
    --save_dir /path/to/experiments \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --args train.learning_rate=1e-4:3e-4:1e-3 \
    --dry 0
```

This launches 3 separate jobs, one for each learning rate.

Combine multiple sweeps:

```bash
python scripts/submit_cluster.py \
    --save_dir /path/to/experiments \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --args train.learning_rate=1e-4:3e-4 env.num_agents=512:1024 \
    --dry 0
```

This launches 4 jobs (2 learning rates × 2 agent counts).

## CLI Options

### Job Management

| Option | Description |
|--------|-------------|
| `--save_dir` | Base directory for experiment outputs (required) |
| `--prefix` | Prefix for job names |
| `--dry` | Dry run (1) or submit (0), default: 1 |
| `--max_pjob` | Max parallel jobs (waits if exceeded) |

### Compute Overrides

Override compute config values from CLI:

| Option | Description |
|--------|-------------|
| `--account` | SLURM account |
| `--partition` | SLURM partition |
| `--cpus` | CPUs per task |
| `--gpus` | GPUs per node |
| `--nodes` | Number of nodes |
| `--gpu_type` | GPU type (a100, v100, etc.) |
| `--nodelist` | Specific nodes to use |
| `--mem` | Memory per node (e.g., 32gb) |
| `--exclude` | Nodes to exclude |
| `--time` | Time limit in minutes |

### Program Options

| Option | Description |
|--------|-------------|
| `--program_config` | YAML file with training args |
| `--args` | Override/sweep args (e.g., `learning_rate=1e-4:3e-4`) |
| `--main` | Main command (default: `-m pufferlib.pufferl train puffer_drive`) |

## Multi-Node Training

For multi-node distributed training, set `nodes > 1`:

```yaml
# compute config
nodes: 2
gpus: 4
```

The script automatically configures `torchrun` with the correct rendezvous settings.

## W&B Logging

To use Weights & Biases logging on cluster nodes:

1. Log in on the login node:
   ```bash
   wandb login
   ```

2. Enable in your program config:
   ```yaml
   wandb: True
   wandb_project: your-project
   wandb_group: your-group
   ```

Credentials are stored in `~/.netrc` and shared with compute nodes.

## Output Structure

Each job creates a directory under `save_dir`:

```
/path/to/experiments/
└── train_base_8a6a467/
    ├── submitit/
    │   ├── 549896_0_log.out    # stdout
    │   ├── 549896_0_log.err    # stderr
    │   └── 549896_submission.sh
    └── checkpoints/
        └── ...
```

## Examples

### Basic training run

```bash
python scripts/submit_cluster.py \
    --save_dir /scratch/$USER/experiments \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --dry 0
```

### Quick test run

```bash
python scripts/submit_cluster.py \
    --save_dir /scratch/$USER/experiments \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --time 30 \
    --args train.total_timesteps=10_000_000 \
    --dry 0
```

### Hyperparameter sweep

```bash
python scripts/submit_cluster.py \
    --save_dir /scratch/$USER/experiments \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --args train.learning_rate=1e-4:3e-4:1e-3 train.gamma=0.99:0.995 \
    --max_pjob 4 \
    --dry 0
```

### Custom compute allocation

```bash
python scripts/submit_cluster.py \
    --save_dir /scratch/$USER/experiments \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --gpus 2 --cpus 32 --mem 64gb --time 720 \
    --dry 0
```
