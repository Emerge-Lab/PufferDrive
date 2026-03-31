# PufferDrive MLOps

Tooling for running PufferDrive experiments on GCP Vertex AI.

## Concepts

- **Experiment**: A group of related training runs (e.g., "hyperparam_sweep", "baseline_comparison")
- **Run**: An individual training job within an experiment

Output is organized hierarchically in the bucket:
```
gs://bucket/experiments/<experiment_name>/<run_name>/
```

## Quick Start

```bash
# Launch a single run
python mlops/mlops.py launch my_experiment my_run --machine T4

# Launch multiple runs from config
python mlops/mlops.py launch-batch mlops/configs/experiments.yaml

# Build base Docker image (only needed after dependency changes)
python mlops/mlops.py build-image
```

## Commands

### `launch` - Run a single training job

```bash
python mlops/mlops.py launch <experiment_name> <run_name> [options]

Options:
  --machine, -m        Machine preset (see table below). Default: T4
  --dataset-path, -d   GCS path to training dataset
  --eval-dataset-path  GCS path to eval dataset
  --mode               Run mode: train or sweep. Default: train
  --project, -p        GCP project (DRILAX, NOA, ...). Default: DRILAX
  --verbose, -v        Show Docker build/push output

# Examples:
python mlops/mlops.py launch baseline run1 --machine T4
python mlops/mlops.py launch hyperparam_sweep lr_1e-3 -m L4

# Hyperparameter sweep (uses [sweep.*] sections from drive.ini)
python mlops/mlops.py launch my_sweep run1 --project NOA --machine T4 --mode sweep --max-runs 5 --dataset-path /gcs/valeo-cp2879-driving-policy/datasets/womd/1000
```

### `launch-batch` - Run multiple experiments/runs

Define experiments in a YAML file:

```yaml
experiments:
  # Experiment with multiple runs
  - name: hyperparam_sweep
    machine: T4
    params:
      env.num-agents: 1024
    runs:
      - name: lr_1e-3
        params:
          train.learning-rate: 0.001
      - name: lr_5e-4
        params:
          train.learning-rate: 0.0005

  # Single run experiment (no 'runs' field — experiment name used as run name)
  - name: baseline_T4
    machine: T4
    params:
      train.minibatch-size: 11648
```

Then run:

```bash
python mlops/mlops.py launch-batch mlops/configs/experiments.yaml
```

One Docker image is built per project, compiled for the union of all CUDA architectures
used by the runs in that project. Mixed machine types in the same batch are supported.

### `train-test` - CI smoke test

Launches the runs defined in `mlops/configs/smoke-tests.yaml`. Called automatically on PR merge.

```bash
python mlops/mlops.py train-test
```

### `build-image` - Build base Docker image

Only needed when Python dependencies change (torch, heavyball, etc.):

```bash
python mlops/mlops.py build-image
```

## Machine Presets

| Preset      | Machine Type       | GPU                  | Count |
|-------------|--------------------|----------------------|-------|
| T4          | n1-standard-16     | NVIDIA T4            | 1     |
| T4_2        | n1-standard-32     | NVIDIA T4            | 2     |
| T4_4        | n1-standard-32     | NVIDIA T4            | 4     |
| L4          | g2-standard-16     | NVIDIA L4            | 1     |
| L4_8        | g2-standard-96     | NVIDIA L4            | 8     |
| A100        | a2-highgpu-1g      | NVIDIA A100          | 1     |
| A100_4      | a2-megagpu-16g     | NVIDIA A100          | 4     |
| RTX6000     | g4-standard-48     | NVIDIA RTX Pro 6000  | 1     |
| RTX6000_2   | g4-standard-96     | NVIDIA RTX Pro 6000  | 2     |
| RTX6000_4   | g4-standard-192    | NVIDIA RTX Pro 6000  | 4     |
| RTX6000_8   | g4-standard-384    | NVIDIA RTX Pro 6000  | 8     |

## File Structure

```
mlops/
├── mlops.py              # Main CLI
├── run_training.sh       # Container entrypoint script
├── configs/
│   ├── smoke-tests.yaml  # CI smoke test runs
│   └── experiments.yaml  # (user-created) batch experiment configs
├── dockerfile/
│   ├── .dockerignore     # Docker ignore file (copied to root during build)
│   ├── build.dockerfile  # Base image with Python deps (rebuild on dep changes)
│   └── launch.dockerfile # Training image (rebuilds on code changes)
```
