# PufferDrive

RL driving simulator built on PufferLib. C simulation core with Python/PyTorch training loop.

## Project structure

- `pufferlib/ocean/drive/` — C simulation: `drive.h` (sim logic), `datatypes.h` (structs), `binding.c` (Python binding), `render.h` (3D rendering), `visualize.c` (offline video binary), `drivenet.h` (C inference)
- `pufferlib/ocean/env_binding.h` — Generic C-Python binding layer (shared memory for obs/actions/rewards/terminals/truncations)
- `pufferlib/ocean/torch.py` — PyTorch policy network
- `pufferlib/pufferl.py` — Training loop (PPO + GAE)
- `pufferlib/ocean/drive/drive.py` — Python env wrapper
- `pufferlib/config/ocean/drive.ini` — Environment config (reward coefficients, dynamics, etc.)
- `scripts/submit_cluster.py` — SLURM job submission with code isolation
- `scripts/cluster_configs/` — YAML configs for cluster compute and training args
- `resources/drive/binaries/` — Map binary files (large, symlinked during code isolation)

## Building locally

```bash
uv pip install -e '.[drive]' --no-build-isolation
```

Always use `uv`, never conda/pip/python directly.

### Visualize binary

```bash
bash scripts/build_ocean.sh visualize fast
```

Use `fast` mode by default. Only use `local` (ASAN) when debugging memory errors — it adds ~5x overhead.

## NYU Greene HPC cluster

### Key paths

- **All files on cluster must be in `/scratch/ev2237/`** — never `/home/ev2237/` (quota is tiny)
- Code: `/scratch/ev2237/code/PufferDrive`
- Experiments: `/scratch/ev2237/experiments`
- Venv: `/scratch/ev2237/venvs/pufferdrive`
- Login venv (for submitit): `/scratch/ev2237/login_venv`
- Container overlay: `/scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3`
- Container image: `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`

### SLURM accounts

Multiple accounts are available. When launching jobs, submit to several accounts in parallel and cancel the remaining once one starts:

- `torch_pr_355_general` (primary)
- `torch_pr_45_general`
- `torch_pr_102_general`
- `torch_pr_104_general`
- `torch_pr_923_general`
- `torch_pr_924_general`
- `torch_pr_355_tandon_priority`
- `torch_pr_924_tandon_priority`
- `torch_pr_45_tandon_advanced`
- `torch_pr_102_tandon_advanced`
- `torch_pr_104_tandon_advanced`
- `torch_pr_355_tandon_advanced`
- `torch_pr_924_tandon_advanced`

### Building on the cluster

**NEVER build on the login node.** The login node will OOM-kill nvcc and has no GPU, producing .so files with wrong torch ABI symbols. Always build via `srun` inside singularity:

```bash
ssh torch "cd /scratch/ev2237/code/PufferDrive && srun \
  --account=torch_pr_355_general --partition=l40s_public,h200_public \
  --cpus-per-task=4 --mem=16gb --time=10 --gres=gpu:1 \
  singularity exec --nv --overlay /scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  bash -c 'source /ext3/env.sh && export TORCH_CUDA_ARCH_LIST=\"8.0;9.0\" && python setup.py build_ext --inplace --force'"
```

- `TORCH_CUDA_ARCH_LIST="8.0;9.0"` is required — the cluster has L40S (8.0), H100 (9.0), and H200 (9.0). Without the right arch, jobs crash with `RuntimeError: no kernel image is available for execution on the device`
- **Must rebuild after every branch checkout** — the .so persists across git switches and silently uses stale C code
- **Must rebuild whenever C code changes** (drive.h, datatypes.h, binding.c, env_binding.h, drivenet.h) — Python-only changes don't require a rebuild
- Do NOT use `uv pip install` or `pip install` on the login node — it builds outside singularity and produces incompatible .so files

### Launching training jobs

Run `submit_cluster.py` from the login node (this is fine — it only submits SLURM jobs).

**Always submit to multiple accounts simultaneously** to maximize scheduling speed. Cancel the remaining jobs once one starts running:

```bash
# Submit the same job under multiple accounts
for acct in torch_pr_355_general torch_pr_45_general torch_pr_102_general torch_pr_924_general; do
  ssh torch "source /scratch/ev2237/login_venv/bin/activate && \
    cd /scratch/ev2237/code/PufferDrive && \
    python scripts/submit_cluster.py \
      --save_dir /scratch/ev2237/experiments \
      --compute_config scripts/cluster_configs/nyu_greene.yaml \
      --program_config scripts/cluster_configs/train_base.yaml \
      --prefix <descriptive-run-name> \
      --account $acct \
      --container \
      --container_overlay /scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3"
done
```

Then monitor with `squeue`, and `scancel` the pending jobs once one starts.

**Wandb naming is required.** Every run must have a descriptive `--prefix` (becomes the wandb run name). Always specify `--wandb-project`. If the project is not obvious from context, ask the user which project to use.

Override args: `--args 'env.goal_behavior=1' 'train.render=True'`

Sweep: `--args 'train.seed=42:55:1'` (colon-separated values produce one job each)

### Maps

Use CARLA_2D maps by default: `env.map_dir=resources/drive/binaries/carla_data`. Do not use 3D maps unless explicitly told to.

### Monitoring jobs

```bash
ssh torch "squeue -u ev2237"                    # all jobs
ssh torch "squeue -u ev2237 -j <jobid>"         # specific job
ssh torch "scancel <jobid>"                     # cancel job
```

When polling, check at most every 30 seconds.

## Git workflow

- **Never push directly to `3.0` or `main`** — always use feature branches and PRs
- **Never merge PRs** — leave merging to the user
- After pushing a PR, check pre-commit CI: `gh pr checks <PR#> | grep pre-commit`. Fix formatting failures immediately.
- Give wandb runs descriptive names via `--prefix` (not random defaults)

## Common pitfalls

- **Stale .so after branch switch**: Always rebuild C extension after `git checkout`
- **Login node OOM**: Never run nvcc, Python training, or env code on the login node
- **`/home` quota**: All scratch data, venvs, experiments must go in `/scratch/ev2237/`
- **Arch mismatch**: Without `9.0` in TORCH_CUDA_ARCH_LIST, H200 jobs crash with "no kernel image available"
- **Memory pressure**: 16 workers + 8 maps needs 64GB; 32GB will hang
