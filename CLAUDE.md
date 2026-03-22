# PufferDrive

## Cluster Access (NYU Torch)

- SSH host: `torch` (configured in ~/.ssh/config, user ev2237)
- Code path on cluster: `/scratch/ev2237/code/PufferDrive`
- Container overlay: `/scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3`
- Container image: `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`
- SLURM accounts (user has multiple):
  - `torch_pr_355_general`
  - `torch_pr_355_tandon_advanced`
  - `torch_pr_355_tandon_priority`
  - `torch_pr_104_general`
  - `torch_pr_104_tandon_advanced`
  - `torch_pr_102_general`
  - `torch_pr_102_tandon_advanced`
  - `torch_pr_45_general`
  - `torch_pr_45_tandon_advanced`

## Job Launch Strategy

When submitting a job, submit it on **all available accounts** simultaneously, monitor which one starts running first, then cancel the rest. This avoids waiting in queue on a single account.

## Building C Extension

Must be done on a **compute node** (not login node) inside singularity. Do NOT specify a partition — let SLURM pick the default:
```bash
srun --account=torch_pr_355_general --gres=gpu:1 --cpus-per-task=4 --mem=16gb --time=15 \
  singularity exec --nv \
    --overlay /scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    bash -c 'source /ext3/env.sh && cd /scratch/ev2237/code/PufferDrive && python setup.py build_ext --inplace --force'
```

Rebuild after every branch checkout before launching jobs.

## Launching Training Jobs

```bash
source /scratch/ev2237/login_venv/bin/activate
cd /scratch/ev2237/code/PufferDrive
python scripts/submit_cluster.py \
  --save_dir /scratch/ev2237/experiments \
  --compute_config scripts/cluster_configs/nyu_greene.yaml \
  --program_config scripts/cluster_configs/train_base.yaml \
  --prefix <run_name> \
  --container \
  --container_overlay /scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3
```

- Default training config: `pufferlib/config/ocean/drive.ini`
- Base cluster program config: `scripts/cluster_configs/train_base.yaml`
- Compute config: `scripts/cluster_configs/nyu_greene.yaml`

## Debugging Memory Issues

When investigating OOM or memory problems, do NOT hypothesize — write a script that actually measures memory usage step by step. Use `resource.getrusage` or `/proc/self/status` to track RSS before and after each allocation/init step.

## Local Development

- Use `uv` (not conda/pip/python) to run Python locally
- Name wandb runs descriptively

## Checking on jobs.
Don't ever wait more than 30 seconds to check on jobs. Always 30 seconds or less.
