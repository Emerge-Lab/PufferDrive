# Modal nightly training

Nightly cron that launches PufferDrive training on Modal — 3 seeds of
`single_agent_speed_run.yaml` plus 3 seeds of `nightly_best.yaml`, all on
1× A100-80GB in parallel, all logging to wandb project `nightly-modal` in the
`emerge_` org.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | CUDA 12.8.1 + cuDNN + Ubuntu 24.04 base, system libs, Python 3.12, torch. Slow layer — rarely rebuilt. |
| `modal_app.py` | Modal app — bakes the repo + builds C extensions on top of the Dockerfile, defines the per-seed `train` function and the `nightly` cron entrypoint. |

The training yamls themselves live in `scripts/cluster_configs/` and are shared
with the Greene-side launcher.

## One-time setup

```bash
# Install + auth Modal CLI (host machine)
pip install modal
modal token new

# Create the wandb secret. Paste the API key from https://wandb.ai/authorize.
modal secret create wandb-emerge WANDB_API_KEY=<key>
```

## Deploy the nightly cron

```bash
modal deploy scripts/modal/modal_app.py
```

Modal hashes the source — re-run after any code change to rebuild the image
and update the deployed cron. The first deploy builds the Dockerfile (~5 min);
subsequent deploys only rebuild the `pip install -e .` layer when repo files
change (~1 min).

The cron is `0 4 * * *` (04:00 UTC daily). Adjust the `modal.Cron(...)` arg in
`modal_app.py` to change the wall-clock time.

## Trigger runs manually

```bash
# Run the full 6-job fan-out now (without waiting for cron):
modal run scripts/modal/modal_app.py::nightly

# Run a single seed/config (useful for smoke tests):
modal run scripts/modal/modal_app.py::train \
    --yaml-path scripts/cluster_configs/single_agent_speed_run.yaml \
    --seed 0 --run-name local_smoke --wandb-group smoke
```

## Inspect / cancel

```bash
modal app list                                      # show deployed apps
modal app logs pufferdrive-nightly                  # tail logs (running app)
modal app stop pufferdrive-nightly                  # remove the cron
```

Per-container logs (one per training run) appear in the Modal dashboard
under the `pufferdrive-nightly` app.

## Cost note

A100-80GB on Modal is ~$3.20/h. A 12 h training run × 6 jobs = ~$230/night.
Bring down by lowering `train.total_timesteps` in the yamls, dropping the
`--gpu` to `A100` (40GB), or limiting to fewer seeds.
