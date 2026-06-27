"""Nightly PufferDrive training on Modal.

Schedules a cron that, every night, launches:
  - 3 seeds × scripts/cluster_configs/single_agent_speed_run.yaml
  - 3 seeds × scripts/cluster_configs/nightly_best.yaml
each on its own 1× A100-80GB container, all in parallel. All 6 runs log to the
same wandb project (overridden here, not in the yamls) under distinct groups.

One-time setup:
  modal token new                     # if not already authenticated
  modal secret create wandb-emerge WANDB_API_KEY=<paste from wandb.ai/authorize>

Deploy the cron (idempotent — re-run after each code change):
  modal deploy scripts/modal/modal_app.py

Trigger the full nightly fan-out manually (without waiting for cron):
  modal run scripts/modal/modal_app.py::nightly

Trigger one run manually:
  modal run scripts/modal/modal_app.py::train \\
      --yaml-path scripts/cluster_configs/single_agent_speed_run.yaml \\
      --seed 0 --run-name local_smoke --wandb-group smoke
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image: Dockerfile base + repo + built C extensions.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "scripts" / "modal" / "Dockerfile"

image = (
    modal.Image.from_dockerfile(
        str(DOCKERFILE),
        context_mount=modal.Mount.from_local_dir(
            str(REPO_ROOT),
            remote_path="/workspace",
            # Skip heavy artefacts that aren't needed at runtime.
            condition=lambda p: not any(
                seg in p
                for seg in (
                    ".git/",
                    ".venv/",
                    "wandb/",
                    "experiments/",
                    "runs/",
                    "build/",
                    "extern/",
                    "__pycache__/",
                    ".pytest_cache/",
                    ".mypy_cache/",
                    ".ruff_cache/",
                )
            ),
        ),
    )
    .workdir("/workspace")
    # `-e .` triggers setup.py build_ext for the C + CUDA extensions.
    # --no-build-isolation lets it see the torch already in the base image
    # (otherwise pip builds a fresh torch in a sandbox — slow).
    .run_commands(
        "pip install --break-system-packages --no-build-isolation -e .",
        # Smoke-import to fail the build early if extensions didn't compile.
        "python -c 'import pufferlib.ocean.drive.binding; "
        "import pufferlib._C; print(\"extensions OK\")'",
    )
)

app = modal.App("pufferdrive-nightly", image=image)

# ---------------------------------------------------------------------------
# Secrets + wandb routing.
# ---------------------------------------------------------------------------
# Create with:
#   modal secret create wandb-emerge WANDB_API_KEY=<key>
WANDB_SECRET = modal.Secret.from_name("wandb-emerge")
WANDB_PROJECT = "nightly-modal"
WANDB_ENTITY = "emerge_"

# ---------------------------------------------------------------------------
# yaml-to-CLI conversion. Mirrors scripts/submit_cluster.py's logic so the
# same yamls work on Modal and Greene.
# ---------------------------------------------------------------------------
BOOLEAN_FLAGS = {"wandb", "neptune"}


def yaml_to_cli_args(cfg: dict) -> list[str]:
    args: list[str] = []
    for key, val in cfg.items():
        cli_key = key.replace("_", "-")
        if key in BOOLEAN_FLAGS:
            if val in (True, "True", "true", 1, "1"):
                args.append(f"--{cli_key}")
            # False: omit the flag entirely
            continue
        args.append(f"--{cli_key}")
        args.append(str(val))
    return args


# ---------------------------------------------------------------------------
# Per-run training function. One container per call.
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    # 12h cap; nightly runs at total_timesteps=1B/10B can take this long. Bump
    # if a real run wedges short of completion.
    timeout=12 * 3600,
    secrets=[WANDB_SECRET],
    # Retries on container infra failure (preempt, OOM at boot). The training
    # job itself shouldn't restart on RL crashes — let the run die so we
    # notice.
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def train(
    yaml_path: str,
    seed: int,
    run_name: str,
    wandb_group: str,
) -> str:
    """Run one training job. Returns the run_name on success."""
    import subprocess

    import yaml

    cfg_path = Path("/workspace") / yaml_path
    cfg = yaml.safe_load(cfg_path.read_text())

    cli_args = yaml_to_cli_args(cfg)
    cli_args += [
        "--train.seed",
        str(seed),
        "--wandb-project",
        WANDB_PROJECT,
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-group",
        wandb_group,
        "--run-name",
        run_name,
    ]

    cmd = ["python", "-m", "pufferlib.pufferl", "train", "puffer_drive", *cli_args]
    print(f"[modal] launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd="/workspace")
    return run_name


# ---------------------------------------------------------------------------
# Cron entrypoint — fans out to 6 parallel `train` calls.
# ---------------------------------------------------------------------------
# Schedule: 04:00 UTC daily ≈ midnight ET / 21:00 PT previous day. Adjust if
# the user wants a different wall-clock time.
@app.function(timeout=60 * 60, secrets=[WANDB_SECRET])
@modal.schedule(modal.Cron("0 4 * * *"))
def nightly() -> None:
    """Fan out to per-seed, per-config training runs in parallel."""
    from datetime import datetime, timezone

    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    seeds = [0, 1, 2]

    jobs: list[tuple[str, int, str, str]] = []
    for seed in seeds:
        jobs.append(
            (
                "scripts/cluster_configs/single_agent_speed_run.yaml",
                seed,
                f"{date}_single_seed{seed}",
                "modal_single_agent",
            )
        )
        jobs.append(
            (
                "scripts/cluster_configs/nightly_best.yaml",
                seed,
                f"{date}_multi_seed{seed}",
                "modal_multi_agent",
            )
        )

    print(f"[modal] nightly fan-out: {len(jobs)} runs", flush=True)
    # starmap blocks until every run finishes. exceptions in a single run
    # propagate; other runs continue.
    results = list(train.starmap(jobs, return_exceptions=True))
    for (yaml_path, seed, run_name, _group), result in zip(jobs, results):
        status = "OK" if not isinstance(result, BaseException) else f"FAIL: {result!r}"
        print(f"[modal] {run_name} ({yaml_path}, seed={seed}): {status}", flush=True)


# ---------------------------------------------------------------------------
# Local entrypoint (modal run scripts/modal/modal_app.py::nightly).
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    """Default `modal run` target — triggers the nightly fan-out immediately."""
    nightly.remote()
