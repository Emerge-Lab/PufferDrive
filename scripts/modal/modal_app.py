"""Nightly PufferDrive training on Modal.

Schedules a cron that, every night, launches:
  - 3 seeds × scripts/cluster_configs/single_agent_speed_run.yaml →
    wandb project `nightly-single`, group = today's UTC date
  - 3 seeds × scripts/cluster_configs/nightly_best.yaml →
    wandb project `nightly-multi`, group = today's UTC date
each on its own 1× A100-80GB container, all in parallel.

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
      --seed 0 --run-name local_smoke --wandb-group smoke \\
      --wandb-project nightly-single
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image: Dockerfile base + repo + built C extensions.
# ---------------------------------------------------------------------------
# Modal re-imports this module inside the container to find function
# definitions, so module-level path resolution has to work in both contexts.
# Locally (modal run / modal deploy) the file lives at
# <repo>/scripts/modal/modal_app.py — three levels deep, setup.py is at the
# resolved repo root. In the container Modal mirrors it to /root/modal_app.py
# where the image is already built and these path references are inert; the
# baked repo lives at /workspace via add_local_dir.
def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        if (ancestor / "setup.py").exists():
            return ancestor
    return Path("/workspace")


REPO_ROOT = _resolve_repo_root()
DOCKERFILE = REPO_ROOT / "scripts" / "modal" / "Dockerfile"

_IGNORE_SEGMENTS = (
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


def _ignore(path) -> bool:
    """add_local_dir ignore callable: True drops the file from the image."""
    s = str(path)
    return any(seg in s for seg in _IGNORE_SEGMENTS)


image = (
    modal.Image.from_dockerfile(str(DOCKERFILE))
    .workdir("/workspace")
    .add_local_dir(str(REPO_ROOT), "/workspace", copy=True, ignore=_ignore)
    # `-e .` triggers setup.py build_ext for the C + CUDA extensions.
    # --no-build-isolation lets uv use the torch already in /opt/venv
    # instead of building it in a fresh sandbox (~2 min saved).
    # The trailing numpy<2 / pandas<2.2 are additional constraints fed
    # to uv's resolver alongside install_requires — without them an
    # unconstrained `pandas` in setup.py would resolve to pandas 3+ and
    # drag numpy 2 in, breaking the C extension's numpy 1 ABI
    # (NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION in setup.py).
    .run_commands(
        'uv pip install --no-build-isolation -e . "numpy<2" "pandas<2.2"',
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
WANDB_PROJECT_SINGLE = "nightly-single"
WANDB_PROJECT_MULTI = "nightly-multi"
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
    # Enough cores for drive.ini's vec.num_workers default (20) plus headroom.
    # PufferLib refuses to start if num_workers > os.cpu_count().
    cpu=24,
    # Enough RAM for the C-side map cache + vec env workers. Bump for the
    # multi-agent (nightly_best) config which loads heavier maps.
    memory=32 * 1024,
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
    wandb_project: str,
) -> str:
    """Run one training job. Returns the run_name on success."""
    import os
    import subprocess

    import yaml

    cfg_path = Path("/workspace") / yaml_path
    cfg = yaml.safe_load(cfg_path.read_text())

    cli_args = yaml_to_cli_args(cfg)
    cli_args += [
        "--train.seed",
        str(seed),
        # PufferLib refuses to start if num_workers > os.cpu_count(). Modal's
        # T4/A100 containers come with ~16 logical cores; pin workers/envs
        # to a safe value regardless of yaml/drive.ini defaults.
        "--vec.num-workers",
        "8",
        "--vec.num-envs",
        "8",
        "--wandb-project",
        wandb_project,
        "--wandb-group",
        wandb_group,
        "--run-name",
        run_name,
    ]

    # pufferl.py has no --wandb-entity flag; wandb picks the entity up from
    # WANDB_ENTITY in the env. WANDB_API_KEY arrives via the wandb-emerge
    # Modal secret.
    env = {**os.environ, "WANDB_ENTITY": WANDB_ENTITY}

    cmd = ["python", "-m", "pufferlib.pufferl", "train", "puffer_drive", *cli_args]
    print(f"[modal] launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd="/workspace", env=env)
    return run_name


# ---------------------------------------------------------------------------
# Cron entrypoint — fans out to 6 parallel `train` calls.
# ---------------------------------------------------------------------------
# Schedule: 04:00 UTC daily ≈ midnight ET / 21:00 PT previous day. Adjust if
# the user wants a different wall-clock time.
@app.function(
    timeout=60 * 60,
    secrets=[WANDB_SECRET],
    schedule=modal.Cron("0 4 * * *"),
)
def nightly() -> None:
    """Fan out to per-seed, per-config training runs in parallel.

    Single-agent runs land in wandb project `nightly-single`, multi-agent in
    `nightly-multi`. Within each project the wandb_group is today's UTC date
    so a night's 3 seeds cluster together.
    """
    from datetime import datetime, timezone

    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    seeds = [0, 1, 2]

    jobs: list[tuple[str, int, str, str, str]] = []
    for seed in seeds:
        jobs.append(
            (
                "scripts/cluster_configs/single_agent_speed_run.yaml",
                seed,
                f"{date}_seed{seed}",
                date,
                WANDB_PROJECT_SINGLE,
            )
        )
        jobs.append(
            (
                "scripts/cluster_configs/nightly_best.yaml",
                seed,
                f"{date}_seed{seed}",
                date,
                WANDB_PROJECT_MULTI,
            )
        )

    print(f"[modal] nightly fan-out: {len(jobs)} runs", flush=True)
    # starmap blocks until every run finishes. exceptions in a single run
    # propagate; other runs continue.
    results = list(train.starmap(jobs, return_exceptions=True))
    for (yaml_path, seed, run_name, _group, project), result in zip(jobs, results):
        status = "OK" if not isinstance(result, BaseException) else f"FAIL: {result!r}"
        print(f"[modal] {project}/{run_name} ({yaml_path}, seed={seed}): {status}", flush=True)


# ---------------------------------------------------------------------------
# Local entrypoint (modal run scripts/modal/modal_app.py::nightly).
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    """Default `modal run` target — triggers the nightly fan-out immediately."""
    nightly.remote()
