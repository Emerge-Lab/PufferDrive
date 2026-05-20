# Failure mining workflow

How to roll a trained policy out, capture compact replays, and produce a
browser-viewable HTML index of episodes. Pairs with `pufferl.mine_failures`
and `pufferlib/mining_viz.py`.

## TL;DR

```bash
# Roll the policy out for 100 episodes, save compact replays for "failures",
# render HTML for each + a sortable index.
puffer mine_failures puffer_drive \
    --load-model-path /path/to/model_011000.pt \
    --mine.output-dir ./failure_mining/baseline_011000 \
    --mine.num-episodes 100 \
    --vec.backend Serial             # see "Multiprocessing hang" below

# Outputs:
#   ./failure_mining/baseline_011000/
#     replays/episode_NNNNNN.replay.zlib   ← one per failed episode
#     renders/episode_NNNNNN.html          ← per-replay viewer
#     renders/index.html                   ← sortable summary
#     episodes.csv                         ← all episodes, all metrics
```

Open the index in a browser:

```bash
open ./failure_mining/baseline_011000/renders/index.html
```

## What gets captured

A compact replay bundle is a pickled+zlib'd `schema_version=2` dict containing
per-step agent state, traffic state, and observation arrays for a single
episode. Bundles are produced **C-side** when `capture_compact_replay=True`
is passed to `Drive(...)`. `mine_failures` sets this automatically.

Each saved bundle is paired with a metadata row in `episodes.csv` including
`episode_return`, `collision_rate`, `offroad_rate`, `num_goals_reached`,
`avg_distance_per_infraction`, etc. The HTML viewer (`pufferlib/mining_viz.py`)
reads the bundle and replays it in-browser on a top-down canvas, with optional
overlays for the agent's observed FOV, partner circle, goal route, and waypoint
markers.

## `mine.score_threshold` — gotcha

The `mine_failures` selection rule is "save replay if and only if
`episode_return < score_threshold`". The docstring claims `-inf` means "capture
every episode" — that's wrong: `episode_return < -inf` is never true, so the
default captures **nothing**. To actually save episodes:

```bash
# Capture every episode (works with any non-degenerate return):
--mine.score-threshold 1e9

# Capture only "true" failures (negative returns):
--mine.score-threshold 0
```

`episodes.csv` always contains all N episodes' metadata regardless of threshold
— only the bundle save + HTML render is gated.

## Multiprocessing hang — use `--vec.backend Serial`

`pufferl.mine_failures` goes through `pufferlib.vector.make(...)` with the
drive.ini default `backend=Multiprocessing`. Even with `num_envs=1,
num_workers=1`, that backend **forks** workers post-torch-import. Forking after
torch has been imported in the parent is a classic deadlock for CUDA — the
child can hang on CUDA initialization, and the parent sits forever on the IPC
pipe.

Symptoms: CPU 100% in the parent, RSS frozen, no `[mine_failures] target
episodes=...` print, never produces output. If you let it sit for ~10 minutes
nothing changes.

Fix: force the in-process backend.

```bash
--vec.backend Serial
```

This keeps the env in the same process as the policy. No fork, no hang. The
single-env nature of mining means the throughput cost is negligible.

## Tuning the rollout config

The mining env config comes from drive.ini's `[mine]` section plus per-CLI
overrides. Useful knobs:

```bash
# Larger output (slower):
--mine.num-episodes 500

# Replay mode (drive recorded nuPlan / Waymo scenarios):
--env.simulation-mode replay \
--env.control-mode control_sdc_only \
--env.map-dir /path/to/recorded_bins \
--env.init-steps 10 \
--env.scenario-length 200

# Looser goal radius (useful if the trained policy struggles with the
# stricter default; default 2m, max 12m under reward randomization):
--env.goal-radius 6

# Closer-spaced goals (mining a policy that wasn't trained on these):
--env.min-waypoint-spacing 10 \
--env.max-waypoint-spacing 15
```

## Resume + obs-shape gotcha

`mine_failures` does **not** read the sibling `config.yaml` next to
`load_model_path` — only `pufferl.train` does. If the checkpoint was trained
with non-default `policy.*` or `rnn.*` dimensions (e.g. `input_size=128`,
`backbone_num_layers=4`), you'll get a shape mismatch on `load_state_dict`
unless you pass them on the CLI:

```bash
--policy.input-size 128 \
--policy.actor-hidden-size 512 \
--policy.actor-num-layers 0 \
--policy.backbone-hidden-size 512 \
--policy.backbone-num-layers 4 \
--policy.critic-hidden-size 512 \
--policy.critic-num-layers 0 \
--policy.encoder-gigaflow True \
--policy.split-network False \
--rnn.hidden-size 512 \
--rnn.input-size 512
```

You can read the right values out of the checkpoint's sibling `config.yaml`
(under `policy:` and `rnn:`) and pass them through.

## On the cluster

Mining is GPU-bound on the policy forward pass but memory-light compared to
training (single env, no rollout buffer, no PPO update). 48 GB RAM and a
60-minute time limit are plenty for 100 episodes. The same `submit_cluster.py`
flow as training works — just override `--main` to invoke `mine_failures`:

```bash
python3 scripts/submit_cluster.py \
    --save_dir /scratch/$USER/runs \
    --prefix mine \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --account <acct> --partition <gpu-partition> --time 60 \
    --mem 48gb --cpus 8 \
    --container \
    --main "-m pufferlib.pufferl mine_failures puffer_drive" \
    --args \
        load_model_path=<path-to-ckpt> \
        mine.output_dir=/scratch/$USER/failure_mining/out \
        mine.num_episodes=100 \
        mine.score_threshold=1e9 \
        vec.backend=Serial
```

See [`docs/cluster_training.md`](cluster_training.md) for the one-time
submitit setup (`pip install --user submitit pyyaml cloudpickle` on the
system python) and the rationale for why `submit_cluster.py` must be run
from the login node rather than inside the container.

Outputs land on `/scratch`; pull them down with `rsync` for in-browser
viewing.

## Viewer features (`mining_viz.py`)

The per-episode HTML viewer supports:

- Frame scrubber + play/pause + speed control.
- Toggle observation overlay (FOV rectangle, partner circle, observed-entity
  highlights, goal route, waypoint markers).
- Toggle road segment / road edge / lane line rendering.
- Map background (CARLA / nuPlan / Waymo road graph from the bundle's
  embedded `simulation_mode`).

The index (`renders/index.html`) is a sortable table linking to each per-episode
HTML, with the metadata columns from `episodes.csv` (failure metrics, scenario
ID, map name).
