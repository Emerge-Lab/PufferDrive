# Failure mining workflow

How to roll a trained policy out, capture compact replays, and produce a
browser-viewable HTML index of episodes. Pairs with `pufferl.mine_failures`
and `pufferlib/mining_viz.py`.

## TL;DR

```bash
# Roll the policy out for 100 episodes, save compact replays for episodes
# whose episode_return falls below the threshold, render HTML for each +
# a sortable index.
puffer mine_failures puffer_drive \
    --load-model-path /path/to/model_011000.pt \
    --mine.output-dir ./failure_mining/baseline_011000 \
    --mine.num-episodes 100 \
    --mine.score-threshold 1e9 \
    --vec.backend Serial
```

Outputs:

```
./failure_mining/baseline_011000/
    replays/episode_NNNNNN.replay.zlib   one per saved episode
    renders/episode_NNNNNN.html          per-replay viewer
    renders/index.html                   sortable summary
    episodes.csv                         all episodes, all metrics
```

Open the index in a browser:

```bash
open ./failure_mining/baseline_011000/renders/index.html
```

## What gets captured

A compact replay bundle is a pickled+zlib'd `schema_version=2` dict containing
per-step agent state, traffic state, and observation arrays for a single
episode. Bundles are produced C-side when `capture_compact_replay=True` is
passed to `Drive(...)`. `mine_failures` sets this automatically.

Each saved bundle is paired with a metadata row in `episodes.csv` including
`episode_return`, `collision_rate`, `offroad_rate`, `num_goals_reached`,
`avg_distance_per_infraction`, etc. The HTML viewer (`pufferlib/mining_viz.py`)
reads the bundle and replays it in-browser on a top-down canvas, with optional
overlays for the agent's observed FOV, partner circle, goal route, and waypoint
markers.

## `mine.score_threshold` selection

The save rule is "write replay if and only if `episode_return < score_threshold`".

- `--mine.score-threshold 1e9` captures every episode (any real return is
  less than 1e9).
- `--mine.score-threshold 0` captures only negative-return ("true failure")
  episodes.
- Default `-inf` captures **nothing** — useful only if you want `episodes.csv`
  metrics without the bundle overhead.

`episodes.csv` always contains all N episodes' metadata regardless of
threshold; only the bundle save + HTML render is gated.

## `--vec.backend Serial`

Mining must use `--vec.backend Serial`. The drive.ini default
`Multiprocessing` backend forks workers post-torch-import, which deadlocks on
CUDA in the child process. Symptom is a parent process at 100% CPU with no
visible progress and no `[mine_failures] target episodes=...` print.

`Serial` keeps the env in the same process as the policy. Mining is a single
env / single rollout workflow, so the throughput cost is negligible.

## Tuning the rollout config

The mining env config comes from drive.ini's `[mine]` section plus per-CLI
overrides:

```bash
# Larger output (slower):
--mine.num-episodes 500

# Replay mode (drive recorded nuPlan / Waymo scenarios):
--env.simulation-mode replay \
--env.control-mode control_sdc_only \
--env.map-dir /path/to/recorded_bins \
--env.init-steps 10 \
--env.scenario-length 200

# Looser goal radius (default 2 m, up to 12 m under reward randomization):
--env.goal-radius 6

# Closer-spaced goals:
--env.min-waypoint-spacing 10 \
--env.max-waypoint-spacing 15
```

## Loading checkpoints with non-default architecture

`mine_failures` does not read the sibling `config.yaml` next to
`load_model_path` (only `pufferl.train` does). If the checkpoint was trained
with non-default `policy.*` or `rnn.*` dimensions (e.g. `input_size=128`,
`backbone_num_layers=4`), pass them on the CLI to match the saved state dict:

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
(under `policy:` and `rnn:`) and pass them through. The error if you forget
is a wall of `size mismatch for ...` lines from `policy.load_state_dict`.

## On the cluster

Mining is GPU-bound on the policy forward pass but memory-light compared to
training (single env, no rollout buffer, no PPO update). 48 GB RAM and a
60-minute time limit are plenty for 100 episodes. The same `submit_cluster.py`
flow as training works — override `--main` to invoke `mine_failures`:

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

See [`docs/cluster_training.md`](cluster_training.md) for one-time setup of
the login-side submitit (`python3 -m pip install --user submitit pyyaml
cloudpickle`).

Outputs land on `/scratch`; pull them down with `rsync` for in-browser viewing.

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
