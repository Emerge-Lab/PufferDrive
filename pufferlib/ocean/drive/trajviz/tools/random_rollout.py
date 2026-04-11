"""Render a full episode of a random-policy rollout to MP4 via trajviz.

Standalone smoke test that exercises the entire trajviz pipeline against
a real Drive simulation (not the synthetic grid in tests/test_main.c):

    1. Spin up a Drive env on one map.
    2. Reset and step with uniformly random actions for the full episode.
    3. Pull per-step (x, y, heading) out via get_sim_trajectories().
    4. Load the source map .bin via map_io and mean-center it with the
       env's world_mean (so road geometry lines up with sim coordinates).
    5. Drive a single Renderer.render_episode call — both top-down and
       BEV views in one pass — and write two MP4s.

Usage:
    python -m pufferlib.ocean.drive.trajviz.tools.random_rollout \\
        [--map pufferlib/resources/drive/binaries/map_001.bin] \\
        [--out-dir /tmp] [--episode-length 91] [--seed 0]

The C extension must already be built (TRAJVIZ=1 python setup.py build_ext --inplace).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import map_io
from pufferlib.ocean.drive.trajviz import Renderer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--map",
        type=Path,
        default=Path("pufferlib/resources/drive/binaries/map_001.bin"),
        help="Path to a single .bin map file.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("/tmp"))
    p.add_argument("--episode-length", type=int, default=91, help="Number of sim steps to roll out.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Cap on TOTAL agent slots in the env. Default 2 matches "
        "the typical WOSAC tracks_to_predict count for one map; "
        "raising it makes Drive instantiate the same map across "
        "multiple sub-envs to fill the cap.",
    )
    p.add_argument(
        "--init-mode",
        default="create_only_controlled",
        choices=("create_all_valid", "create_only_controlled", "init_variable_agent_number"),
        help="How Drive instantiates agents. 'create_only_controlled' "
        "(default here) gives random actions only to the source "
        "scenario's tracks_to_predict agents and replays the rest "
        "from log data — matching real WOSAC behavior. "
        "'create_all_valid' makes every vehicle policy-controlled.",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    if not args.map.exists():
        raise SystemExit(f"map not found: {args.map}")

    # Drive expects a directory of maps and loads num_maps of them sorted.
    # To pin it to one specific map regardless of which alphabetically
    # comes first in the source dir, copy our chosen map into a fresh
    # temp dir and point Drive at that.
    with tempfile.TemporaryDirectory(prefix="trajviz_random_") as tmpdir:
        shutil.copy(args.map, tmpdir)
        print(f"[rollout] map: {args.map}  →  {tmpdir}")

        # init_mode controls which agents become policy-controlled vs
        # expert-replayed. With 'create_only_controlled', random actions
        # only affect the source scenario's tracks_to_predict agents
        # (typically 2 in WOSAC); the rest replay their Waymo log
        # trajectories. With 'create_all_valid' the random actions move
        # everything — useful if you want to see chaos, but doesn't
        # match how the trained policy would actually be used.
        env = Drive(
            map_dir=tmpdir,
            num_maps=1,
            num_agents=args.num_agents,
            episode_length=args.episode_length,
            seed=args.seed,
            init_steps=0,
            init_mode=args.init_mode,
        )
        print(f"[rollout] num_agents={env.num_agents} episode_length={env.episode_length}")

        rng = np.random.default_rng(args.seed)
        env.reset(seed=args.seed)

        # Action buffer was pre-allocated by PufferEnv based on the
        # MultiDiscrete([91]) space — one categorical per agent in [0, 91).
        actions_shape = env.actions.shape
        actions_dtype = env.actions.dtype
        action_high = 91

        # Important: stop ONE step before episode_length so we don't trigger
        # the auto-reset at end-of-episode. The C side increments timestep
        # in c_step and resets it to 0 when timestep == episode_length, which
        # would zero out traj["lengths"]. Stepping episode_length-1 times
        # leaves timestep at episode_length-1 (no reset) with that many
        # frames recorded.
        n_steps = env.episode_length - 1
        for step in range(n_steps):
            actions = rng.integers(0, action_high, size=actions_shape, dtype=actions_dtype)
            obs, reward, done, trunc, info = env.step(actions)
            if trunc.all():
                print(f"[rollout] unexpected truncation at step {step + 1}")
                break

        traj = env.get_sim_trajectories()
        world_mean = np.asarray(env.world_mean, dtype=np.float32)
        print(f"[rollout] world_mean = {world_mean}")
        print(
            f"[rollout] valid lengths: min={int(traj['lengths'].min())} "
            f"max={int(traj['lengths'].max())} "
            f"mean={float(traj['lengths'].mean()):.1f}"
        )

        env.close()

    # Load the same source map and mean-center it. We use the path the
    # user supplied, not the temp copy (which is gone now), but they hold
    # the same bytes.
    roads_raw = map_io.load_map_roads(args.map)
    roads = map_io.mean_center_roads(roads_raw, world_mean)
    road_xy, road_offsets, road_types = map_io.roads_to_csr(roads)
    print(f"[rollout] roads: {len(roads)} polylines, {int(road_xy.shape[0])} verts")

    # Stack into (T, A, 3). get_sim_trajectories returns (A, T) per field.
    num_agents, num_steps = traj["x"].shape
    traj_xyh = np.empty((num_steps, num_agents, 3), dtype=np.float32)
    traj_xyh[..., 0] = traj["x"].T
    traj_xyh[..., 1] = traj["y"].T
    traj_xyh[..., 2] = traj["heading"].T

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_topdown = args.out_dir / "random_topdown.mp4"
    out_bev = args.out_dir / "random_bev.mp4"

    print(f"[rollout] rendering to {out_topdown.name} + {out_bev.name} ...")
    with Renderer(width=args.width, height=args.height) as r:
        r.render_episode(
            road_xy=road_xy,
            road_offsets=road_offsets,
            road_types=road_types,
            traj_xyh=traj_xyh,
            agent_lengths=traj["lengths"].astype(np.int32),
            ego_idx=-1,
            fps=args.fps,
            out_topdown=str(out_topdown),
            out_bev=str(out_bev),
        )
    print(f"[rollout] wrote {out_topdown}")
    print(f"[rollout] wrote {out_bev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
