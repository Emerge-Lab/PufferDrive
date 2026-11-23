"""Render the spiced self-play policy on the three bundled Waymo scenes.

Scenes live in reproduce/data/map_000.bin .. map_002.bin. Human replay keeps
the logged partners and drives the ego with the policy. Videos are mp4 files
written under reproduce/videos/.

    python reproduce/render.py
    python reproduce/render.py --scene 0 --policy regularized
    python reproduce/render.py --mode self_play --scene all
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator
from pufferlib.pufferl import load_env, load_policy


DATA_DIR = ROOT / "reproduce" / "data"
VIDEO_DIR = ROOT / "reproduce" / "videos"
ENV_NAME = "puffer_drive"
MODES = {
    "human_replay": "control_sdc_only",
    "self_play": "control_vehicles",
}


def load_eval_module():
    path = Path(__file__).with_name("eval.py")
    spec = importlib.util.spec_from_file_location("spiced_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_paths():
    paths = sorted(DATA_DIR.glob("map_*.bin"))
    if not paths:
        raise SystemExit(f"No maps in {DATA_DIR}. Expected map_000.bin, map_001.bin, map_002.bin.")
    return paths


def stage_one_map(bin_path: Path) -> tempfile.TemporaryDirectory:
    """The simulator loads map_000.bin .. map_{n-1}.bin from one directory."""
    tmp = tempfile.TemporaryDirectory(prefix="spiced-scene-")
    dest = Path(tmp.name) / "map_000.bin"
    dest.symlink_to(bin_path.resolve())
    return tmp


def newest_mp4(directory: Path, before: set[str]) -> Path | None:
    found = [p for p in directory.glob("*.mp4") if p.name not in before]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def snapshot_png(mp4_path: Path, png_path: Path) -> None:
    """Grab a mid-clip frame. The recorder writes one frame per simulator step at 30 fps."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4_path),
            "-vf",
            "select=eq(n\\,45)",
            "-frames:v",
            "1",
            str(png_path),
        ],
        check=True,
    )


def load_policy_on_scene(eval_mod, config, bin_path: Path):
    """Build the network against a bundled scene so the 10k split is not required."""
    staged = stage_one_map(bin_path)
    try:
        probe_cfg = eval_mod.make_eval_config(config, "control_sdc_only", num_maps=1, num_agents=1, map_dir=staged.name)
        env = load_env(ENV_NAME, probe_cfg)
        try:
            policy = load_policy(probe_cfg, env, ENV_NAME)
            policy.eval()
            return policy
        finally:
            env.close()
    finally:
        staged.cleanup()


def render_scene(eval_mod, policy, config, bin_path: Path, mode: str, out_dir: Path) -> Path:
    control_mode = MODES[mode]
    num_agents = 1 if mode == "human_replay" else 64
    staged = stage_one_map(bin_path)
    try:
        eval_config = eval_mod.make_eval_config(
            config, control_mode, num_maps=1, num_agents=num_agents, map_dir=staged.name
        )
        eval_config["env"]["episode_length"] = 91
        eval_config["env"]["render_mode"] = 1
        env = load_env(eval_mod.ENV_NAME, eval_config)
        try:
            before = {p.name for p in Path.cwd().glob("*.mp4")}
            evaluator = CheckpointEvaluator(eval_config)
            evaluator.rollout(env=env, policy=policy, render_env_idx=0, deterministic=True)
            env.driver_env.stop_recorder(0)
        finally:
            env.close()
    finally:
        staged.cleanup()

    src = newest_mp4(Path.cwd(), before)
    if src is None:
        raise SystemExit(f"No mp4 written for {bin_path.name}. ffmpeg is required for headless rendering.")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{bin_path.stem}_{mode}.mp4"
    shutil.move(src, dest)
    png = dest.with_suffix(".png")
    snapshot_png(dest, png)
    print(f"Saved {dest}")
    print(f"Saved {png}")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("regularized", "unregularized"), default="regularized")
    parser.add_argument("--mode", choices=tuple(MODES), default="human_replay")
    parser.add_argument("--scene", default="all", help="Scene index (0, 1, 2) or 'all'")
    parser.add_argument("--out-dir", type=Path, default=VIDEO_DIR)
    args = parser.parse_args()

    paths = scene_paths()
    if args.scene == "all":
        chosen = list(enumerate(paths))
    else:
        index = int(args.scene)
        if index < 0 or index >= len(paths):
            raise SystemExit(f"--scene must be 0..{len(paths) - 1} or all")
        chosen = [(index, paths[index])]

    eval_mod = load_eval_module()
    policy_path = dict(eval_mod.CHECKPOINTS)[args.policy]
    if not policy_path.exists():
        raise SystemExit(f"Missing checkpoint {policy_path}")

    print(f"Policy: {args.policy} ({policy_path.name})")
    print(f"Mode: {args.mode}")
    fallback = eval_mod.load_base_config()
    config = eval_mod.checkpoint_config(policy_path, fallback)
    policy = load_policy_on_scene(eval_mod, config, paths[0])

    for index, bin_path in chosen:
        print(f"Rendering scene {index}: {bin_path.name}", flush=True)
        render_scene(eval_mod, policy, config, bin_path, args.mode, args.out_dir)


if __name__ == "__main__":
    main()
