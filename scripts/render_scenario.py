#!/usr/bin/env python3
"""Render a single-scenario video from a saved checkpoint.

Produces an mp4 of the policy driving on a chosen CARLA town using the
EGL headless render pipeline (render.h -> ffmpeg).

Usage:
    python scripts/render_scenario.py \\
        --checkpoint path/to/model.pt \\
        --map Town10HD \\
        --output-dir renders/ \\
        --steps 1000 \\
        --num-agents 100 \\
        --view topdown_sim

Views:
    sim_state    - fixed perspective camera with 3D car models (view_mode=0)
    bev          - ego-following ortho camera with wireframe boxes (view_mode=1)
    topdown_sim  - fixed ortho camera over full map with 3D car models (view_mode=2)
"""
import argparse
import os
import sys
import tempfile

# Suppress argparse pollution from pufferl's load_config
sys.argv = ["render_scenario"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--map", default="Town10HD", help="CARLA town name (e.g. Town01, Town10HD)")
    parser.add_argument("--output-dir", default="renders", help="Output directory for mp4s")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps to render")
    parser.add_argument("--num-agents", type=int, default=100, help="Agents per scenario")
    parser.add_argument(
        "--view",
        default="topdown_sim",
        choices=["sim_state", "bev", "topdown_sim"],
        help="Camera view mode",
    )
    parser.add_argument("--num-eval-agents", type=int, default=512, help="Total agent budget for eval vecenv")
    parser.add_argument(
        "--map-dir",
        default=None,
        help="Custom map directory (default: auto-create a temp dir with the chosen map)",
    )
    cli = parser.parse_args()

    view_mode_map = {"sim_state": 0, "bev": 1, "topdown_sim": 2}
    view_mode = view_mode_map[cli.view]

    # Find the project root (parent of scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Set up a single-map directory if not provided
    if cli.map_dir:
        map_dir = cli.map_dir
    else:
        map_dir = tempfile.mkdtemp(prefix=f"render_{cli.map}_")
        map_bin = os.path.join(project_root, "pufferlib/resources/drive/binaries/carla_py123d", f"{cli.map}.bin")
        if not os.path.exists(map_bin):
            print(f"Error: map binary not found at {map_bin}")
            print("Available maps:", os.listdir(os.path.join(project_root, "pufferlib/resources/drive/binaries/carla_py123d")))
            sys.exit(1)
        dst = os.path.join(map_dir, f"{cli.map}.bin")
        if not os.path.exists(dst):
            os.symlink(map_bin, dst)

    os.makedirs(cli.output_dir, exist_ok=True)

    from pufferlib.pufferl import (
        build_eval_overrides,
        eval_multi_scenarios_render,
        load_config,
        load_eval_multi_scenarios_config,
    )

    env_name = "puffer_drive"
    tmp_args = load_config(env_name)

    eval_overrides = build_eval_overrides(
        simulation_mode="gigaflow",
        num_agents=cli.num_eval_agents,
        num_scenarios=1,
        map_dir=map_dir,
        num_carla_maps=1,
    )
    args = load_eval_multi_scenarios_config(env_name, cli.checkpoint, eval_overrides)
    args["env"]["min_agents_per_env"] = cli.num_agents
    args["env"]["max_agents_per_env"] = cli.num_agents
    args["load_model_path"] = cli.checkpoint
    args["num_scenarios"] = 1
    args["num_carla_maps"] = 1
    args["eval_simulation"] = "gigaflow"
    args["render"] = 1
    args["render_obs"] = 0
    args["inline_eval"] = True
    args["eval_results_dir"] = cli.output_dir

    print(f"Rendering {cli.map} with {cli.num_agents} agents, {cli.steps} steps, view={cli.view}")
    print(f"Checkpoint: {cli.checkpoint}")
    print(f"Output: {cli.output_dir}/mp4/")

    eval_multi_scenarios_render(
        env_name=env_name,
        args=dict(args),
        vecenv=None,
        policy=None,
        logger=None,
        metric_prefix=f"render_{cli.map}",
        quiet=False,
        render_backend="egl",
        view_mode=view_mode,
        video_suffix="",
        log_view_label=cli.view,
        render_max_steps=cli.steps,
    )

    mp4_dir = os.path.join(cli.output_dir, "mp4")
    mp4s = [f for f in os.listdir(mp4_dir) if f.endswith(".mp4")] if os.path.isdir(mp4_dir) else []
    if mp4s:
        print(f"\nDone. Output: {os.path.join(mp4_dir, mp4s[0])}")
    else:
        print(f"\nDone. Check {mp4_dir} for output.")


if __name__ == "__main__":
    main()
