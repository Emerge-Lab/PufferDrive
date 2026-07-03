#!/usr/bin/env python3
"""Render a single-scenario video from a saved checkpoint.

Produces an mp4 of the policy driving on a chosen map using the
EGL headless render pipeline (render.h -> ffmpeg).

Usage (gigaflow — random agents on CARLA/WOMD road network):
    python scripts/render_scenario.py \\
        --checkpoint path/to/model.pt \\
        --map Town10HD \\
        --steps 1000 \\
        --num_agents 100 \\
        --view topdown_sim

Usage (replay — policy controls SDC, others follow logged trajectories):
    python scripts/render_scenario.py \\
        --checkpoint path/to/model.pt \\
        --map_dir pufferlib/resources/drive/binaries/sudden_brake_bins/ \\
        --simulation_mode replay \\
        --view bev \\
        --all_maps # Renders all maps in the map-dir

Views:
    sim_state    - fixed perspective camera with 3D car models (view_mode=0)
    bev          - ego-following ortho camera with wireframe boxes (view_mode=1)
    topdown_sim  - fixed ortho camera over full map with 3D car models (view_mode=2)
    bev_all      - ego-following top-down showing all agents (view_mode=3)
"""

import argparse
import os
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--map", default=None, help="Map name for auto-lookup (e.g. Town01, tfrecord-00021-of-00150_24)"
    )
    parser.add_argument("--output_dir", default="renders", help="Output directory for mp4s")
    parser.add_argument("--steps", type=int, default=None, help="Simulation steps (default: 1000 gigaflow, 91 replay)")
    parser.add_argument("--num_agents", type=int, default=None, help="Agents per scenario (gigaflow only, default 100)")
    parser.add_argument(
        "--view",
        default="topdown_sim",
        choices=["sim_state", "bev", "topdown_sim", "bev_all"],
        help="Camera view mode",
    )
    parser.add_argument(
        "--map_dir",
        default=None,
        help="Custom map directory (default: auto-find the map in binaries/)",
    )
    parser.add_argument(
        "--simulation_mode",
        default="gigaflow",
        choices=["gigaflow", "replay"],
        help="Simulation mode: gigaflow (random spawn) or replay (log trajectories, policy controls SDC)",
    )
    parser.add_argument(
        "--init_step", type=int, default=None, help="Timestep to start from (default: 0 gigaflow, 10 replay)"
    )
    parser.add_argument(
        "--control_mode",
        default=None,
        help="Override control mode (default: control_vehicles for gigaflow, control_sdc_only for replay)",
    )
    parser.add_argument(
        "--all_maps",
        action="store_true",
        help="Render one video per .bin file in map-dir (default: render one)",
    )
    cli = parser.parse_args()

    # Suppress argparse pollution from pufferl's load_config after our own parse
    sys.argv = ["render_scenario"]

    seed = int(time.time_ns() & 0xFFFFFFFF)

    # Defaults that depend on simulation mode
    if cli.simulation_mode == "replay":
        steps = cli.steps or 91
        num_agents = cli.num_agents or 1
        control_mode = cli.control_mode or "control_sdc_only"
        init_step = cli.init_step if cli.init_step is not None else 10
        eval_type = "human_replay"
    else:
        steps = cli.steps or 1000
        num_agents = cli.num_agents or 100
        control_mode = cli.control_mode or "control_vehicles"
        init_step = cli.init_step if cli.init_step is not None else 0
        eval_type = "multi_scenario"

    # Find the project root (parent of scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Resolve map directory
    if cli.map_dir:
        map_dir = os.path.abspath(cli.map_dir)
    elif cli.map:
        map_dir = tempfile.mkdtemp(prefix=f"render_{cli.map}_")
        binaries_root = os.path.join(project_root, "pufferlib/resources/drive/binaries")
        map_bin = None
        for search_dir in ["carla_py123d", "dense", "lateral", "longitudinal", "obstacles", "vru", "womd", "carla"]:
            candidate = os.path.join(binaries_root, search_dir, f"{cli.map}.bin")
            if os.path.exists(candidate):
                map_bin = candidate
                break
        if map_bin is None:
            print(f"Error: map binary '{cli.map}.bin' not found in any subdirectory of {binaries_root}")
            for d in sorted(os.listdir(binaries_root)):
                full = os.path.join(binaries_root, d)
                if os.path.isdir(full):
                    bins = [f for f in os.listdir(full) if f.endswith(".bin")]
                    if bins:
                        print(f"  {d}/: {', '.join(sorted(bins)[:5])}{'...' if len(bins) > 5 else ''}")
            sys.exit(1)
        dst = os.path.join(map_dir, f"{cli.map}.bin")
        if not os.path.exists(dst):
            os.symlink(map_bin, dst)
    else:
        print("Error: provide --map_dir or --map")
        sys.exit(1)

    os.makedirs(cli.output_dir, exist_ok=True)

    # Count .bin files now so num_maps and num_agents can be set in env_overrides.
    # --map creates a temp dir with exactly 1 .bin; --map_dir may have many.
    num_bin_files = len([f for f in os.listdir(map_dir) if f.endswith(".bin")])
    if num_bin_files == 0:
        print(f"Error: no .bin files found in {map_dir}")
        sys.exit(1)

    if cli.all_maps:
        num_scenarios = num_bin_files
        print(f"[render_scenario] --all_maps: rendering {num_scenarios} scenarios")
    else:
        num_scenarios = 1

    from pufferlib.pufferl import eval as puffer_eval, load_config

    env_name = "puffer_drive"
    args = load_config(env_name)
    args["load_model_path"] = cli.checkpoint
    args["eval_results_dir"] = cli.output_dir

    env_overrides = {
        "map_dir": map_dir,
        "num_maps": num_bin_files,
        "num_agents": num_agents * num_scenarios,
        "seed": seed,
        "simulation_mode": cli.simulation_mode,
        "control_mode": control_mode,
        "init_step": init_step,
        "partner_blindness_prob": 0.0,
        "phantom_braking_prob": 0.0,
        "phantom_braking_trigger_prob": 0.0,
        "obs_dropout_lane": 0.0,
        "obs_dropout_boundary": 0.0,
    }
    if cli.simulation_mode == "gigaflow":
        env_overrides["min_agents_per_env"] = num_agents
        env_overrides["max_agents_per_env"] = num_agents
    elif cli.simulation_mode == "replay":
        # Don't stop the SDC for offroad/collision so the full trajectory renders
        env_overrides["offroad_behavior"] = 0
        env_overrides["collision_behavior"] = 0
        env_overrides["scenario_length"] = steps
        env_overrides["resample_frequency"] = steps

    eval_section = {
        "type": eval_type,
        "render": True,
        "render_views": [cli.view],
        "render_backend": "egl",
        "mode": "inline",
        "enabled": True,
        "interval": 0,
        "env": env_overrides,
        "eval": {
            "render_num_scenarios": num_scenarios,
            "render_max_steps": steps,
            "num_scenarios": num_scenarios,
        },
    }

    args.setdefault("eval", {})
    args["eval"]["render_cli"] = eval_section

    map_label = os.path.basename(map_dir.rstrip("/"))
    mode_desc = cli.simulation_mode
    if cli.simulation_mode == "replay":
        mode_desc += f" (control_mode={control_mode})"
    print(f"Rendering {map_label} | mode={mode_desc} | {steps} steps | view={cli.view}")
    print(f"Checkpoint: {cli.checkpoint}")
    print(f"Output: {cli.output_dir}/mp4/render_cli/")

    puffer_eval(env_name=env_name, args=args, evaluator_name="render_cli")

    mp4_dir = os.path.join(cli.output_dir, "mp4", "render_cli")
    mp4s = sorted(f for f in os.listdir(mp4_dir) if f.endswith(".mp4")) if os.path.isdir(mp4_dir) else []
    if mp4s:
        print(f"\nDone. Output: {os.path.join(mp4_dir, mp4s[-1])}")
    else:
        print(f"\nDone. Check {mp4_dir} for output.")


if __name__ == "__main__":
    main()
