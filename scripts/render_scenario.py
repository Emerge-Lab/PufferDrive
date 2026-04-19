#!/usr/bin/env python3
"""Render a single-scenario video from a saved checkpoint.

Produces an mp4 of the policy driving on a chosen map using the
EGL headless render pipeline (render.h -> ffmpeg).

Usage (gigaflow — random agents on CARLA/WOMD road network):
    python scripts/render_scenario.py \\
        --checkpoint path/to/model.pt \\
        --map Town10HD \\
        --steps 1000 \\
        --num-agents 100 \\
        --view topdown_sim

Usage (replay — policy controls SDC, others follow logged trajectories):
    python scripts/render_scenario.py \\
        --checkpoint path/to/model.pt \\
        --map tfrecord-00021-of-00150_24 \\
        --simulation-mode replay \\
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--map", default="Town10HD", help="Map name (e.g. Town01, tfrecord-00021-of-00150_24)")
    parser.add_argument("--output-dir", default="renders", help="Output directory for mp4s")
    parser.add_argument("--steps", type=int, default=None, help="Simulation steps (default: 1000 gigaflow, 91 replay)")
    parser.add_argument("--num-agents", type=int, default=None, help="Agents per scenario (gigaflow only, default 100)")
    parser.add_argument(
        "--view",
        default="topdown_sim",
        choices=["sim_state", "bev", "topdown_sim", "bev_all"],
        help="Camera view mode (bev_all = ego-following top-down showing all agents)",
    )
    parser.add_argument("--num-eval-agents", type=int, default=512, help="Total agent budget for eval vecenv")
    parser.add_argument(
        "--map-dir",
        default=None,
        help="Custom map directory (default: auto-find the map in binaries/)",
    )
    parser.add_argument(
        "--simulation-mode",
        default="gigaflow",
        choices=["gigaflow", "replay"],
        help="Simulation mode: gigaflow (random spawn) or replay (log trajectories, policy controls SDC)",
    )
    parser.add_argument(
        "--init-steps", type=int, default=None, help="Timestep to start from (default: 0 gigaflow, 10 replay)"
    )
    parser.add_argument(
        "--control-mode",
        default=None,
        help="Override control mode (default: control_vehicles for gigaflow, control_sdc_only for replay)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip video output and run eval_multi_scenarios to aggregate metrics across all bins in --map-dir. Ignores --map when set.",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help="Number of scenarios to evaluate in --no-render mode (defaults to every bin in --map-dir)",
    )
    cli = parser.parse_args()

    # Suppress argparse pollution from pufferl's load_config after our own parse
    sys.argv = ["render_scenario"]

    # Defaults that depend on simulation mode
    if cli.simulation_mode == "replay":
        steps = cli.steps or 91
        num_agents = cli.num_agents or 1
        control_mode = cli.control_mode or "control_sdc_only"
        init_steps = cli.init_steps if cli.init_steps is not None else 10
    else:
        steps = cli.steps or 1000
        num_agents = cli.num_agents or 100
        control_mode = cli.control_mode or "control_vehicles"
        init_steps = cli.init_steps if cli.init_steps is not None else 0

    view_mode_map = {"sim_state": 0, "bev": 1, "topdown_sim": 2, "bev_all": 3}
    view_mode = view_mode_map[cli.view]

    # Find the project root (parent of scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Set up a single-map directory if not provided
    if cli.map_dir:
        map_dir = os.path.abspath(cli.map_dir)
    elif cli.no_render:
        # No-render mode requires --map-dir explicitly (we aggregate across the
        # whole directory rather than symlink a single bin).
        print("Error: --no-render requires --map-dir pointing at a directory of .bin files.")
        sys.exit(1)
    else:
        map_dir = tempfile.mkdtemp(prefix=f"render_{cli.map}_")
        # Search for the map .bin in known directories
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

    os.makedirs(cli.output_dir, exist_ok=True)

    from pufferlib.pufferl import (
        build_eval_overrides,
        eval_multi_scenarios,
        eval_multi_scenarios_render,
        load_config,
        load_eval_multi_scenarios_config,
    )

    env_name = "puffer_drive"
    tmp_args = load_config(env_name)

    if cli.no_render:
        num_scenarios = cli.num_scenarios or len([f for f in os.listdir(map_dir) if f.endswith(".bin")])
    else:
        num_scenarios = 1

    eval_overrides = build_eval_overrides(
        simulation_mode=cli.simulation_mode,
        num_agents=cli.num_eval_agents,
        num_scenarios=num_scenarios,
        map_dir=map_dir,
        num_carla_maps=num_scenarios if cli.no_render else 1,
        clean=True,
    )
    # Override control_mode and init_steps after build_eval_overrides
    eval_overrides["env"]["control_mode"] = control_mode
    eval_overrides["env"]["init_steps"] = init_steps
    if cli.simulation_mode == "replay":
        # For no-render metrics we want offroad/collision to TERMINATE so
        # the SDC is penalized per the normal eval rules. For rendering we
        # let the SDC keep driving so the video shows the full trajectory
        # even when the policy is far off.
        if not cli.no_render:
            eval_overrides["env"]["offroad_behavior"] = 0
            eval_overrides["env"]["collision_behavior"] = 0
        # Match scenario_length to requested steps so the render loop
        # doesn't cap at the default 91 from build_eval_overrides.
        eval_overrides["env"]["scenario_length"] = steps
        eval_overrides["env"]["resample_frequency"] = steps

    args = load_eval_multi_scenarios_config(env_name, cli.checkpoint, eval_overrides)
    args["load_model_path"] = cli.checkpoint
    args["num_scenarios"] = num_scenarios
    args["num_carla_maps"] = num_scenarios if cli.no_render else 1
    args["eval_simulation"] = cli.simulation_mode
    args["inline_eval"] = True
    args["eval_results_dir"] = cli.output_dir

    if cli.simulation_mode == "gigaflow":
        args["env"]["min_agents_per_env"] = num_agents
        args["env"]["max_agents_per_env"] = num_agents
    elif cli.simulation_mode == "replay":
        pass  # replay uses agents from .bin data as-is

    mode_desc = cli.simulation_mode
    if cli.simulation_mode == "replay":
        mode_desc += f" (control_mode={control_mode})"

    if cli.no_render:
        # Cap vec workers at physical CPU count — pufferlib rejects
        # num_workers > cores (emerge2 has 16 physical cores, ini default 20).
        import psutil

        cpu_cores = psutil.cpu_count(logical=False) or 8
        cap = min(cpu_cores, num_scenarios)
        args["vec"]["num_envs"] = cap
        args["vec"]["num_workers"] = cap
        args["vec"]["batch_size"] = cap
        print(f"No-render eval | mode={mode_desc} | {steps} steps | {num_scenarios} scenarios | {cap} workers")
        print(f"Map dir: {map_dir}")
        print(f"Checkpoint: {cli.checkpoint}")
        print(f"Output: {cli.output_dir}/")
        eval_multi_scenarios(
            env_name=env_name,
            args=dict(args),
            vecenv=None,
            policy=None,
            logger=None,
            metric_prefix="eval",
            quiet=False,
            clean=True,
        )
        print(f"\nDone. Summary CSV: {cli.output_dir}/evaluation_summary.csv")
        return

    args["render"] = 1
    args["render_obs"] = 0
    print(f"Rendering {cli.map} | mode={mode_desc} | {steps} steps | view={cli.view}")
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
        render_max_steps=steps,
    )

    mp4_dir = os.path.join(cli.output_dir, "mp4")
    mp4s = [f for f in os.listdir(mp4_dir) if f.endswith(".mp4")] if os.path.isdir(mp4_dir) else []
    if mp4s:
        print(f"\nDone. Output: {os.path.join(mp4_dir, mp4s[0])}")
    else:
        print(f"\nDone. Check {mp4_dir} for output.")


if __name__ == "__main__":
    main()
