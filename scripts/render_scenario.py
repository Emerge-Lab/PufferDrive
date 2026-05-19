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
        --map-dir pufferlib/resources/drive/binaries/sudden_brake_bins/ \\
        --simulation-mode replay \\
        --view bev \\
        --all-maps # Renders all maps in the map-dir

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


def _patch_policy_args_from_checkpoint(args, checkpoint_path):
    """Infer policy architecture dims from checkpoint weights and patch args["policy"].

    Checkpoints carry no config metadata, so we reverse-engineer the dims:
      - input_size        — ego_encoder.0.weight output dim (shape[0])
      - backbone_hidden_size — backbone.1.weight output dim (shape[0])
      - split_network     — True if critic_backbone weights exist separately

    This lets render_scenario work with checkpoints trained under any config
    without the user needing to know or pass the architecture flags.
    """
    import torch

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Strip DDP wrapper prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # input_size: output dim of the first sub-encoder linear layer
    ego_key = "actor_backbone.ego_encoder.0.weight"
    if ego_key in state_dict:
        inferred_input_size = state_dict[ego_key].shape[0]
        if inferred_input_size != args["policy"]["input_size"]:
            print(
                f"[render_scenario] checkpoint input_size={inferred_input_size} "
                f"(config default was {args['policy']['input_size']}); patching."
            )
            args["policy"]["input_size"] = inferred_input_size

    # backbone_hidden_size: output dim of the first backbone hidden layer
    bb_key = "actor_backbone.backbone.1.weight"
    if bb_key in state_dict:
        inferred_bb_hidden = state_dict[bb_key].shape[0]
        if inferred_bb_hidden != args["policy"]["backbone_hidden_size"]:
            print(
                f"[render_scenario] checkpoint backbone_hidden_size={inferred_bb_hidden} "
                f"(config default was {args['policy']['backbone_hidden_size']}); patching."
            )
            args["policy"]["backbone_hidden_size"] = inferred_bb_hidden

    # split_network: critic has its own weights only when split
    has_split = any(k.startswith("critic_backbone.ego_encoder") for k in state_dict)
    if has_split != args["policy"].get("split_network", False):
        print(
            f"[render_scenario] checkpoint split_network={has_split} "
            f"(config default was {args['policy'].get('split_network', False)}); patching."
        )
        args["policy"]["split_network"] = has_split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--map", default=None, help="Map name for auto-lookup (e.g. Town01, tfrecord-00021-of-00150_24)")
    parser.add_argument("--output-dir", default="renders", help="Output directory for mp4s")
    parser.add_argument("--steps", type=int, default=None, help="Simulation steps (default: 1000 gigaflow, 91 replay)")
    parser.add_argument("--num-agents", type=int, default=None, help="Agents per scenario (gigaflow only, default 100)")
    parser.add_argument(
        "--view",
        default="topdown_sim",
        choices=["sim_state", "bev", "topdown_sim", "bev_all"],
        help="Camera view mode",
    )
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
        "--all-maps",
        action="store_true",
        help="Render one video per .bin file in map-dir (default: render one)",
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
        eval_type = "human_replay"
    else:
        steps = cli.steps or 1000
        num_agents = cli.num_agents or 100
        control_mode = cli.control_mode or "control_vehicles"
        init_steps = cli.init_steps if cli.init_steps is not None else 0
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
        print("Error: provide --map-dir or --map")
        sys.exit(1)

    os.makedirs(cli.output_dir, exist_ok=True)

    from pufferlib.pufferl import eval as puffer_eval, load_config

    env_name = "puffer_drive"
    args = load_config(env_name)
    args["load_model_path"] = cli.checkpoint
    args["eval_results_dir"] = cli.output_dir

    _patch_policy_args_from_checkpoint(args, cli.checkpoint)

    env_overrides = {
        "map_dir": map_dir,
        "simulation_mode": cli.simulation_mode,
        "control_mode": control_mode,
        "init_steps": init_steps,
        # Clean eval — no robustness perturbations
        "partner_blindness_prob": 0.0,
        "phantom_braking_prob": 0.0,
        "phantom_braking_trigger_prob": 0.0,
        "lane_segment_dropout": 0.0,
        "boundary_segment_dropout": 0.0,
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

    if cli.all_maps:
        num_scenarios = len([f for f in os.listdir(map_dir) if f.endswith(".bin")])
        if num_scenarios == 0:
            print(f"Error: no .bin files found in {map_dir}")
            sys.exit(1)
        print(f"[render_scenario] --all-maps: rendering {num_scenarios} scenarios")
    else:
        num_scenarios = 1

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
