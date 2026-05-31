"""Render an agent's observation at multiple points along an episode so we can
see how the GIGAFLOW W_lane coarse view (lane slots) changes with position.

Saves per-frame PNGs and a single side-by-side grid PNG under
benchmark/coarse_view_renders/.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from pufferlib.ocean.drive.drive import Drive
from pufferlib.viz import plot_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="pufferlib/resources/drive/binaries/carla/opendrive__Town10HD.bin")
    parser.add_argument("--frames", type=int, default=8, help="number of obs renders along the episode")
    parser.add_argument("--steps-between", type=int, default=20, help="steps between renders")
    parser.add_argument("--agent-idx", type=int, default=0)
    parser.add_argument("--obs-slots-lane-n", type=int, default=80)
    parser.add_argument("--obs-slots-boundary-n", type=int, default=80)
    parser.add_argument("--obs-range-coarse-m", type=float, default=200.0)
    parser.add_argument("--obs-norm-coarse-dist-m", type=float, default=200.0)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="benchmark/coarse_view_renders")
    parser.add_argument("--dpi", type=int, default=200, help="figure DPI for the captured raster")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams["figure.dpi"] = args.dpi

    env = Drive(
        num_agents=args.num_agents,
        max_agents_per_env=args.num_agents,
        min_agents_per_env=args.num_agents,
        map_dir=args.map,
        num_maps=1,
        obs_slots_lane_n=args.obs_slots_lane_n,
        obs_slots_boundary_n=args.obs_slots_boundary_n,
        obs_range_coarse_m=args.obs_range_coarse_m,
        obs_norm_coarse_dist_m=args.obs_norm_coarse_dist_m,
        num_target_waypoints=1,
        simulation_mode="gigaflow",
        control_mode="control_vehicles",
        use_map_cache=1,
        scenario_length=max(400, args.frames * args.steps_between + 20),
        seed=args.seed,
    )
    env.reset()

    plot_kwargs = dict(
        target_type=env.target_type,
        reward_conditioning=env.reward_conditioning,
        num_target_waypoints=env.num_target_waypoints,
        obs_slots_partners_n=env.obs_slots_partners_n,
        obs_slots_lane_n=env.obs_slots_lane_n,
        obs_slots_boundary_n=env.obs_slots_boundary_n,
        obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
        obs_dropout_lane=env.obs_dropout_lane,
        obs_dropout_boundary=env.obs_dropout_boundary,
        agent_idx=args.agent_idx,
        obs_norm_goal_offset_m=env.obs_norm_goal_offset_m,
        obs_norm_xy_offset_m=env.obs_norm_xy_offset_m,
        obs_norm_veh_width_m=env.obs_norm_veh_width_m,
        obs_norm_veh_length_m=env.obs_norm_veh_length_m,
        obs_norm_road_seg_length_m=env.obs_norm_road_seg_length_m,
        obs_norm_road_seg_width_m=env.obs_norm_road_seg_width_m,
    )

    for f in range(args.frames):
        img = plot_observation(env.observations.copy(), **plot_kwargs)
        path = os.path.join(args.out_dir, f"obs_t{f * args.steps_between:04d}.png")
        plt.imsave(path, img)
        print(f"saved {path} ({img.shape[1]}x{img.shape[0]})")
        for _ in range(args.steps_between):
            env.step(np.zeros(env.actions.shape, dtype=env.actions.dtype))

    env.close()


if __name__ == "__main__":
    main()
