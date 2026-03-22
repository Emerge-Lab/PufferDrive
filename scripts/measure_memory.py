"""Measure memory usage during env initialization step by step.

Usage:
    python scripts/measure_memory.py [map_dir] [num_maps] [num_agents]

Defaults: resources/drive/binaries/carla_data, 2 maps, 1024 agents
"""
import os
import sys
import resource


def get_rss_mb():
    """Get current RSS in MB (ru_maxrss is in KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def print_mem(label, baseline=[None]):
    rss = get_rss_mb()
    if baseline[0] is None:
        baseline[0] = rss
    delta = rss - baseline[0]
    print(f"  [{rss:8.0f} MB RSS, +{delta:8.0f} MB] {label}")
    sys.stdout.flush()
    return rss


print_mem("Python startup")

import numpy as np
print_mem("After numpy import")

# Use pufferlib's own env creator so we match real init exactly
import pufferlib.ocean as ocean
print_mem("After pufferlib.ocean import")

map_dir = sys.argv[1] if len(sys.argv) > 1 else "resources/drive/binaries/carla_data"
num_maps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
num_agents = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

# List map files and sizes
print(f"\nConfig: map_dir={map_dir}, num_maps={num_maps}, num_agents={num_agents}")
for i in range(num_maps):
    path = os.path.join(map_dir, f"map_{i:03d}.bin")
    if os.path.exists(path):
        print(f"  {path}: {os.path.getsize(path)} bytes")
    else:
        print(f"  {path}: MISSING")

# Create a single env to measure per-env cost
print(f"\n--- Creating single env (num_agents={num_agents}, num_maps={num_maps}) ---")
make_env = ocean.env_creator("puffer_drive")

before = get_rss_mb()
env = make_env(
    num_agents=num_agents,
    num_maps=num_maps,
    map_dir=map_dir,
    init_mode="init_variable_agent_number",
    min_agents_per_env=1,
    max_agents_per_env=128,
    observation_window_size=100.0,
    polyline_reduction_threshold=1.0,
    polyline_max_segment_length=10.0,
    episode_length=300,
    dt=0.1,
    action_type="discrete",
    dynamics_model="jerk",
    reward_vehicle_collision=-0.5,
    reward_offroad_collision=-0.5,
    reward_lane_align=1.0,
    reward_lane_center=1.0,
    reward_goal=1.0,
    reward_goal_post_respawn=0.25,
    goal_radius=2.0,
    min_goal_speed=-0.01,
    max_goal_speed=10.0,
    goal_behavior=1,
    min_goal_distance=0.5,
    max_goal_distance=60.0,
    collision_behavior=0,
    offroad_behavior=0,
    termination_mode=1,
    resample_frequency=300,
    control_mode="control_vehicles",
    init_steps=0,
    allow_fewer_maps=True,
    spawn_width_min=1.5,
    spawn_width_max=2.5,
    spawn_length_min=2.0,
    spawn_length_max=5.5,
    spawn_height=1.5,
    reward_randomization=1,
    reward_conditioning=1,
    turn_off_normalization=1,
    reward_bound_goal_radius_min=2.0,
    reward_bound_goal_radius_max=12.0,
    reward_bound_collision_min=-3.0,
    reward_bound_collision_max=-0.1,
    reward_bound_offroad_min=-3.0,
    reward_bound_offroad_max=-0.1,
    reward_bound_comfort_min=-0.1,
    reward_bound_comfort_max=0.0,
    reward_bound_lane_align_min=0.002,
    reward_bound_lane_align_max=0.0025,
    reward_bound_lane_center_min=-0.00075,
    reward_bound_lane_center_max=-0.00065,
    reward_bound_velocity_min=0.0,
    reward_bound_velocity_max=0.005,
    reward_bound_traffic_light_min=-1.0,
    reward_bound_traffic_light_max=0.0,
    reward_bound_center_bias_min=-0.1,
    reward_bound_center_bias_max=0.1,
    reward_bound_vel_align_min=0.0,
    reward_bound_vel_align_max=1.0,
    reward_bound_overspeed_min=-1.0,
    reward_bound_overspeed_max=-0.9,
    reward_bound_timestep_min=-0.00005,
    reward_bound_timestep_max=0.0,
    reward_bound_reverse_min=-0.0075,
    reward_bound_reverse_max=-0.00025,
    reward_bound_throttle_min=0.8,
    reward_bound_throttle_max=1.25,
    reward_bound_steer_min=0.8,
    reward_bound_steer_max=1.25,
    reward_bound_acc_min=0.666,
    reward_bound_acc_max=1.5,
    min_avg_speed_to_consider_goal_attempt=2.0,
)
after = get_rss_mb()
print_mem(f"After single env creation (+{after - before:.0f} MB for this env)")

# Now compare with carla_3D
print(f"\n--- For comparison, creating single env with carla_3D ---")
before2 = get_rss_mb()
env2 = make_env(
    num_agents=num_agents,
    num_maps=3,
    map_dir="resources/drive/binaries/carla_3D",
    init_mode="init_variable_agent_number",
    min_agents_per_env=1,
    max_agents_per_env=128,
    observation_window_size=100.0,
    polyline_reduction_threshold=1.0,
    polyline_max_segment_length=10.0,
    episode_length=300,
    dt=0.1,
    action_type="discrete",
    dynamics_model="jerk",
    reward_vehicle_collision=-0.5,
    reward_offroad_collision=-0.5,
    reward_lane_align=1.0,
    reward_lane_center=1.0,
    reward_goal=1.0,
    reward_goal_post_respawn=0.25,
    goal_radius=2.0,
    min_goal_speed=-0.01,
    max_goal_speed=10.0,
    goal_behavior=1,
    min_goal_distance=0.5,
    max_goal_distance=60.0,
    collision_behavior=0,
    offroad_behavior=0,
    termination_mode=1,
    resample_frequency=300,
    control_mode="control_vehicles",
    init_steps=0,
    allow_fewer_maps=True,
    spawn_width_min=1.5,
    spawn_width_max=2.5,
    spawn_length_min=2.0,
    spawn_length_max=5.5,
    spawn_height=1.5,
    reward_randomization=1,
    reward_conditioning=1,
    turn_off_normalization=1,
    reward_bound_goal_radius_min=2.0,
    reward_bound_goal_radius_max=12.0,
    reward_bound_collision_min=-3.0,
    reward_bound_collision_max=-0.1,
    reward_bound_offroad_min=-3.0,
    reward_bound_offroad_max=-0.1,
    reward_bound_comfort_min=-0.1,
    reward_bound_comfort_max=0.0,
    reward_bound_lane_align_min=0.002,
    reward_bound_lane_align_max=0.0025,
    reward_bound_lane_center_min=-0.00075,
    reward_bound_lane_center_max=-0.00065,
    reward_bound_velocity_min=0.0,
    reward_bound_velocity_max=0.005,
    reward_bound_traffic_light_min=-1.0,
    reward_bound_traffic_light_max=0.0,
    reward_bound_center_bias_min=-0.1,
    reward_bound_center_bias_max=0.1,
    reward_bound_vel_align_min=0.0,
    reward_bound_vel_align_max=1.0,
    reward_bound_overspeed_min=-1.0,
    reward_bound_overspeed_max=-0.9,
    reward_bound_timestep_min=-0.00005,
    reward_bound_timestep_max=0.0,
    reward_bound_reverse_min=-0.0075,
    reward_bound_reverse_max=-0.00025,
    reward_bound_throttle_min=0.8,
    reward_bound_throttle_max=1.25,
    reward_bound_steer_min=0.8,
    reward_bound_steer_max=1.25,
    reward_bound_acc_min=0.666,
    reward_bound_acc_max=1.5,
    min_avg_speed_to_consider_goal_attempt=2.0,
)
after2 = get_rss_mb()
print_mem(f"After carla_3D env creation (+{after2 - before2:.0f} MB for this env)")

print("\nDone.")
