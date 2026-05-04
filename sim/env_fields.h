#ifndef SIM_ENV_FIELDS_H
#define SIM_ENV_FIELDS_H

// Single source of truth for Env fields wired from drive.ini [env] kwargs.
// Add a line here + a default in config/drive.ini to expose a new tunable.

#define ENV_FIELDS(F)                                                                                                  \
    /* Sim mode / control */                                                                                           \
    F(int, simulation_mode)                                                                                            \
    F(int, control_mode)                                                                                               \
    F(int, eval_mode)                                                                                                  \
    F(int, min_agents_per_env)                                                                                         \
    F(int, max_agents_per_env)                                                                                         \
    F(int, replay_expert_actions)                                                                                      \
    F(int, action_type)                                                                                                \
    F(int, dynamics_model)                                                                                             \
    F(float, dt)                                                                                                       \
    F(int, init_step)                                                                                                  \
    F(int, episode_length)                                                                                             \
    F(int, collision_behavior)                                                                                         \
    F(int, offroad_behavior)                                                                                           \
    F(int, red_light_behavior)                                                                                         \
    F(int, termination_mode)                                                                                           \
    F(float, inactive_agent_threshold)                                                                                 \
    /* Goal / target */                                                                                                \
    F(int, target_type)                                                                                                \
    F(float, goal_radius)                                                                                              \
    F(float, goal_speed)                                                                                               \
    F(int, num_target_waypoints)                                                                                       \
    F(float, min_goal_spacing)                                                                                         \
    F(float, max_goal_spacing)                                                                                         \
    F(float, path_spacing)                                                                                             \
    F(float, spawn_initial_speed)                                                                                      \
    /* Reward coefficients */                                                                                          \
    F(float, reward_goal)                                                                                              \
    F(float, reward_collision)                                                                                         \
    F(float, reward_offroad)                                                                                           \
    F(float, reward_stop_line)                                                                                         \
    F(float, reward_comfort)                                                                                           \
    F(float, reward_lane_align)                                                                                        \
    F(float, reward_vel_align)                                                                                         \
    F(float, reward_lane_center)                                                                                       \
    F(float, reward_center_bias)                                                                                       \
    F(float, reward_velocity)                                                                                          \
    F(float, reward_reverse)                                                                                           \
    F(float, reward_timestep)                                                                                          \
    F(float, reward_overspeed)                                                                                         \
    /* Observation limits */                                                                                           \
    F(int, obs_slots_lane)                                                                                             \
    F(int, obs_slots_boundary)                                                                                         \
    F(int, obs_slots_partners)                                                                                         \
    F(int, obs_slots_traffic_controls)                                                                                 \
    F(int, traffic_control_scope)                                                                                      \
    /* Observation normalization */                                                                                    \
    F(float, norm_goal_offset_m)                                                                                       \
    F(float, norm_xy_offset_m)                                                                                         \
    F(float, norm_vehicle_length_m)                                                                                    \
    F(float, norm_vehicle_width_m)                                                                                     \
    F(float, norm_road_segment_length_m)                                                                               \
    F(float, norm_road_segment_width_m)                                                                                \
    /* Observation distances */                                                                                        \
    F(float, obs_range_road_front_m)                                                                                   \
    F(float, obs_range_road_behind_m)                                                                                  \
    F(float, obs_range_road_side_m)                                                                                    \
    F(float, obs_range_partner_m)                                                                                      \
    F(float, obs_range_traffic_control_m)                                                                              \
    /* Robustness features */                                                                                          \
    F(float, partner_blindness_prob)

#endif
