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
    F(int, max_lane_segment_observations)                                                                              \
    F(int, max_boundary_segment_observations)                                                                          \
    F(int, max_partner_observations)                                                                                   \
    F(int, max_traffic_control_observations)                                                                           \
    F(int, traffic_control_scope)                                                                                      \
    /* Observation normalization */                                                                                    \
    F(float, max_goal_position)                                                                                        \
    F(float, max_position)                                                                                             \
    F(float, max_veh_len)                                                                                              \
    F(float, max_veh_width)                                                                                            \
    F(float, max_road_segment_length)                                                                                  \
    F(float, max_road_segment_width)                                                                                   \
    /* Observation distances */                                                                                        \
    F(float, max_traffic_control_distance)                                                                             \
    F(float, agent_obs_max_dist)                                                                                       \
    F(float, road_obs_front_dist)                                                                                      \
    F(float, road_obs_behind_dist)                                                                                     \
    F(float, road_obs_side_dist)

static inline void apply_env_kwargs(Env *env, Dict *kwargs) {
#define APPLY(type, name) env->name = (type) dict_get(kwargs, #name)->value;
    ENV_FIELDS(APPLY)
#undef APPLY
}

#endif
