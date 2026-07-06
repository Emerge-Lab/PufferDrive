#ifndef DRIVE_FIXTURE_H
#define DRIVE_FIXTURE_H

#include "pufferlib/ocean/drive/drive.h"

#ifndef DRIVE_TEST_REPO_ROOT
#define DRIVE_TEST_REPO_ROOT "."
#endif

static inline const char *drive_carla_map(void) {
    return DRIVE_TEST_REPO_ROOT "/pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin";
}

static inline const char *drive_nuplan_map(void) {
    return DRIVE_TEST_REPO_ROOT
        "/pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin";
}

static inline char *drive_test_strdup(const char *src) {
    size_t n = strlen(src) + 1;
    char *dst = (char *) malloc(n);
    memcpy(dst, src, n);
    return dst;
}

static inline Agent drive_test_agent(float x, float y, float heading) {
    Agent agent = {0};
    agent.type = VEHICLE;
    agent.sim_x = x;
    agent.sim_y = y;
    agent.sim_z = 0.0f;
    agent.sim_heading = heading;
    agent.cos_heading = cosf(heading);
    agent.sin_heading = sinf(heading);
    agent.sim_length = 4.0f;
    agent.sim_width = 2.0f;
    agent.sim_height = 1.5f;
    agent.sim_valid = 1;
    agent.current_lane_idx = 1;
    agent.previous_lane_idx = 1;
    agent.wheelbase = 2.7f;
    copy_pose_to_prev(&agent);
    update_agent_radius(&agent);
    return agent;
}

static inline Drive drive_test_env_config(
    const char *map_file,
    int simulation_mode,
    int num_agents,
    int use_map_cache) {
    Drive env = {0};
    env.render_mode = RENDER_WINDOW;
    env.action_type = 0;
    env.dynamics_model = CLASSIC;
    env.reward_goal = 1.0f;
    env.reward_collision = 3.0f;
    env.reward_offroad = 3.0f;
    env.reward_comfort = 0.05f;
    env.reward_lane_align = 0.025f;
    env.reward_vel_align = 1.0f;
    env.reward_lane_center = 0.0038f;
    env.reward_center_bias = 0.0f;
    env.reward_velocity = 0.0025f;
    env.reward_reverse = 0.005f;
    env.reward_stop_line = 1.0f;
    env.reward_timestep = 0.000025f;
    env.reward_overspeed = 0.05f;
    env.reward_ade = 0.0f;
    env.collision_behavior = 0;
    env.offroad_behavior = 0;
    env.traffic_light_behavior = 0;
    env.use_map_cache = use_map_cache;
    env.emit_completed_episodes = 1;
    env.goal_radius = 2.0f;
    env.goal_speed = 3.0f;
    env.min_goal_spacing = 20.0f;
    env.max_goal_spacing = 60.0f;
    env.num_goals = 3;
    env.target_type = TARGET_STATIC;
    env.goal_on_lane = 1;
    env.obs_slots_lane_n = 32;
    env.obs_slots_boundary_n = 32;
    env.obs_slots_lane_kept = 32;
    env.obs_slots_boundary_kept = 32;
    env.obs_slots_partners_n = 16;
    env.obs_slots_traffic_controls_n = 4;
    env.traffic_control_scope = TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS;
    env.dt = 0.1f;
    env.spawn_initial_speed = 0.0f;
    env.scenario_length = 91;
    env.termination_mode = 0;
    env.inactive_agent_threshold = 0.4f;
    env.map_name = drive_test_strdup(map_file);
    env.num_controllable_agents = num_agents;
    env.num_max_agents = 64;
    env.init_step = 0;
    env.timestep = 0;
    env.init_mode = INIT_ALL_VALID;
    env.control_mode = simulation_mode == SIMULATION_REPLAY ? CONTROL_SDC_ONLY : CONTROL_VEHICLES;
    env.sdc_controller = CONTROLLER_POLICY;
    env.non_sdc_controller = CONTROLLER_POLICY;
    env.non_vehicle_controller = CONTROLLER_REPLAY;
    env.simulation_mode = simulation_mode;
    env.reward_conditioning = 0;
    env.reward_randomization = 0;
    env.compute_eval_metrics = 1;
    env.eval_mode = 0;
    env.obs_norm_goal_offset_m = 100.0f;
    env.obs_norm_xy_offset_m = 100.0f;
    env.obs_norm_veh_length_m = 15.0f;
    env.obs_norm_veh_width_m = 10.0f;
    env.obs_norm_road_seg_length_m = 5.0f;
    env.obs_norm_road_seg_width_m = 5.0f;
    env.obs_range_traffic_control_m = 100.0f;
    env.obs_range_partner_m = 100.0f;
    env.obs_range_road_front_m = 120.0f;
    env.obs_range_road_behind_m = 20.0f;
    env.obs_range_road_side_m = 30.0f;
    env.partner_blindness_prob = 0.0f;
    env.partner_blindness_trigger_prob = 0.1f;
    env.phantom_braking_prob = 0.0f;
    env.phantom_braking_trigger_prob = 0.0f;
    env.phantom_braking_duration = 10;
    return env;
}

static inline Drive drive_test_make_env(const char *map_file, int simulation_mode, int num_agents, int use_map_cache) {
    Drive env = drive_test_env_config(map_file, simulation_mode, num_agents, use_map_cache);
    allocate(&env);
    c_reset(&env);
    return env;
}

static inline void drive_set_neutral_actions(Drive *env) {
    if (env->action_type == 0) {
        int *actions = (int *) env->actions;
        int neutral;
        if (env->dynamics_model == JERK) {
            int num_long = sizeof(JERK_LONG) / sizeof(JERK_LONG[0]);
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            neutral = (num_long / 2) * num_lat + (num_lat / 2);
        } else {
            int num_accel = sizeof(ACCELERATION_VALUES) / sizeof(ACCELERATION_VALUES[0]);
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            neutral = (num_accel / 2) * num_steer + (num_steer / 2);
        }
        for (int i = 0; i < env->active_agent_count; i++) {
            actions[i] = neutral;
        }
        return;
    }
    float (*actions)[2] = (float (*)[2]) env->actions;
    for (int i = 0; i < env->active_agent_count; i++) {
        actions[i][0] = 0.0f;
        actions[i][1] = 0.0f;
    }
}

static inline int drive_all_finite(const float *values, int count) {
    for (int i = 0; i < count; i++) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static inline int drive_map_cache_live_count(void) {
    int live = 0;
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] != NULL) {
            live++;
        }
    }
    return live;
}

static inline void drive_map_cache_clear(void) {
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] != NULL) {
            free_shared_map_data(g_map_cache[i]);
        }
    }
    free(g_map_cache);
    g_map_cache = NULL;
    g_map_cache_count = 0;
}

#endif
