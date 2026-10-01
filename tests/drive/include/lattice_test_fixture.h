#ifndef LATTICE_TEST_FIXTURE_H
#define LATTICE_TEST_FIXTURE_H

#include "drive_fixture.h"

// Menu of the plan (puffer_drive.yaml defaults): MultiDiscrete([2, 20, 2, 60, 5])
static inline void lattice_test_default_menu(struct LatticeConfig *cfg, float decision_period_s) {
    const float offsets[] = {-0.9f, 0.0f, 0.9f};
    const float lat_durations[] = {2.4f, 3.6f, 4.8f, 6.0f};
    const float low_speed_distances[] = {5.0f, 10.0f, 15.0f, 20.0f};
    const float speeds[] = {0.0f, 1.0f, 2.5f, 5.0f, 7.5f, 10.0f, 12.5f, 15.0f, 17.5f, 20.0f};
    const float lon_durations[] = {1.2f, 2.4f, 3.6f, 4.8f, 6.0f};
    const float stops[] = {5.0f, 10.0f, 20.0f, 40.0f};
    const float backups[] = {1.0f, 2.0f, 5.0f, 10.0f};
    cfg->lat_offset_count = 3;
    memcpy(cfg->lat_offsets_m, offsets, sizeof offsets);
    cfg->lat_duration_count = 4;
    memcpy(cfg->lat_durations_s, lat_durations, sizeof lat_durations);
    memcpy(cfg->low_speed_distances_m, low_speed_distances, sizeof low_speed_distances);
    cfg->lon_speed_count = 10;
    memcpy(cfg->lon_speeds_mps, speeds, sizeof speeds);
    cfg->lon_duration_count = 5;
    memcpy(cfg->lon_durations_s, lon_durations, sizeof lon_durations);
    cfg->stop_distance_count = 4;
    memcpy(cfg->stop_distances_m, stops, sizeof stops);
    cfg->backup_distance_count = 4;
    memcpy(cfg->backup_distances_m, backups, sizeof backups);
    cfg->low_speed_mps = 3.0f;
    cfg->decision_period_s = decision_period_s;
    cfg->exit_mode = LATTICE_EXIT_MODE_POLICY;
}

static inline Drive make_lattice_env(const char *map_file, int num_agents, float dt) {
    Drive env = drive_test_env_config(map_file, SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    env.action_type = ACTION_TYPE_LATTICE;
    env.dt = dt;
    env.scenario_length = 5000;
    lattice_test_default_menu(&env.lattice, 0.3f);
    allocate(&env);
    c_reset(&env);
    return env;
}

// neutral per-agent coefficients so measured numbers match the plan (c = 1, no erratic behaviour)
static inline void lattice_test_neutral_agent(Agent *agent, float wheelbase) {
    agent->wheelbase = wheelbase;
    agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
    agent->reward_coefs[REWARD_COEF_STEER] = 1.0f;
    agent->reward_coefs[REWARD_COEF_ACC] = 1.0f;
    agent->reward_coefs[REWARD_COEF_SPEED] = 1.0f;
    agent->phantom_braking_counter = 0;
    agent->is_phantom_braker = 0;
    agent->is_blind_partner = 0;
}

// puts active slot 0 on lane_idx at arc_m moving at speed_mps along the lane, then rebuilds its lattice state
static inline void place_lattice_agent(Drive *env, int active_idx, int lane_idx, float arc_m, float speed_mps) {
    Agent *agent = &env->agents[env->active_agent_indices[active_idx]];
    float x, y, heading;
    lattice_lane_point_at_arc(env, lane_idx, arc_m, &x, &y, &heading);
    lattice_test_neutral_agent(agent, 2.7f);
    agent->sim_x = x;
    agent->sim_y = y;
    agent->sim_heading = heading;
    agent->cos_heading = cosf(heading);
    agent->sin_heading = sinf(heading);
    agent->sim_vx = speed_mps * agent->cos_heading;
    agent->sim_vy = speed_mps * agent->sin_heading;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    agent->steering_angle = 0.0f;
    agent->stopped = 0;
    agent->removed = 0;
    const RoadMapElement *lane = &env->road_elements[lane_idx];
    const float *cum = &env->lattice_lane_cum_m[env->lattice_lanes[lane_idx].cum_offset];
    int seg_idx = 0;
    while (seg_idx < lane->segment_size - 2 && cum[seg_idx + 1] < arc_m) {
        seg_idx++;
    }
    agent->sim_z = lane->z[seg_idx];
    update_agent_speed(agent);
    copy_pose_to_prev(agent);
    reset_lattice_state(env);
    compute_observations(env);
}

static inline void lattice_set_action(Drive *env, int active_idx, int lat_gate, int lat_cell, int lon_gate, int lon_cell, int exit_slot) {
    int *action = (int *) env->actions + active_idx * LATTICE_ACTION_FACTORS;
    action[LATTICE_FACTOR_LAT_GATE] = lat_gate;
    action[LATTICE_FACTOR_LAT_CELL] = lat_cell;
    action[LATTICE_FACTOR_LON_GATE] = lon_gate;
    action[LATTICE_FACTOR_LON_CELL] = lon_cell;
    action[LATTICE_FACTOR_EXIT] = exit_slot;
}

#endif
