#include "include/drive_fixture.h"
#include "include/lattice_test_fixture.h"
#include "include/test.h"

#define TOWN06 DRIVE_TEST_REPO_ROOT "/pufferlib/resources/drive/binaries/carla/opendrive__Town06.bin"
#define STRAIGHT_LANE_TOWN06 35
#define CURVED_LANE_TOWN06 33
#define LEFT_NEIGHBOUR_TOWN06 32

static const char *CARLA_TOWNS[] = {"Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"};

static void carla_town_path(char *out, size_t size, const char *town) {
    snprintf(out, size, DRIVE_TEST_REPO_ROOT "/pufferlib/resources/drive/binaries/carla/opendrive__%s.bin", town);
}

static void step_with_action(Drive *env, int lat_gate, int lat_cell, int lon_gate, int lon_cell, int exit_slot) {
    lattice_set_action(env, 0, lat_gate, lat_cell, lon_gate, lon_cell, exit_slot);
    c_step(env);
}

static void step_keep(Drive *env, int steps) {
    for (int step = 0; step < steps; step++) {
        step_with_action(env, 0, 0, 0, 0, 0);
    }
}

static Agent *slot0_agent(Drive *env) {
    return &env->agents[env->active_agent_indices[0]];
}

// fastest valid speed cell <= speed_set_mps (shortest duration first), keep if the committed plan already targets it
static void step_greedy(Drive *env, float speed_set_mps) {
    const struct LatticeConfig *cfg = &env->lattice;
    struct LatticeAgent *lattice_agent = &env->lattice_agents[0];
    const unsigned char *mask = lattice_agent->mask;
    int lon_cells = lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL);
    int keep_valid = mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_GATE) + LATTICE_GATE_KEEP];
    int pick = -1;
    for (int speed_idx = cfg->lon_speed_count - 1; speed_idx >= 0 && pick < 0; speed_idx--) {
        if (cfg->lon_speeds_mps[speed_idx] > speed_set_mps + 1e-6f) {
            continue;
        }
        for (int duration_idx = 0; duration_idx < cfg->lon_duration_count && pick < 0; duration_idx++) {
            int cell = duration_idx * cfg->lon_speed_count + speed_idx;
            pick = mask[lon_cells + cell] ? cell : -1;
        }
    }
    int keep = keep_valid && pick >= 0 && lattice_agent->lon.kind == LATTICE_LON_KIND_SPEED
        && lattice_agent->lon.target_speed_mps == cfg->lon_speeds_mps[pick % cfg->lon_speed_count];
    if (pick < 0) {
        step_with_action(env, 0, 0, 1, cfg->lon_emergency_cell, 0);
    } else {
        step_with_action(env, 0, 0, keep ? 0 : 1, keep ? 0 : pick, 0);
    }
}

static int test_polynomials_and_bellman(void) {
    double c[6];
    lattice_quintic_coefs(1.0, 2.0, -0.5, 12.0, 0.0, 0.0, 4.8, c);
    LatticePlanPoint end = lattice_poly_point(c, 4.8);
    EXPECT_NEAR((float) end.value, 12.0f, 1e-9f);
    EXPECT_NEAR((float) end.first, 0.0f, 1e-9f);
    EXPECT_NEAR((float) end.second, 0.0f, 1e-9f);
    lattice_quartic_speed_coefs(0.0, 3.0, 1.0, 10.0, 3.6, c);
    end = lattice_poly_point(c, 3.6);
    EXPECT_NEAR((float) end.first, 10.0f, 1e-9f);
    EXPECT_NEAR((float) end.second, 0.0f, 1e-9f);
    // a new plan started from the old plan's state at t reproduces the old plan (Bellman consistency)
    double old_plan[6], new_plan[6];
    lattice_quintic_coefs(0.0, 0.3, 0.1, 3.5, 0.0, 0.0, 6.0, old_plan);
    LatticePlanPoint mid = lattice_poly_point(old_plan, 2.1);
    lattice_quintic_coefs(mid.value, mid.first, mid.second, 3.5, 0.0, 0.0, 6.0 - 2.1, new_plan);
    for (int sample = 0; sample <= 10; sample++) {
        double u = 0.39 * sample;
        EXPECT_NEAR((float) (lattice_poly_point(new_plan, u).value - lattice_poly_point(old_plan, 2.1 + u).value), 0.0f, 1e-9f);
    }
    return 0;
}

static int test_menu_layout(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    const struct LatticeConfig *cfg = &env.lattice;
    EXPECT_EQ_INT(cfg->nvec[0], 2);
    EXPECT_EQ_INT(cfg->nvec[1], 20);
    EXPECT_EQ_INT(cfg->nvec[2], 2);
    EXPECT_EQ_INT(cfg->nvec[3], 60);
    EXPECT_EQ_INT(cfg->nvec[4], 5);
    EXPECT_EQ_INT(cfg->mask_feature_count, 89);
    EXPECT_EQ_INT(cfg->lon_stop_cell_base, 50);
    EXPECT_EQ_INT(cfg->lon_stop_line_cell, 54);
    EXPECT_EQ_INT(cfg->lon_emergency_cell, 55);
    EXPECT_EQ_INT(cfg->lon_backup_cell_base, 56);
    EXPECT_EQ_INT(lattice_lat_lane_side(cfg, 14), 1);
    EXPECT_EQ_INT(lattice_lat_cell_duration_idx(cfg, 14), 2);
    EXPECT_EQ_INT(lattice_lat_lane_side(cfg, 10), -1);
    EXPECT_NEAR(lattice_lat_cell_offset(cfg, 12), 0.0f, 0.0f);
    EXPECT_NEAR(lattice_lat_cell_offset(cfg, 13), 0.9f, 1e-6f);
    EXPECT_EQ_INT(cfg->lat_duration_steps[1], 12);
    EXPECT_EQ_INT(compute_observation_size(&env) - (EGO_FEATURES + PARTNER_FEATURES * env.obs_slots_partners_n + LANE_FEATURES * env.obs_slots_lane_kept
        + BOUNDARY_FEATURES * env.obs_slots_boundary_kept + TRAFFIC_CONTROL_FEATURES * env.obs_slots_traffic_controls_n + OBS_VALID_COUNT_FEATURES
        + env.num_goals * GOAL_FEATURES), LATTICE_PLAN_FEATURES + 89);
    free_allocated(&env);
    return 0;
}

// every exit of every split reachable through a slot; slot 0 straightest, then left to right; predecessors invert exits
static int test_exit_slots_and_predecessors_all_towns(void) {
    for (size_t town_idx = 0; town_idx < sizeof(CARLA_TOWNS) / sizeof(CARLA_TOWNS[0]); town_idx++) {
        char path[512];
        carla_town_path(path, sizeof path, CARLA_TOWNS[town_idx]);
        Drive env = make_lattice_env(path, 1, 0.3f);
        for (int lane = 0; lane < env.num_road_elements; lane++) {
            const struct LatticeLaneInfo *info = &env.lattice_lanes[lane];
            EXPECT_TRUE(info->exit_count <= LATTICE_EXIT_SLOTS);
            for (int slot = 1; slot < info->exit_count; slot++) {
                EXPECT_TRUE(fabsf(info->exit_turn_rad[0]) <= fabsf(info->exit_turn_rad[slot]) + 1e-6f);
                if (slot >= 2) {
                    EXPECT_TRUE(info->exit_turn_rad[slot - 1] >= info->exit_turn_rad[slot]);
                }
            }
            for (int slot = 0; slot < info->exit_count; slot++) {
                const struct LatticeLaneInfo *target = &env.lattice_lanes[info->exit_slots[slot]];
                int found = 0;
                for (int pred = 0; pred < target->predecessor_count; pred++) {
                    found |= target->predecessors[pred] == lane;
                }
                EXPECT_TRUE(found);
            }
        }
        free_allocated(&env);
    }
    return 0;
}

static float distance_to_lane_polyline(const Drive *env, int lane_idx, float x, float y) {
    const RoadMapElement *lane = &env->road_elements[lane_idx];
    float best = 1e9f;
    for (int seg = 0; seg + 1 < lane->segment_size; seg++) {
        best = fminf(best, compute_point_to_segment_distance(x, y, lane->x[seg], lane->y[seg], lane->x[seg + 1], lane->y[seg + 1]));
    }
    return best;
}

// rail: exact 0.5 m spacing, smooth heading, bounded curvature, and close to the raw lanes (position filter is local)
static int test_rail_geometry_all_towns(void) {
    int checked_samples = 0;
    float max_offset_m = 0.0f;
    for (size_t town_idx = 0; town_idx < sizeof(CARLA_TOWNS) / sizeof(CARLA_TOWNS[0]); town_idx++) {
        char path[512];
        carla_town_path(path, sizeof path, CARLA_TOWNS[town_idx]);
        Drive env = make_lattice_env(path, 1, 0.3f);
        for (int lane = 0; lane < env.num_road_elements; lane += 7) {
            if (!is_drivable_road_lane(env.road_elements[lane].type) || env.lattice_lanes[lane].length_m < 2.0f) {
                continue;
            }
            place_lattice_agent(&env, 0, lane, 0.5f * env.lattice_lanes[lane].length_m, 5.0f);
            const struct LatticeRail *rail = &env.lattice_agents[0].rail;
            EXPECT_TRUE(env.lattice_agents[0].has_reference);
            EXPECT_TRUE(rail->sample_count > 100);
            for (int sample = 0; sample + 1 < rail->sample_count; sample++) {
                float dx = rail->x[sample + 1] - rail->x[sample];
                float dy = rail->y[sample + 1] - rail->y[sample];
                EXPECT_NEAR(sqrtf(dx * dx + dy * dy), LATTICE_RAIL_SPACING_M, 2e-3f);
                EXPECT_TRUE(fabsf(lattice_wrap_angle(rail->heading[sample + 1] - rail->heading[sample])) < 0.35f);
                int slot = rail->chain_slot[sample];
                float offset_m = distance_to_lane_polyline(&env, rail->lanes[slot], rail->x[sample], rail->y[sample]);
                for (int near_slot = slot > 0 ? slot - 1 : 0; near_slot <= slot + 1 && near_slot < rail->lane_count; near_slot++) {
                    offset_m = fminf(offset_m, distance_to_lane_polyline(&env, rail->lanes[near_slot], rail->x[sample], rail->y[sample]));
                }
                if (sample > 2 * LATTICE_SMOOTH_HALF_WINDOW && sample + 2 * LATTICE_SMOOTH_HALF_WINDOW < rail->sample_count) {
                    max_offset_m = fmaxf(max_offset_m, offset_m);
                    checked_samples++;
                }
            }
        }
        free_allocated(&env);
    }
    printf("  rail samples checked %d, max offset from the raw lane %.3f m\n", checked_samples, max_offset_m);
    EXPECT_TRUE(checked_samples > 10000);
    EXPECT_TRUE(max_offset_m < 1.5f);
    return 0;
}

// rail offset from the raw lane on straight stretches stays small along a long drive (no heading-filter drift)
static int test_long_drive_no_rail_drift(void) {
    char path[512];
    carla_town_path(path, sizeof path, "Town03");
    Drive env = make_lattice_env(path, 1, 0.3f);
    int lane = -1;
    for (int candidate = 0; candidate < env.num_road_elements && lane < 0; candidate++) {
        if (is_drivable_road_lane(env.road_elements[candidate].type) && env.lattice_lanes[candidate].length_m > 40.0f
            && !env.lattice_lanes[candidate].is_connector) {
            lane = candidate;
        }
    }
    EXPECT_TRUE(lane >= 0);
    place_lattice_agent(&env, 0, lane, 5.0f, 0.0f);
    float worst_straight_offset_m = 0.0f;
    int regenerations_before = 0;
    for (int step = 0; step < 1200; step++) {
        step_greedy(&env, 10.0f);
        struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
        const struct LatticeRail *rail = &lattice_agent->rail;
        int sample = lattice_agent->projection_hint;
        int straight = 1;
        for (int window = sample - 16; window <= sample + 16 && straight; window++) {
            straight = window >= 0 && window < rail->sample_count && fabsf(rail->curvature[window]) < 1e-3f;
        }
        if (straight) {
            int slot = rail->chain_slot[sample];
            float offset_m = distance_to_lane_polyline(&env, rail->lanes[slot], rail->x[sample], rail->y[sample]);
            for (int near_slot = slot > 0 ? slot - 1 : 0; near_slot <= slot + 1 && near_slot < rail->lane_count; near_slot++) {
                offset_m = fminf(offset_m, distance_to_lane_polyline(&env, rail->lanes[near_slot], rail->x[sample], rail->y[sample]));
            }
            worst_straight_offset_m = fmaxf(worst_straight_offset_m, offset_m);
        }
        regenerations_before = (int) lattice_agent->counters.rail_regens;
    }
    printf("  1200 greedy steps (10 m/s cap): %.0f m driven, rail regenerations %d, worst rail offset from the lane on straights %.3f m\n", env.lattice_agents[0].sigma_m, regenerations_before, worst_straight_offset_m);
    EXPECT_TRUE(regenerations_before > 5);
    EXPECT_TRUE(worst_straight_offset_m < 0.1f);
    free_allocated(&env);
    return 0;
}

// forward EMERGENCY stop distances vs the plan (c = 1, dt 0.3): 3.73 / 12.10 / 43.92 m, never reversing
static int test_emergency_distances(void) {
    const float speeds[] = {5.0f, 10.0f, 20.0f};
    const float expected_m[] = {3.73f, 12.10f, 43.92f};
    for (int case_idx = 0; case_idx < 3; case_idx++) {
        Drive env = make_lattice_env(TOWN06, 1, 0.3f);
        place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 60.0f, speeds[case_idx]);
        Agent *agent = slot0_agent(&env);
        float min_speed = 1e9f;
        step_with_action(&env, 0, 0, 1, env.lattice.lon_emergency_cell, 0);
        for (int step = 0; step < 60; step++) {
            min_speed = fminf(min_speed, agent->sim_speed_signed);
            step_keep(&env, 1);
        }
        float travelled_m = (float) env.lattice_agents[0].sigma_m;
        printf("  EMERGENCY from %.0f m/s: %.2f m (plan %.2f), end speed %.4f, min speed %.4f\n", speeds[case_idx], travelled_m, expected_m[case_idx], agent->sim_speed_signed, min_speed);
        EXPECT_NEAR(travelled_m, expected_m[case_idx], 0.35f);
            EXPECT_TRUE(min_speed >= -1e-4f);
        EXPECT_NEAR(agent->sim_speed_signed, 0.0f, 1e-4f);
        free_allocated(&env);
    }
    return 0;
}

// fastest-valid-cell keep-lane driving (the plan's regression policy): tracks the rail, never overspeeds the envelope
static int test_keep_lane_tracking(void) {
    const float speed_sets[] = {8.0f, 12.5f, 20.0f};
    for (int case_idx = 0; case_idx < 3; case_idx++) {
        Drive env = make_lattice_env(TOWN06, 1, 0.3f);
        place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 20.0f, 0.0f);
        Agent *agent = slot0_agent(&env);
        struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
        float max_d = 0.0f, max_speed = 0.0f, max_lat_accel = 0.0f;
        for (int step = 0; step < 300; step++) {
            step_greedy(&env, speed_sets[case_idx]);
            LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
            max_d = fmaxf(max_d, fabsf(frenet.d));
            max_speed = fmaxf(max_speed, agent->sim_speed);
            max_lat_accel = fmaxf(max_lat_accel, fabsf(frenet.curvature) * frenet.s_dot * frenet.s_dot);
        }
        printf("  greedy v_set %.1f for 90 s: max |d| %.3f m, max speed %.2f, max k*sdot^2 %.2f, distance %.0f m, lost %.0f, emergency steps %.0f\n", speed_sets[case_idx], max_d,
               max_speed, max_lat_accel, lattice_agent->sigma_m, lattice_agent->counters.lost, lattice_agent->counters.emergency_steps);
        EXPECT_TRUE(max_d < 0.45f);
        EXPECT_TRUE(max_lat_accel < 4.2f);
        EXPECT_TRUE(lattice_agent->sigma_m > 100.0);
        EXPECT_EQ_INT((int) lattice_agent->counters.lost, 0);
        free_allocated(&env);
    }
    return 0;
}

// stop cell from 5 m/s ends exactly at rest near its target
static int test_stop_cell_exact(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 40.0f, 5.0f);
    Agent *agent = slot0_agent(&env);
    int stop_10m_cell = env.lattice.lon_stop_cell_base + 1;
    EXPECT_TRUE(env.lattice_agents[0].mask[lattice_mask_offset(&env.lattice, LATTICE_FACTOR_LON_CELL) + stop_10m_cell]);
    step_with_action(&env, 0, 0, 1, stop_10m_cell, 0);
    step_keep(&env, 60);
    float stopped_at_m = (float) env.lattice_agents[0].sigma_m;
    printf("  stop at 10 m from 5 m/s: stopped after %.3f m, v %.5f a %.5f\n", stopped_at_m, agent->sim_speed_signed, agent->accel_long);
    EXPECT_NEAR(stopped_at_m, 10.0f, 0.3f);
    EXPECT_NEAR(agent->sim_speed_signed, 0.0f, 1e-4f);
    EXPECT_NEAR(agent->accel_long, 0.0f, 1e-4f);
    EXPECT_TRUE(lattice_is_stopped_exactly(agent));
    free_allocated(&env);
    return 0;
}

// back up 5 m from rest: T 4.8 s (c = 1, dt 0.3), lands within 0.2 m past B, ends exactly at rest; forward masked meanwhile
static int test_backup_cell(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 60.0f, 0.0f);
    Agent *agent = slot0_agent(&env);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    int backup_5m_cell = cfg->lon_backup_cell_base + 2;
    EXPECT_NEAR(lattice_agent->backup_duration_s[2], 4.8f, 1e-4f);
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + backup_5m_cell]);
    step_with_action(&env, 0, 0, 1, backup_5m_cell, 0);
    EXPECT_EQ_INT(lattice_agent->gear, -1);
    step_keep(&env, 3);
    EXPECT_TRUE(agent->sim_speed_signed < 0.0f);
    EXPECT_FALSE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + 5]);
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + cfg->lon_emergency_cell]);
    step_keep(&env, 30);
    printf("  back up 5 m: sigma %.3f, v %.5f, a %.5f, gear %d, backed %.2f m\n", lattice_agent->sigma_m, agent->sim_speed_signed, agent->accel_long, lattice_agent->gear, lattice_agent->counters.backup_m);
    EXPECT_TRUE(lattice_agent->sigma_m <= -5.0 + 0.02 && lattice_agent->sigma_m >= -5.2);
    EXPECT_NEAR(agent->sim_speed_signed, 0.0f, 1e-4f);
    EXPECT_NEAR(agent->accel_long, 0.0f, 1e-4f);
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + 10 * 2 + 1]);
    free_allocated(&env);
    return 0;
}

// Town06 lanes 33 -> 32 on the 208 degree curve at 4.5 m/s, [1, 14, 0, 0, 0]: rail switches at the decision
static int test_lane_change_on_curve(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    place_lattice_agent(&env, 0, CURVED_LANE_TOWN06, 20.0f, 4.5f);
    Agent *agent = slot0_agent(&env);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    EXPECT_EQ_INT(lattice_agent->neighbour_lane[1], LEFT_NEIGHBOUR_TOWN06);
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(&env.lattice, LATTICE_FACTOR_LAT_CELL) + 14]);
    step_with_action(&env, 1, 14, 0, 0, 0);
    EXPECT_EQ_INT(lattice_agent->rail.lanes[lattice_agent->rail.chain_slot[lattice_agent->projection_hint]], LEFT_NEIGHBOUR_TOWN06);
    EXPECT_EQ_INT(lattice_agent->rail_changed_flag, 1);
    EXPECT_EQ_INT(lattice_agent->lane_change_active, 1);
    float min_speed = 1e9f, max_speed = 0.0f, max_plan_error = 0.0f;
    for (int step = 0; step < 36; step++) {
        step_keep(&env, 1);
        min_speed = fminf(min_speed, agent->sim_speed);
        max_speed = fmaxf(max_speed, agent->sim_speed);
        LatticeFrenet now = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
        LatticePlanPoint planned = lattice_lat_state(&env, &lattice_agent->lat, env.timestep + 1, now.s, now.s_dot, agent->accel_long);
        max_plan_error = fmaxf(max_plan_error, fabsf((float) planned.value - now.d));
    }
    LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    printf("  lane change 33 -> 32 at 4.5 m/s: lands at d %.3f after 11 s, max |d - plan| %.3f, speed %.2f..%.2f, active %d\n", frenet.d, max_plan_error, min_speed, max_speed, lattice_agent->lane_change_active);
    EXPECT_TRUE(fabsf(frenet.d) < 0.05f);
    EXPECT_TRUE(max_plan_error < 0.15f);
    EXPECT_TRUE(min_speed > 4.3f && max_speed < 4.7f);
    EXPECT_EQ_INT(lattice_agent->lane_change_active, 0);
    EXPECT_EQ_INT(lattice_agent->rail_changed_flag, 0);
    free_allocated(&env);
    return 0;
}

// dt 0.1 with a 0.3 s decision period: actions only act on decision steps; masks are index-0 only between contexts
static int test_decision_period_at_dt_01(void) {
    Drive env = drive_test_env_config(TOWN06, SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    env.action_type = ACTION_TYPE_LATTICE;
    env.dt = 0.1f;
    env.scenario_length = 5000;
    lattice_test_default_menu(&env.lattice, 0.3f);
    allocate(&env);
    c_reset(&env);
    EXPECT_EQ_INT(env.lattice.decision_period_steps, 3);
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 60.0f, 10.0f);
    int obs_size = compute_observation_size(&env);
    int mask_base = obs_size - env.lattice.mask_feature_count;
    int non_context_seen = 0;
    for (int step = 0; step < 9; step++) {
        int is_decision = ((env.timestep + 1 - env.episode_start_step) % 3) == 1;
        step_with_action(&env, 0, 0, 1, env.lattice.lon_emergency_cell, 0);
        int emergency = env.lattice_agents[0].lon.kind == LATTICE_LON_KIND_EMERGENCY;
        if (!is_decision) {
            continue;
        }
        EXPECT_TRUE(emergency);
        break;
    }
    Drive env2 = drive_test_env_config(TOWN06, SIMULATION_MODE_GIGAFLOW, 1, 0);
    env2.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    env2.action_type = ACTION_TYPE_LATTICE;
    env2.dt = 0.1f;
    env2.scenario_length = 5000;
    lattice_test_default_menu(&env2.lattice, 0.3f);
    allocate(&env2);
    c_reset(&env2);
    place_lattice_agent(&env2, 0, STRAIGHT_LANE_TOWN06, 60.0f, 10.0f);
    for (int step = 0; step < 6; step++) {
        int next_is_decision = ((env2.timestep + 1 - env2.episode_start_step) % 3) == 1;
        lattice_set_action(&env2, 0, 0, 0, next_is_decision ? 0 : 1, env2.lattice.lon_emergency_cell, 0);
        c_step(&env2);
        EXPECT_TRUE(env2.lattice_agents[0].lon.kind != LATTICE_LON_KIND_EMERGENCY);
        if (!lattice_is_context_step(&env2)) {
            non_context_seen = 1;
            float *obs = env2.observations;
            EXPECT_NEAR(obs[mask_base + 0], 1.0f, 0.0f);
            EXPECT_NEAR(obs[mask_base + 1], 0.0f, 0.0f);
            EXPECT_NEAR(obs[mask_base + 2 + 20 + 1], 0.0f, 0.0f);
        }
    }
    EXPECT_TRUE(non_context_seen);
    free_allocated(&env);
    free_allocated(&env2);
    return 0;
}

static int test_invalid_actions_fall_back(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 60.0f, 10.0f);
    step_with_action(&env, 7, 99, 5, -3, 11);
    EXPECT_NEAR(env.lattice_agents[0].counters.invalid_actions, 1.0f, 0.0f);
    step_with_action(&env, 1, 0, 1, 0, 0);
    EXPECT_TRUE(env.lattice_agents[0].counters.invalid_actions >= 1.0f);
    step_keep(&env, 5);
    EXPECT_FINITE(slot0_agent(&env)->sim_x);
    free_allocated(&env);
    return 0;
}

static unsigned long long lcg_next(unsigned long long *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state >> 33;
}

// uniformly random valid action per factor, drawn from the observed masks
static void random_valid_actions(Drive *env, unsigned long long *rng_state) {
    int obs_size = compute_observation_size(env);
    const struct LatticeConfig *cfg = &env->lattice;
    for (int active_idx = 0; active_idx < env->active_agent_count; active_idx++) {
        const float *masks = env->observations + active_idx * obs_size + obs_size - cfg->mask_feature_count;
        int *action = (int *) env->actions + active_idx * LATTICE_ACTION_FACTORS;
        int offset = 0;
        for (int factor = 0; factor < LATTICE_ACTION_FACTORS; factor++) {
            int valid_count = 0;
            for (int value = 0; value < cfg->nvec[factor]; value++) {
                valid_count += masks[offset + value] > 0.5f;
            }
            int pick = valid_count > 0 ? (int) (lcg_next(rng_state) % valid_count) : 0;
            action[factor] = 0;
            for (int value = 0; value < cfg->nvec[factor]; value++) {
                if (masks[offset + value] > 0.5f && pick-- == 0) {
                    action[factor] = value;
                    break;
                }
            }
            offset += cfg->nvec[factor];
        }
    }
}

static int test_determinism(void) {
    Drive env_a = make_lattice_env(TOWN06, 8, 0.3f);
    Drive env_b = make_lattice_env(TOWN06, 8, 0.3f);
    unsigned long long rng_a = 11, rng_b = 11;
    int obs_size = compute_observation_size(&env_a);
    for (int step = 0; step < 150; step++) {
        random_valid_actions(&env_a, &rng_a);
        random_valid_actions(&env_b, &rng_b);
        c_step(&env_a);
        c_step(&env_b);
        EXPECT_TRUE(memcmp(env_a.observations, env_b.observations, env_a.active_agent_count * obs_size * sizeof(float)) == 0);
    }
    free_allocated(&env_a);
    free_allocated(&env_b);
    return 0;
}

// many agents, random valid actions, every CARLA town: finite observations, masks never empty, some lane changes and back-ups
static int test_random_rollouts_all_towns(void) {
    float lane_changes = 0.0f, backups = 0.0f, backup_m = 0.0f, decisions = 0.0f, rejects = 0.0f, invalid = 0.0f;
    for (size_t town_idx = 0; town_idx < sizeof(CARLA_TOWNS) / sizeof(CARLA_TOWNS[0]); town_idx++) {
        char path[512];
        carla_town_path(path, sizeof path, CARLA_TOWNS[town_idx]);
        Drive env = make_lattice_env(path, 16, 0.3f);
        unsigned long long rng_state = 5 + town_idx;
        int obs_size = compute_observation_size(&env);
        for (int step = 0; step < 300; step++) {
            random_valid_actions(&env, &rng_state);
            c_step(&env);
            for (int value = 0; value < env.active_agent_count * obs_size; value++) {
                EXPECT_FINITE(env.observations[value]);
            }
        }
        for (int active_idx = 0; active_idx < env.active_agent_count; active_idx++) {
            const struct LatticeCounters *counters = &env.lattice_agents[active_idx].counters;
            lane_changes += counters->chosen_changes;
            backups += counters->backups;
            backup_m += counters->backup_m;
            decisions += counters->decisions;
            rejects += counters->decode_rejects;
            invalid += counters->invalid_actions;
        }
        free_allocated(&env);
    }
    printf("  random rollouts: decisions %.0f, lane changes %.0f, back-ups %.0f (%.1f m), decode rejects %.0f, invalid %.0f\n", decisions, lane_changes, backups, backup_m, rejects, invalid);
    EXPECT_TRUE(lane_changes > 0.0f);
    EXPECT_TRUE(backup_m > 0.0f);
    EXPECT_NEAR(invalid, 0.0f, 0.0f);
    return 0;
}


// plan's gear_sequence: 5 m/s -> stop at 10 m -> back up 5 m returning to centre -> forward again, no creep at rest
static int test_gear_change_sequence(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 40.0f, 5.0f);
    Agent *agent = slot0_agent(&env);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    step_with_action(&env, 0, 0, 1, cfg->lon_stop_cell_base + 1, 0);
    step_keep(&env, 30);
    EXPECT_TRUE(lattice_is_stopped_exactly(agent));
    double stopped_sigma = lattice_agent->sigma_m;
    float max_rest_speed = 0.0f;
    for (int step = 0; step < 3; step++) {
        step_keep(&env, 1);
        max_rest_speed = fmaxf(max_rest_speed, fabsf(agent->sim_speed_signed));
    }
    int backup_5m_cell = cfg->lon_backup_cell_base + 2;
    int centre_last_duration_cell = (cfg->lat_duration_count - 1) * lattice_lat_choice_count(cfg) + 2;
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + backup_5m_cell]);
    step_with_action(&env, 1, centre_last_duration_cell, 1, backup_5m_cell, 0);
    EXPECT_EQ_INT(lattice_agent->gear, -1);
    EXPECT_EQ_INT(lattice_agent->lat.dir, -1);
    step_keep(&env, 30);
    double backed_m = stopped_sigma - lattice_agent->sigma_m;
    EXPECT_TRUE(lattice_is_stopped_exactly(agent));
    for (int step = 0; step < 3; step++) {
        step_keep(&env, 1);
        max_rest_speed = fmaxf(max_rest_speed, fabsf(agent->sim_speed_signed));
    }
    int forward_cell = 2 * cfg->lon_speed_count + 3;
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + forward_cell]);
    step_with_action(&env, 0, 0, 1, forward_cell, 0);
    EXPECT_EQ_INT(lattice_agent->gear, 1);
    step_keep(&env, 20);
    printf("  gear sequence: stopped %.3f m, backed %.3f m, max speed at rest %.4f, forward speed %.2f\n", stopped_sigma, backed_m, max_rest_speed, agent->sim_speed_signed);
    EXPECT_NEAR(stopped_sigma, 10.0f, 0.3f);
    EXPECT_TRUE(backed_m > 4.9 && backed_m < 5.25);
    EXPECT_TRUE(max_rest_speed < 0.01f);
    EXPECT_NEAR(agent->sim_speed_signed, 5.0f, 0.2f);
    free_allocated(&env);
    return 0;
}

// a split within the freeze distance makes the exit factor live once; +-lane is masked on that decision; the slot is applied
static int test_exit_live_once_per_split(void) {
    char path[512];
    carla_town_path(path, sizeof path, "Town03");
    Drive env = make_lattice_env(path, 1, 0.3f);
    int split_lane = -1;
    for (int lane = 0; lane < env.num_road_elements && split_lane < 0; lane++) {
        const struct LatticeLaneInfo *info = &env.lattice_lanes[lane];
        if (info->exit_count >= 2 && info->length_m > 30.0f && !info->is_connector) {
            split_lane = lane;
        }
    }
    EXPECT_TRUE(split_lane >= 0);
    place_lattice_agent(&env, 0, split_lane, env.lattice_lanes[split_lane].length_m - 25.0f, 3.0f);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    EXPECT_TRUE(lattice_agent->live_split_slot >= 0);
    int exit_offset = lattice_mask_offset(cfg, LATTICE_FACTOR_EXIT);
    EXPECT_TRUE(lattice_agent->mask[exit_offset + 1]);
    for (int cell = 0; cell < cfg->lat_cell_count; cell++) {
        if (lattice_lat_lane_side(cfg, cell) != 0) {
            EXPECT_FALSE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL) + cell]);
        }
    }
    int chosen_exit = env.lattice_lanes[split_lane].exit_slots[1];
    step_with_action(&env, 0, 0, 0, 0, 1);
    const struct LatticeRail *rail = &lattice_agent->rail;
    int split_slot = -1;
    for (int lane_slot = 0; lane_slot + 1 < rail->lane_count; lane_slot++) {
        split_slot = rail->lanes[lane_slot] == split_lane && rail->lanes[lane_slot + 1] == chosen_exit ? lane_slot : split_slot;
    }
    EXPECT_TRUE(split_slot >= 0);
    EXPECT_EQ_INT(rail->exit_decided[split_slot], 1);
    EXPECT_TRUE(lattice_agent->live_split_slot != split_slot);
    EXPECT_NEAR(lattice_agent->counters.exit_decisions, 1.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

// on rail 33, the car drifts 2.2 m toward lane 32 (3.5 m away): one drift switch, the plan is kept physically, so the
// car returns to lane 33's centre and switches back once; the 0.3 m band prevents any further flip-flop
static int test_drift_switch_with_hysteresis(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    place_lattice_agent(&env, 0, CURVED_LANE_TOWN06, 20.0f, 4.5f);
    Agent *agent = slot0_agent(&env);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    float offset_m = lattice_agent->neighbour_offset_m[1];
    agent->sim_x -= 2.2f * agent->sin_heading;
    agent->sim_y += 2.2f * agent->cos_heading;
    update_lattice_before_observations(&env);
    compute_observations(&env);
    int lane_after_drift = lattice_agent->rail.lanes[lattice_agent->rail.chain_slot[lattice_agent->projection_hint]];
    LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    float target_after_drift = lattice_agent->lat.target_d_m;
    step_keep(&env, 60);
    int final_lane = lattice_agent->rail.lanes[lattice_agent->rail.chain_slot[lattice_agent->projection_hint]];
    printf("  drift: offset %.2f m, lane after drift %d (d %.2f, target %.2f), switches after 18 s %.0f, final lane %d\n", offset_m, lane_after_drift, frenet.d,
           target_after_drift, lattice_agent->counters.drift_changes, final_lane);
    EXPECT_EQ_INT(lane_after_drift, LEFT_NEIGHBOUR_TOWN06);
    EXPECT_TRUE(fabsf(frenet.d - (2.2f - offset_m)) < 0.3f);
    EXPECT_TRUE(fabsf(target_after_drift + offset_m) < 0.3f);
    EXPECT_TRUE(lattice_agent->counters.drift_changes <= 2.0f);
    free_allocated(&env);
    return 0;
}

// far from every lane: straight reference, lateral keep only, back-ups masked; removed agents get index-0 rows
static int test_no_lane_mode_and_removed_rows(void) {
    Drive env = make_lattice_env(TOWN06, 2, 0.3f);
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 100.0f, 0.0f);
    Agent *agent = slot0_agent(&env);
    agent->sim_x += 40.0f * -agent->sin_heading;
    agent->sim_y += 40.0f * agent->cos_heading;
    Agent *removed = &env.agents[env.active_agent_indices[1]];
    removed->removed = 1;
    reset_lattice_state(&env);
    compute_observations(&env);
    const struct LatticeConfig *cfg = &env.lattice;
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    EXPECT_EQ_INT(lattice_agent->has_reference, 0);
    EXPECT_EQ_INT(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_GATE) + LATTICE_GATE_NEW], 0);
    for (int backup = 0; backup < cfg->backup_distance_count; backup++) {
        EXPECT_FALSE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + cfg->lon_backup_cell_base + backup]);
    }
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + cfg->lon_emergency_cell]);
    int obs_size = compute_observation_size(&env);
    const float *removed_obs = env.observations + obs_size;
    for (int feature = 0; feature < LATTICE_PLAN_FEATURES; feature++) {
        EXPECT_NEAR(removed_obs[EGO_FEATURES + feature], 0.0f, 0.0f);
    }
    int mask_base = obs_size - cfg->mask_feature_count;
    int valid = 0;
    for (int feature = 0; feature < cfg->mask_feature_count; feature++) {
        valid += removed_obs[mask_base + feature] > 0.5f;
    }
    EXPECT_EQ_INT(valid, LATTICE_ACTION_FACTORS);
    step_keep(&env, 5);
    EXPECT_FINITE(agent->sim_x);
    free_allocated(&env);
    return 0;
}

// the rail-changed flag is seen from the chosen change until the next decision (3 steps at dt 0.1)
static int test_rail_changed_flag_timing_dt_01(void) {
    Drive env = drive_test_env_config(TOWN06, SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    env.action_type = ACTION_TYPE_LATTICE;
    env.dt = 0.1f;
    env.scenario_length = 5000;
    lattice_test_default_menu(&env.lattice, 0.3f);
    allocate(&env);
    c_reset(&env);
    place_lattice_agent(&env, 0, CURVED_LANE_TOWN06, 20.0f, 4.5f);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    step_keep(&env, 1);
    int obs_size = compute_observation_size(&env);
    int flag_idx = EGO_FEATURES + 41;
    int flag_steps = 0;
    for (int step = 0; step < 6; step++) {
        int next_is_decision = ((env.timestep + 1 - env.episode_start_step) % 3) == 1;
        if (next_is_decision && flag_steps == 0) {
            EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(&env.lattice, LATTICE_FACTOR_LAT_CELL) + 14]);
            step_with_action(&env, 1, 14, 0, 0, 0);
            EXPECT_NEAR(env.observations[flag_idx], 1.0f, 0.0f);
            flag_steps = 1;
            continue;
        }
        step_keep(&env, 1);
        if (flag_steps > 0 && env.observations[flag_idx] > 0.5f) {
            flag_steps++;
        }
    }
    printf("  rail-changed flag seen on %d consecutive steps at dt 0.1\n", flag_steps);
    EXPECT_EQ_INT(flag_steps, 3);
    (void) obs_size;
    free_allocated(&env);
    return 0;
}

// plan-change RMS is zero while keeping, positive on a lane change, charged once at the consistency coefficient
static int test_plan_change_rms_consistency(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    place_lattice_agent(&env, 0, CURVED_LANE_TOWN06, 20.0f, 4.5f);
    Agent *agent = slot0_agent(&env);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    agent->reward_coefs[REWARD_COEF_TRAJECTORY_CONSISTENCY] = 1.0f;
    step_keep(&env, 1);
    EXPECT_NEAR(lattice_agent->counters.plan_change_rms_m, 0.0f, 0.0f);
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(&env.lattice, LATTICE_FACTOR_LAT_CELL) + 14]);
    step_with_action(&env, 1, 14, 0, 0, 0);
    float change_rms_m = lattice_agent->counters.plan_change_rms_m;
    printf("  lane change plan-change RMS %.3f m, consistency reward %.3f\n", change_rms_m, env.logs[0].reward_trajectory_consistency);
    EXPECT_TRUE(change_rms_m > 0.3f && change_rms_m < LANE_WIDTH);
    EXPECT_NEAR(env.logs[0].reward_trajectory_consistency, -change_rms_m, 1e-6f);
    EXPECT_NEAR(lattice_agent->plan_change_rms_m, 0.0f, 0.0f);
    step_keep(&env, 3);
    EXPECT_NEAR(lattice_agent->counters.plan_change_rms_m, change_rms_m, 0.0f);
    EXPECT_NEAR(env.logs[0].reward_trajectory_consistency, -change_rms_m, 1e-6f);
    free_allocated(&env);
    return 0;
}

// route progress on a straight lane toward a goal 80 m ahead pays one unit per meter driven
static int test_route_progress_reward(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    env.reward_route_progress = 1.0f;
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 40.0f, 5.0f);
    Agent *agent = slot0_agent(&env);
    float goal_x, goal_y, goal_heading;
    lattice_lane_point_at_arc(&env, STRAIGHT_LANE_TOWN06, 120.0f, &goal_x, &goal_y, &goal_heading);
    agent->list_goal_x[0] = goal_x;
    agent->list_goal_y[0] = goal_y;
    agent->list_goal_z[0] = agent->sim_z;
    agent->list_goal_lane[0] = STRAIGHT_LANE_TOWN06;
    agent->goal_count = 1;
    agent->current_goal_idx = 0;
    agent->current_goal_x = goal_x;
    agent->current_goal_y = goal_y;
    agent->current_goal_z = agent->sim_z;
    float start_x = agent->sim_x, start_y = agent->sim_y;
    step_keep(&env, 1);
    float first_x = agent->sim_x, first_y = agent->sim_y;
    EXPECT_NEAR(env.logs[0].reward_route_progress, 0.0f, 0.0f);
    step_keep(&env, 10);
    float driven_m = sqrtf((agent->sim_x - first_x) * (agent->sim_x - first_x) + (agent->sim_y - first_y) * (agent->sim_y - first_y));
    printf("  route progress over %.2f m driven: %.2f (start %.1f m from first step)\n", driven_m, env.logs[0].reward_route_progress,
           sqrtf((first_x - start_x) * (first_x - start_x) + (first_y - start_y) * (first_y - start_y)));
    EXPECT_TRUE(driven_m > 10.0f);
    EXPECT_NEAR(env.logs[0].reward_route_progress, driven_m, 0.3f);
    free_allocated(&env);
    return 0;
}

// the plan block reports the red / yellow state of the light at the next stop line on the chain
static int test_stop_line_light_features(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    int light_lane = -1, approach_lane = -1;
    for (int lane_idx = 0; lane_idx < env.num_road_elements && light_lane < 0; lane_idx++) {
        const struct LatticeLaneInfo *info = &env.lattice_lanes[lane_idx];
        if (!is_drivable_road_lane(env.road_elements[lane_idx].type) || info->traffic_light_idx < 0 || info->predecessor_count == 0) {
            continue;
        }
        int predecessor = info->predecessors[0];
        if (env.lattice_lanes[predecessor].exit_slots[0] == lane_idx && env.lattice_lanes[predecessor].length_m > 25.0f) {
            light_lane = lane_idx;
            approach_lane = predecessor;
        }
    }
    EXPECT_TRUE(light_lane >= 0);
    place_lattice_agent(&env, 0, approach_lane, env.lattice_lanes[approach_lane].length_m - 20.0f, 0.0f);
    TrafficControlElement *light = &env.traffic_elements[env.lattice_lanes[light_lane].traffic_light_idx];
    EXPECT_TRUE(env.timestep < light->state_size);
    int plan_idx = EGO_FEATURES;
    int states[3] = {TRAFFIC_CONTROL_STATE_RED, TRAFFIC_CONTROL_STATE_YELLOW, TRAFFIC_CONTROL_STATE_GREEN};
    for (int state_idx = 0; state_idx < 3; state_idx++) {
        light->states[env.timestep] = states[state_idx];
        compute_observations(&env);
        EXPECT_TRUE(env.observations[plan_idx + 38] < 1.0f);
        EXPECT_NEAR(env.observations[plan_idx + 39], state_idx == 0 ? 1.0f : 0.0f, 0.0f);
        EXPECT_NEAR(env.observations[plan_idx + 40], state_idx == 1 ? 1.0f : 0.0f, 0.0f);
    }
    // out of the traffic-control view: with light_in_view the features report none, without it the rail still does
    light->states[env.timestep] = TRAFFIC_CONTROL_STATE_RED;
    env.obs_range_traffic_control_m = 5.0f;
    env.lattice.light_in_view = 1;
    compute_observations(&env);
    EXPECT_NEAR(env.observations[plan_idx + 38], 1.0f, 0.0f);
    EXPECT_NEAR(env.observations[plan_idx + 39], 0.0f, 0.0f);
    EXPECT_NEAR(env.observations[plan_idx + 40], 0.0f, 0.0f);
    env.lattice.light_in_view = 0;
    compute_observations(&env);
    EXPECT_TRUE(env.observations[plan_idx + 38] < 1.0f);
    EXPECT_NEAR(env.observations[plan_idx + 39], 1.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

// waiting penalty: frac x cheapest infraction per step at rest, half at 0.5 m/s, none at 1 m/s or at a reported red light
static int test_wait_penalty(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    env.reward_wait_penalty_frac = 5e-4f;
    float speeds[3] = {0.0f, 0.5f, 5.0f};
    float expected[3] = {-5e-4f, -2.5e-4f, 0.0f};
    for (int case_idx = 0; case_idx < 3; case_idx++) {
        place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 40.0f, speeds[case_idx]);
        Agent *agent = slot0_agent(&env);
        agent->reward_coefs[REWARD_COEF_COLLISION] = 1.5f;
        agent->reward_coefs[REWARD_COEF_OFFROAD] = 1.5f;
        agent->reward_coefs[REWARD_COEF_STOP_LINE] = 1.0f;
        float before = env.logs[0].reward_wait;
        step_keep(&env, 1);
        printf("  wait penalty at %.1f m/s: %.6f\n", speeds[case_idx], env.logs[0].reward_wait - before);
        EXPECT_NEAR(env.logs[0].reward_wait - before, expected[case_idx], 2e-5f);
    }
    // a 5 m/s full speed also charges slow driving: half at 2.5 m/s, nothing from 5 m/s on
    env.reward_wait_full_speed_mps = 5.0f;
    float slow_speeds[3] = {0.0f, 2.5f, 7.5f};
    float slow_expected[3] = {-5e-4f, -2.5e-4f, 0.0f};
    for (int case_idx = 0; case_idx < 3; case_idx++) {
        place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 40.0f, slow_speeds[case_idx]);
        Agent *agent = slot0_agent(&env);
        agent->reward_coefs[REWARD_COEF_COLLISION] = 1.5f;
        agent->reward_coefs[REWARD_COEF_OFFROAD] = 1.5f;
        agent->reward_coefs[REWARD_COEF_STOP_LINE] = 1.0f;
        float before = env.logs[0].reward_wait;
        step_keep(&env, 1);
        printf(
            "  wait penalty at %.1f m/s with full speed 5 m/s: %.6f\n",
            slow_speeds[case_idx],
            env.logs[0].reward_wait - before);
        EXPECT_NEAR(env.logs[0].reward_wait - before, slow_expected[case_idx], 2e-5f);
    }
    free_allocated(&env);
    Drive light_env = make_lattice_env(TOWN06, 1, 0.3f);
    light_env.reward_wait_penalty_frac = 5e-4f;
    light_env.lattice.light_in_view = 1;
    int light_lane = -1, approach_lane = -1;
    for (int lane_idx = 0; lane_idx < light_env.num_road_elements && light_lane < 0; lane_idx++) {
        const struct LatticeLaneInfo *info = &light_env.lattice_lanes[lane_idx];
        if (!is_drivable_road_lane(light_env.road_elements[lane_idx].type) || info->traffic_light_idx < 0 || info->predecessor_count == 0) {
            continue;
        }
        int predecessor = info->predecessors[0];
        if (light_env.lattice_lanes[predecessor].exit_slots[0] == lane_idx && light_env.lattice_lanes[predecessor].length_m > 25.0f) {
            light_lane = lane_idx;
            approach_lane = predecessor;
        }
    }
    EXPECT_TRUE(light_lane >= 0);
    TrafficControlElement *light = &light_env.traffic_elements[light_env.lattice_lanes[light_lane].traffic_light_idx];
    int states[2] = {TRAFFIC_CONTROL_STATE_RED, TRAFFIC_CONTROL_STATE_GREEN};
    float light_expected[2] = {0.0f, -5e-4f};
    for (int case_idx = 0; case_idx < 2; case_idx++) {
        place_lattice_agent(&light_env, 0, approach_lane, light_env.lattice_lanes[approach_lane].length_m - 20.0f, 0.0f);
        Agent *agent = slot0_agent(&light_env);
        agent->reward_coefs[REWARD_COEF_COLLISION] = 1.5f;
        agent->reward_coefs[REWARD_COEF_OFFROAD] = 1.5f;
        agent->reward_coefs[REWARD_COEF_STOP_LINE] = 1.0f;
        for (int t = 0; t < light->state_size; t++) {
            light->states[t] = states[case_idx];
        }
        float before = light_env.logs[0].reward_wait;
        step_keep(&light_env, 1);
        EXPECT_NEAR(light_env.logs[0].reward_wait - before, light_expected[case_idx], 2e-5f);
    }
    free_allocated(&light_env);
    return 0;
}

// oncoming-lane overtaking: profiles, one-lane scene on Town01, menu layout with the sixth lateral choice
#define TOWN01 DRIVE_TEST_REPO_ROOT "/pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin"
#define ONCOMING_CHOICE 5
#define CENTRE_CHOICE 2

static Drive make_overtake_env(const char *map_file, int num_agents) {
    Drive env = drive_test_env_config(map_file, SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    env.action_type = ACTION_TYPE_LATTICE;
    env.dt = 0.3f;
    env.scenario_length = 5000;
    lattice_test_default_menu(&env.lattice, 0.3f);
    env.lattice.oncoming_overtake = 1;
    allocate(&env);
    c_reset(&env);
    return env;
}

// a long straight non-connector lane ending at a split, with an oncoming lane beside it over its first 100 m
static int find_overtake_lane(const Drive *env) {
    for (int lane_idx = 0; lane_idx < env->num_road_elements; lane_idx++) {
        const struct LatticeLaneInfo *info = &env->lattice_lanes[lane_idx];
        if (!is_drivable_road_lane(env->road_elements[lane_idx].type) || info->is_connector || info->length_m < 120.0f
            || info->max_curvature > 0.002f || info->exit_count < 2) {
            continue;
        }
        int beside = 1;
        for (float arc_m = 0.0f; arc_m <= 100.0f && beside; arc_m += LATTICE_PROFILE_SPACING_M) {
            beside = lattice_profile_at(env, lane_idx, arc_m)->oncoming_lane >= 0;
        }
        if (beside) {
            return lane_idx;
        }
    }
    return -1;
}

static void set_overtake_reward_coefs(Agent *agent) {
    agent->reward_coefs[REWARD_COEF_LANE_ALIGN] = 0.025f;
    agent->reward_coefs[REWARD_COEF_VEL_ALIGN] = 1.0f;
    agent->reward_coefs[REWARD_COEF_VELOCITY] = 2.5e-3f;
    agent->reward_coefs[REWARD_COEF_LANE_CENTER] = 0.001f;
    agent->reward_coefs[REWARD_COEF_CENTER_BIAS] = 0.0f;
    agent->reward_coefs[REWARD_COEF_COLLISION] = 1.5f;
    agent->reward_coefs[REWARD_COEF_OFFROAD] = 1.5f;
    agent->reward_coefs[REWARD_COEF_STOP_LINE] = 1.0f;
}

static int test_oncoming_menu_layout(void) {
    Drive env = make_overtake_env(TOWN06, 1);
    const struct LatticeConfig *cfg = &env.lattice;
    EXPECT_EQ_INT(cfg->nvec[1], 24);
    EXPECT_EQ_INT(cfg->mask_feature_count, 93);
    EXPECT_EQ_INT(lattice_lat_choice_count(cfg), 6);
    EXPECT_TRUE(lattice_lat_is_oncoming(cfg, 1 * 6 + ONCOMING_CHOICE));
    EXPECT_TRUE(!lattice_lat_is_oncoming(cfg, 1 * 6 + 4));
    EXPECT_EQ_INT(lattice_lat_lane_side(cfg, 1 * 6 + 4), 1);
    EXPECT_EQ_INT(lattice_lat_lane_side(cfg, 1 * 6 + ONCOMING_CHOICE), 0);
    EXPECT_EQ_INT(lattice_lat_cell_duration_idx(cfg, 3 * 6 + ONCOMING_CHOICE), 3);
    EXPECT_NEAR(lattice_lat_cell_offset(cfg, 1 * 6 + CENTRE_CHOICE), 0.0f, 0.0f);
    EXPECT_EQ_INT(lattice_plan_feature_count(cfg), LATTICE_PLAN_FEATURES + 1);
    EXPECT_EQ_INT(compute_observation_size(&env) - (EGO_FEATURES + PARTNER_FEATURES * env.obs_slots_partners_n + LANE_FEATURES * env.obs_slots_lane_kept
        + BOUNDARY_FEATURES * env.obs_slots_boundary_kept + TRAFFIC_CONTROL_FEATURES * env.obs_slots_traffic_controls_n + OBS_VALID_COUNT_FEATURES
        + env.num_goals * GOAL_FEATURES), LATTICE_PLAN_FEATURES + 1 + 93);
    free_allocated(&env);
    return 0;
}

// oncoming lanes exist only where one lane runs each way: everywhere on Town01, never on the Town06 highway
static int test_oncoming_profiles(void) {
    for (size_t town_idx = 0; town_idx < sizeof(CARLA_TOWNS) / sizeof(CARLA_TOWNS[0]); town_idx++) {
        char path[512];
        carla_town_path(path, sizeof path, CARLA_TOWNS[town_idx]);
        Drive env = make_overtake_env(path, 1);
        int with_oncoming = 0;
        for (int sample_idx = 0; sample_idx < env.lattice_profile_count; sample_idx++) {
            const struct LatticeProfileSample *sample = &env.lattice_profiles[sample_idx];
            if (sample->oncoming_lane < 0) {
                EXPECT_NEAR(sample->oncoming_offset_m, 0.0f, 0.0f);
                continue;
            }
            with_oncoming++;
            EXPECT_TRUE(sample->neighbour_lane[0] < 0 && sample->neighbour_lane[1] < 0);
            EXPECT_TRUE(sample->oncoming_offset_m >= LATTICE_NEIGHBOUR_MIN_M && sample->oncoming_offset_m <= LATTICE_NEIGHBOUR_MAX_M);
            EXPECT_TRUE(sample->edge_left_m >= sample->oncoming_offset_m);
        }
        float share = (float) with_oncoming / (float) (env.lattice_profile_count > 0 ? env.lattice_profile_count : 1);
        printf("  %s: oncoming lane beside %.1f%% of profile samples\n", CARLA_TOWNS[town_idx], 100.0f * share);
        if (strcmp(CARLA_TOWNS[town_idx], "Town01") == 0) {
            EXPECT_TRUE(share > 0.8f);
        }
        if (strcmp(CARLA_TOWNS[town_idx], "Town06") == 0) {
            EXPECT_EQ_INT(with_oncoming, 0);
        }
        free_allocated(&env);
    }
    return 0;
}

// Town01, stopped car 35 m ahead at 6 m/s: borrow, pass, return; the reward always sees the car's own lane
static int test_oncoming_overtake_stopped_car(void) {
    Drive env = make_overtake_env(TOWN01, 2);
    env.reward_oncoming_penalty_frac = 5e-4f;
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(lane >= 0);
    float offset_m = lattice_profile_at(&env, lane, 30.0f)->oncoming_offset_m;
    place_lattice_agent(&env, 1, lane, 45.0f, 0.0f);
    place_lattice_agent(&env, 0, lane, 10.0f, 6.0f);
    Agent *ego = slot0_agent(&env);
    Agent *blocker = &env.agents[env.active_agent_indices[1]];
    set_overtake_reward_coefs(ego);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    int lat_cells = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL);
    int choice_count = lattice_lat_choice_count(cfg);
    printf("  lane %d, oncoming offset %.2f m, oncoming cells by duration %d %d %d %d\n", lane, offset_m,
           lattice_agent->mask[lat_cells + ONCOMING_CHOICE], lattice_agent->mask[lat_cells + choice_count + ONCOMING_CHOICE],
           lattice_agent->mask[lat_cells + 2 * choice_count + ONCOMING_CHOICE], lattice_agent->mask[lat_cells + 3 * choice_count + ONCOMING_CHOICE]);
    EXPECT_TRUE(lattice_agent->mask[lat_cells + choice_count + ONCOMING_CHOICE]);
    step_with_action(&env, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    EXPECT_NEAR(lattice_agent->lat.target_d_m, offset_m, 0.05f);
    EXPECT_NEAR(lattice_agent->counters.oncoming_starts, 1.0f, 0.0f);
    int plan_flag_idx = EGO_FEATURES + LATTICE_PLAN_FEATURES;
    int returned = 0, borrow_steps = 0, collided = 0, wrong_lane_steps = 0;
    float max_d = 0.0f, min_borrow_align = 1e9f, min_borrow_velocity = 1e9f, borrow_cost = 0.0f;
    for (int step = 0; step < 80; step++) {
        Log before = env.logs[0];
        LatticeFrenet ego_frenet = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
        LatticeFrenet blocker_frenet = lattice_frenet_state(&lattice_agent->rail, blocker, -1);
        if (!returned && ego_frenet.s > blocker_frenet.s + blocker->sim_length + 6.0f) {
            EXPECT_TRUE(lattice_agent->mask[lat_cells + choice_count + CENTRE_CHOICE]);
            step_with_action(&env, 1, choice_count + CENTRE_CHOICE, 0, 0, 0);
            returned = 1;
        } else {
            step_keep(&env, 1);
        }
        LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
        int borrowing = lattice_is_borrowing(&env, lattice_agent, &frenet);
        max_d = fmaxf(max_d, frenet.d);
        collided |= ego->metrics_array[COLLISION_IDX] > 0.0f || ego->stopped;
        wrong_lane_steps += ego->current_lane_idx != lattice_agent->rail.lanes[lattice_agent->rail.chain_slot[frenet.sample_idx]];
        EXPECT_NEAR(env.observations[plan_flag_idx], (float) borrowing, 0.0f);
        if (!borrowing) {
            continue;
        }
        borrow_steps++;
        EXPECT_NEAR(ego->metrics_array[LANE_ANGLE_IDX], cosf(frenet.heading_error), 1e-4f);
        EXPECT_TRUE(ego->metrics_array[LANE_ANGLE_IDX] > 0.9f);
        min_borrow_align = fminf(min_borrow_align, env.logs[0].reward_lane_align - before.reward_lane_align);
        min_borrow_velocity = fminf(min_borrow_velocity, env.logs[0].reward_velocity - before.reward_velocity);
        borrow_cost += env.logs[0].reward_oncoming - before.reward_oncoming;
        EXPECT_NEAR(env.logs[0].reward_oncoming - before.reward_oncoming, -5e-4f, 1e-7f);
    }
    LatticeFrenet end = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
    printf("  overtake: max d %.2f m, %d borrowing steps (cost %.4f), min align %.6f, min velocity %.6f, end d %.3f, s past blocker %.1f m\n",
           max_d, borrow_steps, borrow_cost, min_borrow_align, min_borrow_velocity, end.d,
           end.s - lattice_frenet_state(&lattice_agent->rail, blocker, -1).s);
    EXPECT_TRUE(returned);
    EXPECT_TRUE(!collided);
    EXPECT_EQ_INT(wrong_lane_steps, 0);
    EXPECT_TRUE(max_d > offset_m - 0.2f);
    EXPECT_TRUE(borrow_steps > 10);
    EXPECT_TRUE(min_borrow_align >= 0.0f);
    EXPECT_TRUE(min_borrow_velocity > 0.0f);
    EXPECT_TRUE(fabsf(end.d) < 0.1f);
    EXPECT_NEAR(lattice_agent->counters.oncoming_steps, (float) borrow_steps, 2.0f);
    EXPECT_NEAR(lattice_agent->counters.oncoming_passes, 1.0f, 0.0f);
    free_allocated(&env);
    // the same borrow turned back to the lane centre (6.0 s) one step in never reaches the stopped car: no pass
    Drive early = make_overtake_env(TOWN01, 2);
    place_lattice_agent(&early, 1, lane, 75.0f, 0.0f);
    place_lattice_agent(&early, 0, lane, 10.0f, 6.0f);
    struct LatticeAgent *early_agent = &early.lattice_agents[0];
    step_with_action(&early, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    step_with_action(&early, 1, 3 * choice_count + CENTRE_CHOICE, 0, 0, 0);
    for (int step = 0; step < 20; step++) {
        step_keep(&early, 1);
    }
    printf(
        "  borrow turned back after one step: oncoming starts %.0f, borrowing steps %.0f, passes %.0f\n",
        early_agent->counters.oncoming_starts,
        early_agent->counters.oncoming_steps,
        early_agent->counters.oncoming_passes);
    EXPECT_NEAR(early_agent->counters.oncoming_starts, 1.0f, 0.0f);
    EXPECT_NEAR(early_agent->counters.oncoming_steps, 0.0f, 0.0f);
    EXPECT_NEAR(early_agent->counters.oncoming_passes, 0.0f, 0.0f);
    free_allocated(&early);
    // one borrow past two stopped cars in a row counts two passes
    Drive platoon = make_overtake_env(TOWN01, 3);
    place_lattice_agent(&platoon, 1, lane, 45.0f, 0.0f);
    Agent *front = &platoon.agents[platoon.active_agent_indices[1]];
    place_lattice_agent(&platoon, 2, lane, 45.0f + front->sim_length + 3.0f, 0.0f);
    place_lattice_agent(&platoon, 0, lane, 10.0f, 6.0f);
    Agent *platoon_ego = slot0_agent(&platoon);
    Agent *last = &platoon.agents[platoon.active_agent_indices[2]];
    struct LatticeAgent *platoon_agent = &platoon.lattice_agents[0];
    step_with_action(&platoon, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    int platoon_returned = 0;
    for (int step = 0; step < 100 && !platoon_ego->stopped; step++) {
        LatticeFrenet ego_frenet
            = lattice_frenet_state(&platoon_agent->rail, platoon_ego, platoon_agent->projection_hint);
        LatticeFrenet last_frenet = lattice_frenet_state(&platoon_agent->rail, last, -1);
        if (!platoon_returned && ego_frenet.s > last_frenet.s + last->sim_length + 6.0f) {
            step_with_action(&platoon, 1, choice_count + CENTRE_CHOICE, 0, 0, 0);
            platoon_returned = 1;
        } else {
            step_keep(&platoon, 1);
        }
    }
    printf(
        "  borrow past two stopped cars: returned %d, collision %.0f, passes %.0f\n",
        platoon_returned,
        platoon_ego->metrics_array[COLLISION_IDX],
        platoon_agent->counters.oncoming_passes);
    EXPECT_TRUE(platoon_returned);
    EXPECT_TRUE(!platoon_ego->stopped);
    EXPECT_NEAR(platoon_agent->counters.oncoming_passes, 2.0f, 0.0f);
    free_allocated(&platoon);
    return 0;
}

// stacking diagnostics: queued behind a live and a frozen car, slow-following, a crash just after a borrow
static int test_stacking_counters(void) {
    Drive env = make_overtake_env(TOWN01, 2);
    int lane = find_overtake_lane(&env);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    place_lattice_agent(&env, 1, lane, 45.0f, 0.0f);
    Agent *blocker = &env.agents[env.active_agent_indices[1]];
    place_lattice_agent(&env, 0, lane, 45.0f - blocker->sim_length - 3.0f, 0.0f);
    step_keep(&env, 5);
    float live_queued = lattice_agent->counters.queued_steps, live_frozen = lattice_agent->counters.queued_frozen_steps;
    blocker->stopped = 1;
    step_keep(&env, 5);
    float frozen_queued = lattice_agent->counters.queued_frozen_steps - live_frozen;
    printf(
        "  queued behind a live car %.0f of 5 steps (frozen %.0f), behind a frozen car %.0f of 5\n",
        live_queued,
        live_frozen,
        frozen_queued);
    EXPECT_NEAR(live_queued, 5.0f, 0.0f);
    EXPECT_NEAR(live_frozen, 0.0f, 0.0f);
    EXPECT_NEAR(frozen_queued, 5.0f, 0.0f);
    free_allocated(&env);
    Drive slow = make_overtake_env(TOWN01, 2);
    struct LatticeAgent *slow_agent = &slow.lattice_agents[0];
    place_lattice_agent(&slow, 1, lane, 60.0f, 0.0f);
    place_lattice_agent(&slow, 0, lane, 40.0f, 2.5f);
    int cell = 2 * slow.lattice.lon_speed_count + 2;
    step_with_action(&slow, 0, 0, 1, cell, 0);
    step_keep(&slow, 4);
    printf(
        "  slow-following at 2.5 m/s, car 20 m ahead: %.0f of 5 steps, queued %.0f\n",
        slow_agent->counters.slow_follow_steps,
        slow_agent->counters.queued_steps);
    EXPECT_NEAR(slow_agent->counters.slow_follow_steps, 5.0f, 0.0f);
    EXPECT_NEAR(slow_agent->counters.queued_steps, 0.0f, 0.0f);
    free_allocated(&slow);
    // a borrow turned back while still behind a stopped car, at constant speed, ends in that car: a borrow collision
    Drive crash = make_overtake_env(TOWN01, 2);
    struct LatticeAgent *crash_agent = &crash.lattice_agents[0];
    int choice_count = lattice_lat_choice_count(&crash.lattice);
    place_lattice_agent(&crash, 1, lane, 45.0f, 0.0f);
    place_lattice_agent(&crash, 0, lane, 10.0f, 6.0f);
    step_with_action(&crash, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    for (int step = 1; step < 40 && slot0_agent(&crash)->metrics_array[COLLISION_IDX] == 0.0f; step++) {
        if (step == 10) {
            step_with_action(&crash, 1, choice_count + CENTRE_CHOICE, 0, 0, 0);
        } else {
            step_keep(&crash, 1);
        }
    }
    printf(
        "  borrow turned back behind a stopped car: borrowing steps %.0f, collision %.0f, borrow collisions %.0f\n",
        crash_agent->counters.oncoming_steps,
        slot0_agent(&crash)->metrics_array[COLLISION_IDX],
        crash_agent->counters.oncoming_collisions);
    EXPECT_TRUE(crash_agent->counters.oncoming_steps > 0.0f);
    EXPECT_NEAR(crash_agent->counters.oncoming_collisions, 1.0f, 0.0f);
    free_allocated(&crash);
    return 0;
}

// borrowing toward the split at the lane's end: keep is masked once the 4 s window reaches the junction, return lands first
static int test_oncoming_forced_return_before_junction(void) {
    Drive env = make_overtake_env(TOWN01, 1);
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(lane >= 0);
    place_lattice_agent(&env, 0, lane, 10.0f, 6.0f);
    Agent *ego = slot0_agent(&env);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    int lat_gate = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_GATE);
    int lat_cells = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL);
    int choice_count = lattice_lat_choice_count(cfg);
    step_with_action(&env, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    int forced_step = -1;
    float forced_gap_m = 0.0f;
    for (int step = 0; step < 120 && forced_step < 0; step++) {
        if (!lattice_agent->mask[lat_gate + LATTICE_GATE_KEEP]) {
            forced_step = step;
            LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
            const struct LatticeRail *rail = &lattice_agent->rail;
            forced_gap_m = 1e9f;
            for (int lane_slot = rail->chain_slot[frenet.sample_idx]; lane_slot < rail->lane_count; lane_slot++) {
                if (env.lattice_lanes[rail->lanes[lane_slot]].is_connector) {
                    forced_gap_m = rail->lane_start_s_m[lane_slot] - frenet.s;
                    break;
                }
            }
            for (int duration_idx = 0; duration_idx < cfg->lat_duration_count; duration_idx++) {
                EXPECT_TRUE(!lattice_agent->mask[lat_cells + duration_idx * choice_count + ONCOMING_CHOICE]);
            }
            EXPECT_TRUE(lattice_agent->mask[lat_gate + LATTICE_GATE_NEW]);
            break;
        }
        step_keep(&env, 1);
    }
    printf("  forced return after %d steps, %.1f m before the junction (window %.1f m)\n", forced_step, forced_gap_m,
           fmaxf(LATTICE_ONCOMING_KEEP_MIN_M, LATTICE_ONCOMING_KEEP_S * ego->sim_speed));
    EXPECT_TRUE(forced_step > 0);
    EXPECT_TRUE(forced_gap_m <= fmaxf(LATTICE_ONCOMING_KEEP_MIN_M, LATTICE_ONCOMING_KEEP_S * ego->sim_speed) + LATTICE_PROFILE_SPACING_M);
    int return_cell = -1;
    for (int duration_idx = 0; duration_idx < cfg->lat_duration_count && return_cell < 0; duration_idx++) {
        int cell = duration_idx * choice_count + CENTRE_CHOICE;
        return_cell = lattice_agent->mask[lat_cells + cell] ? cell : -1;
    }
    EXPECT_TRUE(return_cell >= 0);
    step_with_action(&env, 1, return_cell, 0, 0, 0);
    EXPECT_TRUE(lattice_agent->mask[lat_gate + LATTICE_GATE_KEEP]);
    float d_at_junction = 1e9f;
    for (int step = 0; step < 60; step++) {
        step_keep(&env, 1);
        LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
        if (env.lattice_lanes[lattice_agent->rail.lanes[lattice_agent->rail.chain_slot[frenet.sample_idx]]].is_connector) {
            d_at_junction = frenet.d;
            break;
        }
    }
    printf("  offset entering the junction %.3f m\n", d_at_junction);
    EXPECT_TRUE(fabsf(d_at_junction) < LATTICE_BORROW_MIN_D_M);
    free_allocated(&env);
    return 0;
}

// random valid actions with the sixth choice in every town: finite observations, no invalid actions, deterministic
static int test_oncoming_random_rollouts(void) {
    float starts = 0.0f, borrow_steps = 0.0f, steps = 0.0f, invalid = 0.0f;
    for (size_t town_idx = 0; town_idx < sizeof(CARLA_TOWNS) / sizeof(CARLA_TOWNS[0]); town_idx++) {
        char path[512];
        carla_town_path(path, sizeof path, CARLA_TOWNS[town_idx]);
        Drive env = make_overtake_env(path, 16);
        Drive twin = make_overtake_env(path, 16);
        env.reward_oncoming_penalty_frac = 5e-4f;
        twin.reward_oncoming_penalty_frac = 5e-4f;
        unsigned long long rng_state = 21 + town_idx, twin_rng = 21 + town_idx;
        int obs_size = compute_observation_size(&env);
        for (int step = 0; step < 300; step++) {
            random_valid_actions(&env, &rng_state);
            random_valid_actions(&twin, &twin_rng);
            c_step(&env);
            c_step(&twin);
            for (int value = 0; value < env.active_agent_count * obs_size; value++) {
                EXPECT_FINITE(env.observations[value]);
            }
            EXPECT_TRUE(memcmp(env.observations, twin.observations, env.active_agent_count * obs_size * sizeof(float)) == 0);
        }
        for (int active_idx = 0; active_idx < env.active_agent_count; active_idx++) {
            const struct LatticeCounters *counters = &env.lattice_agents[active_idx].counters;
            starts += counters->oncoming_starts;
            borrow_steps += counters->oncoming_steps;
            steps += counters->steps;
            invalid += counters->invalid_actions;
        }
        free_allocated(&env);
        free_allocated(&twin);
    }
    printf("  random rollouts with overtaking: %.0f borrow starts, borrowing %.2f%% of steps, invalid %.0f\n", starts, 100.0f * borrow_steps / fmaxf(steps, 1.0f), invalid);
    EXPECT_TRUE(starts > 0.0f);
    EXPECT_TRUE(borrow_steps > 0.0f);
    EXPECT_NEAR(invalid, 0.0f, 0.0f);
    return 0;
}

// turning around (env.lattice.turnaround): menu, planner, execution on Town01, offer rules, random rollouts
static Drive make_turn_env(const char *map_file, int num_agents, int overtake) {
    Drive env = drive_test_env_config(map_file, SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    env.action_type = ACTION_TYPE_LATTICE;
    env.dt = 0.3f;
    env.scenario_length = 100000;
    lattice_test_default_menu(&env.lattice, 0.3f);
    env.lattice.turnaround = 1;
    env.lattice.oncoming_overtake = overtake;
    allocate(&env);
    c_reset(&env);
    return env;
}

// at rest on lane at arc with the given size; returns the turn cell's mask bit
static int place_turn_car(Drive *env, int active_idx, int lane, float arc_m, float length_m, float width_m) {
    place_lattice_agent(env, active_idx, lane, arc_m, 0.0f);
    Agent *agent = &env->agents[env->active_agent_indices[active_idx]];
    agent->sim_length = length_m;
    agent->sim_width = width_m;
    agent->wheelbase = 0.6f * length_m;
    update_agent_z(env, agent);
    copy_pose_to_prev(agent);
    reset_lattice_state(env);
    compute_observations(env);
    const struct LatticeConfig *cfg = &env->lattice;
    return env->lattice_agents[active_idx].mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + cfg->lon_turn_cell];
}

static int test_turn_menu_layout(void) {
    Drive env = make_turn_env(TOWN06, 1, 0);
    const struct LatticeConfig *cfg = &env.lattice;
    EXPECT_EQ_INT(cfg->nvec[1], 20);
    EXPECT_EQ_INT(cfg->nvec[3], 61);
    EXPECT_EQ_INT(cfg->lon_turn_cell, 60);
    EXPECT_EQ_INT(cfg->mask_feature_count, 90);
    EXPECT_EQ_INT(lattice_plan_feature_count(cfg), LATTICE_PLAN_FEATURES + LATTICE_TURN_PLAN_FEATURES);
    free_allocated(&env);
    Drive both = make_turn_env(TOWN06, 1, 1);
    EXPECT_EQ_INT(both.lattice.nvec[1], 24);
    EXPECT_EQ_INT(both.lattice.nvec[3], 61);
    EXPECT_EQ_INT(both.lattice.mask_feature_count, 94);
    EXPECT_EQ_INT(lattice_plan_feature_count(&both.lattice), LATTICE_PLAN_FEATURES + 1 + LATTICE_TURN_PLAN_FEATURES);
    EXPECT_EQ_INT(
        compute_observation_size(&both)
            - (EGO_FEATURES + PARTNER_FEATURES * both.obs_slots_partners_n + LANE_FEATURES * both.obs_slots_lane_kept
               + BOUNDARY_FEATURES * both.obs_slots_boundary_kept
               + TRAFFIC_CONTROL_FEATURES * both.obs_slots_traffic_controls_n + OBS_VALID_COUNT_FEATURES
               + both.num_goals * GOAL_FEATURES),
        LATTICE_PLAN_FEATURES + 1 + LATTICE_TURN_PLAN_FEATURES + 94);
    free_allocated(&both);
    Drive off = make_lattice_env(TOWN06, 1, 0.3f);
    EXPECT_EQ_INT(off.lattice.lon_turn_cell, -1);
    EXPECT_EQ_INT(off.lattice.nvec[3], 60);
    free_allocated(&off);
    return 0;
}

// legs alternate gear, rotate one way, drive pi / k in total; one-shot for small cars, none for trucks on 8 m roads
static int test_turn_planner_cases(void) {
    Drive env = make_turn_env(TOWN01, 1, 0);
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(lane >= 0);
    const float sizes[3][2] = {{2.0f, 1.5f}, {4.5f, 1.8f}, {5.5f, 2.5f}};
    int legs[3];
    for (int size_idx = 0; size_idx < 3; size_idx++) {
        int offered = place_turn_car(&env, 0, lane, 40.0f, sizes[size_idx][0], sizes[size_idx][1]);
        const struct LatticeTurnPlan *plan = &env.lattice_agents[0].turn.cache_plan;
        legs[size_idx] = offered ? plan->leg_count : 0;
        if (!offered) {
            continue;
        }
        Agent *agent = slot0_agent(&env);
        float driven_m = 0.0f;
        for (int leg_idx = 0; leg_idx < plan->leg_count; leg_idx++) {
            driven_m += plan->legs[leg_idx].length_m;
            EXPECT_TRUE(plan->legs[leg_idx].curvature * plan->legs[leg_idx].gear * plan->sense > 0.0f);
            if (leg_idx > 0) {
                EXPECT_EQ_INT(plan->legs[leg_idx].gear, -plan->legs[leg_idx - 1].gear);
            }
        }
        float target_rotation = fabsf(lattice_wrap_angle(plan->target_heading - agent->sim_heading));
        EXPECT_NEAR(driven_m, target_rotation / fabsf(plan->legs[0].curvature), 0.15f);
        EXPECT_TRUE(cosf(plan->end_heading - agent->sim_heading) < -0.95f);
    }
    printf("  Town01 lane %d: legs small %d, sedan %d, large %d\n", lane, legs[0], legs[1], legs[2]);
    EXPECT_EQ_INT(legs[0], 1);
    EXPECT_TRUE(legs[1] >= 2);
    EXPECT_EQ_INT(legs[2], 0);
    free_allocated(&env);
    Drive highway = make_turn_env(TOWN06, 1, 0);
    int offered = 0;
    for (int lane_idx = 0; lane_idx < highway.num_road_elements && lane_idx < 400; lane_idx += 7) {
        if (!is_drivable_road_lane(highway.road_elements[lane_idx].type) || highway.lattice_lanes[lane_idx].is_connector
            || highway.lattice_lanes[lane_idx].length_m < 30.0f) {
            continue;
        }
        offered += place_turn_car(&highway, 0, lane_idx, 15.0f, 2.0f, 1.5f);
    }
    EXPECT_EQ_INT(offered, 0);
    free_allocated(&highway);
    return 0;
}

// sedan K-turn on Town01: inside the edges every step, keep-only masks, neutral lane rewards, lands aligned on the
// other lane
static int test_turn_execution_town01(void) {
    Drive env = make_turn_env(TOWN01, 1, 0);
    env.reward_wait_penalty_frac = 5e-4f;
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(place_turn_car(&env, 0, lane, 40.0f, 4.5f, 1.8f));
    Agent *agent = slot0_agent(&env);
    set_overtake_reward_coefs(agent);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    struct LatticeTurnEdges edges;
    EXPECT_TRUE(lattice_collect_turn_edges(&env, agent, LATTICE_TURN_RADIUS_COUNT, &edges));
    float start_heading = agent->sim_heading;
    int turn_idx = EGO_FEATURES + LATTICE_PLAN_FEATURES;
    step_with_action(&env, 0, 0, 1, cfg->lon_turn_cell, 0);
    EXPECT_EQ_INT(lattice_agent->turn.active, 1);
    EXPECT_NEAR(lattice_agent->counters.turn_legs, (float) lattice_agent->turn.plan.leg_count, 0.0f);
    int steps = 0, clear_steps = 0, mask_ok = 1, reward_ok = 1;
    float previous_remaining = 2.0f, remaining_rise = 0.0f;
    while (lattice_agent->turn.active && steps < 300) {
        Log before = env.logs[0];
        step_with_action(&env, 1, 3, 1, cfg->lon_emergency_cell, 2);
        steps++;
        LatticePose pose = {agent->sim_x, agent->sim_y, agent->sim_heading};
        clear_steps += lattice_turn_pose_clear(&edges, pose, 0.5f * agent->sim_length, 0.5f * agent->sim_width);
        if (!lattice_agent->turn.active) {
            break;
        }
        for (int factor_idx = 0; factor_idx < LATTICE_ACTION_FACTORS; factor_idx++) {
            int offset = lattice_mask_offset(cfg, factor_idx);
            for (int value = 0; value < cfg->nvec[factor_idx]; value++) {
                mask_ok &= lattice_agent->mask[offset + value] == (value == 0);
            }
        }
        reward_ok &= env.logs[0].reward_lane_align == before.reward_lane_align
            && env.logs[0].reward_lane_center == before.reward_lane_center
            && env.logs[0].reward_reverse == before.reward_reverse && env.logs[0].reward_wait == before.reward_wait;
        float remaining = env.observations[turn_idx + 1];
        EXPECT_NEAR(env.observations[turn_idx], 1.0f, 0.0f);
        EXPECT_NEAR(env.observations[turn_idx + 2], 0.0f, 0.0f);
        remaining_rise = fmaxf(remaining_rise, remaining - previous_remaining);
        previous_remaining = remaining;
    }
    LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    int landing_lane = lattice_agent->rail.lanes[lattice_agent->rail.chain_slot[lattice_agent->projection_hint]];
    printf(
        "  sedan K-turn: %d steps (%.1f s), %d of %d steps clear of edges, heading change %.1f deg, lands on lane %d "
        "at d %.2f cos %.3f, starts %.0f completions %.0f invalid %.0f\n",
        steps,
        steps * env.dt,
        clear_steps,
        steps,
        lattice_wrap_angle(agent->sim_heading - start_heading) * 57.2958f,
        landing_lane,
        frenet.d,
        cosf(frenet.heading_error),
        lattice_agent->counters.turn_starts,
        lattice_agent->counters.turn_completions,
        lattice_agent->counters.invalid_actions);
    EXPECT_EQ_INT(lattice_agent->turn.active, 0);
    EXPECT_EQ_INT(clear_steps, steps);
    EXPECT_TRUE(mask_ok);
    EXPECT_TRUE(reward_ok);
    EXPECT_TRUE(remaining_rise < 0.02f);
    EXPECT_TRUE(agent->metrics_array[OFFROAD_IDX] == 0.0f && !agent->stopped);
    EXPECT_TRUE(cosf(agent->sim_heading - start_heading) < -0.98f);
    EXPECT_TRUE(lattice_agent->has_reference && landing_lane != lane);
    EXPECT_TRUE(cosf(frenet.heading_error) > 0.99f);
    EXPECT_EQ_INT(lattice_agent->gear, 1);
    EXPECT_NEAR(lattice_agent->counters.turn_starts, 1.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.turn_completions, 1.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.turn_aborts, 0.0f, 0.0f);
    EXPECT_TRUE(lattice_agent->counters.invalid_actions >= (float) (steps - 1));
    EXPECT_TRUE(steps * env.dt < 60.0f);
    EXPECT_NEAR(env.observations[turn_idx], 0.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->lat.target_d_m, 0.0f, 0.0f);
    EXPECT_EQ_INT(lattice_agent->turn.landing, fabsf(frenet.d) > LATTICE_TURN_LANDED_D_M);
    EXPECT_EQ_INT(agent->current_lane_idx, landing_lane);
    EXPECT_TRUE(agent->metrics_array[LANE_ANGLE_IDX] > 0.99f);
    EXPECT_NEAR(agent->metrics_array[LANE_DIST_IDX], -frenet.d, 0.05f);
    double landed_sigma_m = lattice_agent->sigma_m;
    int landing_steps = 0;
    float min_align_reward = INFINITY;
    for (int step = 0; step < 60; step++) {
        float align_before = env.logs[0].reward_lane_align;
        step_greedy(&env, 5.0f);
        min_align_reward = fminf(min_align_reward, env.logs[0].reward_lane_align - align_before);
        landing_steps += lattice_agent->turn.landing;
    }
    LatticeFrenet merged = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    printf(
        "  after pulling away %.1f m: d %.2f, offroad %.0f, %d steps on the landing override, min lane-align step "
        "reward %.5f\n",
        lattice_agent->sigma_m - landed_sigma_m,
        merged.d,
        agent->metrics_array[OFFROAD_IDX],
        landing_steps,
        min_align_reward);
    EXPECT_TRUE(fabsf(merged.d) < 0.3f);
    EXPECT_TRUE(agent->metrics_array[OFFROAD_IDX] == 0.0f);
    EXPECT_EQ_INT(lattice_agent->turn.landing, 0);
    EXPECT_TRUE(min_align_reward >= 0.0f);
    free_allocated(&env);
    return 0;
}

// a car the sim stops mid-turn (collision or offroad behaviour) leaves the turn as an abort
static int test_turn_aborts_when_stopped(void) {
    Drive env = make_turn_env(TOWN01, 1, 0);
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(place_turn_car(&env, 0, lane, 40.0f, 4.5f, 1.8f));
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    step_with_action(&env, 0, 0, 1, env.lattice.lon_turn_cell, 0);
    for (int step = 0; step < 15; step++) {
        step_with_action(&env, 0, 0, 0, 0, 0);
    }
    EXPECT_EQ_INT(lattice_agent->turn.active, 1);
    slot0_agent(&env)->stopped = 1;
    step_with_action(&env, 0, 0, 0, 0, 0);
    EXPECT_EQ_INT(lattice_agent->turn.active, 0);
    EXPECT_NEAR(lattice_agent->counters.turn_aborts, 1.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.turn_completions, 0.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

// not offered while moving, with a car parked inside the reach, for a phantom braker, or next to a stop line
static int test_turn_offer_rules(void) {
    Drive env = make_turn_env(TOWN01, 2, 0);
    int lane = find_overtake_lane(&env);
    const struct LatticeConfig *cfg = &env.lattice;
    int turn_bit = lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + cfg->lon_turn_cell;
    place_lattice_agent(&env, 1, lane, 150.0f, 0.0f);
    EXPECT_TRUE(place_turn_car(&env, 0, lane, 40.0f, 2.0f, 1.5f));
    place_lattice_agent(&env, 0, lane, 40.0f, 3.0f);
    EXPECT_EQ_INT(env.lattice_agents[0].mask[turn_bit], 0);
    place_lattice_agent(&env, 1, lane, 46.0f, 0.0f);
    EXPECT_EQ_INT(place_turn_car(&env, 0, lane, 40.0f, 2.0f, 1.5f), 0);
    place_lattice_agent(&env, 1, lane, 150.0f, 0.0f);
    EXPECT_TRUE(place_turn_car(&env, 0, lane, 40.0f, 2.0f, 1.5f));
    slot0_agent(&env)->is_phantom_braker = 1;
    reset_lattice_state(&env);
    compute_observations(&env);
    EXPECT_EQ_INT(env.lattice_agents[0].mask[turn_bit], 0);
    // the plan stays cached at rest but traffic is checked every decision: a car pulling up withdraws the offer
    slot0_agent(&env)->is_phantom_braker = 0;
    EXPECT_TRUE(place_turn_car(&env, 0, lane, 40.0f, 2.0f, 1.5f));
    Agent *other = &env.agents[env.active_agent_indices[1]];
    float far_x = other->sim_x, far_y = other->sim_y, near_x, near_y, near_heading;
    lattice_lane_point_at_arc(&env, lane, 47.0f, &near_x, &near_y, &near_heading);
    float rest_x = slot0_agent(&env)->sim_x;
    other->sim_x = near_x;
    other->sim_y = near_y;
    step_keep(&env, 1);
    EXPECT_NEAR(slot0_agent(&env)->sim_x, rest_x, 0.0f);
    EXPECT_EQ_INT(env.lattice_agents[0].turn.cache_valid, 1);
    EXPECT_EQ_INT(env.lattice_agents[0].mask[turn_bit], 0);
    other->sim_x = far_x;
    other->sim_y = far_y;
    step_keep(&env, 1);
    EXPECT_EQ_INT(env.lattice_agents[0].mask[turn_bit], 1);
    free_allocated(&env);
    Drive lights = make_turn_env(TOWN06, 1, 0);
    int near = 0;
    for (int traffic_idx = 0; traffic_idx < lights.num_traffic_elements && !near; traffic_idx++) {
        const TrafficControlElement *traffic = &lights.traffic_elements[traffic_idx];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        float mid_x = 0.5f * (traffic->stop_line[0] + traffic->stop_line[3]);
        float mid_y = 0.5f * (traffic->stop_line[1] + traffic->stop_line[4]);
        for (int lane_idx = 0; lane_idx < lights.num_road_elements && !near; lane_idx++) {
            if (!is_drivable_road_lane(lights.road_elements[lane_idx].type)
                || lights.lattice_lanes[lane_idx].is_connector) {
                continue;
            }
            float x, y, h;
            float end_arc_m = lights.lattice_lanes[lane_idx].length_m - 6.0f;
            lattice_lane_point_at_arc(&lights, lane_idx, end_arc_m, &x, &y, &h);
            if (hypotf(x - mid_x, y - mid_y) < 8.0f) {
                near = 1;
                EXPECT_EQ_INT(place_turn_car(&lights, 0, lane_idx, end_arc_m, 2.0f, 1.5f), 0);
                EXPECT_TRUE(lattice_stop_line_within(
                    &lights,
                    slot0_agent(&lights),
                    lattice_turn_reach_m(slot0_agent(&lights), lattice_turn_curvature(&lights, slot0_agent(&lights)))));
            }
        }
    }
    EXPECT_TRUE(near);
    free_allocated(&lights);
    return 0;
}

// random valid actions with the turn cell (and overtaking) in every town: finite, deterministic, no invalid actions,
// turns finish
static int test_turn_random_rollouts(void) {
    float starts = 0.0f, completions = 0.0f, aborts = 0.0f, invalid = 0.0f, turn_steps = 0.0f, steps = 0.0f;
    for (size_t town_idx = 0; town_idx < sizeof(CARLA_TOWNS) / sizeof(CARLA_TOWNS[0]); town_idx++) {
        char path[512];
        carla_town_path(path, sizeof path, CARLA_TOWNS[town_idx]);
        Drive env = make_turn_env(path, 16, 1);
        Drive twin = make_turn_env(path, 16, 1);
        env.reward_wait_penalty_frac = 5e-4f;
        twin.reward_wait_penalty_frac = 5e-4f;
        unsigned long long rng_state = 41 + town_idx, twin_rng = 41 + town_idx;
        int obs_size = compute_observation_size(&env);
        for (int step = 0; step < 300; step++) {
            random_valid_actions(&env, &rng_state);
            random_valid_actions(&twin, &twin_rng);
            c_step(&env);
            c_step(&twin);
            for (int value = 0; value < env.active_agent_count * obs_size; value++) {
                EXPECT_FINITE(env.observations[value]);
            }
            EXPECT_TRUE(
                memcmp(env.observations, twin.observations, env.active_agent_count * obs_size * sizeof(float)) == 0);
        }
        for (int active_idx = 0; active_idx < env.active_agent_count; active_idx++) {
            const struct LatticeCounters *counters = &env.lattice_agents[active_idx].counters;
            starts += counters->turn_starts;
            completions += counters->turn_completions;
            aborts += counters->turn_aborts;
            invalid += counters->invalid_actions;
            turn_steps += counters->turn_steps;
            steps += counters->steps;
        }
        free_allocated(&env);
        free_allocated(&twin);
    }
    printf(
        "  random rollouts with turning: %.0f turn starts, %.0f completed, %.0f aborted, %.2f%% of steps turning, "
        "invalid %.0f\n",
        starts,
        completions,
        aborts,
        100.0f * turn_steps / fmaxf(steps, 1.0f),
        invalid);
    EXPECT_TRUE(starts > 0.0f);
    EXPECT_TRUE(completions > 0.0f);
    EXPECT_NEAR(invalid, 0.0f, 0.0f);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_polynomials_and_bellman);
    RUN_TEST(test_menu_layout);
    RUN_TEST(test_exit_slots_and_predecessors_all_towns);
    RUN_TEST(test_rail_geometry_all_towns);
    RUN_TEST(test_long_drive_no_rail_drift);
    RUN_TEST(test_emergency_distances);
    RUN_TEST(test_keep_lane_tracking);
    RUN_TEST(test_stop_cell_exact);
    RUN_TEST(test_backup_cell);
    RUN_TEST(test_lane_change_on_curve);
    RUN_TEST(test_decision_period_at_dt_01);
    RUN_TEST(test_invalid_actions_fall_back);
    RUN_TEST(test_determinism);
    RUN_TEST(test_gear_change_sequence);
    RUN_TEST(test_exit_live_once_per_split);
    RUN_TEST(test_drift_switch_with_hysteresis);
    RUN_TEST(test_no_lane_mode_and_removed_rows);
    RUN_TEST(test_rail_changed_flag_timing_dt_01);
    RUN_TEST(test_plan_change_rms_consistency);
    RUN_TEST(test_route_progress_reward);
    RUN_TEST(test_stop_line_light_features);
    RUN_TEST(test_wait_penalty);
    RUN_TEST(test_random_rollouts_all_towns);
    RUN_TEST(test_oncoming_menu_layout);
    RUN_TEST(test_oncoming_profiles);
    RUN_TEST(test_oncoming_overtake_stopped_car);
    RUN_TEST(test_stacking_counters);
    RUN_TEST(test_oncoming_forced_return_before_junction);
    RUN_TEST(test_oncoming_random_rollouts);
    RUN_TEST(test_turn_menu_layout);
    RUN_TEST(test_turn_planner_cases);
    RUN_TEST(test_turn_execution_town01);
    RUN_TEST(test_turn_aborts_when_stopped);
    RUN_TEST(test_turn_offer_rules);
    RUN_TEST(test_turn_random_rollouts);
    return test_summary(failures);
}
