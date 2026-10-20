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
static void pick_greedy_lon(Drive *env, float speed_set_mps, int *lon_gate, int *lon_cell) {
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
    *lon_gate = pick < 0 || !keep;
    *lon_cell = pick < 0 ? cfg->lon_emergency_cell : (keep ? 0 : pick);
}

static void step_greedy(Drive *env, float speed_set_mps) {
    int lon_gate, lon_cell;
    pick_greedy_lon(env, speed_set_mps, &lon_gate, &lon_cell);
    step_with_action(env, 0, 0, lon_gate, lon_cell, 0);
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

static int lattice_slot_after(const struct LatticeRail *rail, int lane_idx, int next_lane_idx) {
    for (int lane_slot = 0; lane_slot + 1 < rail->lane_count; lane_slot++) {
        if (rail->lanes[lane_slot] == lane_idx && rail->lanes[lane_slot + 1] == next_lane_idx) {
            return lane_slot;
        }
    }
    return -1;
}

// policy exits: a split within the freeze distance stays live until the car reaches it, and the chosen exit can change
static int test_exit_redecidable_until_split(void) {
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
    const struct LatticeRail *rail = &lattice_agent->rail;
    int exit_offset = lattice_mask_offset(cfg, LATTICE_FACTOR_EXIT);
    EXPECT_TRUE(lattice_agent->live_split_slot >= 0);
    EXPECT_TRUE(lattice_agent->mask[exit_offset + 1]);
    int turn_exit = env.lattice_lanes[split_lane].exit_slots[1];
    int straight_exit = env.lattice_lanes[split_lane].exit_slots[0];
    int centre_cell = (cfg->lat_duration_count - 1) * lattice_lat_choice_count(cfg) + 2;
    step_with_action(&env, 0, 0, 0, 0, 1);
    EXPECT_TRUE(lattice_slot_after(rail, split_lane, straight_exit) >= 0);
    EXPECT_NEAR(lattice_agent->counters.exit_decisions, 0.0f, 0.0f);
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL) + centre_cell]);
    step_with_action(&env, 1, centre_cell, 0, 0, 1);
    int split_slot = lattice_slot_after(rail, split_lane, turn_exit);
    EXPECT_TRUE(split_slot >= 0);
    EXPECT_EQ_INT(rail->exit_decided[split_slot], 1);
    EXPECT_EQ_INT(lattice_agent->live_split_slot, split_slot);
    EXPECT_NEAR(lattice_agent->counters.exit_decisions, 1.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.exit_nonstraight, 1.0f, 0.0f);
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL) + centre_cell]);
    step_with_action(&env, 1, centre_cell, 0, 0, 0);
    EXPECT_TRUE(lattice_slot_after(rail, split_lane, straight_exit) >= 0);
    EXPECT_NEAR(lattice_agent->counters.exit_decisions, 1.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.exit_switches, 1.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.exit_nonstraight, 0.0f, 0.0f);
    Agent *agent = slot0_agent(&env);
    int reached_lane = -1;
    for (int step = 0; step < 60 && reached_lane != straight_exit; step++) {
        step_with_action(&env, 0, 0, 0, 0, 0);
        reached_lane = rail->lanes[rail->chain_slot[lattice_agent->projection_hint]];
    }
    printf(
        "  exit re-decided: decisions %.0f, switches %.0f, car reached the straight exit %d, speed %.2f\n",
        lattice_agent->counters.exit_decisions,
        lattice_agent->counters.exit_switches,
        reached_lane == straight_exit,
        agent->sim_speed);
    EXPECT_EQ_INT(reached_lane, straight_exit);
    EXPECT_NEAR(lattice_agent->counters.exit_switches, 1.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

// a turn committed 40 m before the split stays offered while the car drives as fast as its masks allow, and is taken
static int test_committed_exit_stays_offered(void) {
    char path[512];
    carla_town_path(path, sizeof path, "Town03");
    Drive env = make_lattice_env(path, 1, 0.3f);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    int exit_offset = lattice_mask_offset(cfg, LATTICE_FACTOR_EXIT);
    int centre_cell = (cfg->lat_duration_count - 1) * lattice_lat_choice_count(cfg) + 2;
    int tried = 0, reached = 0, revoked = 0;
    for (int lane = 0; lane < env.num_road_elements && tried < 12; lane++) {
        const struct LatticeLaneInfo *info = &env.lattice_lanes[lane];
        if (!is_drivable_road_lane(env.road_elements[lane].type) || info->exit_count < 2 || info->is_connector
            || info->length_m < 45.0f) {
            continue;
        }
        place_lattice_agent(&env, 0, lane, info->length_m - 40.0f, 8.0f);
        if (lattice_agent->live_split_slot < 0 || !lattice_agent->mask[exit_offset + 1]) {
            continue;
        }
        tried++;
        int turn_exit = info->exit_slots[1];
        int on_turn = 0;
        for (int step = 0; step < 80 && !on_turn; step++) {
            int live_here = lattice_agent->live_split_slot >= 0
                && lattice_agent->rail.lanes[lattice_agent->live_split_slot] == lane;
            revoked += live_here && step > 0 && !lattice_agent->mask[exit_offset + 1];
            int lat_ok = lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL) + centre_cell];
            int lon_gate, lon_cell;
            pick_greedy_lon(&env, 20.0f, &lon_gate, &lon_cell);
            step_with_action(&env, live_here && lat_ok, live_here && lat_ok ? centre_cell : 0, lon_gate, lon_cell, 1);
            on_turn = lattice_agent->rail.lanes[lattice_agent->rail.chain_slot[lattice_agent->projection_hint]]
                == turn_exit;
        }
        reached += on_turn;
    }
    printf("  committed turns at greedy speed: tried %d, reached %d, revoked decisions %d\n", tried, reached, revoked);
    EXPECT_TRUE(tried >= 5);
    EXPECT_EQ_INT(revoked, 0);
    EXPECT_EQ_INT(reached, tried);
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

// a light approach on Town06: the controlled lane and its predecessor, longer than min_approach_m
static int find_light_approach(const Drive *env, float min_approach_m, int *approach_lane) {
    for (int lane_idx = 0; lane_idx < env->num_road_elements; lane_idx++) {
        const struct LatticeLaneInfo *info = &env->lattice_lanes[lane_idx];
        if (!is_drivable_road_lane(env->road_elements[lane_idx].type) || info->traffic_light_idx < 0
            || info->predecessor_count == 0) {
            continue;
        }
        int predecessor = info->predecessors[0];
        if (env->lattice_lanes[predecessor].exit_slots[0] == lane_idx
            && env->lattice_lanes[predecessor].length_m > min_approach_m) {
            *approach_lane = predecessor;
            return lane_idx;
        }
    }
    return -1;
}

// the slot state the traffic-control observation shows for this light; -1 when the light is not in a slot
static float light_slot_state(Drive *env, Agent *ego, const TrafficControlElement *light) {
    float slots[16 * TRAFFIC_CONTROL_FEATURES] = {0};
    int count = 0;
    int control_idx[16];
    write_traffic_control_obs(env, ego, slots, 0, &count, control_idx);
    float rel_x, rel_y;
    project_point_to_ego_frame(ego, light->stop_line[0], light->stop_line[1], &rel_x, &rel_y);
    for (int slot = 0; slot < count; slot++) {
        const float *row = &slots[slot * TRAFFIC_CONTROL_FEATURES];
        if (fabsf(row[0] - rel_x / env->obs_norm_xy_offset_m) < 1e-5f
            && fabsf(row[1] - rel_y / env->obs_norm_xy_offset_m) < 1e-5f) {
            return row[6];
        }
    }
    return -1.0f;
}

// obs_light_facing_only: a light's state shows on the approach, not once the car is past its stop line
static int test_light_state_only_when_facing(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    int approach_lane = -1;
    int light_lane = find_light_approach(&env, 25.0f, &approach_lane);
    EXPECT_TRUE(light_lane >= 0);
    TrafficControlElement *light = &env.traffic_elements[env.lattice_lanes[light_lane].traffic_light_idx];
    light->states[env.timestep] = TRAFFIC_CONTROL_STATE_RED;
    EXPECT_EQ_INT(validate_traffic_control_lanes(&env), 0);
    env.obs_light_facing_only = 1;
    place_lattice_agent(&env, 0, approach_lane, env.lattice_lanes[approach_lane].length_m - 20.0f, 0.0f);
    light->states[env.timestep] = TRAFFIC_CONTROL_STATE_RED;
    float approaching = light_slot_state(&env, slot0_agent(&env), light);
    Agent *ego = slot0_agent(&env);
    ego->sim_heading += (float) M_PI;
    float facing_away = light_slot_state(&env, ego, light);
    place_lattice_agent(&env, 0, light_lane, 10.0f, 0.0f);
    light->states[env.timestep] = TRAFFIC_CONTROL_STATE_RED;
    float past_hidden = light_slot_state(&env, slot0_agent(&env), light);
    env.obs_light_facing_only = 0;
    float past_shown = light_slot_state(&env, slot0_agent(&env), light);
    printf(
        "  light slot state: approaching %.0f, turned away %.0f, past the line %.0f (facing only) / %.0f (all)\n",
        approaching,
        facing_away,
        past_hidden,
        past_shown);
    EXPECT_NEAR(approaching, (float) TRAFFIC_CONTROL_STATE_RED, 0.0f);
    EXPECT_NEAR(facing_away, (float) TRAFFIC_CONTROL_STATE_UNKNOWN, 0.0f);
    EXPECT_NEAR(past_hidden, (float) TRAFFIC_CONTROL_STATE_UNKNOWN, 0.0f);
    EXPECT_NEAR(past_shown, (float) TRAFFIC_CONTROL_STATE_RED, 0.0f);
    free_allocated(&env);
    return 0;
}

// with light_in_view the stop-line cell needs the stop line in the traffic-control view
static int test_stop_line_cell_needs_view(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    int approach_lane = -1;
    int light_lane = find_light_approach(&env, 25.0f, &approach_lane);
    EXPECT_TRUE(light_lane >= 0);
    const struct LatticeConfig *cfg = &env.lattice;
    int stop_line_mask = lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + cfg->lon_stop_line_cell;
    int offered[3];
    float ranges[3] = {200.0f, 5.0f, 5.0f};
    int in_view[3] = {1, 1, 0};
    for (int case_idx = 0; case_idx < 3; case_idx++) {
        env.obs_range_traffic_control_m = ranges[case_idx];
        env.lattice.light_in_view = in_view[case_idx];
        place_lattice_agent(&env, 0, approach_lane, env.lattice_lanes[approach_lane].length_m - 20.0f, 3.0f);
        offered[case_idx] = env.lattice_agents[0].mask[stop_line_mask];
    }
    printf(
        "  stop-line cell offered: in view %d, out of view %d, out of view without light_in_view %d\n",
        offered[0],
        offered[1],
        offered[2]);
    EXPECT_EQ_INT(offered[0], 1);
    EXPECT_EQ_INT(offered[1], 0);
    EXPECT_EQ_INT(offered[2], 1);
    free_allocated(&env);
    return 0;
}

// a reported red waives the waiting penalty near its stop line or in a queue, not for a car standing far back alone
static int test_wait_exempt_near_or_queued(void) {
    Drive env = make_lattice_env(TOWN06, 2, 0.3f);
    env.reward_wait_penalty_frac = 5e-4f;
    env.lattice.light_in_view = 1;
    int approach_lane = -1;
    int light_lane = find_light_approach(&env, 2.0f * LATTICE_WAIT_LIGHT_NEAR_M + 20.0f, &approach_lane);
    EXPECT_TRUE(light_lane >= 0);
    TrafficControlElement *light = &env.traffic_elements[env.lattice_lanes[light_lane].traffic_light_idx];
    for (int t = 0; t < light->state_size; t++) {
        light->states[t] = TRAFFIC_CONTROL_STATE_RED;
    }
    float approach_m = env.lattice_lanes[approach_lane].length_m;
    float back_arc_m = approach_m - 2.0f * LATTICE_WAIT_LIGHT_NEAR_M;
    float waits[3];
    for (int case_idx = 0; case_idx < 3; case_idx++) {
        // case 0 alone far back, case 1 far back behind a stopped car 8 m ahead, case 2 alone near the line
        float ego_arc_m = case_idx == 2 ? approach_m - 10.0f : back_arc_m;
        float other_arc_m = case_idx == 1 ? back_arc_m + 8.0f : 5.0f;
        place_lattice_agent(&env, 1, case_idx == 1 ? approach_lane : STRAIGHT_LANE_TOWN06, other_arc_m, 0.0f);
        place_lattice_agent(&env, 0, approach_lane, ego_arc_m, 0.0f);
        Agent *agent = slot0_agent(&env);
        agent->reward_coefs[REWARD_COEF_COLLISION] = 1.5f;
        agent->reward_coefs[REWARD_COEF_OFFROAD] = 1.5f;
        agent->reward_coefs[REWARD_COEF_STOP_LINE] = 1.0f;
        float before = env.logs[0].reward_wait;
        lattice_set_action(&env, 1, 0, 0, 0, 0, 0);
        step_keep(&env, 1);
        waits[case_idx] = env.logs[0].reward_wait - before;
    }
    printf(
        "  wait penalty at a red: alone %.0f m back %.6f, queued there %.6f, alone 10 m before the line %.6f\n",
        2.0f * LATTICE_WAIT_LIGHT_NEAR_M,
        waits[0],
        waits[1],
        waits[2]);
    EXPECT_NEAR(waits[0], -5e-4f, 2e-5f);
    EXPECT_NEAR(waits[1], 0.0f, 0.0f);
    EXPECT_NEAR(waits[2], 0.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

// a plan committed at rest is a start: no consistency penalty, while the RMS metric still records it
static int test_consistency_free_from_rest(void) {
    Drive env = make_lattice_env(TOWN06, 1, 0.3f);
    env.reward_trajectory_consistency = 2e-3f;
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 20.0f, 0.0f);
    Agent *agent = slot0_agent(&env);
    agent->reward_coefs[REWARD_COEF_TRAJECTORY_CONSISTENCY] = 2e-3f;
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    step_keep(&env, 5);
    int start_cell = 1 * cfg->lon_speed_count + 2;
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + start_cell]);
    float before = env.logs[0].reward_trajectory_consistency;
    float rms_before = lattice_agent->counters.plan_change_rms_m;
    step_with_action(&env, 0, 0, 1, start_cell, 0);
    float start_penalty = env.logs[0].reward_trajectory_consistency - before;
    float start_rms = lattice_agent->counters.plan_change_rms_m - rms_before;
    step_keep(&env, 15);
    int change_cell = 2 * cfg->lon_speed_count + 1;
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL) + change_cell]);
    before = env.logs[0].reward_trajectory_consistency;
    step_with_action(&env, 0, 0, 1, change_cell, 0);
    float moving_penalty = env.logs[0].reward_trajectory_consistency - before;
    printf(
        "  consistency: start from rest %.5f (RMS %.2f m), slowing down at %.1f m/s %.5f\n",
        start_penalty,
        start_rms,
        agent->sim_speed,
        moving_penalty);
    EXPECT_NEAR(start_penalty, 0.0f, 0.0f);
    EXPECT_TRUE(start_rms > 0.5f);
    EXPECT_TRUE(moving_penalty < 0.0f);
    free_allocated(&env);
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

// a borrow behind a stopped car keeps its plan until the car is passed though every decision asks to turn back
static int test_overtake_commit_holds_until_passed(void) {
    Drive probe = make_overtake_env(TOWN01, 2);
    int lane = find_overtake_lane(&probe);
    free_allocated(&probe);
    EXPECT_TRUE(lane >= 0);
    float passes[2] = {0}, held_steps[2] = {0}, end_d_m[2] = {0}, collided[2] = {0};
    for (int commit = 0; commit < 2; commit++) {
        Drive env = make_overtake_env(TOWN01, 2);
        env.lattice.overtake_commit = commit;
        place_lattice_agent(&env, 1, lane, 45.0f, 0.0f);
        place_lattice_agent(&env, 0, lane, 10.0f, 6.0f);
        struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
        const struct LatticeConfig *cfg = &env.lattice;
        int choice_count = lattice_lat_choice_count(cfg);
        int lat_gate = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_GATE);
        int lat_cells = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL);
        Agent *ego = slot0_agent(&env);
        step_with_action(&env, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
        EXPECT_EQ_INT(lattice_agent->overtake_target, commit ? env.active_agent_indices[1] : -1);
        for (int step = 0; step < 80 && !ego->stopped; step++) {
            held_steps[commit]
                += lattice_agent->overtake_target >= 0 && !lattice_agent->mask[lat_gate + LATTICE_GATE_NEW];
            int turn_back = lattice_agent->mask[lat_gate + LATTICE_GATE_NEW]
                && lattice_agent->mask[lat_cells + choice_count + CENTRE_CHOICE];
            step_with_action(&env, turn_back, turn_back ? choice_count + CENTRE_CHOICE : 0, 0, 0, 0);
            collided[commit] += ego->metrics_array[COLLISION_IDX] > 0.0f;
        }
        passes[commit] = lattice_agent->counters.oncoming_passes;
        end_d_m[commit] = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint).d;
        printf(
            "  overtake commit %d: commits %.0f, held %.0f steps, completions %.0f, passes %.0f, yields %.0f, abandons "
            "%.0f, collision steps %.0f, end d %.2f m\n",
            commit,
            lattice_agent->counters.overtake_commits,
            held_steps[commit],
            lattice_agent->counters.overtake_completions,
            passes[commit],
            lattice_agent->counters.overtake_yields,
            lattice_agent->counters.overtake_abandons,
            collided[commit],
            end_d_m[commit]);
        EXPECT_NEAR(lattice_agent->counters.overtake_commits, (float) commit, 0.0f);
        EXPECT_NEAR(lattice_agent->counters.overtake_completions, (float) commit, 0.0f);
        EXPECT_NEAR(lattice_agent->counters.overtake_yields, 0.0f, 0.0f);
        EXPECT_NEAR(lattice_agent->counters.overtake_abandons, 0.0f, 0.0f);
        EXPECT_EQ_INT(lattice_agent->overtake_target, -1);
        free_allocated(&env);
    }
    EXPECT_NEAR(passes[0], 0.0f, 0.0f);
    EXPECT_NEAR(passes[1], 1.0f, 0.0f);
    EXPECT_TRUE(held_steps[1] > 10.0f);
    EXPECT_NEAR(collided[1], 0.0f, 0.0f);
    EXPECT_TRUE(fabsf(end_d_m[1]) < 0.3f);
    return 0;
}

// shortest valid lateral cell that returns to the lane centre; -1 when none is offered
static int first_valid_centre_cell(const Drive *env) {
    const struct LatticeConfig *cfg = &env->lattice;
    int lat_cells = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL);
    for (int duration_idx = 0; duration_idx < cfg->lat_duration_count; duration_idx++) {
        int cell = duration_idx * lattice_lat_choice_count(cfg) + CENTRE_CHOICE;
        if (env->lattice_agents[0].mask[lat_cells + cell]) {
            return cell;
        }
    }
    return -1;
}

// a committed overtake yields to a car coming at it; a borrow with that car near or nothing ahead commits to nothing
static int test_overtake_commit_yields(void) {
    Drive env = make_overtake_env(TOWN01, 3);
    env.lattice.overtake_commit = 1;
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(lane >= 0);
    const struct LatticeProfileSample *beside = lattice_profile_at(&env, lane, 95.0f);
    EXPECT_TRUE(beside->turn_lane >= 0);
    place_lattice_agent(&env, 2, beside->turn_lane, beside->turn_arc_m, 8.0f);
    place_lattice_agent(&env, 1, lane, 45.0f, 0.0f);
    place_lattice_agent(&env, 0, lane, 10.0f, 6.0f);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    const struct LatticeConfig *cfg = &env.lattice;
    int choice_count = lattice_lat_choice_count(cfg);
    int lat_gate = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_GATE);
    Agent *ego = slot0_agent(&env);
    Agent *oncoming = &env.agents[env.active_agent_indices[2]];
    step_with_action(&env, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    EXPECT_EQ_INT(lattice_agent->overtake_target, env.active_agent_indices[1]);
    int released_step = -1;
    float collided = 0.0f;
    for (int step = 0; step < 60 && !ego->stopped; step++) {
        released_step = released_step < 0 && lattice_agent->overtake_target < 0 ? step : released_step;
        // steer back with any valid centre cell at speed, then stop behind the stopped car
        int centre_cell = first_valid_centre_cell(&env);
        int turn_back = lattice_agent->mask[lat_gate + LATTICE_GATE_NEW] && centre_cell >= 0;
        float d_m = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint).d;
        int lon_gate, lon_cell;
        pick_greedy_lon(&env, released_step >= 0 && d_m < 0.5f ? 0.0f : 6.0f, &lon_gate, &lon_cell);
        step_with_action(&env, turn_back, turn_back ? centre_cell : 0, lon_gate, lon_cell, 0);
        lattice_set_action(&env, 2, 0, 0, 0, 0, 0);
        collided += ego->metrics_array[COLLISION_IDX] > 0.0f || oncoming->metrics_array[COLLISION_IDX] > 0.0f;
    }
    LatticeFrenet end = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
    printf(
        "  committed overtake meets an oncoming car: released after %d steps, yields %.0f, passes %.0f, collisions "
        "%.0f, end d %.2f m\n",
        released_step,
        lattice_agent->counters.overtake_yields,
        lattice_agent->counters.oncoming_passes,
        collided,
        end.d);
    EXPECT_TRUE(released_step >= 0);
    EXPECT_NEAR(lattice_agent->counters.overtake_yields, 1.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.overtake_completions, 0.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.overtake_abandons, 0.0f, 0.0f);
    EXPECT_NEAR(collided, 0.0f, 0.0f);
    EXPECT_TRUE(end.d < LATTICE_BORROW_MIN_D_M);
    free_allocated(&env);
    Drive threatened = make_overtake_env(TOWN01, 3);
    threatened.lattice.overtake_commit = 1;
    const struct LatticeProfileSample *near = lattice_profile_at(&threatened, lane, 60.0f);
    EXPECT_TRUE(near->turn_lane >= 0);
    place_lattice_agent(&threatened, 2, near->turn_lane, near->turn_arc_m, 8.0f);
    place_lattice_agent(&threatened, 1, lane, 45.0f, 0.0f);
    place_lattice_agent(&threatened, 0, lane, 10.0f, 6.0f);
    step_with_action(&threatened, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    printf(
        "  borrow toward an oncoming car 50 m ahead: borrow starts %.0f, target %d, commits %.0f\n",
        threatened.lattice_agents[0].counters.oncoming_starts,
        threatened.lattice_agents[0].overtake_target,
        threatened.lattice_agents[0].counters.overtake_commits);
    EXPECT_NEAR(threatened.lattice_agents[0].counters.oncoming_starts, 1.0f, 0.0f);
    EXPECT_EQ_INT(threatened.lattice_agents[0].overtake_target, -1);
    EXPECT_NEAR(threatened.lattice_agents[0].counters.overtake_commits, 0.0f, 0.0f);
    free_allocated(&threatened);
    Drive empty = make_overtake_env(TOWN01, 1);
    empty.lattice.overtake_commit = 1;
    place_lattice_agent(&empty, 0, lane, 10.0f, 6.0f);
    step_with_action(&empty, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    printf(
        "  borrow on an empty road: target %d, commits %.0f, new lateral plans offered %d\n",
        empty.lattice_agents[0].overtake_target,
        empty.lattice_agents[0].counters.overtake_commits,
        empty.lattice_agents[0].mask[lattice_mask_offset(&empty.lattice, LATTICE_FACTOR_LAT_GATE) + LATTICE_GATE_NEW]);
    EXPECT_EQ_INT(empty.lattice_agents[0].overtake_target, -1);
    EXPECT_NEAR(empty.lattice_agents[0].counters.overtake_commits, 0.0f, 0.0f);
    EXPECT_EQ_INT(
        empty.lattice_agents[0].mask[lattice_mask_offset(&empty.lattice, LATTICE_FACTOR_LAT_GATE) + LATTICE_GATE_NEW],
        1);
    free_allocated(&empty);
    return 0;
}

// a committed overtake creeping up on a car parked in the oncoming lane yields LATTICE_OVERTAKE_YIELD_GAP_M short of it
static int test_overtake_commit_yields_to_parked_car(void) {
    Drive env = make_overtake_env(TOWN01, 3);
    env.lattice.overtake_commit = 1;
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(lane >= 0);
    const struct LatticeProfileSample *beside = lattice_profile_at(&env, lane, 60.0f);
    EXPECT_TRUE(beside->turn_lane >= 0);
    place_lattice_agent(&env, 2, beside->turn_lane, beside->turn_arc_m, 0.0f);
    place_lattice_agent(&env, 1, lane, 45.0f, 0.0f);
    place_lattice_agent(&env, 0, lane, 10.0f, 6.0f);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    int choice_count = lattice_lat_choice_count(&env.lattice);
    Agent *ego = slot0_agent(&env);
    Agent *parked = &env.agents[env.active_agent_indices[2]];
    step_with_action(&env, 1, choice_count + ONCOMING_CHOICE, 0, 0, 0);
    EXPECT_EQ_INT(lattice_agent->overtake_target, env.active_agent_indices[1]);
    float gap_m = 0.0f, speed_mps = 0.0f, collided = 0.0f;
    int held_steps = 0;
    for (; held_steps < 80 && lattice_agent->overtake_target >= 0; held_steps++) {
        int lon_gate, lon_cell;
        pick_greedy_lon(&env, 2.5f, &lon_gate, &lon_cell);
        step_with_action(&env, 0, 0, lon_gate, lon_cell, 0);
        collided += ego->metrics_array[COLLISION_IDX] > 0.0f;
        LatticeFrenet ego_frenet = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
        gap_m = lattice_frenet_state(&lattice_agent->rail, parked, -1).s - ego_frenet.s
            - 0.5f * (ego->sim_length + parked->sim_length);
        speed_mps = fabsf(ego->sim_speed_signed);
    }
    printf(
        "  committed overtake creeping toward a parked oncoming-lane car: released after %d steps, %.1f m short "
        "at %.1f m/s, yields %.0f, completions %.0f, abandons %.0f, collisions %.0f\n",
        held_steps,
        gap_m,
        speed_mps,
        lattice_agent->counters.overtake_yields,
        lattice_agent->counters.overtake_completions,
        lattice_agent->counters.overtake_abandons,
        collided);
    EXPECT_EQ_INT(lattice_agent->overtake_target, -1);
    EXPECT_NEAR(lattice_agent->counters.overtake_yields, 1.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.overtake_completions, 0.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.overtake_abandons, 0.0f, 0.0f);
    EXPECT_TRUE(gap_m < LATTICE_OVERTAKE_YIELD_GAP_M && gap_m > LATTICE_OVERTAKE_YIELD_GAP_M - 2.0f);
    EXPECT_TRUE(speed_mps * LATTICE_OVERTAKE_YIELD_S < gap_m);
    EXPECT_TRUE(lattice_agent->mask[lattice_mask_offset(&env.lattice, LATTICE_FACTOR_LAT_GATE) + LATTICE_GATE_NEW]);
    EXPECT_NEAR(collided, 0.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

// close ahead in the oncoming lane a standing car is a threat to a slow overtake, a car driving the same way is not
static int test_overtake_threat_standing_or_driving(void) {
    Drive env = make_overtake_env(TOWN01, 2);
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(lane >= 0);
    const struct LatticeProfileSample *beside = lattice_profile_at(&env, lane, 40.0f);
    EXPECT_TRUE(beside->turn_lane >= 0);
    int threat[2] = {0};
    float gap_m = 0.0f;
    for (int driving = 0; driving < 2; driving++) {
        place_lattice_agent(&env, 1, beside->turn_lane, beside->turn_arc_m, 0.0f);
        Agent *other = &env.agents[env.active_agent_indices[1]];
        if (driving) {
            other->sim_heading = lattice_wrap_angle(other->sim_heading + (float) M_PI);
            other->cos_heading = cosf(other->sim_heading);
            other->sin_heading = sinf(other->sim_heading);
            other->sim_vx = 1.5f * other->cos_heading;
            other->sim_vy = 1.5f * other->sin_heading;
            update_agent_speed(other);
            copy_pose_to_prev(other);
        }
        place_lattice_agent(&env, 0, lane, 25.0f, 1.5f);
        struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
        Agent *ego = slot0_agent(&env);
        LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
        gap_m = lattice_frenet_state(&lattice_agent->rail, other, -1).s - frenet.s
            - 0.5f * (ego->sim_length + other->sim_length);
        threat[driving] = lattice_oncoming_threat(&env, lattice_agent, ego, &frenet);
    }
    printf(
        "  oncoming-lane car %.1f m ahead of a car at 1.5 m/s: threat standing %d, driving its way %d\n",
        gap_m,
        threat[0],
        threat[1]);
    EXPECT_TRUE(gap_m < LATTICE_OVERTAKE_YIELD_GAP_M && gap_m > 1.5f * LATTICE_OVERTAKE_YIELD_S);
    EXPECT_EQ_INT(threat[0], 1);
    EXPECT_EQ_INT(threat[1], 0);
    free_allocated(&env);
    return 0;
}

// a commitment level with its slow car outlasts LATTICE_OVERTAKE_COMMIT_MAX_S; one that fell back is abandoned then
static int test_overtake_commit_expiry(void) {
    Drive probe = make_overtake_env(TOWN01, 2);
    int lane = find_overtake_lane(&probe);
    free_allocated(&probe);
    EXPECT_TRUE(lane >= 0);
    int held_steps[2] = {0}, released_step[2] = {-1, -1};
    float collided[2] = {0}, completions[2] = {0}, abandons[2] = {0};
    for (int falls_back = 0; falls_back < 2; falls_back++) {
        Drive env = make_overtake_env(TOWN01, 2);
        env.lattice.overtake_commit = 1;
        place_lattice_agent(&env, 1, lane, 40.0f, 4.0f);
        place_lattice_agent(&env, 0, lane, 10.0f, 6.0f);
        struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
        const struct LatticeConfig *cfg = &env.lattice;
        int lat_gate = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_GATE);
        Agent *ego = slot0_agent(&env);
        step_with_action(&env, 1, lattice_lat_choice_count(cfg) + ONCOMING_CHOICE, 0, 0, 0);
        EXPECT_EQ_INT(lattice_agent->overtake_target, env.active_agent_indices[1]);
        for (int step = 0; step < 90 && !ego->stopped; step++) {
            held_steps[falls_back] += lattice_agent->overtake_target >= 0;
            released_step[falls_back] = released_step[falls_back] < 0 && lattice_agent->overtake_target < 0
                ? step
                : released_step[falls_back];
            int centre_cell = first_valid_centre_cell(&env);
            int turn_back = lattice_agent->mask[lat_gate + LATTICE_GATE_NEW] && centre_cell >= 0
                && (!falls_back || lattice_agent->overtake_target < 0);
            int lon_gate = 0, lon_cell = 0;
            if (falls_back) {
                pick_greedy_lon(&env, 2.5f, &lon_gate, &lon_cell);
            }
            step_with_action(&env, turn_back, turn_back ? centre_cell : 0, lon_gate, lon_cell, 0);
            collided[falls_back] += ego->metrics_array[COLLISION_IDX] > 0.0f;
        }
        completions[falls_back] = lattice_agent->counters.overtake_completions;
        abandons[falls_back] = lattice_agent->counters.overtake_abandons;
        printf(
            "  slow car, %s: held %d steps (%.1f s), released at step %d, completions %.0f, abandons %.0f, "
            "yields %.0f, collision steps %.0f\n",
            falls_back ? "ego falls back" : "ego keeps 6 m/s and asks to turn back",
            held_steps[falls_back],
            (float) held_steps[falls_back] * env.dt,
            released_step[falls_back],
            completions[falls_back],
            abandons[falls_back],
            lattice_agent->counters.overtake_yields,
            collided[falls_back]);
        EXPECT_NEAR(lattice_agent->counters.overtake_yields, 0.0f, 0.0f);
        free_allocated(&env);
    }
    float max_steps = LATTICE_OVERTAKE_COMMIT_MAX_S / 0.3f;
    EXPECT_TRUE((float) held_steps[0] > max_steps);
    EXPECT_NEAR(completions[0], 1.0f, 0.0f);
    EXPECT_NEAR(abandons[0], 0.0f, 0.0f);
    EXPECT_NEAR(collided[0], 0.0f, 0.0f);
    EXPECT_TRUE(fabsf((float) held_steps[1] - max_steps) <= 1.0f);
    EXPECT_NEAR(completions[1], 0.0f, 0.0f);
    EXPECT_NEAR(abandons[1], 1.0f, 0.0f);
    EXPECT_NEAR(collided[1], 0.0f, 0.0f);
    Drive env = make_overtake_env(TOWN01, 2);
    env.lattice.overtake_commit = 1;
    place_lattice_agent(&env, 1, lane, 40.0f, 4.0f);
    place_lattice_agent(&env, 0, lane, 10.0f, 6.0f);
    step_with_action(&env, 1, lattice_lat_choice_count(&env.lattice) + ONCOMING_CHOICE, 0, 0, 0);
    EXPECT_TRUE(env.lattice_agents[0].overtake_target >= 0);
    c_reset(&env);
    EXPECT_EQ_INT(env.lattice_agents[0].overtake_target, -1);
    free_allocated(&env);
    return 0;
}

// a car coming head-on in the car's own lane is neither an overtake target nor counted as passed
static int test_overtake_commit_skips_facing_car(void) {
    Drive env = make_overtake_env(TOWN01, 2);
    env.lattice.overtake_commit = 1;
    int lane = find_overtake_lane(&env);
    EXPECT_TRUE(lane >= 0);
    place_lattice_agent(&env, 1, lane, 55.0f, 4.0f);
    Agent *facing = &env.agents[env.active_agent_indices[1]];
    facing->sim_heading = lattice_wrap_angle(facing->sim_heading + (float) M_PI);
    facing->cos_heading = cosf(facing->sim_heading);
    facing->sin_heading = sinf(facing->sim_heading);
    facing->sim_vx = 4.0f * facing->cos_heading;
    facing->sim_vy = 4.0f * facing->sin_heading;
    update_agent_speed(facing);
    copy_pose_to_prev(facing);
    place_lattice_agent(&env, 0, lane, 10.0f, 6.0f);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    Agent *ego = slot0_agent(&env);
    step_with_action(&env, 1, lattice_lat_choice_count(&env.lattice) + ONCOMING_CHOICE, 0, 0, 0);
    EXPECT_NEAR(lattice_agent->counters.oncoming_starts, 1.0f, 0.0f);
    EXPECT_EQ_INT(lattice_agent->overtake_target, -1);
    EXPECT_EQ_INT(lattice_agent->pass_target, -1);
    float collided = 0.0f;
    for (int step = 0; step < 40; step++) {
        LatticeFrenet ego_frenet = lattice_frenet_state(&lattice_agent->rail, ego, lattice_agent->projection_hint);
        LatticeFrenet facing_frenet = lattice_frenet_state(&lattice_agent->rail, facing, -1);
        int past = ego_frenet.s - 0.5f * ego->sim_length > facing_frenet.s + 0.5f * facing->sim_length + 3.0f;
        int centre_cell = first_valid_centre_cell(&env);
        int turn_back = past && centre_cell >= 0 && lattice_plan_in_oncoming(&env, lattice_agent, &ego_frenet)
            && lattice_agent->mask[lattice_mask_offset(&env.lattice, LATTICE_FACTOR_LAT_GATE) + LATTICE_GATE_NEW];
        step_with_action(&env, turn_back, turn_back ? centre_cell : 0, 0, 0, 0);
        collided += ego->metrics_array[COLLISION_IDX] > 0.0f;
    }
    printf(
        "  borrow past a car coming head-on in the own lane: commits %.0f, passes %.0f, collision steps %.0f\n",
        lattice_agent->counters.overtake_commits,
        lattice_agent->counters.oncoming_passes,
        collided);
    EXPECT_NEAR(lattice_agent->counters.overtake_commits, 0.0f, 0.0f);
    EXPECT_NEAR(lattice_agent->counters.oncoming_passes, 0.0f, 0.0f);
    EXPECT_NEAR(collided, 0.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

// init rejects lattice_overtake_commit outside 0/1 or without lattice_oncoming_overtake
static int test_overtake_commit_config(void) {
    Drive env = make_overtake_env(TOWN01, 1);
    env.lattice.overtake_commit = 1;
    EXPECT_EQ_INT(init_lattice_config(&env), 0);
    env.lattice.oncoming_overtake = 0;
    EXPECT_EQ_INT(init_lattice_config(&env), -1);
    env.lattice.oncoming_overtake = 1;
    env.lattice.overtake_commit = 2;
    EXPECT_EQ_INT(init_lattice_config(&env), -1);
    env.lattice.overtake_commit = 0;
    free_allocated(&env);
    return 0;
}

// a lattice car cruising along its lane earns reward_speed_bonus x dt x its speed's share above the start speed
static int test_speed_bonus_while_cruising(void) {
    float per_step[3] = {0};
    for (int variant = 0; variant < 3; variant++) {
        Drive env = make_overtake_env(TOWN01, 1);
        env.base_max_speed_mps = 20.0f;
        env.reward_speed_bonus = variant > 0 ? 4e-3f : 0.0f;
        env.reward_speed_bonus_from_mps = variant == 2 ? 5.0f : 0.0f;
        int lane = find_overtake_lane(&env);
        EXPECT_TRUE(lane >= 0);
        place_lattice_agent(&env, 0, lane, 10.0f, 10.0f);
        step_keep(&env, 3);
        Log before = env.logs[0];
        step_keep(&env, 10);
        per_step[variant] = (env.logs[0].reward_speed_bonus - before.reward_speed_bonus) / 10.0f;
        EXPECT_NEAR(slot0_agent(&env)->sim_speed, 10.0f, 0.05f);
        free_allocated(&env);
    }
    printf(
        "  speed bonus while cruising at 10 m/s: off %.2e, from 0 %.2e, from 5 m/s %.2e per step\n",
        per_step[0],
        per_step[1],
        per_step[2]);
    EXPECT_NEAR(per_step[0], 0.0f, 0.0f);
    EXPECT_NEAR(per_step[1], 4e-3f * 0.3f * 0.5f, 2e-6f);
    EXPECT_NEAR(per_step[2], 4e-3f * 0.3f / 3.0f, 2e-6f);
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
    float starts = 0.0f, borrow_steps = 0.0f, steps = 0.0f, invalid = 0.0f, commits = 0.0f, completions = 0.0f,
          yields = 0.0f, abandons = 0.0f;
    for (size_t run_idx = 0; run_idx < 2 * sizeof(CARLA_TOWNS) / sizeof(CARLA_TOWNS[0]); run_idx++) {
        size_t town_idx = run_idx / 2;
        char path[512];
        carla_town_path(path, sizeof path, CARLA_TOWNS[town_idx]);
        Drive env = make_overtake_env(path, 16);
        Drive twin = make_overtake_env(path, 16);
        env.lattice.overtake_commit = (int) (run_idx % 2);
        twin.lattice.overtake_commit = (int) (run_idx % 2);
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
            commits += counters->overtake_commits;
            completions += counters->overtake_completions;
            yields += counters->overtake_yields;
            abandons += counters->overtake_abandons;
        }
        free_allocated(&env);
        free_allocated(&twin);
    }
    printf(
        "  random rollouts with overtaking (commit off and on): %.0f borrow starts, borrowing %.2f%% of steps, invalid "
        "%.0f, commits %.0f, completions %.0f, yields %.0f, abandons %.0f\n",
        starts,
        100.0f * borrow_steps / fmaxf(steps, 1.0f),
        invalid,
        commits,
        completions,
        yields,
        abandons);
    EXPECT_TRUE(starts > 0.0f);
    EXPECT_TRUE(borrow_steps > 0.0f);
    EXPECT_NEAR(invalid, 0.0f, 0.0f);
    EXPECT_TRUE(commits > 0.0f);
    EXPECT_TRUE(completions + yields + abandons <= commits);
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

// hybrid menu: 5 offsets, 21 speeds x 3 comfort levels, 8 stop distances -> MultiDiscrete([2, 28, 2, 77, 5])
static int test_hybrid_menu_layout(void) {
    Drive env = make_hybrid_lattice_env(TOWN06, 1, 0.3f);
    const struct LatticeConfig *cfg = &env.lattice;
    EXPECT_EQ_INT(cfg->nvec[1], 28);
    EXPECT_EQ_INT(cfg->nvec[3], 77);
    EXPECT_EQ_INT(cfg->lon_slot_count, 3);
    EXPECT_EQ_INT(cfg->lon_speed_cell_count, 63);
    EXPECT_EQ_INT(cfg->lon_stop_cell_base, 63);
    EXPECT_EQ_INT(cfg->lon_stop_line_cell, 71);
    EXPECT_EQ_INT(cfg->lon_emergency_cell, 72);
    EXPECT_EQ_INT(cfg->lon_backup_cell_base, 73);
    EXPECT_EQ_INT(cfg->mask_feature_count, 2 + 28 + 2 + 77 + 5);
    EXPECT_EQ_INT(lattice_lat_lane_side(cfg, 7 + 6), 1);
    EXPECT_NEAR(lattice_lat_cell_offset(cfg, 7 + 2), -0.45f, 1e-6f);
    free_allocated(&env);
    return 0;
}

static int urgency_steps_at(Drive *env, float speed_mps, float frac) {
    struct LatticeAgent *lattice_agent = &env->lattice_agents[0];
    Agent *agent = &env->agents[env->active_agent_indices[0]];
    LatticeContext ctx = lattice_make_context(env, lattice_agent, agent, env->timestep + 1);
    return lattice_urgency_speed_steps(env, agent, &ctx, speed_mps, frac, 0);
}

// shortest T on the 0.3 s grid with peak accel 1.5|dv|/T and peak jerk 6|dv|/T^2 inside frac of the limits (c = 1)
static int test_urgency_speed_durations(void) {
    Drive env = make_hybrid_lattice_env(TOWN06, 1, 0.3f);
    place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 40.0f, 5.0f);
    EXPECT_EQ_INT(urgency_steps_at(&env, 8.0f, 1.0f), 8);   // 2.4 s: jerk needs 2.12 s
    EXPECT_EQ_INT(urgency_steps_at(&env, 8.0f, 0.7f), 9);   // 2.7 s: accel 1.75 needs 2.57 s
    EXPECT_EQ_INT(urgency_steps_at(&env, 8.0f, 0.4f), 15);  // 4.5 s: accel 1.0 needs exactly 4.5 s
    EXPECT_EQ_INT(urgency_steps_at(&env, 0.0f, 1.0f), 10);  // 3.0 s: end jerk +4 needs 2.74 s
    EXPECT_EQ_INT(urgency_steps_at(&env, 5.0f, 0.4f), 4);   // hold: the shortest 1.2 s
    EXPECT_EQ_INT(urgency_steps_at(&env, 12.0f, 1.0f), 14); // 4.2 s: accel needs 4.2 s
    EXPECT_EQ_INT(urgency_steps_at(&env, 20.0f, 0.4f), 0);  // 15 m/s gently needs 22.5 s
    Agent *agent = slot0_agent(&env);
    agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.25f;
    agent->reward_coefs[REWARD_COEF_ACC] = 1.5f;
    EXPECT_EQ_INT(urgency_steps_at(&env, 8.0f, 1.0f), 7); // 2.1 s: jerk cap 5 needs 1.90 s, accel cap 3.75 needs 1.2 s
    agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
    agent->reward_coefs[REWARD_COEF_ACC] = 1.0f;
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    LatticeContext ctx = lattice_make_context(&env, lattice_agent, agent, env.timestep + 1);
    int cell_8_firm = 2 * env.lattice.lon_speed_count + 8;
    struct LatticeLonPlan plan = {0};
    EXPECT_TRUE(lattice_build_lon_cell(&env, lattice_agent, agent, &ctx, cell_8_firm, NULL, &plan, NULL));
    EXPECT_NEAR(plan.target_speed_mps, 8.0f, 0.0f);
    EXPECT_EQ_INT(plan.end_step - plan.start_step, 8);
    int cell_20_gentle = 20;
    EXPECT_FALSE(lattice_build_lon_cell(&env, lattice_agent, agent, &ctx, cell_20_gentle, NULL, &plan, NULL));
    EXPECT_FALSE(lattice_agent->mask[lattice_mask_offset(&env.lattice, LATTICE_FACTOR_LON_CELL) + cell_20_gentle]);
    free_allocated(&env);
    return 0;
}

// every accepted comfort-level plan, densely sampled, keeps accel and jerk inside its fraction of the car's limits
static int test_urgency_plans_respect_comfort_caps(void) {
    Drive env = make_hybrid_lattice_env(TOWN06, 1, 0.3f);
    const float start_speeds[] = {0.0f, 3.0f, 8.0f, 14.0f};
    const float start_accels[] = {0.0f, 1.5f, -3.0f};
    int accepted = 0;
    for (int speed_case = 0; speed_case < 4; speed_case++) {
        for (int accel_case = 0; accel_case < 3; accel_case++) {
            place_lattice_agent(&env, 0, STRAIGHT_LANE_TOWN06, 40.0f, start_speeds[speed_case]);
            Agent *agent = slot0_agent(&env);
            struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
            LatticeContext ctx = lattice_make_context(&env, lattice_agent, agent, env.timestep + 1);
            ctx.lon_accel0 = start_accels[accel_case];
            for (int cell = 0; cell < env.lattice.lon_speed_cell_count; cell++) {
                float speed_mps = env.lattice.lon_speeds_mps[cell % env.lattice.lon_speed_count];
                float frac = env.lattice.lon_urgency_fracs[cell / env.lattice.lon_speed_count];
                int steps = lattice_urgency_speed_steps(&env, agent, &ctx, speed_mps, frac, 0);
                if (steps == 0) {
                    continue;
                }
                accepted++;
                double coefs[6];
                double horizon = (double) (float) (steps * env.dt);
                lattice_quartic_speed_coefs(0.0, ctx.lon_speed0, ctx.lon_accel0, speed_mps, horizon, coefs);
                double accel_low = fmin(frac * ACCEL_LONG_LIMIT[0], ctx.lon_accel0) - 1e-4;
                double accel_high = fmax(frac * ACCEL_LONG_LIMIT[1], ctx.lon_accel0) + 1e-4;
                for (int sample = 0; sample <= 400; sample++) {
                    double u = horizon * sample / 400.0;
                    LatticePlanPoint point = lattice_poly_point(coefs, u);
                    double jerk = lattice_poly_third(coefs, u);
                    EXPECT_TRUE(point.second >= accel_low && point.second <= accel_high);
                    EXPECT_TRUE(jerk >= frac * JERK_LONG[0] - 1e-4 && jerk <= frac * JERK_LONG[3] + 1e-4);
                }
                EXPECT_NEAR((float) lattice_poly_point(coefs, horizon).first, speed_mps, 1e-3f);
            }
        }
    }
    printf("  comfort-level plans checked: %d\n", accepted);
    EXPECT_TRUE(accepted > 100);
    free_allocated(&env);
    return 0;
}

// comfort fractions must lie in (0, 1], strictly ascending; the flag must be 0 or 1
static int test_urgency_config_rejected(void) {
    Drive control = drive_test_env_config(TOWN06, SIMULATION_MODE_GIGAFLOW, 1, 0);
    control.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    control.action_type = ACTION_TYPE_LATTICE;
    control.dt = 0.3f;
    lattice_test_default_menu(&control.lattice, 0.3f);
    lattice_test_hybrid_menu(&control.lattice);
    EXPECT_EQ_INT(init_lattice_config(&control), 0);
    const float bad_fracs[][3] = {{0.0f, 0.5f, 1.0f}, {0.4f, 0.7f, 1.5f}, {0.7f, 0.4f, 1.0f}};
    for (int case_idx = 0; case_idx < 3; case_idx++) {
        Drive env = drive_test_env_config(TOWN06, SIMULATION_MODE_GIGAFLOW, 1, 0);
        env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
        env.action_type = ACTION_TYPE_LATTICE;
        env.dt = 0.3f;
        lattice_test_default_menu(&env.lattice, 0.3f);
        memcpy(env.lattice.lon_urgency_fracs, bad_fracs[case_idx], sizeof bad_fracs[case_idx]);
        EXPECT_EQ_INT(init_lattice_config(&env), -1);
    }
    Drive env = drive_test_env_config(TOWN06, SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    env.action_type = ACTION_TYPE_LATTICE;
    env.dt = 0.3f;
    lattice_test_default_menu(&env.lattice, 0.3f);
    env.lattice.lon_urgency = 2;
    EXPECT_EQ_INT(init_lattice_config(&env), -1);
    return 0;
}

// hybrid menu: identical runs match bit for bit; random valid actions on two towns keep observations finite
static int test_hybrid_determinism_and_rollouts(void) {
    Drive env_a = make_hybrid_lattice_env(TOWN06, 8, 0.3f);
    Drive env_b = make_hybrid_lattice_env(TOWN06, 8, 0.3f);
    unsigned long long rng_a = 17, rng_b = 17;
    int obs_size = compute_observation_size(&env_a);
    float lon_new = 0.0f, invalid = 0.0f;
    for (int step = 0; step < 300; step++) {
        random_valid_actions(&env_a, &rng_a);
        random_valid_actions(&env_b, &rng_b);
        c_step(&env_a);
        c_step(&env_b);
        EXPECT_TRUE(
            memcmp(env_a.observations, env_b.observations, env_a.active_agent_count * obs_size * sizeof(float)) == 0);
        for (int value = 0; value < env_a.active_agent_count * obs_size; value++) {
            EXPECT_FINITE(env_a.observations[value]);
        }
    }
    for (int active_idx = 0; active_idx < env_a.active_agent_count; active_idx++) {
        lon_new += env_a.lattice_agents[active_idx].counters.lon_new;
        invalid += env_a.lattice_agents[active_idx].counters.invalid_actions;
    }
    printf("  hybrid rollout: new speed plans %.0f, invalid actions %.0f\n", lon_new, invalid);
    EXPECT_TRUE(lon_new > 0.0f);
    EXPECT_NEAR(invalid, 0.0f, 0.0f);
    free_allocated(&env_a);
    free_allocated(&env_b);
    return 0;
}

// ========================================
// Rail-frame observations, longer trail, goal-exit refresh
// ========================================

// configured but not yet allocated hybrid lattice env with training-like observation ranges and road dropout
static Drive begin_rail_obs_env(const char *map_file, int num_agents) {
    Drive env = drive_test_env_config(map_file, SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
    env.action_type = ACTION_TYPE_LATTICE;
    env.dt = 0.3f;
    env.scenario_length = 5000;
    lattice_test_default_menu(&env.lattice, 0.3f);
    lattice_test_hybrid_menu(&env.lattice);
    env.obs_slots_lane_n = 40;
    env.obs_slots_lane_kept = 28;
    env.obs_slots_boundary_n = 40;
    env.obs_slots_boundary_kept = 24;
    env.obs_norm_xy_offset_m = 200.0f;
    env.obs_norm_goal_offset_m = 200.0f;
    env.obs_range_road_front_m = 200.0f;
    env.obs_range_road_behind_m = 60.0f;
    env.obs_range_road_side_m = 50.0f;
    env.obs_range_partner_m = 200.0f;
    env.obs_range_traffic_control_m = 200.0f;
    return env;
}

static void finish_rail_obs_env(Drive *env) {
    allocate(env);
    c_reset(env);
}

static int rail_block_start(const Drive *env) {
    return compute_observation_size((Drive *) env) - env->lattice.mask_feature_count - rail_obs_feature_count(env);
}

// lane-free rail through samples spaced LATTICE_RAIL_SPACING_M apart; the car's z is its level
static void fill_synthetic_rail(struct LatticeRail *rail, const float *x, const float *y, int count) {
    memset(rail, 0, sizeof *rail);
    rail->sample_count = count;
    rail->is_straight_fallback = 1;
    memcpy(rail->x, x, count * sizeof(float));
    memcpy(rail->y, y, count * sizeof(float));
    lattice_finish_rail_geometry(rail);
}

// straight, arc and hairpin rails: feet, ends, level, tight-bend and second-branch rejections
static int test_rail_obs_projection_synthetic(void) {
    static struct LatticeRail rail;
    static float x[LATTICE_RAIL_SAMPLES], y[LATTICE_RAIL_SAMPLES];
    Drive env = {0};
    LatticeRailObsFrame frame;
    LatticeRailFoot foot;
    for (int sample = 0; sample < LATTICE_NOMINAL_RAIL_SAMPLES; sample++) {
        x[sample] = sample * LATTICE_RAIL_SPACING_M;
        y[sample] = 0.0f;
    }
    fill_synthetic_rail(&rail, x, y, LATTICE_NOMINAL_RAIL_SAMPLES);
    build_lattice_rail_obs_frame(&rail, 0.0f, -1e9f, 1e9f, &frame);
    EXPECT_TRUE(lattice_rail_obs_foot(&env, &frame, 50.0f, 3.0f, 0.0f, &foot));
    EXPECT_NEAR(foot.s_m, 50.0f, 1e-3f);
    EXPECT_NEAR(foot.d_m, 3.0f, 1e-3f);
    EXPECT_NEAR(foot.cos_heading, 1.0f, 1e-5f);
    EXPECT_NEAR(foot.sin_heading, 0.0f, 1e-5f);
    EXPECT_TRUE(lattice_rail_obs_foot(&env, &frame, 120.25f, -2.0f, 0.0f, &foot));
    EXPECT_NEAR(foot.s_m, 120.25f, 1e-3f);
    EXPECT_NEAR(foot.d_m, -2.0f, 1e-3f);
    EXPECT_TRUE(!lattice_rail_obs_foot(&env, &frame, 0.4f, 1.0f, 0.0f, &foot));
    EXPECT_TRUE(!lattice_rail_obs_foot(&env, &frame, -5.0f, 0.0f, 0.0f, &foot));
    EXPECT_TRUE(!lattice_rail_obs_foot(&env, &frame, 255.0f, 0.0f, 0.0f, &foot));
    EXPECT_TRUE(!lattice_rail_obs_foot(&env, &frame, 50.0f, 3.0f, 4.5f, &foot));
    EXPECT_TRUE(lattice_rail_obs_foot(&env, &frame, 50.0f, 3.0f, 3.9f, &foot));

    float radius_m = 20.0f;
    int arc_count = 120;
    for (int sample = 0; sample < arc_count; sample++) {
        float angle = sample * LATTICE_RAIL_SPACING_M / radius_m;
        x[sample] = radius_m * sinf(angle);
        y[sample] = radius_m * (1.0f - cosf(angle));
    }
    fill_synthetic_rail(&rail, x, y, arc_count);
    build_lattice_rail_obs_frame(&rail, 0.0f, -1e9f, 1e9f, &frame);
    float probe_angle = 1.5f;
    const float arc_offsets_m[] = {10.0f, -10.0f, -17.0f, 17.0f};
    for (int case_idx = 0; case_idx < 4; case_idx++) {
        float d_m = arc_offsets_m[case_idx];
        float px = (radius_m - d_m) * sinf(probe_angle);
        float py = radius_m - (radius_m - d_m) * cosf(probe_angle);
        int valid = lattice_rail_obs_foot(&env, &frame, px, py, 0.0f, &foot);
        EXPECT_EQ_INT(valid, case_idx < 3);
        if (valid) {
            // a point off a chord polyline projects up to |d| * (spacing / R) / 2 away from the arc's own foot
            EXPECT_NEAR(
                foot.s_m,
                radius_m * probe_angle,
                0.5f * fabsf(d_m) * LATTICE_RAIL_SPACING_M / radius_m + 0.01f);
            EXPECT_NEAR(foot.d_m, d_m, 0.02f);
            EXPECT_NEAR(foot.cos_heading, cosf(probe_angle), 0.02f);
            EXPECT_NEAR(foot.sin_heading, sinf(probe_angle), 0.02f);
        }
    }

    float turn_radius_m = 4.0f;
    float leg_m = 50.0f;
    float bend_m = (float) M_PI * turn_radius_m;
    int hairpin_count = (int) ((2.0f * leg_m + bend_m) / LATTICE_RAIL_SPACING_M);
    for (int sample = 0; sample < hairpin_count; sample++) {
        float s_m = sample * LATTICE_RAIL_SPACING_M;
        float angle = (s_m - leg_m) / turn_radius_m;
        x[sample] = s_m <= leg_m
            ? s_m
            : (s_m <= leg_m + bend_m ? leg_m + turn_radius_m * sinf(angle) : leg_m - (s_m - leg_m - bend_m));
        y[sample] = s_m <= leg_m
            ? 0.0f
            : (s_m <= leg_m + bend_m ? turn_radius_m * (1.0f - cosf(angle)) : 2.0f * turn_radius_m);
    }
    fill_synthetic_rail(&rail, x, y, hairpin_count);
    build_lattice_rail_obs_frame(&rail, 0.0f, -1e9f, 1e9f, &frame);
    EXPECT_TRUE(lattice_rail_obs_foot(&env, &frame, 25.0f, 1.0f, 0.0f, &foot));
    EXPECT_NEAR(foot.s_m, 25.0f, 0.05f);
    EXPECT_NEAR(foot.d_m, 1.0f, 0.02f);
    EXPECT_TRUE(!lattice_rail_obs_foot(&env, &frame, 25.0f, 3.0f, 0.0f, &foot));
    EXPECT_TRUE(lattice_rail_obs_foot(&env, &frame, 25.0f, -3.0f, 0.0f, &foot));
    EXPECT_NEAR(foot.d_m, -3.0f, 0.02f);
    EXPECT_TRUE(lattice_rail_obs_foot(&env, &frame, 25.0f, 7.0f, 0.0f, &foot));
    EXPECT_NEAR(foot.s_m, leg_m + bend_m + 25.0f, 0.05f);
    EXPECT_NEAR(foot.d_m, 1.0f, 0.02f);
    build_lattice_rail_obs_frame(&rail, 0.0f, 30.0f, 1e9f, &frame);
    EXPECT_TRUE(!lattice_rail_obs_foot(&env, &frame, 25.0f, -3.0f, 0.0f, &foot));
    build_lattice_rail_obs_frame(&rail, 0.0f, -1e9f, 20.0f, &frame);
    EXPECT_TRUE(!lattice_rail_obs_foot(&env, &frame, 25.0f, -3.0f, 0.0f, &foot));
    return 0;
}

// a neighbour-lane rail shifts every d by the lane spacing and keeps d - d_ego (ds scales with the curve radius)
static int test_rail_obs_lane_change_invariance(void) {
    Drive env = begin_rail_obs_env(TOWN06, 2);
    env.obs_rail_frame = 1;
    finish_rail_obs_env(&env);
    place_lattice_agent(&env, 1, LEFT_NEIGHBOUR_TOWN06, 40.0f, 4.5f);
    place_lattice_agent(&env, 0, CURVED_LANE_TOWN06, 20.0f, 4.5f);
    int obs_size = compute_observation_size(&env);
    int rail_start = rail_block_start(&env);
    float before[RAIL_PARTNER_FEATURES] = {0};
    for (int slot = 0; slot < env.obs_slots_partners_n; slot++) {
        const float *row = &env.observations[rail_start + slot * RAIL_PARTNER_FEATURES];
        if (row[0] == 1.0f) {
            memcpy(before, row, sizeof before);
        }
    }
    EXPECT_NEAR(before[0], 1.0f, 0.0f);
    struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
    lattice_neighbours_at(&env, lattice_agent);
    EXPECT_EQ_INT(lattice_agent->neighbour_lane[1], LEFT_NEIGHBOUR_TOWN06);
    int sample_idx = build_lattice_fresh_rail(
        &env,
        &lattice_agent->rail,
        slot0_agent(&env),
        LEFT_NEIGHBOUR_TOWN06,
        lattice_agent->neighbour_arc_m[1]);
    EXPECT_TRUE(sample_idx >= 0);
    lattice_agent->projection_hint = sample_idx;
    compute_observations(&env);
    float after[RAIL_PARTNER_FEATURES] = {0};
    for (int slot = 0; slot < env.obs_slots_partners_n; slot++) {
        const float *row = &env.observations[rail_start + slot * RAIL_PARTNER_FEATURES];
        if (row[0] == 1.0f) {
            memcpy(after, row, sizeof after);
        }
    }
    printf(
        "  partner on the left lane: d %.3f -> %.3f lanes, d - d_ego %.3f -> %.3f lanes, ds %.2f -> %.2f m\n",
        before[2],
        after[2],
        before[4],
        after[4],
        before[1] * RAIL_OBS_DS_NORM_M,
        after[1] * RAIL_OBS_DS_NORM_M);
    EXPECT_NEAR(after[0], 1.0f, 0.0f);
    EXPECT_NEAR(after[4], before[4], 0.02f);
    EXPECT_TRUE(before[2] - after[2] > 0.7f);
    EXPECT_NEAR(after[5], before[5], 1e-3f);
    (void) obs_size;
    free_allocated(&env);
    return 0;
}

// hiding the car frame zeroes exactly the planar car-frame columns; the rest, the rail block and the trajectories match
static int test_rail_obs_hide_car_frame_and_layout(void) {
    Drive shown = begin_rail_obs_env(TOWN06, 6);
    Drive hidden = begin_rail_obs_env(TOWN06, 6);
    Drive plain = begin_rail_obs_env(TOWN06, 6);
    shown.obs_rail_frame = 1;
    hidden.obs_rail_frame = 1;
    hidden.obs_hide_car_frame = 1;
    finish_rail_obs_env(&shown);
    finish_rail_obs_env(&hidden);
    finish_rail_obs_env(&plain);
    int obs_size = compute_observation_size(&shown);
    int rail_features = rail_obs_feature_count(&shown);
    EXPECT_EQ_INT(rail_features, 16 * 9 + (28 + 24) * 6 + 4 * 4 + 3 * 4 + 6 * 2);
    EXPECT_EQ_INT(obs_size, compute_observation_size(&plain) + rail_features);
    int plan_start = EGO_FEATURES;
    int partner_start = plan_start + lattice_plan_feature_count(&shown.lattice) + shown.num_goals * GOAL_FEATURES;
    int lane_start = partner_start + shown.obs_slots_partners_n * PARTNER_FEATURES;
    int boundary_start = lane_start + shown.obs_slots_lane_kept * LANE_FEATURES;
    int traffic_start = boundary_start + shown.obs_slots_boundary_kept * BOUNDARY_FEATURES;
    int count_start = traffic_start + shown.obs_slots_traffic_controls_n * TRAFFIC_CONTROL_FEATURES;
    int rail_start = rail_block_start(&shown);
    EXPECT_EQ_INT(rail_start, count_start + OBS_VALID_COUNT_FEATURES);
    unsigned long long rng_shown = 23, rng_hidden = 23, rng_plain = 23;
    float expected[obs_size];
    int valid_rows = 0, listed_rows = 0;
    for (int step = 0; step < 60; step++) {
        random_valid_actions(&shown, &rng_shown);
        random_valid_actions(&hidden, &rng_hidden);
        random_valid_actions(&plain, &rng_plain);
        c_step(&shown);
        c_step(&hidden);
        c_step(&plain);
        for (int active_idx = 0; active_idx < shown.active_agent_count; active_idx++) {
            const float *row = &shown.observations[active_idx * obs_size];
            const float *plain_row = &plain.observations[active_idx * compute_observation_size(&plain)];
            EXPECT_TRUE(memcmp(row, plain_row, count_start * sizeof(float)) == 0);
            memcpy(expected, row, sizeof expected);
            for (int point = 0; point < LATTICE_PREVIEW_POINTS; point++) {
                expected[plan_start + 42 + 3 * point] = 0.0f;
                expected[plan_start + 42 + 3 * point + 1] = 0.0f;
            }
            const int planar[4] = {0, 1, 5, 6};
            for (int slot = 0; slot < shown.obs_slots_partners_n; slot++) {
                for (int col = 0; col < 4; col++) {
                    expected[partner_start + slot * PARTNER_FEATURES + planar[col]] = 0.0f;
                }
            }
            for (int slot = 0; slot < shown.obs_slots_lane_kept + shown.obs_slots_boundary_kept; slot++) {
                for (int col = 0; col < 4; col++) {
                    expected[lane_start + slot * LANE_FEATURES + planar[col]] = 0.0f;
                }
            }
            for (int slot = 0; slot < shown.obs_slots_traffic_controls_n; slot++) {
                for (int col = 0; col < 4; col++) {
                    expected[traffic_start + slot * TRAFFIC_CONTROL_FEATURES + col] = 0.0f;
                }
            }
            const float *hidden_row = &hidden.observations[active_idx * obs_size];
            for (int value = 0; value < obs_size; value++) {
                EXPECT_TRUE(hidden_row[value] == expected[value]);
            }
            int counts[4]
                = {(int) row[count_start],
                   (int) row[count_start + 1],
                   (int) row[count_start + 2],
                   (int) row[count_start + 3]};
            for (int slot = 0; slot < shown.obs_slots_partners_n; slot++) {
                const float *rail_row = &row[rail_start + slot * RAIL_PARTNER_FEATURES];
                EXPECT_TRUE(rail_row[0] == 0.0f || (rail_row[0] == 1.0f && slot < counts[2]));
                valid_rows += rail_row[0] == 1.0f;
                listed_rows += slot < counts[2];
            }
            int road_rail_start = rail_start + shown.obs_slots_partners_n * RAIL_PARTNER_FEATURES;
            for (int slot = 0; slot < shown.obs_slots_lane_kept + shown.obs_slots_boundary_kept; slot++) {
                int is_lane = slot < shown.obs_slots_lane_kept;
                int listed = is_lane ? slot < counts[0] : slot - shown.obs_slots_lane_kept < counts[1];
                const float *rail_row = &row[road_rail_start + slot * RAIL_ROAD_FEATURES];
                EXPECT_TRUE(rail_row[0] == 0.0f || (rail_row[0] == 1.0f && listed));
                EXPECT_TRUE(rail_row[4] * rail_row[4] + rail_row[5] * rail_row[5] <= rail_row[0] + 1e-4f);
            }
        }
    }
    printf("  rail partner rows valid %d of %d listed\n", valid_rows, listed_rows);
    EXPECT_TRUE(valid_rows > 0);
    free_allocated(&shown);
    free_allocated(&hidden);
    free_allocated(&plain);
    return 0;
}

// a 60 m trail adds 60 samples behind the car and keeps the reach ahead; bad trails and refresh pairings are rejected
static int test_rail_trail_keep(void) {
    char path[512];
    carla_town_path(path, sizeof path, "Town03");
    Drive short_trail = begin_rail_obs_env(path, 1);
    Drive long_trail = begin_rail_obs_env(path, 1);
    // the greedy driver picks speed cells of the duration menu
    lattice_test_default_menu(&short_trail.lattice, 0.3f);
    lattice_test_default_menu(&long_trail.lattice, 0.3f);
    long_trail.lattice.trail_keep_m = 60.0f;
    finish_rail_obs_env(&short_trail);
    finish_rail_obs_env(&long_trail);
    int checked_lanes = 0;
    int full_trail_lane = -1;
    for (int lane = 0; lane < short_trail.num_road_elements && checked_lanes < 20; lane++) {
        if (!is_drivable_road_lane(short_trail.road_elements[lane].type)
            || short_trail.lattice_lanes[lane].length_m < 40.0f || short_trail.lattice_lanes[lane].is_connector) {
            continue;
        }
        place_lattice_agent(&short_trail, 0, lane, 5.0f, 5.0f);
        place_lattice_agent(&long_trail, 0, lane, 5.0f, 5.0f);
        const struct LatticeRail *rail_30 = &short_trail.lattice_agents[0].rail;
        const struct LatticeRail *rail_60 = &long_trail.lattice_agents[0].rail;
        float car_s_30
            = lattice_frenet_state(rail_30, slot0_agent(&short_trail), short_trail.lattice_agents[0].projection_hint).s;
        float car_s_60
            = lattice_frenet_state(rail_60, slot0_agent(&long_trail), long_trail.lattice_agents[0].projection_hint).s;
        if (car_s_60 - rail_60->s_start_m < 59.0f || lattice_rail_end_s(rail_30) - car_s_30 < 225.0f) {
            continue;
        }
        EXPECT_NEAR(car_s_30 - rail_30->s_start_m, 30.0f, 0.6f);
        EXPECT_NEAR(car_s_60 - rail_60->s_start_m, 60.0f, 0.6f);
        EXPECT_EQ_INT(rail_30->sample_count, LATTICE_NOMINAL_RAIL_SAMPLES);
        EXPECT_EQ_INT(rail_60->sample_count, LATTICE_NOMINAL_RAIL_SAMPLES + 60);
        EXPECT_NEAR(lattice_rail_end_s(rail_60) - car_s_60, lattice_rail_end_s(rail_30) - car_s_30, 0.6f);
        full_trail_lane = full_trail_lane < 0 ? lane : full_trail_lane;
        checked_lanes++;
    }
    printf("  lanes with a full 60 m trail and 225 m ahead: %d\n", checked_lanes);
    EXPECT_TRUE(checked_lanes >= 5);
    place_lattice_agent(&long_trail, 0, full_trail_lane, 5.0f, 0.0f);
    float min_trail_m = 1e9f, max_trail_m = 0.0f;
    int max_samples = 0;
    for (int step = 0; step < 900; step++) {
        step_greedy(&long_trail, 10.0f);
        struct LatticeAgent *lattice_agent = &long_trail.lattice_agents[0];
        float car_s
            = lattice_frenet_state(&lattice_agent->rail, slot0_agent(&long_trail), lattice_agent->projection_hint).s;
        float trail_m = car_s - lattice_agent->rail.s_start_m;
        max_samples = lattice_agent->rail.sample_count > max_samples ? lattice_agent->rail.sample_count : max_samples;
        max_trail_m = fmaxf(max_trail_m, trail_m);
        if (lattice_agent->counters.rail_regens > 0.0f) {
            min_trail_m = fminf(min_trail_m, trail_m);
        }
    }
    printf(
        "  900 greedy steps with a 60 m trail: regenerations %.0f, trail %.1f..%.1f m, most samples %d\n",
        long_trail.lattice_agents[0].counters.rail_regens,
        min_trail_m,
        max_trail_m,
        max_samples);
    EXPECT_TRUE(long_trail.lattice_agents[0].counters.rail_regens > 5.0f);
    EXPECT_TRUE(min_trail_m > 55.0f);
    EXPECT_TRUE(max_samples <= LATTICE_NOMINAL_RAIL_SAMPLES + 60);
    free_allocated(&short_trail);
    free_allocated(&long_trail);

    const float trails[] = {29.5f, 94.5f, 60.25f, 60.0f, 94.0f};
    for (int case_idx = 0; case_idx < 5; case_idx++) {
        Drive env = begin_rail_obs_env(TOWN06, 1);
        env.lattice.trail_keep_m = trails[case_idx];
        EXPECT_EQ_INT(init_lattice_config(&env), case_idx < 3 ? -1 : 0);
    }
    for (int exit_mode = 0; exit_mode < 2; exit_mode++) {
        Drive env = begin_rail_obs_env(TOWN06, 1);
        env.lattice.exit_mode = exit_mode == 0 ? LATTICE_EXIT_MODE_POLICY : LATTICE_EXIT_MODE_GOAL;
        env.lattice.exit_goal_refresh = 1;
        EXPECT_EQ_INT(init_lattice_config(&env), exit_mode == 0 ? -1 : 0);
    }
    return 0;
}

static void set_single_goal(Drive *env, Agent *agent, int goal_lane) {
    float goal_x, goal_y, goal_heading;
    lattice_lane_point_at_arc(
        env,
        goal_lane,
        0.5f * env->lattice_lanes[goal_lane].length_m,
        &goal_x,
        &goal_y,
        &goal_heading);
    agent->list_goal_x[0] = goal_x;
    agent->list_goal_y[0] = goal_y;
    agent->list_goal_z[0] = agent->sim_z;
    agent->list_goal_lane[0] = goal_lane;
    agent->goal_count = 1;
    agent->current_goal_idx = 0;
    agent->current_goal_x = goal_x;
    agent->current_goal_y = goal_y;
    agent->current_goal_z = agent->sim_z;
}

// goal exits: a new goal switches a decided split ahead only with the refresh on
static int test_exit_goal_refresh(void) {
    char path[512];
    carla_town_path(path, sizeof path, "Town03");
    int switches[2] = {0, 0};
    int took_new_exit[2] = {0, 0};
    for (int refresh = 0; refresh < 2; refresh++) {
        Drive env = begin_rail_obs_env(path, 1);
        env.lattice.exit_mode = LATTICE_EXIT_MODE_GOAL;
        env.lattice.exit_goal_refresh = refresh;
        finish_rail_obs_env(&env);
        int split_lane = -1;
        for (int lane = 0; lane < env.num_road_elements && split_lane < 0; lane++) {
            const struct LatticeLaneInfo *info = &env.lattice_lanes[lane];
            int exits_reach_themselves = info->exit_count >= 2
                && isfinite(lattice_goal_distance_m(&env, info->exit_slots[0], info->exit_slots[0]))
                && isfinite(lattice_goal_distance_m(&env, info->exit_slots[1], info->exit_slots[1]));
            if (exits_reach_themselves && info->length_m > 30.0f && !info->is_connector) {
                split_lane = lane;
            }
        }
        EXPECT_TRUE(split_lane >= 0);
        int straight_exit = env.lattice_lanes[split_lane].exit_slots[0];
        int turn_exit = env.lattice_lanes[split_lane].exit_slots[1];
        float split_arc_m = env.lattice_lanes[split_lane].length_m - 25.0f;
        place_lattice_agent(&env, 0, split_lane, split_arc_m, 3.0f);
        Agent *agent = slot0_agent(&env);
        struct LatticeAgent *lattice_agent = &env.lattice_agents[0];
        set_single_goal(&env, agent, turn_exit);
        place_lattice_agent(&env, 0, split_lane, split_arc_m, 3.0f);
        EXPECT_TRUE(lattice_slot_after(&lattice_agent->rail, split_lane, turn_exit) >= 0);
        set_single_goal(&env, agent, straight_exit);
        step_keep(&env, 1);
        switches[refresh] = (int) lattice_agent->counters.exit_switches;
        took_new_exit[refresh] = lattice_slot_after(&lattice_agent->rail, split_lane, straight_exit) >= 0;
        free_allocated(&env);
    }
    printf(
        "  goal changed after the split was decided: switches without / with refresh %d / %d\n",
        switches[0],
        switches[1]);
    EXPECT_EQ_INT(switches[0], 0);
    EXPECT_EQ_INT(took_new_exit[0], 0);
    EXPECT_EQ_INT(switches[1], 1);
    EXPECT_EQ_INT(took_new_exit[1], 1);
    return 0;
}

// the run configurations on several towns: twins match bit for bit, observations stay finite; prints valid fractions
static int test_rail_obs_random_rollouts(void) {
    const char *towns[] = {"Town01", "Town03", "Town06", "Town10HD"};
    int valid[4] = {0}, listed[4] = {0};
    for (int town_idx = 0; town_idx < 4; town_idx++) {
        char path[512];
        carla_town_path(path, sizeof path, towns[town_idx]);
        Drive env = begin_rail_obs_env(path, 12);
        Drive twin = begin_rail_obs_env(path, 12);
        Drive *pair[2] = {&env, &twin};
        for (int copy = 0; copy < 2; copy++) {
            pair[copy]->obs_rail_frame = 1;
            pair[copy]->obs_hide_car_frame = town_idx % 2;
            pair[copy]->lattice.trail_keep_m = 60.0f;
            pair[copy]->lattice.exit_mode = town_idx < 2 ? LATTICE_EXIT_MODE_GOAL : LATTICE_EXIT_MODE_POLICY;
            pair[copy]->lattice.exit_goal_refresh = town_idx < 2;
            pair[copy]->lattice.oncoming_overtake = 1;
            pair[copy]->lattice.turnaround = 1;
            pair[copy]->goal_source = GOAL_SOURCE_MAP;
            finish_rail_obs_env(pair[copy]);
        }
        int obs_size = compute_observation_size(&env);
        int rail_start = rail_block_start(&env);
        int count_start = rail_start - OBS_VALID_COUNT_FEATURES;
        unsigned long long rng_state = 31 + town_idx, twin_rng = 31 + town_idx;
        for (int step = 0; step < 250; step++) {
            random_valid_actions(&env, &rng_state);
            random_valid_actions(&twin, &twin_rng);
            c_step(&env);
            c_step(&twin);
            EXPECT_TRUE(
                memcmp(env.observations, twin.observations, env.active_agent_count * obs_size * sizeof(float)) == 0);
            for (int active_idx = 0; active_idx < env.active_agent_count; active_idx++) {
                const float *row = &env.observations[active_idx * obs_size];
                for (int value = 0; value < obs_size; value++) {
                    EXPECT_FINITE(row[value]);
                }
                int partner_count = (int) row[count_start + 2];
                int road_listed[2] = {(int) row[count_start], (int) row[count_start + 1]};
                int traffic_count = (int) row[count_start + 3];
                const float *rail = &row[rail_start];
                for (int slot = 0; slot < partner_count; slot++) {
                    valid[0] += rail[slot * RAIL_PARTNER_FEATURES] == 1.0f;
                }
                listed[0] += partner_count;
                rail += env.obs_slots_partners_n * RAIL_PARTNER_FEATURES;
                for (int slot = 0; slot < env.obs_slots_lane_kept + env.obs_slots_boundary_kept; slot++) {
                    int is_lane = slot < env.obs_slots_lane_kept;
                    int listed_row = is_lane ? slot < road_listed[0] : slot - env.obs_slots_lane_kept < road_listed[1];
                    valid[1] += listed_row && rail[slot * RAIL_ROAD_FEATURES] == 1.0f;
                    listed[1] += listed_row;
                }
                rail += (env.obs_slots_lane_kept + env.obs_slots_boundary_kept) * RAIL_ROAD_FEATURES;
                for (int slot = 0; slot < traffic_count; slot++) {
                    valid[2] += rail[slot * RAIL_POINT_FEATURES] == 1.0f;
                }
                listed[2] += traffic_count;
                rail += env.obs_slots_traffic_controls_n * RAIL_POINT_FEATURES;
                const Agent *agent = &env.agents[env.active_agent_indices[active_idx]];
                for (int goal_idx = 0; goal_idx < env.num_goals; goal_idx++) {
                    int present = goal_idx >= agent->current_goal_idx && goal_idx < agent->goal_count;
                    valid[3] += present && rail[goal_idx * RAIL_POINT_FEATURES] == 1.0f;
                    listed[3] += present;
                }
            }
        }
        free_allocated(&env);
        free_allocated(&twin);
    }
    const char *names[] = {"partners", "road segments", "stop lines", "goals"};
    for (int kind = 0; kind < 4; kind++) {
        printf(
            "  rail-valid %s: %d of %d (%.1f%%)\n",
            names[kind],
            valid[kind],
            listed[kind],
            100.0f * valid[kind] / fmaxf(1.0f, (float) listed[kind]));
    }
    EXPECT_TRUE(valid[0] > 0.3f * listed[0]);
    EXPECT_TRUE(valid[1] > 0.3f * listed[1]);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_polynomials_and_bellman);
    RUN_TEST(test_menu_layout);
    RUN_TEST(test_hybrid_menu_layout);
    RUN_TEST(test_urgency_speed_durations);
    RUN_TEST(test_urgency_plans_respect_comfort_caps);
    RUN_TEST(test_urgency_config_rejected);
    RUN_TEST(test_hybrid_determinism_and_rollouts);
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
    RUN_TEST(test_exit_redecidable_until_split);
    RUN_TEST(test_committed_exit_stays_offered);
    RUN_TEST(test_drift_switch_with_hysteresis);
    RUN_TEST(test_no_lane_mode_and_removed_rows);
    RUN_TEST(test_rail_changed_flag_timing_dt_01);
    RUN_TEST(test_plan_change_rms_consistency);
    RUN_TEST(test_route_progress_reward);
    RUN_TEST(test_stop_line_light_features);
    RUN_TEST(test_wait_penalty);
    RUN_TEST(test_light_state_only_when_facing);
    RUN_TEST(test_stop_line_cell_needs_view);
    RUN_TEST(test_wait_exempt_near_or_queued);
    RUN_TEST(test_consistency_free_from_rest);
    RUN_TEST(test_random_rollouts_all_towns);
    RUN_TEST(test_oncoming_menu_layout);
    RUN_TEST(test_oncoming_profiles);
    RUN_TEST(test_oncoming_overtake_stopped_car);
    RUN_TEST(test_overtake_commit_holds_until_passed);
    RUN_TEST(test_overtake_commit_yields);
    RUN_TEST(test_overtake_commit_yields_to_parked_car);
    RUN_TEST(test_overtake_threat_standing_or_driving);
    RUN_TEST(test_overtake_commit_expiry);
    RUN_TEST(test_overtake_commit_skips_facing_car);
    RUN_TEST(test_overtake_commit_config);
    RUN_TEST(test_speed_bonus_while_cruising);
    RUN_TEST(test_stacking_counters);
    RUN_TEST(test_oncoming_forced_return_before_junction);
    RUN_TEST(test_oncoming_random_rollouts);
    RUN_TEST(test_turn_menu_layout);
    RUN_TEST(test_turn_planner_cases);
    RUN_TEST(test_turn_execution_town01);
    RUN_TEST(test_turn_aborts_when_stopped);
    RUN_TEST(test_turn_offer_rules);
    RUN_TEST(test_turn_random_rollouts);
    RUN_TEST(test_rail_obs_projection_synthetic);
    RUN_TEST(test_rail_obs_lane_change_invariance);
    RUN_TEST(test_rail_obs_hide_car_frame_and_layout);
    RUN_TEST(test_rail_trail_keep);
    RUN_TEST(test_exit_goal_refresh);
    RUN_TEST(test_rail_obs_random_rollouts);
    return test_summary(failures);
}
