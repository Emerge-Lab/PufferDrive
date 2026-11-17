#include "include/drive_fixture.h"
#include "include/test.h"

static int test_commit_goals_front_align_fills_every_slot(void) {
    // a full front-aligned set fills every slot and the current_goal_* alias tracks slot 0
    Drive env = {0};
    env.num_goals = 3;
    Agent agent = {0};
    float gx[3] = {10.0f, 20.0f, 30.0f};
    float gy[3] = {11.0f, 21.0f, 31.0f};
    float gz[3] = {1.0f, 2.0f, 3.0f};
    int glane[3] = {4, 5, 6};

    commit_goals(&env, &agent, gx, gy, gz, glane, 3, 0);

    EXPECT_EQ_INT(agent.goal_count, 3);
    EXPECT_EQ_INT(agent.current_goal_idx, 0);
    for (int slot = 0; slot < 3; slot++) {
        EXPECT_NEAR(agent.list_goal_x[slot], gx[slot], 1e-6f);
        EXPECT_NEAR(agent.list_goal_y[slot], gy[slot], 1e-6f);
        EXPECT_NEAR(agent.list_goal_z[slot], gz[slot], 1e-6f);
        EXPECT_EQ_INT(agent.list_goal_lane[slot], glane[slot]);
    }
    EXPECT_NEAR(agent.current_goal_x, gx[0], 1e-6f);
    EXPECT_NEAR(agent.current_goal_y, gy[0], 1e-6f);
    EXPECT_NEAR(agent.current_goal_z, gz[0], 1e-6f);
    return 0;
}

static int test_commit_goals_back_align_pads_front_with_lane_minus_one(void) {
    // a back-aligned partial set: zeroed lane -1 slots first, current_goal_idx on the first placed goal
    Drive env = {0};
    env.num_goals = 3;
    Agent agent = {0};
    float gx[1] = {99.0f};
    float gy[1] = {98.0f};
    float gz[1] = {5.0f};
    int glane[1] = {7};
    int start_slot = env.num_goals - 1; // one valid goal -> back-aligned into the last slot

    commit_goals(&env, &agent, gx, gy, gz, glane, 1, start_slot);

    EXPECT_EQ_INT(agent.goal_count, 3);
    EXPECT_EQ_INT(agent.current_goal_idx, start_slot);
    for (int slot = 0; slot < start_slot; slot++) {
        EXPECT_EQ_INT(agent.list_goal_lane[slot], -1);
        EXPECT_NEAR(agent.list_goal_x[slot], 0.0f, 1e-6f);
        EXPECT_NEAR(agent.list_goal_y[slot], 0.0f, 1e-6f);
        EXPECT_NEAR(agent.list_goal_z[slot], 0.0f, 1e-6f);
    }
    EXPECT_NEAR(agent.list_goal_x[start_slot], gx[0], 1e-6f);
    EXPECT_EQ_INT(agent.list_goal_lane[start_slot], glane[0]);
    EXPECT_NEAR(agent.current_goal_x, gx[0], 1e-6f);
    EXPECT_NEAR(agent.current_goal_y, gy[0], 1e-6f);
    return 0;
}

static int test_map_goal_source_no_attrition(void) {
    // with the fresh-cell retry every active agent spawns, none is removed for a dead-ended seed draw
    srand(7);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    env.goal_source = GOAL_SOURCE_MAP;
    allocate(&env);
    c_reset(&env);

    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        EXPECT_FALSE(agent->removed);
        EXPECT_TRUE(agent->goal_count >= 1 && agent->goal_count <= env.num_goals);
    }
    free_allocated(&env);
    return 0;
}

static int test_map_goals_carry_lane_and_track_slot_zero(void) {
    // every map goal carries a lane for the GPS lane-distance feature; the alias mirrors slot 0
    srand(3);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    env.goal_source = GOAL_SOURCE_MAP;
    allocate(&env);
    c_reset(&env);

    Agent *agent = &env.agents[env.active_agent_indices[0]];
    EXPECT_TRUE(generate_new_goals_from_map(&env, agent));
    EXPECT_TRUE(agent->goal_count >= 1 && agent->goal_count <= env.num_goals);
    EXPECT_EQ_INT(agent->current_goal_idx, 0);
    for (int slot = 0; slot < agent->goal_count; slot++) {
        EXPECT_TRUE(agent->list_goal_lane[slot] >= 0);
        EXPECT_FINITE(agent->list_goal_x[slot]);
        EXPECT_FINITE(agent->list_goal_y[slot]);
    }
    EXPECT_NEAR(agent->current_goal_x, agent->list_goal_x[0], 1e-6f);
    EXPECT_NEAR(agent->current_goal_y, agent->list_goal_y[0], 1e-6f);
    free_allocated(&env);
    return 0;
}

static float lane_heading_at_point(const Drive *env, int lane_idx, float x, float y) {
    const RoadMapElement *lane = &env->road_elements[lane_idx];
    int best_vertex_idx = 0;
    float best_dist_sq = 1e30f;
    for (int vertex_idx = 0; vertex_idx < lane->segment_size; vertex_idx++) {
        float dx = lane->x[vertex_idx] - x;
        float dy = lane->y[vertex_idx] - y;
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_vertex_idx = vertex_idx;
        }
    }
    return lane->headings[best_vertex_idx];
}

static int count_goal_heading_violations(Drive *env, float max_heading_deg, int regen_count, int *out_pair_count) {
    env->goal_heading_max_deg = max_heading_deg;
    float max_heading_rad = 60.0f * (float) M_PI / 180.0f;
    int violation_count = 0;
    *out_pair_count = 0;
    Agent *agent = &env->agents[env->active_agent_indices[0]];
    for (int regen_idx = 0; regen_idx < regen_count; regen_idx++) {
        if (!generate_new_goals_from_map(env, agent)) {
            continue;
        }
        for (int slot = 1; slot < agent->goal_count; slot++) {
            float prev_heading = lane_heading_at_point(
                env,
                agent->list_goal_lane[slot - 1],
                agent->list_goal_x[slot - 1],
                agent->list_goal_y[slot - 1]);
            float heading = lane_heading_at_point(
                env,
                agent->list_goal_lane[slot],
                agent->list_goal_x[slot],
                agent->list_goal_y[slot]);
            (*out_pair_count)++;
            if (fabsf(normalize_heading(heading - prev_heading)) > max_heading_rad) {
                violation_count++;
            }
        }
    }
    return violation_count;
}

static int test_map_goal_heading_constraint_reduces_turns(void) {
    // with goal_heading_max_deg successive goals turn > 60 deg far less often than without (0 = disabled)
    srand(11);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    env.goal_source = GOAL_SOURCE_MAP;
    env.num_goals = 4;
    env.min_goal_spacing = 20.0f;
    env.max_goal_spacing = 200.0f;
    allocate(&env);
    c_reset(&env);

    int free_pair_count = 0, constrained_pair_count = 0;
    int free_violations = count_goal_heading_violations(&env, 0.0f, 2000, &free_pair_count);
    int constrained_violations = count_goal_heading_violations(&env, 60.0f, 2000, &constrained_pair_count);
    printf(
        "  heading>60deg pairs: unconstrained %d/%d, constrained %d/%d\n",
        free_violations,
        free_pair_count,
        constrained_violations,
        constrained_pair_count);
    EXPECT_TRUE(free_pair_count > 500 && constrained_pair_count > 500);
    // Constrained violation rate must be under 60% of the unconstrained rate (Town01 T-junctions leave a residue).
    EXPECT_TRUE(constrained_violations * free_pair_count * 10 < free_violations * constrained_pair_count * 6);
    free_allocated(&env);
    return 0;
}

static int test_route_goals_full_set_or_removed(void) {
    // a live route agent always holds the full front-aligned set; a failed retry removes the agent
    srand(5);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        EXPECT_TRUE(agent->removed || agent->goal_count == env.num_goals);
    }
    free_allocated(&env);
    return 0;
}

static int test_route_goals_front_aligned_with_lanes(void) {
    // Every live route agent front-aligns a full, lane-tagged, finite goal set with the alias on slot 0.
    srand(11);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    int checked = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        if (agent->removed) {
            continue;
        }
        checked++;
        EXPECT_EQ_INT(agent->current_goal_idx, 0);
        EXPECT_EQ_INT(agent->goal_count, env.num_goals);
        for (int slot = 0; slot < agent->goal_count; slot++) {
            EXPECT_TRUE(agent->list_goal_lane[slot] >= 0);
            EXPECT_FINITE(agent->list_goal_x[slot]);
            EXPECT_FINITE(agent->list_goal_y[slot]);
            EXPECT_FINITE(agent->list_goal_z[slot]);
        }
        EXPECT_NEAR(agent->current_goal_x, agent->list_goal_x[0], 1e-6f);
    }
    EXPECT_TRUE(checked > 0);
    free_allocated(&env);
    return 0;
}

static int test_roll_goals_slides_window_and_appends(void) {
    // rolling regen shifts the goals one slot forward and appends one, current_goal_idx stays 0
    srand(13);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    int rolled = 0;
    for (int i = 0; i < env.active_agent_count && rolled == 0; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        if (agent->removed || agent->goal_count != env.num_goals) {
            continue;
        }
        float old_x[MAX_GOALS], old_y[MAX_GOALS];
        int old_lane[MAX_GOALS];
        for (int slot = 0; slot < agent->goal_count; slot++) {
            old_x[slot] = agent->list_goal_x[slot];
            old_y[slot] = agent->list_goal_y[slot];
            old_lane[slot] = agent->list_goal_lane[slot];
        }
        if (roll_goals(&env, agent) != 1) {
            continue; // frontier walk dead-ended for this agent; the caller would regen instead
        }
        rolled++;
        EXPECT_EQ_INT(agent->current_goal_idx, 0);
        for (int slot = 0; slot < env.num_goals - 1; slot++) {
            EXPECT_NEAR(agent->list_goal_x[slot], old_x[slot + 1], 1e-6f);
            EXPECT_NEAR(agent->list_goal_y[slot], old_y[slot + 1], 1e-6f);
            EXPECT_EQ_INT(agent->list_goal_lane[slot], old_lane[slot + 1]);
        }
        int last = env.num_goals - 1;
        EXPECT_FINITE(agent->list_goal_x[last]);
        EXPECT_FINITE(agent->list_goal_y[last]);
        EXPECT_TRUE(agent->list_goal_lane[last] >= 0);
        EXPECT_NEAR(agent->current_goal_x, agent->list_goal_x[0], 1e-6f);
    }
    EXPECT_TRUE(rolled > 0);
    free_allocated(&env);
    return 0;
}

static int test_roll_goals_bails_on_replay_pins(void) {
    // laneless replay goals cannot seed a frontier walk, so roll_goals refuses instead of fabricating one
    Drive env = {0};
    env.goal_source = GOAL_SOURCE_ROUTE;
    Agent agent = {0};
    agent.goal_count = 3;
    for (int g = 0; g < agent.goal_count; g++) {
        agent.list_goal_lane[g] = -1;
    }
    EXPECT_EQ_INT(roll_goals(&env, &agent), 0);
    return 0;
}

static int test_gt_goals_along_trajectory_are_laneless(void) {
    // GT goals: finite, laneless, along the log; a spawn inside the first goal radius may advance the index
    srand(17);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.goal_source = GOAL_SOURCE_GT;
    allocate(&env);
    c_reset(&env);

    EXPECT_TRUE(env.active_agent_count > 0);
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        EXPECT_EQ_INT(agent->goal_count, env.num_goals);
        EXPECT_TRUE(agent->current_goal_idx >= 0 && agent->current_goal_idx <= agent->goal_count);
        for (int slot = 0; slot < agent->goal_count; slot++) {
            EXPECT_EQ_INT(agent->list_goal_lane[slot], -1);
            EXPECT_FINITE(agent->list_goal_x[slot]);
            EXPECT_FINITE(agent->list_goal_y[slot]);
            EXPECT_FINITE(agent->list_goal_z[slot]);
        }
    }
    free_allocated(&env);
    return 0;
}

static int test_gt_map_goals_snap_to_lane_centers(void) {
    // GT_MAP goals snap within the snap radius onto a co-directional lane, the others keep the raw point
    srand(17);
    Drive raw_env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    raw_env.goal_source = GOAL_SOURCE_GT;
    allocate(&raw_env);
    c_reset(&raw_env);
    srand(17);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.goal_source = GOAL_SOURCE_GT_MAP;
    allocate(&env);
    c_reset(&env);

    EXPECT_TRUE(env.active_agent_count > 0);
    EXPECT_EQ_INT(env.active_agent_count, raw_env.active_agent_count);
    int snapped_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        EXPECT_EQ_INT(env.active_agent_indices[i], raw_env.active_agent_indices[i]);
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        Agent *raw_agent = &raw_env.agents[raw_env.active_agent_indices[i]];
        EXPECT_EQ_INT(agent->goal_count, env.num_goals);
        for (int slot = 0; slot < agent->goal_count; slot++) {
            float dx = agent->list_goal_x[slot] - raw_agent->list_goal_x[slot];
            float dy = agent->list_goal_y[slot] - raw_agent->list_goal_y[slot];
            float moved_m = sqrtf(dx * dx + dy * dy);
            EXPECT_FINITE(agent->list_goal_x[slot]);
            EXPECT_FINITE(agent->list_goal_y[slot]);
            EXPECT_NEAR(agent->list_goal_z[slot], raw_agent->list_goal_z[slot], 0.0f);
            EXPECT_TRUE(moved_m <= GOAL_LANE_SNAP_MAX_DIST_M + 1e-3f);
            if (agent->list_goal_lane[slot] == -1) {
                EXPECT_NEAR(moved_m, 0.0f, 0.0f);
                continue;
            }
            EXPECT_TRUE(is_drivable_road_lane(env.road_elements[agent->list_goal_lane[slot]].type));
            snapped_count++;
        }
    }
    printf("  gt_map goals snapped to a lane: %d/%d\n", snapped_count, env.active_agent_count * env.num_goals);
    EXPECT_TRUE(snapped_count > 0);
    free_allocated(&env);
    free_allocated(&raw_env);
    return 0;
}

static int test_step_observations_follow_goal_updates(void) {
    const int goal_sources[] = {GOAL_SOURCE_MAP, GOAL_SOURCE_ROUTE, GOAL_SOURCE_ROUTE};
    const int regen_modes[] = {GOAL_REGEN_FINITE, GOAL_REGEN_FINITE, GOAL_REGEN_ROLLING};
    const int goal_case_count = sizeof(goal_sources) / sizeof(goal_sources[0]);
    for (int case_idx = 0; case_idx < goal_case_count; case_idx++) {
        Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
        env.goal_source = goal_sources[case_idx];
        env.goal_regen_mode = regen_modes[case_idx];
        env.goal_reach_requires_speed = 1;
        allocate(&env);
        c_reset(&env);

        Agent *agent = &env.agents[env.active_agent_indices[0]];
        int reached_goal_idx = env.goal_regen_mode == GOAL_REGEN_ROLLING ? 0 : agent->goal_count - 1;
        agent->current_goal_idx = reached_goal_idx;
        agent->current_goal_x = agent->list_goal_x[reached_goal_idx] = agent->sim_x;
        agent->current_goal_y = agent->list_goal_y[reached_goal_idx] = agent->sim_y;
        agent->current_goal_z = agent->list_goal_z[reached_goal_idx] = agent->sim_z;
        agent->list_goal_lane[reached_goal_idx] = agent->current_lane_idx;
        drive_set_neutral_actions(&env);

        c_step(&env);

        EXPECT_FALSE(agent->removed);
        EXPECT_FALSE(env.terminals[0]);
        EXPECT_FALSE(env.truncations[0]);
        EXPECT_EQ_INT(agent->current_goal_idx, 0);
        EXPECT_NEAR(env.logs[0].reward_goal, env.reward_goal, 1e-6f);
        float goal_dx = agent->current_goal_x - agent->sim_x;
        float goal_dy = agent->current_goal_y - agent->sim_y;
        float goal_norm = fmaxf(env.obs_norm_goal_offset_m, sqrtf(goal_dx * goal_dx + goal_dy * goal_dy));
        EXPECT_NEAR(
            env.observations[EGO_FEATURES],
            (goal_dx * agent->cos_heading + goal_dy * agent->sin_heading) / goal_norm,
            1e-6f);
        EXPECT_NEAR(
            env.observations[EGO_FEATURES + 1],
            (-goal_dx * agent->sin_heading + goal_dy * agent->cos_heading) / goal_norm,
            1e-6f);

        // Recomputing from the returned state must also preserve the lane-to-goal routing features.
        int observation_size = compute_observation_size(&env);
        float step_observations[observation_size];
        memcpy(step_observations, env.observations, sizeof(step_observations));
        compute_observations(&env);
        EXPECT_TRUE(memcmp(step_observations, env.observations, sizeof(step_observations)) == 0);
        free_allocated(&env);
    }
    return 0;
}

static int test_goal_update_skips_stopped_and_removed_agents(void) {
    // neither runs compute_metrics, so a goal flag from an earlier step must not roll or renew their goals
    srand(23);
    const int regen_modes[] = {GOAL_REGEN_ROLLING, GOAL_REGEN_FINITE};
    const int regen_case_count = sizeof(regen_modes) / sizeof(regen_modes[0]);
    for (int case_idx = 0; case_idx < regen_case_count; case_idx++) {
        Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
        env.goal_source = GOAL_SOURCE_MAP;
        env.goal_regen_mode = regen_modes[case_idx];
        allocate(&env);
        c_reset(&env);

        Agent *agent = &env.agents[env.active_agent_indices[0]];
        agent->metrics_array[REACHED_GOAL_IDX] = 1.0f;
        if (env.goal_regen_mode == GOAL_REGEN_ROLLING) {
            agent->stopped = 1;
            agent->current_goal_idx = 1;
        } else {
            invalidate_agent(agent);
            agent->removed = 1;
            agent->current_goal_idx = agent->goal_count;
        }
        int goal_idx_before = agent->current_goal_idx;
        float goal_x_before[MAX_GOALS], goal_y_before[MAX_GOALS];
        memcpy(goal_x_before, agent->list_goal_x, sizeof(goal_x_before));
        memcpy(goal_y_before, agent->list_goal_y, sizeof(goal_y_before));
        drive_set_neutral_actions(&env);

        c_step(&env);

        EXPECT_EQ_INT(agent->current_goal_idx, goal_idx_before);
        EXPECT_TRUE(memcmp(goal_x_before, agent->list_goal_x, sizeof(goal_x_before)) == 0);
        EXPECT_TRUE(memcmp(goal_y_before, agent->list_goal_y, sizeof(goal_y_before)) == 0);
        free_allocated(&env);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_commit_goals_front_align_fills_every_slot);
    RUN_TEST(test_commit_goals_back_align_pads_front_with_lane_minus_one);
    RUN_TEST(test_map_goal_source_no_attrition);
    RUN_TEST(test_map_goals_carry_lane_and_track_slot_zero);
    RUN_TEST(test_map_goal_heading_constraint_reduces_turns);
    RUN_TEST(test_route_goals_full_set_or_removed);
    RUN_TEST(test_route_goals_front_aligned_with_lanes);
    RUN_TEST(test_roll_goals_slides_window_and_appends);
    RUN_TEST(test_roll_goals_bails_on_replay_pins);
    RUN_TEST(test_step_observations_follow_goal_updates);
    RUN_TEST(test_goal_update_skips_stopped_and_removed_agents);
    RUN_TEST(test_gt_goals_along_trajectory_are_laneless);
    RUN_TEST(test_gt_map_goals_snap_to_lane_centers);
    return test_summary(failures);
}
