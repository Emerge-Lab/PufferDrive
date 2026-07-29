#include "include/drive_fixture.h"
#include "include/test.h"

// Covers the goal-system rewrite paths: the commit_goals slot layout (front/back align + lane=-1
// padding), map goal_source spawn/regen carrying lane indices (the GPS lane-distance precondition),
// the route full-set invariant and front-alignment, roll_goals window slide plus its refusal to roll
// trajectory-pinned replay goals, and ground-truth goals placed along the logged trajectory.

// ---------------------------------------------------------------------------
// commit_goals: pure slot-layout logic (reads only env->num_goals).
// ---------------------------------------------------------------------------

static int test_commit_goals_front_align_fills_every_slot(void) {
    // Front-align (start_slot 0) with a full set: all num_goals slots carry a placed goal, goal_count
    // is the full set, and the current_goal_* alias tracks slot 0.
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
    // Back-align a partial set (route/replay exhaustion path): the placed goals sit at the tail, the
    // leading unused slots are zeroed with lane = -1, and current_goal_idx points at the first placed slot.
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

// ---------------------------------------------------------------------------
// Map goal source (free-roam walk, no route).
// ---------------------------------------------------------------------------

static int test_map_goal_source_no_attrition(void) {
    // Map source used to remove any agent whose single seed draw dead-ended, permanently
    // for the env lifetime. With the fresh-cell retry, every active agent spawns.
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
    // Map goals must record a lane index for every goal in the obs window so the GPS lane-distance
    // feature can look them up; a -1 there would silently zero-fill it. current_goal_idx front-aligns
    // at 0 and the current_goal_* alias must mirror slot 0.
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

// ---------------------------------------------------------------------------
// Route goal source (walks the agent's own route).
// ---------------------------------------------------------------------------

static int test_route_goals_full_set_or_removed(void) {
    // The route path front-aligns the full num_goals set or (after a route retry) removes the
    // agent; it must never leave a live agent with a partial set.
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

// ---------------------------------------------------------------------------
// roll_goals: rolling-window regen.
// ---------------------------------------------------------------------------

static int test_roll_goals_slides_window_and_appends(void) {
    // Rolling regen drops the reached goal, shifts every later goal one slot toward the front, and
    // appends a fresh frontier goal in the last slot while keeping current_goal_idx pinned to 0.
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
    // Replay goals track the logged trajectory and carry list_goal_lane = -1. roll_goals cannot
    // seed a frontier walk from a laneless goal, so it must bail (return 0) rather than fabricate
    // one; the goal-update loop then routes replay away from rolling entirely.
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

// ---------------------------------------------------------------------------
// Ground-truth goal source (replay: goals sampled off the logged trajectory).
// ---------------------------------------------------------------------------

static int test_gt_goals_along_trajectory_are_laneless(void) {
    // GT source spaces num_goals goals along the logged trajectory. They carry no lane (list_goal_lane
    // = -1, so no GPS lane-distance) and every coordinate is finite. current_goal_idx starts at 0 but
    // may already have advanced if the agent spawns inside the first goal's radius (metrics run in reset).
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

int main(void) {
    int failures = 0;
    RUN_TEST(test_commit_goals_front_align_fills_every_slot);
    RUN_TEST(test_commit_goals_back_align_pads_front_with_lane_minus_one);
    RUN_TEST(test_map_goal_source_no_attrition);
    RUN_TEST(test_map_goals_carry_lane_and_track_slot_zero);
    RUN_TEST(test_route_goals_full_set_or_removed);
    RUN_TEST(test_route_goals_front_aligned_with_lanes);
    RUN_TEST(test_roll_goals_slides_window_and_appends);
    RUN_TEST(test_roll_goals_bails_on_replay_pins);
    RUN_TEST(test_gt_goals_along_trajectory_are_laneless);
    return test_summary(failures);
}
