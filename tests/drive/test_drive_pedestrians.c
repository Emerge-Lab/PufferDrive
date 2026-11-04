#include "include/drive_fixture.h"
#include "include/test.h"

static Drive make_pedestrian_env(int num_agents, int pedestrian_slots, float spawn_initial_speed) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.scripted_pedestrian_slots = pedestrian_slots;
    env.spawn_initial_speed = spawn_initial_speed;
    env.scenario_length = 400;
    allocate(&env);
    c_reset(&env);
    return env;
}

static int first_crossing_pedestrian(Drive *env) {
    for (int k = 0; k < env->scripted_pedestrian_slots; k++) {
        if (env->pedestrians[k].state == PEDESTRIAN_STATE_CROSSING) {
            return k;
        }
    }
    return -1;
}

static int step_until_crossing(Drive *env, int max_steps) {
    for (int k = 0; k < env->scripted_pedestrian_slots; k++) {
        env->pedestrians[k].timer_s = 0.0f;
    }
    for (int t = 0; t < max_steps; t++) {
        drive_set_neutral_actions(env);
        c_step(env);
        int crossing_k = first_crossing_pedestrian(env);
        if (crossing_k != -1) {
            return crossing_k;
        }
    }
    return -1;
}

static int test_ray_hits_segment(void) {
    float t = -1.0f;
    EXPECT_TRUE(ray_hits_segment(0.0f, 0.0f, 1.0f, 0.0f, 5.0f, -1.0f, 5.0f, 1.0f, &t));
    EXPECT_NEAR(t, 5.0f, 1e-5f);
    EXPECT_FALSE(ray_hits_segment(0.0f, 0.0f, -1.0f, 0.0f, 5.0f, -1.0f, 5.0f, 1.0f, &t));
    EXPECT_FALSE(ray_hits_segment(0.0f, 0.0f, 1.0f, 0.0f, 5.0f, 1.0f, 5.0f, 3.0f, &t));
    EXPECT_FALSE(ray_hits_segment(0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 8.0f, 0.0f, &t));
    EXPECT_TRUE(ray_hits_segment(1.0f, 1.0f, 0.0f, 1.0f, -3.0f, 4.0f, 3.0f, 4.0f, &t));
    EXPECT_NEAR(t, 3.0f, 1e-5f);
    return 0;
}

static int test_pedestrian_slots_are_hidden_static_partners(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 3, 0.0f);
    EXPECT_EQ_INT(env.static_agent_count, 3);
    EXPECT_EQ_INT(env.num_total_agents, 16 + 3);
    EXPECT_EQ_INT(env.num_agents, env.active_agent_count + 3);
    for (int k = 0; k < 3; k++) {
        ScriptedPedestrian *pedestrian = &env.pedestrians[k];
        Agent *agent = &env.agents[pedestrian->agent_idx];
        EXPECT_EQ_INT(env.static_agent_indices[k], pedestrian->agent_idx);
        EXPECT_EQ_INT(agent->type, PEDESTRIAN);
        EXPECT_EQ_INT(agent->sim_valid, 0);
        EXPECT_EQ_INT(agent->active_agent, 0);
        EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_IDLE);
        EXPECT_EQ_INT(pedestrian->target_agent_idx, -1);
        EXPECT_TRUE(pedestrian->timer_s >= PEDESTRIAN_COOLDOWN_MIN_S);
        EXPECT_TRUE(pedestrian->timer_s <= PEDESTRIAN_COOLDOWN_MAX_S);
    }
    free_allocated(&env);
    return 0;
}

static int test_pedestrian_slots_keep_initial_spawns(void) {
    srand(7);
    Drive without = make_pedestrian_env(16, 0, 0.0f);
    srand(7);
    Drive with = make_pedestrian_env(16, 2, 0.0f);
    EXPECT_EQ_INT(with.active_agent_count, without.active_agent_count);
    for (int i = 0; i < without.active_agent_count; i++) {
        Agent *a = &without.agents[without.active_agent_indices[i]];
        Agent *b = &with.agents[with.active_agent_indices[i]];
        EXPECT_NEAR(b->sim_x, a->sim_x, 0.0f);
        EXPECT_NEAR(b->sim_y, a->sim_y, 0.0f);
        EXPECT_NEAR(b->sim_heading, a->sim_heading, 0.0f);
        EXPECT_NEAR(b->sim_length, a->sim_length, 0.0f);
    }
    free_allocated(&without);
    free_allocated(&with);
    return 0;
}

static int test_pedestrian_crossing_geometry_and_lifecycle(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 2, 5.0f);
    EXPECT_TRUE(env.active_agent_count >= 4);
    int crossing_k = step_until_crossing(&env, 60);
    EXPECT_TRUE(crossing_k != -1);
    ScriptedPedestrian *pedestrian = &env.pedestrians[crossing_k];
    Agent *agent = &env.agents[pedestrian->agent_idx];
    EXPECT_TRUE(pedestrian->target_agent_idx >= 0);
    Agent *car = &env.agents[pedestrian->target_agent_idx];

    EXPECT_EQ_INT(agent->sim_valid, 1);
    EXPECT_EQ_INT(agent->type, PEDESTRIAN);
    EXPECT_TRUE(agent->sim_length >= PEDESTRIAN_SIZE_MIN_M && agent->sim_length <= PEDESTRIAN_SIZE_MAX_M);
    EXPECT_TRUE(agent->sim_width >= PEDESTRIAN_SIZE_MIN_M && agent->sim_width <= PEDESTRIAN_SIZE_MAX_M);
    EXPECT_NEAR(agent->sim_height, PEDESTRIAN_HEIGHT_M, 0.0f);
    float min_speed_mps = PEDESTRIAN_LB1_SPEED_BASE_MPS + PEDESTRIAN_LB1_SPEED_PER_LANE_MPS - PEDESTRIAN_SPEED_JITTER_MPS;
    float max_speed_mps
        = PEDESTRIAN_LB1_SPEED_BASE_MPS + 3.0f * PEDESTRIAN_LB1_SPEED_PER_LANE_MPS + PEDESTRIAN_SPEED_JITTER_MPS;
    EXPECT_TRUE(pedestrian->speed_mps >= min_speed_mps && pedestrian->speed_mps <= max_speed_mps);
    EXPECT_NEAR(agent->sim_speed, 0.0f, 0.0f);
    EXPECT_TRUE(pedestrian->timer_s >= PEDESTRIAN_WAIT_MIN_S && pedestrian->timer_s <= PEDESTRIAN_WAIT_MAX_S);
    EXPECT_TRUE(pedestrian->ramp_s >= PEDESTRIAN_RAMP_MIN_S && pedestrian->ramp_s <= PEDESTRIAN_RAMP_MAX_S);
    EXPECT_NEAR(pedestrian->dir_x * pedestrian->dir_x + pedestrian->dir_y * pedestrian->dir_y, 1.0f, 1e-4f);
    EXPECT_NEAR(agent->cos_heading, pedestrian->dir_x, 1e-5f);
    EXPECT_NEAR(agent->sin_heading, pedestrian->dir_y, 1e-5f);
    float min_crossing_m = PEDESTRIAN_KERB_OFFSET_MIN_M + PEDESTRIAN_FAR_KERB_OVERSHOOT_M;
    float max_crossing_m
        = PEDESTRIAN_KERB_OFFSET_MAX_M + PEDESTRIAN_FAR_KERB_OVERSHOOT_M + 2.0f * PEDESTRIAN_KERB_SEARCH_MAX_M;
    EXPECT_TRUE(pedestrian->crossing_m > min_crossing_m && pedestrian->crossing_m < max_crossing_m);
    EXPECT_TRUE(pedestrian->remaining_m > 0.0f && pedestrian->remaining_m <= pedestrian->crossing_m);

    // Walks across the target car's path, starting ahead of it on its right
    float along_car = car->cos_heading * pedestrian->dir_x + car->sin_heading * pedestrian->dir_y;
    EXPECT_TRUE(fabsf(along_car) < 0.5f);
    float rel_x, rel_y;
    project_point_to_ego_frame(car, agent->sim_x, agent->sim_y, &rel_x, &rel_y);
    float max_appear_m
        = PEDESTRIAN_LB1_APPEAR_BASE_M + 2.0f * PEDESTRIAN_LB1_APPEAR_PER_LANE_M + PEDESTRIAN_APPEAR_JITTER_M;
    EXPECT_TRUE(rel_x > 0.0f && rel_x < max_appear_m + 5.0f);
    EXPECT_TRUE(rel_y < 0.0f);

    int target_slot = -1;
    for (int j = 0; j < env.active_agent_count; j++) {
        if (env.active_agent_indices[j] == pedestrian->target_agent_idx) {
            target_slot = j;
        }
    }
    EXPECT_TRUE(target_slot != -1);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_crossings, 1.0f, 0.0f);

    int saw_linger = 0;
    for (int t = 0; t < 250 && pedestrian->state != PEDESTRIAN_STATE_IDLE; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        saw_linger |= pedestrian->state == PEDESTRIAN_STATE_LINGER;
    }
    EXPECT_TRUE(saw_linger);
    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_IDLE);
    EXPECT_EQ_INT(agent->sim_valid, 0);
    EXPECT_EQ_INT(pedestrian->target_agent_idx, -1);
    EXPECT_TRUE(pedestrian->timer_s >= PEDESTRIAN_COOLDOWN_MIN_S);
    free_allocated(&env);
    return 0;
}

static int test_car_hitting_pedestrian_logs_pedestrian_collision(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 2, 5.0f);
    int crossing_k = step_until_crossing(&env, 60);
    EXPECT_TRUE(crossing_k != -1);
    ScriptedPedestrian *pedestrian = &env.pedestrians[crossing_k];
    Agent *agent = &env.agents[pedestrian->agent_idx];

    // Put the walker mid-crossing and park a policy car on it, aligned with the road
    float mid_offset_m = pedestrian->remaining_m - 0.5f * pedestrian->crossing_m;
    float mid_x = agent->sim_x + pedestrian->dir_x * mid_offset_m;
    float mid_y = agent->sim_y + pedestrian->dir_y * mid_offset_m;
    agent->sim_x = mid_x;
    agent->sim_y = mid_y;
    copy_pose_to_prev(agent);

    int car_slot = 0;
    int car_idx = env.active_agent_indices[car_slot];
    Agent *car = &env.agents[car_idx];
    car->sim_x = mid_x;
    car->sim_y = mid_y;
    car->sim_z = agent->sim_z;
    car->sim_heading = atan2f(-pedestrian->dir_x, pedestrian->dir_y);
    car->cos_heading = cosf(car->sim_heading);
    car->sin_heading = sinf(car->sim_heading);
    copy_pose_to_prev(car);
    car->stopped = 0;
    car->removed = 0;
    car->first_collision_partner_idx = -1;
    env.logs[car_slot] = (Log) {0};

    compute_metrics(&env, car_idx, car_slot);
    EXPECT_NEAR(car->metrics_array[OFFROAD_IDX], 0.0f, 0.0f);
    EXPECT_NEAR(car->metrics_array[COLLISION_IDX], 1.0f, 0.0f);
    EXPECT_EQ_INT(car->first_collision_partner_idx, pedestrian->agent_idx);
    compute_rewards(&env, car_slot);
    EXPECT_NEAR(env.logs[car_slot].collision_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.logs[car_slot].pedestrian_collision_rate, 1.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

static int test_pedestrian_stands_then_ramps_to_speed(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 2, 5.0f);
    int crossing_k = step_until_crossing(&env, 60);
    EXPECT_TRUE(crossing_k != -1);
    ScriptedPedestrian *pedestrian = &env.pedestrians[crossing_k];
    Agent *agent = &env.agents[pedestrian->agent_idx];
    float start_x = agent->sim_x;
    float start_y = agent->sim_y;
    int standing_steps = 0;
    while (pedestrian->timer_s > 0.0f && standing_steps < 10) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        standing_steps++;
        EXPECT_NEAR(agent->sim_x, start_x, 0.0f);
        EXPECT_NEAR(agent->sim_y, start_y, 0.0f);
        EXPECT_NEAR(agent->sim_speed, 0.0f, 0.0f);
    }
    EXPECT_TRUE(standing_steps <= (int) ceilf(PEDESTRIAN_WAIT_MAX_S / env.dt) + 1);
    float previous_speed_mps = 0.0f;
    int ramp_steps = (int) ceilf(pedestrian->ramp_s / env.dt);
    for (int t = 0; t < ramp_steps; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        EXPECT_TRUE(agent->sim_speed > previous_speed_mps || previous_speed_mps >= pedestrian->speed_mps - 1e-4f);
        EXPECT_TRUE(agent->sim_speed <= pedestrian->speed_mps + 1e-4f);
        previous_speed_mps = agent->sim_speed;
    }
    EXPECT_NEAR(agent->sim_speed, pedestrian->speed_mps, 1e-4f);
    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_CROSSING);
    EXPECT_TRUE(hypotf(agent->sim_x - start_x, agent->sim_y - start_y) > 0.0f);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_ray_hits_segment);
    RUN_TEST(test_pedestrian_slots_are_hidden_static_partners);
    RUN_TEST(test_pedestrian_slots_keep_initial_spawns);
    RUN_TEST(test_pedestrian_crossing_geometry_and_lifecycle);
    RUN_TEST(test_pedestrian_stands_then_ramps_to_speed);
    RUN_TEST(test_car_hitting_pedestrian_logs_pedestrian_collision);
    return test_summary(failures);
}
