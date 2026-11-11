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

static int first_visible_pedestrian(Drive *env) {
    for (int k = 0; k < env->scripted_pedestrian_slots; k++) {
        if (env->pedestrians[k].state != PEDESTRIAN_STATE_IDLE) {
            return k;
        }
    }
    return -1;
}

// Every slot appears in `mode` as soon as a car takes it
static int step_until_appearance(Drive *env, int mode, int max_steps) {
    for (int k = 0; k < env->scripted_pedestrian_slots; k++) {
        env->pedestrians[k].timer_s = 0.0f;
        env->pedestrians[k].mode = mode;
    }
    for (int t = 0; t < max_steps; t++) {
        drive_set_neutral_actions(env);
        c_step(env);
        int visible_k = first_visible_pedestrian(env);
        if (visible_k != -1) {
            return visible_k;
        }
    }
    return -1;
}

// Road edges between a point and the nearest lane vertex: an odd count means the point lies outside the road
static int road_edges_to_nearest_lane(Drive *env, float px, float py) {
    float nearest_sq = 1e18f;
    float lane_x = px;
    float lane_y = py;
    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *element = &env->road_elements[i];
        if (!is_road_lane(element->type)) {
            continue;
        }
        for (int v = 0; v < element->segment_size; v++) {
            float dist_sq = (element->x[v] - px) * (element->x[v] - px) + (element->y[v] - py) * (element->y[v] - py);
            if (dist_sq < nearest_sq) {
                nearest_sq = dist_sq;
                lane_x = element->x[v];
                lane_y = element->y[v];
            }
        }
    }
    float reach_m = sqrtf(nearest_sq);
    float dir_x = (lane_x - px) / reach_m;
    float dir_y = (lane_y - py) / reach_m;
    int edge_count = 0;
    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *element = &env->road_elements[i];
        if (!is_road_edge(element->type)) {
            continue;
        }
        for (int v = 0; v + 1 < element->segment_size; v++) {
            float t;
            bool hit = ray_hits_segment(
                px,
                py,
                dir_x,
                dir_y,
                element->x[v],
                element->y[v],
                element->x[v + 1],
                element->y[v + 1],
                &t);
            edge_count += hit && t < reach_m;
        }
    }
    return edge_count;
}

static int target_slot_of(Drive *env, const ScriptedPedestrian *pedestrian) {
    for (int j = 0; j < env->active_agent_count; j++) {
        if (env->active_agent_indices[j] == pedestrian->target_agent_idx) {
            return j;
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
        EXPECT_TRUE(pedestrian->mode >= PEDESTRIAN_MODE_POP_IN_CROSS && pedestrian->mode <= PEDESTRIAN_MODE_KERB_STEP);
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

static int test_pop_in_crossing_geometry_and_lifecycle(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 2, 5.0f);
    EXPECT_TRUE(env.active_agent_count >= 4);
    int crossing_k = step_until_appearance(&env, PEDESTRIAN_MODE_POP_IN_CROSS, 60);
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
    float min_speed_mps
        = PEDESTRIAN_LB1_SPEED_BASE_MPS + PEDESTRIAN_LB1_SPEED_PER_LANE_MPS - PEDESTRIAN_SPEED_JITTER_MPS;
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
    EXPECT_TRUE(pedestrian->path_m > min_crossing_m && pedestrian->path_m < max_crossing_m);
    EXPECT_TRUE(pedestrian->remaining_m > 0.0f && pedestrian->remaining_m <= pedestrian->path_m);

    // Walks across the target car's path, starting ahead of it on its right
    float along_car = car->cos_heading * pedestrian->dir_x + car->sin_heading * pedestrian->dir_y;
    EXPECT_TRUE(fabsf(along_car) < 0.5f);
    float rel_x, rel_y;
    project_point_to_ego_frame(car, agent->sim_x, agent->sim_y, &rel_x, &rel_y);
    float max_appear_m
        = PEDESTRIAN_LB1_APPEAR_BASE_M + 2.0f * PEDESTRIAN_LB1_APPEAR_PER_LANE_M + PEDESTRIAN_APPEAR_JITTER_M;
    EXPECT_TRUE(rel_x > 0.0f && rel_x < max_appear_m + 5.0f);
    EXPECT_TRUE(rel_y < 0.0f);

    int target_slot = target_slot_of(&env, pedestrian);
    EXPECT_TRUE(target_slot != -1);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_crossings, 1.0f, 0.0f);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_bystanders, 0.0f, 0.0f);
    EXPECT_TRUE(pedestrian->stand_s >= PEDESTRIAN_LINGER_MIN_S && pedestrian->stand_s <= PEDESTRIAN_LINGER_MAX_S);

    int saw_standing = 0;
    for (int t = 0; t < 250 && pedestrian->state != PEDESTRIAN_STATE_IDLE; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        saw_standing |= pedestrian->state == PEDESTRIAN_STATE_STANDING;
    }
    EXPECT_TRUE(saw_standing);
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
    int crossing_k = step_until_appearance(&env, PEDESTRIAN_MODE_POP_IN_CROSS, 60);
    EXPECT_TRUE(crossing_k != -1);
    ScriptedPedestrian *pedestrian = &env.pedestrians[crossing_k];
    Agent *agent = &env.agents[pedestrian->agent_idx];

    // Put the walker mid-crossing and park a policy car on it, aligned with the road
    float mid_offset_m = pedestrian->remaining_m - 0.5f * pedestrian->path_m;
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

static int test_pop_in_crosser_stands_then_ramps_to_speed(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 2, 5.0f);
    int crossing_k = step_until_appearance(&env, PEDESTRIAN_MODE_POP_IN_CROSS, 60);
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
    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_WALKING);
    EXPECT_TRUE(hypotf(agent->sim_x - start_x, agent->sim_y - start_y) > 0.0f);
    free_allocated(&env);
    return 0;
}

static int test_pedestrian_mode_shares(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 1, 0.0f);
    const int draw_count = 20000;
    int mode_counts[PEDESTRIAN_MODE_KERB_STEP + 1] = {0};
    for (int i = 0; i < draw_count; i++) {
        mode_counts[sample_pedestrian_mode(&env)]++;
    }
    float kerb_step_prob = 1.0f - PEDESTRIAN_POP_IN_CROSS_PROB - PEDESTRIAN_SLOW_CROSS_PROB - PEDESTRIAN_STAND_PROB
        - PEDESTRIAN_KERB_WALK_PROB;
    EXPECT_TRUE(kerb_step_prob > 0.0f);
    EXPECT_NEAR((float) mode_counts[PEDESTRIAN_MODE_POP_IN_CROSS] / draw_count, PEDESTRIAN_POP_IN_CROSS_PROB, 0.02f);
    EXPECT_NEAR((float) mode_counts[PEDESTRIAN_MODE_SLOW_CROSS] / draw_count, PEDESTRIAN_SLOW_CROSS_PROB, 0.02f);
    EXPECT_NEAR((float) mode_counts[PEDESTRIAN_MODE_STAND] / draw_count, PEDESTRIAN_STAND_PROB, 0.02f);
    EXPECT_NEAR((float) mode_counts[PEDESTRIAN_MODE_KERB_WALK] / draw_count, PEDESTRIAN_KERB_WALK_PROB, 0.02f);
    EXPECT_NEAR((float) mode_counts[PEDESTRIAN_MODE_KERB_STEP] / draw_count, kerb_step_prob, 0.02f);
    free_allocated(&env);
    return 0;
}

static int test_slow_crosser_walks_from_a_kerb_across_the_road(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 1, 5.0f);
    int visible_k = step_until_appearance(&env, PEDESTRIAN_MODE_SLOW_CROSS, 100);
    EXPECT_TRUE(visible_k != -1);
    ScriptedPedestrian *pedestrian = &env.pedestrians[visible_k];
    Agent *agent = &env.agents[pedestrian->agent_idx];
    Agent *car = &env.agents[pedestrian->target_agent_idx];

    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_WALKING);
    EXPECT_TRUE(pedestrian->speed_mps >= PEDESTRIAN_SLOW_SPEED_MIN_MPS);
    EXPECT_TRUE(pedestrian->speed_mps <= PEDESTRIAN_SLOW_SPEED_MAX_MPS);
    EXPECT_TRUE(pedestrian->timer_s >= 0.0f && pedestrian->timer_s <= PEDESTRIAN_SLOW_WAIT_MAX_S);
    EXPECT_NEAR(pedestrian->dir_x * pedestrian->dir_x + pedestrian->dir_y * pedestrian->dir_y, 1.0f, 1e-4f);
    EXPECT_TRUE(hypotf(agent->sim_x - car->sim_x, agent->sim_y - car->sim_y) > 0.5f * PEDESTRIAN_SLOW_APPEAR_MIN_M);
    int target_slot = target_slot_of(&env, pedestrian);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_crossings, 1.0f, 0.0f);

    // Kerb, road, kerb
    float end_x = agent->sim_x + pedestrian->dir_x * pedestrian->path_m;
    float end_y = agent->sim_y + pedestrian->dir_y * pedestrian->path_m;
    EXPECT_EQ_INT(road_edges_to_nearest_lane(&env, agent->sim_x, agent->sim_y) % 2, 1);
    EXPECT_EQ_INT(road_edges_to_nearest_lane(&env, end_x, end_y) % 2, 1);
    EXPECT_EQ_INT(road_edges_to_nearest_lane(&env, 0.5f * (agent->sim_x + end_x), 0.5f * (agent->sim_y + end_y)), 0);

    for (int t = 0; t < 350 && pedestrian->state == PEDESTRIAN_STATE_WALKING; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        EXPECT_TRUE(agent->sim_speed <= pedestrian->speed_mps + 1e-4f);
    }
    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_STANDING);
    EXPECT_NEAR(agent->sim_x, end_x, 1e-2f);
    EXPECT_NEAR(agent->sim_y, end_y, 1e-2f);
    free_allocated(&env);
    return 0;
}

static int test_stander_stays_beside_the_road_with_a_running_stopped_time(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 1, 5.0f);
    int visible_k = step_until_appearance(&env, PEDESTRIAN_MODE_STAND, 100);
    EXPECT_TRUE(visible_k != -1);
    ScriptedPedestrian *pedestrian = &env.pedestrians[visible_k];
    Agent *agent = &env.agents[pedestrian->agent_idx];

    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_STANDING);
    EXPECT_TRUE(pedestrian->timer_s >= PEDESTRIAN_BYSTANDER_STAND_MIN_S);
    EXPECT_TRUE(pedestrian->timer_s <= PEDESTRIAN_BYSTANDER_STAND_MAX_S);
    EXPECT_TRUE(agent->seconds_stopped >= 0.0f && agent->seconds_stopped <= MAX_STOPPED_SECONDS + env.dt);
    EXPECT_EQ_INT(road_edges_to_nearest_lane(&env, agent->sim_x, agent->sim_y) % 2, 1);
    int target_slot = target_slot_of(&env, pedestrian);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_bystanders, 1.0f, 0.0f);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_crossings, 0.0f, 0.0f);

    float start_x = agent->sim_x;
    float start_y = agent->sim_y;
    float stopped_s = agent->seconds_stopped;
    int standing_steps = (int) (PEDESTRIAN_BYSTANDER_STAND_MIN_S / env.dt) - 2;
    for (int t = 0; t < standing_steps; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_STANDING);
        EXPECT_NEAR(agent->sim_x, start_x, 0.0f);
        EXPECT_NEAR(agent->sim_y, start_y, 0.0f);
        EXPECT_NEAR(agent->sim_speed, 0.0f, 0.0f);
    }
    EXPECT_NEAR(agent->seconds_stopped, stopped_s + (float) standing_steps * env.dt, 1e-2f);
    free_allocated(&env);
    return 0;
}

static int test_kerb_walker_stays_off_the_road(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 1, 5.0f);
    int visible_k = step_until_appearance(&env, PEDESTRIAN_MODE_KERB_WALK, 150);
    EXPECT_TRUE(visible_k != -1);
    ScriptedPedestrian *pedestrian = &env.pedestrians[visible_k];
    Agent *agent = &env.agents[pedestrian->agent_idx];

    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_WALKING);
    EXPECT_TRUE(pedestrian->speed_mps >= PEDESTRIAN_KERB_WALK_SPEED_MIN_MPS);
    EXPECT_TRUE(pedestrian->speed_mps <= PEDESTRIAN_KERB_WALK_SPEED_MAX_MPS);
    EXPECT_NEAR(agent->sim_speed, pedestrian->speed_mps, 1e-4f);
    EXPECT_NEAR(agent->seconds_stopped, 0.0f, 0.0f);
    EXPECT_TRUE(pedestrian->path_m > PEDESTRIAN_KERB_WALK_MIN_M - 2.0f * PEDESTRIAN_KERB_WALK_BEND_TOLERANCE_M);
    EXPECT_TRUE(pedestrian->path_m < PEDESTRIAN_KERB_WALK_MAX_M + 2.0f * PEDESTRIAN_KERB_WALK_BEND_TOLERANCE_M);
    int target_slot = target_slot_of(&env, pedestrian);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_bystanders, 1.0f, 0.0f);

    float start_x = agent->sim_x;
    float start_y = agent->sim_y;
    float start_z = agent->sim_z;
    float path_m = pedestrian->path_m;
    EXPECT_NEAR(pedestrian->climb_per_m, 0.0f, 0.0f); // Town01 is flat
    pedestrian->climb_per_m = 0.05f;
    for (int t = 0; t < 380 && pedestrian->state == PEDESTRIAN_STATE_WALKING; t++) {
        EXPECT_EQ_INT(road_edges_to_nearest_lane(&env, agent->sim_x, agent->sim_y) % 2, 1);
        drive_set_neutral_actions(&env);
        c_step(&env);
    }
    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_STANDING);
    EXPECT_NEAR(hypotf(agent->sim_x - start_x, agent->sim_y - start_y), path_m, 1e-2f);
    EXPECT_NEAR(agent->sim_z - start_z, 0.05f * path_m, 1e-2f);
    EXPECT_EQ_INT(road_edges_to_nearest_lane(&env, agent->sim_x, agent->sim_y) % 2, 1);
    free_allocated(&env);
    return 0;
}

static int test_kerb_stepper_pops_in_steps_and_stands_off_the_road(void) {
    srand(7);
    Drive env = make_pedestrian_env(16, 1, 5.0f);
    int visible_k = step_until_appearance(&env, PEDESTRIAN_MODE_KERB_STEP, 60);
    EXPECT_TRUE(visible_k != -1);
    ScriptedPedestrian *pedestrian = &env.pedestrians[visible_k];
    Agent *agent = &env.agents[pedestrian->agent_idx];
    Agent *car = &env.agents[pedestrian->target_agent_idx];

    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_WALKING);
    EXPECT_TRUE(pedestrian->timer_s >= PEDESTRIAN_WAIT_MIN_S && pedestrian->timer_s <= PEDESTRIAN_WAIT_MAX_S);
    EXPECT_TRUE(pedestrian->path_m >= 0.0f && pedestrian->path_m <= PEDESTRIAN_KERB_STEP_MAX_M);
    EXPECT_TRUE(pedestrian->stand_s >= PEDESTRIAN_BYSTANDER_STAND_MIN_S);
    EXPECT_TRUE(pedestrian->stand_s <= PEDESTRIAN_BYSTANDER_STAND_MAX_S);
    float rel_x, rel_y;
    project_point_to_ego_frame(car, agent->sim_x, agent->sim_y, &rel_x, &rel_y);
    float max_appear_m
        = PEDESTRIAN_LB1_APPEAR_BASE_M + 2.0f * PEDESTRIAN_LB1_APPEAR_PER_LANE_M + PEDESTRIAN_APPEAR_JITTER_M;
    EXPECT_TRUE(rel_x > 0.0f && rel_x < max_appear_m + 5.0f);
    EXPECT_TRUE(rel_y < 0.0f);
    int target_slot = target_slot_of(&env, pedestrian);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_bystanders, 1.0f, 0.0f);
    EXPECT_NEAR(env.logs[target_slot].pedestrian_crossings, 0.0f, 0.0f);

    float start_x = agent->sim_x;
    float start_y = agent->sim_y;
    float path_m = pedestrian->path_m;
    for (int t = 0; t < 40 && pedestrian->state == PEDESTRIAN_STATE_WALKING; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
    }
    EXPECT_EQ_INT(pedestrian->state, PEDESTRIAN_STATE_STANDING);
    EXPECT_NEAR(hypotf(agent->sim_x - start_x, agent->sim_y - start_y), path_m, 1e-3f);
    EXPECT_EQ_INT(road_edges_to_nearest_lane(&env, agent->sim_x, agent->sim_y) % 2, 1);
    EXPECT_NEAR(agent->sim_speed, 0.0f, 0.0f);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_ray_hits_segment);
    RUN_TEST(test_pedestrian_slots_are_hidden_static_partners);
    RUN_TEST(test_pedestrian_slots_keep_initial_spawns);
    RUN_TEST(test_pedestrian_mode_shares);
    RUN_TEST(test_pop_in_crossing_geometry_and_lifecycle);
    RUN_TEST(test_pop_in_crosser_stands_then_ramps_to_speed);
    RUN_TEST(test_car_hitting_pedestrian_logs_pedestrian_collision);
    RUN_TEST(test_slow_crosser_walks_from_a_kerb_across_the_road);
    RUN_TEST(test_stander_stays_beside_the_road_with_a_running_stopped_time);
    RUN_TEST(test_kerb_walker_stays_off_the_road);
    RUN_TEST(test_kerb_stepper_pops_in_steps_and_stands_off_the_road);
    return test_summary(failures);
}
