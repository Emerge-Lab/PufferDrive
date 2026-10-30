#include "include/drive_fixture.h"
#include "include/test.h"

#define SUBSTEP_TEST_POLICY_DT_SECONDS 0.3f
#define SUBSTEP_TEST_BASE_SPEED_MPS 20.0f
#define SUBSTEP_TEST_SPEED_SCALE 1.5f
#define SUBSTEP_TEST_AGENT_COUNT 4
#define DEMO_SUBSTEP_COUNT 18
#define EXACT_TOLERANCE 0.0f
// One policy step of sub-stepping agrees on speed to well under a millimetre per second; the
// path does not, and diverges without bound over a long horizon (see the long-horizon test).
#define SUBSTEP_SPEED_TOLERANCE_MPS 1e-5f
#define SUBSTEP_FIRST_STEP_POSITION_TOLERANCE_M 0.1f
#define POSE_NOISE_DRAW_SEARCH_LIMIT 2000
#define PARTNER_TRAVEL_MIN_M 1.0f
#define SWEPT_COLLISION_GAP_M 6.0f
#define SWEPT_COLLISION_SPEED_MPS 14.0f
#define PHANTOM_POLICY_STEP_COUNT 60
#define PHANTOM_DURATION_STEPS 3

static Drive substep_env(int num_agents, unsigned int seed) {
    srand(seed);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.max_speed_mps = SUBSTEP_TEST_BASE_SPEED_MPS;
    env.conditioning_accel_scale = 1.0f;
    env.conditioning_speed_scale = SUBSTEP_TEST_SPEED_SCALE;
    env.reset_accel_on_stop = 1;
    env.dt = SUBSTEP_TEST_POLICY_DT_SECONDS;
    env.action_type = ACTION_TYPE_CONTINUOUS;
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    allocate(&env);
    c_reset(&env);
    return env;
}

static void run_policy_step(Drive *env) {
    for (int substep = 0; substep < env->substeps_per_policy_step; substep++) {
        c_substep(env);
    }
    c_step(env);
}

static void set_continuous_actions(Drive *env, float accel_action, float steer_action) {
    float (*actions)[2] = (float (*)[2]) env->actions;
    for (int i = 0; i < env->active_agent_count; i++) {
        actions[i][0] = accel_action;
        actions[i][1] = steer_action;
    }
}

static int trajectories_match(const Drive *left, const Drive *right) {
    if (left->active_agent_count != right->active_agent_count) {
        return 0;
    }
    for (int i = 0; i < left->active_agent_count; i++) {
        const Agent *a = &left->agents[left->active_agent_indices[i]];
        const Agent *b = &right->agents[right->active_agent_indices[i]];
        if (a->sim_x != b->sim_x || a->sim_y != b->sim_y || a->sim_heading != b->sim_heading) {
            return 0;
        }
        if (a->sim_vx != b->sim_vx || a->sim_vy != b->sim_vy || a->yaw_rate != b->yaw_rate) {
            return 0;
        }
        if (a->accel_long != b->accel_long || a->accel_lat != b->accel_lat) {
            return 0;
        }
        if (a->steering_angle != b->steering_angle || a->phantom_braking_counter != b->phantom_braking_counter) {
            return 0;
        }
    }
    return 1;
}

static unsigned long hash_all_agents(const Drive *env) {
    unsigned long hash = 1469598103934665603UL;
    const unsigned char *bytes = (const unsigned char *) env->agents;
    size_t byte_count = (size_t) env->num_total_agents * sizeof(Agent);
    for (size_t i = 0; i < byte_count; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211UL;
    }
    return hash;
}

static int rng_draws_between(Rng before, Rng after, int search_limit) {
    for (int draws = 0; draws <= search_limit; draws++) {
        if (before.s[0] == after.s[0] && before.s[1] == after.s[1] && before.s[2] == after.s[2]
            && before.s[3] == after.s[3]) {
            return draws;
        }
        rng_next(&before);
    }
    return -1;
}

static float agent_travel_m(const Agent *agent, float start_x_m, float start_y_m) {
    float dx = agent->sim_x - start_x_m;
    float dy = agent->sim_y - start_y_m;
    return sqrtf(dx * dx + dy * dy);
}

static void record_phantom_counters(Drive *env, int *counters, int policy_step_count) {
    for (int step = 0; step < policy_step_count; step++) {
        set_continuous_actions(env, 0.5f, 0.0f);
        run_policy_step(env);
        counters[step] = env->agents[env->active_agent_indices[0]].phantom_braking_counter;
    }
}

static int count_phantom_braking_windows(const int *counters, int policy_step_count) {
    int windows = 0;
    int previous_counter = 0;
    for (int step = 0; step < policy_step_count; step++) {
        windows += (counters[step] > 0 && previous_counter == 0);
        previous_counter = counters[step];
    }
    return windows;
}

// One decrement per policy step: a counter above 1 must step down by exactly 1, never hold.
static int phantom_counters_decay_once_per_step(const int *counters, int policy_step_count, int duration) {
    int previous_counter = 0;
    for (int step = 0; step < policy_step_count; step++) {
        int counter = counters[step];
        if (counter < 0 || counter > duration) {
            return 0;
        }
        if (previous_counter >= 2 && counter != previous_counter - 1) {
            return 0;
        }
        if (previous_counter <= 1 && counter != 0 && counter != duration) {
            return 0;
        }
        previous_counter = counter;
    }
    return 1;
}

static void arm_phantom_braking(Drive *env) {
    env->phantom_braking_prob = 1.0f;
    env->phantom_braking_trigger_prob = 0.5f;
    env->phantom_braking_duration = PHANTOM_DURATION_STEPS;
    for (int i = 0; i < env->active_agent_count; i++) {
        env->agents[env->active_agent_indices[i]].is_phantom_braker = 1;
    }
}

static int test_substepping_defaults_to_off(void) {
    Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 201);
    EXPECT_EQ_INT(env.substep_mode, SUBSTEP_MODE_ALL);
    EXPECT_EQ_INT(env.substeps_per_policy_step, 1);
    EXPECT_EQ_INT(env.substeps_taken, 0);
    free_allocated(&env);
    return 0;
}

static int test_single_substep_matches_the_training_path(void) {
    Drive substepped = substep_env(SUBSTEP_TEST_AGENT_COUNT, 203);
    Drive untouched = substep_env(SUBSTEP_TEST_AGENT_COUNT, 203);
    EXPECT_EQ_INT(c_set_substepping(&substepped, SUBSTEP_MODE_ALL, 1), 0);
    for (int step = 0; step < 30; step++) {
        set_continuous_actions(&substepped, 0.5f, 0.1f);
        set_continuous_actions(&untouched, 0.5f, 0.1f);
        run_policy_step(&substepped);
        c_step(&untouched);
        EXPECT_EQ_INT(trajectories_match(&substepped, &untouched), 1);
    }
    free_allocated(&substepped);
    free_allocated(&untouched);
    return 0;
}

static int test_substepping_is_deterministic(void) {
    Drive first_run = substep_env(SUBSTEP_TEST_AGENT_COUNT, 207);
    Drive second_run = substep_env(SUBSTEP_TEST_AGENT_COUNT, 207);
    EXPECT_EQ_INT(c_set_substepping(&first_run, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
    EXPECT_EQ_INT(c_set_substepping(&second_run, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
    for (int step = 0; step < 30; step++) {
        float accel_action = 0.4f + 0.01f * (float) step;
        set_continuous_actions(&first_run, accel_action, 0.2f);
        run_policy_step(&first_run);
    }
    for (int step = 0; step < 30; step++) {
        float accel_action = 0.4f + 0.01f * (float) step;
        set_continuous_actions(&second_run, accel_action, 0.2f);
        run_policy_step(&second_run);
    }
    EXPECT_EQ_INT(trajectories_match(&first_run, &second_run), 1);
    free_allocated(&first_run);
    free_allocated(&second_run);
    return 0;
}

// The configured count is the only thing that decides how many sub-steps a policy step takes.
static int test_substep_count_is_fixed_not_timed(void) {
    Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 209);
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
    for (int step = 0; step < 20; step++) {
        set_continuous_actions(&env, 0.5f, 0.0f);
        for (int substep = 0; substep < DEMO_SUBSTEP_COUNT; substep++) {
            EXPECT_EQ_INT(env.substeps_taken, substep);
            c_substep(&env);
        }
        EXPECT_EQ_INT(env.substeps_taken, DEMO_SUBSTEP_COUNT);
        EXPECT_EQ_INT(env.substeps_per_policy_step, DEMO_SUBSTEP_COUNT);
        c_step(&env);
        EXPECT_EQ_INT(env.substeps_taken, 0);
    }
    EXPECT_NEAR(env.dt, SUBSTEP_TEST_POLICY_DT_SECONDS, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

// Speed is the linear part of the chain, so one policy step of sub-stepping tracks it closely.
static int test_substep_speed_tracks_the_single_step(void) {
    for (int action_index = 0; action_index <= 10; action_index++) {
        float accel_action = -1.0f + 0.2f * (float) action_index;
        Drive single = substep_env(1, 211);
        Drive substepped = substep_env(1, 211);
        EXPECT_EQ_INT(c_set_substepping(&single, SUBSTEP_MODE_ALL, 1), 0);
        EXPECT_EQ_INT(c_set_substepping(&substepped, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
        set_continuous_actions(&single, accel_action, 0.4f);
        set_continuous_actions(&substepped, accel_action, 0.4f);
        run_policy_step(&single);
        run_policy_step(&substepped);

        const Agent *a = &single.agents[single.active_agent_indices[0]];
        const Agent *b = &substepped.agents[substepped.active_agent_indices[0]];
        EXPECT_NEAR(b->sim_speed_signed, a->sim_speed_signed, SUBSTEP_SPEED_TOLERANCE_MPS);
        EXPECT_NEAR(agent_travel_m(b, a->sim_x, a->sim_y), 0.0f, SUBSTEP_FIRST_STEP_POSITION_TOLERANCE_M);
        free_allocated(&single);
        free_allocated(&substepped);
    }
    return 0;
}

// The path diverges without bound over a long horizon; only sanity is asserted, not a distance.
static int test_long_horizon_substep_divergence_stays_sane(void) {
    Drive single = substep_env(1, 213);
    Drive substepped = substep_env(1, 213);
    EXPECT_EQ_INT(c_set_substepping(&single, SUBSTEP_MODE_ALL, 1), 0);
    EXPECT_EQ_INT(c_set_substepping(&substepped, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
    for (int step = 0; step < 100; step++) {
        set_continuous_actions(&single, 0.6f, 0.25f);
        set_continuous_actions(&substepped, 0.6f, 0.25f);
        run_policy_step(&single);
        run_policy_step(&substepped);
    }
    const Agent *a = &single.agents[single.active_agent_indices[0]];
    const Agent *b = &substepped.agents[substepped.active_agent_indices[0]];
    float speed_cap_mps = substepped.base_max_speed_mps * b->reward_coefs[REWARD_COEF_SPEED];
    EXPECT_FINITE(a->sim_x);
    EXPECT_FINITE(b->sim_x);
    EXPECT_FINITE(b->sim_heading);
    EXPECT_TRUE(b->sim_speed_signed <= speed_cap_mps);
    EXPECT_TRUE(b->sim_speed_signed >= MAX_BACKWARD_SPEED);
    free_allocated(&single);
    free_allocated(&substepped);
    return 0;
}

// Neither the sub-step count nor the sub-step mode may change when the per-policy-step draw
// happens: all four configurations must produce one identical counter sequence.
static int test_phantom_braking_fires_once_per_policy_step_in_both_modes(void) {
    static const int substep_counts[] = {1, DEMO_SUBSTEP_COUNT};
    int reference_counters[PHANTOM_POLICY_STEP_COUNT];
    int reference_windows = -1;

    for (int substep_mode = SUBSTEP_MODE_ALL; substep_mode <= SUBSTEP_MODE_HUMAN_ONLY; substep_mode++) {
        for (int count_index = 0; count_index < 2; count_index++) {
            Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 217);
            arm_phantom_braking(&env);
            EXPECT_EQ_INT(c_set_substepping(&env, substep_mode, substep_counts[count_index]), 0);

            int counters[PHANTOM_POLICY_STEP_COUNT];
            record_phantom_counters(&env, counters, PHANTOM_POLICY_STEP_COUNT);
            EXPECT_EQ_INT(
                phantom_counters_decay_once_per_step(counters, PHANTOM_POLICY_STEP_COUNT, PHANTOM_DURATION_STEPS),
                1);

            int windows = count_phantom_braking_windows(counters, PHANTOM_POLICY_STEP_COUNT);
            if (reference_windows < 0) {
                reference_windows = windows;
                memcpy(reference_counters, counters, sizeof(counters));
                EXPECT_TRUE(reference_windows > 0);
            }
            EXPECT_EQ_INT(windows, reference_windows);
            for (int step = 0; step < PHANTOM_POLICY_STEP_COUNT; step++) {
                EXPECT_EQ_INT(counters[step], reference_counters[step]);
            }
            free_allocated(&env);
        }
    }
    return 0;
}

static int test_swept_collision_spans_the_whole_policy_step(void) {
    for (int substeps = 1; substeps <= DEMO_SUBSTEP_COUNT; substeps += DEMO_SUBSTEP_COUNT - 1) {
        Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 223);
        EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, substeps), 0);
        int mover_idx = env.active_agent_indices[0];
        int target_idx = env.active_agent_indices[1];
        Agent *mover = &env.agents[mover_idx];
        Agent *target = &env.agents[target_idx];

        target->sim_x = mover->sim_x + SWEPT_COLLISION_GAP_M * mover->cos_heading;
        target->sim_y = mover->sim_y + SWEPT_COLLISION_GAP_M * mover->sin_heading;
        target->sim_heading = mover->sim_heading;
        target->cos_heading = mover->cos_heading;
        target->sin_heading = mover->sin_heading;
        copy_pose_to_prev(target);
        mover->sim_vx = SWEPT_COLLISION_SPEED_MPS * mover->cos_heading;
        mover->sim_vy = SWEPT_COLLISION_SPEED_MPS * mover->sin_heading;
        update_agent_speed(mover);

        set_continuous_actions(&env, 1.0f, 0.0f);
        run_policy_step(&env);

        EXPECT_NEAR(mover->metrics_array[COLLISION_IDX], 1.0f, EXACT_TOLERANCE);
        float swept_distance_m = agent_travel_m(mover, mover->prev_x, mover->prev_y);
        EXPECT_TRUE(swept_distance_m > SWEPT_COLLISION_SPEED_MPS * SUBSTEP_TEST_POLICY_DT_SECONDS * 0.5f);
        free_allocated(&env);
    }
    return 0;
}

static int test_pose_noise_fires_once_per_policy_step(void) {
    int single_draws = -1;
    for (int substeps = 1; substeps <= DEMO_SUBSTEP_COUNT; substeps += DEMO_SUBSTEP_COUNT - 1) {
        Drive env = substep_env(1, 227);
        env.pose_noise_xy_m = 0.5f;
        env.pose_noise_yaw_rad = 0.1f;
        env.phantom_braking_trigger_prob = 0.0f;
        EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, substeps), 0);

        Rng before = env.rng_state;
        set_continuous_actions(&env, 0.5f, 0.0f);
        run_policy_step(&env);
        int draws = rng_draws_between(before, env.rng_state, POSE_NOISE_DRAW_SEARCH_LIMIT);
        EXPECT_TRUE(draws > 0);
        if (single_draws < 0) {
            single_draws = draws;
        }
        EXPECT_EQ_INT(draws, single_draws);
        free_allocated(&env);
    }
    return 0;
}

static int test_both_modes_still_move_partners(void) {
    for (int substep_mode = SUBSTEP_MODE_ALL; substep_mode <= SUBSTEP_MODE_HUMAN_ONLY; substep_mode++) {
        Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 229);
        int human_idx = env.active_agent_indices[0];
        int partner_idx = env.active_agent_indices[1];
        EXPECT_EQ_INT(c_select_human_agent(&env, human_idx), 0);
        EXPECT_EQ_INT(c_set_human_input(&env, 0.0f, 2.0f), 0);
        EXPECT_EQ_INT(c_set_substepping(&env, substep_mode, DEMO_SUBSTEP_COUNT), 0);

        float human_start_x_m = env.agents[human_idx].sim_x;
        float human_start_y_m = env.agents[human_idx].sim_y;
        float partner_start_x_m = env.agents[partner_idx].sim_x;
        float partner_start_y_m = env.agents[partner_idx].sim_y;
        for (int step = 0; step < 10; step++) {
            set_continuous_actions(&env, 1.0f, 0.0f);
            run_policy_step(&env);
        }
        EXPECT_TRUE(agent_travel_m(&env.agents[human_idx], human_start_x_m, human_start_y_m) > PARTNER_TRAVEL_MIN_M);
        EXPECT_TRUE(
            agent_travel_m(&env.agents[partner_idx], partner_start_x_m, partner_start_y_m) > PARTNER_TRAVEL_MIN_M);
        free_allocated(&env);
    }
    return 0;
}

static int test_human_only_mode_substeps_the_human_alone(void) {
    Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 233);
    int human_idx = env.active_agent_indices[0];
    int partner_idx = env.active_agent_indices[1];
    EXPECT_EQ_INT(c_select_human_agent(&env, human_idx), 0);
    EXPECT_EQ_INT(c_set_human_input(&env, 0.0f, 2.0f), 0);
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_HUMAN_ONLY, DEMO_SUBSTEP_COUNT), 0);

    float partner_start_x_m = env.agents[partner_idx].sim_x;
    float partner_start_y_m = env.agents[partner_idx].sim_y;
    float human_start_x_m = env.agents[human_idx].sim_x;
    float human_start_y_m = env.agents[human_idx].sim_y;

    set_continuous_actions(&env, 1.0f, 0.0f);
    for (int substep = 0; substep < DEMO_SUBSTEP_COUNT; substep++) {
        c_substep(&env);
    }
    EXPECT_TRUE(agent_travel_m(&env.agents[human_idx], human_start_x_m, human_start_y_m) > 0.0f);
    EXPECT_NEAR(env.agents[partner_idx].sim_x, partner_start_x_m, EXACT_TOLERANCE);
    EXPECT_NEAR(env.agents[partner_idx].sim_y, partner_start_y_m, EXACT_TOLERANCE);

    c_step(&env);
    EXPECT_TRUE(agent_travel_m(&env.agents[partner_idx], partner_start_x_m, partner_start_y_m) > 0.0f);
    free_allocated(&env);
    return 0;
}

static int test_render_poses_never_mutate_sim_state(void) {
    for (int substep_mode = SUBSTEP_MODE_ALL; substep_mode <= SUBSTEP_MODE_HUMAN_ONLY; substep_mode++) {
        Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 239);
        int human_idx = env.active_agent_indices[0];
        EXPECT_EQ_INT(c_select_human_agent(&env, human_idx), 0);
        EXPECT_EQ_INT(c_set_human_input(&env, 0.0f, 2.0f), 0);
        EXPECT_EQ_INT(c_set_substepping(&env, substep_mode, DEMO_SUBSTEP_COUNT), 0);
        for (int step = 0; step < 10; step++) {
            set_continuous_actions(&env, 1.0f, 0.0f);
            run_policy_step(&env);
        }

        float render_x_m[64], render_y_m[64], render_heading_rad[64];
        unsigned long hash_before = hash_all_agents(&env);
        for (int tenth = 0; tenth <= 10; tenth++) {
            c_get_render_poses(&env, 0.03f * (float) tenth, render_x_m, render_y_m, render_heading_rad);
            EXPECT_TRUE(drive_all_finite(render_x_m, env.num_agents));
            EXPECT_TRUE(drive_all_finite(render_y_m, env.num_agents));
            EXPECT_TRUE(drive_all_finite(render_heading_rad, env.num_agents));
        }
        EXPECT_TRUE(hash_all_agents(&env) == hash_before);
        free_allocated(&env);
    }
    return 0;
}

static int test_render_poses_are_identity_under_substep_all(void) {
    Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 241);
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
    for (int step = 0; step < 10; step++) {
        set_continuous_actions(&env, 1.0f, 0.2f);
        run_policy_step(&env);
    }
    float render_x_m[64], render_y_m[64], render_heading_rad[64];
    for (int tenth = 0; tenth <= 10; tenth++) {
        c_get_render_poses(&env, 0.03f * (float) tenth, render_x_m, render_y_m, render_heading_rad);
        for (int j = 0; j < env.num_agents; j++) {
            int agent_idx = (j < env.active_agent_count) ? env.active_agent_indices[j]
                                                         : env.static_agent_indices[j - env.active_agent_count];
            const Agent *agent = &env.agents[agent_idx];
            EXPECT_NEAR(render_x_m[j], agent->sim_x + env.world_mean_x, EXACT_TOLERANCE);
            EXPECT_NEAR(render_y_m[j], agent->sim_y + env.world_mean_y, EXACT_TOLERANCE);
            EXPECT_NEAR(render_heading_rad[j], agent->sim_heading, EXACT_TOLERANCE);
        }
    }
    free_allocated(&env);
    return 0;
}

static int test_render_poses_dead_reckon_partners_in_human_only(void) {
    Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 243);
    int human_idx = env.active_agent_indices[0];
    EXPECT_EQ_INT(c_select_human_agent(&env, human_idx), 0);
    EXPECT_EQ_INT(c_set_human_input(&env, 0.0f, 2.0f), 0);
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_HUMAN_ONLY, DEMO_SUBSTEP_COUNT), 0);
    for (int step = 0; step < 20; step++) {
        set_continuous_actions(&env, 1.0f, 0.0f);
        run_policy_step(&env);
    }

    float render_x_m[64], render_y_m[64], render_heading_rad[64];
    c_get_render_poses(&env, 0.0f, render_x_m, render_y_m, render_heading_rad);
    for (int j = 0; j < env.num_agents; j++) {
        int agent_idx = (j < env.active_agent_count) ? env.active_agent_indices[j]
                                                     : env.static_agent_indices[j - env.active_agent_count];
        const Agent *agent = &env.agents[agent_idx];
        EXPECT_NEAR(render_x_m[j], agent->sim_x + env.world_mean_x, 1e-3f);
        EXPECT_NEAR(render_y_m[j], agent->sim_y + env.world_mean_y, 1e-3f);
    }

    c_get_render_poses(&env, SUBSTEP_TEST_POLICY_DT_SECONDS, render_x_m, render_y_m, render_heading_rad);
    int moving_partner_count = 0;
    for (int j = 0; j < env.num_agents; j++) {
        int agent_idx = (j < env.active_agent_count) ? env.active_agent_indices[j]
                                                     : env.static_agent_indices[j - env.active_agent_count];
        const Agent *agent = &env.agents[agent_idx];
        float predicted_travel_m
            = agent_travel_m(agent, render_x_m[j] - env.world_mean_x, render_y_m[j] - env.world_mean_y);
        if (agent_idx == human_idx) {
            EXPECT_NEAR(render_x_m[j], agent->sim_x + env.world_mean_x, EXACT_TOLERANCE);
            EXPECT_NEAR(render_y_m[j], agent->sim_y + env.world_mean_y, EXACT_TOLERANCE);
            EXPECT_NEAR(render_heading_rad[j], agent->sim_heading, EXACT_TOLERANCE);
            continue;
        }
        if (fabsf(agent->sim_speed_signed) > 1.0f) {
            EXPECT_TRUE(predicted_travel_m > 0.0f);
            moving_partner_count++;
        }
    }
    EXPECT_TRUE(moving_partner_count > 0);
    free_allocated(&env);
    return 0;
}

static int test_substepping_validates_its_arguments(void) {
    Drive env = substep_env(1, 251);
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_HUMAN_ONLY, DEMO_SUBSTEP_COUNT), 0);
    EXPECT_EQ_INT(env.substep_mode, SUBSTEP_MODE_HUMAN_ONLY);
    EXPECT_EQ_INT(env.substeps_per_policy_step, DEMO_SUBSTEP_COUNT);

    EXPECT_EQ_INT(c_set_substepping(&env, -1, DEMO_SUBSTEP_COUNT), -1);
    EXPECT_EQ_INT(c_set_substepping(&env, 2, DEMO_SUBSTEP_COUNT), -1);
    EXPECT_EQ_INT(c_set_substepping(&env, 99, DEMO_SUBSTEP_COUNT), -1);
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, 0), -1);
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, -5), -1);
    EXPECT_EQ_INT(env.substep_mode, SUBSTEP_MODE_HUMAN_ONLY);
    EXPECT_EQ_INT(env.substeps_per_policy_step, DEMO_SUBSTEP_COUNT);

    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, 1), 0);
    EXPECT_EQ_INT(env.substeps_taken, 0);
    free_allocated(&env);
    return 0;
}

// Reconfiguring mid-policy-step must restart the count, or the new setting silently no-ops.
static int test_reconfiguring_substepping_restarts_the_count(void) {
    Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 253);
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
    set_continuous_actions(&env, 1.0f, 0.0f);
    for (int substep = 0; substep < DEMO_SUBSTEP_COUNT; substep++) {
        c_substep(&env);
    }
    EXPECT_EQ_INT(env.substeps_taken, DEMO_SUBSTEP_COUNT);

    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, 4), 0);
    EXPECT_EQ_INT(env.substeps_taken, 0);

    int agent_idx = env.active_agent_indices[0];
    float resumed_x_m = env.agents[agent_idx].sim_x;
    float resumed_y_m = env.agents[agent_idx].sim_y;
    c_substep(&env);
    EXPECT_EQ_INT(env.substeps_taken, 1);
    EXPECT_TRUE(agent_travel_m(&env.agents[agent_idx], resumed_x_m, resumed_y_m) > 0.0f);
    free_allocated(&env);
    return 0;
}

static int test_extra_substeps_are_a_no_op(void) {
    Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 257);
    const int configured_substeps = 3;
    EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, configured_substeps), 0);
    set_continuous_actions(&env, 1.0f, 0.1f);
    for (int substep = 0; substep < configured_substeps; substep++) {
        c_substep(&env);
    }
    int agent_idx = env.active_agent_indices[0];
    float settled_x_m = env.agents[agent_idx].sim_x;
    float settled_y_m = env.agents[agent_idx].sim_y;

    for (int extra = 0; extra < 5; extra++) {
        c_substep(&env);
        EXPECT_EQ_INT(env.substeps_taken, configured_substeps);
        EXPECT_NEAR(env.agents[agent_idx].sim_x, settled_x_m, EXACT_TOLERANCE);
        EXPECT_NEAR(env.agents[agent_idx].sim_y, settled_y_m, EXACT_TOLERANCE);
    }
    c_step(&env);
    EXPECT_EQ_INT(env.substeps_taken, 0);
    free_allocated(&env);
    return 0;
}

// A counter surviving a reset would make the next policy step's sub-steps no-ops, freezing the
// whole sim for one step; the player would see a dead stop at every episode boundary.
static int test_substep_counter_is_cleared_by_every_reset_path(void) {
    {
        Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 263);
        EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
        set_continuous_actions(&env, 1.0f, 0.0f);
        for (int substep = 0; substep < DEMO_SUBSTEP_COUNT; substep++) {
            c_substep(&env);
        }
        EXPECT_EQ_INT(env.substeps_taken, DEMO_SUBSTEP_COUNT);

        env.timestep = 5;
        c_reset(&env);
        EXPECT_EQ_INT(env.substeps_taken, 0);

        int agent_idx = env.active_agent_indices[0];
        float resumed_x_m = env.agents[agent_idx].sim_x;
        float resumed_y_m = env.agents[agent_idx].sim_y;
        set_continuous_actions(&env, 1.0f, 0.0f);
        run_policy_step(&env);
        EXPECT_TRUE(agent_travel_m(&env.agents[agent_idx], resumed_x_m, resumed_y_m) > 0.0f);
        free_allocated(&env);
    }
    {
        Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 269);
        EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
        set_continuous_actions(&env, 1.0f, 0.0f);
        for (int substep = 0; substep < DEMO_SUBSTEP_COUNT; substep++) {
            c_substep(&env);
        }
        env.autoreset_pending = 1;
        env.timestep = 5;
        c_step(&env);
        EXPECT_EQ_INT(env.substeps_taken, 0);

        int agent_idx = env.active_agent_indices[0];
        float resumed_x_m = env.agents[agent_idx].sim_x;
        float resumed_y_m = env.agents[agent_idx].sim_y;
        set_continuous_actions(&env, 1.0f, 0.0f);
        run_policy_step(&env);
        EXPECT_TRUE(agent_travel_m(&env.agents[agent_idx], resumed_x_m, resumed_y_m) > 0.0f);
        free_allocated(&env);
    }
    {
        Drive env = substep_env(SUBSTEP_TEST_AGENT_COUNT, 271);
        EXPECT_EQ_INT(c_set_substepping(&env, SUBSTEP_MODE_ALL, DEMO_SUBSTEP_COUNT), 0);
        env.eval_mode = 1;
        env.eval_episode_done = 1;
        set_continuous_actions(&env, 1.0f, 0.0f);
        for (int substep = 0; substep < DEMO_SUBSTEP_COUNT; substep++) {
            c_substep(&env);
        }
        c_step(&env);
        EXPECT_EQ_INT(env.substeps_taken, 0);
        free_allocated(&env);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_substepping_defaults_to_off);
    RUN_TEST(test_single_substep_matches_the_training_path);
    RUN_TEST(test_substepping_is_deterministic);
    RUN_TEST(test_substep_count_is_fixed_not_timed);
    RUN_TEST(test_substep_speed_tracks_the_single_step);
    RUN_TEST(test_long_horizon_substep_divergence_stays_sane);
    RUN_TEST(test_phantom_braking_fires_once_per_policy_step_in_both_modes);
    RUN_TEST(test_swept_collision_spans_the_whole_policy_step);
    RUN_TEST(test_pose_noise_fires_once_per_policy_step);
    RUN_TEST(test_both_modes_still_move_partners);
    RUN_TEST(test_human_only_mode_substeps_the_human_alone);
    RUN_TEST(test_render_poses_never_mutate_sim_state);
    RUN_TEST(test_render_poses_are_identity_under_substep_all);
    RUN_TEST(test_render_poses_dead_reckon_partners_in_human_only);
    RUN_TEST(test_substepping_validates_its_arguments);
    RUN_TEST(test_reconfiguring_substepping_restarts_the_count);
    RUN_TEST(test_extra_substeps_are_a_no_op);
    RUN_TEST(test_substep_counter_is_cleared_by_every_reset_path);
    return test_summary(failures);
}
