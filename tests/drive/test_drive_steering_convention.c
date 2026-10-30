// One place that pins which way the car actually goes, for every path that can steer one:
// the policy's two dynamics models, the human wheel, and the co-sim state setter. Every
// assertion here is on the sign of the heading CHANGE and of yaw_rate, never on steering_angle,
// because a suite written against steering_angle cannot tell left from right.
#include "include/drive_fixture.h"
#include "include/test.h"

#define CONVENTION_TEST_DT_SECONDS 0.1f
#define CONVENTION_TEST_BASE_SPEED_MPS 20.0f
#define CONVENTION_TEST_SPEED_SCALE 1.5f
// Short enough that even a full-lock turn cannot wrap the heading past the half circle, which
// would invert the sign of sim_y and of raw sim_heading while the car is still turning one way.
#define TURN_DIRECTION_STEP_COUNT 3
#define EXACT_TOLERANCE 0.0f
#define COSIM_YAW_RATE_RPS 0.3f
#define COSIM_STEERING_RAD 0.3f
#define COSIM_SPEED_MPS 10.0f

typedef struct {
    Drive env;
    GridMap offgrid_map;
    Agent agent;
    float action_storage[2];
} ConventionBench;

// The zeroed GridMap reports every query off-grid, which makes update_agent_z a no-op.
static void convention_bench_init(ConventionBench *bench, float speed_mps, int dynamics_model) {
    memset(bench, 0, sizeof(ConventionBench));
    bench->agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    for (int coef = 0; coef < NUM_REWARD_COEFS; coef++) {
        bench->agent.reward_coefs[coef] = 1.0f;
    }
    bench->agent.sim_vx = speed_mps;
    update_agent_speed(&bench->agent);
    copy_pose_to_prev(&bench->agent);
    bench->env.agents = &bench->agent;
    bench->env.num_agents = 1;
    bench->env.grid_map = &bench->offgrid_map;
    bench->env.dt = CONVENTION_TEST_DT_SECONDS;
    bench->env.base_max_speed_mps = CONVENTION_TEST_BASE_SPEED_MPS;
    bench->env.conditioning_speed_scale = CONVENTION_TEST_SPEED_SCALE;
    bench->env.action_type = ACTION_TYPE_CONTINUOUS;
    bench->env.dynamics_model = dynamics_model;
    bench->env.human_grip_mode = GRIP_MODE_KINEMATIC;
    bench->env.is_first_integration = 1;
    bench->env.actions = bench->action_storage;
}

static void set_lateral_action(ConventionBench *bench, float lateral_action) {
    float (*actions)[2] = (float (*)[2]) bench->env.actions;
    actions[0][0] = 0.0f;
    actions[0][1] = lateral_action;
}

// Runs a few steps and asserts the car turns the expected way the whole time. Returns 0 on success.
static int expect_turn_direction(ConventionBench *bench, int use_human_path, int expect_left) {
    for (int step = 0; step < TURN_DIRECTION_STEP_COUNT; step++) {
        float heading_before_rad = bench->agent.sim_heading;
        if (use_human_path) {
            move_human(&bench->env, 0);
        } else {
            move_dynamics(&bench->env, 0, 0);
        }
        float heading_change_rad = normalize_heading(bench->agent.sim_heading - heading_before_rad);
        if (expect_left) {
            EXPECT_TRUE(heading_change_rad > 0.0f);
            EXPECT_TRUE(bench->agent.yaw_rate > 0.0f);
        } else {
            EXPECT_TRUE(heading_change_rad < 0.0f);
            EXPECT_TRUE(bench->agent.yaw_rate < 0.0f);
        }
    }
    // Started at the origin heading +x, so left of the start line is positive y.
    if (expect_left) {
        EXPECT_TRUE(bench->agent.sim_y > 0.0f);
    } else {
        EXPECT_TRUE(bench->agent.sim_y < 0.0f);
    }
    EXPECT_TRUE(bench->agent.sim_x > 0.0f);
    return 0;
}

static int test_classic_dynamics_positive_lateral_action_turns_left(void) {
    static const float speeds_mps[] = {3.0f, 8.0f, 15.0f};
    int speed_count = (int) (sizeof(speeds_mps) / sizeof(speeds_mps[0]));
    for (int s = 0; s < speed_count; s++) {
        {
            ConventionBench bench;
            convention_bench_init(&bench, speeds_mps[s], DYNAMICS_MODEL_CLASSIC);
            set_lateral_action(&bench, 1.0f);
            if (expect_turn_direction(&bench, 0, 1)) {
                return 1;
            }
        }
        {
            ConventionBench bench;
            convention_bench_init(&bench, speeds_mps[s], DYNAMICS_MODEL_CLASSIC);
            set_lateral_action(&bench, -1.0f);
            if (expect_turn_direction(&bench, 0, 0)) {
                return 1;
            }
        }
    }
    return 0;
}

static int test_jerk_dynamics_positive_lateral_action_turns_left(void) {
    static const float speeds_mps[] = {3.0f, 8.0f, 15.0f};
    int speed_count = (int) (sizeof(speeds_mps) / sizeof(speeds_mps[0]));
    for (int s = 0; s < speed_count; s++) {
        {
            ConventionBench bench;
            convention_bench_init(&bench, speeds_mps[s], DYNAMICS_MODEL_JERK);
            set_lateral_action(&bench, 1.0f);
            if (expect_turn_direction(&bench, 0, 1)) {
                return 1;
            }
        }
        {
            ConventionBench bench;
            convention_bench_init(&bench, speeds_mps[s], DYNAMICS_MODEL_JERK);
            set_lateral_action(&bench, -1.0f);
            if (expect_turn_direction(&bench, 0, 0)) {
                return 1;
            }
        }
    }
    return 0;
}

static int test_neutral_lateral_action_drives_straight(void) {
    for (int dynamics_model = DYNAMICS_MODEL_CLASSIC; dynamics_model <= DYNAMICS_MODEL_JERK; dynamics_model++) {
        ConventionBench bench;
        convention_bench_init(&bench, 10.0f, dynamics_model);
        set_lateral_action(&bench, 0.0f);
        for (int step = 0; step < TURN_DIRECTION_STEP_COUNT; step++) {
            move_dynamics(&bench.env, 0, 0);
            EXPECT_NEAR(bench.agent.sim_heading, 0.0f, EXACT_TOLERANCE);
            EXPECT_NEAR(bench.agent.sim_y, 0.0f, EXACT_TOLERANCE);
            EXPECT_NEAR(bench.agent.yaw_rate, 0.0f, EXACT_TOLERANCE);
        }
        EXPECT_TRUE(bench.agent.sim_x > 0.0f);
    }
    return 0;
}

// The physical convention is one thing shared by every path: a positive road-wheel angle, a
// positive yaw rate and a positive lateral acceleration all mean the same left turn.
static int test_every_path_shares_one_physical_convention(void) {
    for (int dynamics_model = DYNAMICS_MODEL_CLASSIC; dynamics_model <= DYNAMICS_MODEL_JERK; dynamics_model++) {
        ConventionBench bench;
        convention_bench_init(&bench, 10.0f, dynamics_model);
        set_lateral_action(&bench, 1.0f);
        move_dynamics(&bench.env, 0, 0);
        EXPECT_TRUE(bench.agent.steering_angle > 0.0f);
        EXPECT_TRUE(bench.agent.yaw_rate > 0.0f);
        EXPECT_TRUE(bench.agent.accel_lat > 0.0f);
    }
    {
        ConventionBench bench;
        convention_bench_init(&bench, 10.0f, DYNAMICS_MODEL_JERK);
        EXPECT_EQ_INT(c_set_human_input(&bench.env, -1.0f, 0.0f), 0);
        move_human(&bench.env, 0);
        EXPECT_TRUE(bench.agent.steering_angle > 0.0f);
        EXPECT_TRUE(bench.agent.yaw_rate > 0.0f);
        EXPECT_TRUE(bench.agent.accel_lat > 0.0f);
    }
    return 0;
}

// Deliberate input remap, not a physics difference: the wheel reads +1 at its RIGHT lock, so the
// human command is negated onto the shared convention while the policy's lateral action is not.
static int test_wheel_command_is_negated_onto_the_road_wheel(void) {
    {
        ConventionBench bench;
        convention_bench_init(&bench, 10.0f, DYNAMICS_MODEL_JERK);
        EXPECT_EQ_INT(c_set_human_input(&bench.env, 1.0f, 0.0f), 0);
        if (expect_turn_direction(&bench, 1, 0)) {
            return 1;
        }
    }
    {
        ConventionBench bench;
        convention_bench_init(&bench, 10.0f, DYNAMICS_MODEL_JERK);
        EXPECT_EQ_INT(c_set_human_input(&bench.env, -1.0f, 0.0f), 0);
        if (expect_turn_direction(&bench, 1, 1)) {
            return 1;
        }
    }
    {
        ConventionBench bench;
        convention_bench_init(&bench, 10.0f, DYNAMICS_MODEL_JERK);
        set_lateral_action(&bench, 1.0f);
        if (expect_turn_direction(&bench, 0, 1)) {
            return 1;
        }
    }
    return 0;
}

static Drive convention_full_env(unsigned int seed) {
    srand(seed);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 2, 0);
    env.max_speed_mps = CONVENTION_TEST_BASE_SPEED_MPS;
    env.conditioning_accel_scale = 1.0f;
    env.conditioning_speed_scale = CONVENTION_TEST_SPEED_SCALE;
    env.dt = CONVENTION_TEST_DT_SECONDS;
    env.action_type = ACTION_TYPE_CONTINUOUS;
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    allocate(&env);
    c_reset(&env);
    return env;
}

// An external simulator's cars must steer by the same rule as the policy's, or co-sim traffic
// turns the opposite way from everything around it.
static int test_cosim_state_setter_matches_the_convention(void) {
    Drive env = convention_full_env(401);
    int agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[agent_idx];

    int indices[1] = {agent_idx};
    float world_x_m[1] = {agent->sim_x + env.world_mean_x};
    float world_y_m[1] = {agent->sim_y + env.world_mean_y};
    float world_z_m[1] = {0.0f};
    float heading_rad[1] = {0.0f};
    float velocity_x_mps[1] = {COSIM_SPEED_MPS};
    float velocity_y_mps[1] = {0.0f};
    float yaw_rate_rps[1] = {0.0f};
    float accel_long_mps2[1] = {0.0f};

    for (int sign = -1; sign <= 1; sign += 2) {
        yaw_rate_rps[0] = COSIM_YAW_RATE_RPS * (float) sign;
        EXPECT_EQ_INT(
            c_set_agent_states(
                &env,
                1,
                indices,
                world_x_m,
                world_y_m,
                world_z_m,
                heading_rad,
                velocity_x_mps,
                velocity_y_mps,
                yaw_rate_rps,
                accel_long_mps2,
                NULL,
                NULL),
            0);
        EXPECT_NEAR(agent->yaw_rate, yaw_rate_rps[0], EXACT_TOLERANCE);
        EXPECT_TRUE(agent->steering_angle * (float) sign > 0.0f);
        EXPECT_TRUE(agent->accel_lat * (float) sign > 0.0f);
    }

    for (int sign = -1; sign <= 1; sign += 2) {
        float steering_rad[1] = {COSIM_STEERING_RAD * (float) sign};
        yaw_rate_rps[0] = 0.0f;
        EXPECT_EQ_INT(
            c_set_agent_states(
                &env,
                1,
                indices,
                world_x_m,
                world_y_m,
                world_z_m,
                heading_rad,
                velocity_x_mps,
                velocity_y_mps,
                yaw_rate_rps,
                accel_long_mps2,
                NULL,
                steering_rad),
            0);
        EXPECT_NEAR(agent->steering_angle, steering_rad[0], EXACT_TOLERANCE);
        EXPECT_TRUE(agent->accel_lat * (float) sign > 0.0f);
    }
    free_allocated(&env);
    return 0;
}

// The state the co-sim setter leaves behind must keep turning the same way once the sim steps it.
static int test_cosim_injected_turn_continues_the_same_way(void) {
    for (int sign = -1; sign <= 1; sign += 2) {
        Drive env = convention_full_env(409);
        int agent_idx = env.active_agent_indices[0];
        Agent *agent = &env.agents[agent_idx];
        int log_idx = 0;

        int indices[1] = {agent_idx};
        float world_x_m[1] = {agent->sim_x + env.world_mean_x};
        float world_y_m[1] = {agent->sim_y + env.world_mean_y};
        float world_z_m[1] = {0.0f};
        float heading_rad[1] = {0.0f};
        float velocity_x_mps[1] = {COSIM_SPEED_MPS};
        float velocity_y_mps[1] = {0.0f};
        float yaw_rate_rps[1] = {COSIM_YAW_RATE_RPS * (float) sign};
        float accel_long_mps2[1] = {0.0f};
        EXPECT_EQ_INT(
            c_set_agent_states(
                &env,
                1,
                indices,
                world_x_m,
                world_y_m,
                world_z_m,
                heading_rad,
                velocity_x_mps,
                velocity_y_mps,
                yaw_rate_rps,
                accel_long_mps2,
                NULL,
                NULL),
            0);

        float (*actions)[2] = (float (*)[2]) env.actions;
        actions[log_idx][0] = 0.0f;
        actions[log_idx][1] = 0.0f;
        for (int step = 0; step < TURN_DIRECTION_STEP_COUNT; step++) {
            float heading_before_rad = agent->sim_heading;
            move_dynamics(&env, log_idx, agent_idx);
            float heading_change_rad = normalize_heading(agent->sim_heading - heading_before_rad);
            EXPECT_TRUE(heading_change_rad * (float) sign > 0.0f);
            EXPECT_TRUE(agent->yaw_rate * (float) sign > 0.0f);
        }
        free_allocated(&env);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_classic_dynamics_positive_lateral_action_turns_left);
    RUN_TEST(test_jerk_dynamics_positive_lateral_action_turns_left);
    RUN_TEST(test_neutral_lateral_action_drives_straight);
    RUN_TEST(test_every_path_shares_one_physical_convention);
    RUN_TEST(test_wheel_command_is_negated_onto_the_road_wheel);
    RUN_TEST(test_cosim_state_setter_matches_the_convention);
    RUN_TEST(test_cosim_injected_turn_continues_the_same_way);
    return test_summary(failures);
}
