#include "include/drive_fixture.h"
#include "include/test.h"

#define TRAINING_RENDER_TEST_SEED 42
#define TRAINING_RENDER_TEST_AGENT_COUNT 64
#define TRAINING_RENDER_LIGHT_STATE_COUNT 64

static Drive make_training_render_env(int eval_training_render) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, TRAINING_RENDER_TEST_AGENT_COUNT, 0);
    env.eval_mode = 1;
    env.eval_training_render = eval_training_render;
    env.init_seed = TRAINING_RENDER_TEST_SEED;
    rng_seed(&env.seed_stream_rng, env.init_seed);
    allocate(&env);
    c_reset(&env);
    return env;
}

static int test_eval_vehicle_size_distribution_switch(void) {
    Drive benchmark_env = make_training_render_env(0);
    EXPECT_TRUE(benchmark_env.active_agent_count > 0);
    for (int active_idx = 0; active_idx < benchmark_env.active_agent_count; active_idx++) {
        Agent *agent = &benchmark_env.agents[benchmark_env.active_agent_indices[active_idx]];
        EXPECT_TRUE(agent->sim_length >= 2.0f);
        EXPECT_TRUE(agent->sim_length <= 5.5f);
        EXPECT_TRUE(agent->sim_width >= 1.5f);
        EXPECT_TRUE(agent->sim_width <= 2.5f);
    }
    free_allocated(&benchmark_env);

    Drive training_render_env = make_training_render_env(1);
    EXPECT_TRUE(training_render_env.active_agent_count > 0);
    int saw_training_only_size = 0;
    for (int active_idx = 0; active_idx < training_render_env.active_agent_count; active_idx++) {
        Agent *agent = &training_render_env.agents[training_render_env.active_agent_indices[active_idx]];
        EXPECT_TRUE(agent->sim_length >= 0.8f);
        EXPECT_TRUE(agent->sim_length <= 7.0f);
        EXPECT_TRUE(agent->sim_width >= 0.8f);
        EXPECT_TRUE(agent->sim_width <= 2.7f);
        if (agent->sim_length < 2.0f || agent->sim_length > 5.5f || agent->sim_width < 1.5f
            || agent->sim_width > 2.5f) {
            saw_training_only_size = 1;
        }
    }
    EXPECT_TRUE(saw_training_only_size);
    free_allocated(&training_render_env);
    return 0;
}

static void generate_light_sequence(int eval_mode, int eval_training_render, int *states) {
    TrafficControlElement traffic_light = {0};
    traffic_light.type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    traffic_light.states = states;
    traffic_light.state_size = TRAINING_RENDER_LIGHT_STATE_COUNT;

    Drive env = {0};
    env.eval_mode = eval_mode;
    env.eval_training_render = eval_training_render;
    env.scenario_length = TRAINING_RENDER_LIGHT_STATE_COUNT;
    env.dt = 0.1f;
    env.num_traffic_elements = 1;
    env.traffic_elements = &traffic_light;
    rng_seed(&env.rng_state, TRAINING_RENDER_TEST_SEED);
    generate_traffic_light_states(&env);
}

static int test_eval_traffic_lights_use_training_generation(void) {
    int training_states[TRAINING_RENDER_LIGHT_STATE_COUNT] = {0};
    int training_render_states[TRAINING_RENDER_LIGHT_STATE_COUNT] = {0};
    int benchmark_states[TRAINING_RENDER_LIGHT_STATE_COUNT] = {0};

    generate_light_sequence(0, 0, training_states);
    generate_light_sequence(1, 1, training_render_states);
    generate_light_sequence(1, 0, benchmark_states);

    EXPECT_TRUE(memcmp(training_states, training_render_states, sizeof(training_states)) == 0);
    EXPECT_TRUE(memcmp(training_states, benchmark_states, sizeof(training_states)) != 0);
    return 0;
}

static int test_reward_sampling_remains_config_driven(void) {
    Drive env = {0};
    Agent fixed_agent = {0};
    Agent randomized_agent = {0};
    env.eval_mode = 1;
    env.eval_training_render = 1;
    env.reward_collision = 1.5f;

    env.reward_randomization = 0;
    generate_reward_coefs(&env, &fixed_agent);
    EXPECT_NEAR(fixed_agent.reward_coefs[REWARD_COEF_COLLISION], env.reward_collision, 1e-6f);

    rng_seed(&env.rng_state, TRAINING_RENDER_TEST_SEED);
    env.reward_randomization = 1;
    generate_reward_coefs(&env, &randomized_agent);
    EXPECT_TRUE(randomized_agent.reward_coefs[REWARD_COEF_COLLISION] >= REWARD_BOUNDS[REWARD_COEF_COLLISION].min_val);
    EXPECT_TRUE(randomized_agent.reward_coefs[REWARD_COEF_COLLISION] <= REWARD_BOUNDS[REWARD_COEF_COLLISION].max_val);
    EXPECT_TRUE(
        randomized_agent.reward_coefs[REWARD_COEF_COLLISION] != fixed_agent.reward_coefs[REWARD_COEF_COLLISION]);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_eval_vehicle_size_distribution_switch);
    RUN_TEST(test_eval_traffic_lights_use_training_generation);
    RUN_TEST(test_reward_sampling_remains_config_driven);
    return test_summary(failures);
}
