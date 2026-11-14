#include "include/drive_fixture.h"
#include "include/test.h"

// Three lights of one junction plus one standalone light; eval cycle 10/3/2 s at dt 0.1.
#define LIGHT_COUNT 4
#define STATE_COUNT 900
#define CYCLE_STEPS 150

static int light_states[LIGHT_COUNT][STATE_COUNT];

static Drive drive_test_lights_env(TrafficControlElement *lights, int eval_mode) {
    Drive env = {0};
    env.eval_mode = eval_mode;
    env.dt = 0.1f;
    env.scenario_length = STATE_COUNT;
    env.num_traffic_elements = LIGHT_COUNT;
    env.traffic_elements = lights;
    rng_seed(&env.rng_state, 7);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        lights[i] = (TrafficControlElement) {0};
        lights[i].type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
        lights[i].state_size = STATE_COUNT;
        lights[i].states = light_states[i];
        lights[i].junction_id = i < 3 ? 42 : -1;
    }
    return env;
}

static int count_state(const int *states, int state) {
    int count = 0;
    for (int t = 0; t < STATE_COUNT; t++) {
        count += states[t] == state;
    }
    return count;
}

static int test_every_light_runs_its_own_cycle(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    generate_traffic_light_states(&env);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_GREEN), 600);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_YELLOW), 180);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_RED), 120);
        for (int t = 0; t < STATE_COUNT - CYCLE_STEPS; t++) {
            EXPECT_EQ_INT(lights[i].states[t + CYCLE_STEPS], lights[i].states[t]);
        }
    }
    return 0;
}

// Junction fully off while the standalone light is not: individual removal alone needs three 20% draws to line up.
static int count_junction_only_off_episodes(int seed_count) {
    TrafficControlElement lights[LIGHT_COUNT];
    int junction_only_off_count = 0;
    for (int seed = 0; seed < seed_count; seed++) {
        Drive env = drive_test_lights_env(lights, 0);
        rng_seed(&env.rng_state, (uint64_t) seed);
        generate_traffic_light_states(&env);
        int off_lights = 0;
        for (int i = 0; i < LIGHT_COUNT; i++) {
            off_lights += count_state(lights[i].states, TRAFFIC_CONTROL_STATE_OFF) == STATE_COUNT;
        }
        int standalone_off = count_state(lights[3].states, TRAFFIC_CONTROL_STATE_OFF) == STATE_COUNT;
        junction_only_off_count += off_lights == 3 && !standalone_off;
    }
    return junction_only_off_count;
}

// Expected rate per episode: 0.8 (no episode disable) * ~0.206 (group removed) * 0.8 (standalone kept) ~= 13%.
static int test_training_group_removal_turns_off_whole_junction(void) {
    int junction_only_off_count = count_junction_only_off_episodes(500);
    EXPECT_TRUE(junction_only_off_count >= 40);
    EXPECT_TRUE(junction_only_off_count <= 100);
    return 0;
}

static int test_eval_never_removes_lights(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    for (int seed = 0; seed < 50; seed++) {
        Drive env = drive_test_lights_env(lights, 1);
        rng_seed(&env.rng_state, (uint64_t) seed);
        generate_traffic_light_states(&env);
        for (int i = 0; i < LIGHT_COUNT; i++) {
            EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_OFF), 0);
        }
    }
    return 0;
}

// Gigaflow schedules include the terminal step, so the bootstrap observation reads a real light state
static int test_schedule_covers_terminal_observation(void) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.eval_mode = 1; // default CARLA cycles: no light is removed, so the terminal state is never OFF
    allocate(&env);
    c_reset(&env);
    int light_count = 0;
    TrafficControlElement *probe_light = NULL;
    for (int i = 0; i < env.num_traffic_elements; i++) {
        TrafficControlElement *tc = &env.traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        EXPECT_EQ_INT(tc->state_size, env.scenario_length + 1);
        EXPECT_TRUE(tc->states[env.scenario_length] != TRAFFIC_CONTROL_STATE_OFF);
        probe_light = tc;
        light_count++;
    }
    EXPECT_TRUE(light_count > 0);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_x = (probe_light->stop_line[0] + probe_light->stop_line[3]) * 0.5f;
    agent->sim_y = (probe_light->stop_line[1] + probe_light->stop_line[4]) * 0.5f;
    agent->sim_z = (probe_light->stop_line[2] + probe_light->stop_line[5]) * 0.5f;
    copy_pose_to_prev(agent);
    env.timestep = env.scenario_length;
    compute_observations(&env);
    int valid_base = compute_observation_size(&env) - OBS_VALID_COUNT_FEATURES;
    int traffic_base = valid_base - env.obs_slots_traffic_controls_n * TRAFFIC_CONTROL_FEATURES;
    EXPECT_TRUE(env.observations[valid_base + 3] >= 1.0f);
    int observed_state = (int) env.observations[traffic_base + TRAFFIC_CONTROL_FEATURES - 1];
    EXPECT_EQ_INT(observed_state, probe_light->states[env.scenario_length]);
    free_allocated(&env);
    return 0;
}

// The optional yellow-remaining column (before type/state) reads the consecutive yellow steps ahead in
// seconds over traffic_light_yellow_max_s, saturating at 1; every other state reads 0
static int test_yellow_remaining_obs_counts_down_and_saturates(void) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.eval_mode = 1;
    int plain_obs_size = compute_observation_size(&env);
    env.obs_traffic_light_yellow_remaining = 1;
    int row_features = TRAFFIC_CONTROL_FEATURES + TRAFFIC_CONTROL_YELLOW_REMAINING_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), plain_obs_size + env.obs_slots_traffic_controls_n);
    allocate(&env);
    c_reset(&env);
    TrafficControlElement *probe_light = NULL;
    for (int i = 0; i < env.num_traffic_elements && probe_light == NULL; i++) {
        if (env.traffic_elements[i].type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            probe_light = &env.traffic_elements[i];
        }
    }
    EXPECT_TRUE(probe_light != NULL);
    int yellow_start = 10;
    int yellow_steps = 40; // 4 s at dt 0.1: longer than the 3 s training max
    for (int t = 0; t < probe_light->state_size; t++) {
        probe_light->states[t] = TRAFFIC_CONTROL_STATE_RED;
    }
    for (int t = yellow_start; t < yellow_start + yellow_steps; t++) {
        probe_light->states[t] = TRAFFIC_CONTROL_STATE_YELLOW;
    }
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_x = (probe_light->stop_line[0] + probe_light->stop_line[3]) * 0.5f;
    agent->sim_y = (probe_light->stop_line[1] + probe_light->stop_line[4]) * 0.5f;
    agent->sim_z = (probe_light->stop_line[2] + probe_light->stop_line[5]) * 0.5f;
    copy_pose_to_prev(agent);
    int valid_base = compute_observation_size(&env) - OBS_VALID_COUNT_FEATURES;
    int traffic_base = valid_base - env.obs_slots_traffic_controls_n * row_features;
    int yellow_column = traffic_base + row_features - 3;
    int probe_timesteps[4] = {yellow_start, yellow_start + 20, yellow_start + 39, yellow_start + 40};
    float expected[4] = {1.0f, 2.0f / 3.0f, 0.1f / 3.0f, 0.0f};
    int expected_state[4]
        = {TRAFFIC_CONTROL_STATE_YELLOW,
           TRAFFIC_CONTROL_STATE_YELLOW,
           TRAFFIC_CONTROL_STATE_YELLOW,
           TRAFFIC_CONTROL_STATE_RED};
    for (int k = 0; k < 4; k++) {
        env.timestep = probe_timesteps[k];
        compute_observations(&env);
        EXPECT_TRUE(env.observations[valid_base + 3] >= 1.0f);
        EXPECT_TRUE(fabsf(env.observations[yellow_column] - expected[k]) < 1e-5f);
        EXPECT_EQ_INT((int) env.observations[traffic_base + row_features - 2], TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT);
        EXPECT_EQ_INT((int) env.observations[traffic_base + row_features - 1], expected_state[k]);
    }
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_every_light_runs_its_own_cycle);
    RUN_TEST(test_training_group_removal_turns_off_whole_junction);
    RUN_TEST(test_eval_never_removes_lights);
    RUN_TEST(test_schedule_covers_terminal_observation);
    RUN_TEST(test_yellow_remaining_obs_counts_down_and_saturates);
    return test_summary(failures);
}
