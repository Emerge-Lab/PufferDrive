#include "include/drive_fixture.h"
#include "include/test.h"

#define GROUPED_LIGHT_COUNT 4
#define LIGHT_COUNT 5
#define PHASE_COUNT 3
#define STATE_COUNT 900

static const int GROUPED_PHASE_INDICES[GROUPED_LIGHT_COUNT] = {0, 0, 1, 2};

static Drive drive_test_lights_env(TrafficControlElement *lights, int states[LIGHT_COUNT][STATE_COUNT], int eval_mode) {
    Drive env = {0};
    env.eval_mode = eval_mode;
    env.traffic_light_junction_phases = 1;
    env.dt = 0.1f;
    env.scenario_length = STATE_COUNT;
    env.num_traffic_elements = LIGHT_COUNT;
    env.traffic_elements = lights;
    rng_seed(&env.rng_state, 7);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        lights[i] = (TrafficControlElement) {0};
        lights[i].type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
        lights[i].state_size = STATE_COUNT;
        lights[i].states = states[i];
        lights[i].junction_id = i < GROUPED_LIGHT_COUNT ? 42 : -1;
        lights[i].phase_idx = i < GROUPED_LIGHT_COUNT ? GROUPED_PHASE_INDICES[i] : -1;
    }
    return env;
}

static int count_state(const int *states, int state) {
    int count = 0;
    for (int timestep = 0; timestep < STATE_COUNT; timestep++) {
        count += states[timestep] == state;
    }
    return count;
}

static int phase_is_active(TrafficControlElement *lights, int phase_idx, int timestep) {
    for (int i = 0; i < GROUPED_LIGHT_COUNT; i++) {
        int state = lights[i].states[timestep];
        if (lights[i].phase_idx == phase_idx
            && (state == TRAFFIC_CONTROL_STATE_GREEN || state == TRAFFIC_CONTROL_STATE_YELLOW)) {
            return 1;
        }
    }
    return 0;
}

static int test_evaluation_junction_phases_take_turns(void) {
    int states[LIGHT_COUNT][STATE_COUNT];
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, states, 1);
    generate_traffic_light_states(&env);

    for (int timestep = 0; timestep < STATE_COUNT; timestep++) {
        EXPECT_EQ_INT(lights[0].states[timestep], lights[1].states[timestep]);
        int active_phase_count = 0;
        for (int phase_idx = 0; phase_idx < PHASE_COUNT; phase_idx++) {
            active_phase_count += phase_is_active(lights, phase_idx, timestep);
        }
        EXPECT_TRUE(active_phase_count <= 1);
    }
    for (int i = 0; i < GROUPED_LIGHT_COUNT; i++) {
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_GREEN), 200);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_YELLOW), 60);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_RED), 640);
    }
    return 0;
}

static int test_standalone_evaluation_light_keeps_independent_cycle(void) {
    int states[LIGHT_COUNT][STATE_COUNT];
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, states, 1);
    generate_traffic_light_states(&env);

    EXPECT_EQ_INT(count_state(lights[4].states, TRAFFIC_CONTROL_STATE_GREEN), 600);
    EXPECT_EQ_INT(count_state(lights[4].states, TRAFFIC_CONTROL_STATE_YELLOW), 180);
    EXPECT_EQ_INT(count_state(lights[4].states, TRAFFIC_CONTROL_STATE_RED), 120);
    return 0;
}

static int test_training_grouped_lights_remain_exclusive(void) {
    int states[LIGHT_COUNT][STATE_COUNT];
    TrafficControlElement lights[LIGHT_COUNT];
    for (int seed = 0; seed < 64; seed++) {
        Drive env = drive_test_lights_env(lights, states, 0);
        rng_seed(&env.rng_state, (uint64_t) seed);
        generate_traffic_light_states(&env);
        for (int timestep = 0; timestep < STATE_COUNT; timestep++) {
            int active_phase_count = 0;
            for (int phase_idx = 0; phase_idx < PHASE_COUNT; phase_idx++) {
                active_phase_count += phase_is_active(lights, phase_idx, timestep);
            }
            EXPECT_TRUE(active_phase_count <= 1);
        }
        for (int i = 0; i < GROUPED_LIGHT_COUNT; i++) {
            int removed = count_state(lights[i].states, TRAFFIC_CONTROL_STATE_OFF) == STATE_COUNT;
            EXPECT_TRUE(removed || count_state(lights[i].states, TRAFFIC_CONTROL_STATE_GREEN) < STATE_COUNT);
        }
    }
    return 0;
}

static int test_disabled_coordination_preserves_independent_path(void) {
    int actual_states[LIGHT_COUNT][STATE_COUNT];
    int expected_states[LIGHT_COUNT][STATE_COUNT];
    TrafficControlElement actual_lights[LIGHT_COUNT];
    TrafficControlElement expected_lights[LIGHT_COUNT];
    Drive actual = drive_test_lights_env(actual_lights, actual_states, 0);
    Drive expected = drive_test_lights_env(expected_lights, expected_states, 0);
    actual.traffic_light_junction_phases = 0;

    generate_traffic_light_states(&actual);
    generate_independent_traffic_light_states(&expected);

    EXPECT_TRUE(memcmp(actual_states, expected_states, sizeof(actual_states)) == 0);
    EXPECT_TRUE(memcmp(&actual.rng_state, &expected.rng_state, sizeof(Rng)) == 0);
    return 0;
}

static int test_missing_groups_preserve_independent_path(void) {
    int actual_states[LIGHT_COUNT][STATE_COUNT];
    int expected_states[LIGHT_COUNT][STATE_COUNT];
    TrafficControlElement actual_lights[LIGHT_COUNT];
    TrafficControlElement expected_lights[LIGHT_COUNT];
    Drive actual = drive_test_lights_env(actual_lights, actual_states, 0);
    Drive expected = drive_test_lights_env(expected_lights, expected_states, 0);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        actual_lights[i].junction_id = -1;
        actual_lights[i].phase_idx = -1;
        expected_lights[i].junction_id = -1;
        expected_lights[i].phase_idx = -1;
    }

    generate_traffic_light_states(&actual);
    generate_independent_traffic_light_states(&expected);

    EXPECT_TRUE(memcmp(actual_states, expected_states, sizeof(actual_states)) == 0);
    EXPECT_TRUE(memcmp(&actual.rng_state, &expected.rng_state, sizeof(Rng)) == 0);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_evaluation_junction_phases_take_turns);
    RUN_TEST(test_standalone_evaluation_light_keeps_independent_cycle);
    RUN_TEST(test_training_grouped_lights_remain_exclusive);
    RUN_TEST(test_disabled_coordination_preserves_independent_path);
    RUN_TEST(test_missing_groups_preserve_independent_path);
    return test_summary(failures);
}
