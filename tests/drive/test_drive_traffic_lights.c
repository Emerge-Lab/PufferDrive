#include "include/drive_fixture.h"
#include "include/test.h"

// Three lights of one junction (phases 0..2) plus one standalone light; eval cycle 10/3/2 s at dt 0.1.
#define LIGHT_COUNT 4
#define STATE_COUNT 900
#define SLOT_STEPS 150

static int light_states[LIGHT_COUNT][STATE_COUNT];

static Drive drive_test_lights_env(TrafficControlElement *lights, int eval_mode) {
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
        lights[i].states = light_states[i];
        lights[i].junction_id = i < 3 ? 42 : -1;
        lights[i].phase_idx = i < 3 ? i : -1;
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

static int test_junction_lights_never_green_together(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    generate_traffic_light_states(&env);
    for (int t = 0; t < STATE_COUNT; t++) {
        int non_red = 0;
        for (int i = 0; i < 3; i++) {
            non_red += lights[i].states[t] != TRAFFIC_CONTROL_STATE_RED;
        }
        EXPECT_TRUE(non_red <= 1);
    }
    return 0;
}

static int test_junction_lights_share_cycle_and_take_turns(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    generate_traffic_light_states(&env);
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_GREEN), 200);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_YELLOW), 60);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_RED), 640);
    }
    for (int t = 0; t < STATE_COUNT - SLOT_STEPS; t++) {
        // Green slot of phase k is followed one slot later by the green slot of phase (k+1) % 3.
        for (int i = 0; i < 3; i++) {
            if (lights[i].states[t] != TRAFFIC_CONTROL_STATE_GREEN) {
                continue;
            }
            EXPECT_EQ_INT(lights[(i + 1) % 3].states[t + SLOT_STEPS], TRAFFIC_CONTROL_STATE_GREEN);
        }
    }
    return 0;
}

static int test_standalone_light_keeps_single_phase_cycle(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    generate_traffic_light_states(&env);
    EXPECT_EQ_INT(count_state(lights[3].states, TRAFFIC_CONTROL_STATE_GREEN), 600);
    EXPECT_EQ_INT(count_state(lights[3].states, TRAFFIC_CONTROL_STATE_YELLOW), 180);
    EXPECT_EQ_INT(count_state(lights[3].states, TRAFFIC_CONTROL_STATE_RED), 120);
    for (int t = 0; t < STATE_COUNT - SLOT_STEPS; t++) {
        EXPECT_EQ_INT(lights[3].states[t + SLOT_STEPS], lights[3].states[t]);
    }
    return 0;
}

static int test_junction_phases_disabled_makes_every_light_standalone(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    Drive env = drive_test_lights_env(lights, 1);
    env.traffic_light_junction_phases = 0;
    generate_traffic_light_states(&env);
    for (int i = 0; i < LIGHT_COUNT; i++) {
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_GREEN), 600);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_YELLOW), 180);
        EXPECT_EQ_INT(count_state(lights[i].states, TRAFFIC_CONTROL_STATE_RED), 120);
    }
    return 0;
}

static int test_training_lights_stay_exclusive_unless_removed(void) {
    TrafficControlElement lights[LIGHT_COUNT];
    int exclusive_violations = 0;
    for (int seed = 0; seed < 20; seed++) {
        Drive env = drive_test_lights_env(lights, 0);
        rng_seed(&env.rng_state, (uint64_t) seed);
        generate_traffic_light_states(&env);
        for (int t = 0; t < STATE_COUNT; t++) {
            int cycling_non_red = 0;
            for (int i = 0; i < 3; i++) {
                int state = lights[i].states[t];
                int is_cycling = count_state(lights[i].states, TRAFFIC_CONTROL_STATE_RED) > 0;
                cycling_non_red += is_cycling && state != TRAFFIC_CONTROL_STATE_RED;
            }
            exclusive_violations += cycling_non_red > 1;
        }
    }
    EXPECT_EQ_INT(exclusive_violations, 0);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_junction_lights_never_green_together);
    RUN_TEST(test_junction_lights_share_cycle_and_take_turns);
    RUN_TEST(test_standalone_light_keeps_single_phase_cycle);
    RUN_TEST(test_junction_phases_disabled_makes_every_light_standalone);
    RUN_TEST(test_training_lights_stay_exclusive_unless_removed);
    return test_summary(failures);
}
