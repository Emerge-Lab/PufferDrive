#include "include/drive_fixture.h"
#include "include/test.h"

#define ELEMENT_COUNT 6
#define SCHEDULE_STEPS 12
#define FIXTURE_LIGHT_A_IDX 0
#define FIXTURE_STOP_SIGN_IDX 1
#define FIXTURE_LIGHT_B_IDX 2
#define FIXTURE_UNSCHEDULED_LIGHT_IDX 3
#define FIXTURE_YIELD_SIGN_IDX 4
#define FIXTURE_NONE_IDX 5
#define SENTINEL_STATE -7
#define CO_LOCATED_DISTANCE_M 0.5f
#define OBS_TRAFFIC_VALID_COUNT_IDX 3
#define PAST_THE_SCHEDULE -1

static int schedules[ELEMENT_COUNT][SCHEDULE_STEPS];

static Drive synthetic_env(TrafficControlElement *elements) {
    Drive env = {0};
    env.num_traffic_elements = ELEMENT_COUNT;
    env.traffic_elements = elements;
    int types[ELEMENT_COUNT] = {
        TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT,
        TRAFFIC_CONTROL_TYPE_STOP_SIGN,
        TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT,
        TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT,
        TRAFFIC_CONTROL_TYPE_YIELD_SIGN,
        TRAFFIC_CONTROL_TYPE_NONE,
    };
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        elements[i] = (TrafficControlElement) {0};
        elements[i].type = types[i];
        elements[i].state_size = SCHEDULE_STEPS;
        elements[i].states = schedules[i];
        for (int t = 0; t < SCHEDULE_STEPS; t++) {
            schedules[i][t] = TRAFFIC_CONTROL_STATE_RED + (t + i) % 3;
        }
    }
    elements[FIXTURE_UNSCHEDULED_LIGHT_IDX].states = NULL;
    elements[FIXTURE_UNSCHEDULED_LIGHT_IDX].state_size = 0;
    return env;
}

static void fill_sentinel(int *states) {
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        states[i] = SENTINEL_STATE;
    }
}

static int test_lights_report_their_schedule_at_the_current_timestep(void) {
    TrafficControlElement elements[ELEMENT_COUNT];
    Drive env = synthetic_env(elements);
    int states[ELEMENT_COUNT];
    for (int t = 0; t < SCHEDULE_STEPS; t++) {
        env.timestep = t;
        fill_sentinel(states);
        c_get_traffic_light_states(&env, states);
        EXPECT_EQ_INT(states[FIXTURE_LIGHT_A_IDX], schedules[FIXTURE_LIGHT_A_IDX][t]);
        EXPECT_EQ_INT(states[FIXTURE_LIGHT_B_IDX], schedules[FIXTURE_LIGHT_B_IDX][t]);
    }
    return 0;
}

static int test_non_lights_report_unknown_even_with_a_schedule(void) {
    TrafficControlElement elements[ELEMENT_COUNT];
    Drive env = synthetic_env(elements);
    env.timestep = 4;
    int states[ELEMENT_COUNT];
    fill_sentinel(states);
    c_get_traffic_light_states(&env, states);
    EXPECT_EQ_INT(states[FIXTURE_STOP_SIGN_IDX], TRAFFIC_CONTROL_STATE_UNKNOWN);
    EXPECT_EQ_INT(states[FIXTURE_YIELD_SIGN_IDX], TRAFFIC_CONTROL_STATE_UNKNOWN);
    EXPECT_EQ_INT(states[FIXTURE_NONE_IDX], TRAFFIC_CONTROL_STATE_UNKNOWN);
    return 0;
}

static int test_unscheduled_light_reports_off(void) {
    TrafficControlElement elements[ELEMENT_COUNT];
    Drive env = synthetic_env(elements);
    env.timestep = 2;
    int states[ELEMENT_COUNT];
    fill_sentinel(states);
    c_get_traffic_light_states(&env, states);
    EXPECT_EQ_INT(states[FIXTURE_UNSCHEDULED_LIGHT_IDX], TRAFFIC_CONTROL_STATE_OFF);
    return 0;
}

static int test_timestep_outside_the_schedule_reports_off(void) {
    TrafficControlElement elements[ELEMENT_COUNT];
    Drive env = synthetic_env(elements);
    int outside_timesteps[] = {SCHEDULE_STEPS, SCHEDULE_STEPS + 5, -1};
    int states[ELEMENT_COUNT];
    for (int k = 0; k < 3; k++) {
        env.timestep = outside_timesteps[k];
        fill_sentinel(states);
        c_get_traffic_light_states(&env, states);
        EXPECT_EQ_INT(states[FIXTURE_LIGHT_A_IDX], TRAFFIC_CONTROL_STATE_OFF);
        EXPECT_EQ_INT(states[FIXTURE_LIGHT_B_IDX], TRAFFIC_CONTROL_STATE_OFF);
        EXPECT_EQ_INT(states[FIXTURE_STOP_SIGN_IDX], TRAFFIC_CONTROL_STATE_UNKNOWN);
    }
    return 0;
}

static int test_getter_leaves_the_sim_untouched(void) {
    TrafficControlElement elements[ELEMENT_COUNT];
    Drive env = synthetic_env(elements);
    env.timestep = 3;
    int before[ELEMENT_COUNT][SCHEDULE_STEPS];
    memcpy(before, schedules, sizeof(before));
    int states[ELEMENT_COUNT];
    c_get_traffic_light_states(&env, states);
    EXPECT_TRUE(memcmp(before, schedules, sizeof(before)) == 0);
    EXPECT_EQ_INT(env.timestep, 3);
    return 0;
}

static int light_is_alone_at_its_stop_line(const Drive *env, int light_idx) {
    const TrafficControlElement *light = &env->traffic_elements[light_idx];
    float mid_x = (light->stop_line[0] + light->stop_line[3]) * 0.5f;
    float mid_y = (light->stop_line[1] + light->stop_line[4]) * 0.5f;
    for (int j = 0; j < env->num_traffic_elements; j++) {
        const TrafficControlElement *other = &env->traffic_elements[j];
        float dx = (other->stop_line[0] + other->stop_line[3]) * 0.5f - mid_x;
        float dy = (other->stop_line[1] + other->stop_line[4]) * 0.5f - mid_y;
        if (j != light_idx && dx * dx + dy * dy < CO_LOCATED_DISTANCE_M * CO_LOCATED_DISTANCE_M) {
            return 0;
        }
    }
    return 1;
}

static int observed_nearest_control_state(Drive *env, const TrafficControlElement *light) {
    Agent *agent = &env->agents[env->active_agent_indices[0]];
    agent->sim_x = (light->stop_line[0] + light->stop_line[3]) * 0.5f;
    agent->sim_y = (light->stop_line[1] + light->stop_line[4]) * 0.5f;
    agent->sim_z = (light->stop_line[2] + light->stop_line[5]) * 0.5f;
    copy_pose_to_prev(agent);
    compute_observations(env);
    int valid_base = compute_observation_size(env) - OBS_VALID_COUNT_FEATURES;
    int traffic_base = valid_base - env->obs_slots_traffic_controls_n * TRAFFIC_CONTROL_FEATURES;
    if (env->observations[valid_base + OBS_TRAFFIC_VALID_COUNT_IDX] < 1.0f) {
        return SENTINEL_STATE;
    }
    return (int) env->observations[traffic_base + TRAFFIC_CONTROL_FEATURES - 1];
}

static int expect_getter_matches_observation(int eval_mode, int timestep) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.eval_mode = eval_mode;
    allocate(&env);
    c_reset(&env);
    env.timestep = timestep == PAST_THE_SCHEDULE ? env.scenario_length + 1 : timestep;
    int *states = calloc(env.num_traffic_elements, sizeof(int));
    c_get_traffic_light_states(&env, states);
    int compared = 0;
    int mismatched_idx = -1;
    for (int i = 0; i < env.num_traffic_elements; i++) {
        if (env.traffic_elements[i].type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || !light_is_alone_at_its_stop_line(&env, i)) {
            continue;
        }
        int observed = observed_nearest_control_state(&env, &env.traffic_elements[i]);
        if (observed != states[i]) {
            mismatched_idx = i;
        }
        compared++;
    }
    free(states);
    free_allocated(&env);
    EXPECT_EQ_INT(mismatched_idx, -1);
    EXPECT_TRUE(compared > 0);
    return 0;
}

static int test_getter_matches_the_observation_in_eval(void) {
    EXPECT_EQ_INT(expect_getter_matches_observation(1, 0), 0);
    EXPECT_EQ_INT(expect_getter_matches_observation(1, 37), 0);
    return 0;
}

static int test_getter_matches_the_observation_with_lights_removed_in_training(void) {
    EXPECT_EQ_INT(expect_getter_matches_observation(0, 5), 0);
    return 0;
}

static int test_getter_matches_the_observation_past_the_schedule(void) {
    EXPECT_EQ_INT(expect_getter_matches_observation(1, PAST_THE_SCHEDULE), 0);
    return 0;
}

static int test_getter_reads_back_an_override(void) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.eval_mode = 1;
    allocate(&env);
    c_reset(&env);
    env.timestep = 10;
    int *wanted = calloc(env.num_traffic_elements, sizeof(int));
    int *read_back = calloc(env.num_traffic_elements, sizeof(int));
    for (int i = 0; i < env.num_traffic_elements; i++) {
        wanted[i] = TRAFFIC_CONTROL_STATE_RED + i % 3;
    }
    int set_result = c_set_traffic_light_states(&env, wanted);
    c_get_traffic_light_states(&env, read_back);
    int mismatched_idx = -1;
    for (int i = 0; i < env.num_traffic_elements; i++) {
        int expected = env.traffic_elements[i].type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT ? wanted[i]
                                                                                         : TRAFFIC_CONTROL_STATE_UNKNOWN;
        if (read_back[i] != expected) {
            mismatched_idx = i;
        }
    }
    free(wanted);
    free(read_back);
    free_allocated(&env);
    EXPECT_EQ_INT(set_result, 0);
    EXPECT_EQ_INT(mismatched_idx, -1);
    return 0;
}

static int test_lane_speed_limit_target_reads_the_lane_or_falls_back(void) {
    float lane_limits_mps[3] = {13.9f, 0.0f, -2.0f};
    Drive env = {0};
    env.lane_speed_limit_mps = lane_limits_mps;
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    agent.current_lane_idx = 0;
    EXPECT_NEAR(lane_speed_limit_target(&env, &agent), 13.9f, 0.0f);
    agent.current_lane_idx = 1;
    EXPECT_NEAR(lane_speed_limit_target(&env, &agent), UNKNOWN_LANE_SPEED_LIMIT_MPS, 0.0f);
    agent.current_lane_idx = 2;
    EXPECT_NEAR(lane_speed_limit_target(&env, &agent), UNKNOWN_LANE_SPEED_LIMIT_MPS, 0.0f);
    agent.current_lane_idx = -1;
    EXPECT_NEAR(lane_speed_limit_target(&env, &agent), UNKNOWN_LANE_SPEED_LIMIT_MPS, 0.0f);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_lights_report_their_schedule_at_the_current_timestep);
    RUN_TEST(test_non_lights_report_unknown_even_with_a_schedule);
    RUN_TEST(test_unscheduled_light_reports_off);
    RUN_TEST(test_timestep_outside_the_schedule_reports_off);
    RUN_TEST(test_getter_leaves_the_sim_untouched);
    RUN_TEST(test_getter_matches_the_observation_in_eval);
    RUN_TEST(test_getter_matches_the_observation_with_lights_removed_in_training);
    RUN_TEST(test_getter_matches_the_observation_past_the_schedule);
    RUN_TEST(test_getter_reads_back_an_override);
    RUN_TEST(test_lane_speed_limit_target_reads_the_lane_or_falls_back);
    return test_summary(failures);
}
