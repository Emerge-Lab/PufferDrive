#include "drive.h"

#include "../env_config.h"
#include "drivenet.h"

#include <string.h>

// Use this test if the network changes to ensure that the forward pass
// matches the torch implementation to the 3rd or ideally 4th decimal place
void test_drivenet() {
    int num_obs = 1848;
    int num_actions = 2;
    int num_agents = 4;

    float *observations = calloc(num_agents * num_obs, sizeof(float));
    for (int i = 0; i < num_obs * num_agents; i++) {
        observations[i] = i % 7;
    }

    int *actions = calloc(num_agents * num_actions, sizeof(int));

    // Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin");
    Weights *weights = load_weights("puffer_drive_weights.bin");
    DriveNet *net = init_drivenet(weights, num_agents, CLASSIC);

    forward(net, observations, actions);
    for (int i = 0; i < num_agents * num_actions; i++) {
        printf("idx: %d, action: %d, logits:", i, actions[i]);
        for (int j = 0; j < num_actions; j++) {
            printf(" %.6f", net->actor->output[i * num_actions + j]);
        }
        printf("\n");
    }
    free_drivenet(net);
    free(weights);
}

static inline int compute_effective_road_obs_count(int max_count, float dropout) {
    if (max_count <= 0) {
        return 0;
    }
    float clipped_dropout = clip(dropout, 0.0f, 1.0f);
    return (int) (max_count * (1.0f - clipped_dropout));
}

void demo() {
    // Read configuration from INI file
    env_init_config conf = {0};
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    if (load_env_config(ini_file, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    Drive env = {
        .dynamics_model = conf.dynamics_model,
        .reward_collision = conf.reward_collision,
        .reward_offroad = conf.reward_offroad,
        .reward_ade = conf.reward_ade,
        .goal_radius = conf.goal_radius,
        .dt = conf.dt,
        .spawn_initial_speed = conf.spawn_initial_speed,
        .goal_speed = conf.goal_speed,
        .map_name = "pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin",
        .init_step = conf.init_step,
        .collision_behavior = conf.collision_behavior,
        .offroad_behavior = conf.offroad_behavior,
        .compute_eval_metrics = conf.compute_eval_metrics,
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights *weights = load_weights("resources/drive/puffer_drive_weights.bin");
    DriveNet *net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);
    // Client* client = make_client(&env);
    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        // Handle camera controls
        int (*actions)[2] = (int (*)[2]) env.actions;
        forward(net, env.observations, env.actions);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[0][0] = 3;
            actions[0][1] = 6;
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                actions[0][0] += accel_delta;
                // Cap acceleration to maximum of 6
                if (actions[0][0] > 6) {
                    actions[0][0] = 6;
                }
            }
            if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                actions[0][0] -= accel_delta;
                // Cap acceleration to minimum of 0
                if (actions[0][0] < 0) {
                    actions[0][0] = 0;
                }
            }
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                actions[0][1] += steer_delta;
                // Cap steering to minimum of 0
                if (actions[0][1] < 0) {
                    actions[0][1] = 0;
                }
            }
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                actions[0][1] -= steer_delta;
                // Cap steering to maximum of 12
                if (actions[0][1] > 12) {
                    actions[0][1] = 12;
                }
            }
        }
        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
}

void performance_test() {
    // Read configuration from INI file
    env_init_config conf = {0};
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    if (load_env_config(ini_file, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    long test_time = 10;
    Drive env = {
        .map_name = strdup("pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin"),
        .num_controllable_agents = conf.max_agents_per_env,
        // From conf
        .action_type = conf.action_type,
        .dynamics_model = conf.dynamics_model,
        .reward_collision = conf.reward_collision,
        .reward_offroad = conf.reward_offroad,
        .reward_stop_line = conf.reward_stop_line,
        .reward_goal = conf.reward_goal,
        .reward_ade = conf.reward_ade,
        .reward_overspeed = conf.reward_overspeed,
        .reward_comfort = conf.reward_comfort,
        .reward_velocity = conf.reward_velocity,
        .reward_lane_align = conf.reward_lane_align,
        .reward_vel_align = conf.reward_vel_align,
        .reward_lane_center = conf.reward_lane_center,
        .reward_center_bias = conf.reward_center_bias,
        .reward_reverse = conf.reward_reverse,
        .reward_timestep = conf.reward_timestep,
        .goal_radius = conf.goal_radius,
        .collision_behavior = conf.collision_behavior,
        .offroad_behavior = conf.offroad_behavior,
        .traffic_light_behavior = conf.traffic_light_behavior,
        .dt = conf.dt,
        .spawn_initial_speed = conf.spawn_initial_speed,
        .goal_speed = conf.goal_speed,
        .target_type = conf.target_type,
        .scenario_length = conf.scenario_length,
        .termination_mode = conf.termination_mode,
        .init_step = conf.init_step,
        .init_mode = conf.init_mode,
        .control_mode = conf.control_mode,
        .simulation_mode = conf.simulation_mode,
        .min_waypoint_spacing = conf.min_waypoint_spacing,
        .max_waypoint_spacing = conf.max_waypoint_spacing,
        .num_target_waypoints = conf.num_target_waypoints,
        .reward_conditioning = conf.reward_conditioning,
        .reward_randomization = conf.reward_randomization,
        .compute_eval_metrics = conf.compute_eval_metrics,
        .num_max_agents = conf.max_agents_per_env,
        .obs_slots_lane_n = conf.obs_slots_lane_n,
        .obs_slots_boundary_n = conf.obs_slots_boundary_n,
        .obs_slots_partners_n = conf.obs_slots_partners_n,
        .obs_slots_traffic_controls_n = conf.obs_slots_traffic_controls_n,
        .traffic_control_scope = conf.traffic_control_scope,
        .partner_blindness_prob = conf.partner_blindness_prob,
        .partner_blindness_trigger_prob = conf.partner_blindness_trigger_prob,
        .phantom_braking_prob = conf.phantom_braking_prob,
        .phantom_braking_trigger_prob = conf.phantom_braking_trigger_prob,
        .phantom_braking_duration = conf.phantom_braking_duration,

    };
    env.obs_slots_lane_kept = compute_effective_road_obs_count(env.obs_slots_lane, conf.obs_dropout_lane);
    env.obs_slots_boundary_kept = compute_effective_road_obs_count(env.obs_slots_boundary, conf.obs_dropout_boundary);

    struct timespec ts_total_start, ts_total_end;
    struct timespec ts_init_start, ts_init_end;
    struct timespec ts_step_start, ts_step_end;
    double init_time = 0, step_time = 0, total_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts_total_start);

    clock_gettime(CLOCK_MONOTONIC, &ts_init_start);
    allocate(&env);
    c_reset(&env);
    clock_gettime(CLOCK_MONOTONIC, &ts_init_end);
    init_time = (ts_init_end.tv_sec - ts_init_start.tv_sec) + (ts_init_end.tv_nsec - ts_init_start.tv_nsec) / 1e9;
    printf("Init time: %.4f s\n", init_time);

    long start = time(NULL);
    int i = 0;

    // Reallocate actions buffer for trajectory mode (needs num_trajectory_scaling_factors per agent)

    clock_gettime(CLOCK_MONOTONIC, &ts_step_start);
    while (time(NULL) - start < test_time) {
        // Set random discrete actions for all agents
        int (*actions)[2] = (int (*)[2]) env.actions;
        for (int j = 0; j < env.active_agent_count; j++) {
            actions[j][0] = rand_r(&env.rng_state) % 7;
            actions[j][1] = rand_r(&env.rng_state) % 13;
        }
        c_step(&env);
        i++;
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_step_end);
    step_time = (ts_step_end.tv_sec - ts_step_start.tv_sec) + (ts_step_end.tv_nsec - ts_step_start.tv_nsec) / 1e9;

    long end = time(NULL);
    printf("Steps: %d | Agents: %d\n", i, env.active_agent_count);
    printf("Step loop time: %.4f s\n", step_time);
    printf("SPS: %ld\n", (i * env.active_agent_count) / (end - start));

    free_allocated(&env);

    clock_gettime(CLOCK_MONOTONIC, &ts_total_end);
    total_time = (ts_total_end.tv_sec - ts_total_start.tv_sec) + (ts_total_end.tv_nsec - ts_total_start.tv_nsec) / 1e9;
    printf("Total time: %.4f s\n", total_time);
}

int main() {
    performance_test();
    // demo();
    // test_drivenet();
    return 0;
}
