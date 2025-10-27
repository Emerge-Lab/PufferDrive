#include "drive.h"
#include "drivenet.h"
#include <string.h>
#include "ini.h"

// Config struct for parsing INI files
typedef struct {
    int action_type;
    int dynamics_model;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_goal;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float reward_ade;
    float goal_radius;
    int spawn_immunity_timer;
    float dt;
    int use_goal_generation;
    int control_non_vehicles;
    int scenario_length;
    int init_steps;
    int control_all_agents;
    int num_policy_controlled_agents;
    int deterministic_agent_selection;
} env_init_config;

// INI file parser handler
static int handler(void* config, const char* section, const char* name, const char* value) {
    env_init_config* env_config = (env_init_config*)config;
    #define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

    if (MATCH("env", "action_type")) {
        env_config->action_type = (strcmp(value, "\"discrete\"") == 0) ? 0 : 1;
    } else if (MATCH("env", "dynamics_model")) {
        if (strcmp(value, "\"classic\"") == 0 || strcmp(value, "classic") == 0) {
            env_config->dynamics_model = 0;  // CLASSIC
        } else if (strcmp(value, "\"jerk\"") == 0 || strcmp(value, "jerk") == 0) {
            env_config->dynamics_model = 1;  // JERK
        } else {
            printf("Warning: Unknown dynamics_model value '%s', defaulting to JERK\n", value);
            env_config->dynamics_model = 1;  // Default to JERK
        }
    } else if (MATCH("env", "use_goal_generation")) {
        env_config->use_goal_generation = (strcmp(value, "True") == 0) ? 1 : 0;
    } else if (MATCH("env", "reward_vehicle_collision")) {
        env_config->reward_vehicle_collision = atof(value);
    } else if (MATCH("env", "reward_offroad_collision")) {
        env_config->reward_offroad_collision = atof(value);
    } else if (MATCH("env", "reward_goal")) {
        env_config->reward_goal = atof(value);
    } else if (MATCH("env", "reward_goal_post_respawn")) {
        env_config->reward_goal_post_respawn = atof(value);
    } else if (MATCH("env", "reward_vehicle_collision_post_respawn")) {
        env_config->reward_vehicle_collision_post_respawn = atof(value);
    } else if (MATCH("env", "reward_ade")) {
        env_config->reward_ade = atof(value);
    } else if (MATCH("env", "goal_radius")) {
        env_config->goal_radius = atof(value);
    } else if (MATCH("env", "spawn_immunity_timer")) {
        env_config->spawn_immunity_timer = atoi(value);
    } else if (MATCH("env", "dt")) {
        env_config->dt = atof(value);
    } else if (MATCH("env", "control_non_vehicles")) {
        env_config->control_non_vehicles = (strcmp(value, "True") == 0) ? 1 : 0;
    } else if (MATCH("env", "scenario_length")) {
        env_config->scenario_length = atoi(value);
    } else if (MATCH("env", "init_steps")) {
        env_config->init_steps = atoi(value);
    } else if (MATCH("env", "control_all_agents")) {
        env_config->control_all_agents = (strcmp(value, "True") == 0) ? 1 : 0;
    } else if (MATCH("env", "num_policy_controlled_agents")) {
        env_config->num_policy_controlled_agents = atoi(value);
    } else if (MATCH("env", "deterministic_agent_selection")) {
        env_config->deterministic_agent_selection = (strcmp(value, "True") == 0) ? 1 : 0;
    }

    #undef MATCH
    return 1;
}

// Use this test if the network changes to ensure that the forward pass
// matches the torch implementation to the 3rd or ideally 4th decimal place
void test_drivenet() {
    int num_obs = 1848;
    int num_actions = 2;
    int num_agents = 4;

    float* observations = calloc(num_agents*num_obs, sizeof(float));
    for (int i = 0; i < num_obs*num_agents; i++) {
        observations[i] = i % 7;
    }

    int* actions = calloc(num_agents*num_actions, sizeof(int));

    //Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin");
    Weights* weights = load_weights("puffer_drive_weights.bin");
    DriveNet* net = init_drivenet(weights, num_agents);

    forward(net, observations, actions);
    for (int i = 0; i < num_agents*num_actions; i++) {
        printf("idx: %d, action: %d, logits:", i, actions[i]);
        for (int j = 0; j < num_actions; j++) {
            printf(" %.6f", net->actor->output[i*num_actions + j]);
        }
        printf("\n");
    }
    free_drivenet(net);
    free(weights);
}

void demo() {
    // Read configuration from INI file
    env_init_config conf = {0};
    const char* ini_file = "pufferlib/config/ocean/drive.ini";
    if(ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    Drive env = {
        .human_agent_idx = 0,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_ade = conf.reward_ade,
        .goal_radius = conf.goal_radius,
        .dt = conf.dt,
	    .map_name = "resources/drive/binaries/map_000.bin",
        .control_non_vehicles = conf.control_non_vehicles,
        .init_steps = conf.init_steps,
        .control_all_agents = conf.control_all_agents,
        .policy_agents_per_env = conf.num_policy_controlled_agents,
        .deterministic_agent_selection = conf.deterministic_agent_selection,
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin");
    DriveNet* net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);
    //Client* client = make_client(&env);
    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        // Handle camera controls
        int (*actions)[2] = (int(*)[2])env.actions;
        forward(net, env.observations, env.actions);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[env.human_agent_idx][0] = 3;
            actions[env.human_agent_idx][1] = 6;
            if(IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)){
                actions[env.human_agent_idx][0] += accel_delta;
                // Cap acceleration to maximum of 6
                if(actions[env.human_agent_idx][0] > 6) {
                    actions[env.human_agent_idx][0] = 6;
                }
            }
            if(IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)){
                actions[env.human_agent_idx][0] -= accel_delta;
                // Cap acceleration to minimum of 0
                if(actions[env.human_agent_idx][0] < 0) {
                    actions[env.human_agent_idx][0] = 0;
                }
            }
            if(IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)){
                actions[env.human_agent_idx][1] += steer_delta;
                // Cap steering to minimum of 0
                if(actions[env.human_agent_idx][1] < 0) {
                    actions[env.human_agent_idx][1] = 0;
                }
            }
            if(IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)){
                actions[env.human_agent_idx][1] -= steer_delta;
                // Cap steering to maximum of 12
                if(actions[env.human_agent_idx][1] > 12) {
                    actions[env.human_agent_idx][1] = 12;
                }
            }
            if(IsKeyPressed(KEY_TAB)){
                env.human_agent_idx = (env.human_agent_idx + 1) % env.active_agent_count;
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
    const char* ini_file = "pufferlib/config/ocean/drive.ini";
    if(ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    long test_time = 10;
    Drive env = {
        .human_agent_idx = 0,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_ade = conf.reward_ade,
        .goal_radius = conf.goal_radius,
        .dt = conf.dt,
	    .map_name = "resources/drive/binaries/map_000.bin",
        .control_non_vehicles = conf.control_non_vehicles,
        .init_steps = conf.init_steps,
        .control_all_agents = conf.control_all_agents,
        .policy_agents_per_env = conf.num_policy_controlled_agents,
        .deterministic_agent_selection = conf.deterministic_agent_selection,
    };
    clock_t start_time, end_time;
    double cpu_time_used;
    start_time = clock();
    allocate(&env);
    c_reset(&env);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Init time: %f\n", cpu_time_used);

    long start = time(NULL);
    int i = 0;
    int (*actions)[2] = (int(*)[2])env.actions;

    while (time(NULL) - start < test_time) {
        // Set random actions for all agents
        for(int j = 0; j < env.active_agent_count; j++) {
            int accel = rand() % 7;
            int steer = rand() % 13;
            actions[j][0] = accel;  // -1, 0, or 1
            actions[j][1] = steer;  // Random steering
        }

        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i*env.active_agent_count) / (end - start));
    free_allocated(&env);
}

int main() {
    //performance_test();
    demo();
    //test_drivenet();
    return 0;
}
