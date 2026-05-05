#ifndef ENV_CONFIG_H
#define ENV_CONFIG_H

#include <../../inih-r62/ini.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Config struct for parsing INI files - contains all environment configuration
typedef struct {
    int render_mode;
    int action_type;
    int dynamics_model;
    bool fix_rewards;
    bool fix_lambdas;
    float lambda_value;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_goal;
    float goal_radius;
    float goal_speed;
    int collision_behavior;
    int offroad_behavior;
    int spawn_immunity_timer;
    float dt;
    int goal_behavior;
    float goal_target_distance;
    int episode_length;
    int termination_mode;
    int init_steps;
    int init_mode;
    int control_mode;
    int sdc_controller;
    int non_sdc_controller;
    int non_vehicle_controller;
    int max_controlled_agents;
    char map_dir[256];
    char anchor_policy_path[256];
    int reg_mode;
    float obs_partner_noise_pos;
    float obs_partner_noise_speed;
    bool async_resets;
} env_init_config;

// INI file parser handler - parses all environment configuration from drive.ini
static int handler(void *config, const char *section, const char *name, const char *value) {
    env_init_config *env_config = (env_init_config *)config;
#define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

    if (MATCH("env", "action_type")) {
        if (strcmp(value, "\"discrete\"") == 0 || strcmp(value, "discrete") == 0) {
            env_config->action_type = 0; // DISCRETE
        } else if (strcmp(value, "\"continuous\"") == 0 || strcmp(value, "continuous") == 0) {
            env_config->action_type = 1; // CONTINUOUS
        } else {
            printf("Warning: Unknown action_type value '%s', defaulting to DISCRETE\n", value);
            env_config->action_type = 0; // Default to DISCRETE
        }
    } else if (MATCH("env", "dynamics_model")) {
        if (strcmp(value, "\"classic\"") == 0 || strcmp(value, "classic") == 0) {
            env_config->dynamics_model = 0; // CLASSIC
        } else if (strcmp(value, "\"jerk\"") == 0 || strcmp(value, "jerk") == 0) {
            env_config->dynamics_model = 1; // JERK
        } else if (strcmp(value, "\"delta_local\"") == 0 || strcmp(value, "delta_local") == 0) {
            env_config->dynamics_model = 2; // DELTA_LOCAL
        } else {
            printf("Warning: Unknown dynamics_model value '%s', defaulting to CLASSIC\n", value);
            env_config->dynamics_model = 0; // Default to CLASSIC
        }
    } else if (MATCH("env", "goal_behavior")) {
        env_config->goal_behavior = atoi(value);
    } else if (MATCH("env", "goal_target_distance")) {
        env_config->goal_target_distance = atof(value);
    } else if (MATCH("env", "reward_vehicle_collision")) {
        env_config->reward_vehicle_collision = atof(value);
    } else if (MATCH("env", "obs_partner_noise_speed")) {
        env_config->obs_partner_noise_speed = atof(value);
    } else if (MATCH("env", "obs_partner_noise_pos")) {
        env_config->obs_partner_noise_pos = atof(value);
    } else if (MATCH("env", "reward_offroad_collision")) {
        env_config->reward_offroad_collision = atof(value);
    } else if (MATCH("env", "reward_goal")) {
        env_config->reward_goal = atof(value);
    } else if (MATCH("env", "goal_radius")) {
        env_config->goal_radius = atof(value);
    } else if (MATCH("env", "goal_speed")) {
        env_config->goal_speed = atof(value);
    } else if (MATCH("env", "collision_behavior")) {
        env_config->collision_behavior = atoi(value);
    } else if (MATCH("env", "offroad_behavior")) {
        env_config->offroad_behavior = atoi(value);
    } else if (MATCH("env", "spawn_immunity_timer")) {
        env_config->spawn_immunity_timer = atoi(value);
    } else if (MATCH("env", "dt")) {
        env_config->dt = atof(value);
    } else if (MATCH("env", "episode_length")) {
        env_config->episode_length = atoi(value);
    } else if (MATCH("env", "termination_mode")) {
        env_config->termination_mode = atoi(value);
    } else if (MATCH("env", "init_steps")) {
        env_config->init_steps = atoi(value);
    } else if (MATCH("env", "init_mode")) {
        if (strcmp(value, "\"create_all_valid\"") == 0 || strcmp(value, "create_all_valid") == 0) {
            env_config->init_mode = 0;
        } else if (strcmp(value, "\"create_only_controlled\"") == 0 || strcmp(value, "create_only_controlled") == 0) {
            env_config->init_mode = 1;
        } else {
            printf("Warning: Unknown init_mode value '%s', defaulting to CREATE_ALL_VALID\n", value);
            env_config->init_mode = 0; // Default to CREATE_ALL_VALID
        }
    } else if (MATCH("env", "control_mode")) {
        if (strcmp(value, "\"control_vehicles\"") == 0 || strcmp(value, "control_vehicles") == 0) {
            env_config->control_mode = 0;
        } else if (strcmp(value, "\"control_agents\"") == 0 || strcmp(value, "control_agents") == 0) {
            env_config->control_mode = 1;
        } else if (strcmp(value, "\"control_wosac\"") == 0 || strcmp(value, "control_wosac") == 0) {
            env_config->control_mode = 2;
        } else if (strcmp(value, "\"control_sdc_only\"") == 0 || strcmp(value, "control_sdc_only") == 0) {
            env_config->control_mode = 3;
        } else if (strcmp(value, "\"control_mixed_play\"") == 0 || strcmp(value, "control_mixed_play") == 0) {
            env_config->control_mode = 4;
        } else {
            printf("Warning: Unknown control_mode value '%s', defaulting to CONTROL_VEHICLES\n", value);
            env_config->control_mode = 0; // Default to CONTROL_VEHICLES
        }
    } else if (MATCH("env", "sdc_controller")) {
        if (strcmp(value, "\"static\"") == 0 || strcmp(value, "static") == 0) {
            env_config->sdc_controller = 0;
        } else if (strcmp(value, "\"policy\"") == 0 || strcmp(value, "policy") == 0) {
            env_config->sdc_controller = 1;
        } else if (strcmp(value, "\"replay\"") == 0 || strcmp(value, "replay") == 0) {
            env_config->sdc_controller = 2;
        } else if (strcmp(value, "\"idm\"") == 0 || strcmp(value, "idm") == 0) {
            env_config->sdc_controller = 3;
        } else if (strcmp(value, "\"corridor_idm\"") == 0 || strcmp(value, "corridor_idm") == 0) {
            env_config->sdc_controller = 4;
        } else {
            printf("Warning: Unknown sdc_controller value '%s', defaulting to POLICY\n", value);
            env_config->sdc_controller = 1;
        }
    } else if (MATCH("env", "non_sdc_controller")) {
        if (strcmp(value, "\"static\"") == 0 || strcmp(value, "static") == 0) {
            env_config->non_sdc_controller = 0;
        } else if (strcmp(value, "\"policy\"") == 0 || strcmp(value, "policy") == 0) {
            env_config->non_sdc_controller = 1;
        } else if (strcmp(value, "\"replay\"") == 0 || strcmp(value, "replay") == 0) {
            env_config->non_sdc_controller = 2;
        } else if (strcmp(value, "\"idm\"") == 0 || strcmp(value, "idm") == 0) {
            env_config->non_sdc_controller = 3;
        } else if (strcmp(value, "\"corridor_idm\"") == 0 || strcmp(value, "corridor_idm") == 0) {
            env_config->non_sdc_controller = 4;
        } else {
            printf("Warning: Unknown non_sdc_controller value '%s', defaulting to POLICY\n", value);
            env_config->non_sdc_controller = 1;
        }
    } else if (MATCH("env", "non_vehicle_controller")) {
        if (strcmp(value, "\"auto\"") == 0 || strcmp(value, "auto") == 0) {
            if (env_config->non_sdc_controller == 3 || env_config->non_sdc_controller == 4) {
                env_config->non_vehicle_controller = 2;
            } else {
                env_config->non_vehicle_controller =
                    env_config->non_sdc_controller != 0 ? env_config->non_sdc_controller : 1;
            }
        } else if (strcmp(value, "\"static\"") == 0 || strcmp(value, "static") == 0) {
            env_config->non_vehicle_controller = 0;
        } else if (strcmp(value, "\"policy\"") == 0 || strcmp(value, "policy") == 0) {
            env_config->non_vehicle_controller = 1;
        } else if (strcmp(value, "\"replay\"") == 0 || strcmp(value, "replay") == 0) {
            env_config->non_vehicle_controller = 2;
        } else if (strcmp(value, "\"idm\"") == 0 || strcmp(value, "idm") == 0) {
            env_config->non_vehicle_controller = 3;
        } else if (strcmp(value, "\"corridor_idm\"") == 0 || strcmp(value, "corridor_idm") == 0) {
            env_config->non_vehicle_controller = 4;
        } else {
            printf("Warning: Unknown non_vehicle_controller value '%s', defaulting to REPLAY\n", value);
            env_config->non_vehicle_controller = 2;
        }
    } else if (MATCH("env", "reg_mode")) {
        if (strcmp(value, "\"None\"") == 0 || strcmp(value, "None") == 0 || strcmp(value, "\"none\"") == 0 ||
            strcmp(value, "none") == 0) {
            env_config->reg_mode = 0; // NONE
        } else if (strcmp(value, "\"log_prob_direct\"") == 0 || strcmp(value, "log_prob_direct") == 0) {
            env_config->reg_mode = 1; // LOG_PROB_DIRECT
        } else if (strcmp(value, "\"kl_anchor\"") == 0 || strcmp(value, "kl_anchor") == 0) {
            env_config->reg_mode = 2; // KL_ANCHOR
        } else {
            printf("Warning: Unknown reg_mode value '%s', defaulting to NONE\n", value);
            env_config->reg_mode = 0; // Default to NONE
        }
    } else if (MATCH("env", "map_dir")) {
        if (sscanf(value, "\"%255[^\"]\"", env_config->map_dir) != 1) {
            strncpy(env_config->map_dir, value, sizeof(env_config->map_dir) - 1);
            env_config->map_dir[sizeof(env_config->map_dir) - 1] = '\0';
        }
        // printf("Parsed map_dir: '%s'\n", env_config->map_dir);
    } else if (MATCH("env", "max_controlled_agents")) {
        env_config->max_controlled_agents = atoi(value);
    } else if (MATCH("env", "anchor_policy_path")) {
        if (sscanf(value, "\"%255[^\"]\"", env_config->anchor_policy_path) != 1) {
            strncpy(env_config->anchor_policy_path, value, sizeof(env_config->anchor_policy_path) - 1);
            env_config->anchor_policy_path[sizeof(env_config->anchor_policy_path) - 1] = '\0';
        }
    } else if (MATCH("env", "async_resets")) {
        env_config->async_resets =
            (strcmp(value, "True") == 0 || strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
    } else if (MATCH("env", "fix_rewards")) {
        env_config->fix_rewards = (strcmp(value, "True") == 0 || strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
    } else if (MATCH("env", "fix_lambdas")) {
        env_config->fix_lambdas = (strcmp(value, "True") == 0 || strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
    } else if (MATCH("env", "lambda_value")) {
        env_config->lambda_value = atof(value);
    } else {
        return 0; // Unknown section/name, indicate failure to handle
    }

#undef MATCH
    return 1;
}

#endif // ENV_CONFIG_H
