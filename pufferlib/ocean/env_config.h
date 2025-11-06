#ifndef ENV_CONFIG_H
#define ENV_CONFIG_H

#include <../../inih-r62/ini.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Config struct for parsing INI files - contains all environment configuration
typedef struct
{
    int action_type;
    int dynamics_model;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_goal;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float reward_ade;
    float goal_radius;
    int collision_behaviour;
    int offroad_behaviour;
    int spawn_immunity_timer;
    float dt;
    int use_goal_generation;
    int control_non_vehicles;
    int scenario_length;
    int init_steps;
    int control_all_agents;
    int num_policy_controlled_agents;
    int deterministic_agent_selection;
    // Conditioning weight bounds
    float reward_collision_weight_lb;
    float reward_collision_weight_ub;
    float reward_offroad_weight_lb;
    float reward_offroad_weight_ub;
    float reward_goal_weight_lb;
    float reward_goal_weight_ub;
    float entropy_weight_lb;
    float entropy_weight_ub;
    float discount_weight_lb;
    float discount_weight_ub;
} env_init_config;

// INI file parser handler - parses all environment configuration from drive.ini
static int handler(
    void* config,
    const char* section,
    const char* name,
    const char* value
) {
    env_init_config* env_config = (env_init_config*)config;
    #define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

    if (MATCH("env", "action_type")) {
        if (strcmp(value, "\"discrete\"") == 0 ||strcmp(value, "discrete") == 0) {
            env_config->action_type = 0;  // DISCRETE
        } else if (strcmp(value, "\"continuous\"") == 0 || strcmp(value, "continuous") == 0) {
            env_config->action_type = 1;  // CONTINUOUS
        } else {
            printf("Warning: Unknown action_type value '%s', defaulting to DISCRETE\n", value);
            env_config->action_type = 0;  // Default to DISCRETE
        }
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
    } else if(MATCH("env", "collision_behaviour")){
        env_config->collision_behaviour = atoi(value);
    } else if(MATCH("env", "offroad_behaviour")){
        env_config->offroad_behaviour = atoi(value);
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

#endif // ENV_CONFIG_H
