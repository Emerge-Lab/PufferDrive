#ifndef ENV_CONFIG_H
#define ENV_CONFIG_H

#include <../../inih-r62/ini.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Config struct for parsing INI files - contains all environment configuration
typedef struct {
    int action_type;
    int dynamics_model;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_traffic_light_violation;
    float reward_goal;
    float reward_goal_post_respawn;
    float reward_ade;
    float reward_overspeed;
    float reward_comfort;
    float reward_velocity;
    float reward_lane_align;
    float reward_lane_center;
    float reward_timestep;
    float reward_reverse;
    float goal_radius;
    int collision_behavior;
    int offroad_behavior;
    int traffic_light_behavior;
    int end_sdc_path_behavior;
    int spawn_immunity_timer;
    float dt;
    int reach_goal_behavior;
    int target_type;
    int scenario_length;
    int termination_mode;
    int init_steps;
    int init_mode;
    int control_mode;
    int simulation_mode;
    char map_dir[256];
    float min_goal_spacing;
    float max_goal_spacing;
    int num_target_waypoints;
    int reward_conditioning;
    int reward_randomization;
    int max_agents_per_env;
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
        } else {
            printf("Warning: Unknown dynamics_model value '%s', defaulting to JERK\n", value);
            env_config->dynamics_model = 1; // Default to JERK
        }
    } else if (MATCH("env", "collision_behavior")) {
        env_config->collision_behavior = atoi(value);
    } else if (MATCH("env", "offroad_behavior")) {
        env_config->offroad_behavior = atoi(value);
    } else if (MATCH("env", "traffic_light_behavior")) {
        env_config->traffic_light_behavior = atoi(value);
    } else if (MATCH("env", "end_sdc_path_behavior")) {
        env_config->end_sdc_path_behavior = atoi(value);
    } else if (MATCH("env", "reach_goal_behavior")) {
        env_config->reach_goal_behavior = atoi(value);
    } else if (MATCH("env", "target_type")) {
        if (strcmp(value, "\"static\"") == 0 || strcmp(value, "static") == 0) {
            env_config->target_type = 0; // TARGET_STATIC
        } else if (strcmp(value, "\"dynamic\"") == 0 || strcmp(value, "dynamic") == 0) {
            env_config->target_type = 1; // TARGET_DYNAMIC
        } else {
            printf("Warning: Unknown target_type value '%s', defaulting to static\n", value);
            env_config->target_type = 0;
        }
    } else if (MATCH("env", "reward_vehicle_collision")) {
        env_config->reward_vehicle_collision = atof(value);
    } else if (MATCH("env", "reward_offroad_collision")) {
        env_config->reward_offroad_collision = atof(value);
    } else if (MATCH("env", "reward_traffic_light_violation")) {
        env_config->reward_traffic_light_violation = atof(value);
    } else if (MATCH("env", "reward_goal")) {
        env_config->reward_goal = atof(value);
    } else if (MATCH("env", "reward_goal_post_respawn")) {
        env_config->reward_goal_post_respawn = atof(value);
    } else if (MATCH("env", "reward_ade")) {
        env_config->reward_ade = atof(value);
    } else if (MATCH("env", "reward_overspeed")) {
        env_config->reward_overspeed = atof(value);
    } else if (MATCH("env", "reward_comfort")) {
        env_config->reward_comfort = atof(value);
    } else if (MATCH("env", "reward_velocity")) {
        env_config->reward_velocity = atof(value);
    } else if (MATCH("env", "reward_lane_align")) {
        env_config->reward_lane_align = atof(value);
    } else if (MATCH("env", "reward_lane_center")) {
        env_config->reward_lane_center = atof(value);
    } else if (MATCH("env", "reward_timestep")) {
        env_config->reward_timestep = atof(value);
    } else if (MATCH("env", "reward_reverse")) {
        env_config->reward_reverse = atof(value);
    } else if (MATCH("env", "goal_radius")) {
        env_config->goal_radius = atof(value);
    } else if (MATCH("env", "spawn_immunity_timer")) {
        env_config->spawn_immunity_timer = atoi(value);
    } else if (MATCH("env", "dt")) {
        env_config->dt = atof(value);
    } else if (MATCH("env", "scenario_length")) {
        env_config->scenario_length = atoi(value);
    } else if (MATCH("env", "termination_mode")) {
        env_config->termination_mode = atoi(value);
    } else if (MATCH("env", "init_steps")) {
        env_config->init_steps = atoi(value);
    } else if (MATCH("env", "max_agents_per_env")) {
        env_config->max_agents_per_env = atoi(value);
    } else if (MATCH("env", "init_mode")) {
        env_config->init_mode = atoi(value);
    } else if (MATCH("env", "control_mode")) {
        env_config->control_mode = atoi(value);
    } else if (MATCH("env", "simulation_mode")) {
        env_config->simulation_mode = atoi(value);
    } else if (MATCH("env", "map_dir")) {
        if (sscanf(value, "\"%255[^\"]\"", env_config->map_dir) != 1) {
            strncpy(env_config->map_dir, value, sizeof(env_config->map_dir) - 1);
            env_config->map_dir[sizeof(env_config->map_dir) - 1] = '\0';
        }
    } else if (MATCH("env", "min_goal_spacing")) {
        env_config->min_goal_spacing = atof(value);
    } else if (MATCH("env", "max_goal_spacing")) {
        env_config->max_goal_spacing = atof(value);
    } else if (MATCH("env", "num_target_waypoints")) {
        env_config->num_target_waypoints = atoi(value);
    } else if (MATCH("env", "reward_conditioning")) {
        if (strcmp(value, "True") == 0 || strcmp(value, "true") == 0 || strcmp(value, "1") == 0) {
            env_config->reward_conditioning = 1;
        } else {
            env_config->reward_conditioning = 0;
        }
    } else if (MATCH("env", "reward_randomization")) {
        if (strcmp(value, "True") == 0 || strcmp(value, "true") == 0 || strcmp(value, "1") == 0) {
            env_config->reward_randomization = 1;
        } else {
            env_config->reward_randomization = 0;
        }
    } else {
        return 0; // Unknown section/name, indicate failure to handle
    }

#undef MATCH
    return 1;
}

#endif // ENV_CONFIG_H
