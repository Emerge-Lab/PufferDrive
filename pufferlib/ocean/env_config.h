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
    float reward_lane_center;
    float reward_lane_align;
    float reward_goal;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float goal_radius;
    float goal_speed;
    int collision_behavior;
    int offroad_behavior;
    int spawn_immunity_timer;
    float dt;
    int goal_behavior;
    int reward_randomization;
    int reward_conditioning;
    float goal_target_distance;

    float reward_bound_collision_min;
    float reward_bound_goal_radius_min;
    float reward_bound_goal_radius_max;
    float reward_bound_collision_max;
    float reward_bound_offroad_min;
    float reward_bound_offroad_max;
    float reward_bound_comfort_min;
    float reward_bound_comfort_max;
    float reward_bound_lane_align_min;
    float reward_bound_lane_align_max;
    float reward_bound_lane_center_min;
    float reward_bound_lane_center_max;
    float reward_bound_velocity_min;
    float reward_bound_velocity_max;
    float reward_bound_traffic_light_min;
    float reward_bound_traffic_light_max;
    float reward_bound_center_bias_min;
    float reward_bound_center_bias_max;
    float reward_bound_vel_align_min;
    float reward_bound_vel_align_max;
    float reward_bound_overspeed_min;
    float reward_bound_overspeed_max;
    float reward_bound_timestep_min;
    float reward_bound_timestep_max;
    float reward_bound_reverse_min;
    float reward_bound_reverse_max;
    float reward_bound_throttle_min;
    float reward_bound_throttle_max;
    float reward_bound_steer_min;
    float reward_bound_steer_max;
    float reward_bound_acc_min;
    float reward_bound_acc_max;

    int episode_length;
    int termination_mode;
    int init_steps;
    int init_mode;
    int control_mode;
    int num_maps;
    char map_dir[256];
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
    } else if (MATCH("env", "goal_behavior")) {
        env_config->goal_behavior = atoi(value);
    } else if (MATCH("env", "reward_randomization")) {
        env_config->reward_randomization = atoi(value);
    } else if (MATCH("env", "reward_conditioning")) {
        env_config->reward_conditioning = atoi(value);
    } else if (MATCH("env", "goal_target_distance")) {
        env_config->goal_target_distance = atof(value);
    } else if (MATCH("env", "reward_lane_center")) {
        env_config->reward_lane_center = atof(value);
    } else if (MATCH("env", "reward_lane_align")) {
        env_config->reward_lane_align = atof(value);
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
    } else if (MATCH("env", "goal_radius")) {
        env_config->goal_radius = atof(value);
    } else if (MATCH("env", "reward_bound_collision_min")) {
        env_config->reward_bound_collision_min = atof(value);
    } else if (MATCH("env", "reward_bound_goal_radius_min")) {
        env_config->reward_bound_goal_radius_min = atof(value);
    } else if (MATCH("env", "reward_bound_goal_radius_max")) {
        env_config->reward_bound_goal_radius_max = atof(value);
    } else if (MATCH("env", "reward_bound_collision_max")) {
        env_config->reward_bound_collision_max = atof(value);
    } else if (MATCH("env", "reward_bound_offroad_min")) {
        env_config->reward_bound_offroad_min = atof(value);
    } else if (MATCH("env", "reward_bound_offroad_max")) {
        env_config->reward_bound_offroad_max = atof(value);
    } else if (MATCH("env", "reward_bound_comfort_min")) {
        env_config->reward_bound_comfort_min = atof(value);
    } else if (MATCH("env", "reward_bound_comfort_max")) {
        env_config->reward_bound_comfort_max = atof(value);
    } else if (MATCH("env", "reward_bound_lane_align_min")) {
        env_config->reward_bound_lane_align_min = atof(value);
    } else if (MATCH("env", "reward_bound_lane_align_max")) {
        env_config->reward_bound_lane_align_max = atof(value);
    } else if (MATCH("env", "reward_bound_lane_center_min")) {
        env_config->reward_bound_lane_center_min = atof(value);
    } else if (MATCH("env", "reward_bound_lane_center_max")) {
        env_config->reward_bound_lane_center_max = atof(value);
    } else if (MATCH("env", "reward_bound_velocity_min")) {
        env_config->reward_bound_velocity_min = atof(value);
    } else if (MATCH("env", "reward_bound_velocity_max")) {
        env_config->reward_bound_velocity_max = atof(value);
    } else if (MATCH("env", "reward_bound_traffic_light_min")) {
        env_config->reward_bound_traffic_light_min = atof(value);
    } else if (MATCH("env", "reward_bound_traffic_light_max")) {
        env_config->reward_bound_traffic_light_max = atof(value);
    } else if (MATCH("env", "reward_bound_center_bias_min")) {
        env_config->reward_bound_center_bias_min = atof(value);
    } else if (MATCH("env", "reward_bound_center_bias_max")) {
        env_config->reward_bound_center_bias_max = atof(value);
    } else if (MATCH("env", "reward_bound_vel_align_min")) {
        env_config->reward_bound_vel_align_min = atof(value);
    } else if (MATCH("env", "reward_bound_vel_align_max")) {
        env_config->reward_bound_vel_align_max = atof(value);
    } else if (MATCH("env", "reward_bound_overspeed_min")) {
        env_config->reward_bound_overspeed_min = atof(value);
    } else if (MATCH("env", "reward_bound_overspeed_max")) {
        env_config->reward_bound_overspeed_max = atof(value);
    } else if (MATCH("env", "reward_bound_timestep_min")) {
        env_config->reward_bound_timestep_min = atof(value);
    } else if (MATCH("env", "reward_bound_timestep_max")) {
        env_config->reward_bound_timestep_max = atof(value);
    } else if (MATCH("env", "reward_bound_reverse_min")) {
        env_config->reward_bound_reverse_min = atof(value);
    } else if (MATCH("env", "reward_bound_reverse_max")) {
        env_config->reward_bound_reverse_max = atof(value);
    } else if (MATCH("env", "reward_bound_throttle_min")) {
        env_config->reward_bound_throttle_min = atof(value);
    } else if (MATCH("env", "reward_bound_throttle_max")) {
        env_config->reward_bound_throttle_max = atof(value);
    } else if (MATCH("env", "reward_bound_steer_min")) {
        env_config->reward_bound_steer_min = atof(value);
    } else if (MATCH("env", "reward_bound_steer_max")) {
        env_config->reward_bound_steer_max = atof(value);
    } else if (MATCH("env", "reward_bound_acc_min")) {
        env_config->reward_bound_acc_min = atof(value);
    } else if (MATCH("env", "reward_bound_acc_max")) {
        env_config->reward_bound_acc_max = atof(value);
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
        env_config->init_mode = atoi(value);
    } else if (MATCH("env", "control_mode")) {
        env_config->control_mode = atoi(value);
    } else if (MATCH("env", "map_dir")) {
        if (sscanf(value, "\"%255[^\"]\"", env_config->map_dir) != 1) {
            strncpy(env_config->map_dir, value, sizeof(env_config->map_dir) - 1);
            env_config->map_dir[sizeof(env_config->map_dir) - 1] = '\0';
        }
        // printf("Parsed map_dir: '%s'\n", env_config->map_dir);
    } else if (MATCH("env", "num_maps")) {
        env_config->num_maps = atoi(value);
    } else {
        return 0; // Unknown section/name, indicate failure to handle
    }

#undef MATCH
    return 1;
}

#endif // ENV_CONFIG_H
