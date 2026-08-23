#ifndef ENV_CONFIG_H
#define ENV_CONFIG_H

#include <../../inih-r62/ini.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Config struct for parsing INI files - contains all environment configuration
typedef struct {
    int action_type;
    int dynamics_model;
    float reward_collision;
    float reward_offroad;
    float reward_stop_line;
    float reward_goal;
    float reward_ade;
    float reward_overspeed;
    float reward_comfort;
    float reward_velocity;
    float reward_lane_align;
    float reward_vel_align;
    float reward_lane_center;
    float reward_center_bias;
    float reward_timestep;
    float reward_reverse;
    float goal_radius;
    float spawn_initial_speed;
    float vehicle_max_speed_mps;
    float goal_speed;
    int collision_behavior;
    int offroad_behavior;
    int traffic_light_behavior;
    int use_map_cache;
    int use_neighbor_cache;
    float dt;
    int goal_regen_mode;
    int goal_source;
    int obs_goal_lane_distance;
    int scenario_length;
    int termination_mode;
    int init_step;
    int init_mode;
    int control_mode;
    int simulation_mode;
    char map_dir[256];
    float min_goal_spacing;
    float max_goal_spacing;
    int num_goals;
    int reward_conditioning;
    int reward_randomization;
    int compute_eval_metrics;
    int max_agents_per_env;
    int obs_slots_lane_n;
    int obs_slots_boundary_n;
    int obs_lane_stride;
    int obs_boundary_stride;
    float obs_dropout_lane;
    float obs_dropout_boundary;
    int obs_slots_partners_n;
    int obs_slots_traffic_controls_n;
    int traffic_control_scope;
    float obs_norm_goal_offset_m;
    float obs_norm_xy_offset_m;
    float obs_norm_veh_length_m;
    float obs_norm_veh_width_m;
    float obs_norm_road_seg_length_m;
    float obs_norm_road_seg_width_m;
    float obs_range_traffic_control_m;
    float obs_range_partner_m;
    float obs_range_road_front_m;
    float obs_range_road_behind_m;
    float obs_range_road_side_m;
    float partner_blindness_prob;
    float partner_blindness_trigger_prob;
    float partner_blindness_duration_seconds;
    float phantom_braking_prob;
    float phantom_braking_trigger_prob;
    float phantom_braking_duration_seconds;
} env_init_config;

// Shared "ignore"/"stop"/"remove" enum for the collision/offroad/traffic-light
// behavior keys. Values mirror INFRACTION_BEHAVIOR_IGNORE/INFRACTION_BEHAVIOR_STOP/INFRACTION_BEHAVIOR_REMOVE in
// drive.h.
static int parse_infraction_behavior(const char *name, const char *value) {
    if (strcmp(value, "\"ignore\"") == 0 || strcmp(value, "ignore") == 0) {
        return 0; // INFRACTION_BEHAVIOR_IGNORE
    }
    if (strcmp(value, "\"stop\"") == 0 || strcmp(value, "stop") == 0) {
        return 1; // INFRACTION_BEHAVIOR_STOP
    }
    if (strcmp(value, "\"remove\"") == 0 || strcmp(value, "remove") == 0) {
        return 2; // INFRACTION_BEHAVIOR_REMOVE
    }
    fprintf(stderr, "Invalid %s value '%s': must be \"ignore\", \"stop\", or \"remove\"\n", name, value);
    exit(1);
}

// INI file parser handler - parses all environment configuration from drive.ini
static int handler(void *config, const char *section, const char *name, const char *value) {
    env_init_config *env_config = (env_init_config *) config;
#define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

    if (MATCH("env", "action_type")) {
        if (strcmp(value, "\"discrete\"") == 0 || strcmp(value, "discrete") == 0) {
            env_config->action_type = 0; // ACTION_TYPE_DISCRETE
        } else if (strcmp(value, "\"continuous\"") == 0 || strcmp(value, "continuous") == 0) {
            env_config->action_type = 1; // ACTION_TYPE_CONTINUOUS
        } else {
            fprintf(stderr, "Invalid action_type value '%s': must be \"discrete\" or \"continuous\"\n", value);
            exit(1);
        }
    } else if (MATCH("env", "dynamics_model")) {
        if (strcmp(value, "\"classic\"") == 0 || strcmp(value, "classic") == 0) {
            env_config->dynamics_model = 0; // DYNAMICS_MODEL_CLASSIC
        } else if (strcmp(value, "\"jerk\"") == 0 || strcmp(value, "jerk") == 0) {
            env_config->dynamics_model = 1; // DYNAMICS_MODEL_JERK
        } else {
            fprintf(stderr, "Invalid dynamics_model value '%s': must be \"classic\" or \"jerk\"\n", value);
            exit(1);
        }
    } else if (MATCH("env", "collision_behavior")) {
        env_config->collision_behavior = parse_infraction_behavior(name, value);
    } else if (MATCH("env", "offroad_behavior")) {
        env_config->offroad_behavior = parse_infraction_behavior(name, value);
    } else if (MATCH("env", "traffic_light_behavior")) {
        env_config->traffic_light_behavior = parse_infraction_behavior(name, value);
    } else if (MATCH("env", "use_map_cache")) {
        env_config->use_map_cache = atoi(value);
    } else if (MATCH("env", "use_neighbor_cache")) {
        env_config->use_neighbor_cache = atoi(value);
    } else if (MATCH("env", "goal_regen_mode")) {
        if (strcmp(value, "\"finite\"") == 0 || strcmp(value, "finite") == 0) {
            env_config->goal_regen_mode = 0; // GOAL_REGEN_FINITE
        } else if (strcmp(value, "\"rolling\"") == 0 || strcmp(value, "rolling") == 0) {
            env_config->goal_regen_mode = 1; // GOAL_REGEN_ROLLING
        } else {
            fprintf(stderr, "Invalid goal_regen_mode value '%s': must be \"finite\" or \"rolling\"\n", value);
            exit(1);
        }
    } else if (MATCH("env", "goal_source")) {
        if (strcmp(value, "\"route\"") == 0 || strcmp(value, "route") == 0) {
            env_config->goal_source = 0; // GOAL_SOURCE_ROUTE
        } else if (strcmp(value, "\"map\"") == 0 || strcmp(value, "map") == 0) {
            env_config->goal_source = 1; // GOAL_SOURCE_MAP
        } else if (strcmp(value, "\"gt\"") == 0 || strcmp(value, "gt") == 0) {
            env_config->goal_source = 2; // GOAL_SOURCE_GT
        } else {
            fprintf(stderr, "Invalid goal_source value '%s': must be \"route\", \"map\", or \"gt\"\n", value);
            exit(1);
        }
    } else if (MATCH("env", "obs_goal_lane_distance")) {
        if (strcmp(value, "True") == 0 || strcmp(value, "true") == 0 || strcmp(value, "1") == 0) {
            env_config->obs_goal_lane_distance = 1;
        } else {
            env_config->obs_goal_lane_distance = 0;
        }
    } else if (MATCH("env", "reward_collision")) {
        env_config->reward_collision = atof(value);
    } else if (MATCH("env", "reward_offroad")) {
        env_config->reward_offroad = atof(value);
    } else if (MATCH("env", "reward_stop_line")) {
        env_config->reward_stop_line = atof(value);
    } else if (MATCH("env", "reward_goal")) {
        env_config->reward_goal = atof(value);
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
    } else if (MATCH("env", "reward_vel_align")) {
        env_config->reward_vel_align = atof(value);
    } else if (MATCH("env", "reward_lane_center")) {
        env_config->reward_lane_center = atof(value);
    } else if (MATCH("env", "reward_center_bias")) {
        env_config->reward_center_bias = atof(value);
    } else if (MATCH("env", "reward_timestep")) {
        env_config->reward_timestep = atof(value);
    } else if (MATCH("env", "reward_reverse")) {
        env_config->reward_reverse = atof(value);
    } else if (MATCH("env", "goal_radius")) {
        env_config->goal_radius = atof(value);
    } else if (MATCH("env", "spawn_initial_speed")) {
        env_config->spawn_initial_speed = atof(value);
    } else if (MATCH("env", "vehicle_max_speed_mps")) {
        env_config->vehicle_max_speed_mps = atof(value);
    } else if (MATCH("env", "goal_speed")) {
        env_config->goal_speed = atof(value);
    } else if (MATCH("env", "dt")) {
        env_config->dt = atof(value);
    } else if (MATCH("env", "scenario_length")) {
        env_config->scenario_length = atoi(value);
    } else if (MATCH("env", "termination_mode")) {
        env_config->termination_mode = atoi(value);
    } else if (MATCH("env", "init_step")) {
        env_config->init_step = atoi(value);
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
    } else if (MATCH("env", "num_goals")) {
        env_config->num_goals = atoi(value);
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
    } else if (MATCH("env", "compute_eval_metrics")) {
        if (strcmp(value, "True") == 0 || strcmp(value, "true") == 0 || strcmp(value, "1") == 0) {
            env_config->compute_eval_metrics = 1;
        } else {
            env_config->compute_eval_metrics = 0;
        }
    } else if (MATCH("env", "obs_slots_boundary_n")) {
        env_config->obs_slots_boundary_n = atoi(value);
    } else if (MATCH("env", "obs_slots_lane_n")) {
        env_config->obs_slots_lane_n = atoi(value);
    } else if (MATCH("env", "obs_lane_stride")) {
        env_config->obs_lane_stride = atoi(value);
    } else if (MATCH("env", "obs_boundary_stride")) {
        env_config->obs_boundary_stride = atoi(value);
    } else if (MATCH("env", "obs_dropout_lane")) {
        env_config->obs_dropout_lane = atof(value);
    } else if (MATCH("env", "obs_dropout_boundary")) {
        env_config->obs_dropout_boundary = atof(value);
    } else if (MATCH("env", "obs_slots_partners_n")) {
        env_config->obs_slots_partners_n = atoi(value);
    } else if (MATCH("env", "obs_slots_traffic_controls_n")) {
        env_config->obs_slots_traffic_controls_n = atoi(value);
    } else if (MATCH("env", "traffic_control_scope")) {
        env_config->traffic_control_scope = atoi(value);
    } else if (MATCH("env", "obs_norm_goal_offset_m")) {
        env_config->obs_norm_goal_offset_m = atof(value);
    } else if (MATCH("env", "obs_norm_xy_offset_m")) {
        env_config->obs_norm_xy_offset_m = atof(value);
    } else if (MATCH("env", "obs_norm_veh_length_m")) {
        env_config->obs_norm_veh_length_m = atof(value);
    } else if (MATCH("env", "obs_norm_veh_width_m")) {
        env_config->obs_norm_veh_width_m = atof(value);
    } else if (MATCH("env", "obs_norm_road_seg_length_m")) {
        env_config->obs_norm_road_seg_length_m = atof(value);
    } else if (MATCH("env", "obs_norm_road_seg_width_m")) {
        env_config->obs_norm_road_seg_width_m = atof(value);
    } else if (MATCH("env", "obs_range_traffic_control_m")) {
        env_config->obs_range_traffic_control_m = atof(value);
    } else if (MATCH("env", "obs_range_partner_m")) {
        env_config->obs_range_partner_m = atof(value);
    } else if (MATCH("env", "obs_range_road_front_m")) {
        env_config->obs_range_road_front_m = atof(value);
    } else if (MATCH("env", "obs_range_road_behind_m")) {
        env_config->obs_range_road_behind_m = atof(value);
    } else if (MATCH("env", "obs_range_road_side_m")) {
        env_config->obs_range_road_side_m = atof(value);
    } else if (MATCH("env", "partner_blindness_prob")) {
        env_config->partner_blindness_prob = atof(value);
    } else if (MATCH("env", "partner_blindness_trigger_prob")) {
        env_config->partner_blindness_trigger_prob = atof(value);
    } else if (MATCH("env", "partner_blindness_duration_seconds")) {
        env_config->partner_blindness_duration_seconds = atof(value);
    } else if (MATCH("env", "phantom_braking_prob")) {
        env_config->phantom_braking_prob = atof(value);
    } else if (MATCH("env", "phantom_braking_trigger_prob")) {
        env_config->phantom_braking_trigger_prob = atof(value);
    } else if (MATCH("env", "phantom_braking_duration_seconds")) {
        env_config->phantom_braking_duration_seconds = atof(value);
    } else {
        return 0; // Unknown section/name, indicate failure to handle
    }

#undef MATCH
    return 1;
}

static int load_env_config(const char *ini_file, env_init_config *config) {
    return ini_parse(ini_file, handler, config);
}

#endif // ENV_CONFIG_H
