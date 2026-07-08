#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>
#include "error.h"
#include "datatypes.h"

#define INVALID_POSITION -10000.0f

// Initialization modes
#define INIT_ALL_VALID 0
#define INIT_ONLY_CONTROLLABLE_AGENTS 1

// Control modes
#define CONTROL_VEHICLES 0
#define CONTROL_AGENTS 1
#define CONTROL_WOSAC 2
#define CONTROL_SDC_ONLY 3

// Controller modes
#define CONTROLLER_STATIC 0
#define CONTROLLER_POLICY 1
#define CONTROLLER_REPLAY 2
#define CONTROLLER_IDM 3
#define CONTROLLER_CORRIDOR_IDM 4
#define CONTROLLER_PDM 5
#define CONTROLLER_NUPLAN_IDM 6

// Simulation modes
#define SIMULATION_GIGAFLOW 0
#define SIMULATION_REPLAY 1

// Lane selection scoring
#define LANE_SELECTION_DISTANCE_WEIGHT 0.7f
#define LANE_SELECTION_HEADING_WEIGHT 0.3f
#define LANE_DISTANCE_NORMALIZATION 4.0f
#define LANE_SWITCH_THRESHOLD 0.05f // Hysteresis: new lane must be 5% better to switch
#define LANE_ALIGN_COS_THRESHOLD 0.5f

// Collision and distance thresholds
#define MAX_CHECKED_LANES 32
#define COLLISION_QUICK_CHECK_DIST 15.0f  // Quick distance check before OBB SAT
#define INIT_COLLISION_SHRINK_FACTOR 0.7f // Shrink agent dims at init to prevent collisions
#define AGENT_STOPPED_SPEED_THRESHOLD 0.2f
#define TRAFFIC_LIGHT_DISTANCE_THRESHOLD 10.0f
#define STOP_LINE_EXTENSION_FACTOR 1.5f
#define RED_LIGHT_HEADING_THRESHOLD (M_PI / 4.0f)
#define NUPLAN_BEHIND_COS_THRESHOLD -0.8660254f // cos(150 degrees)

// TTC default value when no vehicle ahead
#define DEFAULT_TTC 5.0f
// TTC violation threshold for "within bound" rate
#define TTC_VIOLATION_THRESHOLD 0.95f
// Multi-lane detection thresholds
#define LANE_WIDTH 3.7f
#define LANE_MARGIN 0.2f
#define MULTI_LANE_THRESHOLD (LANE_WIDTH / 2.0f + LANE_MARGIN) // 2.05m
#define MULTI_LANE_FULL_SCORE_TIME 3.4f                        // seconds
#define MULTI_LANE_HALF_SCORE_TIME 5.7f                        // seconds
#define COMFORT_ACCEL_THRESHOLD 3.0f                           // m/s^2
#define COMFORT_JERK_THRESHOLD 5.0f                            // m/s^3

// Avoidability-by-braking: counterfactual "could the target have braked to avoid this
// collision" analysis, evaluated over the per-agent rolling trajectory history
// (brake_hist_*, see Agent in datatypes.h; TARGET_BRAKE_HISTORY_LEN there).
#define TARGET_AVOIDABILITY_BRAKE_DECEL 5.0f // m/s^2 braking magnitude (matches planner max braking)
// Max poses in a full braking rollout to a stop: ceil(MAX_SPEED / DECEL / dt=0.1) + 1 = 81
// (index 0 is the current pose). Bounds the fixed-size adversary extension array.
#define TARGET_AVOIDABILITY_MAX_EXT_STEPS 81
#define TARGET_AVOIDABILITY_NOT_AVOIDABLE (-1.0f) // no braking within history avoids the crash
#define TARGET_AVOIDABILITY_UNSET (-2.0f)         // per-episode "not yet computed" sentinel

// nuPlan-style collision classification
#define COLLISION_TYPE_NONE 0
#define COLLISION_TYPE_STOPPED_EGO 1
#define COLLISION_TYPE_STOPPED_TRACK 2
#define COLLISION_TYPE_ACTIVE_REAR 3
#define COLLISION_TYPE_ACTIVE_FRONT 4
#define COLLISION_TYPE_ACTIVE_LATERAL 5
// Collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2
#define TRAFFIC_LIGHT_VIOLATION 3

// Collision/Infraction behaviors
#define STOP_AGENT 1
#define REMOVE_AGENT 2

#define MAX_SPEED 40.0f

// Grid cell size
#define GRID_CELL_SIZE 5.0f
// Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution)
// => For each entity type in gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2
#define MAX_ENTITIES_PER_CELL 30
// GridMapEntity types
#define ENTITY_TYPE_ROAD_ELEMENT 1
#define ENTITY_TYPE_TRAFFIC_CONTROL 2

// Traffic control scope constants
#define TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS 0
#define TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS_STOP_SIGN 1
#define TRAFFIC_CONTROL_SCOPE_ALL 2

// TARGET_TYPE modes (controls what target info is in observations)
#define TARGET_STATIC 0
#define TARGET_DYNAMIC 1

// Observation feature counts
#define EGO_FEATURES_CLASSIC 7
#define EGO_FEATURES_JERK 9
#define ROAD_FEATURES 7
#define PARTNER_FEATURES 10
#define TRAFFIC_CONTROL_FEATURES 7
#define STATIC_TARGET_FEATURES 3
#define DYNAMIC_TARGET_FEATURES 5
#define OBS_VALID_COUNT_FEATURES 4

// GIGAFLOW specific
#define MAX_ROUTE_LENGTH 64
// Traffic light generation
#define TL_DEFAULT_RED_DURATION 2.0f
#define TL_DEFAULT_YELLOW_DURATION 3.0f
#define TL_DEFAULT_GREEN_DURATION 10.0f
#define TL_EPISODE_DISABLE_PROB 0.20f
#define TL_INDIVIDUAL_REMOVE_PROB 0.20f
#define TL_ALWAYS_GREEN_PROB 0.05f

// Dynamics Models
#define CLASSIC 0
#define JERK 1

static const float ACCEL_LONG_LIMIT[2] = {-5.0f, 2.5f};
static const float ACCEL_LAT_LIMIT[2] = {-4.0f, 4.0f};
#define STEERING_LIMIT 0.667f

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space (for CLASSIC dynamics model)
static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
static const float STEERING_VALUES[9] = {-0.667f, -0.500f, -0.333f, -0.167f, 0.000f, 0.167f, 0.333f, 0.500f, 0.667f};

static const float offsets[4][2] = {
    {-1, 1}, // top-left
    {1, 1},  // top-right
    {1, -1}, // bottom-right
    {-1, -1} // bottom-left
};

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, // Top row
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, // Second row
    {-2, 0},  {-1, 0},  {0, 0},  {1, 0},  {2, 0},  // Middle row (including center)
    {-2, 1},  {-1, 1},  {0, 1},  {1, 1},  {2, 1},  // Fourth row
    {-2, 2},  {-1, 2},  {0, 2},  {1, 2},  {2, 2}   // Bottom row
};

#define Z_COMPUTATION_OFFSET_COUNT 9
#define Z_BUFFER 4.0f
#define Z_NUM_PT_AVG 30

static const int z_computation_offsets[Z_COMPUTATION_OFFSET_COUNT][2] = {
    {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {0, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1},
};

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;
typedef struct Agent Agent;
typedef struct RoadMapElement RoadMapElement;
typedef struct TrafficControlElement TrafficControlElement;

#define COMPLETED_EPISODE_QUEUE_CAPACITY 256
#define COMPLETED_EPISODE_MAP_NAME_SIZE 512

struct Log {
    float n;
    float target_n;
    float episode_return;
    float episode_return_collision;
    float episode_return_offroad;
    float episode_return_drive;
    float episode_return_adversarial;
    float target_episode_return;
    float target_episode_return_collision;
    float target_episode_return_offroad;
    float target_episode_return_drive;
    float episode_length;
    float target_episode_length;
    float expert_static_car_count;
    float static_car_count;
    float score;
    float offroad_rate;
    float collision_rate;
    float adversaries_offroad_rate;
    float adversaries_collision_rate;
    float adversaries_target_collision_rate;
    float adversaries_adversary_collision_rate;
    float adversaries_red_light_violation_rate;
    float adversaries_at_fault_collision_rate;
    float adversaries_making_progress_rate;
    float adversaries_comfort_score;
    float adversaries_comfort_violation_rate;
    float adversaries_comfort_longitudinal_accel_violations_per_timestep;
    float adversaries_comfort_lateral_accel_violations_per_timestep;
    float adversaries_comfort_jerk_violations_per_timestep;
    float adversaries_uncomfortable_timestep_rate;
    float adversaries_avg_speed;
    float adversaries_lane_center_rate;
    float adversaries_lane_heading_aligned_rate;
    float adversaries_velocity_progress;
    float adversaries_speed_limit_compliance;
    float adversaries_driving_direction_score;
    float adversaries_multi_lane_time;
    float adversaries_multi_lane_score;
    float adversaries_dnf_rate;
    float adversaries_score;
    float adversaries_num_waypoints_reached;
    float did_target_collide;
    float did_target_offroad;
    float did_target_run_light;
    float did_target_fail;
    float did_target_make_progress;
    float did_target_have_at_fault_collision;
    float target_num_goals_reached;
    float target_ttc_within_bound_rate;
    float target_progress_ratio;
    float target_puffer_score;
    float target_comfort_violations_per_timestep;
    float target_comfort_longitudinal_accel_violations_per_timestep;
    float target_comfort_lateral_accel_violations_per_timestep;
    float target_comfort_jerk_violations_per_timestep;
    float target_uncomfortable_timestep_rate;
    float target_collision_count;
    float target_collision_severity;
    float target_collision_responsibility;
    float target_collision_impact_zone;
    float target_collision_type;
    float target_collision_target_speed;
    float target_collision_other_speed;
    float target_collision_relative_speed;
    float target_collision_other_active;
    float target_collision_other_stopped;
    float target_collision_other_removed;
    float target_avoidability_by_braking;
    float adversaries_collision_count;
    float adversaries_collision_severity;
    float adversaries_collision_responsibility;
    float adversaries_collision_impact_zone;
    float target_hit_count;
    float target_hit_responsibility;
    float target_hit_low_responsibility_rate;
    float target_hit_at_fault_count;
    float target_hit_at_fault_rate;
    float red_light_violation_rate;
    float num_waypoints_reached;
    float num_goals_reached;
    float comfort_violation_count;
    float comfort_longitudinal_accel_violation_count;
    float comfort_lateral_accel_violation_count;
    float comfort_jerk_violation_count;
    float uncomfortable_timestep_count;
    float velocity_progress_sum;
    float lane_center_rate;
    float lane_heading_aligned_rate;
    float dnf_rate;
    float avg_displacement_error;
    float avg_speed_per_agent;
    // Puffer score components
    float at_fault_collision_rate;
    float ttc_within_bound_rate;
    float driving_direction_score;
    float speed_limit_compliance;
    float making_progress_rate;
    float progress_ratio;
    float comfort_score;
    float puffer_score;
    // Puffer score intermediate accumulators (for aggregation)
    float wrong_way_distance;
    float speed_violation_sum;
    float ttc_violations;
    float ttc_samples;
    float multi_lane_time;
    float multi_lane_score;
    float total_distance_travelled;
    float total_infractions;
};

typedef struct CompletedEpisodeSummary CompletedEpisodeSummary;
struct CompletedEpisodeSummary {
    Log log;
    float n;
    float target_n;
    int active_agent_count;
    int timestep;
    char map_name[COMPLETED_EPISODE_MAP_NAME_SIZE];
    char scenario_id[128];
    char dataset_name[32];
};

typedef struct GridMapEntity GridMapEntity;
struct GridMapEntity {
    int entity_type;  // Entity type: 1=Agent, 2=RoadMapElement, 3=TrafficControlElement
    int entity_idx;   // Index into the corresponding typed array
    int geometry_idx; // Index into entity's trajectory/geometry array
};

typedef struct GridMap GridMap;
struct GridMap {
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int grid_cols;
    int grid_rows;
    int cell_size_x;
    int cell_size_y;
    int *cell_entities_count; // number of entities in each cell of the GridMap
    GridMapEntity **cells;    // list of gridEntities in each cell of the GridMap
    // Extras/Optimizations
    int vision_range;
    int *neighbor_cache_count;               // number of entities in each cells neighbor cache
    GridMapEntity **neighbor_cache_entities; // preallocated array to hold neighbor entities
    int *grid_index_drivable;
    int num_drivable_grid_cell;
};

typedef struct DriveMap DriveMap;
struct DriveMap {
    char *map_name;
    RoadMapElement *road_elements;
    TrafficControlElement *traffic_elements;
    GridMap *grid_map;
    struct LaneGraph lane_graph;
    int num_road_elements;
    int num_traffic_elements;
    int num_objects;
};

static void free_drive_map(DriveMap *map);

typedef struct DriveMapCache DriveMapCache;
struct DriveMapCache {
    DriveMap **maps;
    int count;
    int capacity;
    int cache_hits;
    int cache_misses;
};

static DriveMapCache *drive_map_cache_create(void) {
    DriveMapCache *cache = (DriveMapCache *)calloc(1, sizeof(DriveMapCache));
    return cache;
}

static void drive_map_cache_close(DriveMapCache *cache) {
    if (cache == NULL) {
        return;
    }

    for (int i = 0; i < cache->count; i++) {
        free_drive_map(cache->maps[i]);
    }
    free(cache->maps);
    free(cache);
}

typedef struct {
    float collision;
    float offroad;
    float red_light;
    float drive;
    float adversarial;
} RewardTerms;

struct Drive {
    Client *client;
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;
    unsigned char *masks;
    Log log;
    Log *logs;
    int num_controllable_agents; // Max number of controllable agents
    int active_agent_count;      // Current number of controllable agents
    int *active_agent_indices;
    int num_total_agents; // Total agents in a log scenario
    int num_max_agents;   // Max agents allocated in the environment
    int num_agents;       // Current number of agents in the environment
    int action_type;
    int human_agent_idx;
    Agent *agents;
    RoadMapElement *road_elements;
    TrafficControlElement *traffic_elements;
    DriveMap *shared_map;
    DriveMapCache *map_cache;
    int owns_map_data;
    int owns_traffic_data;
    int num_road_elements;
    int num_traffic_elements;
    int num_objects;
    struct LaneGraph lane_graph;
    int static_agent_count;
    int *static_agent_indices;
    int expert_static_agent_count;
    int *expert_static_agent_indices;
    int timestep;
    int init_steps;
    int dynamics_model;
    GridMap *grid_map;
    int *neighbor_offsets;
    int scenario_length;
    float reward_goal;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_comfort;
    float reward_lane_align;
    float reward_vel_align;
    float reward_lane_center;
    float reward_center_bias;
    float reward_velocity;
    float reward_reverse;
    float reward_stop_line;
    float reward_timestep;
    float reward_overspeed;
    float reward_ade;
    float adv_reward_weight_collision;
    float adv_reward_weight_offroad;
    float adv_reward_weight_traffic_light;
    float adv_reward_weight_drive;
    float adv_target_offroad_reward;
    float adv_target_collision_reward;
    float adv_target_hit_at_fault_bonus;
    float adv_target_hit_low_responsibility_threshold;
    float adv_target_hit_low_responsibility_penalty;
    int adv_target_hit_low_responsibility_behavior;
    int target_hit_this_step;
    int target_hit_hitter_idx_this_step;
    int target_hit_at_fault_this_step;
    float target_hit_responsibility_this_step;
    float target_hit_count_episode;
    float target_hit_responsibility_episode;
    float target_hit_low_responsibility_count_episode;
    float target_hit_at_fault_count_episode;
    int target_collision_type_episode;
    float target_collision_target_speed_episode;
    float target_collision_other_speed_episode;
    float target_collision_relative_speed_episode;
    float target_collision_other_active_episode;
    float target_collision_other_stopped_episode;
    float target_collision_other_removed_episode;
    float target_avoidability_by_braking_episode;
    char *map_name;
    float world_mean_x;
    float world_mean_y;
    float dt;
    float pdm_horizon;
    float pdm_planning_dt;
    float spawn_initial_speed;
    int targeted_spawn_mode;
    float targeted_spawn_radius;
    int targeted_spawn_attempts;
    float goal_radius;
    float goal_speed;
    float min_waypoint_spacing;
    float max_waypoint_spacing;
    int num_target_waypoints;
    int logs_capacity;
    int target_type;
    char *ini_file;
    int collision_behavior; // 0 = none, 1=stop, 2 = remove
    int ignore_target_collision_behavior;
    int remove_target_on_collision_or_offroad;
    int offroad_behavior;       // 0 = none, 1=stop, 2 = remove
    int traffic_light_behavior; // 0 = none, 1=stop, 2 = remove
    int deterministic_traffic_lights;
    // Metadata fields
    char scenario_id[128];
    char dataset_name[32];
    int log_length;
    float log_dt;
    int num_objects_of_interest;
    int *objects_of_interest;
    int num_tracks_to_predict;
    int *tracks_to_predict;
    int init_mode;
    int control_mode;
    int sdc_controller;
    int non_sdc_controller;
    int simulation_mode;
    int termination_mode;
    float inactive_agent_threshold;
    int adversarial_termination_mode;
    int terminate_ep_on_target_failure;
    int reward_conditioning;
    int reward_randomization;
    int compute_eval_metrics;
    int max_boundary_segment_observations;
    int max_lane_segment_observations;
    int max_partner_observations;
    int target_max_partner_observations;
    int adversary_max_partner_observations;
    int max_traffic_control_observations;
    int traffic_control_scope;
    int obs_lane_segment_count;
    int obs_boundary_segment_count;
    int road_dropout_enabled;
    // Robustness features
    float partner_blindness_prob;
    float phantom_braking_prob;
    float phantom_braking_trigger_prob;
    int phantom_braking_duration;
    int eval_mode;

    // Observation normalization constants
    float max_goal_position;
    float max_position;
    float max_veh_len;
    float max_veh_width;
    float max_road_segment_length;
    float max_road_segment_width;
    float max_traffic_control_distance;
    float agent_obs_max_dist;
    float road_obs_front_dist;
    float road_obs_behind_dist;
    float road_obs_side_dist;
    CompletedEpisodeSummary completed_episode_queue[COMPLETED_EPISODE_QUEUE_CAPACITY];
    int completed_episode_queue_head;
    int completed_episode_queue_tail;
    int completed_episode_queue_count;
};

// ========================================
// Utility Functions
// ========================================

typedef struct {
    int index;
    float dist_sq;
    float dx; // Store dx/dy to avoid re-calculating
    float dy;
    float dz;
} AgentDistance;

static inline float total_reward_terms(const RewardTerms *terms) {
    return terms->collision + terms->offroad + terms->red_light + terms->drive + terms->adversarial;
}

static inline void apply_reward_terms(Drive *env, int i, const RewardTerms *terms) {
    float total_reward = total_reward_terms(terms);
    env->rewards[i] += total_reward;
    env->logs[i].episode_return += total_reward;
    env->logs[i].episode_return_collision += terms->collision;
    env->logs[i].episode_return_offroad += terms->offroad;
    env->logs[i].episode_return_drive += terms->drive;
    env->logs[i].episode_return_adversarial += terms->adversarial;
}

static inline void accumulate_log_totals(Log *dst, const Log *src) {
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < num_keys; i++) {
        ((float *)dst)[i] += ((const float *)src)[i];
    }
}

static inline void normalize_log_for_output(Log *log, float n, float target_n) {
    if (n <= 0.0f) {
        return;
    }

    float episode_length = log->episode_length;
    float target_episode_length = log->target_episode_length;
    float episode_return = log->episode_return;
    float episode_return_collision = log->episode_return_collision;
    float episode_return_offroad = log->episode_return_offroad;
    float episode_return_drive = log->episode_return_drive;
    float episode_return_adversarial = log->episode_return_adversarial;
    float target_episode_return = log->target_episode_return;
    float target_episode_return_collision = log->target_episode_return_collision;
    float target_episode_return_offroad = log->target_episode_return_offroad;
    float target_episode_return_drive = log->target_episode_return_drive;
    float did_target_collide = log->did_target_collide;
    float did_target_offroad = log->did_target_offroad;
    float did_target_run_light = log->did_target_run_light;
    float did_target_fail = log->did_target_fail;
    float did_target_make_progress = log->did_target_make_progress;
    float did_target_have_at_fault_collision = log->did_target_have_at_fault_collision;
    float adversaries_offroad_rate = log->adversaries_offroad_rate;
    float adversaries_collision_rate = log->adversaries_collision_rate;
    float adversaries_target_collision_rate = log->adversaries_target_collision_rate;
    float adversaries_adversary_collision_rate = log->adversaries_adversary_collision_rate;
    float adversaries_red_light_violation_rate = log->adversaries_red_light_violation_rate;
    float adversaries_at_fault_collision_rate = log->adversaries_at_fault_collision_rate;
    float adversaries_making_progress_rate = log->adversaries_making_progress_rate;
    float adversaries_comfort_score = log->adversaries_comfort_score;
    float adversaries_comfort_violation_rate = log->adversaries_comfort_violation_rate;
    float adversaries_comfort_longitudinal_accel_violations_per_timestep =
        log->adversaries_comfort_longitudinal_accel_violations_per_timestep;
    float adversaries_comfort_lateral_accel_violations_per_timestep =
        log->adversaries_comfort_lateral_accel_violations_per_timestep;
    float adversaries_comfort_jerk_violations_per_timestep = log->adversaries_comfort_jerk_violations_per_timestep;
    float adversaries_uncomfortable_timestep_rate = log->adversaries_uncomfortable_timestep_rate;
    float adversaries_avg_speed = log->adversaries_avg_speed;
    float adversaries_lane_center_rate = log->adversaries_lane_center_rate;
    float adversaries_lane_heading_aligned_rate = log->adversaries_lane_heading_aligned_rate;
    float adversaries_velocity_progress = log->adversaries_velocity_progress;
    float adversaries_speed_limit_compliance = log->adversaries_speed_limit_compliance;
    float adversaries_driving_direction_score = log->adversaries_driving_direction_score;
    float adversaries_multi_lane_time = log->adversaries_multi_lane_time;
    float adversaries_multi_lane_score = log->adversaries_multi_lane_score;
    float adversaries_dnf_rate = log->adversaries_dnf_rate;
    float adversaries_score = log->adversaries_score;
    float adversaries_num_waypoints_reached = log->adversaries_num_waypoints_reached;
    float target_num_goals_reached = log->target_num_goals_reached;
    float target_ttc_within_bound_rate = log->target_ttc_within_bound_rate;
    float target_progress_ratio = log->target_progress_ratio;
    float target_puffer_score = log->target_puffer_score;
    float target_comfort_violations_per_timestep = log->target_comfort_violations_per_timestep;
    float target_comfort_longitudinal_accel_violations_per_timestep =
        log->target_comfort_longitudinal_accel_violations_per_timestep;
    float target_comfort_lateral_accel_violations_per_timestep =
        log->target_comfort_lateral_accel_violations_per_timestep;
    float target_comfort_jerk_violations_per_timestep = log->target_comfort_jerk_violations_per_timestep;
    float target_uncomfortable_timestep_rate = log->target_uncomfortable_timestep_rate;
    float target_collision_count = log->target_collision_count;
    float target_collision_severity = log->target_collision_severity;
    float target_collision_responsibility = log->target_collision_responsibility;
    float target_collision_impact_zone = log->target_collision_impact_zone;
    float target_collision_type = log->target_collision_type;
    float target_collision_target_speed = log->target_collision_target_speed;
    float target_collision_other_speed = log->target_collision_other_speed;
    float target_collision_relative_speed = log->target_collision_relative_speed;
    float target_collision_other_active = log->target_collision_other_active;
    float target_collision_other_stopped = log->target_collision_other_stopped;
    float target_collision_other_removed = log->target_collision_other_removed;
    float target_avoidability_by_braking = log->target_avoidability_by_braking;
    float adversaries_collision_count = log->adversaries_collision_count;
    float adversaries_collision_severity = log->adversaries_collision_severity;
    float adversaries_collision_responsibility = log->adversaries_collision_responsibility;
    float adversaries_collision_impact_zone = log->adversaries_collision_impact_zone;
    float target_hit_count = log->target_hit_count;
    float target_hit_responsibility = log->target_hit_responsibility;
    float target_hit_low_responsibility_rate = log->target_hit_low_responsibility_rate;
    float target_hit_at_fault_count = log->target_hit_at_fault_count;
    float num_goals_reached = log->num_goals_reached;
    float ttc_within_bound_rate = log->ttc_within_bound_rate;
    float progress_ratio = log->progress_ratio;
    float puffer_score = log->puffer_score;

    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < num_keys; i++) {
        ((float *)log)[i] /= n;
    }

    if (target_n > 0.0f) {
        log->target_episode_length = target_episode_length / target_n;
        log->target_episode_return = target_episode_return / target_n;
        log->target_episode_return_collision = target_episode_return_collision / target_n;
        log->target_episode_return_offroad = target_episode_return_offroad / target_n;
        log->target_episode_return_drive = target_episode_return_drive / target_n;
        log->did_target_collide = did_target_collide / target_n;
        log->did_target_offroad = did_target_offroad / target_n;
        log->did_target_run_light = did_target_run_light / target_n;
        log->did_target_fail = did_target_fail / target_n;
        log->did_target_make_progress = did_target_make_progress / target_n;
        log->did_target_have_at_fault_collision = did_target_have_at_fault_collision / target_n;
        log->target_num_goals_reached = target_num_goals_reached / target_n;
        log->target_ttc_within_bound_rate = target_ttc_within_bound_rate / target_n;
        log->target_progress_ratio = target_progress_ratio / target_n;
        log->target_puffer_score = target_puffer_score / target_n;
        log->target_comfort_violations_per_timestep = target_comfort_violations_per_timestep / target_n;
        log->target_comfort_longitudinal_accel_violations_per_timestep =
            target_comfort_longitudinal_accel_violations_per_timestep / target_n;
        log->target_comfort_lateral_accel_violations_per_timestep =
            target_comfort_lateral_accel_violations_per_timestep / target_n;
        log->target_comfort_jerk_violations_per_timestep = target_comfort_jerk_violations_per_timestep / target_n;
        log->target_uncomfortable_timestep_rate = target_uncomfortable_timestep_rate / target_n;
    }

    if (target_collision_count > 0.0f) {
        log->target_collision_severity = target_collision_severity / target_collision_count;
        log->target_collision_responsibility = target_collision_responsibility / target_collision_count;
        log->target_collision_impact_zone = target_collision_impact_zone / target_collision_count;
        log->target_collision_type = target_collision_type / target_collision_count;
        log->target_collision_target_speed = target_collision_target_speed / target_collision_count;
        log->target_collision_other_speed = target_collision_other_speed / target_collision_count;
        log->target_collision_relative_speed = target_collision_relative_speed / target_collision_count;
        log->target_collision_other_active = target_collision_other_active / target_collision_count;
        log->target_collision_other_stopped = target_collision_other_stopped / target_collision_count;
        log->target_collision_other_removed = target_collision_other_removed / target_collision_count;
        log->target_avoidability_by_braking = target_avoidability_by_braking / target_collision_count;
    }
    log->target_collision_count = target_collision_count;

    if (adversaries_collision_count > 0.0f) {
        log->adversaries_collision_severity = adversaries_collision_severity / adversaries_collision_count;
        log->adversaries_collision_responsibility = adversaries_collision_responsibility / adversaries_collision_count;
        log->adversaries_collision_impact_zone = adversaries_collision_impact_zone / adversaries_collision_count;
    }
    log->adversaries_collision_count = adversaries_collision_count;

    if (target_hit_count > 0.0f) {
        log->target_hit_responsibility = target_hit_responsibility / target_hit_count;
        log->target_hit_low_responsibility_rate = target_hit_low_responsibility_rate / target_hit_count;
        log->target_hit_at_fault_rate = target_hit_at_fault_count / target_hit_count;
    }
    log->target_hit_count = target_hit_count;
    log->target_hit_at_fault_count = target_hit_at_fault_count;

    float adversary_n = n - target_n;
    log->episode_length = adversary_n > 0.0f ? (episode_length - target_episode_length) / adversary_n : 0.0f;
    log->episode_return = adversary_n > 0.0f ? (episode_return - target_episode_return) / adversary_n : 0.0f;
    log->episode_return_collision =
        adversary_n > 0.0f ? (episode_return_collision - target_episode_return_collision) / adversary_n : 0.0f;
    log->episode_return_offroad =
        adversary_n > 0.0f ? (episode_return_offroad - target_episode_return_offroad) / adversary_n : 0.0f;
    log->episode_return_drive =
        adversary_n > 0.0f ? (episode_return_drive - target_episode_return_drive) / adversary_n : 0.0f;
    log->episode_return_adversarial = adversary_n > 0.0f ? episode_return_adversarial / adversary_n : 0.0f;
    log->adversaries_offroad_rate = adversary_n > 0.0f ? adversaries_offroad_rate / adversary_n : 0.0f;
    log->adversaries_collision_rate = adversary_n > 0.0f ? adversaries_collision_rate / adversary_n : 0.0f;
    log->adversaries_target_collision_rate =
        adversary_n > 0.0f ? adversaries_target_collision_rate / adversary_n : 0.0f;
    log->adversaries_adversary_collision_rate =
        adversary_n > 0.0f ? adversaries_adversary_collision_rate / adversary_n : 0.0f;
    log->adversaries_red_light_violation_rate =
        adversary_n > 0.0f ? adversaries_red_light_violation_rate / adversary_n : 0.0f;
    log->adversaries_at_fault_collision_rate =
        adversary_n > 0.0f ? adversaries_at_fault_collision_rate / adversary_n : 0.0f;
    log->adversaries_making_progress_rate = adversary_n > 0.0f ? adversaries_making_progress_rate / adversary_n : 0.0f;
    log->adversaries_comfort_score = adversary_n > 0.0f ? adversaries_comfort_score / adversary_n : 0.0f;
    log->adversaries_comfort_violation_rate =
        adversary_n > 0.0f ? adversaries_comfort_violation_rate / adversary_n : 0.0f;
    log->adversaries_comfort_longitudinal_accel_violations_per_timestep =
        adversary_n > 0.0f ? adversaries_comfort_longitudinal_accel_violations_per_timestep / adversary_n : 0.0f;
    log->adversaries_comfort_lateral_accel_violations_per_timestep =
        adversary_n > 0.0f ? adversaries_comfort_lateral_accel_violations_per_timestep / adversary_n : 0.0f;
    log->adversaries_comfort_jerk_violations_per_timestep =
        adversary_n > 0.0f ? adversaries_comfort_jerk_violations_per_timestep / adversary_n : 0.0f;
    log->adversaries_uncomfortable_timestep_rate =
        adversary_n > 0.0f ? adversaries_uncomfortable_timestep_rate / adversary_n : 0.0f;
    log->adversaries_avg_speed = adversary_n > 0.0f ? adversaries_avg_speed / adversary_n : 0.0f;
    log->adversaries_lane_center_rate = adversary_n > 0.0f ? adversaries_lane_center_rate / adversary_n : 0.0f;
    log->adversaries_lane_heading_aligned_rate =
        adversary_n > 0.0f ? adversaries_lane_heading_aligned_rate / adversary_n : 0.0f;
    log->adversaries_velocity_progress = adversary_n > 0.0f ? adversaries_velocity_progress / adversary_n : 0.0f;
    log->adversaries_speed_limit_compliance =
        adversary_n > 0.0f ? adversaries_speed_limit_compliance / adversary_n : 0.0f;
    log->adversaries_driving_direction_score =
        adversary_n > 0.0f ? adversaries_driving_direction_score / adversary_n : 0.0f;
    log->adversaries_multi_lane_time = adversary_n > 0.0f ? adversaries_multi_lane_time / adversary_n : 0.0f;
    log->adversaries_multi_lane_score = adversary_n > 0.0f ? adversaries_multi_lane_score / adversary_n : 0.0f;
    log->adversaries_dnf_rate = adversary_n > 0.0f ? adversaries_dnf_rate / adversary_n : 0.0f;
    log->adversaries_score = adversary_n > 0.0f ? adversaries_score / adversary_n : 0.0f;
    log->adversaries_num_waypoints_reached =
        adversary_n > 0.0f ? adversaries_num_waypoints_reached / adversary_n : 0.0f;
    log->num_goals_reached = adversary_n > 0.0f ? (num_goals_reached - target_num_goals_reached) / adversary_n : 0.0f;
    log->ttc_within_bound_rate =
        adversary_n > 0.0f ? (ttc_within_bound_rate - target_ttc_within_bound_rate) / adversary_n : 0.0f;
    log->progress_ratio = adversary_n > 0.0f ? (progress_ratio - target_progress_ratio) / adversary_n : 0.0f;
    log->puffer_score = adversary_n > 0.0f ? (puffer_score - target_puffer_score) / adversary_n : 0.0f;
}

static inline void enqueue_completed_episode_summary(Drive *env, const Log *episode_log) {
    CompletedEpisodeSummary summary = {0};
    summary.log = *episode_log;
    summary.n = episode_log->n;
    summary.target_n = episode_log->target_n;
    summary.active_agent_count = env->active_agent_count;
    summary.timestep = env->timestep;

    if (env->map_name != NULL) {
        snprintf(summary.map_name, sizeof(summary.map_name), "%s", env->map_name);
    }
    if (env->scenario_id[0] != '\0') {
        snprintf(summary.scenario_id, sizeof(summary.scenario_id), "%s", env->scenario_id);
    }
    if (env->dataset_name[0] != '\0') {
        snprintf(summary.dataset_name, sizeof(summary.dataset_name), "%s", env->dataset_name);
    }

    normalize_log_for_output(&summary.log, summary.n, summary.target_n);

    if (env->completed_episode_queue_count == COMPLETED_EPISODE_QUEUE_CAPACITY) {
        env->completed_episode_queue_head = (env->completed_episode_queue_head + 1) % COMPLETED_EPISODE_QUEUE_CAPACITY;
        env->completed_episode_queue_count--;
    }

    env->completed_episode_queue[env->completed_episode_queue_tail] = summary;
    env->completed_episode_queue_tail = (env->completed_episode_queue_tail + 1) % COMPLETED_EPISODE_QUEUE_CAPACITY;
    env->completed_episode_queue_count++;
}

static inline int pop_completed_episode_summary(Drive *env, CompletedEpisodeSummary *out_summary) {
    if (env->completed_episode_queue_count <= 0) {
        return 0;
    }

    *out_summary = env->completed_episode_queue[env->completed_episode_queue_head];
    env->completed_episode_queue_head = (env->completed_episode_queue_head + 1) % COMPLETED_EPISODE_QUEUE_CAPACITY;
    env->completed_episode_queue_count--;
    return 1;
}
static float compute_euclidean_distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx * dx + dy * dy);
    return distance;
}

static int compare_depthpoint(const void *a, const void *b) {
    float diff = ((const DepthPoint *)a)->euclidean_dis - ((const DepthPoint *)b)->euclidean_dis;
    return (diff > 0.0f) - (diff < 0.0f);
}

static float clip(float value, float min, float max) {
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

// Normalize heading to [-pi, pi]
static float normalize_heading(float heading) {
    heading = fmodf(heading, 2.0f * M_PI);
    if (heading > M_PI)
        heading -= 2.0f * M_PI;
    else if (heading < -M_PI)
        heading += 2.0f * M_PI;
    return heading;
}

static float compute_heading_diff(float heading1, float heading2) {
    float heading_diff = heading1 - heading2;
    return normalize_heading(heading_diff);
}

static int traffic_control_in_scope(int type, int scope) {
    if (scope == TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS) {
        return type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    } else if (scope == TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS_STOP_SIGN) {
        return type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || type == TRAFFIC_CONTROL_TYPE_STOP_SIGN;
    } else if (scope == TRAFFIC_CONTROL_SCOPE_ALL) {
        return type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || type == TRAFFIC_CONTROL_TYPE_STOP_SIGN ||
               type == TRAFFIC_CONTROL_TYPE_YIELD_SIGN;
    } else {
        return type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    }
}

static void invalidate_agent(Agent *agent) {
    agent->sim_x = INVALID_POSITION;
    agent->sim_y = INVALID_POSITION;
    agent->sim_z = 0.0f;
    agent->prev_sim_x = INVALID_POSITION;
    agent->prev_sim_y = INVALID_POSITION;
    agent->prev_sim_heading = 0.0f;
    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    agent->sim_vx = 0.0f;
    agent->sim_vy = 0.0f;
    agent->yaw_rate = 0.0f;
    agent->sim_speed = 0.0f;
    agent->sim_speed_signed = 0.0f;
    agent->a_long = 0.0f;
    agent->a_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
    agent->steering_angle = 0.0f;
    agent->sim_valid = 0;
}

static void update_agent_speed(Agent *agent) {
    float speed = sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy);
    float v_dot_heading = agent->sim_vx * agent->cos_heading + agent->sim_vy * agent->sin_heading;
    agent->sim_speed = speed;
    agent->sim_speed_signed = copysignf(speed, v_dot_heading);
}

// Trajectory planning/control can operate in the rear-axle frame even though sim state stays center-based.
static inline float compute_log_yaw_rate(const Agent *agent, int timestep, float dt) {
    if (dt <= 0.0f)
        return 0.0f;

    const int prev_t = timestep - 1;
    const int next_t = timestep + 1;
    const int has_prev = (prev_t >= 0) && (agent->log_valid[prev_t] == 1);
    const int has_next = (next_t < agent->trajectory_length) && (agent->log_valid[next_t] == 1);

    if (has_prev && has_next) {
        float dtheta = normalize_heading(agent->log_heading[next_t] - agent->log_heading[prev_t]);
        return dtheta / (2.0f * dt);
    }
    if (has_next) {
        float dtheta = normalize_heading(agent->log_heading[next_t] - agent->log_heading[timestep]);
        return dtheta / dt;
    }
    if (has_prev) {
        float dtheta = normalize_heading(agent->log_heading[timestep] - agent->log_heading[prev_t]);
        return dtheta / dt;
    }

    return 0.0f;
}

// Random uniform in [min, max]
static float random_uniform(float min_val, float max_val) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min_val + scale * (max_val - min_val);
}

static inline void clamp_nonneg_speed(float *v, float *a) {
    if (*v < 0.0f) {
        *v = 0.0f;
        *a = 0.0f;
    }
}

static inline void snapshot_previous_pose(Agent *agent) {
    agent->prev_sim_x = agent->sim_x;
    agent->prev_sim_y = agent->sim_y;
    agent->prev_sim_heading = agent->sim_heading;
}

// Pushes an agent's current (pre-step) state into its rolling brake-history buffer.
// Buffer index TARGET_BRAKE_HISTORY_LEN-1 holds the state one step ago; index 0 holds the
// state TARGET_BRAKE_HISTORY_LEN steps ago. "Now" (this step's post-move state) is read
// directly off the live Agent by the avoidability-by-braking analysis, not stored here.
static void push_brake_history(Agent *agent) {
    if (agent->removed || agent->sim_x == INVALID_POSITION) {
        return;
    }

    for (int k = 0; k < TARGET_BRAKE_HISTORY_LEN - 1; k++) {
        agent->brake_hist_x[k] = agent->brake_hist_x[k + 1];
        agent->brake_hist_y[k] = agent->brake_hist_y[k + 1];
        agent->brake_hist_z[k] = agent->brake_hist_z[k + 1];
        agent->brake_hist_heading[k] = agent->brake_hist_heading[k + 1];
        agent->brake_hist_speed_signed[k] = agent->brake_hist_speed_signed[k + 1];
    }

    int last = TARGET_BRAKE_HISTORY_LEN - 1;
    agent->brake_hist_x[last] = agent->sim_x;
    agent->brake_hist_y[last] = agent->sim_y;
    agent->brake_hist_z[last] = agent->sim_z;
    agent->brake_hist_heading[last] = agent->sim_heading;
    agent->brake_hist_speed_signed[last] = agent->sim_speed_signed;
    if (agent->brake_hist_count < TARGET_BRAKE_HISTORY_LEN) {
        agent->brake_hist_count++;
    }
}

// Mixed uniform distribution X(a) = 0.5*U(1/a, 1) + 0.5*U(1, a)
static float mixed_uniform(float a) {
    if ((float)rand() / (float)RAND_MAX < 0.5f) {
        return random_uniform(1.0f / a, 1.0f);
    } else {
        return random_uniform(1.0f, a);
    }
}

typedef struct {
    float min_val;
    float max_val;
} RewardBound;

static const RewardBound REWARD_BOUNDS[NUM_REWARD_COEFS] = {
    {2.0f, 12.0f},      // REWARD_COEF_GOAL_RADIUS     δ_goal ~ U(2, 12)
    {0.0f, 20.0f},      // REWARD_COEF_GOAL_SPEED
    {0.0f, 3.0f},       // REWARD_COEF_COLLISION       α_collision ~ U(0, 3)
    {0.0f, 3.0f},       // REWARD_COEF_OFFROAD         α_boundary ~ U(0, 3)
    {0.0f, 0.1f},       // REWARD_COEF_COMFORT         α_comfort ~ U(0, 0.1)
    {2.5e-4f, 2.5e-2f}, // REWARD_COEF_LANE_ALIGN      α_l-align ~ U(2.5e-4, 2.5e-2)
    {0.0f, 1.0f},       // REWARD_COEF_VEL_ALIGN       α_vel-align ~ U(0, 1)
    {2.5e-4f, 7.5e-3f}, // REWARD_COEF_LANE_CENTER     α_l-center ~ U(2.5e-4, 7.5e-3)
    {-0.5f, 0.5f},      // REWARD_COEF_CENTER_BIAS     α_center-bias ~ U(-0.5, 0.5)
    {0.0f, 5e-3f},      // REWARD_COEF_VELOCITY        α_velocity = 2.5e-3 (fixed)
    {2.5e-4f, 7.5e-3f}, // REWARD_COEF_REVERSE         α_reverse ~ U(2.5e-4, 7.5e-3)
    {0.0f, 1.0f},       // REWARD_COEF_STOP_LINE       α_stop-line ~ U(0, 1)
    {0.0f, 5e-5f},      // REWARD_COEF_TIMESTEP        α_timestep = 2.5e-5 (fixed)
    {0.0f, 1.0f},       // REWARD_COEF_OVERSPEED
    {0.8f, 1.25f},      // REWARD_COEF_THROTTLE        C_throttle
    {0.8f, 1.25f},      // REWARD_COEF_STEER           C_steer
    {0.666f, 1.5f},     // REWARD_COEF_ACC             C_acc
};

// Forward declarations
void move_expert(Drive *env, float *actions, int agent_idx);

// Generate per-agent reward conditioning coefficients
static void generate_reward_coefs(Drive *env, Agent *agent) {
    if (env->reward_randomization) {
        // Standard Uniform Randomizations (referencing the bounds array)
        agent->reward_coefs[REWARD_COEF_GOAL_RADIUS] = random_uniform(REWARD_BOUNDS[REWARD_COEF_GOAL_RADIUS].min_val,
                                                                      REWARD_BOUNDS[REWARD_COEF_GOAL_RADIUS].max_val);
        agent->reward_coefs[REWARD_COEF_GOAL_SPEED] = random_uniform(REWARD_BOUNDS[REWARD_COEF_GOAL_SPEED].min_val,
                                                                     REWARD_BOUNDS[REWARD_COEF_GOAL_SPEED].max_val);
        agent->reward_coefs[REWARD_COEF_COLLISION] =
            random_uniform(REWARD_BOUNDS[REWARD_COEF_COLLISION].min_val, REWARD_BOUNDS[REWARD_COEF_COLLISION].max_val);
        agent->reward_coefs[REWARD_COEF_OFFROAD] =
            random_uniform(REWARD_BOUNDS[REWARD_COEF_OFFROAD].min_val, REWARD_BOUNDS[REWARD_COEF_OFFROAD].max_val);
        agent->reward_coefs[REWARD_COEF_COMFORT] =
            random_uniform(REWARD_BOUNDS[REWARD_COEF_COMFORT].min_val, REWARD_BOUNDS[REWARD_COEF_COMFORT].max_val);
        agent->reward_coefs[REWARD_COEF_LANE_ALIGN] = random_uniform(REWARD_BOUNDS[REWARD_COEF_LANE_ALIGN].min_val,
                                                                     REWARD_BOUNDS[REWARD_COEF_LANE_ALIGN].max_val);
        agent->reward_coefs[REWARD_COEF_LANE_CENTER] = random_uniform(REWARD_BOUNDS[REWARD_COEF_LANE_CENTER].min_val,
                                                                      REWARD_BOUNDS[REWARD_COEF_LANE_CENTER].max_val);
        agent->reward_coefs[REWARD_COEF_STOP_LINE] =
            random_uniform(REWARD_BOUNDS[REWARD_COEF_STOP_LINE].min_val, REWARD_BOUNDS[REWARD_COEF_STOP_LINE].max_val);
        agent->reward_coefs[REWARD_COEF_CENTER_BIAS] = random_uniform(REWARD_BOUNDS[REWARD_COEF_CENTER_BIAS].min_val,
                                                                      REWARD_BOUNDS[REWARD_COEF_CENTER_BIAS].max_val);
        agent->reward_coefs[REWARD_COEF_VEL_ALIGN] =
            random_uniform(REWARD_BOUNDS[REWARD_COEF_VEL_ALIGN].min_val, REWARD_BOUNDS[REWARD_COEF_VEL_ALIGN].max_val);
        agent->reward_coefs[REWARD_COEF_OVERSPEED] =
            random_uniform(REWARD_BOUNDS[REWARD_COEF_OVERSPEED].min_val, REWARD_BOUNDS[REWARD_COEF_OVERSPEED].max_val);
        agent->reward_coefs[REWARD_COEF_REVERSE] =
            random_uniform(REWARD_BOUNDS[REWARD_COEF_REVERSE].min_val, REWARD_BOUNDS[REWARD_COEF_REVERSE].max_val);
        // Fixed values (Must fall within the bounds defined above)
        agent->reward_coefs[REWARD_COEF_VELOCITY] = 2.5e-3f;
        agent->reward_coefs[REWARD_COEF_TIMESTEP] = 2.5e-5f;
        // Dynamic conditioning (Mixed Uniform)
        agent->reward_coefs[REWARD_COEF_THROTTLE] = mixed_uniform(1.25f);
        agent->reward_coefs[REWARD_COEF_STEER] = mixed_uniform(1.25f);
        agent->reward_coefs[REWARD_COEF_ACC] = mixed_uniform(1.5f);
    } else {
        // Fixed coefficients
        agent->reward_coefs[REWARD_COEF_GOAL_RADIUS] = env->goal_radius;
        agent->reward_coefs[REWARD_COEF_GOAL_SPEED] = env->goal_speed;
        agent->reward_coefs[REWARD_COEF_COLLISION] = env->reward_vehicle_collision;
        agent->reward_coefs[REWARD_COEF_OFFROAD] = env->reward_offroad_collision;
        agent->reward_coefs[REWARD_COEF_COMFORT] = env->reward_comfort;
        agent->reward_coefs[REWARD_COEF_LANE_ALIGN] = env->reward_lane_align;
        agent->reward_coefs[REWARD_COEF_LANE_CENTER] = env->reward_lane_center;
        agent->reward_coefs[REWARD_COEF_VELOCITY] = env->reward_velocity;
        agent->reward_coefs[REWARD_COEF_STOP_LINE] = env->reward_stop_line;
        agent->reward_coefs[REWARD_COEF_CENTER_BIAS] = env->reward_center_bias;
        agent->reward_coefs[REWARD_COEF_VEL_ALIGN] = env->reward_vel_align;
        agent->reward_coefs[REWARD_COEF_OVERSPEED] = env->reward_overspeed;
        agent->reward_coefs[REWARD_COEF_TIMESTEP] = env->reward_timestep;
        agent->reward_coefs[REWARD_COEF_REVERSE] = env->reward_reverse;
        // Dynamic conditioning coefficients
        agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
        agent->reward_coefs[REWARD_COEF_STEER] = 1.0f;
        agent->reward_coefs[REWARD_COEF_ACC] = 1.0f;
    }
}

// Generate procedural traffic light states for GIGAFLOW mode
static void generate_traffic_light_states(Drive *env) {
    int steps = env->scenario_length;
    float dt = env->dt;
    int deterministic_traffic_lights = env->eval_mode || env->deterministic_traffic_lights;

    // 20% chance: disable ALL lights for this episode
    int disable_all = (!deterministic_traffic_lights) && (random_uniform(0.0f, 1.0f) < TL_EPISODE_DISABLE_PROB);

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || traffic->states == NULL ||
            traffic->state_length <= 0)
            continue;

        int fill_steps = steps;
        if (traffic->state_length < fill_steps)
            fill_steps = traffic->state_length;

        if (disable_all) {
            for (int t = 0; t < fill_steps; t++)
                traffic->states[t] = TRAFFIC_CONTROL_STATE_OFF;
            continue;
        }

        if (!deterministic_traffic_lights) {
            // Individual removal
            if (random_uniform(0.0f, 1.0f) < TL_INDIVIDUAL_REMOVE_PROB) {
                for (int t = 0; t < fill_steps; t++)
                    traffic->states[t] = TRAFFIC_CONTROL_STATE_OFF;
                continue;
            }
            // Always green
            if (random_uniform(0.0f, 1.0f) < TL_ALWAYS_GREEN_PROB) {
                for (int t = 0; t < fill_steps; t++)
                    traffic->states[t] = TRAFFIC_CONTROL_STATE_GREEN;
                continue;
            }
        }

        // Compute phase durations
        float dur_green, dur_yellow, dur_red;
        if (deterministic_traffic_lights) {
            dur_green = TL_DEFAULT_GREEN_DURATION;
            dur_yellow = TL_DEFAULT_YELLOW_DURATION;
            dur_red = TL_DEFAULT_RED_DURATION;
        } else {
            dur_green = random_uniform(0.1 * TL_DEFAULT_GREEN_DURATION, TL_DEFAULT_GREEN_DURATION);
            dur_yellow = random_uniform(0.5f * TL_DEFAULT_YELLOW_DURATION, 0.75f * TL_DEFAULT_YELLOW_DURATION);
            dur_red = random_uniform(0.15f * TL_DEFAULT_RED_DURATION, 5.0f * TL_DEFAULT_RED_DURATION);
        }

        int steps_green = (int)(dur_green / dt);
        if (steps_green < 1)
            steps_green = 1;
        int steps_yellow = (int)(dur_yellow / dt);
        if (steps_yellow < 1)
            steps_yellow = 1;
        int steps_red = (int)(dur_red / dt);
        if (steps_red < 1)
            steps_red = 1;
        int cycle_length = steps_green + steps_yellow + steps_red;

        // Random phase offset
        int offset = rand() % cycle_length;

        // Fill states: GREEN -> YELLOW -> RED -> repeat
        for (int t = 0; t < fill_steps; t++) {
            int phase = (t + offset) % cycle_length;
            if (phase < steps_green)
                traffic->states[t] = TRAFFIC_CONTROL_STATE_GREEN;
            else if (phase < steps_green + steps_yellow)
                traffic->states[t] = TRAFFIC_CONTROL_STATE_YELLOW;
            else
                traffic->states[t] = TRAFFIC_CONTROL_STATE_RED;
        }
    }
}

static void subsample_road_observation_rows(float *buffer, int collected_count, int keep_count) {
    if (keep_count <= 0 || collected_count <= keep_count)
        return;
    float tmp[ROAD_FEATURES];
    for (int sample_idx = 0; sample_idx < keep_count; sample_idx++) {
        int remaining = collected_count - sample_idx;
        int swap_idx = (remaining > 1) ? sample_idx + (rand() % remaining) : sample_idx;
        if (swap_idx == sample_idx)
            continue;
        float *a = &buffer[sample_idx * ROAD_FEATURES];
        float *b = &buffer[swap_idx * ROAD_FEATURES];
        memcpy(tmp, a, sizeof(tmp));
        memcpy(a, b, sizeof(tmp));
        memcpy(b, tmp, sizeof(tmp));
    }
}

// ========================================
// Grid Map Functions
// ========================================

static int get_grid_index(Drive *env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x ||
        env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1; // Invalid grid coordinates
    }

    float relativeX = x1 - env->grid_map->top_left_x;     // Distance from left
    float relativeY = y1 - env->grid_map->bottom_right_y; // Distance from bottom
    int gridX = (int)(relativeX / GRID_CELL_SIZE);        // Column index
    int gridY = (int)(relativeY / GRID_CELL_SIZE);        // Row index
    if (gridX < 0 || gridX >= env->grid_map->grid_cols || gridY < 0 || gridY >= env->grid_map->grid_rows) {
        return -1; // Return -1 for out of bounds
    }
    int index = (gridY * env->grid_map->grid_cols) + gridX;
    return index;
}

static void add_entity_to_grid(Drive *env, int grid_index, int entity_type, int entity_idx, int geometry_idx,
                               int *cell_entities_insert_index) {
    if (grid_index == -1) {
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if (count >= env->grid_map->cell_entities_count[grid_index]) {
        printf("Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): "
               "%d\n",
               grid_index, count, env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_type = entity_type;
    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    cell_entities_insert_index[grid_index] = count + 1;
}

static void init_grid_map(Drive *env) {
    // Allocate memory for the grid map structure
    env->grid_map = (GridMap *)malloc(sizeof(GridMap));
    env->grid_map->num_drivable_grid_cell = 0;

    // Find top left and bottom right points of the map
    float top_left_x = 0.0f;
    float top_left_y = 0.0f;
    float bottom_right_x = 0.0f;
    float bottom_right_y = 0.0f;
    bool first_valid_point = false;
    for (int i = 0; i < env->num_road_elements; i++) {
        // Check all points in the geometry for road elements (ROAD_LANE, ROAD_LINE, ROAD_EDGE)
        if (!is_road(env->road_elements[i].type))
            continue;
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_length; j++) {
            if (element->x[j] == INVALID_POSITION)
                continue;
            if (element->y[j] == INVALID_POSITION)
                continue;
            if (!first_valid_point) {
                top_left_x = bottom_right_x = element->x[j];
                top_left_y = bottom_right_y = element->y[j];
                first_valid_point = true;
                continue;
            }
            if (element->x[j] < top_left_x)
                top_left_x = element->x[j];
            if (element->x[j] > bottom_right_x)
                bottom_right_x = element->x[j];
            if (element->y[j] > top_left_y)
                top_left_y = element->y[j];
            if (element->y[j] < bottom_right_y)
                bottom_right_y = element->y[j];
        }
    }

    env->grid_map->top_left_x = top_left_x;
    env->grid_map->top_left_y = top_left_y;
    env->grid_map->bottom_right_x = bottom_right_x;
    env->grid_map->bottom_right_y = bottom_right_y;
    env->grid_map->cell_size_x = GRID_CELL_SIZE;
    env->grid_map->cell_size_y = GRID_CELL_SIZE;

    // Calculate grid dimensions
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->cells = (GridMapEntity **)calloc(grid_cell_count, sizeof(GridMapEntity *));
    env->grid_map->cell_entities_count = (int *)calloc(grid_cell_count, sizeof(int));

    // Calculate number of entities in each grid cell
    for (int i = 0; i < env->num_road_elements; i++) {
        for (int j = 0; j < env->road_elements[i].segment_length - 1; j++) {
            float x_center = (env->road_elements[i].x[j] + env->road_elements[i].x[j + 1]) / 2;
            float y_center = (env->road_elements[i].y[j] + env->road_elements[i].y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            if (grid_index == -1)
                continue; // Skip out-of-bounds entities
            env->grid_map->cell_entities_count[grid_index]++;
        }
    }

    int *cell_entities_insert_index = (int *)calloc(grid_cell_count, sizeof(int));

    // Initialize grid cells
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        env->grid_map->cells[grid_index] =
            (GridMapEntity *)calloc(env->grid_map->cell_entities_count[grid_index], sizeof(GridMapEntity));
    }
    for (int i = 0; i < grid_cell_count; i++) {
        if (cell_entities_insert_index[i] != 0) {
            printf("Error: cell_entities_insert_index[%d] not zero during initialization.\n", i);
            cell_entities_insert_index[i] = 0;
        }
    }

    // Track which grid cells contain drivable lanes (for spawning)
    bool *drivable_grid_seen = (bool *)calloc(grid_cell_count, sizeof(bool));

    // Populate grid cells and count unique drivable grid cells
    for (int i = 0; i < env->num_road_elements; i++) {
        for (int j = 0; j < env->road_elements[i].segment_length - 1; j++) {
            float x_center = (env->road_elements[i].x[j] + env->road_elements[i].x[j + 1]) / 2;
            float y_center = (env->road_elements[i].y[j] + env->road_elements[i].y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            if (grid_index == -1)
                continue; // Skip out-of-bounds entities
            add_entity_to_grid(env, grid_index, ENTITY_TYPE_ROAD_ELEMENT, i, j, cell_entities_insert_index);
            // Count unique drivable grid cells
            if (is_drivable_road_lane(env->road_elements[i].type) && !drivable_grid_seen[grid_index]) {
                drivable_grid_seen[grid_index] = true;
                env->grid_map->num_drivable_grid_cell++;
            }
        }
    }

    // Allocate and fill drivable grid index array
    env->grid_map->grid_index_drivable = (int *)malloc(env->grid_map->num_drivable_grid_cell * sizeof(int));
    int drivable_idx = 0;
    for (int i = 0; i < grid_cell_count; i++) {
        if (drivable_grid_seen[i]) {
            env->grid_map->grid_index_drivable[drivable_idx++] = i;
        }
    }

    free(drivable_grid_seen);
    free(cell_entities_insert_index);
}

static void init_neighbor_offsets(Drive *env) {
    // Allocate memory for the offsets
    env->neighbor_offsets = (int *)calloc(env->grid_map->vision_range * env->grid_map->vision_range * 2, sizeof(int));
    // neighbor offsets in a spiral pattern
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0;                  // Current x offset
    int y = 0;                  // Current y offset
    int dir = 0;                // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1;      // Number of steps in current direction
    int steps_taken = 0;        // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0;              // Total offsets added
    int max_offsets = env->grid_map->vision_range * env->grid_map->vision_range;
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0; // x offset
    env->neighbor_offsets[curr_idx++] = 0; // y offset
    total++;
    // Generate spiral pattern
    while (total < max_offsets) {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        // Only add if within vision range bounds
        if (abs(x) <= env->grid_map->vision_range / 2 && abs(y) <= env->grid_map->vision_range / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        // Check if we need to change direction
        if (steps_taken != steps_to_take)
            continue;
        steps_taken = 0;     // Reset steps taken
        dir = (dir + 1) % 4; // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        // Increase step length every two direction changes
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

static void cache_neighbor_offsets(Drive *env) {
    int count = 0;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity **)calloc(cell_count, sizeof(GridMapEntity *));
    env->grid_map->neighbor_cache_count = (int *)calloc(cell_count + 1, sizeof(int));
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if (current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i] =
            (GridMapEntity *)calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];

            // Skip if no entities or source is NULL
            if (grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }

            int src_idx = grid_index;
            int dst_idx = base_index;
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(&env->grid_map->neighbor_cache_entities[i][dst_idx], env->grid_map->cells[src_idx],
                   grid_count * sizeof(GridMapEntity));
            base_index += grid_count;
        }
    }
}

static int get_neighbors_entities(Drive *env, float x, float y, GridMapEntity *entity_list, int max_size,
                                  const int (*local_offsets)[2], int offset_size) {
    // Get the grid index for the given position (x, y)
    int index = get_grid_index(env, x, y);
    if (index == -1)
        return 0; // Return 0 size if position invalid
    // Calculate 2D grid coordinates
    int cellsX = env->grid_map->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows)
            continue;
        int neighborIndex = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighborIndex];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            int entityId = env->grid_map->cells[neighborIndex][j].entity_idx;
            int geometry_idx = env->grid_map->cells[neighborIndex][j].geometry_idx;
            int entity_type = env->grid_map->cells[neighborIndex][j].entity_type;
            entity_list[entity_list_count].entity_idx = entityId;
            entity_list[entity_list_count].geometry_idx = geometry_idx;
            entity_list[entity_list_count].entity_type = entity_type;
            entity_list_count += 1;
        }
    }
    return entity_list_count;
}

// ========================================
// Map Loading Functions
// ========================================

int load_map_binary(const char *filename, Drive *drive) {
    FILE *file = fopen(filename, "rb");
    if (!file)
        return -1;

    int num_total_agents, num_roads, num_traffic, num_objects;
    if (fread(&num_total_agents, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&num_roads, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&num_traffic, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&num_objects, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    drive->num_total_agents = num_total_agents;
    drive->num_road_elements = num_roads;
    drive->num_traffic_elements = num_traffic;
    drive->num_objects = num_objects;

    if (num_total_agents > 0) {
        drive->agents = (Agent *)calloc(num_total_agents, sizeof(Agent));
    }

    if (num_roads > 0) {
        drive->road_elements = (RoadMapElement *)calloc(num_roads, sizeof(RoadMapElement));
    }

    if (num_traffic > 0) {
        drive->traffic_elements = (TrafficControlElement *)calloc(num_traffic, sizeof(TrafficControlElement));
    }

    for (int i = 0; i < num_total_agents; i++) {
        Agent *agent = &drive->agents[i];
        int agent_id;

        if (fread(&agent_id, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (agent_id != i) {
            printf("[ERROR] -> Agent id %d != idx %d. Binary must be reindexed (id == idx).\n", agent_id, i);
            fclose(file);
            return -1;
        }
        if (fread(&agent->type, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&agent->trajectory_length, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        int tlen = agent->trajectory_length;
        agent->log_trajectory_x = (float *)malloc(tlen * sizeof(float));
        agent->log_trajectory_y = (float *)malloc(tlen * sizeof(float));
        agent->log_trajectory_z = (float *)malloc(tlen * sizeof(float));
        agent->log_heading = (float *)malloc(tlen * sizeof(float));
        agent->log_velocity_x = (float *)malloc(tlen * sizeof(float));
        agent->log_velocity_y = (float *)malloc(tlen * sizeof(float));
        agent->log_length = (float *)malloc(tlen * sizeof(float));
        agent->log_width = (float *)malloc(tlen * sizeof(float));
        agent->log_height = (float *)malloc(tlen * sizeof(float));
        agent->log_valid = (int *)malloc(tlen * sizeof(int));

        if ((size_t)tlen > 0 && fread(agent->log_trajectory_x, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_trajectory_y, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_trajectory_z, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_heading, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_velocity_x, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_velocity_y, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_length, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_width, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_height, sizeof(float), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t)tlen > 0 && fread(agent->log_valid, sizeof(int), tlen, file) != (size_t)tlen) {
            fclose(file);
            return -1;
        }

        if (fread(&agent->route_length, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        if (agent->route_length > 0) {
            agent->route = (int *)malloc(agent->route_length * sizeof(int));
            if (fread(agent->route, sizeof(int), agent->route_length, file) != (size_t)agent->route_length) {
                fclose(file);
                return -1;
            }
        } else {
            agent->route = NULL;
        }

        if (fread(&agent->route_gt_len, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        if (fread(&agent->goal_position_x, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&agent->goal_position_y, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&agent->goal_position_z, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&agent->mark_as_expert, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
    }

    for (int i = 0; i < num_roads; i++) {
        RoadMapElement *road = &drive->road_elements[i];
        int road_id;

        if (fread(&road_id, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (road_id != i) {
            printf("[ERROR] -> Road element id %d != idx %d. Binary must be reindexed (id == idx).\n", road_id, i);
            fclose(file);
            return -1;
        }
        if (fread(&road->type, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&road->segment_length, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        int slen = road->segment_length;

        road->x = (float *)malloc(slen * sizeof(float));
        road->y = (float *)malloc(slen * sizeof(float));
        road->z = (float *)malloc(slen * sizeof(float));

        if ((size_t)slen > 0 && fread(road->x, sizeof(float), slen, file) != (size_t)slen) {
            fclose(file);
            return -1;
        }
        if ((size_t)slen > 0 && fread(road->y, sizeof(float), slen, file) != (size_t)slen) {
            fclose(file);
            return -1;
        }
        if ((size_t)slen > 0 && fread(road->z, sizeof(float), slen, file) != (size_t)slen) {
            fclose(file);
            return -1;
        }

        road->headings = (float *)malloc(slen * sizeof(float));
        if ((size_t)slen > 0 && fread(road->headings, sizeof(float), slen, file) != (size_t)slen) {
            fclose(file);
            return -1;
        }

        if (is_road_lane(road->type)) {
            if (fread(&road->num_entries, sizeof(int), 1, file) != 1) {
                fclose(file);
                return -1;
            }
            if (road->num_entries > 0) {
                road->entry_lanes = (int *)malloc(road->num_entries * sizeof(int));
                if (fread(road->entry_lanes, sizeof(int), road->num_entries, file) != (size_t)road->num_entries) {
                    fclose(file);
                    return -1;
                }
            } else {
                road->entry_lanes = NULL;
            }

            if (fread(&road->num_exits, sizeof(int), 1, file) != 1) {
                fclose(file);
                return -1;
            }
            if (road->num_exits > 0) {
                road->exit_lanes = (int *)malloc(road->num_exits * sizeof(int));
                if (fread(road->exit_lanes, sizeof(int), road->num_exits, file) != (size_t)road->num_exits) {
                    fclose(file);
                    return -1;
                }
            } else {
                road->exit_lanes = NULL;
            }

            if (fread(&road->speed_limit, sizeof(float), 1, file) != 1) {
                fclose(file);
                return -1;
            }
            float lane_length;
            if (fread(&lane_length, sizeof(float), 1, file) != 1) {
                fclose(file);
                return -1;
            }
            if (slen > 0 && fseek(file, slen * sizeof(float), SEEK_CUR) != 0) {
                fclose(file);
                return -1;
            }
        } else {
            road->num_entries = 0;
            road->num_exits = 0;
            road->entry_lanes = NULL;
            road->exit_lanes = NULL;
            road->speed_limit = 0.0f;
        }
    }

    for (int i = 0; i < num_traffic; i++) {
        TrafficControlElement *traffic = &drive->traffic_elements[i];
        int traffic_id;

        if (fread(&traffic_id, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (traffic_id != i) {
            printf("[ERROR] -> Traffic element id %d != idx %d. Binary must be reindexed (id == idx).\n", traffic_id,
                   i);
            fclose(file);
            return -1;
        }
        if (fread(&traffic->type, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(traffic->stop_line, sizeof(float), 6, file) != 6) {
            fclose(file);
            return -1;
        }
        if (fread(&traffic->heading, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&traffic->state_length, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        int state_len = traffic->state_length;

        traffic->states = (int *)malloc(state_len * sizeof(int));
        if ((size_t)state_len > 0 && fread(traffic->states, sizeof(int), state_len, file) != (size_t)state_len) {
            fclose(file);
            return -1;
        }

        if (fread(&traffic->num_controlled_lanes, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (traffic->num_controlled_lanes > 0) {
            traffic->controlled_lanes = (int *)malloc(traffic->num_controlled_lanes * sizeof(int));
            if (fread(traffic->controlled_lanes, sizeof(int), traffic->num_controlled_lanes, file) !=
                (size_t)traffic->num_controlled_lanes) {
                fclose(file);
                return -1;
            }
        } else {
            traffic->controlled_lanes = NULL;
        }
    }

    // Skip objects section
    for (int i = 0; i < num_objects; i++) {
        int obj_id, obj_type, T;
        if (fread(&obj_id, sizeof(int), 1, file) != 1 || fread(&obj_type, sizeof(int), 1, file) != 1 ||
            fread(&T, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        // Skip: x,y,z,heading,vx,vy,length,width,height (9 float arrays) + valid (1 int array)
        fseek(file, 9 * T * sizeof(float) + T * sizeof(int), SEEK_CUR);
    }

    // Lane graph section
    int n_lanes_graph;
    if (fread(&n_lanes_graph, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    drive->lane_graph.n_lanes = n_lanes_graph;
    drive->lane_graph.lane_ids = NULL;
    drive->lane_graph.lane_lengths = NULL;
    drive->lane_graph.distances = NULL;
    if (n_lanes_graph > 0) {
        drive->lane_graph.lane_ids = (int *)malloc(n_lanes_graph * sizeof(int));
        if (fread(drive->lane_graph.lane_ids, sizeof(int), n_lanes_graph, file) != (size_t)n_lanes_graph) {
            fclose(file);
            return -1;
        }
        drive->lane_graph.distances = (float *)malloc(n_lanes_graph * n_lanes_graph * sizeof(float));
        if (fread(drive->lane_graph.distances, sizeof(float), n_lanes_graph * n_lanes_graph, file) !=
            (size_t)(n_lanes_graph * n_lanes_graph)) {
            fclose(file);
            return -1;
        }
    }

    // Metadata
    if (fread(drive->scenario_id, sizeof(char), 128, file) != 128) {
        fclose(file);
        return -1;
    }
    if (fread(drive->dataset_name, sizeof(char), 32, file) != 32) {
        fclose(file);
        return -1;
    }
    if (fread(&drive->log_length, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&drive->log_dt, sizeof(float), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&drive->num_objects_of_interest, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    if (drive->num_objects_of_interest > 0) {
        drive->objects_of_interest = (int *)malloc(drive->num_objects_of_interest * sizeof(int));
        if (fread(drive->objects_of_interest, sizeof(int), drive->num_objects_of_interest, file) !=
            (size_t)drive->num_objects_of_interest) {
            fclose(file);
            return -1;
        }
    } else {
        drive->objects_of_interest = NULL;
    }

    if (fread(&drive->num_tracks_to_predict, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    if (drive->num_tracks_to_predict > 0) {
        drive->tracks_to_predict = (int *)malloc(drive->num_tracks_to_predict * sizeof(int));
        if (fread(drive->tracks_to_predict, sizeof(int), drive->num_tracks_to_predict, file) !=
            (size_t)drive->num_tracks_to_predict) {
            fclose(file);
            return -1;
        }
    } else {
        drive->tracks_to_predict = NULL;
    }

    fclose(file);
    return 0;
}

// ========================================
// Road Utility Functions
// ========================================

// Compute multi-segment average heading around a center segment
static float compute_multi_segment_alignment(RoadMapElement *element, int center_seg_idx) {
    // NOTE: This function returns the average heading in radians for a lane segment,
    // with more weight given to the center segment.

    float avg_heading = 0.0f;
    float total_weight = 0.0f;

    int start = (center_seg_idx > 0) ? (center_seg_idx - 1) : center_seg_idx;
    int end = (center_seg_idx < element->segment_length - 2) ? (center_seg_idx + 1) : (element->segment_length - 2);

    for (int seg_idx = start; seg_idx <= end; seg_idx++) {
        if (seg_idx < 0 || seg_idx >= element->segment_length - 1)
            continue;

        float seg_heading = element->headings[seg_idx];

        float weight = (seg_idx == center_seg_idx) ? 2.0f : 1.0f;

        if (total_weight == 0.0f) {
            avg_heading = seg_heading;
        } else {
            float angle_diff = compute_heading_diff(seg_heading, avg_heading);
            avg_heading += weight * angle_diff / (total_weight + weight);
        }
        total_weight += weight;
    }

    return avg_heading;
}

// Compute the length of a lane
static float compute_lane_length(RoadMapElement *lane) {
    float length = 0.0f;
    for (int i = 1; i < lane->segment_length; i++) {
        float dx = lane->x[i] - lane->x[i - 1];
        float dy = lane->y[i] - lane->y[i - 1];
        length += sqrtf(dx * dx + dy * dy);
    }
    return length;
}

// Compute the remaining distance on a lane from a given position to the end of the lane
static float compute_remaining_lane_distance(RoadMapElement *lane, float pos_x, float pos_y) {
    // Find the closest segment to the position
    int closest_seg = 0;
    float closest_t = 0.0f;
    float min_dist_sq = 1e30f;

    for (int i = 0; i < lane->segment_length - 1; i++) {
        float x0 = lane->x[i];
        float y0 = lane->y[i];
        float x1 = lane->x[i + 1];
        float y1 = lane->y[i + 1];

        float dx = x1 - x0;
        float dy = y1 - y0;
        float seg_len_sq = dx * dx + dy * dy;

        float t = 0.0f;
        if (seg_len_sq > 1e-6f) {
            t = ((pos_x - x0) * dx + (pos_y - y0) * dy) / seg_len_sq;
            t = fmaxf(0.0f, fminf(1.0f, t));
        }

        float proj_x = x0 + t * dx;
        float proj_y = y0 + t * dy;
        float dist_sq = (pos_x - proj_x) * (pos_x - proj_x) + (pos_y - proj_y) * (pos_y - proj_y);

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_seg = i;
            closest_t = t;
        }
    }

    // Compute remaining distance from closest point to end of lane
    float remaining = 0.0f;

    // Partial distance in current segment (from t to end of segment)
    float dx = lane->x[closest_seg + 1] - lane->x[closest_seg];
    float dy = lane->y[closest_seg + 1] - lane->y[closest_seg];
    float seg_len = sqrtf(dx * dx + dy * dy);
    remaining += (1.0f - closest_t) * seg_len;

    // Full distance of remaining segments
    for (int i = closest_seg + 1; i < lane->segment_length - 1; i++) {
        dx = lane->x[i + 1] - lane->x[i];
        dy = lane->y[i + 1] - lane->y[i];
        remaining += sqrtf(dx * dx + dy * dy);
    }

    return remaining;
}

static float compute_lane_end_distance_sq(RoadMapElement *lane, float origin_x, float origin_y) {
    if (lane->segment_length <= 0) {
        return 0.0f;
    }

    int last_idx = lane->segment_length - 1;
    float dx = lane->x[last_idx] - origin_x;
    float dy = lane->y[last_idx] - origin_y;
    return dx * dx + dy * dy;
}

// Returns signed distance to lane (left of lane = negative, right = positive)
static float find_closest_segment_on_lane(RoadMapElement *lane, float agent_x, float agent_y, int *out_segment_idx) {
    int num_segments = lane->segment_length - 1;
    if (num_segments < 1) {
        *out_segment_idx = 0;
        return 1e9f;
    }

    float min_dist_sq = 1e18f;
    int closest_idx = 0;
    float closest_cross = 0.0f;

    for (int seg_idx = 0; seg_idx < num_segments; seg_idx++) {
        float seg_start_x = lane->x[seg_idx];
        float seg_start_y = lane->y[seg_idx];
        float seg_end_x = lane->x[seg_idx + 1];
        float seg_end_y = lane->y[seg_idx + 1];

        float seg_dx = seg_end_x - seg_start_x;
        float seg_dy = seg_end_y - seg_start_y;
        float seg_length_sq = seg_dx * seg_dx + seg_dy * seg_dy;

        float to_agent_x = agent_x - seg_start_x;
        float to_agent_y = agent_y - seg_start_y;

        // cross > 0 means agent is left of lane direction
        float cross = seg_dx * to_agent_y - seg_dy * to_agent_x;

        float dist_sq;
        if (seg_length_sq > 1e-6f) {
            float t = (to_agent_x * seg_dx + to_agent_y * seg_dy) / seg_length_sq;
            if (t <= 0.0f) {
                dist_sq = to_agent_x * to_agent_x + to_agent_y * to_agent_y;
            } else if (t >= 1.0f) {
                float dx = agent_x - seg_end_x;
                float dy = agent_y - seg_end_y;
                dist_sq = dx * dx + dy * dy;
            } else {
                dist_sq = (cross * cross) / seg_length_sq;
            }
        } else {
            dist_sq = to_agent_x * to_agent_x + to_agent_y * to_agent_y;
        }

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_idx = seg_idx;
            closest_cross = cross;
        }
    }

    *out_segment_idx = closest_idx;
    float abs_dist = sqrtf(min_dist_sq);
    return (closest_cross >= 0.0f) ? -abs_dist : abs_dist;
}

static float compute_progression(Agent *agent) {
    int num_wp = agent->path->num_waypoints;
    if (num_wp < 2)
        return agent->path->waypoints[0].s;

    int idx = agent->closest_path_idx_wp;
    if (idx >= num_wp)
        idx = num_wp - 1;

    // closest_path_idx_wp is the closest waypoint IN FRONT of the agent,
    // so the agent is on the segment ending at that waypoint (idx-1 to idx).
    // Also check the next segment (idx to idx+1) in case agent is at the waypoint.
    int start_seg = (idx > 0) ? (idx - 1) : 0;
    int end_seg = (idx < num_wp - 1) ? (idx + 1) : (num_wp - 1);

    float min_dist_sq = 1e18f;
    float best_s = agent->path->waypoints[0].s;

    for (int i = start_seg; i < end_seg; ++i) {
        const struct Waypoint wp_a = agent->path->waypoints[i];
        const struct Waypoint wp_b = agent->path->waypoints[i + 1];

        const float seg_dx = wp_b.x - wp_a.x;
        const float seg_dy = wp_b.y - wp_a.y;
        const float seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy;

        float t;
        if (seg_len_sq < 1e-6f) {
            t = 0.0f;
        } else {
            const float agent_dx = agent->sim_x - wp_a.x;
            const float agent_dy = agent->sim_y - wp_a.y;
            t = (agent_dx * seg_dx + agent_dy * seg_dy) / seg_len_sq;
        }

        const float clamped_t = fmaxf(0.0f, fminf(1.0f, t));
        const float closest_x = wp_a.x + clamped_t * seg_dx;
        const float closest_y = wp_a.y + clamped_t * seg_dy;
        const float dist_sq = (agent->sim_x - closest_x) * (agent->sim_x - closest_x) +
                              (agent->sim_y - closest_y) * (agent->sim_y - closest_y);

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_s = wp_a.s + clamped_t * (wp_b.s - wp_a.s);
        }
    }

    return best_s;
}

static float score_lane_candidate(Drive *env, float agent_heading, int *checked_lanes, int *num_checked_lanes,
                                  int *closest_segment_index, GridMapEntity entity, float agent_x, float agent_y,
                                  float max_distance_threshold, int current_lane_idx, float *signed_dist_out,
                                  float *lane_heading_out) {
    if (entity.entity_idx == -1 || entity.entity_type != ENTITY_TYPE_ROAD_ELEMENT)
        return -1;

    int entity_idx = entity.entity_idx;
    RoadMapElement *element = &env->road_elements[entity_idx];

    if (!is_drivable_road_lane(element->type))
        return -1;

    int already_checked = 0;
    for (int c = 0; c < *num_checked_lanes; c++) {
        if (checked_lanes[c] == entity_idx) {
            already_checked = 1;
            break;
        }
    }
    if (already_checked)
        return -1;
    if (*num_checked_lanes < MAX_CHECKED_LANES) {
        checked_lanes[*num_checked_lanes] = entity_idx;
        (*num_checked_lanes)++;
    }

    // Find closest segment on this lane
    float signed_dist = find_closest_segment_on_lane(element, agent_x, agent_y, closest_segment_index);
    float abs_dist = fabsf(signed_dist);

    if (abs_dist > max_distance_threshold)
        return -1;

    // Compute lane heading using multi-segment alignment
    float avg_lane_heading = compute_multi_segment_alignment(element, *closest_segment_index);

    // Compute heading alignment penalty (0.0 = perfect, 1.0 = opposite)
    float heading_diff = compute_heading_diff(agent_heading, avg_lane_heading);
    float heading_penalty = fabsf(heading_diff) / M_PI; // Normalize to [0, 1]

    // Normalize distance for scoring
    float distance_penalty = abs_dist / LANE_DISTANCE_NORMALIZATION;

    // Combined score using defined weights
    float score = LANE_SELECTION_DISTANCE_WEIGHT * distance_penalty + LANE_SELECTION_HEADING_WEIGHT * heading_penalty;

    // Hysteresis: penalize switching away from current lane
    if (current_lane_idx != entity_idx && current_lane_idx != -1) {
        score += LANE_SWITCH_THRESHOLD;
    }

    *signed_dist_out = signed_dist;
    *lane_heading_out = avg_lane_heading;
    return score;
}

static DepthPoint compute_z_distance_to_road_segment(const Agent *agent, const RoadMapElement *lane, int geometry_idx) {
    float dx = agent->sim_x - lane->x[geometry_idx];
    float dy = agent->sim_y - lane->y[geometry_idx];
    float dz = agent->sim_z - lane->z[geometry_idx];

    DepthPoint point;
    point.z_dis = fabsf(dz);
    point.euclidean_dis = sqrtf(dx * dx + dy * dy + dz * dz);
    point.z = lane->z[geometry_idx];
    return point;
}

// ========================================
// Route/Path/Goal Functions
// ========================================

static int get_closest_waypoint_index_on_path(Drive *env, int agent_idx) {
    const float MAX_DIST_SQ = 10000.0f;
    const float MAX_ANGLE_DIFF = M_PI_2; // 90 degrees
    const int WINDOW_FORWARD = 20;       // 40m ahead (20 * 2m waypoint spacing)
    const int WINDOW_BACKWARD = 10;      // 20m behind

    Agent *agent = &env->agents[agent_idx];
    if (agent->path == NULL || agent->path->num_waypoints == 0)
        return 0;

    int num_wp = agent->path->num_waypoints;
    int prev_idx = agent->closest_path_idx_wp;
    if (prev_idx >= num_wp)
        prev_idx = 0;

    float heading_x = cosf(agent->sim_heading);
    float heading_y = sinf(agent->sim_heading);

    int best_idx = 0;
    float min_dist_sq = MAX_DIST_SQ;

    // Try windowed search first, fallback to full search if no candidate found
    int start_idx = fmaxf(0, prev_idx - WINDOW_BACKWARD);
    int end_idx = fminf(num_wp, prev_idx + WINDOW_FORWARD);

    for (int pass = 0; pass < 2; pass++) {
        for (int i = start_idx; i < end_idx; i++) {
            float dx = agent->path->waypoints[i].x - agent->sim_x;
            float dy = agent->path->waypoints[i].y - agent->sim_y;

            // Skip waypoints behind agent
            if (dx * heading_x + dy * heading_y <= 0)
                continue;

            // Skip waypoints with heading too different from agent
            float angle_diff = agent->sim_heading - agent->path->waypoints[i].heading;
            angle_diff = atan2f(sinf(angle_diff), cosf(angle_diff));

            if (fabsf(angle_diff) > MAX_ANGLE_DIFF)
                continue;

            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_idx = i;
            }
        }

        // If found in windowed search, done
        if (min_dist_sq < MAX_DIST_SQ)
            break;

        // Fallback: full search
        start_idx = 0;
        end_idx = num_wp;
    }

    return best_idx;
}

static inline void initialize_agent_progression(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    if (agent->path == NULL || agent->path->num_waypoints == 0) {
        agent->closest_path_idx_wp = 0;
        agent->path_progression = 0.0f;
        agent->distance_since_spawn = 0.0f;
        return;
    }

    agent->closest_path_idx_wp = get_closest_waypoint_index_on_path(env, agent_idx);
    float baseline_progression = compute_progression(agent);
    agent->path_progression = baseline_progression;
    agent->distance_since_spawn = 0.0f;
}

static inline void reset_agent_path_progression(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->path == NULL || agent->path->num_waypoints == 0) {
        agent->closest_path_idx_wp = 0;
        agent->path_progression = 0.0f;
        return;
    }

    agent->closest_path_idx_wp = get_closest_waypoint_index_on_path(env, agent_idx);
    agent->path_progression = compute_progression(agent);
}

static void build_path(Drive *env, int agent_idx) {
    // NOTE: This function assumes the agent's route is already set.
    // It interpolates waypoints along the route lanes at fixed spacing.
    // It is a mid-level representation between route and low-level goals waypoints.

    float waypoints_spacing = env->min_waypoint_spacing;

    Agent *agent = &env->agents[agent_idx];

    if (agent->path != NULL)
        free(agent->path);
    agent->path = (struct Path *)malloc(sizeof(struct Path));

    int wp_count = 0;
    float prev_x, prev_y, prev_z, prev_s;

    // Interpolate waypoints along route lanes
    for (int route_idx = 0; route_idx < agent->route_length && wp_count < MAX_NUM_WP_PATH; route_idx++) {
        int lane_idx = agent->route[route_idx];
        RoadMapElement *lane = &env->road_elements[lane_idx];

        for (int i = 0; i < lane->segment_length && wp_count < MAX_NUM_WP_PATH; i++) {
            float curr_x = lane->x[i];
            float curr_y = lane->y[i];
            float curr_z = lane->z[i];

            // First point: add directly
            if (wp_count == 0) {
                agent->path->waypoints[0] =
                    (struct Waypoint){.x = curr_x, .y = curr_y, .z = curr_z, .s = 0.0f, .lane_idx = lane_idx};
                prev_x = curr_x;
                prev_y = curr_y;
                prev_z = curr_z;
                prev_s = 0.0f;
                wp_count++;
                continue;
            }

            float dx = curr_x - prev_x;
            float dy = curr_y - prev_y;
            float dz = curr_z - prev_z;
            float seg_len = sqrtf(dx * dx + dy * dy + dz * dz);
            if (seg_len < 1e-6f) {
                prev_x = curr_x;
                prev_y = curr_y;
                prev_z = curr_z;
                continue;
            }

            float curr_s = prev_s + seg_len;

            // Interpolate waypoints within this segment
            float target_s = (float)wp_count * waypoints_spacing;
            while (target_s < curr_s && wp_count < MAX_NUM_WP_PATH) {
                float t = (target_s - prev_s) / seg_len;
                agent->path->waypoints[wp_count] = (struct Waypoint){
                    .x = prev_x + t * dx,
                    .y = prev_y + t * dy,
                    .z = prev_z + t * dz,
                    .s = target_s,
                    .lane_idx = lane_idx,
                };
                wp_count++;
                target_s = (float)wp_count * waypoints_spacing;
            }

            prev_x = curr_x;
            prev_y = curr_y;
            prev_z = curr_z;
            prev_s = curr_s;
        }
    }

    agent->path->num_waypoints = wp_count;
    if (wp_count < 2)
        return;

    // Compute heading (tangent angle) and cache trig values
    for (int i = 0; i < wp_count - 1; i++) {
        float dx = agent->path->waypoints[i + 1].x - agent->path->waypoints[i].x;
        float dy = agent->path->waypoints[i + 1].y - agent->path->waypoints[i].y;
        float heading = atan2f(dy, dx);
        agent->path->waypoints[i].heading = heading;
        agent->path->waypoints[i].cos_heading = cosf(heading);
        agent->path->waypoints[i].sin_heading = sinf(heading);
    }
    // Last waypoint copies from second-to-last
    agent->path->waypoints[wp_count - 1].heading = agent->path->waypoints[wp_count - 2].heading;
    agent->path->waypoints[wp_count - 1].cos_heading = agent->path->waypoints[wp_count - 2].cos_heading;
    agent->path->waypoints[wp_count - 1].sin_heading = agent->path->waypoints[wp_count - 2].sin_heading;

    // Compute kappa (curvature = d_heading / ds)
    for (int i = 1; i < wp_count - 1; i++) {
        float d_heading =
            compute_heading_diff(agent->path->waypoints[i + 1].heading, agent->path->waypoints[i].heading);
        agent->path->waypoints[i].kappa = d_heading / waypoints_spacing;
    }
    agent->path->waypoints[0].kappa = agent->path->waypoints[1].kappa;
    agent->path->waypoints[wp_count - 1].kappa = agent->path->waypoints[wp_count - 2].kappa;
}

// Generate a route by random walk through lane graph until target distance is reached
static int generate_random_route(Drive *env, int start_lane_idx, float target_distance, int *route,
                                 int max_route_length, float agent_x, float agent_y) {
    // NOTE: This function performs a random walk through the lane connectivity graph,starting from start_lane_idx,
    // until the accumulated distance exceeds target_distance or max_route_length is reached.
    // Cycles are avoided by tracking visited lanes and trying to increase distance.

    int route_length = 0;
    float accumulated_distance = 0.0f;
    int current_lane_idx = start_lane_idx;
    float start_x = agent_x;
    float start_y = agent_y;

    // Track visited lanes to avoid loops
    int visited_ids[MAX_ROUTE_LENGTH];
    int visited_count = 0;

    // Add start lane to route
    route[route_length++] = current_lane_idx;
    visited_ids[visited_count++] = current_lane_idx;

    RoadMapElement *current_lane = &env->road_elements[current_lane_idx];
    // Use remaining distance from agent position instead of full lane length
    accumulated_distance += compute_remaining_lane_distance(current_lane, agent_x, agent_y);
    float max_end_distance_sq = compute_lane_end_distance_sq(current_lane, start_x, start_y);

    // Random walk through lane graph
    while (accumulated_distance < target_distance && route_length < max_route_length) {
        if (current_lane_idx == -1)
            break;

        current_lane = &env->road_elements[current_lane_idx];

        // Collect valid (unvisited, drivable) exit lanes
        int valid_exits[8];
        float valid_exit_dist_sq[8];
        int num_valid_exits = 0;

        int progressing_exits[8];
        float progressing_dist_sq[8];
        int num_progressing_exits = 0;

        for (int e = 0; e < current_lane->num_exits && num_valid_exits < 8; e++) {
            int exit_lane_idx = current_lane->exit_lanes[e];

            // Check if already visited
            int already_visited = 0;
            for (int v = 0; v < visited_count; v++) {
                if (visited_ids[v] == exit_lane_idx) {
                    already_visited = 1;
                    break;
                }
            }
            if (already_visited)
                continue;

            // Check if exit lane is drivable
            if (exit_lane_idx == -1)
                continue;

            // NOTE: Dummy logic to prevent cycle, can be improved with better graph traversal
            float exit_end_distance_sq =
                compute_lane_end_distance_sq(&env->road_elements[exit_lane_idx], start_x, start_y);
            valid_exits[num_valid_exits] = exit_lane_idx;
            valid_exit_dist_sq[num_valid_exits] = exit_end_distance_sq;
            num_valid_exits++;

            if (exit_end_distance_sq > max_end_distance_sq) {
                progressing_exits[num_progressing_exits] = exit_lane_idx;
                progressing_dist_sq[num_progressing_exits] = exit_end_distance_sq;
                num_progressing_exits++;
            }
        }

        // If no valid exits, we've reached a dead end
        if (num_valid_exits == 0)
            break;

        // Pick a progressing exit lane if possible, otherwise pick the farthest available
        int chosen_exit_idx;
        float chosen_exit_dist_sq;
        if (num_progressing_exits > 0) {
            int chosen_idx = rand() % num_progressing_exits;
            chosen_exit_idx = progressing_exits[chosen_idx];
            chosen_exit_dist_sq = progressing_dist_sq[chosen_idx];
        } else {
            int best_idx = 0;
            float best_dist_sq = valid_exit_dist_sq[0];
            for (int i = 1; i < num_valid_exits; i++) {
                if (valid_exit_dist_sq[i] > best_dist_sq) {
                    best_dist_sq = valid_exit_dist_sq[i];
                    best_idx = i;
                }
            }
            chosen_exit_idx = valid_exits[best_idx];
            chosen_exit_dist_sq = valid_exit_dist_sq[best_idx];
        }

        if (chosen_exit_idx == -1)
            break;

        // Add to route
        route[route_length++] = chosen_exit_idx;
        visited_ids[visited_count++] = chosen_exit_idx;

        // Accumulate distance
        RoadMapElement *exit_lane = &env->road_elements[chosen_exit_idx];
        accumulated_distance += compute_lane_length(exit_lane);
        if (chosen_exit_dist_sq > max_end_distance_sq) {
            max_end_distance_sq = chosen_exit_dist_sq;
        }

        // Move to next lane
        current_lane_idx = chosen_exit_idx;
    }

    return route_length;
}

// NOTE: Only works for closed maps with infinite looping routes
static int compute_new_route(Drive *env, int agent_idx, int current_lane_idx) {
    Agent *agent = &env->agents[agent_idx];

    // Generate route by random walk through lane graph
    // Use agent's current position to compute remaining distance on start lane
    int num_target_waypoints = env->num_target_waypoints;
    if (num_target_waypoints > MAX_TARGET_WAYPOINTS) {
        num_target_waypoints = MAX_TARGET_WAYPOINTS;
    }

    float min_route_distance;
    // NOTE: make both multipliers config values and tune from a metric (route regenerations per 1k env steps).
    if (env->target_type == TARGET_STATIC) {
        min_route_distance = env->max_waypoint_spacing * num_target_waypoints * 2.0f;
    } else {
        min_route_distance = env->min_waypoint_spacing * num_target_waypoints * 20.0f;
    }

    int temp_route[MAX_ROUTE_LENGTH];
    int route_length = generate_random_route(env, current_lane_idx, min_route_distance, temp_route, MAX_ROUTE_LENGTH,
                                             agent->sim_x, agent->sim_y);

    if (route_length == 0) {
        printf("[GIGAFLOW WARNING] -> Failed to generate route for agent %d\n", agent_idx);
        agent->removed = 1;
        return 0;
    }

    // Free old route and allocate new one
    if (agent->route != NULL)
        free(agent->route);

    agent->route_length = route_length;
    agent->route = (int *)malloc(route_length * sizeof(int));

    for (int i = 0; i < route_length; i++) {
        agent->route[i] = temp_route[i];
    }

    agent->current_route_index = 0;

    // Update path
    build_path(env, agent_idx);
    agent->closest_path_idx_wp = 0; // Reset before search (old index invalid on new path)
    reset_agent_path_progression(env, agent_idx);

    return 1; // Success
}

static int compute_goals(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    struct Path *path = agent->path;

    // Validate path exists
    if (path == NULL || path->num_waypoints == 0) {
        printf("[GIGAFLOW WARNING] -> Agent %d has no valid path\n", agent_idx);
        agent->removed = 1;
        return 0;
    }

    int num_target_waypoints = env->num_target_waypoints;
    // Validate waypoint count in bounds
    if (num_target_waypoints <= 0 || num_target_waypoints > MAX_TARGET_WAYPOINTS) {
        num_target_waypoints = MAX_TARGET_WAYPOINTS;
    }

    float goal_spacings[MAX_TARGET_WAYPOINTS];

    // Iterative replacement for former recursion (bounded at 4 retries)
    for (int iter = 0; iter <= 4; iter++) {
        float total_spacing = 0.0f;
        for (int i = 0; i < num_target_waypoints; i++) {
            goal_spacings[i] = random_uniform(env->min_waypoint_spacing, env->max_waypoint_spacing);
            total_spacing += goal_spacings[i];
        }

        // On iter 3, reset to path start to escape a short-path cycle
        int base_idx = (iter == 3) ? 0 : get_closest_waypoint_index_on_path(env, agent_idx);
        float base_s = path->waypoints[base_idx].s;
        float needed_s = base_s + total_spacing;
        float path_end_s = path->waypoints[path->num_waypoints - 1].s;

        // If we reached the end of the current path, compute a new route and retry.
        // Bounded by iter <= 4 to prevent infinite loops on degenerate maps.
        if (needed_s >= path_end_s) {
            if (iter > 3) {
                printf("[GIGAFLOW WARNING] -> Max iterations in compute_goals for agent %d\n", agent_idx);
                agent->removed = 1;
                return 0;
            }
            if (env->simulation_mode == SIMULATION_GIGAFLOW) {
                int route_ok = compute_new_route(env, agent_idx, path->waypoints[base_idx].lane_idx);
                if (route_ok == 0) {
                    agent->removed = 1;
                    return 0;
                }
                path = agent->path;
                continue;
            }
        }

        // Place N goals along the path at random spacing intervals from current position
        float cumulative_spacing = 0.0f;
        for (int i = 0; i < num_target_waypoints; i++) {
            cumulative_spacing += goal_spacings[i];
            float target_s = base_s + cumulative_spacing;
            // Find waypoint at or past target_s
            int wp_idx = path->num_waypoints - 1;
            for (int j = base_idx + 1; j < path->num_waypoints; j++) {
                if (path->waypoints[j].s >= target_s) {
                    wp_idx = j;
                    break;
                }
            }
            agent->goal_positions_x[i] = path->waypoints[wp_idx].x;
            agent->goal_positions_y[i] = path->waypoints[wp_idx].y;
            agent->goal_positions_z[i] = path->waypoints[wp_idx].z;
        }

        // Reset goal index and update alias
        agent->current_goal_idx = 0;
        agent->goal_position_x = agent->goal_positions_x[0];
        agent->goal_position_y = agent->goal_positions_y[0];
        agent->goal_position_z = agent->goal_positions_z[0];
        return 1;
    }

    printf("[GIGAFLOW ERROR] -> Failed to compute goals for agent %d after multiple attempts\n", agent_idx);
    agent->removed = 1;
    return 0;
}

// ========================================
// Metrics/Collision Functions
// ========================================

static float compute_displacement_error(Agent *agent, int timestep) {
    // Check if timestep is within valid range
    if (timestep < 0 || timestep >= agent->trajectory_length) {
        return 0.0f;
    }

    // Check if reference trajectory is valid at this timestep
    if (!agent->log_valid[timestep]) {
        return 0.0f;
    }

    // Get reference position from logged trajectory at current timestep
    float ref_x = agent->log_trajectory_x[timestep];
    float ref_y = agent->log_trajectory_y[timestep];

    if (ref_x == INVALID_POSITION || ref_y == INVALID_POSITION) {
        return 0.0f;
    }

    // Compute deltas: Euclidean distance between simulated and reference position
    float dx = agent->sim_x - ref_x;
    float dy = agent->sim_y - ref_y;
    float displacement = sqrtf(dx * dx + dy * dy);

    return displacement;
}

static bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmaxf(p1[0], p2[0]) < fminf(q1[0], q2[0]) || fminf(p1[0], p2[0]) > fmaxf(q1[0], q2[0]) ||
        fmaxf(p1[1], p2[1]) < fminf(q1[1], q2[1]) || fminf(p1[1], p2[1]) > fmaxf(q1[1], q2[1]))
        return false;

    // Calculate vectors
    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];

    // Calculate cross products
    float cross = dx1 * dy2 - dy1 * dx2;

    // If lines are parallel
    if (cross == 0)
        return false;

    // Calculate relative vectors between start points
    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];

    // Calculate parameters for intersection point
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;

    // Check if intersection point lies within both line segments
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

static void compute_agent_corners(const Agent *agent, float corners[4][2]);

static bool check_stop_line_crossing(Drive *env, Agent *agent, int current_lane_idx, float corners[4][2]) {
    float agent_x = agent->sim_x;
    float agent_y = agent->sim_y;

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];

        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT)
            continue;
        if (traffic->num_controlled_lanes == 0)
            continue;

        int controls_lane = 0;
        for (int j = 0; j < traffic->num_controlled_lanes; j++) {
            if (traffic->controlled_lanes[j] == current_lane_idx) {
                controls_lane = 1;
                break;
            }
        }
        if (!controls_lane)
            continue;
        if (env->timestep >= traffic->state_length)
            continue;
        if (traffic->states[env->timestep] != TRAFFIC_CONTROL_STATE_RED)
            continue;

        // Pre-filter: distance to stop line midpoint
        float mid_x = (traffic->stop_line[0] + traffic->stop_line[3]) * 0.5f;
        float mid_y = (traffic->stop_line[1] + traffic->stop_line[4]) * 0.5f;
        float dx = agent_x - mid_x;
        float dy = agent_y - mid_y;
        if (dx * dx + dy * dy > TRAFFIC_LIGHT_DISTANCE_THRESHOLD * TRAFFIC_LIGHT_DISTANCE_THRESHOLD)
            continue;

        // Heading check: agent must be heading towards the stop line
        float heading_diff = compute_heading_diff(agent->sim_heading, traffic->heading);
        if (fabsf(heading_diff) > RED_LIGHT_HEADING_THRESHOLD)
            continue;

        // Extend stop line endpoints by STOP_LINE_EXTENSION_FACTOR
        float sl_dx = traffic->stop_line[3] - traffic->stop_line[0];
        float sl_dy = traffic->stop_line[4] - traffic->stop_line[1];
        float ext = (STOP_LINE_EXTENSION_FACTOR - 1.0f) * 0.5f;
        float ext_p1[2] = {traffic->stop_line[0] - ext * sl_dx, traffic->stop_line[1] - ext * sl_dy};
        float ext_p2[2] = {traffic->stop_line[3] + ext * sl_dx, traffic->stop_line[4] + ext * sl_dy};

        // Check front + side edges vs extended stop line (skip back edge k=2)
        for (int k = 0; k < 4; k++) {
            if (k == 2)
                continue;
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], ext_p1, ext_p2))
                return true;
        }
    }
    return false;
}

static bool check_lane_change_red_light(Drive *env, Agent *agent) {
    if (agent->previous_lane_idx == agent->current_lane_idx)
        return false;
    if (agent->previous_lane_idx == -1 || agent->current_lane_idx == -1)
        return false;

    float agent_x = agent->sim_x;
    float agent_y = agent->sim_y;

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];

        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT)
            continue;
        if (traffic->num_controlled_lanes == 0)
            continue;
        if (env->timestep >= traffic->state_length)
            continue;
        if (traffic->states[env->timestep] != TRAFFIC_CONTROL_STATE_RED)
            continue;

        for (int j = 0; j < traffic->num_controlled_lanes; j++) {
            if (traffic->controlled_lanes[j] != agent->current_lane_idx)
                continue;

            float mid_x = (traffic->stop_line[0] + traffic->stop_line[3]) * 0.5f;
            float mid_y = (traffic->stop_line[1] + traffic->stop_line[4]) * 0.5f;
            float dx = agent_x - mid_x;
            float dy = agent_y - mid_y;
            if (dx * dx + dy * dy > TRAFFIC_LIGHT_DISTANCE_THRESHOLD * TRAFFIC_LIGHT_DISTANCE_THRESHOLD)
                continue;

            return true;
        }
    }
    return false;
}

static bool check_red_light_violation(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    int current_lane_idx = agent->current_lane_idx;

    if (current_lane_idx == -1)
        return false;

    float corners[4][2];
    compute_agent_corners(agent, corners);

    if (check_stop_line_crossing(env, agent, current_lane_idx, corners))
        return true;

    if (check_lane_change_red_light(env, agent))
        return true;

    return false;
}

static void compute_agent_corners(const Agent *agent, float corners[4][2]) {
    float half_length = agent->sim_length * 0.5f;
    float half_width = agent->sim_width * 0.5f;
    float cos_heading = agent->cos_heading;
    float sin_heading = agent->sin_heading;

    // Front-right, front-left, rear-left, rear-right.
    corners[0][0] = agent->sim_x + half_length * cos_heading - half_width * sin_heading;
    corners[0][1] = agent->sim_y + half_length * sin_heading + half_width * cos_heading;
    corners[1][0] = agent->sim_x + half_length * cos_heading + half_width * sin_heading;
    corners[1][1] = agent->sim_y + half_length * sin_heading - half_width * cos_heading;
    corners[2][0] = agent->sim_x - half_length * cos_heading + half_width * sin_heading;
    corners[2][1] = agent->sim_y - half_length * sin_heading - half_width * cos_heading;
    corners[3][0] = agent->sim_x - half_length * cos_heading - half_width * sin_heading;
    corners[3][1] = agent->sim_y - half_length * sin_heading + half_width * cos_heading;
}

static bool point_inside_agent_box(float x, float y, const Agent *agent) {
    float dx = x - agent->sim_x;
    float dy = y - agent->sim_y;
    float local_long = dx * agent->cos_heading + dy * agent->sin_heading;
    float local_lat = -dx * agent->sin_heading + dy * agent->cos_heading;
    return fabsf(local_long) <= 0.5f * agent->sim_length && fabsf(local_lat) <= 0.5f * agent->sim_width;
}

static bool segment_intersects_agent_box(float p1[2], float p2[2], const Agent *agent) {
    if (point_inside_agent_box(p1[0], p1[1], agent) || point_inside_agent_box(p2[0], p2[1], agent))
        return true;

    float corners[4][2];
    compute_agent_corners(agent, corners);
    for (int i = 0; i < 4; i++) {
        int next = (i + 1) % 4;
        if (check_line_intersection(p1, p2, corners[i], corners[next]))
            return true;
    }
    return false;
}

static bool front_bumper_intersects_agent(const Agent *ego, const Agent *other) {
    float corners[4][2];
    compute_agent_corners(ego, corners);
    return segment_intersects_agent_box(corners[0], corners[1], other);
}

// OBB collision via SAT (Separating Axis Theorem).
// Projects both boxes onto 4 axes (2 per car) and checks for overlap on all axes.
// When intersecting, returns the minimum-penetration axis oriented from car1 toward car2.
static bool check_obb_collision_with_normal(Agent *car1, Agent *car2, float *normal_x, float *normal_y,
                                            float *penetration_depth) {
    float car1_corners[4][2];
    float car2_corners[4][2];
    compute_agent_corners(car1, car1_corners);
    compute_agent_corners(car2, car2_corners);

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {car1->cos_heading, car1->sin_heading},  // Car1's length axis
        {-car1->sin_heading, car1->cos_heading}, // Car1's width axis
        {car2->cos_heading, car2->sin_heading},  // Car2's length axis
        {-car2->sin_heading, car2->cos_heading}, // Car2's width axis
    };

    float best_overlap = INFINITY;
    float best_axis_x = 0.0f;
    float best_axis_y = 0.0f;

    // Check each axis
    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;

        // Project car1's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj);
            max1 = fmaxf(max1, proj);
        }

        // Project car2's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj);
            max2 = fmaxf(max2, proj);
        }

        // If there's a gap on this axis, the boxes don't intersect
        if (max1 < min2 || min1 > max2) {
            return false; // No collision
        }

        float overlap = fminf(max1, max2) - fmaxf(min1, min2);
        if (overlap < best_overlap) {
            best_overlap = overlap;
            best_axis_x = axes[i][0];
            best_axis_y = axes[i][1];
        }
    }

    // If we get here, there's no separating axis, so the boxes intersect
    float center_dx = car2->sim_x - car1->sim_x;
    float center_dy = car2->sim_y - car1->sim_y;
    if (center_dx * best_axis_x + center_dy * best_axis_y < 0.0f) {
        best_axis_x = -best_axis_x;
        best_axis_y = -best_axis_y;
    }

    if (normal_x != NULL)
        *normal_x = best_axis_x;
    if (normal_y != NULL)
        *normal_y = best_axis_y;
    if (penetration_depth != NULL)
        *penetration_depth = best_overlap;
    return true;
}

static bool check_obb_collision(Agent *car1, Agent *car2) {
    return check_obb_collision_with_normal(car1, car2, NULL, NULL, NULL);
}

static bool check_z_collision_possibility(const Agent *car1, const Agent *car2) {
    float car1_bottom = car1->sim_z;
    float car1_top = car1->sim_z + car1->sim_height;
    float car2_bottom = car2->sim_z;
    float car2_top = car2->sim_z + car2->sim_height;

    return !(car1_top < car2_bottom || car2_top < car1_bottom);
}

static Agent interpolate_agent_pose_for_collision(const Agent *agent, float alpha) {
    Agent interpolated = *agent;
    float heading_delta = normalize_heading(agent->sim_heading - agent->prev_sim_heading);
    interpolated.sim_x = agent->prev_sim_x + alpha * (agent->sim_x - agent->prev_sim_x);
    interpolated.sim_y = agent->prev_sim_y + alpha * (agent->sim_y - agent->prev_sim_y);
    interpolated.sim_heading = normalize_heading(agent->prev_sim_heading + alpha * heading_delta);
    interpolated.cos_heading = cosf(interpolated.sim_heading);
    interpolated.sin_heading = sinf(interpolated.sim_heading);
    return interpolated;
}

static void project_agent_on_axis(const Agent *agent, float axis_x, float axis_y, float *min_out, float *max_out) {
    float corners[4][2];
    compute_agent_corners(agent, corners);

    float min_proj = INFINITY;
    float max_proj = -INFINITY;
    for (int i = 0; i < 4; i++) {
        float proj = corners[i][0] * axis_x + corners[i][1] * axis_y;
        min_proj = fminf(min_proj, proj);
        max_proj = fmaxf(max_proj, proj);
    }

    *min_out = min_proj;
    *max_out = max_proj;
}

static float interval_gap(float min_a, float max_a, float min_b, float max_b) {
    if (max_a < min_b)
        return min_b - max_a;
    if (max_b < min_a)
        return min_a - max_b;
    return -fminf(max_a, max_b) + fmaxf(min_a, min_b);
}

static float interval_overlap(float min_a, float max_a, float min_b, float max_b) {
    return fminf(max_a, max_b) - fmaxf(min_a, min_b);
}

static void orient_axis_from_agent_to_agent(const Agent *agent_a, const Agent *agent_b, float *axis_x, float *axis_y) {
    float center_dx = agent_b->sim_x - agent_a->sim_x;
    float center_dy = agent_b->sim_y - agent_a->sim_y;
    if (center_dx * (*axis_x) + center_dy * (*axis_y) < 0.0f) {
        *axis_x = -*axis_x;
        *axis_y = -*axis_y;
    }
}

static int compute_first_contact_collision_normal(Agent *agent_a, Agent *agent_b, float *normal_x, float *normal_y) {
    if (agent_a->prev_sim_x == INVALID_POSITION || agent_b->prev_sim_x == INVALID_POSITION)
        return 0;

    Agent prev_a = interpolate_agent_pose_for_collision(agent_a, 0.0f);
    Agent prev_b = interpolate_agent_pose_for_collision(agent_b, 0.0f);
    if (check_obb_collision(&prev_a, &prev_b))
        return 0;
    if (!check_obb_collision(agent_a, agent_b))
        return 0;

    float lo = 0.0f;
    float hi = 1.0f;
    for (int iter = 0; iter < 18; iter++) {
        float mid = 0.5f * (lo + hi);
        Agent mid_a = interpolate_agent_pose_for_collision(agent_a, mid);
        Agent mid_b = interpolate_agent_pose_for_collision(agent_b, mid);
        if (check_obb_collision(&mid_a, &mid_b)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Agent impact_a = interpolate_agent_pose_for_collision(agent_a, hi);
    Agent impact_b = interpolate_agent_pose_for_collision(agent_b, hi);
    float axes[4][2] = {
        {impact_a.cos_heading, impact_a.sin_heading},
        {-impact_a.sin_heading, impact_a.cos_heading},
        {impact_b.cos_heading, impact_b.sin_heading},
        {-impact_b.sin_heading, impact_b.cos_heading},
    };

    float best_axis_x = 0.0f;
    float best_axis_y = 0.0f;
    float best_entry_alpha = -INFINITY;
    float best_gap = INFINITY;
    int found = 0;

    for (int i = 0; i < 4; i++) {
        float axis_x = axes[i][0];
        float axis_y = axes[i][1];
        float prev_min_a, prev_max_a, prev_min_b, prev_max_b;
        float curr_min_a, curr_max_a, curr_min_b, curr_max_b;
        project_agent_on_axis(&prev_a, axis_x, axis_y, &prev_min_a, &prev_max_a);
        project_agent_on_axis(&prev_b, axis_x, axis_y, &prev_min_b, &prev_max_b);
        project_agent_on_axis(agent_a, axis_x, axis_y, &curr_min_a, &curr_max_a);
        project_agent_on_axis(agent_b, axis_x, axis_y, &curr_min_b, &curr_max_b);

        float prev_gap = interval_gap(prev_min_a, prev_max_a, prev_min_b, prev_max_b);
        float curr_overlap = interval_overlap(curr_min_a, curr_max_a, curr_min_b, curr_max_b);
        if (curr_overlap < -1e-5f)
            continue;

        float entry_alpha = 0.0f;
        if (prev_gap > 1e-5f) {
            entry_alpha = prev_gap / fmaxf(prev_gap + curr_overlap, 1e-6f);
        }

        if (prev_gap >= -1e-4f && (!found || entry_alpha > best_entry_alpha ||
                                   (fabsf(entry_alpha - best_entry_alpha) < 1e-4f && fabsf(prev_gap) < best_gap))) {
            found = 1;
            best_entry_alpha = entry_alpha;
            best_gap = fabsf(prev_gap);
            best_axis_x = axis_x;
            best_axis_y = axis_y;
        }
    }

    if (!found)
        return 0;

    orient_axis_from_agent_to_agent(&impact_a, &impact_b, &best_axis_x, &best_axis_y);
    *normal_x = best_axis_x;
    *normal_y = best_axis_y;
    return 1;
}

static int classify_impact_zone_from_normal(const Agent *agent, float normal_x, float normal_y) {
    float heading = (agent->prev_sim_x != INVALID_POSITION) ? agent->prev_sim_heading : agent->sim_heading;
    float heading_cos = cosf(heading);
    float heading_sin = sinf(heading);
    float local_long = normal_x * heading_cos + normal_y * heading_sin;
    float local_lat = -normal_x * heading_sin + normal_y * heading_cos;

    if (fabsf(local_long) >= fabsf(local_lat)) {
        return (local_long >= 0.0f) ? COLLISION_IMPACT_ZONE_FRONT : COLLISION_IMPACT_ZONE_REAR;
    }

    return (local_lat >= 0.0f) ? COLLISION_IMPACT_ZONE_LEFT : COLLISION_IMPACT_ZONE_RIGHT;
}

static void evaluate_collision_pair(const Agent *agent_a, const Agent *agent_b, float normal_x, float normal_y,
                                    float *severity, float *responsibility) {
    float mass_a = fmaxf(agent_a->sim_length * agent_a->sim_width, 1e-6f);
    float mass_b = fmaxf(agent_b->sim_length * agent_b->sim_width, 1e-6f);
    float rel_vx = agent_a->sim_vx - agent_b->sim_vx;
    float rel_vy = agent_a->sim_vy - agent_b->sim_vy;
    float closing_speed = fmaxf(0.0f, rel_vx * normal_x + rel_vy * normal_y);
    float reduced_mass = (mass_a * mass_b) / fmaxf(mass_a + mass_b, 1e-6f);
    float contrib_a = fmaxf(0.0f, agent_a->sim_vx * normal_x + agent_a->sim_vy * normal_y);
    float contrib_b = fmaxf(0.0f, -(agent_b->sim_vx * normal_x + agent_b->sim_vy * normal_y));

    if (severity != NULL) {
        *severity = reduced_mass * closing_speed * closing_speed;
    }
    if (responsibility != NULL) {
        *responsibility = contrib_a / fmaxf(contrib_a + contrib_b, 1e-6f);
    }
}

static int collision_check(Drive *env, int agent_idx, float *collision_normal_x, float *collision_normal_y,
                           float *penetration_depth) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->sim_x == INVALID_POSITION || agent->removed)
        return -1;

    int car_collided_with_index = -1;

    // O(N) linear scan over all agents (active + static); no spatial grid used here.
    // COLLISION_QUICK_CHECK_DIST (15m) is the real bottleneck — the 5m-cell grid
    // neighborhood covers ~50m but this quick check prunes at 15m.
    for (int i = 0; i < env->num_agents; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1)
            continue;
        if (index == agent_idx)
            continue;

        Agent *other_agent = &env->agents[index];
        if (other_agent->removed || other_agent->sim_x == INVALID_POSITION)
            continue;

        float dist_sq = ((other_agent->sim_x - agent->sim_x) * (other_agent->sim_x - agent->sim_x) +
                         (other_agent->sim_y - agent->sim_y) * (other_agent->sim_y - agent->sim_y));
        if (dist_sq > COLLISION_QUICK_CHECK_DIST * COLLISION_QUICK_CHECK_DIST)
            continue;
        if (!check_z_collision_possibility(agent, other_agent))
            continue;
        float sat_normal_x = 0.0f;
        float sat_normal_y = 0.0f;
        float sat_penetration = 0.0f;
        if (check_obb_collision_with_normal(agent, other_agent, &sat_normal_x, &sat_normal_y, &sat_penetration)) {
            car_collided_with_index = index;
            if (collision_normal_x != NULL && collision_normal_y != NULL) {
                int target_agent_idx = env->active_agent_count > 0 ? env->active_agent_indices[0] : -1;
                int target_involved = agent_idx == target_agent_idx || index == target_agent_idx;
                if (!target_involved || !compute_first_contact_collision_normal(agent, other_agent, collision_normal_x,
                                                                                collision_normal_y)) {
                    *collision_normal_x = sat_normal_x;
                    *collision_normal_y = sat_normal_y;
                }
            }
            if (penetration_depth != NULL)
                *penetration_depth = sat_penetration;
            break;
        }
    }

    return car_collided_with_index;
}

// Adversary's future path under max braking + constant heading, from the collision moment
// (index 0 = current pose) until it stops. Only x/y vary; heading, z and dimensions stay at
// the adversary's current values, so callers rebuild a pose via `Agent sample = *adversary;
// sample.sim_x = ext.x[j]; sample.sim_y = ext.y[j];`.
typedef struct {
    int num_steps; // valid poses in [0, num_steps); >= 1
    float x[TARGET_AVOIDABILITY_MAX_EXT_STEPS];
    float y[TARGET_AVOIDABILITY_MAX_EXT_STEPS];
} AdversaryBrakeTrajectory;

// Fills `out` with the adversary braking to a stop at TARGET_AVOIDABILITY_BRAKE_DECEL along
// its current heading. Reverse motion (sim_speed_signed < 0) keeps moving backward along the
// heading while |speed| decays to 0. The signed distance is clamped at the stop time so the
// pose holds still afterward (the braking parabola would otherwise reverse past its peak).
static inline void build_adversary_brake_trajectory(const Agent *adversary, float dt, AdversaryBrakeTrajectory *out) {
    float v0 = adversary->sim_speed_signed;
    float decel = TARGET_AVOIDABILITY_BRAKE_DECEL;
    float stop_time = fabsf(v0) / decel; // seconds until |speed| reaches 0

    out->num_steps = 0;
    for (int j = 0; j < TARGET_AVOIDABILITY_MAX_EXT_STEPS; j++) {
        float t = fminf(j * dt, stop_time);
        float d = v0 * t - copysignf(0.5f * decel * t * t, v0);
        out->x[j] = adversary->sim_x + d * adversary->cos_heading;
        out->y[j] = adversary->sim_y + d * adversary->sin_heading;
        out->num_steps++;
        if (j * dt >= stop_time) {
            break; // fully stopped; further steps would just repeat this pose
        }
    }
}

// Samples the target's braking path at arc-length `s`: the recorded rail
// (brake_hist[idx_start..LEN-1] then the live pose), and once `s` exceeds the rail length, a
// straight continuation from the rail end along the end heading. Heading is the local path
// direction (equivalently the box orientation, invariant to the 180-deg flip of a reversing
// car). Invariant: the caller passes idx_start in [0, LEN-1], so num_points >= 2.
static void target_brake_path_sample(Agent *target, int idx_start, float s, float *out_x, float *out_y, float *out_z,
                                     float *out_heading) {
    float px[TARGET_BRAKE_HISTORY_LEN + 1];
    float py[TARGET_BRAKE_HISTORY_LEN + 1];
    float pz[TARGET_BRAKE_HISTORY_LEN + 1];
    int num_points = 0;
    for (int k = idx_start; k < TARGET_BRAKE_HISTORY_LEN; k++) {
        px[num_points] = target->brake_hist_x[k];
        py[num_points] = target->brake_hist_y[k];
        pz[num_points] = target->brake_hist_z[k];
        num_points++;
    }
    px[num_points] = target->sim_x;
    py[num_points] = target->sim_y;
    pz[num_points] = target->sim_z;
    num_points++;

    float traveled = 0.0f;
    for (int k = 0; k < num_points - 1; k++) {
        float dx = px[k + 1] - px[k];
        float dy = py[k + 1] - py[k];
        float dz = pz[k + 1] - pz[k];
        float seg_len = sqrtf(dx * dx + dy * dy);
        if (traveled + seg_len >= s) {
            float t = seg_len > 1e-6f ? clip((s - traveled) / seg_len, 0.0f, 1.0f) : 0.0f;
            *out_x = px[k] + t * dx;
            *out_y = py[k] + t * dy;
            *out_z = pz[k] + t * dz;
            *out_heading = seg_len > 1e-6f ? atan2f(dy, dx) : target->sim_heading;
            return;
        }
        traveled += seg_len;
    }

    // `s` exceeds the rail: continue straight from the rail end along the last-segment heading.
    int last = num_points - 1;
    float dx = px[last] - px[last - 1];
    float dy = py[last] - py[last - 1];
    float seg_len = sqrtf(dx * dx + dy * dy);
    float heading = seg_len > 1e-6f ? atan2f(dy, dx) : target->sim_heading;
    float overshoot = s - traveled; // distance past the rail end (>= 0)
    *out_x = px[last] + overshoot * cosf(heading);
    *out_y = py[last] + overshoot * sinf(heading);
    *out_z = pz[last];
    *out_heading = heading;
}

// True if the target, braking (TARGET_AVOIDABILITY_BRAKE_DECEL) along its rail then straight
// from `steps_back` steps before the collision, avoids overlapping the adversary's extended
// trajectory. Collisions are checked per timestep from the collision moment until the target
// stops; a target that stops at or before the collision moment counts as avoided (the
// adversary is not then checked against the stationary target).
static bool target_avoidability_is_avoidable(Agent *target, const Agent *adversary,
                                             const AdversaryBrakeTrajectory *adv_ext, float dt, int steps_back) {
    int idx_start = TARGET_BRAKE_HISTORY_LEN - steps_back;
    float v0 = fabsf(target->brake_hist_speed_signed[idx_start]);
    if (v0 <= 0.0f) {
        return true; // already stationary at brake-start
    }

    float decel = TARGET_AVOIDABILITY_BRAKE_DECEL;
    float stop_time = v0 / decel;
    if (steps_back * dt >= stop_time) {
        return true; // target stops at or before the collision moment
    }

    Agent sample = *target;
    Agent adv_sample = *adversary; // heading/z/dims constant; only x/y vary per step
    for (int j = 0; j < TARGET_AVOIDABILITY_MAX_EXT_STEPS; j++) {
        float tau = steps_back * dt + j * dt; // total braking time at this step
        if (tau >= stop_time) {
            return true; // target has stopped without hitting the adversary
        }

        float s = v0 * tau - 0.5f * decel * tau * tau;
        float sx, sy, sz, sheading;
        target_brake_path_sample(target, idx_start, s, &sx, &sy, &sz, &sheading);
        sample.sim_x = sx;
        sample.sim_y = sy;
        sample.sim_z = sz;
        sample.sim_heading = normalize_heading(sheading);
        sample.cos_heading = cosf(sample.sim_heading);
        sample.sin_heading = sinf(sample.sim_heading);

        int adv_idx = j < adv_ext->num_steps ? j : adv_ext->num_steps - 1; // stopped adversary stays put
        adv_sample.sim_x = adv_ext->x[adv_idx];
        adv_sample.sim_y = adv_ext->y[adv_idx];

        if (check_z_collision_possibility(&sample, &adv_sample) && check_obb_collision(&sample, &adv_sample)) {
            return false;
        }
    }

    return true; // loop always exits via the stop condition before this; bound is a safety cap
}

// Last braking lead-time (seconds) that would have let the target avoid the collision with
// other_agent_idx, or TARGET_AVOIDABILITY_NOT_AVOIDABLE if no braking within the recorded
// history avoids it. Linear scan from the latest feasible brake point (1 step) outward.
static float last_avoidable_time_by_braking(Drive *env, int target_agent_idx, int other_agent_idx) {
    Agent *target = &env->agents[target_agent_idx];
    Agent *adversary = &env->agents[other_agent_idx];
    AdversaryBrakeTrajectory adv_ext;
    build_adversary_brake_trajectory(adversary, env->dt, &adv_ext);

    for (int steps_back = 1; steps_back <= target->brake_hist_count; steps_back++) {
        if (target_avoidability_is_avoidable(target, adversary, &adv_ext, env->dt, steps_back)) {
            return steps_back * env->dt;
        }
    }
    return TARGET_AVOIDABILITY_NOT_AVOIDABLE;
}

static bool is_adversarial_agent(Drive *env, int agent_idx) {
    for (int i = 1; i < env->active_agent_count; i++) {
        if (env->active_agent_indices[i] == agent_idx) {
            return true;
        }
    }
    return false;
}

static bool is_agent_behind_rear_axle(const Agent *ego, const Agent *other) {
    // nuPlan uses is_agent_behind(ego.rear_axle, other.box.center), where behind
    // means relative angle > 150 degrees. Use the equivalent normalized dot test.
    float rear_x = ego->sim_x - 0.5f * ego->sim_length * ego->cos_heading;
    float rear_y = ego->sim_y - 0.5f * ego->sim_length * ego->sin_heading;
    float dx = other->sim_x - rear_x;
    float dy = other->sim_y - rear_y;
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist < 1e-6f)
        return false;

    float dot = (ego->cos_heading * dx + ego->sin_heading * dy) / dist;
    return dot < NUPLAN_BEHIND_COS_THRESHOLD;
}

static bool is_agent_cleanly_in_lane(const Agent *agent) {
    if (agent->current_lane_idx == -1)
        return false;

    float edge_dist = fabsf(agent->metrics_array[LANE_DIST_IDX]) + 0.5f * agent->sim_width;
    return edge_dist <= MULTI_LANE_THRESHOLD;
}

static int classify_collision_type(const Agent *ego, const Agent *other) {
    if (ego->sim_speed <= AGENT_STOPPED_SPEED_THRESHOLD)
        return COLLISION_TYPE_STOPPED_EGO;

    if (other->sim_speed <= AGENT_STOPPED_SPEED_THRESHOLD)
        return COLLISION_TYPE_STOPPED_TRACK;

    if (is_agent_behind_rear_axle(ego, other))
        return COLLISION_TYPE_ACTIVE_REAR;

    if (front_bumper_intersects_agent(ego, other))
        return COLLISION_TYPE_ACTIVE_FRONT;

    return COLLISION_TYPE_ACTIVE_LATERAL;
}

// Classify whether a collision is at-fault for the ego agent.
static bool is_at_fault_collision(Drive *env, int agent_idx, int other_idx) {
    Agent *agent = &env->agents[agent_idx];
    Agent *other = &env->agents[other_idx];
    int collision_type = classify_collision_type(agent, other);

    if (collision_type == COLLISION_TYPE_STOPPED_TRACK)
        return true;

    if (collision_type == COLLISION_TYPE_ACTIVE_FRONT)
        return true;

    if (collision_type == COLLISION_TYPE_ACTIVE_LATERAL)
        return !is_agent_cleanly_in_lane(agent);

    return false;
}

static inline struct ttc_result default_ttc_result(void) {
    return (struct ttc_result){DEFAULT_TTC, -1, INFINITY, 0.0f};
}

static inline void ttc_update_min_result(Agent *ego, int other_idx, float distance_to_collision, float closing_speed,
                                         float ttc) {
    if (ttc < ego->cached_ttc.min_ttc) {
        ego->cached_ttc.min_ttc = ttc;
        ego->cached_ttc.other_idx = other_idx;
        ego->cached_ttc.distance_to_collision = distance_to_collision;
        ego->cached_ttc.closing_speed = closing_speed;
    }
}

// Compute TTC using ego front and other rear points with ahead and lateral corridor filters.
static inline void compute_pairwise_ttc(Agent *ego, int ego_idx, Agent *other, int other_idx) {
    if (other_idx == ego_idx)
        return;
    if (other->sim_x == INVALID_POSITION)
        return;

    float ego_x = ego->sim_x;
    float ego_y = ego->sim_y;
    float other_x = other->sim_x;
    float other_y = other->sim_y;

    float ego_heading_x = ego->cos_heading;
    float ego_heading_y = ego->sin_heading;
    float other_heading_x = other->cos_heading;
    float other_heading_y = other->sin_heading;

    float ego_front_x = ego_x + 0.5f * ego->sim_length * ego_heading_x;
    float ego_front_y = ego_y + 0.5f * ego->sim_length * ego_heading_y;
    float other_rear_x = other_x - 0.5f * other->sim_length * other_heading_x;
    float other_rear_y = other_y - 0.5f * other->sim_length * other_heading_y;

    float rel_x = other_rear_x - ego_front_x;
    float rel_y = other_rear_y - ego_front_y;
    float ahead = rel_x * ego_heading_x + rel_y * ego_heading_y;
    if (ahead <= 0.0f)
        return;

    float lateral = fabsf(rel_x * ego_heading_y - rel_y * ego_heading_x);
    float allowed = 0.5f * (ego->sim_width + other->sim_width);
    if (lateral > allowed)
        return;

    float distance_to_collision = sqrtf(rel_x * rel_x + rel_y * rel_y);
    float ego_radius = 0.5f * ego->sim_width;
    float other_radius = 0.5f * other->sim_width;
    float combined_radius = ego_radius + other_radius;

    float rel_vx = other->sim_vx - ego->sim_vx;
    float rel_vy = other->sim_vy - ego->sim_vy;
    float a = rel_vx * rel_vx + rel_vy * rel_vy;
    float c = rel_x * rel_x + rel_y * rel_y - combined_radius * combined_radius;
    if (c <= 0.0f) {
        ttc_update_min_result(ego, other_idx, distance_to_collision, INFINITY, 0.0f);
        return;
    }
    if (a < 1e-6f)
        return;

    float b = 2.0f * (rel_x * rel_vx + rel_y * rel_vy);
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f)
        return;

    float sqrt_disc = sqrtf(disc);
    float inv_two_a = 0.5f / a;
    float t1 = (-b - sqrt_disc) * inv_two_a;
    float t2 = (-b + sqrt_disc) * inv_two_a;
    float ttc = INFINITY;
    if (t1 > 0.0f) {
        ttc = t1;
    } else if (t2 > 0.0f) {
        ttc = t2;
    }
    if (!isfinite(ttc))
        return;

    float closing_speed = sqrtf(a);
    ttc_update_min_result(ego, other_idx, distance_to_collision, closing_speed, ttc);
}

// Compute TTC for a single ego agent against all other agents using ahead and lateral filters.
static void compute_agent_ttc(Drive *env, int ego_idx) {
    Agent *ego = &env->agents[ego_idx];
    ego->cached_ttc = default_ttc_result();

    if (ego->sim_x == INVALID_POSITION)
        return;

    for (int j = 0; j < env->num_agents; j++) {
        int other_idx;
        if (j < env->active_agent_count) {
            other_idx = env->active_agent_indices[j];
        } else {
            other_idx = env->static_agent_indices[j - env->active_agent_count];
        }
        if (other_idx == -1)
            continue;
        compute_pairwise_ttc(ego, ego_idx, &env->agents[other_idx], other_idx);
    }
}

// Puffer score computation
// Uses hybrid weighted average: multiplier weights (binary gates) + average weights (continuous)
static float calculate_puffer_score(Log *log_agent, int scenario_length, float dt) {
    if (!log_agent) {
        return 0.0f;
    }

    float T = scenario_length * dt;
    if (T <= 0.0f)
        T = 1.0f; // Avoid division by zero

    float no_at_fault = (log_agent->at_fault_collision_rate > 0) ? 0.0f : 1.0f;
    float no_offroad = (log_agent->offroad_rate > 0) ? 0.0f : 1.0f;
    float no_red_light = (log_agent->red_light_violation_rate > 0) ? 0.0f : 1.0f;
    float making_progress = (log_agent->progress_ratio > 0.2f) ? 1.0f : 0.0f;

    // Driving direction: 1.0 if <=2m, 0.5 if 2-6m, 0 if >6m wrong-way distance
    float wrong_dist = log_agent->wrong_way_distance;
    float direction_compliance = (wrong_dist <= 2.0f) ? 1.0f : (wrong_dist <= 6.0f) ? 0.5f : 0.0f;

    float multiplier = no_at_fault * no_offroad * no_red_light * making_progress * direction_compliance;

    if (multiplier == 0.0f)
        return 0.0f;

    // TTC within bound (>0.95s): weight 5
    float ttc_score = log_agent->ttc_within_bound_rate; // Already 0-1

    // Progress ratio (capped at 1): weight 5
    float progress_score = fminf(log_agent->progress_ratio, 1.0f);

    // Speed compliance (nuPlan formula): max(0, 1 - sum(violation * dt) / T): weight 4
    float speed_threshold = fmaxf(T, 1e-3f);
    float speed_score = fmaxf(0.0f, 1.0f - log_agent->speed_violation_sum / speed_threshold);

    // Comfort (binary per episode): weight 2
    float comfort_score = log_agent->comfort_score; // 0 or 1

    // Multi-lane (weight 3): tiered score based on accumulated time
    float multi_lane_score = log_agent->multi_lane_score;

    // Weighted average
    float weighted_sum =
        5 * ttc_score + 5 * progress_score + 4 * speed_score + 3 * multi_lane_score + 2 * comfort_score;
    float total_weight = 5 + 5 + 4 + 3 + 2; // = 19

    return multiplier * (weighted_sum / total_weight);
}

static void build_episode_log_contributions(Drive *env, Log *episode_log) {
    int safe_timestep = (env->timestep > 0) ? env->timestep : 1;
    const float progress_ref_speed = 10.0f;
    int target_agent_idx = env->active_agent_count > 0 ? env->active_agent_indices[0] : -1;
    for (int i = 0; i < env->active_agent_count; i++) {
        Agent *agent = &env->agents[env->active_agent_indices[i]];
        float episode_duration_s = env->logs[i].episode_length * env->dt;
        float reference_progress_distance = progress_ref_speed * episode_duration_s;
        reference_progress_distance = fmaxf(reference_progress_distance, 1.0f);
        env->logs[i].progress_ratio = agent->distance_since_spawn / reference_progress_distance;

        int offroad = env->logs[i].offroad_rate;
        episode_log->offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        episode_log->collision_rate += collided;
        if (i > 0) {
            episode_log->adversaries_offroad_rate += offroad;
            episode_log->adversaries_collision_rate += collided;
            episode_log->adversaries_red_light_violation_rate += env->logs[i].red_light_violation_rate;
            episode_log->adversaries_at_fault_collision_rate += env->logs[i].at_fault_collision_rate;

            if (collided > 0.0f) {
                if (agent->collided_with_agent_idx == target_agent_idx) {
                    episode_log->adversaries_target_collision_rate += 1.0f;
                } else if (is_adversarial_agent(env, agent->collided_with_agent_idx)) {
                    episode_log->adversaries_adversary_collision_rate += 1.0f;
                }
            }
        }
        if (i > 0 && collided > 0.0f) {
            episode_log->adversaries_collision_count += 1.0f;
            episode_log->adversaries_collision_severity += agent->collision_severity;
            episode_log->adversaries_collision_responsibility += agent->collision_responsibility;
            episode_log->adversaries_collision_impact_zone += (float)agent->collision_impact_zone;
        }
        int red_light_violations = env->logs[i].red_light_violation_rate;
        episode_log->red_light_violation_rate += red_light_violations;
        int total_infractions = (offroad || collided || red_light_violations) ? 1 : 0;
        float avg_speed_per_agent = env->logs[i].avg_speed_per_agent;
        episode_log->avg_speed_per_agent += avg_speed_per_agent / safe_timestep;
        if (i > 0) {
            episode_log->adversaries_avg_speed += avg_speed_per_agent / safe_timestep;
        }
        int num_waypoints_reached = env->logs[i].num_waypoints_reached;
        episode_log->num_waypoints_reached += num_waypoints_reached;
        if (i > 0) {
            episode_log->adversaries_num_waypoints_reached += num_waypoints_reached;
        }
        int num_goals_reached = env->logs[i].num_goals_reached;
        episode_log->num_goals_reached += num_goals_reached;
        // TODO: define better scoring criteria ?
        // FIXME
        if (num_goals_reached >= 4 && !agent->removed && !agent->stopped) {
            episode_log->score += 1.0f;
            if (i > 0) {
                episode_log->adversaries_score += 1.0f;
            }
        }
        if (!offroad && !collided && !red_light_violations && num_waypoints_reached < 1) {
            episode_log->dnf_rate += 1.0f;
            if (i > 0) {
                episode_log->adversaries_dnf_rate += 1.0f;
            }
        }
        episode_log->total_distance_travelled += agent->distance_since_spawn;
        if (total_infractions > 0) {
            episode_log->total_infractions += 1.0f;
        }
        float displacement_error = env->logs[i].avg_displacement_error;
        episode_log->avg_displacement_error += displacement_error;
        episode_log->episode_length += env->logs[i].episode_length;
        episode_log->episode_return += env->logs[i].episode_return;
        episode_log->episode_return_collision += env->logs[i].episode_return_collision;
        episode_log->episode_return_offroad += env->logs[i].episode_return_offroad;
        episode_log->episode_return_drive += env->logs[i].episode_return_drive;
        episode_log->episode_return_adversarial += env->logs[i].episode_return_adversarial;
        // Comfort and velocity metrics (normalized per timestep)
        float comfort_violations_per_timestep = env->logs[i].comfort_violation_count / safe_timestep;
        float comfort_longitudinal_accel_violations_per_timestep =
            env->logs[i].comfort_longitudinal_accel_violation_count / safe_timestep;
        float comfort_lateral_accel_violations_per_timestep =
            env->logs[i].comfort_lateral_accel_violation_count / safe_timestep;
        float comfort_jerk_violations_per_timestep = env->logs[i].comfort_jerk_violation_count / safe_timestep;
        float uncomfortable_timestep_rate = env->logs[i].uncomfortable_timestep_count / safe_timestep;

        episode_log->comfort_violation_count += comfort_violations_per_timestep;
        episode_log->comfort_longitudinal_accel_violation_count += comfort_longitudinal_accel_violations_per_timestep;
        episode_log->comfort_lateral_accel_violation_count += comfort_lateral_accel_violations_per_timestep;
        episode_log->comfort_jerk_violation_count += comfort_jerk_violations_per_timestep;
        episode_log->uncomfortable_timestep_count += uncomfortable_timestep_rate;
        if (i > 0) {
            episode_log->adversaries_comfort_violation_rate += comfort_violations_per_timestep;
            episode_log->adversaries_comfort_longitudinal_accel_violations_per_timestep +=
                comfort_longitudinal_accel_violations_per_timestep;
            episode_log->adversaries_comfort_lateral_accel_violations_per_timestep +=
                comfort_lateral_accel_violations_per_timestep;
            episode_log->adversaries_comfort_jerk_violations_per_timestep += comfort_jerk_violations_per_timestep;
            episode_log->adversaries_uncomfortable_timestep_rate += uncomfortable_timestep_rate;
        } else {
            episode_log->target_comfort_violations_per_timestep += comfort_violations_per_timestep;
            episode_log->target_comfort_longitudinal_accel_violations_per_timestep +=
                comfort_longitudinal_accel_violations_per_timestep;
            episode_log->target_comfort_lateral_accel_violations_per_timestep +=
                comfort_lateral_accel_violations_per_timestep;
            episode_log->target_comfort_jerk_violations_per_timestep += comfort_jerk_violations_per_timestep;
            episode_log->target_uncomfortable_timestep_rate += uncomfortable_timestep_rate;
        }
        episode_log->velocity_progress_sum += env->logs[i].velocity_progress_sum / safe_timestep;
        if (i > 0) {
            episode_log->adversaries_velocity_progress += env->logs[i].velocity_progress_sum / safe_timestep;
        }
        // Lane metrics (normalized per timestep for average per episode)
        episode_log->lane_center_rate += env->logs[i].lane_center_rate / safe_timestep;
        episode_log->lane_heading_aligned_rate += env->logs[i].lane_heading_aligned_rate / safe_timestep;
        if (i > 0) {
            episode_log->adversaries_lane_center_rate += env->logs[i].lane_center_rate / safe_timestep;
            episode_log->adversaries_lane_heading_aligned_rate +=
                env->logs[i].lane_heading_aligned_rate / safe_timestep;
        }
        if (env->compute_eval_metrics) {
            env->logs[i].progress_ratio = agent->distance_since_spawn / reference_progress_distance;
            episode_log->at_fault_collision_rate += env->logs[i].at_fault_collision_rate;
            episode_log->ttc_within_bound_rate += env->logs[i].ttc_within_bound_rate;
            episode_log->wrong_way_distance += env->logs[i].wrong_way_distance;
            episode_log->speed_violation_sum += env->logs[i].speed_violation_sum;
            episode_log->progress_ratio += env->logs[i].progress_ratio;
            episode_log->comfort_score += env->logs[i].comfort_score;
            if (i > 0) {
                episode_log->adversaries_comfort_score += env->logs[i].comfort_score;
            }
            episode_log->ttc_violations += env->logs[i].ttc_violations;
            episode_log->ttc_samples += env->logs[i].ttc_samples;
            episode_log->multi_lane_time += env->logs[i].multi_lane_time;
            episode_log->multi_lane_score += env->logs[i].multi_lane_score;

            float wrong_dist = env->logs[i].wrong_way_distance;
            float direction_score = (wrong_dist <= 2.0f) ? 1.0f : (wrong_dist <= 6.0f) ? 0.5f : 0.0f;
            episode_log->driving_direction_score += direction_score;
            if (i > 0) {
                episode_log->adversaries_driving_direction_score += direction_score;
            }

            float T = safe_timestep * env->dt;
            float speed_compliance = fmaxf(0.0f, 1.0f - env->logs[i].speed_violation_sum / fmaxf(T, 1e-3f));
            episode_log->speed_limit_compliance += speed_compliance;
            if (i > 0) {
                episode_log->adversaries_speed_limit_compliance += speed_compliance;
            }

            float making_progress = (env->logs[i].progress_ratio > 0.2f) ? 1.0f : 0.0f;
            episode_log->making_progress_rate += making_progress;
            if (i > 0) {
                episode_log->adversaries_making_progress_rate += making_progress;
            }
            episode_log->puffer_score += calculate_puffer_score(&env->logs[i], safe_timestep, env->dt);
            if (i > 0) {
                episode_log->adversaries_multi_lane_time += env->logs[i].multi_lane_time;
                episode_log->adversaries_multi_lane_score += env->logs[i].multi_lane_score;
            }
        }

        episode_log->n += 1;
    }

    if (env->active_agent_count > 0) {
        int target_agent_idx = env->active_agent_indices[0];
        Agent *target_agent = &env->agents[target_agent_idx];
        episode_log->target_n += 1.0f;
        episode_log->target_episode_length += env->logs[0].episode_length;
        episode_log->target_episode_return += env->logs[0].episode_return;
        episode_log->target_episode_return_collision += env->logs[0].episode_return_collision;
        episode_log->target_episode_return_offroad += env->logs[0].episode_return_offroad;
        episode_log->target_episode_return_drive += env->logs[0].episode_return_drive;
        episode_log->target_num_goals_reached += env->logs[0].num_goals_reached;

        if (env->compute_eval_metrics) {
            float target_progress_ratio = env->logs[0].progress_ratio;
            episode_log->target_progress_ratio += target_progress_ratio;
            episode_log->did_target_make_progress += (target_progress_ratio > 0.2f) ? 1.0f : 0.0f;
            episode_log->did_target_have_at_fault_collision += env->logs[0].at_fault_collision_rate;
            episode_log->target_ttc_within_bound_rate += env->logs[0].ttc_within_bound_rate;
            episode_log->target_puffer_score += calculate_puffer_score(&env->logs[0], safe_timestep, env->dt);
        }

        if (env->logs[0].collision_rate > 0.0f || env->target_collision_type_episode != COLLISION_TYPE_NONE) {
            episode_log->did_target_collide += 1.0f;
            episode_log->did_target_fail += 1.0f;
            episode_log->target_collision_count += 1.0f;
            episode_log->target_collision_severity += target_agent->collision_severity;
            episode_log->target_collision_responsibility += target_agent->collision_responsibility;
            episode_log->target_collision_impact_zone += (float)target_agent->collision_impact_zone;
            episode_log->target_collision_type += (float)env->target_collision_type_episode;
            episode_log->target_collision_target_speed += env->target_collision_target_speed_episode;
            episode_log->target_collision_other_speed += env->target_collision_other_speed_episode;
            episode_log->target_collision_relative_speed += env->target_collision_relative_speed_episode;
            episode_log->target_collision_other_active += env->target_collision_other_active_episode;
            episode_log->target_collision_other_stopped += env->target_collision_other_stopped_episode;
            episode_log->target_collision_other_removed += env->target_collision_other_removed_episode;
            episode_log->target_avoidability_by_braking += env->target_avoidability_by_braking_episode;
        }

        if (env->logs[0].offroad_rate > 0.0f) {
            episode_log->did_target_offroad += 1.0f;
            episode_log->did_target_fail += 1.0f;
        }

        if (env->logs[0].red_light_violation_rate > 0.0f) {
            episode_log->did_target_run_light += 1.0f;
            episode_log->did_target_fail += 1.0f;
        }
    }

    // Log composition counts per agent so vec_log averaging recovers the per-env value
    episode_log->expert_static_car_count += env->expert_static_agent_count;
    episode_log->static_car_count += env->static_agent_count;
}

static void add_log(Drive *env) {
    Log episode_log = {0};
    build_episode_log_contributions(env, &episode_log);
    episode_log.target_hit_count += env->target_hit_count_episode;
    episode_log.target_hit_responsibility += env->target_hit_responsibility_episode;
    episode_log.target_hit_low_responsibility_rate += env->target_hit_low_responsibility_count_episode;
    episode_log.target_hit_at_fault_count += env->target_hit_at_fault_count_episode;
    accumulate_log_totals(&env->log, &episode_log);
    enqueue_completed_episode_summary(env, &episode_log);
}

// ========================================
// Initialization Functions
// ========================================

static void reset_agent_metrics(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    for (int i = 0; i < NUM_METRICS; i++) {
        agent->metrics_array[i] = 0.0f;
    }
}

static void reset_agent_state(Agent *agent) {
    agent->cumulative_displacement = 0.0f;
    agent->displacement_sample_count = 0;
    agent->stopped = 0;
    agent->removed = 0;
    agent->current_lane_idx = -1;
    agent->previous_lane_idx = -1;
    agent->current_route_index = 0;
    agent->a_long = 0.0f;
    agent->a_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
    agent->steering_angle = 0.0f;
    agent->path_progression = 0.0f;
    agent->distance_since_spawn = 0.0f;
    agent->closest_path_idx_wp = 0;
    // Puffer score tracking reset
    agent->wrong_way_distance = 0.0f;
    agent->speed_violation_sum = 0.0f;
    agent->cached_ttc = default_ttc_result();
    agent->ttc_violations = 0;
    agent->ttc_samples = 0;
    agent->at_fault_collision = 0;
    agent->collided_with_agent_idx = -1;
    agent->collision_normal_x = 0.0f;
    agent->collision_normal_y = 0.0f;
    agent->collision_severity = 0.0f;
    agent->collision_responsibility = 0.0f;
    agent->collision_impact_zone = COLLISION_IMPACT_ZONE_NONE;
    agent->multi_lane_time = 0.0f;
    agent->phantom_braking_counter = 0;
    agent->is_blind_partner = 0;
    agent->is_phantom_braker = 0;
}

// Check if a spawn position collides with any existing agent
static bool check_spawn_collision(Drive *env, int num_existing_agents, float spawn_x, float spawn_y, float spawn_z,
                                  float spawn_heading, float spawn_length, float spawn_width, float spawn_height) {
    // Create a temporary agent structure for collision checking
    Agent temp_agent;
    temp_agent.sim_x = spawn_x;
    temp_agent.sim_y = spawn_y;
    temp_agent.sim_z = spawn_z;
    temp_agent.sim_heading = spawn_heading;
    temp_agent.cos_heading = cosf(spawn_heading);
    temp_agent.sin_heading = sinf(spawn_heading);
    temp_agent.yaw_rate = 0.0f;
    temp_agent.sim_length = spawn_length;
    temp_agent.sim_width = spawn_width;
    temp_agent.sim_height = spawn_height;

    // Minimum safe distance
    float min_safe_dist_sq = (spawn_length + 5.0f) * (spawn_length + 5.0f);

    for (int i = 0; i < num_existing_agents; i++) {
        Agent *other = &env->agents[i];

        // Skip invalid agents
        if (other->sim_x == INVALID_POSITION || other->sim_valid != 1)
            continue;

        // Quick distance check first
        float dx = other->sim_x - spawn_x;
        float dy = other->sim_y - spawn_y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq > min_safe_dist_sq)
            continue;
        if (!check_z_collision_possibility(&temp_agent, other))
            continue;
        if (check_obb_collision(&temp_agent, other))
            return true; // Collision detected
    }

    return false; // No collision
}

static bool check_spawn_offroad(Drive *env, float spawn_x, float spawn_y, float spawn_heading, float spawn_length,
                                float spawn_width, float spawn_z) {
    Agent temp_agent = {0};
    temp_agent.sim_x = spawn_x;
    temp_agent.sim_y = spawn_y;
    temp_agent.sim_heading = spawn_heading;
    temp_agent.cos_heading = cosf(spawn_heading);
    temp_agent.sin_heading = sinf(spawn_heading);
    // Slightly inflate dimensions to reject near-boundary spawn candidates.
    temp_agent.sim_length = spawn_length * 1.1f;
    temp_agent.sim_width = spawn_width * 1.1f;
    float corners[4][2];
    compute_agent_corners(&temp_agent, corners);

    // Get neighboring road elements
    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
    int list_size =
        get_neighbors_entities(env, spawn_x, spawn_y, entity_list, MAX_ENTITIES_PER_CELL * 25, collision_offsets, 25);

    // Check intersection with road edges
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_type != ENTITY_TYPE_ROAD_ELEMENT)
            continue;

        int entity_idx = entity_list[i].entity_idx;
        int geometry_idx = entity_list[i].geometry_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];

        if (is_road_edge(element->type)) {
            float abs_dz = fabsf(element->z[geometry_idx] - spawn_z);
            if (abs_dz > Z_BUFFER)
                continue;
            float start[2] = {element->x[geometry_idx], element->y[geometry_idx]};
            float end[2] = {element->x[geometry_idx + 1], element->y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) {
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end))
                    return true; // Offroad detected
            }
        }
    }
    return false;
}

static bool check_spawn_red_light_violation(Drive *env, float spawn_x, float spawn_y, float spawn_z,
                                            float spawn_heading, float spawn_length, float spawn_width,
                                            float spawn_height, int lane_idx) {
    Agent temp_agent;
    temp_agent.sim_x = spawn_x;
    temp_agent.sim_y = spawn_y;
    temp_agent.sim_z = spawn_z;
    temp_agent.sim_heading = spawn_heading;
    temp_agent.cos_heading = cosf(spawn_heading);
    temp_agent.sin_heading = sinf(spawn_heading);
    temp_agent.sim_length = spawn_length;
    temp_agent.sim_width = spawn_width;
    temp_agent.sim_height = spawn_height;
    temp_agent.current_lane_idx = lane_idx;

    float corners[4][2];
    compute_agent_corners(&temp_agent, corners);

    return check_stop_line_crossing(env, &temp_agent, lane_idx, corners);
}

static bool sample_drivable_lane_candidate_global(Drive *env, int *start_lane_idx, int *geometry_idx) {
    if (env->grid_map->num_drivable_grid_cell <= 0)
        return false;

    int list_idx = rand() % env->grid_map->num_drivable_grid_cell;
    int grid_idx = env->grid_map->grid_index_drivable[list_idx];

    GridMapEntity cell_candidates[MAX_ENTITIES_PER_CELL];
    int candidate_count = 0;

    for (int i = 0; i < env->grid_map->cell_entities_count[grid_idx]; i++) {
        GridMapEntity entity = env->grid_map->cells[grid_idx][i];
        if (entity.entity_type != ENTITY_TYPE_ROAD_ELEMENT)
            continue;
        if (!is_drivable_road_lane(env->road_elements[entity.entity_idx].type))
            continue;
        cell_candidates[candidate_count++] = entity;
    }

    if (candidate_count == 0)
        return false;

    GridMapEntity chosen_entity = cell_candidates[rand() % candidate_count];
    *start_lane_idx = chosen_entity.entity_idx;
    *geometry_idx = chosen_entity.geometry_idx;
    return true;
}

static bool sample_drivable_lane_candidate_near_target(Drive *env, float target_x, float target_y, float radius,
                                                       int *start_lane_idx, int *geometry_idx) {
    if (env->grid_map->num_drivable_grid_cell <= 0 || radius <= 0.0f)
        return false;

    float radius_sq = radius * radius;
    int candidate_count = 0;
    int chosen_lane_idx = -1;
    int chosen_geometry_idx = -1;

    for (int i = 0; i < env->grid_map->num_drivable_grid_cell; i++) {
        int grid_idx = env->grid_map->grid_index_drivable[i];

        for (int j = 0; j < env->grid_map->cell_entities_count[grid_idx]; j++) {
            GridMapEntity entity = env->grid_map->cells[grid_idx][j];
            if (entity.entity_type != ENTITY_TYPE_ROAD_ELEMENT)
                continue;
            if (!is_drivable_road_lane(env->road_elements[entity.entity_idx].type))
                continue;

            RoadMapElement *lane = &env->road_elements[entity.entity_idx];
            float dx = lane->x[entity.geometry_idx] - target_x;
            float dy = lane->y[entity.geometry_idx] - target_y;
            if (dx * dx + dy * dy > radius_sq)
                continue;

            candidate_count++;
            if (rand() % candidate_count == 0) {
                chosen_lane_idx = entity.entity_idx;
                chosen_geometry_idx = entity.geometry_idx;
            }
        }
    }

    if (candidate_count == 0)
        return false;

    *start_lane_idx = chosen_lane_idx;
    *geometry_idx = chosen_geometry_idx;
    return true;
}

// NOTE: type of function -> void, int, bool ?
static int spawn_agent(Drive *env, int agent_idx, int num_agents) {
    Agent *agent = &env->agents[agent_idx];

    // Free existing route on reset
    if (agent->route != NULL) {
        free(agent->route);
        agent->route = NULL;
    }

    // Initialize identity fields
    agent->type = VEHICLE;
    agent->active_agent = 1;
    agent->mark_as_expert = 0;
    agent->stopped = 0;
    agent->removed = 0;

    // Default vehicle dimensions
    // length: [0.8, 7.0] m
    // width: [0.8, 3.0] m
    // width = min(width, length)
    float spawn_length, spawn_width;
    if (env->eval_mode) {
        // Fixed size for eval mode
        spawn_length = random_uniform(2.0f, 5.5f);
        spawn_width = random_uniform(1.5f, 2.5f);
    } else {
        // Random size for training mode
        spawn_length = random_uniform(0.8f, 7.0f);
        spawn_width = random_uniform(0.8f, 3.0f);
    }
    if (spawn_width > spawn_length)
        spawn_width = spawn_length;
    float spawn_height = 1.5f; // Fixed height

    // Set spawn position on start lane
    float spawn_x, spawn_y, spawn_z, spawn_heading;
    RoadMapElement *start_lane;
    int start_lane_idx;
    int success = 0;

    // Sampling rejection loop
    // TARGET: Only one attempt should be sufficient in most cases
    const int MAX_SPAWN_ATTEMPTS = 30;
    int local_attempts = env->targeted_spawn_attempts;
    if (local_attempts < 0)
        local_attempts = 0;
    if (local_attempts > MAX_SPAWN_ATTEMPTS)
        local_attempts = MAX_SPAWN_ATTEMPTS;

    for (int attempt = 0; attempt < MAX_SPAWN_ATTEMPTS; attempt++) {
        int candidate_geometry_idx = -1;
        bool sampled = false;
        bool try_targeted_spawn = env->targeted_spawn_mode == 1 && agent_idx > 0 && attempt < local_attempts &&
                                  env->agents[0].sim_valid == 1 && env->agents[0].sim_x != INVALID_POSITION;

        if (try_targeted_spawn) {
            sampled = sample_drivable_lane_candidate_near_target(env, env->agents[0].sim_x, env->agents[0].sim_y,
                                                                 env->targeted_spawn_radius, &start_lane_idx,
                                                                 &candidate_geometry_idx);
        }

        if (!sampled) {
            sampled = sample_drivable_lane_candidate_global(env, &start_lane_idx, &candidate_geometry_idx);
        }

        if (!sampled)
            continue;

        start_lane = &env->road_elements[start_lane_idx];
        spawn_x = start_lane->x[candidate_geometry_idx];
        spawn_y = start_lane->y[candidate_geometry_idx];
        spawn_z = start_lane->z[candidate_geometry_idx];
        spawn_heading = start_lane->headings[candidate_geometry_idx];

        // Check for collision with existing/already-reset agents
        if (check_spawn_collision(env, num_agents, spawn_x, spawn_y, spawn_z, spawn_heading, spawn_length, spawn_width,
                                  spawn_height))
            continue;

        // Check for offroad (vehicle corners intersecting road edges)
        if (check_spawn_offroad(env, spawn_x, spawn_y, spawn_heading, spawn_length, spawn_width, spawn_z))
            continue;

        // Check for red light violation at spawn (vehicle corners intersecting stop lines)
        if (check_spawn_red_light_violation(env, spawn_x, spawn_y, spawn_z, spawn_heading, spawn_length, spawn_width,
                                            spawn_height, start_lane_idx))
            continue;

        success = 1;
        break;
    }

    if (!success) {
        printf("[GIGAFLOW WARNING] -> Failed to find a collision-free spawn position for agent %d\n", agent_idx);
        return 0; // Failed to find collision-free spawn
    }

    // Update simulation state
    agent->sim_x = spawn_x;
    agent->sim_y = spawn_y;
    agent->sim_z = spawn_z;
    agent->sim_heading = spawn_heading;
    agent->cos_heading = cosf(spawn_heading);
    agent->sin_heading = sinf(spawn_heading);
    agent->sim_length = spawn_length;
    agent->sim_width = spawn_width;
    agent->sim_height = spawn_height;
    agent->sim_valid = 1;
    agent->wheelbase = 0.6f * spawn_length;

    float spawn_speed = clip(env->spawn_initial_speed, 0.0f, MAX_SPEED);
    agent->sim_vx = spawn_speed * agent->cos_heading;
    agent->sim_vy = spawn_speed * agent->sin_heading;
    agent->yaw_rate = 0.0f;
    update_agent_speed(agent);

    // Compute initial route
    if (!compute_new_route(env, agent_idx, start_lane_idx)) {
        printf("[GIGAFLOW WARNING] -> Failed to compute a new route for agent %d\n", agent_idx);
        return 0; // Failed to compute new goal
    }

    // Compute initial goal
    if (!compute_goals(env, agent_idx) || agent->removed) {
        printf("[GIGAFLOW WARNING] -> Failed to compute initial goals for agent %d\n", agent_idx);
        return 0;
    }

    return 1; // Success
}

static void set_start_position(Drive *env) {
    bool is_log_replay = (env->control_mode == CONTROL_SDC_ONLY);

    for (int i = 0; i < env->num_total_agents; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Agent *agent = &env->agents[i];

        // Initialize simulation trajectory from logged trajectory at init_steps
        if (env->simulation_mode == SIMULATION_REPLAY) {
            // Clamp init_steps to ensure we don't go out of bounds
            int step = env->init_steps;
            if (step >= agent->trajectory_length)
                step = agent->trajectory_length - 1;
            if (step < 0)
                step = 0;

            // For agents invalid at init_steps, set INVALID_POSITION
            // move_expert will update them when they become valid
            if (agent->log_valid[step] != 1) {
                invalidate_agent(agent);
                agent->sim_length = agent->log_length[step];
                agent->sim_width = agent->log_width[step];
                agent->sim_height = agent->log_height[step];
                continue;
            }

            agent->sim_x = agent->log_trajectory_x[step];
            agent->sim_y = agent->log_trajectory_y[step];
            agent->sim_z = agent->log_trajectory_z[step];
            agent->prev_sim_x = agent->sim_x;
            agent->prev_sim_y = agent->sim_y;
            agent->prev_sim_heading = agent->log_heading[step];
            agent->sim_heading = agent->log_heading[step];
            agent->cos_heading = cosf(agent->sim_heading);
            agent->sin_heading = sinf(agent->sim_heading);
            agent->sim_valid = agent->log_valid[step];
            agent->sim_length = agent->log_length[step];
            agent->sim_width = agent->log_width[step];
            agent->sim_height = agent->log_height[step];
            // Estimate wheelbase as 60% of length
            agent->wheelbase = 0.6f * agent->sim_length;

            if (agent->type == UNKNOWN)
                continue;

            if (is_active == 0) {
                agent->sim_vx = 0.0f;
                agent->sim_vy = 0.0f;
                agent->yaw_rate = 0.0f;
                agent->sim_speed = 0.0f;
                agent->sim_speed_signed = 0.0f;
            } else {
                agent->yaw_rate = compute_log_yaw_rate(agent, step, env->dt);
                agent->sim_vx = agent->log_velocity_x[step];
                agent->sim_vy = agent->log_velocity_y[step];
                update_agent_speed(agent);
            }

            // Shrink width and length slightly to avoid initial collisions (not in log-replay)
            if (!is_log_replay) {
                agent->sim_length *= INIT_COLLISION_SHRINK_FACTOR;
                agent->sim_width *= INIT_COLLISION_SHRINK_FACTOR;
            }
        }

        // Reset agent metrics and state
        reset_agent_metrics(env, i);
        reset_agent_state(agent);
        generate_reward_coefs(env, agent);
    }
}

static bool should_control_agent(Drive *env, int agent_idx) {
    // Check if we have room for more agents or are already at capacity
    if (env->num_controllable_agents != 0 && env->active_agent_count >= env->num_controllable_agents) {
        return false;
    }

    Agent *agent = &env->agents[agent_idx];

    if (env->control_mode == CONTROL_SDC_ONLY) {
        return agent_idx == 0 && agent->route_length != 0;
    }

    if (env->control_mode == CONTROL_WOSAC) {
        for (int j = 0; j < env->num_tracks_to_predict; j++) {
            if (env->tracks_to_predict[j] == agent_idx) {
                return true;
            }
        }
        return false;
    }

    // Standard mode: check type, distance to goal, and expert status
    bool type_is_controllable = false;
    if (env->control_mode == CONTROL_VEHICLES) {
        type_is_controllable = (agent->type == VEHICLE);
    } else { // CONTROL_AGENTS mode
        type_is_controllable = is_controllable_agent(agent->type);
    }

    if (!type_is_controllable || agent->mark_as_expert)
        return false;

    // Control if the agent has a route to follow
    return agent->route_length != 0;
}

static int resolve_agent_controller(Drive *env, int agent_idx, int is_active, int replay_by_default) {
    int requested_controller = (agent_idx == 0) ? env->sdc_controller : env->non_sdc_controller;

    if (requested_controller == CONTROLLER_POLICY && !is_active) {
        return replay_by_default ? CONTROLLER_REPLAY : CONTROLLER_STATIC;
    }

    return requested_controller;
}

void set_active_agents(Drive *env) {
    // Initialize
    env->active_agent_count = 0;        // Foreground agents with observations/rewards/actions
    env->static_agent_count = 0;        // Background agents without RL buffer slots
    env->expert_static_agent_count = 0; // Legacy subset: background log-replay agents
    env->num_agents = 0;                // Total agents created

    // In GIGAFLOW mode, spawn agents dynamically on the map
    if (env->simulation_mode == SIMULATION_GIGAFLOW) {
        int num_agents_to_create = env->num_controllable_agents;

        // Initialize agents for GIGAFLOW mode
        env->agents = (Agent *)calloc(num_agents_to_create, sizeof(Agent));

        int successfully_created = 0;
        for (int i = 0; i < num_agents_to_create; i++) {
            // Pass the number of already successfully created agents for collision checking
            if (spawn_agent(env, i, successfully_created)) {
                successfully_created++;
            } else {
                // Failed spawn: ensure agent is properly invalidated
                invalidate_agent(&env->agents[i]);
                env->agents[i].removed = 1;
            }
        }

        env->num_total_agents = successfully_created;

        // Set up active agent indices
        env->active_agent_indices = (int *)malloc(env->num_total_agents * sizeof(int));
        env->static_agent_indices = NULL;
        env->expert_static_agent_indices = NULL;

        for (int i = 0; i < env->num_total_agents; i++) {
            env->active_agent_indices[i] = i;
            env->agents[i].controller = resolve_agent_controller(env, i, 1, 0);
        }

        env->active_agent_count = env->num_total_agents;
        env->num_agents = env->num_total_agents;
        env->static_agent_count = 0;
        env->expert_static_agent_count = 0;

        return;
    }

    // In REPLAY mode, determine which agents to control
    bool is_log_replay = (env->control_mode == CONTROL_SDC_ONLY);
    // In log-replay mode, no cap on actors
    int max_agents = is_log_replay ? env->num_total_agents : env->num_max_agents;

    int *active_agent_indices = (int *)malloc(max_agents * sizeof(int));
    int *static_agent_indices = (int *)malloc(max_agents * sizeof(int));
    int *expert_static_agent_indices = (int *)malloc(max_agents * sizeof(int));

    // In replay mode, agent 0 is the canonical target / SDC for the
    // adversarial setup. Seed it first so downstream code can rely on the
    // invariant that the target occupies active-agent slot 0.
    if (env->simulation_mode == SIMULATION_REPLAY && env->num_total_agents > 0) {
        active_agent_indices[0] = 0;
        env->active_agent_count++;
        env->num_agents++;
        env->agents[0].active_agent = 1;
        env->agents[0].controller = resolve_agent_controller(env, 0, 1, 0);
    }

    // Iterate through entities to find agents to create and/or control
    for (int i = 0; i < env->num_total_agents && env->num_agents < max_agents; i++) {
        if (env->simulation_mode == SIMULATION_REPLAY && i == 0) {
            continue;
        }

        Agent *agent = &env->agents[i];

        // Skip if not valid at initialization
        if (agent->log_valid[env->init_steps] != 1 && !is_log_replay) {
            continue;
        }

        // Determine if entity should be created
        bool should_create = false;
        if (is_log_replay) {
            should_create = true; // Log-replay: all valid agents
        } else if (env->init_mode == INIT_ALL_VALID) {
            should_create = true; // All valid entities
        } else if (env->control_mode == CONTROL_VEHICLES) {
            should_create = (agent->type == VEHICLE);
        } else { // Control all agents
            should_create = (is_controllable_agent(agent->type));
        }

        if (!should_create)
            continue;

        env->num_agents++;

        // Determine if this agent should be policy-controlled
        bool is_controlled = should_control_agent(env, i);

        if (is_controlled) {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->agents[i].active_agent = 1;
            env->agents[i].controller = resolve_agent_controller(env, i, 1, 0);
        } else if (is_log_replay || env->init_mode != INIT_ONLY_CONTROLLABLE_AGENTS) {
            int replay_by_default =
                is_log_replay || env->agents[i].mark_as_expert == 1 || env->active_agent_count == env->num_max_agents;
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->agents[i].active_agent = 0;
            env->agents[i].controller = resolve_agent_controller(env, i, 0, replay_by_default);
            if (env->agents[i].controller == CONTROLLER_REPLAY) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->agents[i].mark_as_expert = 1;
            }
        }
    }

    // Set up initial active agents
    env->active_agent_indices = (int *)malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int *)malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int *)malloc(env->expert_static_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; i++) {
        env->active_agent_indices[i] = active_agent_indices[i];
    };
    for (int i = 0; i < env->static_agent_count; i++) {
        env->static_agent_indices[i] = static_agent_indices[i];
    }
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        env->expert_static_agent_indices[i] = expert_static_agent_indices[i];
    }

    // Free temporary buffers
    free(active_agent_indices);
    free(static_agent_indices);
    free(expert_static_agent_indices);

    if (env->num_controllable_agents > 0 && env->active_agent_count != env->num_controllable_agents) {
        printf("ERROR Between my_shared and init : Mismatch in active agent count: %d vs %d\n", env->active_agent_count,
               env->num_controllable_agents);
    }

    return;
}

void move_expert(Drive *env, float *actions, int agent_idx) {
    if (env->simulation_mode == SIMULATION_GIGAFLOW) {
        printf("[GIGAFLOW ERROR] -> move_expert() called in GIGAFLOW mode\n");
        return;
    }

    bool is_log_replay = (env->control_mode == CONTROL_SDC_ONLY);

    Agent *agent = &env->agents[agent_idx];
    int t = env->timestep;

    // If agent is invalid at this timestep, set simulated state to invalid
    if (t < 0 || t >= agent->trajectory_length || agent->log_valid[t] == 0) {
        invalidate_agent(agent);
        return;
    }

    // Copy from logged trajectory to simulated state
    agent->sim_x = agent->log_trajectory_x[t];
    agent->sim_y = agent->log_trajectory_y[t];
    agent->sim_z = agent->log_trajectory_z[t];
    agent->prev_sim_x = agent->sim_x;
    agent->prev_sim_y = agent->sim_y;
    agent->prev_sim_heading = agent->log_heading[t];
    agent->sim_heading = agent->log_heading[t];
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->sim_valid = agent->log_valid[t];

    if (is_log_replay) {
        agent->sim_length = agent->log_length[t];
        agent->sim_width = agent->log_width[t];
        agent->sim_height = agent->log_height[t];
        agent->wheelbase = 0.6f * agent->sim_length;
    }

    agent->yaw_rate = compute_log_yaw_rate(agent, t, env->dt);
    agent->sim_vx = agent->log_velocity_x[t];
    agent->sim_vy = agent->log_velocity_y[t];

    update_agent_speed(agent);
    agent->sim_valid = agent->log_valid[t];
}

void remove_bad_trajectories(Drive *env) {

    if (env->control_mode == CONTROL_WOSAC) {
        return; // Leave all trajectories in WOSAC control mode
    }

    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; ++i) {
        collided_with_indices[i] = -1;
    }
    // move experts through trajectories to check for collisions and remove as illegal agents
    for (int t = 0; t < env->scenario_length; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            move_expert(env, env->actions, agent_idx);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->agents[expert_idx].sim_x == INVALID_POSITION)
                continue;
            move_expert(env, env->actions, expert_idx);
        }
        // check collisions
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int collided_with_index = collision_check(env, agent_idx, NULL, NULL, NULL);
            if ((collided_with_index >= 0) && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with_index;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        if (collided_with_indices[i] == -1)
            continue;
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_agent_idx = env->static_agent_indices[j];
            if (static_agent_idx != collided_with_indices[i])
                continue;
            env->agents[static_agent_idx].log_trajectory_x[0] = INVALID_POSITION;
            env->agents[static_agent_idx].log_trajectory_y[0] = INVALID_POSITION;
            env->agents[static_agent_idx].log_valid[0] = 0;
        }
    }
    env->timestep = 0;
}

static void init_owned_map_data(Drive *env) {
    env->shared_map = NULL;
    env->owns_map_data = 1;
    env->owns_traffic_data = 1;
    load_map_binary(env->map_name, env);
    env->road_dropout_enabled = (env->obs_lane_segment_count < env->max_lane_segment_observations) ||
                                (env->obs_boundary_segment_count < env->max_boundary_segment_observations);
    init_grid_map(env);
    int vision_half_range = (int)ceilf(
        fmaxf(fmaxf(env->road_obs_front_dist, env->road_obs_behind_dist), env->road_obs_side_dist) / GRID_CELL_SIZE);
    env->grid_map->vision_range = 2 * vision_half_range + 1;
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
}

static void free_grid_map(GridMap *grid_map) {
    if (grid_map == NULL) {
        return;
    }

    int grid_cell_count = grid_map->grid_cols * grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(grid_map->cells[grid_index]);
    }
    free(grid_map->cells);
    free(grid_map->cell_entities_count);
    free(grid_map->grid_index_drivable);

    for (int i = 0; i < grid_cell_count; i++) {
        free(grid_map->neighbor_cache_entities[i]);
    }
    free(grid_map->neighbor_cache_entities);
    free(grid_map->neighbor_cache_count);
    free(grid_map);
}

static void free_drive_map(DriveMap *map) {
    if (map == NULL) {
        return;
    }

    for (int i = 0; i < map->num_road_elements; i++)
        free_road_element(&map->road_elements[i]);
    for (int i = 0; i < map->num_traffic_elements; i++)
        free_traffic_element(&map->traffic_elements[i]);
    free(map->road_elements);
    free(map->traffic_elements);
    free_grid_map(map->grid_map);
    free_lane_graph(&map->lane_graph);
    free(map->map_name);
    free(map);
}

static TrafficControlElement *clone_traffic_elements(const TrafficControlElement *src, int count) {
    if (src == NULL || count <= 0) {
        return NULL;
    }

    TrafficControlElement *dst = (TrafficControlElement *)calloc(count, sizeof(TrafficControlElement));
    if (dst == NULL) {
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        dst[i] = src[i];
        dst[i].states = NULL;
        dst[i].controlled_lanes = NULL;

        if (src[i].state_length > 0 && src[i].states != NULL) {
            dst[i].states = (int *)malloc(src[i].state_length * sizeof(int));
            if (dst[i].states == NULL) {
                for (int j = 0; j < i; j++)
                    free_traffic_element(&dst[j]);
                free(dst);
                return NULL;
            }
            memcpy(dst[i].states, src[i].states, src[i].state_length * sizeof(int));
        }

        if (src[i].num_controlled_lanes > 0 && src[i].controlled_lanes != NULL) {
            dst[i].controlled_lanes = (int *)malloc(src[i].num_controlled_lanes * sizeof(int));
            if (dst[i].controlled_lanes == NULL) {
                for (int j = 0; j <= i; j++)
                    free_traffic_element(&dst[j]);
                free(dst);
                return NULL;
            }
            memcpy(dst[i].controlled_lanes, src[i].controlled_lanes, src[i].num_controlled_lanes * sizeof(int));
        }
    }

    return dst;
}

static DriveMap *drive_map_cache_find(DriveMapCache *cache, const char *map_name) {
    if (cache == NULL || map_name == NULL) {
        return NULL;
    }

    for (int i = 0; i < cache->count; i++) {
        if (cache->maps[i] && cache->maps[i]->map_name && strcmp(cache->maps[i]->map_name, map_name) == 0) {
            cache->cache_hits++;
            return cache->maps[i];
        }
    }

    return NULL;
}

static int drive_map_cache_append(DriveMapCache *cache, DriveMap *map) {
    if (cache->count == cache->capacity) {
        int new_capacity = cache->capacity == 0 ? 8 : cache->capacity * 2;
        DriveMap **new_maps = (DriveMap **)realloc(cache->maps, new_capacity * sizeof(DriveMap *));
        if (new_maps == NULL) {
            return 0;
        }
        cache->maps = new_maps;
        cache->capacity = new_capacity;
    }

    cache->maps[cache->count++] = map;
    return 1;
}

static void free_loaded_agents(Drive *env) {
    for (int i = 0; i < env->num_total_agents; i++)
        free_agent(&env->agents[i]);
    free(env->agents);
}

static DriveMap *drive_map_cache_load(DriveMapCache *cache, const char *map_name, Drive *env) {
    Drive temp = {0};
    temp.map_name = (char *)map_name;
    temp.road_obs_front_dist = env->road_obs_front_dist;
    temp.road_obs_behind_dist = env->road_obs_behind_dist;
    temp.road_obs_side_dist = env->road_obs_side_dist;
    temp.max_lane_segment_observations = env->max_lane_segment_observations;
    temp.max_boundary_segment_observations = env->max_boundary_segment_observations;
    temp.obs_lane_segment_count = env->obs_lane_segment_count;
    temp.obs_boundary_segment_count = env->obs_boundary_segment_count;

    if (load_map_binary(map_name, &temp) != 0) {
        return NULL;
    }

    temp.road_dropout_enabled = (temp.obs_lane_segment_count < temp.max_lane_segment_observations) ||
                                (temp.obs_boundary_segment_count < temp.max_boundary_segment_observations);
    init_grid_map(&temp);
    int vision_half_range = (int)ceilf(
        fmaxf(fmaxf(temp.road_obs_front_dist, temp.road_obs_behind_dist), temp.road_obs_side_dist) / GRID_CELL_SIZE);
    temp.grid_map->vision_range = 2 * vision_half_range + 1;
    init_neighbor_offsets(&temp);
    cache_neighbor_offsets(&temp);

    DriveMap *map = (DriveMap *)calloc(1, sizeof(DriveMap));
    if (map == NULL) {
        free_loaded_agents(&temp);
        for (int i = 0; i < temp.num_road_elements; i++)
            free_road_element(&temp.road_elements[i]);
        for (int i = 0; i < temp.num_traffic_elements; i++)
            free_traffic_element(&temp.traffic_elements[i]);
        free(temp.road_elements);
        free(temp.traffic_elements);
        free_grid_map(temp.grid_map);
        free(temp.neighbor_offsets);
        free_lane_graph(&temp.lane_graph);
        free(temp.objects_of_interest);
        free(temp.tracks_to_predict);
        return NULL;
    }

    map->map_name = strdup(map_name);
    map->road_elements = temp.road_elements;
    map->traffic_elements = temp.traffic_elements;
    map->grid_map = temp.grid_map;
    map->lane_graph = temp.lane_graph;
    map->num_road_elements = temp.num_road_elements;
    map->num_traffic_elements = temp.num_traffic_elements;
    map->num_objects = temp.num_objects;

    free_loaded_agents(&temp);
    free(temp.neighbor_offsets);
    free(temp.objects_of_interest);
    free(temp.tracks_to_predict);

    if (map->map_name == NULL || !drive_map_cache_append(cache, map)) {
        free_drive_map(map);
        return NULL;
    }

    cache->cache_misses++;
    return map;
}

static DriveMap *drive_map_cache_get_or_load(DriveMapCache *cache, const char *map_name, Drive *env) {
    DriveMap *map = drive_map_cache_find(cache, map_name);
    if (map != NULL) {
        return map;
    }
    return drive_map_cache_load(cache, map_name, env);
}

static void free_owned_map_data(Drive *env) {
    if (!env->owns_map_data) {
        return;
    }

    for (int i = 0; i < env->num_road_elements; i++)
        free_road_element(&env->road_elements[i]);
    free(env->road_elements);
    free_grid_map(env->grid_map);
    free(env->neighbor_offsets);
    free_lane_graph(&env->lane_graph);
}

static void free_owned_traffic_data(Drive *env) {
    if (!env->owns_traffic_data) {
        return;
    }

    for (int i = 0; i < env->num_traffic_elements; i++)
        free_traffic_element(&env->traffic_elements[i]);
    free(env->traffic_elements);
}

static int init_cached_gigaflow_map_data(Drive *env) {
    DriveMap *map = drive_map_cache_get_or_load(env->map_cache, env->map_name, env);
    if (map == NULL) {
        return 0;
    }

    env->shared_map = map;
    env->road_elements = map->road_elements;
    env->grid_map = map->grid_map;
    env->lane_graph = map->lane_graph;
    env->num_road_elements = map->num_road_elements;
    env->num_traffic_elements = map->num_traffic_elements;
    env->num_objects = map->num_objects;
    env->traffic_elements = clone_traffic_elements(map->traffic_elements, map->num_traffic_elements);
    if (map->num_traffic_elements > 0 && env->traffic_elements == NULL) {
        return 0;
    }

    env->road_dropout_enabled = (env->obs_lane_segment_count < env->max_lane_segment_observations) ||
                                (env->obs_boundary_segment_count < env->max_boundary_segment_observations);
    env->owns_map_data = 0;
    env->owns_traffic_data = 1;
    env->neighbor_offsets = NULL;
    return 1;
}

void init(Drive *env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    if (env->simulation_mode == SIMULATION_GIGAFLOW && env->map_cache != NULL) {
        if (!init_cached_gigaflow_map_data(env)) {
            init_owned_map_data(env);
        }
    } else {
        init_owned_map_data(env);
    }
    env->logs_capacity = 0;
    set_active_agents(env);
    env->logs_capacity = env->active_agent_count;
    if (env->simulation_mode == SIMULATION_REPLAY) {
        remove_bad_trajectories(env);
    }
    set_start_position(env);
    if (env->simulation_mode == SIMULATION_GIGAFLOW) {
        int steps = env->scenario_length;
        if (steps > 0) {
            for (int i = 0; i < env->num_traffic_elements; i++) {
                TrafficControlElement *traffic = &env->traffic_elements[i];
                if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT)
                    continue;
                if (traffic->states && traffic->state_length != steps) {
                    free(traffic->states);
                    traffic->states = NULL;
                }
                if (traffic->states == NULL) {
                    traffic->states = (int *)malloc(steps * sizeof(int));
                    if (traffic->states == NULL) {
                        traffic->state_length = 0;
                        continue;
                    }
                }
                traffic->state_length = steps;
            }
        }
        generate_traffic_light_states(env);
    }
    env->logs = (Log *)calloc(env->active_agent_count, sizeof(Log));

    if (env->simulation_mode == SIMULATION_REPLAY) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            build_path(env, agent_idx);
            compute_goals(env, agent_idx);
        }
    }
}

void c_close(Drive *env) {
    for (int i = 0; i < env->num_total_agents; i++)
        free_agent(&env->agents[i]);
    free(env->agents);
    free(env->active_agent_indices);
    free(env->logs);
    free_owned_map_data(env);
    free_owned_traffic_data(env);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->objects_of_interest);
    free(env->tracks_to_predict);
    free(env->map_name);
    free(env->ini_file);
}

static int compute_observation_size(Drive *env) {
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int num_target_waypoints = env->num_target_waypoints;
    if (num_target_waypoints > MAX_TARGET_WAYPOINTS) {
        num_target_waypoints = MAX_TARGET_WAYPOINTS;
    }

    int max_obs = ego_dim + PARTNER_FEATURES * env->max_partner_observations +
                  ROAD_FEATURES * (env->obs_lane_segment_count + env->obs_boundary_segment_count) +
                  TRAFFIC_CONTROL_FEATURES * env->max_traffic_control_observations + OBS_VALID_COUNT_FEATURES;
    if (env->reward_conditioning) {
        max_obs += NUM_REWARD_COEFS;
    }
    if (env->target_type == TARGET_STATIC) {
        max_obs += num_target_waypoints * STATIC_TARGET_FEATURES;
    } else if (env->target_type == TARGET_DYNAMIC) {
        max_obs += num_target_waypoints * DYNAMIC_TARGET_FEATURES;
    }

    return max_obs;
}

void allocate(Drive *env) {
    init(env);
    int max_obs = compute_observation_size(env);

    env->observations = (float *)calloc(env->active_agent_count * max_obs, sizeof(float));
    env->actions = (float *)calloc(env->active_agent_count * 2, sizeof(float));
    env->rewards = (float *)calloc(env->active_agent_count, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
    env->truncations = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
    env->masks = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
    free(env->masks);
    c_close(env);
}

// ========================================
// Extra C API Functions
// ========================================

int get_track_id_or_placeholder(Drive *env, int agent_idx) {
    if (env->tracks_to_predict == NULL || env->num_tracks_to_predict == 0) {
        return -1;
    }
    for (int k = 0; k < env->num_tracks_to_predict; k++) {
        if (env->tracks_to_predict[k] == agent_idx) {
            return env->tracks_to_predict[k];
        }
    }
    return -1;
}

void c_get_global_agent_state(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out, int *id_out,
                              float *length_out, float *width_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];

        // For WOSAC, we need the original world coordinates, so we add the world means back
        x_out[i] = agent->sim_x + env->world_mean_x;
        y_out[i] = agent->sim_y + env->world_mean_y;
        z_out[i] = agent->sim_z;
        heading_out[i] = agent->sim_heading;
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        length_out[i] = agent->sim_length;
        width_out[i] = agent->sim_width;
    }
}

void c_get_global_ground_truth_trajectories(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out,
                                            int *valid_out, int *id_out, int *scenario_id_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        scenario_id_out[i] = 0; // TODO: FIXME

        for (int t = env->init_steps; t < agent->trajectory_length; t++) {
            int out_idx = i * (agent->trajectory_length - env->init_steps) + (t - env->init_steps);
            // Add world means back to get original world coordinates
            x_out[out_idx] = agent->log_trajectory_x[t] + env->world_mean_x;
            y_out[out_idx] = agent->log_trajectory_y[t] + env->world_mean_y;
            z_out[out_idx] = agent->log_trajectory_z[t];
            heading_out[out_idx] = agent->log_heading[t];
            valid_out[out_idx] = agent->log_valid[t];
        }
    }
}

void c_get_road_edge_counts(Drive *env, int *num_polylines_out, int *total_points_out) {
    int count = 0, points = 0;
    for (int i = 0; i < env->num_road_elements; i++) {
        if (is_road_edge(env->road_elements[i].type)) {
            count++;
            points += env->road_elements[i].segment_length;
        }
    }
    *num_polylines_out = count;
    *total_points_out = points;
}

void c_get_road_edge_polylines(Drive *env, float *x_out, float *y_out, int *lengths_out, int *scenario_ids_out) {
    int poly_idx = 0, pt_idx = 0;
    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *e = &env->road_elements[i];
        if (is_road_edge(e->type)) {
            lengths_out[poly_idx] = e->segment_length;
            scenario_ids_out[poly_idx] = 0; // TODO: FIXME
            for (int j = 0; j < e->segment_length; j++) {
                x_out[pt_idx] = e->x[j] + env->world_mean_x;
                y_out[pt_idx] = e->y[j] + env->world_mean_y;
                pt_idx++;
            }
            poly_idx++;
        }
    }
}

// ========================================
// Core Simulation Functions
// ========================================

static void record_target_hit_responsibility(Drive *env, int agent_idx, int other_agent_idx, float normal_x,
                                             float normal_y);
static void record_target_collision_diagnostics(Drive *env, int target_agent_idx, int other_agent_idx);

static void compute_metrics(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    reset_agent_metrics(env, agent_idx);

    if (agent->sim_x == INVALID_POSITION)
        return; // invalid agent position

    bool episode_started = env->timestep > env->init_steps;

    // Current agent is offgrid, treat as offroad
    if (get_grid_index(env, agent->sim_x, agent->sim_y) == -1) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        if (episode_started) {
            int target_agent_idx = env->active_agent_count > 0 ? env->active_agent_indices[0] : -1;
            if (env->remove_target_on_collision_or_offroad && agent_idx == target_agent_idx) {
                agent->removed = 1;
            } else if (env->offroad_behavior == STOP_AGENT && !agent->stopped) {
                agent->stopped = 1;
            } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
                agent->removed = 1;
            }
        }
        return;
    }

    // Compute log-replay metrics
    if (env->simulation_mode == SIMULATION_REPLAY) {
        // Compute displacement error
        float displacement_error = compute_displacement_error(agent, env->timestep);
        if (displacement_error > 0.0f) { // Only count valid displacements
            agent->cumulative_displacement += displacement_error;
            agent->displacement_sample_count++;

            // Compute running average
            agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] =
                agent->cumulative_displacement / agent->displacement_sample_count;
        }
    }

    bool is_offroad = false;
    float half_width = agent->sim_width / 2.0f;

    // Track best candidate by combined distance/heading score
    float best_score = 1e9f;
    int best_candidate_entity_idx = -1;
    int best_candidate_geometry_idx = -1;
    float best_candidate_signed_lane_distance = 0.0f;
    float best_candidate_lane_heading = 0.0f;

    float corners[4][2];
    compute_agent_corners(agent, corners);

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25]; // Array big enough for all neighboring cells
    int list_size = get_neighbors_entities(env, agent->sim_x, agent->sim_y, entity_list, MAX_ENTITIES_PER_CELL * 25,
                                           collision_offsets, 25);

    // Vehicle-width based distance threshold (3x width)
    float max_distance_threshold = 3.0f * agent->sim_width;

    // Track already-checked drivable lanes to avoid redundant processing
    int checked_lanes[MAX_CHECKED_LANES];
    int num_checked_lanes = 0;

    // Loop through road entities and compute associated metrics (offroad, lane alignment)
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1)
            continue;

        // Get the road element (only road elements are in grid)
        if (entity_list[i].entity_type != ENTITY_TYPE_ROAD_ELEMENT)
            continue;

        int entity_idx = entity_list[i].entity_idx;
        int geometry_idx = entity_list[i].geometry_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];

        // Check for offroad collision with road edges
        if (is_road_edge(element->type)) {
            float start[2] = {element->x[geometry_idx], element->y[geometry_idx]};
            float end[2] = {element->x[geometry_idx + 1], element->y[geometry_idx + 1]};
            float abs_dz = fabsf(element->z[geometry_idx] - agent->sim_z);
            if (abs_dz > Z_BUFFER)
                continue;
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    is_offroad = true;
                    break;
                }
            }
        }

        if (is_offroad)
            break;

        int closest_seg_idx;
        float signed_dist_out;
        float lane_heading_out;
        float score = score_lane_candidate(env, agent->sim_heading, checked_lanes, &num_checked_lanes, &closest_seg_idx,
                                           entity_list[i], agent->sim_x, agent->sim_y, max_distance_threshold,
                                           agent->current_lane_idx, &signed_dist_out, &lane_heading_out);

        if (score >= 0.0f && score < best_score) {
            best_score = score;
            best_candidate_entity_idx = entity_idx;
            best_candidate_geometry_idx = closest_seg_idx;
            best_candidate_signed_lane_distance = signed_dist_out;
            best_candidate_lane_heading = lane_heading_out;
        }
    }

    // Update lane alignment metric (running average)
    if (best_candidate_entity_idx != -1) {
        agent->previous_lane_idx = agent->current_lane_idx;
        agent->current_lane_idx = best_candidate_entity_idx;
        agent->current_lane_geometry_idx = best_candidate_geometry_idx;

        // Lane distance and angle metrics (GIGAFLOW Frenet coordinates)
        // x_f = lateral offset from lane center (left = negative, right = positive)
        agent->metrics_array[LANE_DIST_IDX] = best_candidate_signed_lane_distance;
        // Multi-lane detection: vehicle edge exceeds lane boundary
        float edge_dist = fabsf(best_candidate_signed_lane_distance) + half_width;
        if (env->compute_eval_metrics && edge_dist > MULTI_LANE_THRESHOLD && agent->sim_speed > 0.0f) {
            agent->multi_lane_time += env->dt;
        }
        // theta_f = angle relative to lane heading
        float theta_f = compute_heading_diff(agent->sim_heading, best_candidate_lane_heading);
        agent->metrics_array[LANE_ANGLE_IDX] = cosf(theta_f); // Store cos(θ_f)
    } else {
        // Agent not on any lane - use "bad" values to indicate offroad state
        agent->previous_lane_idx = -1;
        agent->current_lane_idx = -1;
        agent->current_lane_geometry_idx = -1;
        agent->metrics_array[LANE_DIST_IDX] = LANE_DISTANCE_NORMALIZATION; // Max distance (far from lane)
        agent->metrics_array[LANE_ANGLE_IDX] = 0.0f;                       // Perpendicular (no alignment)
    }

    agent->closest_path_idx_wp = get_closest_waypoint_index_on_path(env, agent_idx);

    // Speed limit metric (CUSTOM)
    float target_speed = 15.0f; // Default target speed
    int current_lane_idx = agent->current_lane_idx;
    if (current_lane_idx != -1 && env->road_elements[current_lane_idx].speed_limit > 0)
        target_speed = env->road_elements[current_lane_idx].speed_limit;

    // Binary overspeed metric, 1.0 if overspeeding by more than 2 m/s
    agent->metrics_array[SPEED_LIMIT_IDX] = (agent->sim_speed > target_speed + 2.0f) ? 1.0f : 0.0f;
    if (env->compute_eval_metrics) {
        agent->speed_violation_sum += fmaxf(agent->sim_speed - target_speed, 0.0f) * env->dt;
    }

    // Velocity metric (GIGAFLOW) - forward progress aligned with lane
    const float VELOCITY_MIN_SPEED = 2.5f; // m/s
    if (agent->sim_speed_signed > VELOCITY_MIN_SPEED && best_candidate_entity_idx != -1) {
        float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = fmaxf(cos_theta, 0.0f);
        if (env->compute_eval_metrics && cos_theta < 0.0f) {
            agent->wrong_way_distance += agent->sim_speed_signed * env->dt;
        }
    } else {
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = 0.0f;
    }

    // Comfort metric (GIGAFLOW)
    int longitudinal_accel_violation = fabsf(agent->a_long) > COMFORT_ACCEL_THRESHOLD;
    int lateral_accel_violation = fabsf(agent->a_lat) > COMFORT_ACCEL_THRESHOLD;
    int jerk_violation =
        (fabsf(agent->jerk_long) > COMFORT_JERK_THRESHOLD || fabsf(agent->jerk_lat) > COMFORT_JERK_THRESHOLD) ? 1 : 0;
    agent->metrics_array[COMFORT_VIOLATION_IDX] =
        (float)(longitudinal_accel_violation + lateral_accel_violation + jerk_violation);

    // Handle terminal events - NOTE: move it elsewhere?
    // IMPORTANT: early returns after offroad and collision enforce mutual exclusivity of terminal flags.
    // Order matters: offroad > collision > red_light.

    // Priority 1: Handle offroad
    if (is_offroad) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        int target_agent_idx = env->active_agent_count > 0 ? env->active_agent_indices[0] : -1;
        if (!episode_started) {
            return;
        } else if (env->remove_target_on_collision_or_offroad && agent_idx == target_agent_idx) {
            agent->removed = 1;
        } else if (env->offroad_behavior == STOP_AGENT && !agent->stopped) { // Stop
            agent->stopped = 1;
        } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
        }
        return; // early return: no other terminal flags set when offroad
    }

    // Priority 2: Handle vehicle collision
    agent->collided_with_agent_idx = -1;
    float collision_normal_x = 0.0f;
    float collision_normal_y = 0.0f;
    float collision_penetration = 0.0f;
    int car_collided_with_index =
        collision_check(env, agent_idx, &collision_normal_x, &collision_normal_y, &collision_penetration);

    if (car_collided_with_index != -1) {
        agent->metrics_array[COLLISION_IDX] = 1.0f;
        agent->collided_with_agent_idx = car_collided_with_index;
        if (env->compute_eval_metrics) {
            Agent *other_agent = &env->agents[car_collided_with_index];
            agent->collision_normal_x = collision_normal_x;
            agent->collision_normal_y = collision_normal_y;
            agent->collision_impact_zone =
                classify_impact_zone_from_normal(agent, collision_normal_x, collision_normal_y);
            evaluate_collision_pair(agent, other_agent, collision_normal_x, collision_normal_y,
                                    &agent->collision_severity, &agent->collision_responsibility);
            (void)collision_penetration;
        }
        // Track at-fault collisions for evaluation metrics.
        if (env->compute_eval_metrics && is_at_fault_collision(env, agent_idx, car_collided_with_index)) {
            agent->at_fault_collision = 1;
            agent->metrics_array[AT_FAULT_COLLISION_IDX] = 1.0f;
        }
        int target_agent_idx = env->active_agent_count > 0 ? env->active_agent_indices[0] : -1;
        bool target_involved_collision = agent_idx == target_agent_idx || car_collided_with_index == target_agent_idx;
        if (episode_started && target_involved_collision) {
            record_target_hit_responsibility(env, agent_idx, car_collided_with_index, collision_normal_x,
                                             collision_normal_y);
            int other_idx = (agent_idx == target_agent_idx) ? car_collided_with_index : agent_idx;
            record_target_collision_diagnostics(env, target_agent_idx, other_idx);
            if (target_agent_idx >= 0 && other_idx >= 0 &&
                env->target_avoidability_by_braking_episode == TARGET_AVOIDABILITY_UNSET) {
                env->target_avoidability_by_braking_episode =
                    last_avoidable_time_by_braking(env, target_agent_idx, other_idx);
            }
            if (target_agent_idx >= 0 && other_idx >= 0 && is_at_fault_collision(env, target_agent_idx, other_idx)) {
                env->target_hit_at_fault_this_step = 1;
            }
        }
        if (episode_started && env->remove_target_on_collision_or_offroad && target_involved_collision &&
            target_agent_idx >= 0) {
            env->agents[target_agent_idx].removed = 1;
        }
        bool ignore_collision_behavior = target_involved_collision && (env->ignore_target_collision_behavior ||
                                                                       env->remove_target_on_collision_or_offroad);
        if (episode_started && !ignore_collision_behavior) {
            if (env->collision_behavior == STOP_AGENT && !agent->stopped) { // Stop
                agent->stopped = 1;
            } else if (env->collision_behavior == REMOVE_AGENT && !agent->removed) {
                agent->removed = 1;
            }
        }

        return; // early return: red_light not checked after collision
    }

    // Priority 3: Handle red light violation
    if (env->max_traffic_control_observations && check_red_light_violation(env, agent_idx)) {
        agent->metrics_array[RED_LIGHT_IDX] = 1.0f;
        if (episode_started) {
            if (env->traffic_light_behavior == STOP_AGENT && !agent->stopped) {
                agent->stopped = 1;
            } else if (env->traffic_light_behavior == REMOVE_AGENT && !agent->removed) {
                agent->removed = 1;
            }
        }

        return; // early return: no goal reaching when red light violation
    }

    float distance_to_goal =
        compute_euclidean_distance(agent->sim_x, agent->sim_y, agent->goal_position_x, agent->goal_position_y);
    float goal_z_dist = fabsf(agent->sim_z - agent->goal_position_z);

    // Goal reaching
    if (distance_to_goal < agent->reward_coefs[REWARD_COEF_GOAL_RADIUS] && goal_z_dist < Z_BUFFER) {
        agent->metrics_array[REACHED_GOAL_IDX] = 1.0f;
        agent->current_goal_idx++;
    }

    return;
}

static void record_target_hit_responsibility(Drive *env, int agent_idx, int other_agent_idx, float normal_x,
                                             float normal_y) {
    if (env->active_agent_count <= 0)
        return;

    int target_agent_idx = env->active_agent_indices[0];
    if (agent_idx != target_agent_idx && other_agent_idx != target_agent_idx)
        return;

    int hitter_idx = (agent_idx == target_agent_idx) ? other_agent_idx : agent_idx;
    if (!is_adversarial_agent(env, hitter_idx))
        return;

    Agent *hitter = &env->agents[hitter_idx];
    Agent *target = &env->agents[target_agent_idx];
    float hitter_responsibility = 0.0f;
    if (agent_idx == hitter_idx) {
        evaluate_collision_pair(hitter, target, normal_x, normal_y, NULL, &hitter_responsibility);
    } else {
        evaluate_collision_pair(target, hitter, normal_x, normal_y, NULL, &hitter_responsibility);
        hitter_responsibility = 1.0f - hitter_responsibility;
    }

    float target_responsibility = fmaxf(0.0f, fminf(1.0f, 1.0f - hitter_responsibility));
    if (!env->target_hit_this_step || target_responsibility > env->target_hit_responsibility_this_step) {
        env->target_hit_this_step = 1;
        env->target_hit_hitter_idx_this_step = hitter_idx;
        env->target_hit_responsibility_this_step = target_responsibility;
    }
}

static void record_target_collision_diagnostics(Drive *env, int target_agent_idx, int other_agent_idx) {
    if (!env->compute_eval_metrics)
        return;

    if (env->target_collision_type_episode != COLLISION_TYPE_NONE)
        return;

    if (target_agent_idx < 0 || other_agent_idx < 0)
        return;

    Agent *target = &env->agents[target_agent_idx];
    Agent *other = &env->agents[other_agent_idx];
    float rel_vx = target->sim_vx - other->sim_vx;
    float rel_vy = target->sim_vy - other->sim_vy;

    env->target_collision_type_episode = classify_collision_type(target, other);
    env->target_collision_target_speed_episode = target->sim_speed;
    env->target_collision_other_speed_episode = other->sim_speed;
    env->target_collision_relative_speed_episode = sqrtf(rel_vx * rel_vx + rel_vy * rel_vy);
    env->target_collision_other_stopped_episode = other->stopped ? 1.0f : 0.0f;
    env->target_collision_other_removed_episode = other->removed ? 1.0f : 0.0f;
    env->target_collision_other_active_episode = (!other->stopped && !other->removed) ? 1.0f : 0.0f;
}

static RewardTerms compute_rewards(Drive *env, int i) {
    int agent_idx = env->active_agent_indices[i];
    Agent *agent = &env->agents[agent_idx];
    RewardTerms terms = {0};

    // NOTE: compute_metrics enforces offroad > collision priority via early returns.
    // All penalty terms below are applied independently (no early returns here).

    // Collision reward (GIGAFLOW)
    if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
        int target_agent_idx = (env->active_agent_count > 0) ? env->active_agent_indices[0] : -1;
        int collided_with_idx = agent->collided_with_agent_idx;
        Agent *target_agent = (target_agent_idx >= 0) ? &env->agents[target_agent_idx] : NULL;
        bool hit_inactive_target = (collided_with_idx == target_agent_idx) && target_agent != NULL &&
                                   (target_agent->stopped || target_agent->removed);
        bool apply_collision_penalty =
            (agent_idx == target_agent_idx) || is_adversarial_agent(env, collided_with_idx) || hit_inactive_target;

        // Velocity-dependent penalty: incentivizes braking before unavoidable collision.
        // At max speed (~20 m/s): extra -2.0 on top of base coefficient.
        float reward_collision =
            apply_collision_penalty ? -(agent->reward_coefs[REWARD_COEF_COLLISION] + 0.1f * agent->sim_speed) : 0.0f;

        terms.collision += reward_collision;
        env->logs[i].collision_rate = 1.0f;
    }

    // Offroad reward (GIGAFLOW)
    if (agent->metrics_array[OFFROAD_IDX] > 0.0f) {
        float reward_offroad = -agent->reward_coefs[REWARD_COEF_OFFROAD];

        terms.offroad += reward_offroad;
        env->logs[i].offroad_rate = 1.0f;
    }

    // Red light violation reward (GIGAFLOW)
    if (agent->metrics_array[RED_LIGHT_IDX] > 0.0f) {
        float reward_red_light = -agent->reward_coefs[REWARD_COEF_STOP_LINE];

        terms.red_light += reward_red_light;
        env->logs[i].red_light_violation_rate = 1.0f;
    }

    // Goal reward
    if (agent->metrics_array[REACHED_GOAL_IDX] > 0.0f) {
        float weight = 1.0f;
        if (env->simulation_mode == SIMULATION_GIGAFLOW) {
            if (agent->current_goal_idx == env->num_target_waypoints &&
                agent->sim_speed > agent->reward_coefs[REWARD_COEF_GOAL_SPEED])
                weight = 0.0f;
        }

        terms.drive += env->reward_goal * weight;
        env->logs[i].num_waypoints_reached += 1;
    }

    // Get lane angle metric: cos(θ_f) where θ_f = heading diff from lane
    float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
    float theta_f = acosf(fminf(fmaxf(cos_theta, -1.0f), 1.0f)); // Get |θ_f| from cos
    env->logs[i].lane_heading_aligned_rate += (cos_theta >= LANE_ALIGN_COS_THRESHOLD) ? 1.0f : 0.0f;

    // Rl-align (GIGAFLOW): min(cos,0) + vel_align*min(cos*v,0) + 0.0025*(1-|θ|/(π/2))
    float against_lane_penalty = fminf(cos_theta, 0.0f); // negative when >90 degrees off
    float vel_aligned_penalty =
        agent->reward_coefs[REWARD_COEF_VEL_ALIGN] * fminf(cos_theta * agent->sim_speed_signed, 0.0f);
    float alignment_bonus = 0.0025f * (1.0f - theta_f / (M_PI / 2.0f));

    float lane_align_reward = agent->reward_coefs[REWARD_COEF_LANE_ALIGN] * env->dt *
                              (against_lane_penalty + vel_aligned_penalty + alignment_bonus);

    terms.drive += lane_align_reward;

    // Rl-center (GIGAFLOW): -α * dt * (|x_f - bias| - 0.05 / exp(|x_f - bias| - 0.5))
    float lane_center_distance = agent->metrics_array[LANE_DIST_IDX];
    float adjusted_dist = fabsf(lane_center_distance - agent->reward_coefs[REWARD_COEF_CENTER_BIAS]);
    float exp_decay = 0.05f / expf(adjusted_dist - 0.5f);

    float lane_center_reward =
        -agent->reward_coefs[REWARD_COEF_LANE_CENTER] * env->dt * ((cos_theta > 0.5f) * adjusted_dist - exp_decay);

    env->logs[i].lane_center_rate += fabsf(lane_center_distance) < 0.5f ? 1.0f : 0.0f;
    terms.drive += lane_center_reward;

    // Comfort reward (GIGAFLOW)
    float comfort_violations = agent->metrics_array[COMFORT_VIOLATION_IDX];
    float comfort_penalty = -agent->reward_coefs[REWARD_COEF_COMFORT] * comfort_violations;

    env->logs[i].comfort_violation_count += comfort_violations;
    env->logs[i].comfort_longitudinal_accel_violation_count +=
        fabsf(agent->a_long) > COMFORT_ACCEL_THRESHOLD ? 1.0f : 0.0f;
    env->logs[i].comfort_lateral_accel_violation_count += fabsf(agent->a_lat) > COMFORT_ACCEL_THRESHOLD ? 1.0f : 0.0f;
    env->logs[i].comfort_jerk_violation_count +=
        (fabsf(agent->jerk_long) > COMFORT_JERK_THRESHOLD || fabsf(agent->jerk_lat) > COMFORT_JERK_THRESHOLD) ? 1.0f
                                                                                                              : 0.0f;
    env->logs[i].uncomfortable_timestep_count += comfort_violations > 0.0f ? 1.0f : 0.0f;
    terms.drive += comfort_penalty;

    // Velocity reward (GIGAFLOW)
    float velocity_progress = agent->metrics_array[VELOCITY_PROGRESS_IDX];
    float velocity_reward = agent->reward_coefs[REWARD_COEF_VELOCITY] * env->dt * velocity_progress;

    terms.drive += velocity_reward;
    env->logs[i].velocity_progress_sum += velocity_progress;

    // Timestep reward (GIGAFLOW)
    float accel = sqrtf(agent->a_long * agent->a_long + agent->a_lat * agent->a_lat);
    // Only penalize when moving (v > 0) or accelerating (a > 0)
    if (agent->sim_speed > 0.01f || accel > 0.01f) {
        float timestep_penalty = -agent->reward_coefs[REWARD_COEF_TIMESTEP] * env->dt;

        terms.drive += timestep_penalty;
    }

    // Reverse reward (GIGAFLOW)
    if (agent->sim_speed_signed < -0.01f) {
        float reverse_penalty = -agent->reward_coefs[REWARD_COEF_REVERSE] * env->dt;

        terms.drive += reverse_penalty;
    }

    // Over speed reward (GIGAFLOW++)
    float speed_reward = -agent->reward_coefs[REWARD_COEF_OVERSPEED] * agent->metrics_array[SPEED_LIMIT_IDX];

    env->logs[i].avg_speed_per_agent += agent->sim_speed;
    agent->distance_since_spawn += agent->sim_speed * env->dt;
    terms.drive += speed_reward;

    // ADE reward (CUSTOM)
    float current_ade = agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX];
    if (current_ade > 0.0f && env->reward_ade != 0.0f) {
        float ade_reward = env->reward_ade * current_ade;

        terms.drive += ade_reward;
    }
    env->logs[i].avg_displacement_error = current_ade;

    if (env->compute_eval_metrics) {
        if (agent->at_fault_collision > 0) {
            env->logs[i].at_fault_collision_rate = 1.0f;
        }

        env->logs[i].wrong_way_distance = agent->wrong_way_distance;
        env->logs[i].speed_violation_sum = agent->speed_violation_sum;
        env->logs[i].multi_lane_time = agent->multi_lane_time;
        float ml_time = env->logs[i].multi_lane_time;
        float ml_score = (ml_time <= MULTI_LANE_FULL_SCORE_TIME)   ? 1.0f
                         : (ml_time <= MULTI_LANE_HALF_SCORE_TIME) ? 0.5f
                                                                   : 0.0f;
        env->logs[i].multi_lane_score = ml_score;
        agent->metrics_array[MULTI_LANE_TIME_IDX] = ml_time;
        agent->metrics_array[MULTI_LANE_SCORE_IDX] = ml_score;

        compute_agent_ttc(env, agent_idx);
        if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
            agent->cached_ttc.min_ttc = 0.0f;
            agent->cached_ttc.other_idx = -1;
            agent->cached_ttc.distance_to_collision = 0.0f;
            agent->cached_ttc.closing_speed = INFINITY;
        }
        struct ttc_result ttc_agents = agent->cached_ttc;
        float min_vehicle_ttc = ttc_agents.min_ttc;
        agent->metrics_array[TTC_IDX] = min_vehicle_ttc;
        agent->metrics_array[TTC_TFL_IDX] = DEFAULT_TTC;
        agent->ttc_samples++;
        if (min_vehicle_ttc < TTC_VIOLATION_THRESHOLD) {
            agent->ttc_violations++;
        }

        env->logs[i].ttc_violations = (float)agent->ttc_violations;
        env->logs[i].ttc_samples = (float)agent->ttc_samples;
        if (agent->ttc_samples > 0) {
            env->logs[i].ttc_within_bound_rate = 1.0f - ((float)agent->ttc_violations / (float)agent->ttc_samples);
        } else {
            env->logs[i].ttc_within_bound_rate = 1.0f;
        }

        env->logs[i].comfort_score = (env->logs[i].comfort_violation_count > 0) ? 0.0f : 1.0f;
    } else {
        struct ttc_result default_ttc = default_ttc_result();
        agent->metrics_array[TTC_IDX] = default_ttc.min_ttc;
        agent->metrics_array[TTC_TFL_IDX] = default_ttc.min_ttc;
        agent->metrics_array[MULTI_LANE_TIME_IDX] = 0.0f;
        agent->metrics_array[MULTI_LANE_SCORE_IDX] = 0.0f;
        agent->metrics_array[AT_FAULT_COLLISION_IDX] = 0.0f;
    }

    return terms;
}

static void compute_observations(Drive *env) {
    int max_obs = compute_observation_size(env);

    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &observations[i][0];
        int obs_idx = 0;
        Agent *ego_entity = &env->agents[env->active_agent_indices[i]];

        // ====== Ego observations ======
        // Ego speed
        obs[obs_idx++] = ego_entity->sim_speed_signed / MAX_SPEED;
        // Ego width
        obs[obs_idx++] = ego_entity->sim_width / env->max_veh_width;
        // Ego length
        obs[obs_idx++] = ego_entity->sim_length / env->max_veh_len;
        // Ego steering angle
        obs[obs_idx++] = ego_entity->steering_angle / STEERING_LIMIT;
        if (env->dynamics_model == JERK) {
            // Ego longitudinal acceleration
            obs[obs_idx++] = ego_entity->a_long / fabsf(ACCEL_LONG_LIMIT[0]);
            // Ego lateral acceleration
            obs[obs_idx++] = ego_entity->a_lat / ACCEL_LAT_LIMIT[1];
        }
        // Ego distance to lane center
        obs[obs_idx++] =
            fmaxf(-1.0f, fminf(1.0f, ego_entity->metrics_array[LANE_DIST_IDX] / LANE_DISTANCE_NORMALIZATION));
        // Ego angle to lane heading (cos(θ_f))
        obs[obs_idx++] = ego_entity->metrics_array[LANE_ANGLE_IDX];
        // Current lane speed limit
        float lane_speed_limit =
            (ego_entity->current_lane_idx != -1) ? env->road_elements[ego_entity->current_lane_idx].speed_limit : -1.0f;
        obs[obs_idx++] = lane_speed_limit / MAX_SPEED;

        // ====== Reward conditioning and target observations ======
        if (env->reward_conditioning) {
            for (int c = 0; c < NUM_REWARD_COEFS; c++) {
                // Normalize to [0, 1]
                float normalized = (ego_entity->reward_coefs[c] - REWARD_BOUNDS[c].min_val) /
                                   ((REWARD_BOUNDS[c].max_val - REWARD_BOUNDS[c].min_val) + 1e-8f);
                // Clamp to [0, 1] to handle floating point noise
                normalized = fmaxf(0.0f, fminf(1.0f, normalized));
                // Scale to [-1, 1]
                obs[obs_idx++] = 2.0f * normalized - 1.0f;
            }
        }
        // Target observations (static or dynamic)
        if (env->target_type == TARGET_STATIC) {
            for (int wp = 0; wp < env->num_target_waypoints; wp++) {
                if (wp < ego_entity->current_goal_idx) {
                    // Already reached - zeroed
                    obs[obs_idx++] = 0.0f;
                    obs[obs_idx++] = 0.0f;
                    obs[obs_idx++] = 0.0f;
                } else {
                    float gx = ego_entity->goal_positions_x[wp] - ego_entity->sim_x;
                    float gy = ego_entity->goal_positions_y[wp] - ego_entity->sim_y;
                    obs[obs_idx++] =
                        (gx * ego_entity->cos_heading + gy * ego_entity->sin_heading) / env->max_goal_position;
                    obs[obs_idx++] =
                        (-gx * ego_entity->sin_heading + gy * ego_entity->cos_heading) / env->max_goal_position;
                    obs[obs_idx++] = (ego_entity->goal_positions_z[wp] - ego_entity->sim_z) / Z_BUFFER;
                }
            }
        } else if (env->target_type == TARGET_DYNAMIC) {
            if (ego_entity->path != NULL && ego_entity->path->num_waypoints > 0) {
                for (int wp = 0; wp < env->num_target_waypoints; wp++) {
                    int wp_index = fmin(ego_entity->closest_path_idx_wp + wp, ego_entity->path->num_waypoints - 1);
                    if (wp_index < 0)
                        wp_index = 0;
                    struct Waypoint *wp = &ego_entity->path->waypoints[wp_index];
                    float wp_x = wp->x - ego_entity->sim_x;
                    float wp_y = wp->y - ego_entity->sim_y;
                    float wp_z = wp->z - ego_entity->sim_z;
                    // Use pre-computed trig values from build_path
                    float wp_cos_h = wp->cos_heading;
                    float wp_sin_h = wp->sin_heading;
                    float rel_wp_x = wp_x * ego_entity->cos_heading + wp_y * ego_entity->sin_heading;
                    float rel_wp_y = -wp_x * ego_entity->sin_heading + wp_y * ego_entity->cos_heading;
                    float rel_heading_x = wp_cos_h * ego_entity->cos_heading + wp_sin_h * ego_entity->sin_heading;
                    float rel_heading_y = wp_sin_h * ego_entity->cos_heading - wp_cos_h * ego_entity->sin_heading;
                    obs[obs_idx++] = rel_wp_x / env->max_position;
                    obs[obs_idx++] = rel_wp_y / env->max_position;
                    obs[obs_idx++] = wp_z / Z_BUFFER;
                    obs[obs_idx++] = rel_heading_x;
                    obs[obs_idx++] = rel_heading_y;
                }
            } else {
                // No valid path - zero out
                obs_idx += DYNAMIC_TARGET_FEATURES * env->num_target_waypoints;
            }
        }

        // ===== Partner observations =====
        int partner_observation_limit = env->max_partner_observations;
        if (env->sdc_controller == CONTROLLER_POLICY) {
            int is_target = env->active_agent_indices[i] == env->active_agent_indices[0];
            partner_observation_limit =
                is_target ? env->target_max_partner_observations : env->adversary_max_partner_observations;
            if (partner_observation_limit > env->max_partner_observations) {
                partner_observation_limit = env->max_partner_observations;
            }
        }
        int cars_seen = 0;
        if (ego_entity->is_blind_partner) {
            int total_partner_floats = env->max_partner_observations * PARTNER_FEATURES;
            memset(&obs[obs_idx], 0, total_partner_floats * sizeof(float));
            obs_idx += total_partner_floats;
        } else {
            // Collect candidate agents within max observation distance, then sort and select closest ones.
            AgentDistance candidates[env->num_agents];
            int candidate_count = 0;
            for (int j = 0; j < env->num_agents; j++) {
                int index = -1;
                if (j < env->active_agent_count) {
                    index = env->active_agent_indices[j];
                } else if (j < env->num_agents) {
                    index = env->static_agent_indices[j - env->active_agent_count];
                }
                if (index == -1)
                    continue;
                if (index == env->active_agent_indices[i])
                    continue; // Skip self, but don't increment obs_idx
                Agent *other_entity = &env->agents[index];
                // Store original relative positions
                float dx = other_entity->sim_x - ego_entity->sim_x;
                float dy = other_entity->sim_y - ego_entity->sim_y;
                float dz = other_entity->sim_z - ego_entity->sim_z;
                float abs_dz = fabsf(dz);
                float dist_sq = dx * dx + dy * dy + dz * dz;
                if (dist_sq > env->agent_obs_max_dist * env->agent_obs_max_dist || abs_dz > Z_BUFFER)
                    continue;
                // Add to candidate list
                candidates[candidate_count].index = index;
                candidates[candidate_count].dist_sq = dist_sq;
                candidates[candidate_count].dx = dx;
                candidates[candidate_count].dy = dy;
                candidates[candidate_count].dz = dz;
                candidate_count++;
            }
            if (candidate_count > 0) {
                int num_agents_to_observe =
                    (candidate_count < partner_observation_limit) ? candidate_count : partner_observation_limit;

                // Partial selection sort: find the k-th smallest for each k
                for (int k = 0; k < num_agents_to_observe; k++) {
                    // Find minimum in remaining elements [k, candidate_count)
                    int min_idx = k;
                    for (int j = k + 1; j < candidate_count; j++) {
                        if (candidates[j].dist_sq < candidates[min_idx].dist_sq) {
                            min_idx = j;
                        }
                    }
                    // Swap to position k
                    if (min_idx != k) {
                        AgentDistance tmp = candidates[k];
                        candidates[k] = candidates[min_idx];
                        candidates[min_idx] = tmp;
                    }
                }

                for (int k = 0; k < num_agents_to_observe; k++) {
                    // Get the data for the k-th closest agent
                    int index = candidates[k].index;
                    float dx = candidates[k].dx;
                    float dy = candidates[k].dy;
                    float dz = candidates[k].dz;
                    Agent *other_entity = &env->agents[index];
                    // Rotate to ego vehicle's frame
                    float rel_x = dx * ego_entity->cos_heading + dy * ego_entity->sin_heading;
                    float rel_y = -dx * ego_entity->sin_heading + dy * ego_entity->cos_heading;
                    // Rotate heading to ego frame using trig difference identities (cos(a-b), sin(a-b))
                    float rel_heading_x = other_entity->cos_heading * ego_entity->cos_heading +
                                          other_entity->sin_heading * ego_entity->sin_heading;
                    float rel_heading_y = other_entity->sin_heading * ego_entity->cos_heading -
                                          other_entity->cos_heading * ego_entity->sin_heading;

                    // Partner center x
                    obs[obs_idx++] = rel_x / env->max_position;
                    // Partner center y
                    obs[obs_idx++] = rel_y / env->max_position;
                    // Partner center z
                    obs[obs_idx++] = dz / Z_BUFFER;
                    // Partner length
                    obs[obs_idx++] = other_entity->sim_length / env->max_veh_len;
                    // Partner width
                    obs[obs_idx++] = other_entity->sim_width / env->max_veh_width;
                    // Partner orientation (cosine)
                    obs[obs_idx++] = rel_heading_x;
                    // Partner orientation (sine)
                    obs[obs_idx++] = rel_heading_y;
                    // Partner speed
                    obs[obs_idx++] = other_entity->sim_speed_signed / MAX_SPEED;
                    // Target marker for adversarial policy.
                    obs[obs_idx++] = (index == env->active_agent_indices[0]) ? 1.0f : 0.0f;
                    // Alive marker for adversarial policy.
                    obs[obs_idx++] = (other_entity->stopped || other_entity->removed) ? 0.0f : 1.0f;
                    cars_seen++;
                }
            }
            int remaining_partner_obs = (env->max_partner_observations - cars_seen) * PARTNER_FEATURES;
            memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
            obs_idx += remaining_partner_obs;
        }

        // ===== Road observations =====
        int grid_idx = get_grid_index(env, ego_entity->sim_x, ego_entity->sim_y);
        int list_size = 0;
        const GridMapEntity *entity_list = NULL;
        if (!(grid_idx < 0 || grid_idx >= (env->grid_map->grid_cols * env->grid_map->grid_rows))) {
            list_size = env->grid_map->neighbor_cache_count[grid_idx];
            entity_list = env->grid_map->neighbor_cache_entities[grid_idx];
        }

        int lane_obs_idx = obs_idx;
        int boundary_obs_idx = lane_obs_idx + env->obs_lane_segment_count * ROAD_FEATURES;
        obs_idx = boundary_obs_idx + env->obs_boundary_segment_count * ROAD_FEATURES;

        // Slow path collects into scratch so we can sub-sample; fast path writes straight to obs[].
        float lanes_buffer[env->max_lane_segment_observations * ROAD_FEATURES];
        float boundaries_scratch[env->max_boundary_segment_observations * ROAD_FEATURES];
        float *lanes_dest = env->road_dropout_enabled ? lanes_buffer : &obs[lane_obs_idx];
        float *boundaries_dest = env->road_dropout_enabled ? boundaries_scratch : &obs[boundary_obs_idx];
        int lanes_collected = 0;
        int boundaries_collected = 0;

        for (int k = 0; k < list_size; k++) {
            if (lanes_collected >= env->max_lane_segment_observations &&
                boundaries_collected >= env->max_boundary_segment_observations) {
                break;
            }
            int entity_type = entity_list[k].entity_type;
            int entity_idx = entity_list[k].entity_idx;
            int geometry_idx = entity_list[k].geometry_idx;

            if (entity_type != ENTITY_TYPE_ROAD_ELEMENT)
                continue;
            if (entity_idx < 0 || entity_idx >= env->num_road_elements)
                continue;

            RoadMapElement *element = &env->road_elements[entity_idx];
            int is_lane = is_road_lane(element->type);
            int is_edge = is_road_edge(element->type);
            if (!is_lane && !is_edge)
                continue;
            if (geometry_idx < 0 || geometry_idx >= element->segment_length - 1)
                continue;

            float start_x = element->x[geometry_idx];
            float start_y = element->y[geometry_idx];
            float start_z = element->z[geometry_idx];
            float end_x = element->x[geometry_idx + 1];
            float end_y = element->y[geometry_idx + 1];
            float end_z = element->z[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float mid_z = (start_z + end_z) / 2.0f;
            float rel_x = mid_x - ego_entity->sim_x;
            float rel_y = mid_y - ego_entity->sim_y;
            float rel_z = mid_z - ego_entity->sim_z;
            float x_obs = rel_x * ego_entity->cos_heading + rel_y * ego_entity->sin_heading;
            float y_obs = -rel_x * ego_entity->sin_heading + rel_y * ego_entity->cos_heading;

            // Filter by asymmetric vision rectangle
            if (x_obs < -env->road_obs_behind_dist || x_obs > env->road_obs_front_dist)
                continue;
            if (fabsf(y_obs) > env->road_obs_side_dist)
                continue;
            if (fabsf(rel_z) > Z_BUFFER)
                continue;

            // Compute segment direction and length
            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float length = sqrtf(dx * dx + dy * dy);
            float dx_norm = (length > 0) ? dx / length : dx;
            float dy_norm = (length > 0) ? dy / length : dy;
            float cos_angle = dx_norm * ego_entity->cos_heading + dy_norm * ego_entity->sin_heading;
            float sin_angle = -dx_norm * ego_entity->sin_heading + dy_norm * ego_entity->cos_heading;
            if (is_edge && length > 0) {
                float angle = atan2f(sin_angle, cos_angle);
                if (angle > (float)M_PI / 2.0f) {
                    angle -= (float)M_PI;
                } else if (angle < -(float)M_PI / 2.0f) {
                    angle += (float)M_PI;
                }
                cos_angle = cosf(angle);
                sin_angle = sinf(angle);
            }

            float *target;
            int *counter;
            int cap;
            if (is_lane) {
                target = lanes_dest;
                counter = &lanes_collected;
                cap = env->max_lane_segment_observations;
            } else {
                target = boundaries_dest;
                counter = &boundaries_collected;
                cap = env->max_boundary_segment_observations;
            }
            if (*counter >= cap)
                continue;
            int base = (*counter)++ * ROAD_FEATURES;

            // Road segment x
            target[base] = x_obs / env->max_position;
            // Road segment y
            target[base + 1] = y_obs / env->max_position;
            // Road segment z
            target[base + 2] = rel_z / Z_BUFFER;
            // Road segment length
            target[base + 3] = length / env->max_road_segment_length;
            // Road segment width
            target[base + 4] = LANE_WIDTH / env->max_road_segment_width;
            // Road segment orientation (cosine)
            target[base + 5] = cos_angle;
            // Road segment orientation (sine)
            target[base + 6] = sin_angle;
        }

        if (env->road_dropout_enabled) {
            int lane_to_write =
                (lanes_collected < env->obs_lane_segment_count) ? lanes_collected : env->obs_lane_segment_count;
            int boundary_to_write = (boundaries_collected < env->obs_boundary_segment_count)
                                        ? boundaries_collected
                                        : env->obs_boundary_segment_count;
            subsample_road_observation_rows(lanes_buffer, lanes_collected, lane_to_write);
            subsample_road_observation_rows(boundaries_scratch, boundaries_collected, boundary_to_write);
            memcpy(&obs[lane_obs_idx], lanes_buffer, lane_to_write * ROAD_FEATURES * sizeof(float));
            memset(&obs[lane_obs_idx + lane_to_write * ROAD_FEATURES], 0,
                   (env->obs_lane_segment_count - lane_to_write) * ROAD_FEATURES * sizeof(float));
            memcpy(&obs[boundary_obs_idx], boundaries_scratch, boundary_to_write * ROAD_FEATURES * sizeof(float));
            memset(&obs[boundary_obs_idx + boundary_to_write * ROAD_FEATURES], 0,
                   (env->obs_boundary_segment_count - boundary_to_write) * ROAD_FEATURES * sizeof(float));
            lanes_collected = lane_to_write;
            boundaries_collected = boundary_to_write;
        } else {
            memset(&obs[lane_obs_idx + lanes_collected * ROAD_FEATURES], 0,
                   (env->obs_lane_segment_count - lanes_collected) * ROAD_FEATURES * sizeof(float));
            memset(&obs[boundary_obs_idx + boundaries_collected * ROAD_FEATURES], 0,
                   (env->obs_boundary_segment_count - boundaries_collected) * ROAD_FEATURES * sizeof(float));
        }

        // ===== Traffic control observations =====
        typedef struct {
            int idx;
            float dist_sq;
        } TrafficControlDist;
        TrafficControlDist traffic_controls[env->num_traffic_elements > 0 ? env->num_traffic_elements : 1];
        int num_visible_controls = 0;

        // Collect traffic controls within range
        for (int j = 0; j < env->num_traffic_elements; j++) {
            TrafficControlElement *traffic = &env->traffic_elements[j];
            if (!traffic_control_in_scope(traffic->type, env->traffic_control_scope))
                continue;

            float mid_x = (traffic->stop_line[0] + traffic->stop_line[3]) * 0.5f;
            float mid_y = (traffic->stop_line[1] + traffic->stop_line[4]) * 0.5f;
            float mid_z = (traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f;
            float dx = mid_x - ego_entity->sim_x;
            float dy = mid_y - ego_entity->sim_y;
            float dz = mid_z - ego_entity->sim_z;
            float abs_dz = fabsf(dz);
            float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq > env->max_traffic_control_distance * env->max_traffic_control_distance || abs_dz > Z_BUFFER)
                continue;

            traffic_controls[num_visible_controls].idx = j;
            traffic_controls[num_visible_controls].dist_sq = dist_sq;
            num_visible_controls++;
        }

        // Partial selection sort: find K closest (O(N*K))
        int num_controls_to_observe = (num_visible_controls < env->max_traffic_control_observations)
                                          ? num_visible_controls
                                          : env->max_traffic_control_observations;
        for (int k = 0; k < num_controls_to_observe; k++) {
            int min_idx = k;
            for (int j = k + 1; j < num_visible_controls; j++) {
                if (traffic_controls[j].dist_sq < traffic_controls[min_idx].dist_sq) {
                    min_idx = j;
                }
            }
            if (min_idx != k) {
                TrafficControlDist temp = traffic_controls[k];
                traffic_controls[k] = traffic_controls[min_idx];
                traffic_controls[min_idx] = temp;
            }
        }

        // Add observations for closest traffic controls
        int controls_added = 0;
        for (int j = 0; j < num_controls_to_observe && controls_added < env->max_traffic_control_observations; j++) {
            TrafficControlElement *traffic = &env->traffic_elements[traffic_controls[j].idx];
            // Stop line start point
            float dx1 = traffic->stop_line[0] - ego_entity->sim_x;
            float dy1 = traffic->stop_line[1] - ego_entity->sim_y;
            float rel_x1 = dx1 * ego_entity->cos_heading + dy1 * ego_entity->sin_heading;
            float rel_y1 = -dx1 * ego_entity->sin_heading + dy1 * ego_entity->cos_heading;
            // Stop line end point
            float dx2 = traffic->stop_line[3] - ego_entity->sim_x;
            float dy2 = traffic->stop_line[4] - ego_entity->sim_y;
            float rel_x2 = dx2 * ego_entity->cos_heading + dy2 * ego_entity->sin_heading;
            float rel_y2 = -dx2 * ego_entity->sin_heading + dy2 * ego_entity->cos_heading;
            float rel_z = (traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f - ego_entity->sim_z;

            int state;
            if (traffic->type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
                if (env->timestep >= 0 && env->timestep < traffic->state_length && traffic->states != NULL) {
                    state = traffic->states[env->timestep];
                } else {
                    state = TRAFFIC_CONTROL_STATE_OFF;
                }
            } else {
                state = TRAFFIC_CONTROL_STATE_UNKNOWN;
            }

            // Traffic control stop line start point x
            obs[obs_idx++] = rel_x1 / env->max_position;
            // Traffic control stop line start point y
            obs[obs_idx++] = rel_y1 / env->max_position;
            // Traffic control stop line end point x
            obs[obs_idx++] = rel_x2 / env->max_position;
            // Traffic control stop line end point y
            obs[obs_idx++] = rel_y2 / env->max_position;
            // Traffic control stop line midpoint z
            obs[obs_idx++] = rel_z / Z_BUFFER;
            // Traffic control type
            obs[obs_idx++] = traffic->type;
            // Traffic control state
            obs[obs_idx++] = state;
            controls_added++;
        }

        // Zero out remaining traffic control slots
        int remaining_traffic_obs = (env->max_traffic_control_observations - controls_added) * TRAFFIC_CONTROL_FEATURES;
        memset(&obs[obs_idx], 0, remaining_traffic_obs * sizeof(float));
        obs_idx += remaining_traffic_obs;

        obs[obs_idx++] = (float)lanes_collected;
        obs[obs_idx++] = (float)boundaries_collected;
        obs[obs_idx++] = (float)cars_seen;
        obs[obs_idx++] = (float)controls_added;
    }
}

static void move_dynamics(Drive *env, int action_idx, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    // If agent is removed, set position to invalid and return
    if (agent->removed) {
        invalidate_agent(agent);
        return;
    }

    // If agent is stopped, zero out velocities and accelerations
    if (agent->stopped) {
        agent->sim_vx = 0.0f;
        agent->sim_vy = 0.0f;
        agent->yaw_rate = 0.0f;
        agent->sim_speed = 0.0f;
        agent->sim_speed_signed = 0.0f;
        agent->a_long = 0.0f;
        agent->a_lat = 0.0f;
        agent->jerk_long = 0.0f;
        agent->jerk_lat = 0.0f;
        agent->steering_angle = 0.0f;
        return;
    }

    // Phantom braking: override action with max braking
    int phantom_braking_active = 0;
    if (agent->phantom_braking_counter > 0) {
        agent->phantom_braking_counter--;
        phantom_braking_active = 1;
    } else if (agent->is_phantom_braker && env->phantom_braking_trigger_prob > 0.0f &&
               random_uniform(0.0f, 1.0f) < env->phantom_braking_trigger_prob) {
        agent->phantom_braking_counter = env->phantom_braking_duration - 1;
        phantom_braking_active = 1;
    }

    if (env->dynamics_model == CLASSIC) {
        // Classic dynamics model
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 0) { // discrete
            // Interpret action as a single integer: a = accel_idx * num_steer + steer_idx
            int *action_array = (int *)env->actions;
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            int action_val = action_array[action_idx];
            int acceleration_index = action_val / num_steer;
            int steering_index = action_val % num_steer;
            acceleration = ACCELERATION_VALUES[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        } else if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            acceleration = action_array_f[action_idx][0];
            steering = action_array_f[action_idx][1];

            acceleration *= ACCELERATION_VALUES[6];
            steering *= STEERING_VALUES[8];
        }

        if (phantom_braking_active) {
            acceleration = ACCELERATION_VALUES[0]; // max braking
            steering = 0.0f;
        }

        // Limit the steering rate similar to the jerk model
        float delta_steer = clip(steering - agent->steering_angle, -0.6f * env->dt, 0.6f * env->dt);
        steering = clip(agent->steering_angle + delta_steer, -STEERING_LIMIT, STEERING_LIMIT);
        agent->steering_angle = steering;

        // Current state
        float x = agent->sim_x;
        float y = agent->sim_y;
        float heading = agent->sim_heading;
        float speed = agent->sim_speed_signed;

        // Update speed with acceleration
        speed += acceleration * env->dt;
        if (phantom_braking_active)
            clamp_nonneg_speed(&speed, &acceleration);
        speed = clip(speed, -MAX_SPEED, MAX_SPEED);
        // Compute yaw rate
        float beta = atanf(0.5f * tanf(steering));
        // New heading
        float yaw_rate = (speed * cosf(beta) * tanf(steering)) / agent->wheelbase;

        // New velocity
        float new_vx = speed * cosf(heading + beta);
        float new_vy = speed * sinf(heading + beta);

        // Update position
        x = x + (new_vx * env->dt);
        y = y + (new_vy * env->dt);
        heading = heading + yaw_rate * env->dt;

        // Apply updates to the agent's state
        agent->sim_x = x;
        agent->sim_y = y;
        agent->sim_heading = normalize_heading(heading);
        agent->cos_heading = cosf(agent->sim_heading);
        agent->sin_heading = sinf(agent->sim_heading);
        agent->sim_vx = new_vx;
        agent->sim_vy = new_vy;
        agent->yaw_rate = yaw_rate;
        update_agent_speed(agent);

        // Compute acceleration and jerk from finite differences (for comfort metric)
        float new_a_long = acceleration;    // commanded longitudinal acceleration
        float new_a_lat = speed * yaw_rate; // centripetal: v * omega
        agent->jerk_long = (new_a_long - agent->a_long) / env->dt;
        agent->jerk_lat = (new_a_lat - agent->a_lat) / env->dt;
        agent->a_long = new_a_long;
        agent->a_lat = new_a_lat;
    } else {
        // JERK dynamics model
        // Extract jerk action components
        float j_long, j_lat;
        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;

            // Asymmetric scaling for longitudinal jerk to match discrete action space
            // Discrete: JERK_LONG = [-15, -4, 0, 4] (more braking than acceleration)
            float j_long_action = action_array_f[action_idx][0]; // [-1, 1]
            if (j_long_action < 0) {
                j_long = j_long_action * (-JERK_LONG[0]); // Negative: [-1, 0] → [-15, 0] (braking)
            } else {
                j_long = j_long_action * JERK_LONG[3]; // Positive: [0, 1] → [0, 4] (acceleration)
            }

            // Symmetric scaling for lateral jerk
            j_lat = action_array_f[action_idx][1] * JERK_LAT[2];
        } else if (env->action_type == 0) { // discrete
            // Interpret action as a single integer: a = long_idx * num_lat + lat_idx
            int *action_array = (int *)env->actions;
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            int action_val = action_array[action_idx];
            int j_long_idx = action_val / num_lat;
            int j_lat_idx = action_val % num_lat;
            j_long = JERK_LONG[j_long_idx];
            j_lat = JERK_LAT[j_lat_idx];
        }

        if (phantom_braking_active) {
            j_long = JERK_LONG[0]; // max braking jerk
            j_lat = 0.0f;
        }

        // Get dynamic conditioning coefficients
        float c_throttle = agent->reward_coefs[REWARD_COEF_THROTTLE];
        float c_steer = agent->reward_coefs[REWARD_COEF_STEER];
        float c_acc = agent->reward_coefs[REWARD_COEF_ACC];

        // Calculate new longitudinal acceleration from jerk (Eq. 1 in paper)
        float a_long_new = agent->a_long + c_throttle * j_long * env->dt;

        // Zero-crossing: snap to 0 when crossing zero
        if (agent->a_long * a_long_new < 0) {
            a_long_new = 0.0f;
        } else {
            a_long_new = clip(a_long_new, ACCEL_LONG_LIMIT[0], ACCEL_LONG_LIMIT[1] * c_acc);
        }

        // Calculate new lateral acceleration from jerk (Eq. 2 in paper)
        float a_lat_new = agent->a_lat + c_steer * j_lat * env->dt;

        // Zero-crossing: snap to 0 when crossing zero
        if (agent->a_lat * a_lat_new < 0) {
            a_lat_new = 0.0f;
        } else {
            a_lat_new = clip(a_lat_new, ACCEL_LAT_LIMIT[0], ACCEL_LAT_LIMIT[1]);
        }

        float heading_x = agent->cos_heading;
        float heading_y = agent->sin_heading;

        // Calculate new velocity using trapezoidal integration
        float v_dot_heading = agent->sim_vx * heading_x + agent->sim_vy * heading_y;
        float signed_v = copysignf(sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy), v_dot_heading);
        float v_new = signed_v + 0.5f * (a_long_new + agent->a_long) * env->dt;

        // Zero-crossing: snap to 0 when crossing zero
        if (signed_v * v_new < 0) {
            v_new = 0.0f;
        } else {
            v_new = clip(v_new, -2.0f, MAX_SPEED);
        }

        // If phantom braking is active, prevent speed from going negative
        if (phantom_braking_active)
            clamp_nonneg_speed(&v_new, &a_long_new);

        // GIGAFLOW paper approach: a_lat → curvature → steering
        // v_eff = max(|v|, 1.0) to avoid division issues at low speed
        float v_eff = fmaxf(fabsf(v_new), 1.0f);
        float signed_curvature = a_lat_new / (v_eff * v_eff);

        // Convert curvature to steering angle
        float steering_angle = atanf(signed_curvature * agent->wheelbase);

        // Apply steering rate limit (±0.6 rad/s)
        float delta_steer = clip(steering_angle - agent->steering_angle, -0.6f * env->dt, 0.6f * env->dt);

        // Apply steering position limit (±0.55 rad)
        float new_steering_angle = clip(agent->steering_angle + delta_steer, -0.55f, 0.55f);

        // Recalculate curvature from limited steering
        signed_curvature = tanf(new_steering_angle) / agent->wheelbase;

        // Recalculate lateral acceleration from actual curvature
        a_lat_new = v_new * v_new * signed_curvature;

        // Calculate resulting movement using bicycle dynamics
        float d = 0.5f * (v_new + signed_v) * env->dt;
        float theta = d * signed_curvature;
        float dx_local, dy_local;

        if (fabsf(signed_curvature) < 1e-5f || fabsf(theta) < 1e-5f) {
            dx_local = d;
            dy_local = 0.0f;
        } else {
            dx_local = sinf(theta) / signed_curvature;
            dy_local = (1.0f - cosf(theta)) / signed_curvature;
        }

        float dx = dx_local * heading_x - dy_local * heading_y;
        float dy = dx_local * heading_y + dy_local * heading_x;

        // Update agent state
        agent->sim_x += dx;
        agent->sim_y += dy;
        agent->sim_heading = normalize_heading(agent->sim_heading + theta);
        agent->cos_heading = cosf(agent->sim_heading);
        agent->sin_heading = sinf(agent->sim_heading);
        agent->sim_vx = v_new * cosf(agent->sim_heading);
        agent->sim_vy = v_new * sinf(agent->sim_heading);
        const float yaw_rate = v_new * signed_curvature;
        agent->yaw_rate = yaw_rate;

        update_agent_speed(agent);
        // Update jerk and acceleration
        agent->jerk_long = (a_long_new - agent->a_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
        agent->a_long = a_long_new;
        agent->a_lat = a_lat_new;
        agent->steering_angle = new_steering_angle;
    }

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * Z_COMPUTATION_OFFSET_COUNT];
    int list_size = get_neighbors_entities(env, agent->sim_x, agent->sim_y, entity_list,
                                           MAX_ENTITIES_PER_CELL * Z_COMPUTATION_OFFSET_COUNT, z_computation_offsets,
                                           Z_COMPUTATION_OFFSET_COUNT);
    if (list_size > 0) {
        DepthPoint road_neighbors[list_size];
        DepthPoint current_lane_neighbors[list_size];
        int valid_count = 0;
        int current_lane_count = 0;
        for (int i = 0; i < list_size; i++) {
            if (entity_list[i].entity_idx == -1)
                continue;
            if (entity_list[i].entity_type != ENTITY_TYPE_ROAD_ELEMENT)
                continue;

            const RoadMapElement *entity = &env->road_elements[entity_list[i].entity_idx];
            DepthPoint point = compute_z_distance_to_road_segment(agent, entity, entity_list[i].geometry_idx);
            if (point.z_dis < Z_BUFFER) {
                road_neighbors[valid_count++] = point;
                if (entity_list[i].entity_idx == agent->current_lane_idx) {
                    current_lane_neighbors[current_lane_count++] = point;
                }
            }
        }

        int neighbor_count = (current_lane_count > 0) ? current_lane_count : valid_count;
        if (neighbor_count > 0) {
            DepthPoint *neighbors = (current_lane_count > 0) ? current_lane_neighbors : road_neighbors;
            qsort(neighbors, neighbor_count, sizeof(DepthPoint), compare_depthpoint);
            int check_count = (neighbor_count < Z_NUM_PT_AVG) ? neighbor_count : Z_NUM_PT_AVG;
            float sum_z = 0.0f;
            for (int i = 0; i < check_count; i++) {
                sum_z += neighbors[i].z;
            }
            agent->sim_z = sum_z / check_count;
        }
    }

    return;
}

#include "idm.h"
#include "pdm.h"

static void move_agent_with_controller(Drive *env, int action_idx, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    int controller = agent->controller;

    if (controller == CONTROLLER_STATIC) {
        return;
    }

    if (controller == CONTROLLER_CORRIDOR_IDM) {
        move_corridor_idm(env, agent_idx);
        return;
    }

    if (controller == CONTROLLER_IDM) {
        move_idm(env, agent_idx);
        return;
    }

    if (controller == CONTROLLER_NUPLAN_IDM) {
        move_nuplan_idm(env, agent_idx);
        return;
    }

    if (controller == CONTROLLER_PDM) {
        move_pdm(env, agent_idx);
        return;
    }

    if (controller == CONTROLLER_REPLAY) {
        if (env->simulation_mode == SIMULATION_REPLAY) {
            move_expert(env, env->actions, agent_idx);
        }
        return;
    }

    if (action_idx >= 0) {
        move_dynamics(env, action_idx, agent_idx);
    }
}

static inline void sample_erratic_flags(Drive *env, Agent *agent) {
    agent->is_blind_partner =
        (env->partner_blindness_prob > 0.0f && random_uniform(0.0f, 1.0f) < env->partner_blindness_prob) ? 1 : 0;
    agent->is_phantom_braker =
        (env->phantom_braking_prob > 0.0f && random_uniform(0.0f, 1.0f) < env->phantom_braking_prob) ? 1 : 0;
    agent->phantom_braking_counter = 0;
}

static inline float compute_adversarial_target_bonus(Drive *env, RewardTerms *target_terms) {
    float bonus = 0.0f;
    if (target_terms->offroad < 0.0f) {
        bonus += env->adv_target_offroad_reward;
    }

    if (!env->target_hit_this_step) {
        return bonus;
    }

    float responsibility = fmaxf(0.0f, fminf(1.0f, env->target_hit_responsibility_this_step));
    int low_responsibility = env->adv_target_hit_low_responsibility_threshold >= 0.0f &&
                             responsibility < env->adv_target_hit_low_responsibility_threshold;
    if (low_responsibility) {
        return bonus - env->adv_target_hit_low_responsibility_penalty;
    }

    bonus += responsibility * env->adv_target_collision_reward;
    if (env->target_hit_at_fault_this_step) {
        bonus += env->adv_target_hit_at_fault_bonus;
    }
    return bonus;
}

void c_reset(Drive *env) {
    env->target_hit_this_step = 0;
    env->target_hit_hitter_idx_this_step = -1;
    env->target_hit_at_fault_this_step = 0;
    env->target_hit_responsibility_this_step = 0.0f;
    env->target_hit_count_episode = 0.0f;
    env->target_hit_responsibility_episode = 0.0f;
    env->target_hit_low_responsibility_count_episode = 0.0f;
    env->target_hit_at_fault_count_episode = 0.0f;
    env->target_collision_type_episode = COLLISION_TYPE_NONE;
    env->target_collision_target_speed_episode = 0.0f;
    env->target_collision_other_speed_episode = 0.0f;
    env->target_collision_relative_speed_episode = 0.0f;
    env->target_collision_other_active_episode = 0.0f;
    env->target_collision_other_stopped_episode = 0.0f;
    env->target_collision_other_removed_episode = 0.0f;
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].brake_hist_count = 0;
    }
    env->target_avoidability_by_braking_episode = TARGET_AVOIDABILITY_UNSET;

    // Replay envs should expose the constructor-time initial state on the first reset.
    // Gigaflow envs need to fully respawn so explicit reset seeds affect the first scenario.
    if (env->timestep == 0 && env->simulation_mode != SIMULATION_GIGAFLOW) {
        for (int x = 0; x < env->active_agent_count; x++) {
            env->logs[x] = (Log){0};
            int agent_idx = env->active_agent_indices[x];
            sample_erratic_flags(env, &env->agents[agent_idx]);
            initialize_agent_progression(env, agent_idx);
            compute_metrics(env, agent_idx);
        }
        compute_observations(env);
        return;
    }

    env->timestep = env->init_steps;

    if (env->simulation_mode == SIMULATION_GIGAFLOW) {
        generate_traffic_light_states(env);
        int num_reset = 0;
        for (int x = 0; x < env->active_agent_count; x++) {
            int agent_idx = env->active_agent_indices[x];

            // Respawn agent at new random position
            if (spawn_agent(env, agent_idx, num_reset)) {
                num_reset++;
            } else {
                // Failed spawn: ensure agent is properly invalidated
                invalidate_agent(&env->agents[agent_idx]);
                env->agents[agent_idx].removed = 1;
            }
        }

        if (num_reset != env->active_agent_count) {
            printf("[GIGAFLOW ERROR] -> Only respawned %d out of %d agents during reset\n", num_reset,
                   env->active_agent_count);
        }

        // GIGAFLOW: spawn_agent already set positions, routes, paths, goals.
        // Only need to generate reward coefs and compute initial metrics.
        for (int x = 0; x < env->active_agent_count; x++) {
            env->logs[x] = (Log){0};
            int agent_idx = env->active_agent_indices[x];
            Agent *agent = &env->agents[agent_idx];
            if (agent->removed)
                continue;
            reset_agent_metrics(env, agent_idx);
            reset_agent_state(agent);
            sample_erratic_flags(env, agent);
            generate_reward_coefs(env, agent);
            initialize_agent_progression(env, agent_idx);
            compute_metrics(env, agent_idx);
        }
        compute_observations(env);
        return;
    }

    set_start_position(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        Agent *agent = &env->agents[agent_idx];

        // Common resets
        reset_agent_metrics(env, agent_idx);
        reset_agent_state(agent);
        sample_erratic_flags(env, agent);
        generate_reward_coefs(env, agent);

        compute_goals(env, agent_idx);
        initialize_agent_progression(env, agent_idx);
        compute_metrics(env, agent_idx);
    }
    compute_observations(env);
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    RewardTerms reward_terms[(env->active_agent_count > 0) ? env->active_agent_count : 1];
    memset(reward_terms, 0, env->active_agent_count * sizeof(RewardTerms));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));
    env->target_hit_this_step = 0;
    env->target_hit_hitter_idx_this_step = -1;
    env->target_hit_at_fault_this_step = 0;
    env->target_hit_responsibility_this_step = 0.0f;

    // Update masks: exclude stopped/removed agents AND erratic drivers (blind/phantom-braker)
    // from the training rollout buffer, per GIGAFLOW paper Appendix B.4.
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *a = &env->agents[agent_idx];
        if (a->stopped || a->removed || a->is_blind_partner || a->is_phantom_braker) {
            env->masks[i] = 0;
        } else {
            env->masks[i] = 1;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        snapshot_previous_pose(&env->agents[i]);
        push_brake_history(&env->agents[i]);
    }

    env->timestep++;

    // -> 1. Apply actions and move agents
    // Move background agents according to their per-agent controller.
    for (int i = 0; i < env->static_agent_count; i++) {
        int background_idx = env->static_agent_indices[i];
        move_agent_with_controller(env, -1, background_idx);
    }
    // Move active agents with policy actions
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        if (!(agent->stopped || agent->removed)) {
            env->logs[i].episode_length += 1;
        }
        move_agent_with_controller(env, i, agent_idx);
    }

    // -> 2. Compute metrics and rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];

        if (env->agents[agent_idx].stopped || env->agents[agent_idx].removed)
            continue;

        // Compute metrics
        compute_metrics(env, agent_idx);

        // Compute rewards
        reward_terms[i] = compute_rewards(env, i);
    }

    if (env->target_hit_this_step) {
        env->target_hit_count_episode += 1.0f;
        env->target_hit_responsibility_episode += env->target_hit_responsibility_this_step;
        if (env->target_hit_at_fault_this_step) {
            env->target_hit_at_fault_count_episode += 1.0f;
        }
        if (env->adv_target_hit_low_responsibility_threshold >= 0.0f &&
            env->target_hit_responsibility_this_step < env->adv_target_hit_low_responsibility_threshold) {
            env->target_hit_low_responsibility_count_episode += 1.0f;
        }
    }

    // Adversarial reward: agent 0 is the target in both replay and gigaflow modes.
    if (env->active_agent_count > 0) {
        float adversarial_bonus = compute_adversarial_target_bonus(env, &reward_terms[0]);

        for (int i = 1; i < env->active_agent_count; i++) {
            reward_terms[i].collision *= env->adv_reward_weight_collision;
            reward_terms[i].offroad *= env->adv_reward_weight_offroad;
            reward_terms[i].red_light *= env->adv_reward_weight_traffic_light;
            reward_terms[i].drive *= env->adv_reward_weight_drive;

            // Assign adversarial reward.
            reward_terms[i].adversarial = adversarial_bonus;
            // reward_terms[i].adversarial = 0.0f;
            // reward_terms[i].collision = 0.0f;
            // reward_terms[i].offroad = 0.0f;
        }
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        apply_reward_terms(env, i, &reward_terms[i]);
    }

    // Mark terminals for stopped or removed agents
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->agents[agent_idx].stopped || env->agents[agent_idx].removed) {
            env->terminals[i] = 1;
        }
    }

    int bad_target_hit = env->target_hit_this_step && env->adv_target_hit_low_responsibility_threshold >= 0.0f &&
                         env->target_hit_responsibility_this_step < env->adv_target_hit_low_responsibility_threshold;
    if (bad_target_hit && (env->adv_target_hit_low_responsibility_behavior == 1 ||
                           env->adv_target_hit_low_responsibility_behavior == 2)) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            if (env->adv_target_hit_low_responsibility_behavior == 1) {
                env->truncations[i] = 0;
                env->terminals[i] = 1;
            } else if (env->adv_target_hit_low_responsibility_behavior == 2) {
                if (agent_idx == env->target_hit_hitter_idx_this_step) {
                    env->truncations[i] = 0;
                    env->terminals[i] = 1;
                }
            }
        }
        if (env->adv_target_hit_low_responsibility_behavior == 1) {
            add_log(env);
            c_reset(env);
            return;
        }
    }

    // -> 3. Check for episode truncation
    int early_reset = 0;
    if (env->simulation_mode == SIMULATION_GIGAFLOW && env->termination_mode == 1) {
        int count_inactive = 0;
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            if (env->agents[agent_idx].removed || env->agents[agent_idx].stopped) {
                count_inactive++;
            }
        }
        float ratio_inactive = (float)count_inactive / (float)env->active_agent_count;
        if (ratio_inactive > env->inactive_agent_threshold) {
            early_reset = 1;
        }
    }

    int adversarial_early_reset = 0;
    int target_failure_early_reset = 0;
    if (env->active_agent_count > 0 && env->adversarial_termination_mode > 0) {
        int target_agent_idx = env->active_agent_indices[0];
        int target_inactive = env->agents[target_agent_idx].removed || env->agents[target_agent_idx].stopped;

        int alive_adversaries = 0;
        for (int i = 1; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            if (!(env->agents[agent_idx].removed || env->agents[agent_idx].stopped)) {
                alive_adversaries++;
            }
        }

        int no_adversaries_alive = (alive_adversaries == 0);
        switch (env->adversarial_termination_mode) {
        case 1:
            adversarial_early_reset = no_adversaries_alive;
            break;
        case 2:
            adversarial_early_reset = target_inactive;
            target_failure_early_reset = target_inactive;
            break;
        case 3:
            adversarial_early_reset = no_adversaries_alive || target_inactive;
            target_failure_early_reset = target_inactive;
            break;
        default:
            adversarial_early_reset = 0;
            break;
        }
    }

    if (env->timestep == env->scenario_length || early_reset || adversarial_early_reset) {
        for (int i = 0; i < env->active_agent_count; i++) {
            if (env->terminate_ep_on_target_failure && target_failure_early_reset) {
                env->terminals[i] = 1;
            } else {
                env->truncations[i] = 1;
            }
        }
        add_log(env);
        c_reset(env);
        return;
    }

    // -> 4. Compute observations
    compute_observations(env);

    // -> 5. Update goals for agents that reached their goal
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        if (agent->metrics_array[REACHED_GOAL_IDX] > 0.0f) {
            if (agent->current_goal_idx == env->num_target_waypoints) {
                // Last goal reached - generate new set of goals
                env->logs[i].num_goals_reached += 1;
                compute_goals(env, agent_idx);
            } else {
                // Advance alias to next goal
                agent->goal_position_x = agent->goal_positions_x[agent->current_goal_idx];
                agent->goal_position_y = agent->goal_positions_y[agent->current_goal_idx];
                agent->goal_position_z = agent->goal_positions_z[agent->current_goal_idx];
            }
        }
    }
}

#include "render.h"
