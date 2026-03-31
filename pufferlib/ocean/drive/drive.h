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

// Collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2
#define TRAFFIC_LIGHT_VIOLATION 3

// Collision/Infraction behaviors
#define STOP_AGENT 1
#define REMOVE_AGENT 2

// Grid cell size
#define GRID_CELL_SIZE 5.0f
// Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution)
// => For each entity type in gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2
#define MAX_ENTITIES_PER_CELL 30
// GridMapEntity types
#define ENTITY_TYPE_ROAD_ELEMENT 1
#define ENTITY_TYPE_TRAFFIC_CONTROL 2

// Max road segment observation
#define ROAD_OBS_FRONT_DIST 100.0f
#define ROAD_OBS_BEHIND_DIST 20.0f
#define ROAD_OBS_SIDE_DIST 40.0f
#define VISION_HALF_RANGE                                                                                              \
    ((int)ceilf(fmaxf(fmaxf(ROAD_OBS_FRONT_DIST, ROAD_OBS_BEHIND_DIST), ROAD_OBS_SIDE_DIST) / GRID_CELL_SIZE))

// Traffic light observations
#define MAX_TRAFFIC_LIGHT_DISTANCE 100.0f

// Agent observations
#define AGENT_OBS_MAX_DIST 100.0f // Max distance for observing other agents

// TARGET_TYPE modes (controls what target info is in observations)
#define TARGET_STATIC 0
#define TARGET_DYNAMIC 1

// Observation normalization constants
#define MAX_GOAL_POSITION 100.0f       // Goal position scaling (m)
#define MAX_POSITION 100.0f            // Max position for normalization (m)
#define MAX_SPEED 40.0f                // Max speed for normalization (m/s)
#define MAX_VEH_LEN 15.0f              // Max vehicle length for normalization (m)
#define MAX_VEH_WIDTH 10.0f            // Max vehicle width for normalization (m)
#define MAX_ROAD_SCALE 100.0f          // Max road feature scaling (m)
#define MAX_ROAD_SEGMENT_LENGTH 100.0f // Max road segment length (m)

// Observation feature counts
#define EGO_FEATURES_CLASSIC 7
#define EGO_FEATURES_JERK 9
#define ROAD_FEATURES 7
#define PARTNER_FEATURES 8
#define TRAFFIC_LIGHT_FEATURES 6
#define STOP_SIGN_FEATURES 3
#define STATIC_TARGET_FEATURES 3
#define DYNAMIC_TARGET_FEATURES 5

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

struct Log {
    float n;
    float episode_return;
    float episode_length;
    float expert_static_car_count;
    float static_car_count;
    float score;
    float offroad_rate;
    float collision_rate;
    float red_light_violation_rate;
    float num_waypoints_reached;
    float num_goals_reached;
    float comfort_violation_count;
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
    float reward_traffic_light_violation;
    float reward_timestep;
    float reward_overspeed;
    float reward_ade;
    char *map_name;
    float world_mean_x;
    float world_mean_y;
    float dt;
    float spawn_initial_speed;
    float goal_radius;
    float goal_speed_threshold;
    float min_waypoint_spacing;
    float max_waypoint_spacing;
    int num_target_waypoints;
    int logs_capacity;
    int target_type;
    char *ini_file;
    int collision_behavior;     // 0 = none, 1=stop, 2 = remove
    int offroad_behavior;       // 0 = none, 1=stop, 2 = remove
    int traffic_light_behavior; // 0 = none, 1=stop, 2 = remove
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
    int simulation_mode;
    int termination_mode;
    float inactive_agent_threshold;
    int reward_conditioning;
    int reward_randomization;
    int compute_eval_metrics;
    int use_rear_axle;
    int max_boundary_segment_observations;
    int max_lane_segment_observations;
    int max_partner_observations;
    int max_traffic_light_observations;
    int max_stop_sign_observations;
    int eval_mode;
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

static void invalidate_agent(Agent *agent) {
    agent->sim_x = INVALID_POSITION;
    agent->sim_y = INVALID_POSITION;
    agent->sim_z = 0.0f;
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
    {0.0f, 3.0f},       // REWARD_COEF_COLLISION       α_collision ~ U(0, 3)
    {0.0f, 3.0f},       // REWARD_COEF_OFFROAD         α_boundary ~ U(0, 3)
    {0.0f, 0.1f},       // REWARD_COEF_COMFORT         α_comfort ~ U(0, 0.1)
    {2.5e-4f, 2.5e-2f}, // REWARD_COEF_LANE_ALIGN      α_l-align ~ U(2.5e-4, 2.5e-2)
    {0.0f, 1.0f},       // REWARD_COEF_VEL_ALIGN       α_vel-align ~ U(0, 1)
    {2.5e-4f, 7.5e-3f}, // REWARD_COEF_LANE_CENTER     α_l-center ~ U(2.5e-4, 7.5e-3)
    {-0.5f, 0.5f},      // REWARD_COEF_CENTER_BIAS     α_center-bias ~ U(-0.5, 0.5)
    {0.0f, 5e-3f},      // REWARD_COEF_VELOCITY        α_velocity = 2.5e-3 (fixed)
    {2.5e-4f, 7.5e-3f}, // REWARD_COEF_REVERSE         α_reverse ~ U(2.5e-4, 7.5e-3)
    {0.0f, 1.0f},       // REWARD_COEF_TRAFFIC_LIGHT   α_stop-line ~ U(0, 1)
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
        agent->reward_coefs[REWARD_COEF_TRAFFIC_LIGHT] = random_uniform(
            REWARD_BOUNDS[REWARD_COEF_TRAFFIC_LIGHT].min_val, REWARD_BOUNDS[REWARD_COEF_TRAFFIC_LIGHT].max_val);
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
        agent->reward_coefs[REWARD_COEF_COLLISION] = env->reward_vehicle_collision;
        agent->reward_coefs[REWARD_COEF_OFFROAD] = env->reward_offroad_collision;
        agent->reward_coefs[REWARD_COEF_COMFORT] = env->reward_comfort;
        agent->reward_coefs[REWARD_COEF_LANE_ALIGN] = env->reward_lane_align;
        agent->reward_coefs[REWARD_COEF_LANE_CENTER] = env->reward_lane_center;
        agent->reward_coefs[REWARD_COEF_VELOCITY] = env->reward_velocity;
        agent->reward_coefs[REWARD_COEF_TRAFFIC_LIGHT] = env->reward_traffic_light_violation;
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

    // 20% chance: disable ALL lights for this episode
    int disable_all = (!env->eval_mode) && (random_uniform(0.0f, 1.0f) < TL_EPISODE_DISABLE_PROB);

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (traffic->type != TRAFFIC_LIGHT || traffic->states == NULL || traffic->state_length <= 0)
            continue;

        int fill_steps = steps;
        if (traffic->state_length < fill_steps)
            fill_steps = traffic->state_length;

        if (disable_all) {
            memset(traffic->states, TL_STATE_DISABLED, fill_steps * sizeof(int));
            continue;
        }

        if (!env->eval_mode) {
            // Individual removal
            if (random_uniform(0.0f, 1.0f) < TL_INDIVIDUAL_REMOVE_PROB) {
                memset(traffic->states, TL_STATE_DISABLED, fill_steps * sizeof(int));
                continue;
            }
            // Always green
            if (random_uniform(0.0f, 1.0f) < TL_ALWAYS_GREEN_PROB) {
                for (int t = 0; t < fill_steps; t++)
                    traffic->states[t] = TL_STATE_GREEN;
                continue;
            }
        }

        // Compute phase durations
        float dur_green, dur_yellow, dur_red;
        if (env->eval_mode) {
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
                traffic->states[t] = TL_STATE_GREEN;
            else if (phase < steps_green + steps_yellow)
                traffic->states[t] = TL_STATE_YELLOW;
            else
                traffic->states[t] = TL_STATE_RED;
        }
    }
}

// Normalize a reward coefficient to [-1, 1] range for observations
static float normalize_reward_coef(float value, int coef_idx) {
    // NOTE: This prevents having coefficients outside of hardcoded bounds
    // What if we want to allow that?
    float min_value = REWARD_BOUNDS[coef_idx].min_val;
    float max_value = REWARD_BOUNDS[coef_idx].max_val;
    float range = max_value - min_value;

    // Safety: prevent division by zero if range is singular
    if (range < 1e-9f) {
        return 0.0f;
    }

    // Normalize to [0, 1]
    float normalized = (value - min_value) / range;

    // Clamp to [0, 1] to handle floating point noise
    if (normalized < 0.0f)
        normalized = 0.0f;
    if (normalized > 1.0f)
        normalized = 1.0f;

    // Scale to [-1, 1]
    return 2.0f * normalized - 1.0f;
}

// Find entity index given a lane ID
static int find_lane_index_by_id(Drive *env, int lane_id) {
    // NOTE: Maps should ensure lane IDs == Indexes for efficiency
    for (int i = 0; i < env->num_road_elements; i++) {
        if (env->road_elements[i].id == lane_id && is_drivable_road_lane(env->road_elements[i].type)) {
            return i;
        }
    }
    return -1;
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

static const GridMapEntity *get_neighbor_cache_entities(Drive *env, int cell_idx, int *out_count) {
    GridMap *grid_map = env->grid_map;
    if (cell_idx < 0 || cell_idx >= (grid_map->grid_cols * grid_map->grid_rows)) {
        *out_count = 0;
        return NULL;
    }

    *out_count = grid_map->neighbor_cache_count[cell_idx];
    return grid_map->neighbor_cache_entities[cell_idx];
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

        if (fread(&agent->id, sizeof(int), 1, file) != 1) {
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

        if (fread(&road->id, sizeof(int), 1, file) != 1) {
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

        if (road->type >= 0 && road->type <= 9) {
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

        if (fread(&traffic->id, sizeof(int), 1, file) != 1) {
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
        drive->lane_graph.lane_lengths = (float *)malloc(n_lanes_graph * sizeof(float));
        if (fread(drive->lane_graph.lane_lengths, sizeof(float), n_lanes_graph, file) != (size_t)n_lanes_graph) {
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
                                  float max_distance_threshold, int current_lane_index, float *signed_dist_out,
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
    if (current_lane_index != entity_idx && current_lane_index != -1) {
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
        int lane_idx = find_lane_index_by_id(env, agent->route[route_idx]);
        if (lane_idx == -1)
            continue;
        RoadMapElement *lane = &env->road_elements[lane_idx];

        for (int i = 0; i < lane->segment_length && wp_count < MAX_NUM_WP_PATH; i++) {
            float curr_x = lane->x[i];
            float curr_y = lane->y[i];
            float curr_z = lane->z[i];

            // First point: add directly
            if (wp_count == 0) {
                agent->path->waypoints[0] =
                    (struct Waypoint){.x = curr_x, .y = curr_y, .z = curr_z, .s = 0.0f, .lane_id = lane->id};
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
                    .lane_id = lane->id,
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
static int generate_random_route(Drive *env, int start_lane_id, float target_distance, int *route, int max_route_length,
                                 float agent_x, float agent_y) {
    // NOTE: This function performs a random walk through the lane connectivity graph,starting from start_lane_id,
    // until the accumulated distance exceeds target_distance or max_route_length is reached.
    // Cycles are avoided by tracking visited lanes and trying to increase distance.

    int route_length = 0;
    float accumulated_distance = 0.0f;
    int current_lane_id = start_lane_id;
    float start_x = agent_x;
    float start_y = agent_y;

    // Track visited lanes to avoid loops
    int visited_ids[MAX_ROUTE_LENGTH];
    int visited_count = 0;

    // Add start lane to route
    route[route_length++] = current_lane_id;
    visited_ids[visited_count++] = current_lane_id;

    // Get start lane and add its length
    int current_lane_idx = find_lane_index_by_id(env, current_lane_id);
    if (current_lane_idx == -1) {
        return 0; // Invalid start lane
    }

    RoadMapElement *current_lane = &env->road_elements[current_lane_idx];
    // Use remaining distance from agent position instead of full lane length
    accumulated_distance += compute_remaining_lane_distance(current_lane, agent_x, agent_y);
    float max_end_distance_sq = compute_lane_end_distance_sq(current_lane, start_x, start_y);

    // Random walk through lane graph
    while (accumulated_distance < target_distance && route_length < max_route_length) {
        current_lane_idx = find_lane_index_by_id(env, current_lane_id);
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
            int exit_id = current_lane->exit_lanes[e];

            // Check if already visited
            int already_visited = 0;
            for (int v = 0; v < visited_count; v++) {
                if (visited_ids[v] == exit_id) {
                    already_visited = 1;
                    break;
                }
            }
            if (already_visited)
                continue;

            // Check if exit lane is drivable
            int exit_lane_idx = find_lane_index_by_id(env, exit_id);
            if (exit_lane_idx == -1)
                continue;

            // NOTE: Dummy logic to prevent cycle, can be improved with better graph traversal
            float exit_end_distance_sq =
                compute_lane_end_distance_sq(&env->road_elements[exit_lane_idx], start_x, start_y);
            valid_exits[num_valid_exits] = exit_id;
            valid_exit_dist_sq[num_valid_exits] = exit_end_distance_sq;
            num_valid_exits++;

            if (exit_end_distance_sq > max_end_distance_sq) {
                progressing_exits[num_progressing_exits] = exit_id;
                progressing_dist_sq[num_progressing_exits] = exit_end_distance_sq;
                num_progressing_exits++;
            }
        }

        // If no valid exits, we've reached a dead end
        if (num_valid_exits == 0)
            break;

        // Pick a progressing exit lane if possible, otherwise pick the farthest available
        int chosen_exit_id;
        float chosen_exit_dist_sq;
        if (num_progressing_exits > 0) {
            int chosen_idx = rand() % num_progressing_exits;
            chosen_exit_id = progressing_exits[chosen_idx];
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
            chosen_exit_id = valid_exits[best_idx];
            chosen_exit_dist_sq = valid_exit_dist_sq[best_idx];
        }
        int chosen_exit_idx = find_lane_index_by_id(env, chosen_exit_id);

        if (chosen_exit_idx == -1)
            break;

        // Add to route
        route[route_length++] = chosen_exit_id;
        visited_ids[visited_count++] = chosen_exit_id;

        // Accumulate distance
        RoadMapElement *exit_lane = &env->road_elements[chosen_exit_idx];
        accumulated_distance += compute_lane_length(exit_lane);
        if (chosen_exit_dist_sq > max_end_distance_sq) {
            max_end_distance_sq = chosen_exit_dist_sq;
        }

        // Move to next lane
        current_lane_id = chosen_exit_id;
    }

    return route_length;
}

// NOTE: Only works for closed maps with infinite looping routes
static int compute_new_route(Drive *env, int agent_idx, int current_lane_id) {
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
    int route_length = generate_random_route(env, current_lane_id, min_route_distance, temp_route, MAX_ROUTE_LENGTH,
                                             agent->sim_x, agent->sim_y);

    if (route_length == 0) {
        printf("[GIGAFLOW WARNING] -> Failed to generate route for agent %d\n", agent->id);
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

static void compute_goals(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    struct Path *path = agent->path;

    // Validate path exists
    if (path == NULL || path->num_waypoints == 0) {
        printf("[GIGAFLOW WARNING] -> Agent %d has no valid path\n", agent->id);
        agent->removed = 1;
        return;
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
                printf("[GIGAFLOW WARNING] -> Max iterations in compute_goals for agent %d\n", agent->id);
                agent->removed = 1;
                return;
            }
            if (env->simulation_mode == SIMULATION_GIGAFLOW) {
                int route_ok = compute_new_route(env, agent_idx, path->waypoints[base_idx].lane_id);
                if (route_ok == 0) {
                    agent->removed = 1;
                    return;
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
        return;
    }

    printf("[GIGAFLOW ERROR] -> Failed to compute goals for agent %d after multiple attempts\n", agent->id);
    agent->removed = 1;
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

static bool check_stop_line_crossing(Drive *env, Agent *agent, int current_lane_id, float corners[4][2]) {
    float agent_x = agent->sim_x;
    float agent_y = agent->sim_y;

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];

        if (traffic->type != TRAFFIC_LIGHT)
            continue;
        if (traffic->num_controlled_lanes == 0)
            continue;

        int controls_lane = 0;
        for (int j = 0; j < traffic->num_controlled_lanes; j++) {
            if (traffic->controlled_lanes[j] == current_lane_id) {
                controls_lane = 1;
                break;
            }
        }
        if (!controls_lane)
            continue;
        if (env->timestep >= traffic->state_length)
            continue;
        if (!is_traffic_light_red(traffic->states[env->timestep]))
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
    if (agent->previous_lane_index == agent->current_lane_index)
        return false;
    if (agent->previous_lane_index == -1 || agent->current_lane_index == -1)
        return false;

    int new_lane_id = env->road_elements[agent->current_lane_index].id;
    float agent_x = agent->sim_x;
    float agent_y = agent->sim_y;

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];

        if (traffic->type != TRAFFIC_LIGHT)
            continue;
        if (traffic->num_controlled_lanes == 0)
            continue;
        if (env->timestep >= traffic->state_length)
            continue;
        if (!is_traffic_light_red(traffic->states[env->timestep]))
            continue;

        for (int j = 0; j < traffic->num_controlled_lanes; j++) {
            if (traffic->controlled_lanes[j] != new_lane_id)
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
    int current_lane_index = agent->current_lane_index;

    if (current_lane_index == -1)
        return false;

    // Compute bounding box corners: front-right(0), front-left(1), back-left(2), back-right(3)
    static const float offsets[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
    float half_length = agent->sim_length / 2.0f;
    float half_width = agent->sim_width / 2.0f;
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->sim_x + (offsets[i][0] * half_length * agent->cos_heading -
                                        offsets[i][1] * half_width * agent->sin_heading);
        corners[i][1] = agent->sim_y + (offsets[i][0] * half_length * agent->sin_heading +
                                        offsets[i][1] * half_width * agent->cos_heading);
    }

    int current_lane_id = env->road_elements[current_lane_index].id;

    if (check_stop_line_crossing(env, agent, current_lane_id, corners))
        return true;

    if (check_lane_change_red_light(env, agent))
        return true;

    return false;
}

// OBB collision via SAT (Separating Axis Theorem).
// Projects both boxes onto 4 axes (2 per car) and checks for overlap on all axes.
// No epsilon tolerance: exact boundary contact may flicker across steps.
static bool check_obb_collision(Agent *car1, Agent *car2) {
    // Get car corners in world space
    float cos1 = car1->cos_heading;
    float sin1 = car1->sin_heading;

    float cos2 = car2->cos_heading;
    float sin2 = car2->sin_heading;

    // Calculate half dimensions
    float half_len1 = car1->sim_length * 0.5f;
    float half_width1 = car1->sim_width * 0.5f;
    float half_len2 = car2->sim_length * 0.5f;
    float half_width2 = car2->sim_width * 0.5f;

    // Calculate car1's corners in world space
    float car1_corners[4][2] = {
        {car1->sim_x + (half_len1 * cos1 - half_width1 * sin1), car1->sim_y + (half_len1 * sin1 + half_width1 * cos1)},
        {car1->sim_x + (half_len1 * cos1 + half_width1 * sin1), car1->sim_y + (half_len1 * sin1 - half_width1 * cos1)},
        {car1->sim_x + (-half_len1 * cos1 - half_width1 * sin1),
         car1->sim_y + (-half_len1 * sin1 + half_width1 * cos1)},
        {car1->sim_x + (-half_len1 * cos1 + half_width1 * sin1),
         car1->sim_y + (-half_len1 * sin1 - half_width1 * cos1)}};

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->sim_x + (half_len2 * cos2 - half_width2 * sin2), car2->sim_y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->sim_x + (half_len2 * cos2 + half_width2 * sin2), car2->sim_y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->sim_x + (-half_len2 * cos2 - half_width2 * sin2),
         car2->sim_y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->sim_x + (-half_len2 * cos2 + half_width2 * sin2),
         car2->sim_y + (-half_len2 * sin2 - half_width2 * cos2)}};

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {cos1, sin1},  // Car1's length axis
        {-sin1, cos1}, // Car1's width axis
        {cos2, sin2},  // Car2's length axis
        {-sin2, cos2}  // Car2's width axis
    };

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
    }

    // If we get here, there's no separating axis, so the boxes intersect
    return true;
}

static bool check_z_collision_possibility(const Agent *car1, const Agent *car2) {
    float car1_bottom = car1->sim_z;
    float car1_top = car1->sim_z + car1->sim_height;
    float car2_bottom = car2->sim_z;
    float car2_top = car2->sim_z + car2->sim_height;

    return !(car1_top < car2_bottom || car2_top < car1_bottom);
}

static int collision_check(Drive *env, int agent_idx) {
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

        float dist_sq = ((other_agent->sim_x - agent->sim_x) * (other_agent->sim_x - agent->sim_x) +
                         (other_agent->sim_y - agent->sim_y) * (other_agent->sim_y - agent->sim_y));
        if (dist_sq > COLLISION_QUICK_CHECK_DIST * COLLISION_QUICK_CHECK_DIST)
            continue;
        if (!check_z_collision_possibility(agent, other_agent))
            continue;
        if (check_obb_collision(agent, other_agent)) {
            car_collided_with_index = index;
            break;
        }
    }

    return car_collided_with_index;
}

// Classify whether a collision is at-fault for the ego agent.
static bool is_at_fault_collision(Drive *env, int agent_idx, int other_idx) {
    Agent *agent = &env->agents[agent_idx];
    Agent *other = &env->agents[other_idx];

    // Rule 1: Collision with stopped vehicle = always at-fault.
    if (other->sim_speed < AGENT_STOPPED_SPEED_THRESHOLD)
        return true;

    // Rule 2: If ego is stopped = never at-fault.
    if (agent->sim_speed < AGENT_STOPPED_SPEED_THRESHOLD)
        return false;

    // Rule 3: Rear-bumper collision = not at-fault.
    // Check if the other car hit our rear using the heading-aligned relative position.
    float dx = other->sim_x - agent->sim_x;
    float dy = other->sim_y - agent->sim_y;
    float dot = dx * agent->cos_heading + dy * agent->sin_heading;
    if (dot < 0)
        return false;

    return true;
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

static void add_log(Drive *env) {
    int safe_timestep = (env->timestep > 0) ? env->timestep : 1;
    const float progress_ref_speed = 10.0f;
    for (int i = 0; i < env->active_agent_count; i++) {
        Agent *agent = &env->agents[env->active_agent_indices[i]];
        float episode_duration_s = env->logs[i].episode_length * env->dt;
        float reference_progress_distance = progress_ref_speed * episode_duration_s;
        reference_progress_distance = fmaxf(reference_progress_distance, 1.0f);
        env->logs[i].progress_ratio = agent->distance_since_spawn / reference_progress_distance;

        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        int red_light_violations = env->logs[i].red_light_violation_rate;
        env->log.red_light_violation_rate += red_light_violations;
        int total_infractions = (offroad || collided || red_light_violations) ? 1 : 0;
        float avg_speed_per_agent = env->logs[i].avg_speed_per_agent;
        env->log.avg_speed_per_agent += avg_speed_per_agent / safe_timestep;
        int num_waypoints_reached = env->logs[i].num_waypoints_reached;
        env->log.num_waypoints_reached += num_waypoints_reached;
        int num_goals_reached = env->logs[i].num_goals_reached;
        env->log.num_goals_reached += num_goals_reached;
        // TODO: define better scoring criteria ?
        if (num_goals_reached >= 1 && !agent->removed && !agent->stopped) {
            env->log.score += 1.0f;
        }
        if (!offroad && !collided && !red_light_violations && num_waypoints_reached < 1) {
            env->log.dnf_rate += 1.0f;
        }
        env->log.total_distance_travelled += agent->distance_since_spawn;
        if (total_infractions > 0) {
            env->log.total_infractions += 1.0f;
        }
        float displacement_error = env->logs[i].avg_displacement_error;
        env->log.avg_displacement_error += displacement_error;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        // Comfort and velocity metrics (normalized per timestep)
        env->log.comfort_violation_count += env->logs[i].comfort_violation_count / safe_timestep;
        env->log.velocity_progress_sum += env->logs[i].velocity_progress_sum / safe_timestep;
        // Lane metrics (normalized per timestep for average per episode)
        env->log.lane_center_rate += env->logs[i].lane_center_rate / safe_timestep;
        env->log.lane_heading_aligned_rate += env->logs[i].lane_heading_aligned_rate / safe_timestep;
        if (env->compute_eval_metrics) {
            env->logs[i].progress_ratio = agent->distance_since_spawn / reference_progress_distance;
            env->log.at_fault_collision_rate += env->logs[i].at_fault_collision_rate;
            env->log.ttc_within_bound_rate += env->logs[i].ttc_within_bound_rate;
            env->log.wrong_way_distance += env->logs[i].wrong_way_distance;
            env->log.speed_violation_sum += env->logs[i].speed_violation_sum;
            env->log.progress_ratio += env->logs[i].progress_ratio;
            env->log.comfort_score += env->logs[i].comfort_score;
            env->log.ttc_violations += env->logs[i].ttc_violations;
            env->log.ttc_samples += env->logs[i].ttc_samples;
            env->log.multi_lane_time += env->logs[i].multi_lane_time;
            env->log.multi_lane_score += env->logs[i].multi_lane_score;

            float wrong_dist = env->logs[i].wrong_way_distance;
            float direction_score = (wrong_dist <= 2.0f) ? 1.0f : (wrong_dist <= 6.0f) ? 0.5f : 0.0f;
            env->log.driving_direction_score += direction_score;

            float T = safe_timestep * env->dt;
            float speed_compliance = fmaxf(0.0f, 1.0f - env->logs[i].speed_violation_sum / fmaxf(T, 1e-3f));
            env->log.speed_limit_compliance += speed_compliance;

            float making_progress = (env->logs[i].progress_ratio > 0.2f) ? 1.0f : 0.0f;
            env->log.making_progress_rate += making_progress;
            env->log.puffer_score += calculate_puffer_score(&env->logs[i], safe_timestep, env->dt);
        }

        env->log.n += 1;
    }
    // Log composition counts per agent so vec_log averaging recovers the per-env value
    env->log.expert_static_car_count += env->expert_static_agent_count;
    env->log.static_car_count += env->static_agent_count;
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
    agent->current_lane_index = -1;
    agent->previous_lane_index = -1;
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
    agent->multi_lane_time = 0.0f;
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
    // Compute bounding box corners (same as offroad detection in step)
    static const float offsets[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
    // Increase length and width slightly for spawn offroad check
    float half_length = (spawn_length * 1.1f) / 2.0f;
    float half_width = (spawn_width * 1.1f) / 2.0f;
    float cos_heading = cosf(spawn_heading);
    float sin_heading = sinf(spawn_heading);

    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            spawn_x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] =
            spawn_y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

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
                                            float spawn_height, int lane_id) {
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

    static const float offsets[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
    float half_length = spawn_length / 2.0f;
    float half_width = spawn_width / 2.0f;
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = spawn_x + (offsets[i][0] * half_length * temp_agent.cos_heading -
                                   offsets[i][1] * half_width * temp_agent.sin_heading);
        corners[i][1] = spawn_y + (offsets[i][0] * half_length * temp_agent.sin_heading +
                                   offsets[i][1] * half_width * temp_agent.cos_heading);
    }

    return check_stop_line_crossing(env, &temp_agent, lane_id, corners);
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
    agent->id = agent_idx;
    agent->type = VEHICLE;
    agent->active_agent = 1;
    agent->mark_as_expert = 0;

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
    for (int attempt = 0; attempt < MAX_SPAWN_ATTEMPTS; attempt++) {
        int chosen_lane_idx = -1;

        int list_idx = rand() % env->grid_map->num_drivable_grid_cell;
        int grid_idx = env->grid_map->grid_index_drivable[list_idx];

        GridMapEntity cell_candidates[MAX_ENTITIES_PER_CELL];
        int candidate_count = 0;

        for (int i = 0; i < env->grid_map->cell_entities_count[grid_idx]; i++) {
            GridMapEntity entity = env->grid_map->cells[grid_idx][i];

            if (entity.entity_type == ENTITY_TYPE_ROAD_ELEMENT) {
                if (is_drivable_road_lane(env->road_elements[entity.entity_idx].type)) {
                    cell_candidates[candidate_count++] = entity;
                }
            }
        }

        if (candidate_count == 0)
            continue;

        GridMapEntity chosen_entity = cell_candidates[rand() % candidate_count];
        chosen_lane_idx = chosen_entity.entity_idx;

        start_lane_idx = chosen_lane_idx;
        start_lane = &env->road_elements[start_lane_idx];

        spawn_x = start_lane->x[chosen_entity.geometry_idx];
        spawn_y = start_lane->y[chosen_entity.geometry_idx];
        spawn_z = start_lane->z[chosen_entity.geometry_idx];
        spawn_heading = start_lane->headings[chosen_entity.geometry_idx];

        // Check for collision with existing/already-reset agents
        if (check_spawn_collision(env, num_agents, spawn_x, spawn_y, spawn_z, spawn_heading, spawn_length, spawn_width,
                                  spawn_height))
            continue;

        // Check for offroad (vehicle corners intersecting road edges)
        if (check_spawn_offroad(env, spawn_x, spawn_y, spawn_heading, spawn_length, spawn_width, spawn_z))
            continue;

        // Check for red light violation at spawn (vehicle corners intersecting stop lines)
        if (check_spawn_red_light_violation(env, spawn_x, spawn_y, spawn_z, spawn_heading, spawn_length, spawn_width,
                                            spawn_height, start_lane->id))
            continue;

        success = 1;
        break;
    }

    if (!success) {
        printf("[GIGAFLOW WARNING] -> Failed to find a collision-free spawn position for agent %d\n", agent->id);
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
    if (!compute_new_route(env, agent_idx, start_lane->id)) {
        printf("[GIGAFLOW WARNING] -> Failed to compute a new route for agent %d\n", agent->id);
        return 0; // Failed to compute new goal
    }

    // Compute initial goal
    compute_goals(env, agent_idx);

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

void set_active_agents(Drive *env) {
    // Initialize
    env->active_agent_count = 0;        // Policy-controlled agents
    env->static_agent_count = 0;        // Non-moving background agents
    env->expert_static_agent_count = 0; // Expert replay agents (non-controlled)
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

        for (int i = 0; i < env->num_total_agents; i++)
            env->active_agent_indices[i] = i;

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

    // Iterate through entities to find agents to create and/or control
    for (int i = 0; i < env->num_total_agents && env->num_agents < max_agents; i++) {

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
        } else if (is_log_replay || env->init_mode != INIT_ONLY_CONTROLLABLE_AGENTS) {
            // In log-replay mode, all non-controlled agents become expert_static
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->agents[i].active_agent = 0;
            if (is_log_replay || env->agents[i].mark_as_expert == 1 || env->active_agent_count == env->num_max_agents) {
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
            int collided_with_index = collision_check(env, agent_idx);
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

void init(Drive *env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    load_map_binary(env->map_name, env);
    init_grid_map(env);
    env->grid_map->vision_range = 2 * VISION_HALF_RANGE + 1;
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
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
                if (traffic->type != TRAFFIC_LIGHT)
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
    for (int i = 0; i < env->num_road_elements; i++)
        free_road_element(&env->road_elements[i]);
    for (int i = 0; i < env->num_traffic_elements; i++)
        free_traffic_element(&env->traffic_elements[i]);
    free(env->agents);
    free(env->road_elements);
    free(env->traffic_elements);
    free(env->active_agent_indices);
    free(env->logs);
    // GridMap cleanup
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(env->grid_map->cells[grid_index]);
    }
    free(env->grid_map->cells);
    free(env->grid_map->cell_entities_count);
    free(env->grid_map->grid_index_drivable);
    free(env->neighbor_offsets);

    for (int i = 0; i < grid_cell_count; i++) {
        free(env->grid_map->neighbor_cache_entities[i]);
    }
    free(env->grid_map->neighbor_cache_entities);
    free(env->grid_map->neighbor_cache_count);
    free(env->grid_map);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->objects_of_interest);
    free(env->tracks_to_predict);
    free_lane_graph(&env->lane_graph);
    free(env->map_name);
    free(env->ini_file);
}

static int compute_observation_size(Drive *env) {
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int num_target_waypoints = env->num_target_waypoints;
    if (num_target_waypoints > MAX_TARGET_WAYPOINTS) {
        num_target_waypoints = MAX_TARGET_WAYPOINTS;
    }

    if (env->max_stop_sign_observations > 0) {
        ego_dim += 1;
    }

    int max_obs = ego_dim + PARTNER_FEATURES * env->max_partner_observations +
                  ROAD_FEATURES * (env->max_lane_segment_observations + env->max_boundary_segment_observations) +
                  TRAFFIC_LIGHT_FEATURES * env->max_traffic_light_observations +
                  STOP_SIGN_FEATURES * env->max_stop_sign_observations;
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

static void compute_metrics(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    reset_agent_metrics(env, agent_idx);

    if (agent->sim_x == INVALID_POSITION)
        return; // invalid agent position

    // Current agent is offgrid, treat as offroad
    if (get_grid_index(env, agent->sim_x, agent->sim_y) == -1) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        if (env->offroad_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
        } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
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
    float half_length = agent->sim_length / 2.0f;
    float half_width = agent->sim_width / 2.0f;
    // Use cached trig values from move_dynamics
    float cos_heading = agent->cos_heading;
    float sin_heading = agent->sin_heading;

    // Track best candidate by combined distance/heading score
    float best_score = 1e9f;
    int best_candidate_entity_idx = -1;
    int best_candidate_geometry_idx = -1;
    float best_candidate_signed_lane_distance = 0.0f;
    float best_candidate_lane_heading = 0.0f;

    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            agent->sim_x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] =
            agent->sim_y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

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
                                           agent->current_lane_index, &signed_dist_out, &lane_heading_out);

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
        agent->previous_lane_index = agent->current_lane_index;
        agent->current_lane_index = best_candidate_entity_idx;
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
        agent->previous_lane_index = -1;
        agent->current_lane_index = -1;
        agent->current_lane_geometry_idx = -1;
        agent->metrics_array[LANE_DIST_IDX] = LANE_DISTANCE_NORMALIZATION; // Max distance (far from lane)
        agent->metrics_array[LANE_ANGLE_IDX] = 0.0f;                       // Perpendicular (no alignment)
    }

    agent->closest_path_idx_wp = get_closest_waypoint_index_on_path(env, agent_idx);

    // Speed limit metric (CUSTOM)
    float target_speed = 15.0f; // Default target speed
    int current_lane_index = agent->current_lane_index;
    if (current_lane_index != -1 && env->road_elements[current_lane_index].speed_limit > 0) {
        target_speed = env->road_elements[current_lane_index].speed_limit;
    }
    // Binary overspeed metric, 1.0 if overspeeding by more than 2 m/s
    agent->metrics_array[SPEED_LIMIT_IDX] = (agent->sim_speed > target_speed + 2.0f) ? 1.0f : 0.0f;
    if (env->compute_eval_metrics) {
        agent->speed_violation_sum += fmaxf(agent->sim_speed - target_speed, 0.0f) * env->dt;
    }

    // Velocity metric (GIGAFLOW) - forward progress aligned with lane
    const float VELOCITY_MIN_SPEED = 2.5f; // m/s
    if (agent->sim_speed > VELOCITY_MIN_SPEED && best_candidate_entity_idx != -1) {
        float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = fmaxf(cos_theta, 0.0f);
        if (env->compute_eval_metrics && cos_theta < 0.0f) {
            agent->wrong_way_distance += agent->sim_speed * env->dt;
        }
    } else {
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = 0.0f;
    }

    // Comfort metric (GIGAFLOW)
    const float COMFORT_ACCEL_THRESHOLD = 3.0f; // m/s²
    const float COMFORT_JERK_THRESHOLD = 5.0f;  // m/s³
    int accel_violation =
        (fabsf(agent->a_long) > COMFORT_ACCEL_THRESHOLD) + (fabsf(agent->a_lat) > COMFORT_ACCEL_THRESHOLD);
    int jerk_violation =
        (fabsf(agent->jerk_long) > COMFORT_JERK_THRESHOLD || fabsf(agent->jerk_lat) > COMFORT_JERK_THRESHOLD) ? 1 : 0;
    agent->metrics_array[COMFORT_VIOLATION_IDX] = (float)(accel_violation + jerk_violation);

    // Handle terminal events - NOTE: move it elsewhere?
    // IMPORTANT: early returns after offroad and collision enforce mutual exclusivity of
    // terminal flags. Red_light and stop_sign have NO early return and can coexist.
    // Order matters: offroad > collision > red_light/stop_sign.

    // Priority 1: Handle offroad
    if (is_offroad) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        if (env->offroad_behavior == STOP_AGENT && !agent->stopped) { // Stop
            agent->stopped = 1;
        } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
        }
        return; // early return: no other terminal flags set when offroad
    }

    // Priority 2: Handle vehicle collision
    int car_collided_with_index = collision_check(env, agent_idx);

    if (car_collided_with_index != -1) {
        agent->metrics_array[COLLISION_IDX] = 1.0f;
        // Track at-fault collisions for evaluation metrics.
        if (env->compute_eval_metrics && is_at_fault_collision(env, agent_idx, car_collided_with_index)) {
            agent->at_fault_collision = 1;
            agent->metrics_array[AT_FAULT_COLLISION_IDX] = 1.0f;
        }
        if (env->collision_behavior == STOP_AGENT && !agent->stopped) { // Stop
            agent->stopped = 1;
        } else if (env->collision_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
        }

        return; // early return: red_light/stop_sign not checked after collision
    }

    // Priority 3: Handle red light violation
    // Guard: violations are invisible to rewards when max_traffic_light_observations == 0
    if (env->max_traffic_light_observations && check_red_light_violation(env, agent_idx)) {
        agent->metrics_array[RED_LIGHT_IDX] = 1.0f;
        if (env->traffic_light_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
        } else if (env->traffic_light_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
        }
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

static void compute_rewards(Drive *env, int i) {
    int agent_idx = env->active_agent_indices[i];
    Agent *agent = &env->agents[agent_idx];

    // NOTE: compute_metrics enforces offroad > collision priority via early returns.
    // Red_light and stop_sign can coexist with each other (both flags may be set).
    // All penalty terms below are applied independently (no early returns here).

    // Collision reward (GIGAFLOW)
    if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
        // Velocity-dependent penalty: incentivizes braking before unavoidable collision.
        // At max speed (~20 m/s): extra -2.0 on top of base coefficient.
        float reward_collision = -(agent->reward_coefs[REWARD_COEF_COLLISION] + 0.1f * agent->sim_speed);

        env->rewards[i] += reward_collision;
        env->logs[i].episode_return += reward_collision;
        env->logs[i].collision_rate = 1.0f;
    }

    // Offroad reward (GIGAFLOW)
    if (agent->metrics_array[OFFROAD_IDX] > 0.0f) {
        float reward_offroad = -agent->reward_coefs[REWARD_COEF_OFFROAD];

        env->rewards[i] += reward_offroad;
        env->logs[i].offroad_rate = 1.0f;
        env->logs[i].episode_return += reward_offroad;
    }

    // Red light violation reward (GIGAFLOW)
    if (agent->metrics_array[RED_LIGHT_IDX] > 0.0f) {
        float reward_red_light = -agent->reward_coefs[REWARD_COEF_TRAFFIC_LIGHT];

        env->rewards[i] += reward_red_light;
        env->logs[i].red_light_violation_rate = 1.0f;
        env->logs[i].episode_return += reward_red_light;
    }

    // Goal reward
    if (agent->metrics_array[REACHED_GOAL_IDX] > 0.0f) {
        float weight = 1.0f;
        if (env->simulation_mode == SIMULATION_GIGAFLOW) {
            // current_goal_idx is already incremented in compute_metrics when goal is reached.
            // == num_target_waypoints means the last waypoint was just reached.
            // weight=0 suppresses the final goal reward when agent is already moving (> 3 m/s).
            // This prevents rewarding a drive-through of the last waypoint.
            if (agent->current_goal_idx == env->num_target_waypoints && agent->sim_speed > env->goal_speed_threshold)
                weight = 0.0f;
        }

        env->rewards[i] += env->reward_goal * weight;
        env->logs[i].episode_return += env->reward_goal * weight;
        env->logs[i].num_waypoints_reached += 1;
    }

    // Get lane angle metric: cos(θ_f) where θ_f = heading diff from lane
    float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
    float theta_f = acosf(fminf(fmaxf(cos_theta, -1.0f), 1.0f)); // Get |θ_f| from cos
    env->logs[i].lane_heading_aligned_rate += (cos_theta >= LANE_ALIGN_COS_THRESHOLD) ? 1.0f : 0.0f;

    // Rl-align (GIGAFLOW): min(cos,0) + vel_align*min(cos*v,0) + 0.0025*(1-|θ|/(π/2))
    float against_lane_penalty = fminf(cos_theta, 0.0f); // negative when >90 degrees off
    float vel_aligned_penalty = agent->reward_coefs[REWARD_COEF_VEL_ALIGN] * fminf(cos_theta * agent->sim_speed, 0.0f);
    float alignment_bonus = 0.0025f * (1.0f - theta_f / (M_PI / 2.0f));

    float lane_align_reward = agent->reward_coefs[REWARD_COEF_LANE_ALIGN] * env->dt *
                              (against_lane_penalty + vel_aligned_penalty + alignment_bonus);

    env->rewards[i] += lane_align_reward;
    env->logs[i].episode_return += lane_align_reward;

    // Rl-center (GIGAFLOW): -α * dt * (|x_f - bias| - 0.05 / exp(|x_f - bias| - 0.5))
    float lane_center_distance = agent->metrics_array[LANE_DIST_IDX];
    float adjusted_dist = fabsf(lane_center_distance - agent->reward_coefs[REWARD_COEF_CENTER_BIAS]);
    float exp_decay = 0.05f / expf(adjusted_dist - 0.5f);

    float lane_center_reward =
        -agent->reward_coefs[REWARD_COEF_LANE_CENTER] * env->dt * ((cos_theta > 0.5f) * adjusted_dist - exp_decay);

    env->rewards[i] += lane_center_reward;
    env->logs[i].lane_center_rate += fabsf(lane_center_distance) < 0.5f ? 1.0f : 0.0f;
    env->logs[i].episode_return += lane_center_reward;

    // Comfort reward (GIGAFLOW)
    float comfort_violations = agent->metrics_array[COMFORT_VIOLATION_IDX];
    float comfort_penalty = -agent->reward_coefs[REWARD_COEF_COMFORT] * comfort_violations;

    env->rewards[i] += comfort_penalty;
    env->logs[i].comfort_violation_count += comfort_violations;
    env->logs[i].episode_return += comfort_penalty;

    // Velocity reward (GIGAFLOW)
    float velocity_progress = agent->metrics_array[VELOCITY_PROGRESS_IDX];
    float velocity_reward = agent->reward_coefs[REWARD_COEF_VELOCITY] * env->dt * velocity_progress;

    env->rewards[i] += velocity_reward;
    env->logs[i].episode_return += velocity_reward;
    env->logs[i].velocity_progress_sum += velocity_progress;

    // Timestep reward (GIGAFLOW)
    float accel = sqrtf(agent->a_long * agent->a_long + agent->a_lat * agent->a_lat);
    // Only penalize when moving (v > 0) or accelerating (a > 0)
    if (agent->sim_speed > 0.01f || accel > 0.01f) {
        float timestep_penalty = -agent->reward_coefs[REWARD_COEF_TIMESTEP] * env->dt;

        env->rewards[i] += timestep_penalty;
        env->logs[i].episode_return += timestep_penalty;
    }

    // Reverse reward (GIGAFLOW)
    if (agent->sim_speed_signed < -0.01f) {
        float reverse_penalty = -agent->reward_coefs[REWARD_COEF_REVERSE] * env->dt;

        env->rewards[i] += reverse_penalty;
        env->logs[i].episode_return += reverse_penalty;
    }

    // Over speed reward (GIGAFLOW++)
    float speed_reward = -agent->reward_coefs[REWARD_COEF_OVERSPEED] * agent->metrics_array[SPEED_LIMIT_IDX];

    env->rewards[i] += speed_reward;
    env->logs[i].avg_speed_per_agent += agent->sim_speed;
    agent->distance_since_spawn += agent->sim_speed * env->dt;
    env->logs[i].episode_return += speed_reward;

    // ADE reward (CUSTOM)
    float current_ade = agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX];
    if (current_ade > 0.0f && env->reward_ade != 0.0f) {
        float ade_reward = env->reward_ade * current_ade;

        env->rewards[i] += ade_reward;
        env->logs[i].episode_return += ade_reward;
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
}

static void compute_observations(Drive *env) {
    int max_obs = compute_observation_size(env);

    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &observations[i][0];
        int obs_idx = 0;
        Agent *ego_entity = &env->agents[env->active_agent_indices[i]];

        // Use cached trig values (updated in move_dynamics/spawn)
        float cos_heading = ego_entity->cos_heading;
        float sin_heading = ego_entity->sin_heading;

        obs[obs_idx++] = ego_entity->sim_speed_signed / MAX_SPEED;
        obs[obs_idx++] = ego_entity->sim_width / MAX_VEH_WIDTH;
        obs[obs_idx++] = ego_entity->sim_length / MAX_VEH_LEN;
        obs[obs_idx++] = ego_entity->steering_angle / STEERING_VALUES[8];

        if (env->dynamics_model == JERK) {
            obs[obs_idx++] =
                (ego_entity->a_long < 0) ? ego_entity->a_long / (-JERK_LONG[0]) : ego_entity->a_long / JERK_LONG[3];
            obs[obs_idx++] = ego_entity->a_lat / JERK_LAT[2];
        }

        float lane_center_dist = ego_entity->metrics_array[LANE_DIST_IDX] / LANE_DISTANCE_NORMALIZATION;
        lane_center_dist = fmaxf(-1.0f, fminf(1.0f, lane_center_dist));
        obs[obs_idx++] = lane_center_dist;
        obs[obs_idx++] = ego_entity->metrics_array[LANE_ANGLE_IDX];

        float speed_limit = 20.0f;
        int current_lane_index = ego_entity->current_lane_index;
        if (current_lane_index != -1 && env->road_elements[current_lane_index].speed_limit > 0) {
            speed_limit = env->road_elements[current_lane_index].speed_limit;
        }
        obs[obs_idx++] = fminf(speed_limit / MAX_SPEED, 1.0f);

        // Encoder -> Conditioning and goal waypoints
        if (env->reward_conditioning) {
            for (int c = 0; c < NUM_REWARD_COEFS; c++) {
                obs[obs_idx++] = normalize_reward_coef(ego_entity->reward_coefs[c], c);
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
                    obs[obs_idx++] = (gx * cos_heading + gy * sin_heading) / MAX_GOAL_POSITION;
                    obs[obs_idx++] = (-gx * sin_heading + gy * cos_heading) / MAX_GOAL_POSITION;
                    obs[obs_idx++] = (ego_entity->goal_positions_z[wp] - ego_entity->sim_z) / MAX_POSITION;
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
                    float rel_wp_x = wp_x * cos_heading + wp_y * sin_heading;
                    float rel_wp_y = -wp_x * sin_heading + wp_y * cos_heading;
                    float rel_heading_x = wp_cos_h * cos_heading + wp_sin_h * sin_heading;
                    float rel_heading_y = wp_sin_h * cos_heading - wp_cos_h * sin_heading;
                    obs[obs_idx++] = rel_wp_x / MAX_POSITION;
                    obs[obs_idx++] = rel_wp_y / MAX_POSITION;
                    obs[obs_idx++] = rel_heading_x;
                    obs[obs_idx++] = rel_heading_y;
                    obs[obs_idx++] = wp_z / MAX_POSITION;
                }
            } else {
                // No valid path - zero out
                obs_idx += DYNAMIC_TARGET_FEATURES * env->num_target_waypoints;
            }
        }

        // Encode -> Partner vehicle
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
            if (env->agents[index].type >= OTHER)
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
            if (dist_sq > AGENT_OBS_MAX_DIST * AGENT_OBS_MAX_DIST || abs_dz > Z_BUFFER)
                continue;
            // Add to candidate list
            candidates[candidate_count].index = index;
            candidates[candidate_count].dist_sq = dist_sq;
            candidates[candidate_count].dx = dx;
            candidates[candidate_count].dy = dy;
            candidates[candidate_count].dz = dz;
            candidate_count++;
        }
        int cars_seen = 0;
        // Find K closest candidates using partial selection (O(N*K) instead of O(N log N) qsort)
        if (candidate_count > 0) {
            int num_agents_to_observe =
                (candidate_count < env->max_partner_observations) ? candidate_count : env->max_partner_observations;

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
                float rel_x = dx * cos_heading + dy * sin_heading;
                float rel_y = -dx * sin_heading + dy * cos_heading;
                // Store observations with correct indexing
                obs[obs_idx++] = rel_x / MAX_POSITION;
                obs[obs_idx++] = rel_y / MAX_POSITION;
                obs[obs_idx++] = other_entity->sim_width / MAX_VEH_WIDTH;
                obs[obs_idx++] = other_entity->sim_length / MAX_VEH_LEN;
                // relative heading
                // Use cached trig values from other agent
                float other_cos = other_entity->cos_heading;
                float other_sin = other_entity->sin_heading;
                float rel_heading_x =
                    other_cos * cos_heading + other_sin * sin_heading; // cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
                float rel_heading_y =
                    other_sin * cos_heading - other_cos * sin_heading; // sin(a-b) = sin(a)cos(b) - cos(a)sin(b)

                obs[obs_idx++] = rel_heading_x;
                obs[obs_idx++] = rel_heading_y;
                obs[obs_idx++] = other_entity->sim_speed / MAX_SPEED;
                obs[obs_idx++] = dz / MAX_POSITION;
                cars_seen++;
            }
        }
        int remaining_partner_obs = (env->max_partner_observations - cars_seen) * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;

        // Encode -> Road segments (lanes and boundaries)
        int grid_idx = get_grid_index(env, ego_entity->sim_x, ego_entity->sim_y);
        int list_size;
        const GridMapEntity *entity_list = get_neighbor_cache_entities(env, grid_idx, &list_size);

        int lane_added = 0;
        int boundary_added = 0;
        int lane_obs_idx = obs_idx;
        int boundary_obs_idx = lane_obs_idx + env->max_lane_segment_observations * ROAD_FEATURES;

        for (int k = 0; k < list_size; k++) {
            if (lane_added >= env->max_lane_segment_observations &&
                boundary_added >= env->max_boundary_segment_observations) {
                break;
            }
            int entity_type = entity_list[k].entity_type;
            int entity_idx = entity_list[k].entity_idx;
            int geometry_idx = entity_list[k].geometry_idx;

            // Only process road elements in observations
            if (entity_type != ENTITY_TYPE_ROAD_ELEMENT)
                continue;

            // Validate entity_idx before accessing
            if (entity_idx < 0 || entity_idx >= env->num_road_elements) {
                printf("ERROR: Invalid road element idx %d (max: %d)\n", entity_idx, env->num_road_elements - 1);
                continue;
            }

            RoadMapElement *element = &env->road_elements[entity_idx];
            int is_lane = is_road_lane(element->type);
            int is_edge = is_road_edge(element->type);
            if (!is_lane && !is_edge) {
                continue;
            }

            // Validate geometry_idx before accessing
            if (geometry_idx < 0 || geometry_idx >= element->segment_length - 1) {
                printf("ERROR: Invalid geometry_idx %d for road element %d (max: %d)\n", geometry_idx, entity_idx,
                       element->segment_length - 2);
                continue;
            }
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
            float x_obs = rel_x * cos_heading + rel_y * sin_heading;
            float y_obs = -rel_x * sin_heading + rel_y * cos_heading;

            // Filter by asymmetric vision rectangle
            if (x_obs < -ROAD_OBS_BEHIND_DIST || x_obs > ROAD_OBS_FRONT_DIST)
                continue;
            if (fabsf(y_obs) > ROAD_OBS_SIDE_DIST)
                continue;
            if (fabsf(rel_z) > Z_BUFFER)
                continue;

            // Compute segment direction and length (single sqrt instead of two)
            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float length = sqrtf(dx * dx + dy * dy);
            float dx_norm = (length > 0) ? dx / length : dx;
            float dy_norm = (length > 0) ? dy / length : dy;
            float cos_angle = dx_norm * cos_heading + dy_norm * sin_heading;
            float sin_angle = -dx_norm * sin_heading + dy_norm * cos_heading;

            int write_idx = 0;
            if (is_lane) {
                if (lane_added >= env->max_lane_segment_observations) {
                    continue;
                }
                write_idx = lane_obs_idx + lane_added * ROAD_FEATURES;
                lane_added++;
            } else {
                if (boundary_added >= env->max_boundary_segment_observations) {
                    continue;
                }
                write_idx = boundary_obs_idx + boundary_added * ROAD_FEATURES;
                boundary_added++;
            }

            obs[write_idx] = x_obs / MAX_POSITION;
            obs[write_idx + 1] = y_obs / MAX_POSITION;
            obs[write_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[write_idx + 3] = 0.1f / MAX_ROAD_SCALE;
            obs[write_idx + 4] = cos_angle;
            obs[write_idx + 5] = sin_angle;
            obs[write_idx + 6] = rel_z / MAX_POSITION;
        }
        int remaining_lane_obs = (env->max_lane_segment_observations - lane_added) * ROAD_FEATURES;
        memset(&obs[lane_obs_idx + lane_added * ROAD_FEATURES], 0, remaining_lane_obs * sizeof(float));
        int remaining_boundary_obs = (env->max_boundary_segment_observations - boundary_added) * ROAD_FEATURES;
        memset(&obs[boundary_obs_idx + boundary_added * ROAD_FEATURES], 0, remaining_boundary_obs * sizeof(float));
        obs_idx = boundary_obs_idx + env->max_boundary_segment_observations * ROAD_FEATURES;

        // Traffic light observations
        typedef struct {
            int idx;
            float dist_sq;
        } TrafficLightDist;
        TrafficLightDist traffic_lights[env->num_traffic_elements > 0 ? env->num_traffic_elements : 1];
        int num_visible_lights = 0;

        // Collect traffic lights within range
        for (int j = 0; j < env->num_traffic_elements; j++) {
            TrafficControlElement *traffic = &env->traffic_elements[j];
            if (traffic->type != TRAFFIC_LIGHT)
                continue; // Only traffic lights

            float mid_x = (traffic->stop_line[0] + traffic->stop_line[3]) * 0.5f;
            float mid_y = (traffic->stop_line[1] + traffic->stop_line[4]) * 0.5f;
            float mid_z = (traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f;
            float dx = mid_x - ego_entity->sim_x;
            float dy = mid_y - ego_entity->sim_y;
            float dz = mid_z - ego_entity->sim_z;
            float abs_dz = fabsf(dz);
            float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq > MAX_TRAFFIC_LIGHT_DISTANCE * MAX_TRAFFIC_LIGHT_DISTANCE || abs_dz > Z_BUFFER)
                continue;

            traffic_lights[num_visible_lights].idx = j;
            traffic_lights[num_visible_lights].dist_sq = dist_sq;
            num_visible_lights++;
        }

        // Partial selection sort: find K closest (O(N*K) instead of O(N²) bubble sort)
        int num_lights_to_observe = (num_visible_lights < env->max_traffic_light_observations)
                                        ? num_visible_lights
                                        : env->max_traffic_light_observations;
        for (int k = 0; k < num_lights_to_observe; k++) {
            int min_idx = k;
            for (int j = k + 1; j < num_visible_lights; j++) {
                if (traffic_lights[j].dist_sq < traffic_lights[min_idx].dist_sq) {
                    min_idx = j;
                }
            }
            if (min_idx != k) {
                TrafficLightDist temp = traffic_lights[k];
                traffic_lights[k] = traffic_lights[min_idx];
                traffic_lights[min_idx] = temp;
            }
        }

        // Add observations for closest traffic lights
        int lights_added = 0;
        for (int j = 0; j < num_lights_to_observe && lights_added < env->max_traffic_light_observations; j++) {
            TrafficControlElement *traffic = &env->traffic_elements[traffic_lights[j].idx];

            // Stop line endpoint 1
            float dx1 = traffic->stop_line[0] - ego_entity->sim_x;
            float dy1 = traffic->stop_line[1] - ego_entity->sim_y;
            float rel_x1 = dx1 * cos_heading + dy1 * sin_heading;
            float rel_y1 = -dx1 * sin_heading + dy1 * cos_heading;

            // Stop line endpoint 2
            float dx2 = traffic->stop_line[3] - ego_entity->sim_x;
            float dy2 = traffic->stop_line[4] - ego_entity->sim_y;
            float rel_x2 = dx2 * cos_heading + dy2 * sin_heading;
            float rel_y2 = -dx2 * sin_heading + dy2 * cos_heading;
            float rel_z = ((traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f - ego_entity->sim_z) / MAX_POSITION;

            // Get current state with bounds check
            int t = env->timestep;
            int state = TL_STATE_DISABLED;
            if (t >= 0 && t < traffic->state_length) {
                state = traffic->states[t];
            }

            obs[obs_idx++] = rel_x1 / MAX_POSITION;
            obs[obs_idx++] = rel_y1 / MAX_POSITION;
            obs[obs_idx++] = rel_x2 / MAX_POSITION;
            obs[obs_idx++] = rel_y2 / MAX_POSITION;
            obs[obs_idx++] = rel_z;
            obs[obs_idx++] = normalize_traffic_light_state(state);
            lights_added++;
        }

        // Zero out remaining traffic light slots
        int remaining_traffic_obs = (env->max_traffic_light_observations - lights_added) * TRAFFIC_LIGHT_FEATURES;
        memset(&obs[obs_idx], 0, remaining_traffic_obs * sizeof(float));
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

        // Limit the steering rate similar to the jerk model
        float delta_steer = clip(steering - agent->steering_angle, -0.6f * env->dt, 0.6f * env->dt);
        steering = clip(agent->steering_angle + delta_steer, -0.667f, 0.667f);
        agent->steering_angle = steering;

        // Current state
        float x = agent->sim_x;
        float y = agent->sim_y;
        float heading = agent->sim_heading;
        float speed = agent->sim_speed_signed;

        // Update speed with acceleration
        speed += acceleration * env->dt;
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
            a_long_new = clip(a_long_new, -5.0f, 2.5f * c_acc);
        }

        // Calculate new lateral acceleration from jerk (Eq. 2 in paper)
        float a_lat_new = agent->a_lat + c_steer * j_lat * env->dt;

        // Zero-crossing: snap to 0 when crossing zero
        if (agent->a_lat * a_lat_new < 0) {
            a_lat_new = 0.0f;
        } else {
            a_lat_new = clip(a_lat_new, -4.0f, 4.0f);
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
            v_new = clip(v_new, -2.0f, 20.0f);
        }

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
                if (entity_list[i].entity_idx == agent->current_lane_index) {
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

void c_reset(Drive *env) {
    if (env->timestep == 0) {
        for (int x = 0; x < env->active_agent_count; x++) {
            env->logs[x] = (Log){0};
            int agent_idx = env->active_agent_indices[x];
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
        generate_reward_coefs(env, agent);

        compute_goals(env, agent_idx);
        initialize_agent_progression(env, agent_idx);
        compute_metrics(env, agent_idx);
    }
    compute_observations(env);
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));

    // Update masks for stopped/removed agents
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->agents[agent_idx].stopped || env->agents[agent_idx].removed) {
            env->masks[i] = 0;
        } else {
            env->masks[i] = 1;
        }
    }

    env->timestep++;

    // -> 1. Check for episode termination
    int early_reset = 0;
    // Special early reset condition for GIGAFLOW
    if (env->simulation_mode == SIMULATION_GIGAFLOW && env->termination_mode == 1) {
        int count_inactive = 0;
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            if (env->agents[agent_idx].removed || env->agents[agent_idx].stopped) {
                count_inactive++;
            }
        }
        float ratio_inactive = (float)count_inactive / (float)env->active_agent_count;
        // If more than inactive_agent_threshold of agents are removed/stopped, reset the environment
        if (ratio_inactive > env->inactive_agent_threshold) {
            early_reset = 1;
        }
    }

    if (env->timestep == env->scenario_length || early_reset) {
        for (int i = 0; i < env->active_agent_count; i++) {
            env->truncations[i] = 1;
        }
        add_log(env);
        c_reset(env);
        return;
    }

    // -> 2. Apply actions and move agents
    // Move static experts
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        move_expert(env, env->actions, expert_idx);
    }
    // Move active agents with policy actions
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        move_dynamics(env, i, agent_idx);
        // move_expert(env, env->actions, agent_idx);
    }

    // -> 3. Compute metrics and rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];

        if (env->agents[agent_idx].stopped || env->agents[agent_idx].removed)
            continue;

        // Compute metrics
        compute_metrics(env, agent_idx);

        // Compute rewards
        compute_rewards(env, i);
    }

    // Mark terminals for stopped or removed agents
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->agents[agent_idx].stopped || env->agents[agent_idx].removed) {
            env->terminals[i] = 1;
        }
    }

    // -> 5. Compute observations
    compute_observations(env);

    // -> 6. Update goals for agents that reached their goal
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
