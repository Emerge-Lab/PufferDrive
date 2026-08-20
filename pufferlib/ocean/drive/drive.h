// _GNU_SOURCE is set via -D_GNU_SOURCE in setup.py's drive extension build
// flags so GNU extensions (F_SETPIPE_SZ, writev, etc.) are visible regardless
// of which header is included first.
#include "datatypes.h"
#include "error.h"
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "rng.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// EGL is optional: only compile in the EGL headless path if the headers
// are available. CI environments without libegl1-mesa-dev skip this entirely
// and fall back to Xvfb/Mesa software rendering.
#if defined(__linux__) && defined(__has_include)
#if __has_include(<EGL/egl.h>)
#define DRIVE_HAS_EGL 1
#endif
#endif

#ifdef DRIVE_HAS_EGL
// GL_GLEXT_PROTOTYPES must come before any GL/gl.h include so glext declares
// the modern buffer-object entry points (glGenBuffers, glBindBuffer,
// glBufferData, glMapBuffer, glUnmapBuffer, glDeleteBuffers). Without a
// declaration, gcc defaults their return type to implicit int, and
// glMapBuffer's void* pointer gets truncated to 32 bits and sign-extended,
// producing EFAULT writes like 0xffffffff9cbf1000.
#define GL_GLEXT_PROTOTYPES 1
#include "egl_headless.h"

#include <GL/gl.h>
#include <GL/glext.h>
#endif

typedef struct {
    float z_dis;
    float euclidean_dis;
    float z;
} DepthPoint;

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;
typedef struct Agent Agent;
typedef struct RoadMapElement RoadMapElement;
typedef struct TrafficControlElement TrafficControlElement;
typedef struct GridMapEntity GridMapEntity;
typedef struct GridMap GridMap;

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
    float num_goals_reached;
    float comfort_violation_count;
    float comfort_violation_window_count;
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
    // Agent-only puffer display fields (for serialization)
    float no_at_fault;
    float no_offroad;
    float no_red_light;
    float making_progress;
    float ttc_puffer_rate;
    float multiplier;
    float weighted_average;
    float reached_goal_gt;
    float reward_collision;
    float reward_offroad;
    float reward_red_light;
    float reward_goal;
    float reward_lane_align;
    float reward_lane_center;
    float reward_comfort;
    float reward_velocity;
    float reward_timestep;
    float reward_reverse;
    float reward_overspeed;
    float reward_ade;
};

struct GridMapEntity {
    int entity_idx;    // Index into the road_elements array
    int geometry_idx;  // Index into element's geometry array
    int valid_for_obs; // Whether this entity should be included in observations
};

struct GridMap {
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int grid_cols;
    int grid_rows;
    int vision_range;
    int *cell_entities_count;
    int *neighbor_cache_count;
    int *grid_index_drivable;
    int num_drivable_grid_cell;
    int total_entities;
    GridMapEntity **cells;
    GridMapEntity **neighbor_cache_entities;
};

// Static, read-only map geometry shared across envs loading the same map file
// when use_map_cache is set: road geometry, spatial grid (cells + neighbor
// cache), and lane graph. Per-env mutable data (agents, traffic-light states)
// is never shared.
//
// Reference-counted. owner_pid is set at create_shared_map_data; c_close frees
// the entry only when owner_pid == getpid(), so a process that inherits an
// entry via fork-COW does not free it.
struct SharedMapData {
    char *map_name;
    RoadMapElement *road_elements;
    int num_road_elements;
    GridMap *grid_map;
    int *neighbor_offsets;
    struct LaneGraph lane_graph;
    int obs_lane_stride;
    int obs_boundary_stride;
    int ref_count;
    pid_t owner_pid;
};

// Per-process map cache. Built lazily in init(); freeing is gated by per-entry
// owner_pid in c_close.
static struct SharedMapData **g_map_cache = NULL;
static int g_map_cache_count = 0;

struct Drive {
    // Buffers
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;
    unsigned char *masks;
    // Agents
    Agent *agents;
    int num_controllable_agents;
    int active_agent_count;
    int *active_agent_indices;
    int num_total_agents;
    int num_max_agents;
    int num_agents;
    int action_type;
    int human_agent_idx;
    int static_agent_count;
    int *static_agent_indices;
    int expert_static_agent_count;
    int *expert_static_agent_indices;
    // Map and spatial queries
    char *map_name;
    RoadMapElement *road_elements;
    int num_road_elements;
    TrafficControlElement *traffic_elements;
    int num_traffic_elements;
    struct LaneGraph lane_graph;
    GridMap *grid_map;
    GridMapEntity *obs_neighbor_scratch;
    int *neighbor_offsets;
    int use_map_cache;
    int use_neighbor_cache;
    struct SharedMapData *shared_map;
    float world_mean_x;
    float world_mean_y;
    // Scenario data
    char *ini_file;
    char scenario_id[128];
    char dataset_name[32];
    int scenario_length;
    int log_length;
    float log_dt;
    int num_objects;
    int num_objects_of_interest;
    int *objects_of_interest;
    int num_tracks_to_predict;
    int *tracks_to_predict;
    // Simulation
    int timestep;
    int init_step;
    float dt;
    float spawn_initial_speed;
    int dynamics_model;
    int reset_accel_on_stop;
    int init_mode;
    int control_mode;
    int collision_behavior;
    int offroad_behavior;
    int traffic_light_behavior;
    int sdc_controller;
    int non_sdc_controller;
    int non_vehicle_controller;
    int simulation_mode;
    int termination_mode;
    float inactive_agent_threshold;
    int terminate_on_goal;
    int eval_mode;
    int compute_eval_metrics;
    // Rewards
    float reward_goal;
    float reward_collision;
    float reward_offroad;
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
    int reward_conditioning;
    int reward_randomization;
    int reward_log_sampling;
    // Goals
    float goal_radius;
    float goal_speed;
    float min_goal_spacing;
    float max_goal_spacing;
    int num_goals;
    int goal_regen_mode;
    int goal_source;
    int obs_goal_lane_distance;
    // Observations
    int obs_slots_boundary_n;
    int obs_slots_lane_n;
    int obs_slots_partners_n;
    int obs_slots_traffic_controls_n;
    int traffic_control_scope;
    int obs_lane_stride;
    int obs_boundary_stride;
    int obs_slots_lane_kept;
    int obs_slots_boundary_kept;
    int road_dropout_enabled;
    float obs_norm_goal_offset_m;
    float obs_norm_xy_offset_m;
    float obs_norm_veh_length_m;
    float obs_norm_veh_width_m;
    float obs_norm_road_seg_length_m;
    float obs_norm_road_seg_width_m;
    float obs_norm_z_m;
    float eval_perceived_size_margin_m;
    float obs_range_traffic_control_m;
    float obs_range_partner_m;
    float obs_range_road_front_m;
    float obs_range_road_behind_m;
    float obs_range_road_side_m;
    // Robustness
    float partner_blindness_prob;
    float partner_blindness_trigger_prob;
    int partner_blindness_duration;
    float phantom_braking_prob;
    float phantom_braking_trigger_prob;
    int phantom_braking_duration;
    // Logging
    Log log;
    Log *logs;
    int logs_capacity;
    // Seed
    int eval_episode_done;
    int use_exact_episode_seed;
    Rng rng_state;
    Rng seed_stream_rng;
    uint64_t init_seed;
    uint64_t episode_seed;
    uint64_t log_episode_seed;
    // Runtime
    Client *client;
    int render_mode;
    // Rendering
    char video_suffix[64];
    char resource_root[512];
};

typedef struct {
    float min_val;
    float max_val;
    int log_scale;
} RewardBound;

static const RewardBound REWARD_BOUNDS[NUM_REWARD_COEFS] = {
    {2.0f, 12.0f, 0},      // REWARD_COEF_GOAL_RADIUS     δ_goal ~ U(2, 12)
    {0.0f, 20.0f, 0},      // REWARD_COEF_GOAL_SPEED      δ_goal-speed ~ U(0, 20)
    {0.0f, 3.0f, 0},       // REWARD_COEF_COLLISION       α_collision ~ U(0, 3)
    {0.0f, 3.0f, 0},       // REWARD_COEF_OFFROAD         α_boundary ~ U(0, 3)
    {0.0f, 0.1f, 0},       // REWARD_COEF_COMFORT         α_comfort ~ U(0, 0.1)
    {2.5e-4f, 2.5e-2f, 0}, // REWARD_COEF_LANE_ALIGN      α_l-align ~ U(2.5e-4, 2.5e-2)
    {0.0f, 1.0f, 0},       // REWARD_COEF_VEL_ALIGN       α_vel-align ~ U(0, 1)
    {2.5e-4f, 7.5e-3f, 0}, // REWARD_COEF_LANE_CENTER     α_l-center ~ U(2.5e-4, 7.5e-3)
    {-0.5f, 0.5f, 0},      // REWARD_COEF_CENTER_BIAS     α_center-bias ~ U(-0.5, 0.5)
    {0.0f, 5e-3f, 0},      // REWARD_COEF_VELOCITY        α_velocity ~ U(0, 5e-3f)
    {2.5e-4f, 7.5e-3f, 0}, // REWARD_COEF_REVERSE         α_reverse ~ U(2.5e-4, 7.5e-3)
    {0.0f, 1.0f, 0},       // REWARD_COEF_STOP_LINE       α_stop-line ~ U(0, 1)
    {0.0f, 5e-5f, 0},      // REWARD_COEF_TIMESTEP        α_timestep ~ U(0, 5e-5f)
    {0.0f, 1.0f, 0},       // REWARD_COEF_OVERSPEED       α_overspeed ~ U(0, 1)
    {0.8f, 1.25f, 0},      // REWARD_COEF_THROTTLE        C_throttle
    {0.8f, 1.25f, 0},      // REWARD_COEF_STEER           C_steer
    {0.666f, 1.5f, 0},     // REWARD_COEF_ACC             C_acc
};

// Meaning of the values: [min_range, max_range, use_log_scale]
static const RewardBound REWARD_BOUNDS_LOG[NUM_REWARD_COEFS] = {
    {2.0f, 12.0f, 0},      // REWARD_COEF_GOAL_RADIUS     δ_goal ~ U(2, 12)
    {0.0f, 20.0f, 0},      // REWARD_COEF_GOAL_SPEED      δ_goal-speed ~ U(0, 20)
    {0.0f, 3.0f, 0},       // REWARD_COEF_COLLISION       α_collision ~ U(0, 3)
    {0.0f, 3.0f, 0},       // REWARD_COEF_OFFROAD         α_boundary ~ U(0, 3)
    {1e-5f, 0.1f, 1},      // REWARD_COEF_COMFORT         α_comfort ~ logU(1e-5, 0.1)
    {2.5e-4f, 2.5e-2f, 1}, // REWARD_COEF_LANE_ALIGN      α_l-align ~ logU(2.5e-4, 2.5e-2)
    {0.0f, 1.0f, 0},       // REWARD_COEF_VEL_ALIGN       α_vel-align ~ U(0, 1)
    {2.5e-4f, 7.5e-3f, 1}, // REWARD_COEF_LANE_CENTER     α_l-center ~ logU(2.5e-4, 7.5e-3)
    {-0.5f, 0.5f, 0},      // REWARD_COEF_CENTER_BIAS     α_center-bias ~ U(-0.5, 0.5)
    {0.0f, 5e-3f, 0},      // REWARD_COEF_VELOCITY        α_velocity ~ U(0, 5e-3f)
    {2.5e-4f, 7.5e-3f, 1}, // REWARD_COEF_REVERSE         α_reverse ~ logU(2.5e-4, 7.5e-3)
    {0.0f, 1.0f, 0},       // REWARD_COEF_STOP_LINE       α_stop-line ~ U(0, 1)
    {0.0f, 5e-5f, 0},      // REWARD_COEF_TIMESTEP        α_timestep ~ U(0, 5e-5f)
    {0.0f, 1.0f, 0},       // REWARD_COEF_OVERSPEED       α_overspeed ~ U(0, 1)
    {0.8f, 1.25f, 0},      // REWARD_COEF_THROTTLE        C_throttle
    {0.8f, 1.25f, 0},      // REWARD_COEF_STEER           C_steer
    {0.666f, 1.5f, 0},     // REWARD_COEF_ACC             C_acc
};

// ========================================
// Utility Functions
// ========================================

static float compute_euclidean_distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

static float compute_point_to_segment_distance(float px, float py, float x0, float y0, float x1, float y1) {
    // Minimum (perpendicular or endpoint) distance from point (px, py) to segment (x0, y0)->(x1, y1).
    // t is the closest point's clamped projection param along the segment; degenerate (zero-length) segment uses t=0.
    float dx = x1 - x0;
    float dy = y1 - y0;
    float seg_len_sq = dx * dx + dy * dy;
    float t = 0.0f;
    if (seg_len_sq > 1e-6f) {
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len_sq;
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    return compute_euclidean_distance(px, py, x0 + t * dx, y0 + t * dy);
}

static int compare_depthpoint(const void *a, const void *b) {
    float diff = ((const DepthPoint *) a)->euclidean_dis - ((const DepthPoint *) b)->euclidean_dis;
    return (diff > 0.0f) - (diff < 0.0f);
}

static float clip(float value, float min, float max) {
    return value < min ? min : (value > max ? max : value);
}

static float normalize_heading(float heading) {
    heading = fmodf(heading, 2.0f * M_PI);
    if (heading > M_PI) {
        heading -= 2.0f * M_PI;
    } else if (heading < -M_PI) {
        heading += 2.0f * M_PI;
    }
    return heading;
}

static float compute_heading_diff(float heading1, float heading2) {
    return normalize_heading(heading1 - heading2);
}

static float sample_uniform(Rng *rng_state, float min_val, float max_val) {
    return min_val + rng_uniform_f32(rng_state) * (max_val - min_val);
}

static float sample_log_uniform(Rng *rng_state, float min_val, float max_val) {
    return expf(sample_uniform(rng_state, logf(min_val), logf(max_val)));
}

static float sample_mixed_uniform(Rng *rng_state, float a) {
    // Mixed uniform distribution X(a) = 0.5*U(1/a, 1) + 0.5*U(1, a)
    if (rng_uniform_f32(rng_state) < 0.5f) {
        return sample_uniform(rng_state, 1.0f / a, 1.0f);
    }
    return sample_uniform(rng_state, 1.0f, a);
}

static void begin_episode_rng(Drive *env) {
    // Standalone/test envs never pass through my_init; all-zero state marks an unseeded stream.
    uint64_t *s = env->seed_stream_rng.s;
    if ((s[0] | s[1] | s[2] | s[3]) == 0) {
        rng_seed(&env->seed_stream_rng, env->init_seed);
    }
    if (env->use_exact_episode_seed) {
        env->episode_seed = env->init_seed;
    } else {
        // 63-bit so the logged seed survives int64 round-trips (CSV, numpy, JSON)
        env->episode_seed = rng_next(&env->seed_stream_rng) >> 1;
    }
    rng_seed(&env->rng_state, env->episode_seed);
}

static inline void clear_agent_motion(Agent *agent) {
    agent->sim_vx = 0.0f;
    agent->sim_vy = 0.0f;
    agent->yaw_rate = 0.0f;
    agent->sim_speed = 0.0f;
    agent->sim_speed_signed = 0.0f;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
}

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
    agent->current_route_idx = 0;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
    agent->steering_angle = 0.0f;
    agent->distance_since_spawn = 0.0f;
    agent->seconds_stopped = 0.0f;
    agent->comfort_violation_last_window_idx = -1;
    agent->phantom_braking_counter = 0;
    agent->partner_blindness_counter = 0;
    agent->is_blind_partner = 0;
    agent->is_phantom_braker = 0;
}

static void invalidate_agent(Agent *agent) {
    agent->sim_x = INVALID_POSITION;
    agent->sim_y = INVALID_POSITION;
    agent->sim_z = 0.0f;
    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    clear_agent_motion(agent);
    agent->steering_angle = 0.0f;
    agent->sim_valid = 0;
}

static void copy_pose_to_prev(Agent *agent) {
    agent->prev_x = agent->sim_x;
    agent->prev_y = agent->sim_y;
    agent->prev_cos_heading = agent->cos_heading;
    agent->prev_sin_heading = agent->sin_heading;
}

static inline void update_agent_radius(Agent *agent) {
    agent->radius = 0.5f * sqrtf(agent->sim_length * agent->sim_length + agent->sim_width * agent->sim_width);
}

static inline void apply_infraction_behavior(Agent *agent, int behavior) {
    if (behavior == INFRACTION_BEHAVIOR_STOP && !agent->stopped) {
        agent->stopped = 1;
    } else if (behavior == INFRACTION_BEHAVIOR_REMOVE && !agent->removed) {
        agent->removed = 1;
    }
}

static inline void update_agent_speed(Agent *agent) {
    float speed = sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy);
    float v_dot_heading = agent->sim_vx * agent->cos_heading + agent->sim_vy * agent->sin_heading;
    agent->sim_speed = speed;
    agent->sim_speed_signed = copysignf(speed, v_dot_heading);
}

static inline float compute_log_yaw_rate(Agent *agent, int timestep, float dt) {
    int prev_t = timestep - 1;
    int next_t = timestep + 1;
    int has_prev = (prev_t >= 0) && (agent->log_valid[prev_t] == 1);
    int has_next = (next_t < agent->trajectory_size) && (agent->log_valid[next_t] == 1);

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

static inline void project_vector_to_local(
    float world_vec_x,
    float world_vec_y,
    float cos_heading,
    float sin_heading,
    float *local_x,
    float *local_y) {
    // Rotate a world-frame vector into a local frame with heading (cos_heading, sin_heading). Rotation only.
    *local_x = world_vec_x * cos_heading + world_vec_y * sin_heading;
    *local_y = -world_vec_x * sin_heading + world_vec_y * cos_heading;
}

static inline void project_point_to_local(
    float world_x,
    float world_y,
    float center_x,
    float center_y,
    float cos_heading,
    float sin_heading,
    float *local_x,
    float *local_y) {
    // Transform a world point into a local frame centered at (center_x,center_y) with heading
    // (cos_heading,sin_heading). Translate to the center, then rotate.
    project_vector_to_local(world_x - center_x, world_y - center_y, cos_heading, sin_heading, local_x, local_y);
}

static inline void project_point_to_ego_frame(
    const Agent *ego,
    float world_x,
    float world_y,
    float *rel_x,
    float *rel_y) {
    project_point_to_local(world_x, world_y, ego->sim_x, ego->sim_y, ego->cos_heading, ego->sin_heading, rel_x, rel_y);
}

static inline void project_vector_to_ego_frame(
    const Agent *ego,
    float world_vec_x,
    float world_vec_y,
    float *rel_x,
    float *rel_y) {
    project_vector_to_local(world_vec_x, world_vec_y, ego->cos_heading, ego->sin_heading, rel_x, rel_y);
}

#include "map_data.h"

// ========================================
// Road Utility Functions
// ========================================

static float compute_lane_progress(
    RoadMapElement *lane,
    float pos_x,
    float pos_y,
    float cos_heading,
    float sin_heading,
    bool align_heading,
    float *out_dist_sq) {
    // Arc-length progress (s, meters) of a world position along a lane polyline,
    // via closest-point projection onto its segments. Pass 0 only considers
    // segments pointing the same way as the heading (self-intersecting lanes);
    // pass 1 drops that filter as fallback if pass 0 matched nothing.
    float best_progress = 0.0f;
    float best_dist_sq = 1e30f;

    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < lane->segment_size - 1; i++) {
            float x0 = lane->x[i];
            float y0 = lane->y[i];
            float x1 = lane->x[i + 1];
            float y1 = lane->y[i + 1];
            float dx = x1 - x0;
            float dy = y1 - y0;
            float seg_len_sq = dx * dx + dy * dy;
            float seg_s = lane->cum_lengths[i + 1] - lane->cum_lengths[i];

            // Skip degenerate (zero-length) segments
            if (seg_len_sq <= 1e-6f || seg_s <= 1e-6f) {
                continue;
            }
            // Pass 0: reject segments facing away from the agent heading
            if (align_heading && pass == 0 && dx * cos_heading + dy * sin_heading < 0.0f) {
                continue;
            }

            // Project position onto segment, clamped to its endpoints
            float t = ((pos_x - x0) * dx + (pos_y - y0) * dy) / seg_len_sq;
            t = fmaxf(0.0f, fminf(1.0f, t));
            float proj_x = x0 + t * dx;
            float proj_y = y0 + t * dy;
            float dist_sq = (pos_x - proj_x) * (pos_x - proj_x) + (pos_y - proj_y) * (pos_y - proj_y);
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_progress = lane->cum_lengths[i] + t * seg_s;
            }
        }
        // Second pass only needed when heading filter rejected every segment
        if (!align_heading || best_dist_sq < 1e30f) {
            break;
        }
    }

    if (out_dist_sq != NULL) {
        *out_dist_sq = best_dist_sq;
    }
    return best_progress;
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

static void update_agent_z(Drive *env, Agent *agent) {
    static const int z_offsets[9][2] = {
        {-1, -1},
        {0, -1},
        {1, -1},
        {-1, 0},
        {0, 0},
        {1, 0},
        {-1, 1},
        {0, 1},
        {1, 1},
    };

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 9];
    int list_size
        = get_neighbors_entities(env, agent->sim_x, agent->sim_y, entity_list, MAX_ENTITIES_PER_CELL * 9, z_offsets, 9);
    if (list_size <= 0) {
        return;
    }

    DepthPoint road_neighbors[list_size];
    DepthPoint current_lane_neighbors[list_size];
    int valid_count = 0;
    int current_lane_count = 0;
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1) {
            continue;
        }

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
    if (neighbor_count <= 0) {
        return;
    }

    DepthPoint *neighbors = (current_lane_count > 0) ? current_lane_neighbors : road_neighbors;
    qsort(neighbors, neighbor_count, sizeof(DepthPoint), compare_depthpoint);
    int check_count = (neighbor_count < Z_NUM_PT_AVG) ? neighbor_count : Z_NUM_PT_AVG;
    float sum_z = 0.0f;
    for (int i = 0; i < check_count; i++) {
        sum_z += neighbors[i].z;
    }
    agent->sim_z = sum_z / check_count;
}

static int pick_random_exit_lane(
    Rng *rng_state,
    RoadMapElement *road_elements,
    int lane_idx,
    const int *route,
    int route_length) {
    RoadMapElement *lane = &road_elements[lane_idx];
    int exits[ROUTE_EXIT_MAX_CANDIDATES];
    int fresh_exits[ROUTE_EXIT_MAX_CANDIDATES]; // exits not already visited in `route`
    int num_exits = 0;
    int num_fresh_exits = 0;
    for (int e = 0; e < lane->num_exits; e++) {
        int ex = lane->exit_lanes[e];
        if (ex == -1) {
            continue;
        }
        exits[num_exits++] = ex;
        bool in_route = false;
        for (int r = 0; r < route_length; r++) {
            if (route[r] == ex) {
                in_route = true;
                break;
            }
        }
        if (!in_route) {
            fresh_exits[num_fresh_exits++] = ex;
        }
        if (num_exits >= ROUTE_EXIT_MAX_CANDIDATES) {
            break;
        }
    }

    if (num_exits == 0) {
        return -1; // dead-end lane: expected (caller ends the walk / route)
    }
    // Prefer an exit not yet in the route to avoid looping; fall back to any exit when all are visited.
    if (num_fresh_exits > 0) {
        return fresh_exits[rng_below(rng_state, num_fresh_exits)];
    }
    return exits[rng_below(rng_state, num_exits)];
}

// ========================================
// Route/Path/Goal Functions
// ========================================

static bool route_point_at_distance(
    Drive *env,
    const int *route,
    int route_length,
    int start_cursor_idx,
    float start_s_on_lane,
    float distance_meters,
    float *out_x,
    float *out_y,
    float *out_z,
    int *out_lane_idx,
    int *out_cursor_idx,
    float *out_s_on_lane) {
    // Walk `distance_meters` forward from cursor `start_cursor_idx` (at arc-length start_s_on_lane) and return
    // the point there. The cursor's meaning depends on mode:
    //   route mode (route != NULL): start_cursor_idx indexes route[]; seed lane = route[start_cursor_idx].
    //   free-roam  (route == NULL): start_cursor_idx is a lane index; the walk follows random forward exits.
    // out_lane_idx is always the actual landed lane in road_elements. out_cursor_idx is the landed cursor to
    // feed back as the next start_cursor_idx (route index in route mode, lane index in free-roam) so callers
    // chain goals without rescanning. Returns false when an explicit route is exhausted (cursor ==
    // route_length) or a free-roam walk dead-ends.

    // Seed the walk: explicit route reads its first lane from the route array; free-roam walks the lane graph
    // directly and is bounded by MAX_ROUTE_LENGTH hops.
    int current_lane_idx, max_lane_hops;
    if (route != NULL) {
        current_lane_idx = route[start_cursor_idx];
        max_lane_hops = route_length;
    } else {
        current_lane_idx = start_cursor_idx;
        max_lane_hops = MAX_ROUTE_LENGTH;
    }
    int cursor_idx = start_cursor_idx; // route index (route mode) or lane index (free-roam) of current lane
    float remaining_distance_meters = distance_meters;
    float s_on_lane = start_s_on_lane;

    for (int lane_hop = 0; lane_hop < max_lane_hops; lane_hop++) {
        RoadMapElement *lane = &env->road_elements[current_lane_idx];
        float lane_remaining_meters = lane->length - s_on_lane;

        // Target falls on this lane: interpolate the exact point between the two bounding polyline vertices.
        if (remaining_distance_meters <= lane_remaining_meters) {
            float target_s_meters = s_on_lane + remaining_distance_meters;
            int end_vertex_idx = lane->segment_size - 1;
            for (int vertex_idx = 1; vertex_idx < lane->segment_size; vertex_idx++) {
                if (lane->cum_lengths[vertex_idx] >= target_s_meters) {
                    end_vertex_idx = vertex_idx;
                    break;
                }
            }
            int start_vertex_idx = end_vertex_idx - 1;
            float start_vertex_s_meters = lane->cum_lengths[start_vertex_idx];
            float end_vertex_s_meters = lane->cum_lengths[end_vertex_idx];
            float segment_span_meters = end_vertex_s_meters - start_vertex_s_meters;
            float interp_frac = clip((target_s_meters - start_vertex_s_meters) / segment_span_meters, 0.0f, 1.0f);
            // Interpolate the 3D point along the lane segment.
            *out_x = lane->x[start_vertex_idx] + interp_frac * (lane->x[end_vertex_idx] - lane->x[start_vertex_idx]);
            *out_y = lane->y[start_vertex_idx] + interp_frac * (lane->y[end_vertex_idx] - lane->y[start_vertex_idx]);
            *out_z = lane->z[start_vertex_idx] + interp_frac * (lane->z[end_vertex_idx] - lane->z[start_vertex_idx]);
            *out_lane_idx = current_lane_idx;                                  // actual landed lane in road_elements
            *out_cursor_idx = (route != NULL) ? cursor_idx : current_lane_idx; // next start_cursor_idx
            *out_s_on_lane = target_s_meters;
            return true;
        }

        // Target is past this lane: consume the rest of it and advance to the next lane.
        remaining_distance_meters -= lane_remaining_meters;
        if (route != NULL) {
            cursor_idx++;
            if (cursor_idx >= route_length) {
                return false; // route exhausted: expected, caller extends/back-aligns goals
            }
            current_lane_idx = route[cursor_idx];
        } else {
            current_lane_idx = pick_random_exit_lane(&env->rng_state, env->road_elements, current_lane_idx, NULL, 0);
            if (current_lane_idx == -1) {
                return false; // dead-end: no outgoing lane to continue the free-roam walk
            }
        }
        s_on_lane = 0.0f; // new lane: start measuring arc-length from its beginning
    }
    printf("[ERROR] -> route_point_at_distance: walk did not reach target within %d lanes\n", max_lane_hops);
    return false;
}

static int chain_goals(
    Drive *env,
    const int *route,
    int route_length,
    int start_lane_idx,
    float start_s_on_lane,
    const float *goal_spacings_meters,
    int requested_goal_count,
    float *out_goal_x,
    float *out_goal_y,
    float *out_goal_z,
    int *out_goal_lane) {
    // Chains up to `requested_goal_count` goals forward from (start_lane_idx, start_s_on_lane), advancing
    // goal_spacings_meters[i] each step. route == NULL -> random map walk seeded from lane start_lane_idx.
    // Stops early on route exhaustion / dead-end; returns the number actually written into the out arrays.
    int placed_goal_count = 0;
    int carry_idx = start_lane_idx; // route index (route mode) or lane index (map mode) of the last landing
    float s_on_lane = start_s_on_lane;
    for (int goal_idx = 0; goal_idx < requested_goal_count; goal_idx++) {
        // carry_idx and s_on_lane are carried forward so each goal starts where the previous one landed.
        float goal_x, goal_y, goal_z;
        int goal_lane;
        int next_cursor_idx;
        if (!route_point_at_distance(
                env,
                route,
                route_length,
                carry_idx,
                s_on_lane,
                goal_spacings_meters[goal_idx],
                &goal_x,
                &goal_y,
                &goal_z,
                &goal_lane,
                &next_cursor_idx,
                &s_on_lane)) {
            break;
        }
        out_goal_x[placed_goal_count] = goal_x;
        out_goal_y[placed_goal_count] = goal_y;
        out_goal_z[placed_goal_count] = goal_z;
        out_goal_lane[placed_goal_count] = goal_lane;
        placed_goal_count++;
        // Cursor is route index (route mode) or landed lane (free-roam); carry it to start the next goal.
        carry_idx = next_cursor_idx;
    }
    return placed_goal_count;
}

static void commit_goals(
    Drive *env,
    Agent *agent,
    const float *goal_x,
    const float *goal_y,
    const float *goal_z,
    const int *goal_lane,
    int valid_goal_count,
    int start_slot_idx) {
    // Writes `valid_goal_count` placed goals into the agent's goal list starting at slot start_slot_idx
    // (front-aligned: start_slot_idx 0; back-aligned: start_slot_idx = num_goals - valid_goal_count).
    // Unused slots are zeroed with lane = -1. The obs window is [current_goal_idx, goal_count).
    for (int slot_idx = 0; slot_idx < env->num_goals; slot_idx++) {
        int source_idx = slot_idx - start_slot_idx; // index into the placed-goal arrays for this slot
        if (source_idx >= 0 && source_idx < valid_goal_count) {
            agent->list_goal_x[slot_idx] = goal_x[source_idx];
            agent->list_goal_y[slot_idx] = goal_y[source_idx];
            agent->list_goal_z[slot_idx] = goal_z[source_idx];
            agent->list_goal_lane[slot_idx] = goal_lane[source_idx];
        } else {
            agent->list_goal_x[slot_idx] = 0.0f;
            agent->list_goal_y[slot_idx] = 0.0f;
            agent->list_goal_z[slot_idx] = 0.0f;
            agent->list_goal_lane[slot_idx] = -1;
        }
    }
    agent->goal_count = start_slot_idx + valid_goal_count;
    agent->current_goal_idx = start_slot_idx;
    agent->current_goal_x = agent->list_goal_x[start_slot_idx];
    agent->current_goal_y = agent->list_goal_y[start_slot_idx];
    agent->current_goal_z = agent->list_goal_z[start_slot_idx];
}

static bool compute_new_route(Drive *env, Agent *agent, int current_lane_idx) {
    RoadMapElement *road_elements = env->road_elements;
    int candidate_route[MAX_ROUTE_LENGTH];
    float route_distance_meters = 0.0f;
    int route_length = 0;
    candidate_route[route_length++] = current_lane_idx;

    // First lane only contributes the arc-length still ahead of the agent, not its full length.
    RoadMapElement *current_lane = &road_elements[current_lane_idx];
    route_distance_meters += current_lane->length
        - compute_lane_progress(current_lane,
                                agent->sim_x,
                                agent->sim_y,
                                agent->cos_heading,
                                agent->sin_heading,
                                true,
                                NULL);

    // Random walk through the lane graph, appending exit lanes until we cover the target distance.
    while (route_distance_meters < ROUTE_TARGET_DISTANCE && route_length < MAX_ROUTE_LENGTH) {
        if (current_lane_idx == -1) {
            break;
        }
        current_lane = &road_elements[current_lane_idx];
        int chosen_exit_lane_idx
            = pick_random_exit_lane(&env->rng_state, road_elements, current_lane_idx, candidate_route, route_length);
        if (chosen_exit_lane_idx == -1) {
            break; // dead-end lane: stop here, the route is as long as the graph allows
        }
        candidate_route[route_length++] = chosen_exit_lane_idx;
        route_distance_meters += road_elements[chosen_exit_lane_idx].length;
        current_lane_idx = chosen_exit_lane_idx;
    }

    if (route_length == 0) {
        return false;
    }

    // Replace any previously owned route buffer with the freshly walked one.
    if (agent->route != NULL) {
        free(agent->route);
    }
    agent->route_length = route_length;
    agent->route = (int *) malloc(route_length * sizeof(int));
    for (int i = 0; i < route_length; i++) {
        agent->route[i] = candidate_route[i];
    }
    agent->current_route_idx = 0;

    return true;
}

static bool generate_new_goals_from_route(Drive *env, Agent *agent) {
    // Places num_goals goals along the agent's route by native lane arc-length.
    // Replay follows the loaded route to its end; gigaflow route source random-walks a fresh route
    // when the current one runs out.
    if (agent->route == NULL || agent->route_length <= 0) {
        invalidate_agent(agent);
        agent->removed = 1;
        return false;
    }

    // Localize the agent on its route by actual position: from the last known slot forward, pick the route
    // slot whose lane the agent is physically closest to, and take its arc-length on that lane as the base.
    // current_lane_idx (nearest lane by distance+heading) is frequently off-route, so matching by lane index
    // could leave the base on a stale earlier slot and place goals behind the agent; projecting the position
    // guarantees base_s_on_lane is the agent's true progress and every chained goal sits ahead of it.
    int base_route_idx = agent->current_route_idx;
    if (base_route_idx < 0 || base_route_idx >= agent->route_length) {
        base_route_idx = 0;
    }
    float base_s_on_lane = 0.0f;
    float best_dist_sq = 1e30f;
    for (int route_idx = base_route_idx; route_idx < agent->route_length; route_idx++) {
        float lane_dist_sq;
        float lane_progress = compute_lane_progress(
            &env->road_elements[agent->route[route_idx]],
            agent->sim_x,
            agent->sim_y,
            agent->cos_heading,
            agent->sin_heading,
            true,
            &lane_dist_sq);
        if (lane_dist_sq < best_dist_sq) {
            best_dist_sq = lane_dist_sq;
            base_route_idx = route_idx;
            base_s_on_lane = lane_progress;
        }
    }
    agent->current_route_idx = base_route_idx;

    // Remaining route distance from the agent's base to the route end (rest of the base lane + every later lane).
    float route_remaining_meters = env->road_elements[agent->route[base_route_idx]].length - base_s_on_lane;
    for (int route_idx = base_route_idx + 1; route_idx < agent->route_length; route_idx++) {
        route_remaining_meters += env->road_elements[agent->route[route_idx]].length;
    }

    // Replay: once the agent is essentially at the end of its logged route, retire it.
    if (env->simulation_mode == SIMULATION_MODE_REPLAY && route_remaining_meters <= env->goal_radius) {
        invalidate_agent(agent);
        agent->removed = 1;
        return false;
    }

    // Sample a spacing per goal, then walk the route placing goals at those forward distances.
    float goal_spacings_meters[MAX_GOALS];
    for (int goal_idx = 0; goal_idx < env->num_goals; goal_idx++) {
        goal_spacings_meters[goal_idx] = sample_uniform(&env->rng_state, env->min_goal_spacing, env->max_goal_spacing);
    }

    float goal_x[MAX_GOALS], goal_y[MAX_GOALS], goal_z[MAX_GOALS];
    int goal_lane[MAX_GOALS];
    int placed_goal_count = chain_goals(
        env,
        agent->route,
        agent->route_length,
        base_route_idx,
        base_s_on_lane,
        goal_spacings_meters,
        env->num_goals,
        goal_x,
        goal_y,
        goal_z,
        goal_lane);

    // Common case: the route was long enough to hold every goal, front-aligned at slot 0.
    if (placed_goal_count == env->num_goals) {
        commit_goals(env, agent, goal_x, goal_y, goal_z, goal_lane, placed_goal_count, 0);
        return true;
    }

    // Route exhausted before all goals fit.
    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        // Free-roam route source: random-walk a fresh route from the current lane, then retry once.
        int start_lane_idx = (agent->current_lane_idx != -1) ? agent->current_lane_idx : agent->route[base_route_idx];
        if (!compute_new_route(env, agent, start_lane_idx)) {
            invalidate_agent(agent);
            agent->removed = 1;
            printf(
                "[GIGAFLOW WARNING] -> Failed to compute new route for agent %d. Removing from simulation.\n",
                agent->id);
            return false;
        }
        agent->current_route_idx = 0;
        float new_base_s_on_lane = compute_lane_progress(
            &env->road_elements[agent->route[0]],
            agent->sim_x,
            agent->sim_y,
            agent->cos_heading,
            agent->sin_heading,
            true,
            NULL);
        placed_goal_count = chain_goals(
            env,
            agent->route,
            agent->route_length,
            0,
            new_base_s_on_lane,
            goal_spacings_meters,
            env->num_goals,
            goal_x,
            goal_y,
            goal_z,
            goal_lane);
        if (placed_goal_count < env->num_goals) {
            invalidate_agent(agent);
            agent->removed = 1;
            printf(
                "[GIGAFLOW ERROR] -> New route for agent %d is too short for goal generation. Removing from "
                "simulation.\n",
                agent->id);
            return false;
        }
        commit_goals(env, agent, goal_x, goal_y, goal_z, goal_lane, placed_goal_count, 0);
        return true;
    }

    // Replay: back-align the placed goals and pin the route endpoint (last vertex of the last lane) as the
    // final goal, so the agent's last target is exactly where its logged trajectory ends.
    int last_lane_idx = agent->route[agent->route_length - 1];
    RoadMapElement *last_lane = &env->road_elements[last_lane_idx];
    int last_vertex_idx = last_lane->segment_size - 1;
    goal_x[placed_goal_count] = last_lane->x[last_vertex_idx];
    goal_y[placed_goal_count] = last_lane->y[last_vertex_idx];
    goal_z[placed_goal_count] = last_lane->z[last_vertex_idx];
    goal_lane[placed_goal_count] = last_lane_idx;
    int valid_goal_count = placed_goal_count + 1;
    commit_goals(env, agent, goal_x, goal_y, goal_z, goal_lane, valid_goal_count, env->num_goals - valid_goal_count);
    return true;
}

static bool generate_new_goals_from_map(Drive *env, Agent *agent) {
    // Map goal source: seed from a uniform map lane, then chain 1..num_goals goals forward
    // via a route-free random lane-walk. No route is stored; the agent navigates by the GPS lane-distance feature,
    // not by following a path.

    // Pick a uniform drivable grid cell, then collect every drivable lane entity inside it.
    int drivable_cell_list_idx = rng_below(&env->rng_state, env->grid_map->num_drivable_grid_cell);
    int grid_cell_idx = env->grid_map->grid_index_drivable[drivable_cell_list_idx];
    GridMapEntity drivable_lane_candidates[MAX_ENTITIES_PER_CELL];
    int candidate_count = 0;
    for (int entity_idx = 0; entity_idx < env->grid_map->cell_entities_count[grid_cell_idx]; entity_idx++) {
        GridMapEntity entity = env->grid_map->cells[grid_cell_idx][entity_idx];
        if (is_drivable_road_lane(env->road_elements[entity.entity_idx].type)) {
            drivable_lane_candidates[candidate_count++] = entity;
        }
    }
    if (candidate_count == 0) {
        return false; // cell held no drivable lane: caller retries with another cell
    }

    // Seed pose: a uniformly chosen drivable lane and the polyline vertex that landed in this cell.
    GridMapEntity seed_entity = drivable_lane_candidates[rng_below(&env->rng_state, candidate_count)];
    int seed_lane_idx = seed_entity.entity_idx;
    RoadMapElement *seed_lane = &env->road_elements[seed_lane_idx];
    float seed_x = seed_lane->x[seed_entity.geometry_idx];
    float seed_y = seed_lane->y[seed_entity.geometry_idx];
    float seed_heading = seed_lane->headings[seed_entity.geometry_idx];
    float seed_s_on_lane
        = compute_lane_progress(seed_lane, seed_x, seed_y, cosf(seed_heading), sinf(seed_heading), true, NULL);

    // Random goal count in [1, num_goals], each at its own sampled forward spacing.
    int requested_goal_count = 1 + rng_below(&env->rng_state, env->num_goals);
    float goal_spacings_meters[MAX_GOALS];
    for (int goal_idx = 0; goal_idx < requested_goal_count; goal_idx++) {
        goal_spacings_meters[goal_idx] = sample_uniform(&env->rng_state, env->min_goal_spacing, env->max_goal_spacing);
    }
    float goal_x[MAX_GOALS], goal_y[MAX_GOALS], goal_z[MAX_GOALS];
    int goal_lane[MAX_GOALS];
    int placed_goal_count = chain_goals(
        env,
        NULL,
        0,
        seed_lane_idx,
        seed_s_on_lane,
        goal_spacings_meters,
        requested_goal_count,
        goal_x,
        goal_y,
        goal_z,
        goal_lane);
    if (placed_goal_count == 0) {
        return false;
    }
    commit_goals(env, agent, goal_x, goal_y, goal_z, goal_lane, placed_goal_count, 0);
    return true;
}

static int roll_goals(Drive *env, Agent *agent) {
    // Rolling target type: drop the reached goal, slide the window left, and append one new goal at the
    // frontier by walking forward from the previous last goal (along the route for route source, else
    // free-roam). Keeps current_goal_idx at 0 so the
    // obs always shows goal_count goals ahead. Returns 0 if the walk dead-ends (caller falls back to regen).
    // Seed the new frontier goal from the current last goal's lane and arc-length position.
    int last_goal_idx = agent->goal_count - 1;
    int seed_lane_idx = agent->list_goal_lane[last_goal_idx];
    if (seed_lane_idx < 0) {
        return 0;
    }
    float seed_s_on_lane = compute_lane_progress(
        &env->road_elements[seed_lane_idx],
        agent->list_goal_x[last_goal_idx],
        agent->list_goal_y[last_goal_idx],
        0.0f,
        0.0f,
        false,
        NULL);

    // Follow the agent's route when goals come from a route source, otherwise free-roam. Falls back to
    // free-roam if the seed lane isn't on the route; on route exhaustion the caller regenerates a route.
    const int *walk_route = NULL;
    int walk_route_length = 0;
    int walk_start_idx = seed_lane_idx;
    if (env->goal_source == GOAL_SOURCE_ROUTE && agent->route != NULL) {
        int reached_lane_idx = agent->list_goal_lane[agent->current_goal_idx - 1];
        for (int route_idx = agent->current_route_idx; route_idx < agent->route_length; route_idx++) {
            if (agent->route[route_idx] != reached_lane_idx) {
                continue;
            }
            agent->current_route_idx = route_idx;
            break;
        }
        for (int route_idx = agent->current_route_idx; route_idx < agent->route_length; route_idx++) {
            if (agent->route[route_idx] == seed_lane_idx) {
                walk_route = agent->route;
                walk_route_length = agent->route_length;
                walk_start_idx = route_idx;
                break;
            }
        }
    }

    // Walk one spacing forward to find the appended goal before touching the window.
    float spacing_meters = sample_uniform(&env->rng_state, env->min_goal_spacing, env->max_goal_spacing);
    float next_x, next_y, next_z, next_s_on_lane;
    int next_lane_idx, next_cursor_idx; // next_cursor_idx unused: single-step append, no chaining
    if (!route_point_at_distance(
            env,
            walk_route,
            walk_route_length,
            walk_start_idx,
            seed_s_on_lane,
            spacing_meters,
            &next_x,
            &next_y,
            &next_z,
            &next_lane_idx,
            &next_cursor_idx,
            &next_s_on_lane)) {
        return 0;
    }

    // Slide the window left by one (drop the reached goal) and append the new goal at the frontier.
    for (int goal_idx = 0; goal_idx < last_goal_idx; goal_idx++) {
        agent->list_goal_x[goal_idx] = agent->list_goal_x[goal_idx + 1];
        agent->list_goal_y[goal_idx] = agent->list_goal_y[goal_idx + 1];
        agent->list_goal_z[goal_idx] = agent->list_goal_z[goal_idx + 1];
        agent->list_goal_lane[goal_idx] = agent->list_goal_lane[goal_idx + 1];
    }
    agent->list_goal_x[last_goal_idx] = next_x;
    agent->list_goal_y[last_goal_idx] = next_y;
    agent->list_goal_z[last_goal_idx] = next_z;
    agent->list_goal_lane[last_goal_idx] = next_lane_idx;
    agent->current_goal_idx = 0;
    agent->current_goal_x = agent->list_goal_x[0];
    agent->current_goal_y = agent->list_goal_y[0];
    agent->current_goal_z = agent->list_goal_z[0];
    return 1;
}

// ========================================
// Metrics/Collision Functions
// ========================================

static float compute_displacement_error(Agent *agent, int timestep) {
    // Check if timestep is within valid range
    if (timestep < 0 || timestep >= agent->trajectory_size) {
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

static void compute_bounding_box_corners(
    float x,
    float y,
    float cos_h,
    float sin_h,
    float half_l,
    float half_w,
    float corners[4][2]) {
    static const float offsets[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
    for (int i = 0; i < 4; i++) {
        corners[i][0] = x + (offsets[i][0] * half_l * cos_h - offsets[i][1] * half_w * sin_h);
        corners[i][1] = y + (offsets[i][0] * half_l * sin_h + offsets[i][1] * half_w * cos_h);
    }
}

static bool check_segment_intersects_aabb(float p0[2], float p1[2], float half_l, float half_w) {
    // Liang-Barsky slab clip of segment p0->p1 against the origin-centered AABB
    // [-half_l,half_l] x [-half_w,half_w]. True if the segment crosses the box or starts inside it.
    // A degenerate (zero-length) segment reduces to a point-in-box test.
    float dx = p1[0] - p0[0];
    float dy = p1[1] - p0[1];
    float t0 = 0.0f;
    float t1 = 1.0f;
    float p[4] = {-dx, dx, -dy, dy};
    float q[4] = {p0[0] + half_l, half_l - p0[0], p0[1] + half_w, half_w - p0[1]};
    for (int i = 0; i < 4; i++) {
        if (p[i] == 0.0f) {
            if (q[i] < 0.0f) {
                return false; // parallel to this slab and outside it
            }
            continue;
        }
        float r = q[i] / p[i];
        if (p[i] < 0.0f) {
            if (r > t1) {
                return false;
            }
            if (r > t0) {
                t0 = r;
            }
        } else {
            if (r < t0) {
                return false;
            }
            if (r < t1) {
                t1 = r;
            }
        }
    }
    return true;
}

static bool check_segment_crosses_moving_box(float ax, float ay, float bx, float by, Agent *agent) {
    // Segment AB is static; the box sweeps from prev to cur pose. Projecting AB into the box-local frame
    // at both poses makes the box a fixed origin-centered AABB and AB a quad with corners a_prev, b_prev,
    // a_cur, b_cur. Any of the quad's 4 edges hitting the AABB means the box crossed AB this step.
    float a_prev[2], b_prev[2], a_cur[2], b_cur[2];
    float half_length = agent->sim_length / 2.0f, half_width = agent->sim_width / 2.0f;
    project_point_to_local(
        ax,
        ay,
        agent->prev_x,
        agent->prev_y,
        agent->prev_cos_heading,
        agent->prev_sin_heading,
        &a_prev[0],
        &a_prev[1]);
    project_point_to_local(
        bx,
        by,
        agent->prev_x,
        agent->prev_y,
        agent->prev_cos_heading,
        agent->prev_sin_heading,
        &b_prev[0],
        &b_prev[1]);
    project_point_to_local(
        ax,
        ay,
        agent->sim_x,
        agent->sim_y,
        agent->cos_heading,
        agent->sin_heading,
        &a_cur[0],
        &a_cur[1]);
    project_point_to_local(
        bx,
        by,
        agent->sim_x,
        agent->sim_y,
        agent->cos_heading,
        agent->sin_heading,
        &b_cur[0],
        &b_cur[1]);
    if (check_segment_intersects_aabb(a_prev, b_prev, half_length, half_width)
        || check_segment_intersects_aabb(a_cur, b_cur, half_length, half_width)
        || check_segment_intersects_aabb(a_prev, a_cur, half_length, half_width)
        || check_segment_intersects_aabb(b_prev, b_cur, half_length, half_width)) {
        return true;
    }

    // All edges can miss while the swept region still covers the box center: consistent cross-product
    // sign means the origin is inside the quad. The quad chords the prev/cur segments, but the true
    // endpoint paths are arcs, so at large per-step yaw the chords under-cover and a swept center reads
    // as outside (missed); exact for normal driving's small yaw. Degenerate prev == cur is a line:
    // opposite signs => false, and the boundary tests above already caught any overlap.
    float swept_quad[4][2] = {
        {a_prev[0], a_prev[1]},
        {b_prev[0], b_prev[1]},
        {b_cur[0], b_cur[1]},
        {a_cur[0], a_cur[1]},
    };
    bool has_positive_cross = false;
    bool has_negative_cross = false;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        float edge_x = swept_quad[j][0] - swept_quad[i][0];
        float edge_y = swept_quad[j][1] - swept_quad[i][1];
        float cross = edge_x * -swept_quad[i][1] - edge_y * -swept_quad[i][0];
        has_positive_cross = has_positive_cross || cross > 0.0f;
        has_negative_cross = has_negative_cross || cross < 0.0f;
    }
    return has_positive_cross != has_negative_cross;
}

static bool check_agent_on_stop_line(Drive *env, Agent *agent, bool include_yellow_violation) {
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];

        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (tc->num_controlled_lanes == 0) {
            continue;
        }

        int controls_lane = 0;
        for (int j = 0; j < tc->num_controlled_lanes; j++) {
            if (tc->controlled_lanes[j] == agent->current_lane_idx) {
                controls_lane = 1;
                break;
            }
        }
        if (!controls_lane) {
            continue;
        }
        if (env->timestep >= tc->state_size) {
            continue;
        }
        int light_state = tc->states[env->timestep];
        if (light_state != TRAFFIC_CONTROL_STATE_RED
            && !(include_yellow_violation && light_state == TRAFFIC_CONTROL_STATE_YELLOW)) {
            continue;
        }

        // Pre-filter: distance to stop line midpoint
        float mid_x = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
        float mid_y = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;
        float dx = agent->sim_x - mid_x;
        float dy = agent->sim_y - mid_y;
        if (dx * dx + dy * dy > STOP_LINE_DIST_SQ) {
            continue;
        }

        // Heading check: agent must be heading towards the stop line
        float heading_diff = compute_heading_diff(agent->sim_heading, tc->heading);
        if (fabsf(heading_diff) > STOP_LINE_HEADING_THRESHOLD) {
            continue;
        }

        // Stop line segment vector (endpoint 0 -> endpoint 1)
        float sl_dx = tc->stop_line[3] - tc->stop_line[0];
        float sl_dy = tc->stop_line[4] - tc->stop_line[1];
        // Lengthen the segment by STOP_LINE_EXTENSION_FACTOR, growing equally
        // from both endpoints so agents crossing near the edges are still caught.
        float ext = (STOP_LINE_EXTENSION_FACTOR - 1.0f) * 0.5f;
        float ext_p1[2] = {tc->stop_line[0] - ext * sl_dx, tc->stop_line[1] - ext * sl_dy};
        float ext_p2[2] = {tc->stop_line[3] + ext * sl_dx, tc->stop_line[4] + ext * sl_dy};

        if (check_segment_crosses_moving_box(ext_p1[0], ext_p1[1], ext_p2[0], ext_p2[1], agent)) {
            return true;
        }
    }
    return false;
}

static bool check_stop_line_crossing_event(Drive *env, Agent *agent) {
    // Violation = the rear bumper crossing the stop line while red, as a one-step
    // event (CARLA leaderboard convention: flagged once the car fully enters on red).
    float rear_offset = -0.5f * agent->sim_length;
    float rear_x = agent->sim_x + rear_offset * agent->cos_heading;
    float rear_y = agent->sim_y + rear_offset * agent->sin_heading;
    float prev_rear_x = agent->prev_x + rear_offset * agent->prev_cos_heading;
    float prev_rear_y = agent->prev_y + rear_offset * agent->prev_sin_heading;

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || tc->num_controlled_lanes == 0) {
            continue;
        }
        if (env->timestep >= tc->state_size) {
            continue;
        }
        if (tc->states[env->timestep] != TRAFFIC_CONTROL_STATE_RED) {
            continue;
        }
        float mid_x = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
        float mid_y = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;
        float dx = agent->sim_x - mid_x;
        float dy = agent->sim_y - mid_y;
        if (dx * dx + dy * dy > STOP_LINE_DIST_SQ) {
            continue;
        }
        if (fabsf(compute_heading_diff(agent->sim_heading, tc->heading)) > STOP_LINE_HEADING_THRESHOLD) {
            continue;
        }

        float line_dx = tc->stop_line[3] - tc->stop_line[0];
        float line_dy = tc->stop_line[4] - tc->stop_line[1];
        float line_len = sqrtf(line_dx * line_dx + line_dy * line_dy);
        if (line_len <= 0.0f) {
            continue;
        }
        float line_ux = line_dx / line_len, line_uy = line_dy / line_len;
        // normal oriented along travel direction, so s < 0 is before the line
        float normal_x = -line_uy, normal_y = line_ux;
        if (normal_x * cosf(tc->heading) + normal_y * sinf(tc->heading) < 0.0f) {
            normal_x = -normal_x;
            normal_y = -normal_y;
        }
        float s_prev = (prev_rear_x - mid_x) * normal_x + (prev_rear_y - mid_y) * normal_y;
        float s_cur = (rear_x - mid_x) * normal_x + (rear_y - mid_y) * normal_y;
        if (!(s_prev < 0.0f && s_cur >= 0.0f)) {
            continue;
        }
        float crossing_frac = s_prev / (s_prev - s_cur);
        float cross_x = prev_rear_x + crossing_frac * (rear_x - prev_rear_x);
        float cross_y = prev_rear_y + crossing_frac * (rear_y - prev_rear_y);
        float lateral = (cross_x - mid_x) * line_ux + (cross_y - mid_y) * line_uy;
        if (fabsf(lateral) <= 0.5f * STOP_LINE_EXTENSION_FACTOR * line_len) {
            return true;
        }
    }
    return false;
}

static bool lane_has_traffic_light(Drive *env, int lane_idx) {
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        for (int j = 0; j < tc->num_controlled_lanes; j++) {
            if (tc->controlled_lanes[j] == lane_idx) {
                return true;
            }
        }
    }
    return false;
}

static bool check_lane_change_red_light(Drive *env, Agent *agent) {
    if (agent->previous_lane_idx == agent->current_lane_idx) {
        return false;
    }
    if (agent->previous_lane_idx == -1 || agent->current_lane_idx == -1) {
        return false;
    }
    if (lane_has_traffic_light(env, agent->previous_lane_idx)) {
        return false;
    }

    float agent_x = agent->sim_x;
    float agent_y = agent->sim_y;

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];

        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (tc->num_controlled_lanes == 0) {
            continue;
        }
        if (env->timestep >= tc->state_size) {
            continue;
        }
        if (tc->states[env->timestep] != TRAFFIC_CONTROL_STATE_RED) {
            continue;
        }
        if (env->timestep < 1 || tc->states[env->timestep - 1] != TRAFFIC_CONTROL_STATE_RED) {
            continue;
        }

        for (int j = 0; j < tc->num_controlled_lanes; j++) {
            if (tc->controlled_lanes[j] != agent->current_lane_idx) {
                continue;
            }

            float mid_x = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
            float mid_y = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;
            float dx = agent_x - mid_x;
            float dy = agent_y - mid_y;
            if (dx * dx + dy * dy > STOP_LINE_DIST_SQ) {
                continue;
            }
            float line_dx = tc->stop_line[3] - tc->stop_line[0];
            float line_dy = tc->stop_line[4] - tc->stop_line[1];
            float normal_x = -line_dy, normal_y = line_dx;
            if (normal_x * cosf(tc->heading) + normal_y * sinf(tc->heading) < 0.0f) {
                normal_x = -normal_x;
                normal_y = -normal_y;
            }
            if ((agent_x - mid_x) * normal_x + (agent_y - mid_y) * normal_y < 0.0f) {
                continue;
            }

            return true;
        }
    }
    return false;
}

static bool check_red_light_violation(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    if (agent->current_lane_idx == -1) {
        return false;
    }

    if (check_stop_line_crossing_event(env, agent)) {
        return true;
    }

    if (check_lane_change_red_light(env, agent)) {
        return true;
    }

    return false;
}

static bool check_obb_collision(Agent *car1, Agent *car2) {
    // OBB collision via SAT (Separating Axis Theorem).
    // Projects both boxes onto 4 axes (2 per car) and checks for overlap on all axes.
    // No epsilon tolerance: exact boundary contact may flicker across steps.

    // Early z-axis rejection
    float car1_top = car1->sim_z + car1->sim_height;
    float car2_top = car2->sim_z + car2->sim_height;
    if (car1_top < car2->sim_z || car2_top < car1->sim_z) {
        return false;
    }

    float car1_corners[4][2];
    compute_bounding_box_corners(
        car1->sim_x,
        car1->sim_y,
        car1->cos_heading,
        car1->sin_heading,
        car1->sim_length / 2.0f,
        car1->sim_width / 2.0f,
        car1_corners);
    float car2_corners[4][2];
    compute_bounding_box_corners(
        car2->sim_x,
        car2->sim_y,
        car2->cos_heading,
        car2->sin_heading,
        car2->sim_length / 2.0f,
        car2->sim_width / 2.0f,
        car2_corners);

    float axes[4][2]
        = {{car1->cos_heading, car1->sin_heading},
           {-car1->sin_heading, car1->cos_heading},
           {car2->cos_heading, car2->sin_heading},
           {-car2->sin_heading, car2->cos_heading}};

    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;
        for (int j = 0; j < 4; j++) {
            float proj1 = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj1);
            max1 = fmaxf(max1, proj1);
            float proj2 = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj2);
            max2 = fmaxf(max2, proj2);
        }
        if (max1 < min2 || min1 > max2) {
            return false;
        }
    }
    return true;
}

static bool check_moving_obb_collision(Agent *a, Agent *b, float a_disp, float b_disp) {
    // Swept-OBB collision for the prev->cur step (tunnelling-safe).

    // Early z-axis rejection
    float a_top = a->sim_z + a->sim_height;
    float b_top = b->sim_z + b->sim_height;
    if (a_top < b->sim_z || b_top < a->sim_z) {
        return false;
    }

    // Current pose overlap (incl. cross/plus with no corner inside).
    if (check_obb_collision(a, b)) {
        return true;
    }

    // Both nearly static: a sub-threshold step can't open a tunnelling gap the sweep would catch.
    if (a_disp < COLLISION_SKIP_DISP_M && b_disp < COLLISION_SKIP_DISP_M) {
        return false;
    }

    // Sweep each of other's corners along its prev->cur segment, transformed into ego's prev/cur
    // local frame, and test vs ego's origin-centered AABB. Run both orderings: a corner of one box
    // can sweep through the other even when the reverse ordering misses.
    for (int d = 0; d < 2; d++) {
        Agent *ego = d ? a : b;
        Agent *other = d ? b : a;
        float oc_prev[4][2], oc_cur[4][2];
        compute_bounding_box_corners(
            other->prev_x,
            other->prev_y,
            other->prev_cos_heading,
            other->prev_sin_heading,
            other->sim_length / 2.0f,
            other->sim_width / 2.0f,
            oc_prev);
        compute_bounding_box_corners(
            other->sim_x,
            other->sim_y,
            other->cos_heading,
            other->sin_heading,
            other->sim_length / 2.0f,
            other->sim_width / 2.0f,
            oc_cur);
        float ego_half_l = ego->sim_length / 2.0f, ego_half_w = ego->sim_width / 2.0f;
        for (int k = 0; k < 4; k++) {
            float q_prev[2], q_cur[2];
            project_point_to_local(
                oc_prev[k][0],
                oc_prev[k][1],
                ego->prev_x,
                ego->prev_y,
                ego->prev_cos_heading,
                ego->prev_sin_heading,
                &q_prev[0],
                &q_prev[1]);
            project_point_to_local(
                oc_cur[k][0],
                oc_cur[k][1],
                ego->sim_x,
                ego->sim_y,
                ego->cos_heading,
                ego->sin_heading,
                &q_cur[0],
                &q_cur[1]);
            if (check_segment_intersects_aabb(q_prev, q_cur, ego_half_l, ego_half_w)) {
                return true;
            }
        }
    }
    return false;
}

static int collision_check(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->sim_x == INVALID_POSITION || agent->removed) {
        return -1;
    }

    int car_collided_with_index = -1;
    float ego_disp = compute_euclidean_distance(agent->sim_x, agent->sim_y, agent->prev_x, agent->prev_y);

    // Linear over all actors; pair radius quick-check prunes before OBB SAT.
    for (int i = 0; i < env->num_agents; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == agent_idx) {
            continue;
        }

        Agent *other_agent = &env->agents[index];
        if (other_agent->sim_x == INVALID_POSITION || other_agent->removed || other_agent->sim_valid != 1) {
            continue;
        }

        float other_disp = compute_euclidean_distance(
            other_agent->sim_x,
            other_agent->sim_y,
            other_agent->prev_x,
            other_agent->prev_y);
        float threshold = agent->radius + other_agent->radius + COLLISION_PAIR_MARGIN_M + ego_disp + other_disp;
        float ddx = other_agent->sim_x - agent->sim_x;
        float ddy = other_agent->sim_y - agent->sim_y;
        if (ddx * ddx + ddy * ddy > threshold * threshold) {
            continue;
        }
        if (check_moving_obb_collision(agent, other_agent, ego_disp, other_disp)) {
            car_collided_with_index = index;
            break;
        }
    }

    return car_collided_with_index;
}

static bool is_at_fault_collision(Drive *env, int agent_idx, int other_idx) {
    Agent *agent = &env->agents[agent_idx];
    Agent *other = &env->agents[other_idx];

    if (agent->sim_speed <= AGENT_STOPPED_SPEED_THRESHOLD) {
        return false;
    }

    if (other->sim_speed <= AGENT_STOPPED_SPEED_THRESHOLD) {
        return true;
    }

    float rear_x = agent->sim_x - 0.5f * agent->sim_length * agent->cos_heading;
    float rear_y = agent->sim_y - 0.5f * agent->sim_length * agent->sin_heading;
    float dx = other->sim_x - rear_x;
    float dy = other->sim_y - rear_y;
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist >= 1e-6f) {
        float rear_cos = (agent->cos_heading * dx + agent->sin_heading * dy) / dist;
        if (rear_cos < BEHIND_COS_THRESHOLD) {
            return false;
        }
    }

    float agent_corners[4][2];
    compute_bounding_box_corners(
        agent->sim_x,
        agent->sim_y,
        agent->cos_heading,
        agent->sin_heading,
        agent->sim_length / 2.0f,
        agent->sim_width / 2.0f,
        agent_corners);

    // Front bumper = segment between front-left (corner 0) and front-right (corner 1).
    // Transform into other's local frame and test vs its origin-centered AABB: this catches both
    // a bumper corner inside other and the bumper edge crossing other's boundary.
    float front_left_local[2], front_right_local[2];
    project_point_to_local(
        agent_corners[0][0],
        agent_corners[0][1],
        other->sim_x,
        other->sim_y,
        other->cos_heading,
        other->sin_heading,
        &front_left_local[0],
        &front_left_local[1]);
    project_point_to_local(
        agent_corners[1][0],
        agent_corners[1][1],
        other->sim_x,
        other->sim_y,
        other->cos_heading,
        other->sin_heading,
        &front_right_local[0],
        &front_right_local[1]);
    bool front_bumper_intersects = check_segment_intersects_aabb(
        front_left_local,
        front_right_local,
        other->sim_length / 2.0f,
        other->sim_width / 2.0f);

    if (front_bumper_intersects) {
        return true;
    }

    if (agent->current_lane_idx == -1) {
        return true;
    }

    float edge_dist = fabsf(agent->metrics_array[LANE_DIST_IDX]) + 0.5f * agent->sim_width;
    return edge_dist > MULTI_LANE_THRESHOLD;
}

static inline void compute_pairwise_ttc(Agent *ego, Agent *other) {
    if (other->sim_x == INVALID_POSITION) {
        return;
    }

    float ego_x = ego->sim_x;
    float ego_y = ego->sim_y;
    float other_x = other->sim_x;
    float other_y = other->sim_y;

    float center_dx = other_x - ego_x;
    float center_dy = other_y - ego_y;
    float center_distance = sqrtf(center_dx * center_dx + center_dy * center_dy);
    if (center_distance <= 1e-6f || center_distance >= DEFAULT_DTC) {
        return;
    }

    float ego_heading_x = ego->cos_heading;
    float ego_heading_y = ego->sin_heading;
    float other_heading_x = other->cos_heading;
    float other_heading_y = other->sin_heading;

    float forward_cos = (center_dx * ego_heading_x + center_dy * ego_heading_y) / center_distance;
    float heading_align_cos = ego_heading_x * other_heading_x + ego_heading_y * other_heading_y;
    if (forward_cos < DTC_FRONT_CONE_COS_THRESHOLD || heading_align_cos <= DTC_OPPOSITE_HEADING_COS_THRESHOLD) {
        return;
    }

    float ego_corners[4][2];
    float other_corners[4][2];
    compute_bounding_box_corners(
        ego->sim_x,
        ego->sim_y,
        ego->cos_heading,
        ego->sin_heading,
        ego->sim_length / 2.0f,
        ego->sim_width / 2.0f,
        ego_corners);
    compute_bounding_box_corners(
        other->sim_x,
        other->sim_y,
        other->cos_heading,
        other->sin_heading,
        other->sim_length / 2.0f,
        other->sim_width / 2.0f,
        other_corners);

    float min_dtc_sq = DEFAULT_DTC * DEFAULT_DTC;
    for (int ego_corner = 0; ego_corner < 4; ego_corner++) {
        for (int other_corner = 0; other_corner < 4; other_corner++) {
            float dx = ego_corners[ego_corner][0] - other_corners[other_corner][0];
            float dy = ego_corners[ego_corner][1] - other_corners[other_corner][1];
            min_dtc_sq = fminf(min_dtc_sq, dx * dx + dy * dy);
        }
    }

    float distance_to_collision = sqrtf(min_dtc_sq);
    if (distance_to_collision < ego->metrics_array[DISTANCE_TO_COLLISION_IDX]) {
        ego->metrics_array[DISTANCE_TO_COLLISION_IDX] = distance_to_collision;
    }

    float rel_vx = other->sim_vx - ego->sim_vx;
    float rel_vy = other->sim_vy - ego->sim_vy;
    float rel_dot_v = center_dx * rel_vx + center_dy * rel_vy;
    float closing_speed = fmaxf(0.0f, -rel_dot_v / center_distance);
    if (distance_to_collision <= 1e-4f) {
        ego->metrics_array[TTC_IDX] = 0.0f;
        return;
    }
    if (closing_speed <= 1e-4f) {
        return;
    }

    float ttc = fminf(DEFAULT_TTC, distance_to_collision / closing_speed);
    if (!isfinite(ttc)) {
        return;
    }

    if (ttc < ego->metrics_array[TTC_IDX]) {
        ego->metrics_array[TTC_IDX] = ttc;
    }
}

static void compute_agent_ttc(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    agent->metrics_array[TTC_IDX] = DEFAULT_TTC;
    agent->metrics_array[DISTANCE_TO_COLLISION_IDX] = DEFAULT_DTC;

    if (agent->sim_x == INVALID_POSITION) {
        return;
    }

    for (int j = 0; j < env->num_agents; j++) {
        int other_idx;
        if (j < env->active_agent_count) {
            other_idx = env->active_agent_indices[j];
        } else {
            other_idx = env->static_agent_indices[j - env->active_agent_count];
        }
        if (other_idx == agent_idx) {
            continue;
        }
        compute_pairwise_ttc(agent, &env->agents[other_idx]);
    }
}

// Puffer score computation
// Uses hybrid weighted average: multiplier weights (binary gates) + average weights (continuous)
static float calculate_duration_scaled_violation_score(float violation_timestep_count, float duration_steps, float dt) {
    float duration_s = fmaxf(duration_steps * dt, dt);
    float windows = ceilf(duration_s / METRIC_SCORE_WINDOW_SECONDS);
    float safe_windows = fmaxf(windows, 1.0f);
    float score = 1.0f - (violation_timestep_count / safe_windows);

    return fmaxf(0.0f, fminf(1.0f, score));
}

static float calculate_puffer_score(Log *agent_log, float duration_steps, float dt) {
    if (!agent_log) {
        return 0.0f;
    }

    float safe_duration_steps = fmaxf(duration_steps, 1.0f);
    float episode_duration_s = fmaxf(safe_duration_steps * dt, dt);

    float no_at_fault = (agent_log->at_fault_collision_rate > 0) ? 0.0f : 1.0f;
    float no_offroad = (agent_log->offroad_rate > 0) ? 0.0f : 1.0f;
    float no_red_light = (agent_log->red_light_violation_rate > 0) ? 0.0f : 1.0f;
    float making_progress = (agent_log->progress_ratio > 0.2f) ? 1.0f : 0.0f;

    // Driving direction: 1.0 if <=2m, 0.5 if 2-6m, 0 if >6m wrong-way distance
    float wrong_dist = agent_log->wrong_way_distance;
    float direction_compliance = (wrong_dist <= 2.0f) ? 1.0f : (wrong_dist <= 6.0f) ? 0.5f : 0.0f;

    float multiplier = no_at_fault * no_offroad * no_red_light * making_progress * direction_compliance;

    // TTC within bound (>0.95s): weight 5
    float ttc_score = agent_log->ttc_within_bound_rate; // Already 0-1

    // Progress ratio (capped at 1): weight 5
    float progress_score = fminf(agent_log->progress_ratio, 1.0f);

    // Speed compliance (nuPlan formula): max(0, 1 - sum(violation * dt) / T): weight 4
    float speed_threshold = fmaxf(episode_duration_s, 1e-3f);
    float speed_score = fmaxf(0.0f, 1.0f - agent_log->speed_violation_sum / speed_threshold);

    // Comfort (duration-scaled 10s windows): weight 2
    float comfort_score = agent_log->comfort_score; // 0-1

    // Multi-lane (weight 3): tiered score based on accumulated time
    float multi_lane_score = agent_log->multi_lane_score;

    // Weighted average
    float weighted_sum
        = 5 * ttc_score + 5 * progress_score + 4 * speed_score + 3 * multi_lane_score + 2 * comfort_score;
    float total_weight = 5 + 5 + 4 + 3 + 2; // = 19
    float weighted_avg = weighted_sum / total_weight;

    // Store agent-only display fields
    agent_log->no_at_fault = no_at_fault;
    agent_log->no_offroad = no_offroad;
    agent_log->no_red_light = no_red_light;
    agent_log->making_progress = making_progress;
    agent_log->driving_direction_score = direction_compliance;
    agent_log->ttc_puffer_rate = ttc_score;
    agent_log->speed_limit_compliance = speed_score;
    agent_log->multiplier = multiplier;
    agent_log->weighted_average = weighted_avg;
    agent_log->puffer_score = multiplier * weighted_avg;

    return agent_log->puffer_score;
}

static void add_log(Drive *env) {
    int safe_timestep = (env->timestep > 0) ? env->timestep : 1;
    Log episode_log = {0};
    for (int i = 0; i < env->active_agent_count; i++) {
        Agent *agent = &env->agents[env->active_agent_indices[i]];
        float episode_duration_s = env->logs[i].episode_length * env->dt;
        float reference_progress_distance = PUFFER_PROGRESS_REFERENCE_SPEED * episode_duration_s;
        reference_progress_distance = fmaxf(reference_progress_distance, 1.0f);
        env->logs[i].progress_ratio = agent->distance_since_spawn / reference_progress_distance;

        int offroad = env->logs[i].offroad_rate;
        episode_log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        episode_log.collision_rate += collided;
        int red_light_violations = env->logs[i].red_light_violation_rate;
        episode_log.red_light_violation_rate += red_light_violations;
        int total_infractions = (offroad || collided || red_light_violations) ? 1 : 0;
        float avg_speed_per_agent = env->logs[i].avg_speed_per_agent;
        episode_log.avg_speed_per_agent += avg_speed_per_agent / safe_timestep;
        int num_goals_reached = env->logs[i].num_goals_reached;
        episode_log.num_goals_reached += num_goals_reached;
        // Score: 1 per agent that reached its full goal set without being removed/stopped.
        if (num_goals_reached >= env->num_goals && !agent->removed && !agent->stopped) {
            episode_log.score += 1.0f;
        }
        if (!offroad && !collided && !red_light_violations && num_goals_reached < 1) {
            episode_log.dnf_rate += 1.0f;
        }
        episode_log.total_distance_travelled += agent->distance_since_spawn;
        if (total_infractions > 0) {
            episode_log.total_infractions += 1.0f;
        }
        float displacement_error = env->logs[i].avg_displacement_error;
        episode_log.avg_displacement_error += displacement_error;
        episode_log.episode_length += env->logs[i].episode_length;
        episode_log.episode_return += env->logs[i].episode_return;
        // Per-component reward sums (mirrors compute_rewards' env->rewards[i]+= sites).
        episode_log.reward_collision += env->logs[i].reward_collision;
        episode_log.reward_offroad += env->logs[i].reward_offroad;
        episode_log.reward_red_light += env->logs[i].reward_red_light;
        episode_log.reward_goal += env->logs[i].reward_goal;
        episode_log.reward_lane_align += env->logs[i].reward_lane_align;
        episode_log.reward_lane_center += env->logs[i].reward_lane_center;
        episode_log.reward_comfort += env->logs[i].reward_comfort;
        episode_log.reward_velocity += env->logs[i].reward_velocity;
        episode_log.reward_timestep += env->logs[i].reward_timestep;
        episode_log.reward_reverse += env->logs[i].reward_reverse;
        episode_log.reward_overspeed += env->logs[i].reward_overspeed;
        episode_log.reward_ade += env->logs[i].reward_ade;
        // Comfort and velocity metrics (normalized per timestep)
        episode_log.comfort_violation_count += env->logs[i].comfort_violation_count / safe_timestep;
        episode_log.velocity_progress_sum += env->logs[i].velocity_progress_sum / safe_timestep;
        // Lane metrics (normalized per timestep for average per episode)
        episode_log.lane_center_rate += env->logs[i].lane_center_rate / safe_timestep;
        episode_log.lane_heading_aligned_rate += env->logs[i].lane_heading_aligned_rate / safe_timestep;
        if (env->compute_eval_metrics) {
            env->logs[i].progress_ratio = agent->distance_since_spawn / reference_progress_distance;
            env->logs[i].comfort_score = calculate_duration_scaled_violation_score(
                env->logs[i].comfort_violation_window_count,
                env->logs[i].episode_length,
                env->dt);
            calculate_puffer_score(&env->logs[i], env->logs[i].episode_length, env->dt);
            episode_log.at_fault_collision_rate += env->logs[i].at_fault_collision_rate;
            episode_log.ttc_within_bound_rate += env->logs[i].ttc_within_bound_rate;
            episode_log.wrong_way_distance += env->logs[i].wrong_way_distance;
            episode_log.speed_violation_sum += env->logs[i].speed_violation_sum;
            episode_log.progress_ratio += env->logs[i].progress_ratio;
            episode_log.comfort_score += env->logs[i].comfort_score;
            episode_log.ttc_violations += env->logs[i].ttc_violations;
            episode_log.ttc_samples += env->logs[i].ttc_samples;
            episode_log.multi_lane_time += env->logs[i].multi_lane_time;
            episode_log.multi_lane_score += env->logs[i].multi_lane_score;

            float wrong_dist = env->logs[i].wrong_way_distance;
            float direction_score = (wrong_dist <= 2.0f) ? 1.0f : (wrong_dist <= 6.0f) ? 0.5f : 0.0f;
            episode_log.driving_direction_score += direction_score;

            float safe_duration_s = safe_timestep * env->dt;
            float speed_compliance
                = fmaxf(0.0f, 1.0f - env->logs[i].speed_violation_sum / fmaxf(safe_duration_s, 1e-3f));
            episode_log.speed_limit_compliance += speed_compliance;

            float making_progress = (env->logs[i].progress_ratio > 0.2f) ? 1.0f : 0.0f;
            episode_log.making_progress_rate += making_progress;
            episode_log.puffer_score += env->logs[i].puffer_score;
        }
        episode_log.n += 1;
    }
    // Log composition counts per agent so vec_log averaging recovers the per-env value
    episode_log.expert_static_car_count += env->expert_static_agent_count;
    episode_log.static_car_count += env->static_agent_count;

    // Fold this episode into the cumulative log that vec_log averages and resets.
    int num_log_fields = sizeof(Log) / sizeof(float);
    for (int field_idx = 0; field_idx < num_log_fields; field_idx++) {
        ((float *) &env->log)[field_idx] += ((float *) &episode_log)[field_idx];
    }
    env->log_episode_seed = env->episode_seed;
}

// ========================================
// Initialization Functions
// ========================================

static inline void sample_erratic_flags(Drive *env, Agent *agent) {
    agent->is_blind_partner = (env->partner_blindness_prob > 0.0f
                               && sample_uniform(&env->rng_state, 0.0f, 1.0f) < env->partner_blindness_prob)
        ? 1
        : 0;
    agent->is_phantom_braker
        = (env->phantom_braking_prob > 0.0f && sample_uniform(&env->rng_state, 0.0f, 1.0f) < env->phantom_braking_prob)
        ? 1
        : 0;
    agent->phantom_braking_counter = 0;
    agent->partner_blindness_counter = 0;
}

static void generate_reward_coefs(Drive *env, Agent *agent) {
    if (env->reward_randomization) {
        static const int random_coefs[] = {
            REWARD_COEF_GOAL_RADIUS,
            REWARD_COEF_GOAL_SPEED,
            REWARD_COEF_COLLISION,
            REWARD_COEF_OFFROAD,
            REWARD_COEF_COMFORT,
            REWARD_COEF_LANE_ALIGN,
            REWARD_COEF_LANE_CENTER,
            REWARD_COEF_STOP_LINE,
            REWARD_COEF_CENTER_BIAS,
            REWARD_COEF_VEL_ALIGN,
            REWARD_COEF_OVERSPEED,
            REWARD_COEF_REVERSE,
        };
        const RewardBound *bounds = env->reward_log_sampling ? REWARD_BOUNDS_LOG : REWARD_BOUNDS;
        for (int i = 0; i < (int) (sizeof(random_coefs) / sizeof(random_coefs[0])); i++) {
            int c = random_coefs[i];
            agent->reward_coefs[c] = bounds[c].log_scale
                ? sample_log_uniform(&env->rng_state, bounds[c].min_val, bounds[c].max_val)
                : sample_uniform(&env->rng_state, bounds[c].min_val, bounds[c].max_val);
        }
        agent->reward_coefs[REWARD_COEF_VELOCITY] = 2.5e-3f;
        agent->reward_coefs[REWARD_COEF_TIMESTEP] = 2.5e-5f;
        agent->reward_coefs[REWARD_COEF_THROTTLE] = sample_mixed_uniform(&env->rng_state, 1.25f);
        agent->reward_coefs[REWARD_COEF_STEER] = sample_mixed_uniform(&env->rng_state, 1.25f);
        agent->reward_coefs[REWARD_COEF_ACC] = sample_mixed_uniform(&env->rng_state, 1.5f);
    } else {
        agent->reward_coefs[REWARD_COEF_GOAL_RADIUS] = env->goal_radius;
        agent->reward_coefs[REWARD_COEF_GOAL_SPEED] = env->goal_speed;
        agent->reward_coefs[REWARD_COEF_COLLISION] = env->reward_collision;
        agent->reward_coefs[REWARD_COEF_OFFROAD] = env->reward_offroad;
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
        agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
        agent->reward_coefs[REWARD_COEF_STEER] = 1.0f;
        agent->reward_coefs[REWARD_COEF_ACC] = 1.0f;
    }
}

static void generate_traffic_light_states(Drive *env) {
    int steps = env->scenario_length;
    float dt = env->dt;

    // 20% chance: disable ALL lights for this episode
    int disable_all = (!env->eval_mode) && (sample_uniform(&env->rng_state, 0.0f, 1.0f) < TL_EPISODE_DISABLE_PROB);

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || tc->states == NULL || tc->state_size <= 0) {
            continue;
        }

        int fill_steps = steps;
        if (tc->state_size < fill_steps) {
            fill_steps = tc->state_size;
        }

        if (disable_all) {
            for (int t = 0; t < fill_steps; t++) {
                tc->states[t] = TRAFFIC_CONTROL_STATE_OFF;
            }
            continue;
        }

        if (!env->eval_mode) {
            // Individual removal
            if (sample_uniform(&env->rng_state, 0.0f, 1.0f) < TL_INDIVIDUAL_REMOVE_PROB) {
                for (int t = 0; t < fill_steps; t++) {
                    tc->states[t] = TRAFFIC_CONTROL_STATE_OFF;
                }
                continue;
            }
            // Always green
            if (sample_uniform(&env->rng_state, 0.0f, 1.0f) < TL_ALWAYS_GREEN_PROB) {
                for (int t = 0; t < fill_steps; t++) {
                    tc->states[t] = TRAFFIC_CONTROL_STATE_GREEN;
                }
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
            dur_green = sample_uniform(&env->rng_state, 0.1 * TL_DEFAULT_GREEN_DURATION, TL_DEFAULT_GREEN_DURATION);
            dur_yellow = sample_uniform(
                &env->rng_state,
                0.5f * TL_DEFAULT_YELLOW_DURATION,
                0.75f * TL_DEFAULT_YELLOW_DURATION);
            dur_red = sample_uniform(&env->rng_state, 0.15f * TL_DEFAULT_RED_DURATION, 5.0f * TL_DEFAULT_RED_DURATION);
        }

        int steps_green = (int) (dur_green / dt);
        if (steps_green < 1) {
            steps_green = 1;
        }
        int steps_yellow = (int) (dur_yellow / dt);
        if (steps_yellow < 1) {
            steps_yellow = 1;
        }
        int steps_red = (int) (dur_red / dt);
        if (steps_red < 1) {
            steps_red = 1;
        }
        int cycle_length = steps_green + steps_yellow + steps_red;

        // Random phase offset
        int offset = rng_below(&env->rng_state, cycle_length);

        // Fill states: GREEN -> YELLOW -> RED -> repeat
        for (int t = 0; t < fill_steps; t++) {
            int phase = (t + offset) % cycle_length;
            if (phase < steps_green) {
                tc->states[t] = TRAFFIC_CONTROL_STATE_GREEN;
            } else if (phase < steps_green + steps_yellow) {
                tc->states[t] = TRAFFIC_CONTROL_STATE_YELLOW;
            } else {
                tc->states[t] = TRAFFIC_CONTROL_STATE_RED;
            }
        }
    }
}

static bool check_spawn_collision(Drive *env, int num_existing_agents, Agent *tmp_agent) {
    float min_safe_dist_sq = (tmp_agent->sim_length + 5.0f) * (tmp_agent->sim_length + 5.0f);

    for (int i = 0; i < num_existing_agents; i++) {
        Agent *other = &env->agents[i];

        if (other->sim_x == INVALID_POSITION || other->sim_valid != 1) {
            continue;
        }

        float dx = other->sim_x - tmp_agent->sim_x;
        float dy = other->sim_y - tmp_agent->sim_y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq > min_safe_dist_sq) {
            continue;
        }
        if (check_obb_collision(tmp_agent, other)) {
            return true;
        }
    }

    return false;
}

static bool check_spawn_offroad(Drive *env, Agent *tmp_agent) {
    // Increase length and width slightly for spawn offroad check
    Agent scaled = *tmp_agent;
    scaled.sim_length *= 1.1f;
    scaled.sim_width *= 1.1f;

    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int list_size = get_neighbors_entities(
        env,
        tmp_agent->sim_x,
        tmp_agent->sim_y,
        entity_list,
        ROAD_QUERY_ENTITY_COUNT,
        ROAD_OFFSETS,
        25);

    for (int i = 0; i < list_size; i++) {
        int entity_idx = entity_list[i].entity_idx;
        int geometry_idx = entity_list[i].geometry_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];

        if (is_road_edge(element->type)) {
            float abs_dz = fabsf(element->z[geometry_idx] - tmp_agent->sim_z);
            if (abs_dz > Z_BUFFER) {
                continue;
            }
            if (check_segment_crosses_moving_box(
                    element->x[geometry_idx],
                    element->y[geometry_idx],
                    element->x[geometry_idx + 1],
                    element->y[geometry_idx + 1],
                    &scaled)) {
                return true;
            }
        }
    }
    return false;
}

static bool spawn_agent(Drive *env, int agent_idx, int num_agents) {
    Agent *agent = &env->agents[agent_idx];

    // Free existing route on reset
    if (agent->route != NULL) {
        free(agent->route);
        agent->route = NULL;
    }

    agent->id = num_agents;

    // Initialize identity fields
    agent->type = VEHICLE;
    agent->active_agent = 1;
    agent->mark_as_expert = 0;

    float spawn_length, spawn_width;
    if (env->eval_mode) {
        // Eval: uniform random car-sized boxes
        spawn_length = sample_uniform(&env->rng_state, 2.0f, 5.5f);
        spawn_width = sample_uniform(&env->rng_state, 1.5f, 2.5f);
    } else {
        // Training: random size
        spawn_length = sample_uniform(&env->rng_state, 0.8f, 7.0f);
        spawn_width = sample_uniform(&env->rng_state, 0.8f, 2.7f);
    }
    if (spawn_width > spawn_length) {
        spawn_width = spawn_length;
    }
    float spawn_height = 1.5f;
    float spawn_wheelbase = 0.6f * spawn_length;

    // Set spawn position on start lane
    float spawn_x, spawn_y, spawn_z, spawn_heading;
    RoadMapElement *start_lane;
    int start_lane_idx;
    bool is_agent_spawned = false;

    // Sampling rejection loop
    // TARGET: Only one attempt should be sufficient in most cases
    const int MAX_SPAWN_ATTEMPTS = 30;
    for (int attempt = 0; attempt < MAX_SPAWN_ATTEMPTS; attempt++) {
        int chosen_lane_idx = -1;

        int list_idx = rng_below(&env->rng_state, env->grid_map->num_drivable_grid_cell);
        int grid_idx = env->grid_map->grid_index_drivable[list_idx];

        GridMapEntity cell_candidates[MAX_ENTITIES_PER_CELL];
        int candidate_count = 0;

        for (int i = 0; i < env->grid_map->cell_entities_count[grid_idx]; i++) {
            GridMapEntity entity = env->grid_map->cells[grid_idx][i];

            if (is_drivable_road_lane(env->road_elements[entity.entity_idx].type)) {
                cell_candidates[candidate_count++] = entity;
            }
        }

        if (candidate_count == 0) {
            continue;
        }

        GridMapEntity chosen_entity = cell_candidates[rng_below(&env->rng_state, candidate_count)];
        chosen_lane_idx = chosen_entity.entity_idx;

        start_lane_idx = chosen_lane_idx;
        start_lane = &env->road_elements[start_lane_idx];

        spawn_x = start_lane->x[chosen_entity.geometry_idx];
        spawn_y = start_lane->y[chosen_entity.geometry_idx];
        spawn_z = start_lane->z[chosen_entity.geometry_idx];
        spawn_heading = start_lane->headings[chosen_entity.geometry_idx];

        Agent tmp_agent = {0};
        tmp_agent.sim_x = spawn_x;
        tmp_agent.sim_y = spawn_y;
        tmp_agent.sim_z = spawn_z;
        tmp_agent.sim_heading = spawn_heading;
        tmp_agent.cos_heading = cosf(spawn_heading);
        tmp_agent.sin_heading = sinf(spawn_heading);
        // Spawn pose is static: prev == curr makes the moving-box checks degenerate to static.
        tmp_agent.prev_x = spawn_x;
        tmp_agent.prev_y = spawn_y;
        tmp_agent.prev_cos_heading = tmp_agent.cos_heading;
        tmp_agent.prev_sin_heading = tmp_agent.sin_heading;
        tmp_agent.yaw_rate = 0.0f;
        tmp_agent.sim_length = spawn_length;
        tmp_agent.sim_width = spawn_width;
        tmp_agent.sim_height = spawn_height;
        update_agent_radius(&tmp_agent);
        tmp_agent.current_lane_idx = start_lane_idx;

        if (check_spawn_collision(env, num_agents, &tmp_agent)) {
            continue;
        }

        if (check_spawn_offroad(env, &tmp_agent)) {
            continue;
        }

        if (check_agent_on_stop_line(env, &tmp_agent, true)) {
            continue;
        }

        is_agent_spawned = true;
        break;
    }

    if (!is_agent_spawned) {
        printf("[GIGAFLOW WARNING] -> Failed to find a collision-free spawn position for agent %d\n", agent->id);
        return is_agent_spawned;
    }

    // Update simulation state
    agent->sim_x = spawn_x;
    agent->sim_y = spawn_y;
    agent->sim_z = spawn_z;
    agent->sim_heading = spawn_heading;
    agent->cos_heading = cosf(spawn_heading);
    agent->sin_heading = sinf(spawn_heading);
    copy_pose_to_prev(agent);
    agent->sim_length = spawn_length;
    agent->sim_width = spawn_width;
    agent->sim_height = spawn_height;
    update_agent_radius(agent);
    agent->sim_valid = 1;
    agent->wheelbase = spawn_wheelbase;
    agent->current_lane_idx = start_lane_idx;
    float spawn_speed = clip(env->spawn_initial_speed, 0.0f, MAX_SPEED);
    agent->sim_vx = spawn_speed * agent->cos_heading;
    agent->sim_vy = spawn_speed * agent->sin_heading;
    agent->yaw_rate = 0.0f;
    update_agent_speed(agent);

    if (env->goal_source == GOAL_SOURCE_MAP) {
        if (!generate_new_goals_from_map(env, agent)) {
            printf("[GIGAFLOW WARNING] -> Failed to generate map goals for agent %d\n", agent_idx);
            return false;
        }
        return true;
    }

    if (!compute_new_route(env, agent, start_lane_idx)) {
        printf("[GIGAFLOW WARNING] -> Failed to compute a new route for agent %d\n", agent_idx);
        return false; // Failed to compute new goal
    }

    // Compute initial goal
    if (!generate_new_goals_from_route(env, agent)) {
        return false;
    }

    return true;
}

static void set_start_position(Drive *env) {
    for (int i = 0; i < env->num_total_agents; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Agent *agent = &env->agents[i];

        // Initialize simulation trajectory from logged trajectory at init_step
        if (env->simulation_mode == SIMULATION_MODE_REPLAY) {
            // Clamp init_step to ensure we don't go out of bounds
            int step = env->init_step;
            if (step >= agent->trajectory_size) {
                step = agent->trajectory_size - 1;
            }
            if (step < 0) {
                step = 0;
            }

            // For agents invalid at init_step, set INVALID_POSITION
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
            update_agent_radius(agent);
            agent->wheelbase = 0.6f * agent->sim_length;
            copy_pose_to_prev(agent);

            if (agent->type == UNKNOWN) {
                continue;
            }

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

    if (env->control_mode == CONTROL_MODE_SDC_ONLY) {
        return agent_idx == EGO_IDX && agent->route_length != 0;
    }

    if (env->control_mode == CONTROL_MODE_WOSAC) {
        for (int j = 0; j < env->num_tracks_to_predict; j++) {
            if (env->tracks_to_predict[j] == agent_idx) {
                return true;
            }
        }
        return false;
    }

    // Standard mode: check type, distance to goal, and expert status
    bool type_is_controllable = false;
    if (env->control_mode == CONTROL_MODE_VEHICLES) {
        type_is_controllable = (agent->type == VEHICLE);
    } else { // CONTROL_MODE_AGENTS mode
        type_is_controllable = is_controllable_agent(agent->type);
    }

    if (!type_is_controllable || agent->mark_as_expert) {
        return false;
    }

    // In REPLAY mode without route data, control agents spawning far enough from their goal
    if (env->goal_source == GOAL_SOURCE_GT && agent->route_length == 0) {
        float dx = agent->gt_goal_x - agent->log_trajectory_x[env->init_step];
        float dy = agent->gt_goal_y - agent->log_trajectory_y[env->init_step];
        float dz = agent->gt_goal_z - agent->log_trajectory_z[env->init_step];
        return sqrtf(dx * dx + dy * dy + dz * dz) > env->goal_radius;
    }

    // Control if the agent has a route to follow
    return agent->route_length != 0;
}

static int resolve_agent_controller(Drive *env, int agent_idx, int is_active, int replay_by_default) {
    if (replay_by_default) {
        return CONTROLLER_REPLAY;
    }

    Agent *agent = &env->agents[agent_idx];
    int requested_controller = CONTROLLER_STATIC;
    if (agent_idx == EGO_IDX) {
        requested_controller = env->sdc_controller;
    } else if (agent->type == VEHICLE) {
        requested_controller = env->non_sdc_controller;
    } else {
        requested_controller = env->non_vehicle_controller;
    }

    if (requested_controller == CONTROLLER_POLICY && !is_active) {
        return CONTROLLER_STATIC;
    }

    return requested_controller;
}

void set_active_agents(Drive *env) {
    // Initialize
    env->active_agent_count = 0;        // Policy-controlled agents
    env->static_agent_count = 0;        // Non-moving background agents
    env->expert_static_agent_count = 0; // Expert replay agents (non-controlled)
    env->num_agents = 0;                // Total agents created

    // In GIGAFLOW mode, spawn agents dynamically on the map
    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        int num_agents_to_create = env->num_controllable_agents;

        // Initialize agents for GIGAFLOW mode
        env->agents = (Agent *) calloc(num_agents_to_create, sizeof(Agent));
        int *active_agent_indices = (int *) malloc(num_agents_to_create * sizeof(int));

        int successfully_created = 0;
        for (int i = 0; i < num_agents_to_create; i++) {
            if (spawn_agent(env, i, i)) {
                active_agent_indices[successfully_created] = i;
                successfully_created++;
            } else {
                // Failed spawn: ensure agent is properly invalidated
                invalidate_agent(&env->agents[i]);
                env->agents[i].removed = 1;
            }
        }

        env->num_total_agents = num_agents_to_create;
        env->active_agent_indices = (int *) malloc(successfully_created * sizeof(int));
        env->static_agent_indices = NULL;
        env->expert_static_agent_indices = NULL;

        for (int i = 0; i < successfully_created; i++) {
            env->active_agent_indices[i] = active_agent_indices[i];
            env->agents[active_agent_indices[i]].controller
                = resolve_agent_controller(env, active_agent_indices[i], 1, 0);
        }
        free(active_agent_indices);

        env->active_agent_count = successfully_created;
        env->num_agents = successfully_created;

        return;
    }

    // In REPLAY mode, determine which agents to control
    bool is_log_replay = (env->control_mode == CONTROL_MODE_SDC_ONLY);
    // Eval and log-replay keep the whole scene; training stays capped at num_max_agents.
    int max_agents = (env->eval_mode || is_log_replay) ? env->num_total_agents : env->num_max_agents;

    int *active_agent_indices = (int *) malloc(max_agents * sizeof(int));
    int *static_agent_indices = (int *) malloc(max_agents * sizeof(int));
    int *expert_static_agent_indices = (int *) malloc(max_agents * sizeof(int));

    // Iterate through entities to find agents to create and/or control
    for (int i = 0; i < env->num_total_agents && env->num_agents < max_agents; i++) {
        Agent *agent = &env->agents[i];

        // Skip if not valid at initialization
        if (agent->log_valid[env->init_step] != 1 && !is_log_replay) {
            continue;
        }

        // Determine if entity should be created
        bool should_create = false;
        if (is_log_replay) {
            should_create = true; // Log-replay: all valid agents
        } else if (env->init_mode == INIT_MODE_CREATE_ALL_VALID) {
            should_create = true; // All valid entities
        } else if (env->control_mode == CONTROL_MODE_VEHICLES) {
            should_create = (agent->type == VEHICLE);
        } else { // Control all agents
            should_create = (is_controllable_agent(agent->type));
        }

        if (!should_create) {
            continue;
        }

        env->num_agents++;

        // Determine if this agent should be policy-controlled
        bool is_controlled = should_control_agent(env, i);

        if (is_controlled) {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->agents[i].active_agent = 1;
            env->agents[i].controller = resolve_agent_controller(env, i, 1, 0);
        } else if (is_log_replay || env->init_mode != INIT_MODE_CREATE_ONLY_CONTROLLED) {
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->agents[i].active_agent = 0;
            int replay_by_default
                = is_log_replay || env->agents[i].mark_as_expert == 1 || env->active_agent_count == env->num_max_agents;
            env->agents[i].controller = resolve_agent_controller(env, i, 0, replay_by_default);
            if (env->agents[i].controller == CONTROLLER_REPLAY) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->agents[i].mark_as_expert = 1;
            }
        }
    }

    // Set up initial active agents
    env->active_agent_indices = (int *) malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int *) malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int *) malloc(env->expert_static_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; i++) {
        env->active_agent_indices[i] = active_agent_indices[i];
    }
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
        printf(
            "ERROR Between my_shared and init : Mismatch in active agent count: %d vs %d\n",
            env->active_agent_count,
            env->num_controllable_agents);
    }

    return;
}

void move_expert(Drive *env, int agent_idx) {
    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        printf("[GIGAFLOW ERROR] -> move_expert() called in GIGAFLOW mode\n");
        return;
    }
    bool is_log_replay = (env->control_mode == CONTROL_MODE_SDC_ONLY);

    Agent *agent = &env->agents[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->trajectory_size || agent->log_valid[t] == 0) {
        invalidate_agent(agent);
        return;
    }
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
        update_agent_radius(agent);
        agent->wheelbase = 0.6f * agent->sim_length;
    }
    agent->yaw_rate = compute_log_yaw_rate(agent, t, env->dt);
    agent->sim_vx = agent->log_velocity_x[t];
    agent->sim_vy = agent->log_velocity_y[t];
    update_agent_speed(agent);
    agent->sim_valid = agent->log_valid[t];

    if (t == 0 || agent->log_valid[t - 1] == 0) {
        copy_pose_to_prev(agent);
    } else {
        agent->prev_x = agent->log_trajectory_x[t - 1];
        agent->prev_y = agent->log_trajectory_y[t - 1];
        agent->prev_cos_heading = cosf(agent->log_heading[t - 1]);
        agent->prev_sin_heading = sinf(agent->log_heading[t - 1]);
    }
}

void remove_bad_trajectories(Drive *env) {
    if (env->control_mode == CONTROL_MODE_WOSAC) {
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
            move_expert(env, agent_idx);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->agents[expert_idx].sim_x == INVALID_POSITION) {
                continue;
            }
            move_expert(env, expert_idx);
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
        if (collided_with_indices[i] == -1) {
            continue;
        }
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_agent_idx = env->static_agent_indices[j];
            if (static_agent_idx != collided_with_indices[i]) {
                continue;
            }
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
    struct SharedMapData *shared = env->use_map_cache ? map_cache_lookup(env) : NULL;
    if (shared != NULL) {
        // Cache hit: load only the per-env data (agents, traffic-control elements),
        // then discard the freshly-loaded geometry and borrow the shared copy.
        if (load_map_binary(env->map_name, env) != 0) {
            fprintf(stderr, "[ERROR] -> Failed to load map binary: %s\n", env->map_name);
            return;
        }
        for (int i = 0; i < env->num_road_elements; i++) {
            free_road_element(&env->road_elements[i]);
        }
        free(env->road_elements);
        free_lane_graph(&env->lane_graph);
        env->road_elements = shared->road_elements;
        env->num_road_elements = shared->num_road_elements;
        env->grid_map = shared->grid_map;
        env->neighbor_offsets = shared->neighbor_offsets;
        env->lane_graph = shared->lane_graph;
        env->shared_map = shared;
        shared->ref_count++;
    } else {
        // Cache miss (or caching off): load and build the geometry as usual.
        if (load_map_binary(env->map_name, env) != 0) {
            fprintf(stderr, "[ERROR] -> Failed to load map binary: %s\n", env->map_name);
            return;
        }
        if (init_grid_map(env) != 0) {
            fprintf(stderr, "[ERROR] -> Failed to build grid map for map: %s\n", env->map_name);
            return;
        }
        int vision_half_range = (int) ceilf(
            fmaxf(fmaxf(env->obs_range_road_front_m, env->obs_range_road_behind_m), env->obs_range_road_side_m)
            / GRID_CELL_SIZE);
        env->grid_map->vision_range = 2 * vision_half_range + 1;
        init_neighbor_offsets(env);
        if (env->use_map_cache) {
            // Transfer the just-built geometry into a shared, ref-counted entry that
            // this env borrows (ref_count starts at 1).
            struct SharedMapData *entry = (struct SharedMapData *) calloc(1, sizeof(struct SharedMapData));
            entry->map_name = strdup(env->map_name);
            entry->road_elements = env->road_elements;
            entry->num_road_elements = env->num_road_elements;
            entry->grid_map = env->grid_map;
            entry->neighbor_offsets = env->neighbor_offsets;
            entry->lane_graph = env->lane_graph;
            entry->obs_lane_stride = env->obs_lane_stride;
            entry->obs_boundary_stride = env->obs_boundary_stride;
            entry->ref_count = 1;
            entry->owner_pid = getpid();
            map_cache_insert(entry);
            env->shared_map = entry;
        }
    }
    if (env->use_neighbor_cache && env->grid_map->neighbor_cache_entities == NULL) {
        cache_neighbor_offsets(env);
    }
    if (!env->use_neighbor_cache) {
        env->obs_neighbor_scratch = (GridMapEntity *) malloc(env->grid_map->total_entities * sizeof(GridMapEntity));
    }
    env->road_dropout_enabled = (env->obs_slots_lane_kept < env->obs_slots_lane_n)
        || (env->obs_slots_boundary_kept < env->obs_slots_boundary_n);
    env->logs_capacity = 0;
    begin_episode_rng(env);
    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        int steps = env->scenario_length;
        if (steps > 0) {
            for (int i = 0; i < env->num_traffic_elements; i++) {
                TrafficControlElement *traffic = &env->traffic_elements[i];
                if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
                    continue;
                }
                if (traffic->states && traffic->state_size != steps) {
                    free(traffic->states);
                    traffic->states = NULL;
                }
                if (traffic->states == NULL) {
                    traffic->states = (int *) malloc(steps * sizeof(int));
                    if (traffic->states == NULL) {
                        traffic->state_size = 0;
                        continue;
                    }
                }
                traffic->state_size = steps;
            }
        }
        generate_traffic_light_states(env);
    }
    set_active_agents(env);
    env->logs_capacity = env->active_agent_count;
    if (env->simulation_mode == SIMULATION_MODE_REPLAY) {
        remove_bad_trajectories(env);
    }
    set_start_position(env);
    env->logs = (Log *) calloc(env->active_agent_count, sizeof(Log));

    if (env->goal_source == GOAL_SOURCE_GT) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            Agent *agent = &env->agents[agent_idx];
            // For replay mode, always place goals along the logged
            // trajectory. Route-based goal generation can produce goals that
            // diverge from the actual path the SDC should follow.
            {
                int start = env->init_step > 0 ? env->init_step : 0;
                int remaining = agent->trajectory_size - 1 - start;
                if (remaining < 1) {
                    remaining = 1;
                }
                int num_wp = env->num_goals;
                for (int g = 0; g < num_wp; g++) {
                    int t = start + (g + 1) * remaining / num_wp;
                    if (t >= agent->trajectory_size) {
                        t = agent->trajectory_size - 1;
                    }
                    agent->list_goal_x[g] = agent->log_trajectory_x[t];
                    agent->list_goal_y[g] = agent->log_trajectory_y[t];
                    agent->list_goal_z[g] = agent->log_trajectory_z[t];
                    agent->list_goal_lane[g] = -1; // logged goals have no lane idx (no GPS lane-distance)
                }
                agent->goal_count = num_wp;
                agent->current_goal_idx = 0;
                agent->current_goal_x = agent->list_goal_x[0];
                agent->current_goal_y = agent->list_goal_y[0];
                agent->current_goal_z = agent->list_goal_z[0];
            }
        }
    } else if (env->goal_source == GOAL_SOURCE_MAP) {
        for (int i = 0; i < env->active_agent_count; i++) {
            Agent *agent = &env->agents[env->active_agent_indices[i]];
            generate_new_goals_from_map(env, agent);
        }
    } else if (env->goal_source == GOAL_SOURCE_ROUTE) {
        for (int i = 0; i < env->active_agent_count; i++) {
            Agent *agent = &env->agents[env->active_agent_indices[i]];
            generate_new_goals_from_route(env, agent);
        }
    }
}

void c_close(Drive *env) {
    for (int i = 0; i < env->num_total_agents; i++) {
        free_agent(&env->agents[i]);
    }
    for (int i = 0; i < env->num_traffic_elements; i++) {
        free_traffic_element(&env->traffic_elements[i]);
    }
    free(env->agents);
    free(env->traffic_elements);
    free(env->active_agent_indices);
    free(env->logs);
    if (env->shared_map != NULL) {
        // Geometry is borrowed from the cache. Release our reference; free the
        // entry only on the last reference, and only in the process that built it.
        env->shared_map->ref_count--;
        if (env->shared_map->ref_count <= 0 && env->shared_map->owner_pid == getpid()) {
            free_shared_map_data(env->shared_map);
        }
        env->shared_map = NULL;
    } else {
        // Geometry is owned by this env: free it.
        for (int i = 0; i < env->num_road_elements; i++) {
            free_road_element(&env->road_elements[i]);
        }
        free(env->road_elements);
        free(env->neighbor_offsets);
        free_grid_map(env->grid_map);
        free_lane_graph(&env->lane_graph);
    }

    free(env->obs_neighbor_scratch);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->objects_of_interest);
    free(env->tracks_to_predict);
    free(env->map_name);
    free(env->ini_file);
}

static int compute_observation_size(Drive *env) {
    return EGO_FEATURES + PARTNER_FEATURES * env->obs_slots_partners_n + LANE_FEATURES * env->obs_slots_lane_kept
        + BOUNDARY_FEATURES * env->obs_slots_boundary_kept
        + TRAFFIC_CONTROL_FEATURES * env->obs_slots_traffic_controls_n + OBS_VALID_COUNT_FEATURES
        + env->reward_conditioning * NUM_REWARD_COEFS + env->num_goals * GOAL_FEATURES;
}

void allocate(Drive *env) {
    init(env);
    int max_obs = compute_observation_size(env);
    env->observations = (float *) calloc(env->active_agent_count * max_obs, sizeof(float));
    env->actions = (float *) calloc(env->active_agent_count * 2, sizeof(float));
    env->rewards = (float *) calloc(env->active_agent_count, sizeof(float));
    env->terminals = (unsigned char *) calloc(env->active_agent_count, sizeof(unsigned char));
    env->truncations = (unsigned char *) calloc(env->active_agent_count, sizeof(unsigned char));
    env->masks = (unsigned char *) calloc(env->active_agent_count, sizeof(unsigned char));
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

void c_get_global_agent_state(
    Drive *env,
    float *x_out,
    float *y_out,
    float *z_out,
    float *heading_out,
    int *id_out,
    float *length_out,
    float *width_out) {
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

void c_get_global_ground_truth_trajectories(
    Drive *env,
    float *x_out,
    float *y_out,
    float *z_out,
    float *heading_out,
    int *valid_out,
    int *id_out,
    int *scenario_id_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        scenario_id_out[i] = 0; // TODO: FIXME

        for (int t = env->init_step; t < agent->trajectory_size; t++) {
            int out_idx = i * (agent->trajectory_size - env->init_step) + (t - env->init_step);
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
            points += env->road_elements[i].segment_size;
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
            lengths_out[poly_idx] = e->segment_size;
            scenario_ids_out[poly_idx] = 0; // TODO: FIXME
            for (int j = 0; j < e->segment_size; j++) {
                x_out[pt_idx] = e->x[j] + env->world_mean_x;
                y_out[pt_idx] = e->y[j] + env->world_mean_y;
                pt_idx++;
            }
            poly_idx++;
        }
    }
}

// ========================================
// Noise & Robustness Functions
// ========================================

static void subsample_road_observation_rows(
    Rng *rng_state,
    float *buffer,
    int collected_count,
    int keep_count,
    int feature_count) {
    if (keep_count <= 0 || collected_count <= keep_count) {
        return;
    }
    float tmp[feature_count];
    for (int sample_idx = 0; sample_idx < keep_count; sample_idx++) {
        int remaining = collected_count - sample_idx;
        int swap_idx = (remaining > 1) ? sample_idx + rng_below(rng_state, remaining) : sample_idx;
        if (swap_idx == sample_idx) {
            continue;
        }
        float *a = &buffer[sample_idx * feature_count];
        float *b = &buffer[swap_idx * feature_count];
        memcpy(tmp, a, sizeof(tmp));
        memcpy(a, b, sizeof(tmp));
        memcpy(b, tmp, sizeof(tmp));
    }
}

// ========================================
// Core Simulation Functions
// ========================================

static void compute_metrics(Drive *env, int agent_idx, int log_idx) {
    Agent *agent = &env->agents[agent_idx];
    Log *agent_log = &env->logs[log_idx];

    reset_agent_metrics(env, agent_idx);

    if (agent->sim_x == INVALID_POSITION) {
        return; // invalid agent position
    }
    if (get_grid_index(env, agent->sim_x, agent->sim_y) == -1) {
        // Current agent is offgrid, treat as offroad
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->offroad_behavior);
        return;
    }

    // Compute log-replay metrics
    if (env->simulation_mode == SIMULATION_MODE_REPLAY) {
        // Compute displacement error
        float displacement_error = compute_displacement_error(agent, env->timestep);
        if (displacement_error > 0.0f) { // Only count valid displacements
            agent->cumulative_displacement += displacement_error;
            agent->displacement_sample_count++;

            // Compute running average
            agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX]
                = agent->cumulative_displacement / agent->displacement_sample_count;
        }
    }

    bool is_offroad = false;

    // Track best candidate by combined distance/heading score
    float best_score = 1e9f;
    int lane_idx = -1;
    float signed_lane_distance = 0.0f, lane_heading = 0.0f;

    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int list_size = get_neighbors_entities(
        env,
        agent->sim_x,
        agent->sim_y,
        entity_list,
        ROAD_QUERY_ENTITY_COUNT,
        ROAD_OFFSETS,
        25);

    if (list_size <= 0) {
        is_offroad = true;
    }

    // Vehicle-width based distance threshold (3x width)
    float max_distance_threshold = 3.0f * agent->sim_width;

    int checked_lanes[MAX_CHECKED_LANES];
    int num_checked_lanes = 0;

    // Loop through road entities and compute associated metrics (offroad, lane alignment)
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1) {
            continue;
        }

        int entity_idx = entity_list[i].entity_idx;
        int geometry_idx = entity_list[i].geometry_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];

        // Check for offroad crossing with road edges
        if (is_road_edge(element->type)) {
            float abs_dz = fabsf(element->z[geometry_idx] - agent->sim_z);
            if (abs_dz > Z_BUFFER) {
                continue;
            }
            is_offroad = check_segment_crosses_moving_box(
                element->x[geometry_idx],
                element->y[geometry_idx],
                element->x[geometry_idx + 1],
                element->y[geometry_idx + 1],
                agent);
        }

        if (is_offroad) {
            break;
        }

        if (!is_drivable_road_lane(element->type)) {
            continue;
        }

        int already_checked = 0;
        for (int c = 0; c < num_checked_lanes; c++) {
            if (checked_lanes[c] == entity_idx) {
                already_checked = 1;
                break;
            }
        }
        if (already_checked) {
            continue;
        }
        if (num_checked_lanes < MAX_CHECKED_LANES) {
            checked_lanes[num_checked_lanes++] = entity_idx;
        }

        // Find closest segment on this lane (signed distance: left = negative, right = positive)
        int closest_seg_idx = 0;
        float signed_dist = 1e9f;
        int num_segments = element->segment_size - 1;
        if (num_segments >= 1) {
            float min_dist_sq = 1e18f;
            float closest_cross = 0.0f;
            for (int seg_idx = 0; seg_idx < num_segments; seg_idx++) {
                if (fabsf(element->z[seg_idx] - agent->sim_z) > Z_BUFFER) {
                    continue;
                }
                float seg_start_x = element->x[seg_idx];
                float seg_start_y = element->y[seg_idx];
                float seg_end_x = element->x[seg_idx + 1];
                float seg_end_y = element->y[seg_idx + 1];
                float seg_dx = seg_end_x - seg_start_x;
                float seg_dy = seg_end_y - seg_start_y;
                float seg_length_sq = seg_dx * seg_dx + seg_dy * seg_dy;
                float to_agent_x = agent->sim_x - seg_start_x;
                float to_agent_y = agent->sim_y - seg_start_y;
                float cross = seg_dx * to_agent_y - seg_dy * to_agent_x;
                float dist_sq;
                if (seg_length_sq > 1e-6f) {
                    float t = (to_agent_x * seg_dx + to_agent_y * seg_dy) / seg_length_sq;
                    if (t <= 0.0f) {
                        dist_sq = to_agent_x * to_agent_x + to_agent_y * to_agent_y;
                    } else if (t >= 1.0f) {
                        float dxe = agent->sim_x - seg_end_x;
                        float dye = agent->sim_y - seg_end_y;
                        dist_sq = dxe * dxe + dye * dye;
                    } else {
                        dist_sq = (cross * cross) / seg_length_sq;
                    }
                } else {
                    dist_sq = to_agent_x * to_agent_x + to_agent_y * to_agent_y;
                }
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_seg_idx = seg_idx;
                    closest_cross = cross;
                }
            }
            float abs_dist_val = sqrtf(min_dist_sq);
            signed_dist = (closest_cross >= 0.0f) ? -abs_dist_val : abs_dist_val;
        }

        float abs_dist = fabsf(signed_dist);
        if (abs_dist > max_distance_threshold) {
            continue;
        }

        // Multi-segment lane heading (more weight on center segment)
        float avg_lane_heading = 0.0f;
        float total_weight = 0.0f;
        int seg_start = (closest_seg_idx > 0) ? (closest_seg_idx - 1) : closest_seg_idx;
        int seg_end
            = (closest_seg_idx < element->segment_size - 2) ? (closest_seg_idx + 1) : (element->segment_size - 2);
        for (int seg_idx = seg_start; seg_idx <= seg_end; seg_idx++) {
            if (seg_idx < 0 || seg_idx >= element->segment_size - 1) {
                continue;
            }
            float seg_heading = element->headings[seg_idx];
            float weight = (seg_idx == closest_seg_idx) ? 2.0f : 1.0f;
            if (total_weight == 0.0f) {
                avg_lane_heading = seg_heading;
            } else {
                float angle_diff = compute_heading_diff(seg_heading, avg_lane_heading);
                avg_lane_heading += weight * angle_diff / (total_weight + weight);
            }
            total_weight += weight;
        }

        float heading_diff = compute_heading_diff(agent->sim_heading, avg_lane_heading);
        float heading_penalty = fabsf(heading_diff) / M_PI;
        float distance_penalty = abs_dist / LANE_DISTANCE_NORMALIZATION;
        float score
            = LANE_SELECTION_DISTANCE_WEIGHT * distance_penalty + LANE_SELECTION_HEADING_WEIGHT * heading_penalty;
        if (agent->current_lane_idx != entity_idx && agent->current_lane_idx != -1) {
            score += LANE_SWITCH_THRESHOLD;
        }

        if (score < best_score) {
            best_score = score;
            lane_idx = entity_idx;
            signed_lane_distance = signed_dist;
            lane_heading = avg_lane_heading;
        }
    }

    // Update lane alignment metric (running average)
    if (lane_idx != -1) {
        agent->previous_lane_idx = agent->current_lane_idx;
        agent->current_lane_idx = lane_idx;

        // Lane distance and angle metrics
        // x_f = lateral offset from lane center (left = negative, right = positive)
        agent->metrics_array[LANE_DIST_IDX] = signed_lane_distance;
        // Multi-lane detection: vehicle edge exceeds lane boundary
        float edge_dist = fabsf(signed_lane_distance) + agent->sim_width / 2.0f;
        if (env->compute_eval_metrics && edge_dist > MULTI_LANE_THRESHOLD && agent->sim_speed > 0.0f) {
            agent_log->multi_lane_time += env->dt;
        }
        // theta_f = angle relative to lane heading
        float theta_f = compute_heading_diff(agent->sim_heading, lane_heading);
        agent->metrics_array[LANE_ANGLE_IDX] = cosf(theta_f); // Store cos(θ_f)
    } else {
        // Agent not on any lane
        agent->previous_lane_idx = -1;
        agent->current_lane_idx = -1;
        agent->metrics_array[LANE_DIST_IDX] = LANE_DISTANCE_NORMALIZATION; // Max distance (far from lane)
        agent->metrics_array[LANE_ANGLE_IDX] = 0.0f;                       // Perpendicular (no alignment)
    }

    // Update cumulative metrics
    agent->distance_since_spawn += agent->sim_speed * env->dt;
    agent_log->avg_speed_per_agent += agent->sim_speed;

    // Speed limit metric (CUSTOM)
    float target_speed = 15.0f; // Default target speed
    int current_lane_idx = agent->current_lane_idx;
    if (current_lane_idx != -1 && env->road_elements[current_lane_idx].speed_limit > 0) {
        target_speed = env->road_elements[current_lane_idx].speed_limit;
    }
    // Binary overspeed metric, 1.0 if overspeeding by more than 2 m/s
    agent->metrics_array[SPEED_LIMIT_IDX] = (agent->sim_speed > target_speed + 2.0f) ? 1.0f : 0.0f;
    if (env->compute_eval_metrics) {
        agent_log->speed_violation_sum += fmaxf(agent->sim_speed - target_speed, 0.0f) * env->dt;
    }

    // Velocity metric - forward progress aligned with lane
    const float VELOCITY_MIN_SPEED = 2.5f; // m/s
    if (agent->sim_speed_signed > VELOCITY_MIN_SPEED && lane_idx != -1) {
        float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = fmaxf(cos_theta, 0.0f);
        if (env->compute_eval_metrics && cos_theta < 0.0f) {
            agent_log->wrong_way_distance += agent->sim_speed_signed * env->dt;
        }
    } else {
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = 0.0f;
    }

    // Comfort metric
    const float COMFORT_ACCEL_THRESHOLD = 3.0f; // m/s²
    const float COMFORT_JERK_THRESHOLD = 5.0f;  // m/s³
    int accel_violation
        = (fabsf(agent->accel_long) > COMFORT_ACCEL_THRESHOLD) + (fabsf(agent->accel_lat) > COMFORT_ACCEL_THRESHOLD);
    int jerk_violation
        = (fabsf(agent->jerk_long) > COMFORT_JERK_THRESHOLD || fabsf(agent->jerk_lat) > COMFORT_JERK_THRESHOLD) ? 1 : 0;
    agent->metrics_array[COMFORT_VIOLATION_IDX] = (float) (accel_violation + jerk_violation);

    // Handle terminal events - NOTE: move it elsewhere?
    // IMPORTANT: early returns after offroad and collision enforce mutual exclusivity of terminal flags.
    // Order matters: offroad > collision > red_light.

    // Priority 1: Handle offroad
    if (is_offroad) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->offroad_behavior);
        return;
    }

    // Priority 2: Handle vehicle collision
    int car_collided_with_index = collision_check(env, agent_idx);
    if (car_collided_with_index != -1) {
        agent->metrics_array[COLLISION_IDX] = 1.0f;
        if (env->compute_eval_metrics && is_at_fault_collision(env, agent_idx, car_collided_with_index)) {
            agent_log->at_fault_collision_rate = 1.0f;
            agent->metrics_array[AT_FAULT_COLLISION_IDX] = 1.0f;
        }
        apply_infraction_behavior(agent, env->collision_behavior);
        return;
    }

    // Priority 3: Handle red light violation
    if (env->obs_slots_traffic_controls_n && check_red_light_violation(env, agent_idx)) {
        agent->metrics_array[RED_LIGHT_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->traffic_light_behavior);
        return;
    }

    // Goal reaching: swept check against the step's motion segment (prev -> sim),
    // so a high dt cannot jump over the goal disc between two states.
    float distance_to_goal = compute_point_to_segment_distance(
        agent->current_goal_x,
        agent->current_goal_y,
        agent->prev_x,
        agent->prev_y,
        agent->sim_x,
        agent->sim_y);
    float goal_z_dist = fabsf(agent->sim_z - agent->current_goal_z);
    if (agent->current_goal_idx < agent->goal_count && distance_to_goal < agent->reward_coefs[REWARD_COEF_GOAL_RADIUS]
        && goal_z_dist < Z_BUFFER) {
        agent->metrics_array[REACHED_GOAL_IDX] = 1.0f;
        agent_log->num_goals_reached += 1;
        agent->current_goal_idx++;
        if (agent->current_goal_idx < agent->goal_count) {
            agent->current_goal_x = agent->list_goal_x[agent->current_goal_idx];
            agent->current_goal_y = agent->list_goal_y[agent->current_goal_idx];
            agent->current_goal_z = agent->list_goal_z[agent->current_goal_idx];
        }
    }

    float distance_to_goal_gt = compute_point_to_segment_distance(
        agent->gt_goal_x,
        agent->gt_goal_y,
        agent->prev_x,
        agent->prev_y,
        agent->sim_x,
        agent->sim_y);
    float goal_gt_z_dist = fabsf(agent->sim_z - agent->gt_goal_z);
    if (distance_to_goal_gt < GT_GOAL_RADIUS_M && goal_gt_z_dist < Z_BUFFER) {
        agent_log->reached_goal_gt = 1.0f;
    }
}

static void compute_rewards(Drive *env, int i) {
    int agent_idx = env->active_agent_indices[i];
    Agent *agent = &env->agents[agent_idx];
    Log *agent_log = &env->logs[i];

    // Collision reward
    if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
        // Velocity-dependent penalty: incentivizes braking before unavoidable collision.
        // At max speed (~20 m/s): extra -2.0 on top of base coefficient.
        float reward_collision = -(agent->reward_coefs[REWARD_COEF_COLLISION] + 0.1f * agent->sim_speed);
        env->rewards[i] += reward_collision;
        agent_log->collision_rate = 1.0f;
        agent_log->reward_collision += reward_collision;
    }

    // Offroad reward
    if (agent->metrics_array[OFFROAD_IDX] > 0.0f) {
        float reward_offroad = -agent->reward_coefs[REWARD_COEF_OFFROAD];
        env->rewards[i] += reward_offroad;
        agent_log->offroad_rate = 1.0f;
        agent_log->reward_offroad += reward_offroad;
    }

    // Red light violation reward
    if (agent->metrics_array[RED_LIGHT_IDX] > 0.0f) {
        float reward_red_light = -agent->reward_coefs[REWARD_COEF_STOP_LINE];
        env->rewards[i] += reward_red_light;
        agent_log->red_light_violation_rate = 1.0f;
        agent_log->reward_red_light += reward_red_light;
    }

    // Goal reward
    if (agent->metrics_array[REACHED_GOAL_IDX] > 0.0f) {
        bool final_waypoint = (agent->current_goal_idx == agent->goal_count);
        bool speeding = (agent->sim_speed > agent->reward_coefs[REWARD_COEF_GOAL_SPEED]);
        float reward_goal = (final_waypoint && speeding) ? 0.0f : env->reward_goal;
        env->rewards[i] += reward_goal;
        agent_log->reward_goal += reward_goal;
    }

    // Get lane angle metric: cos(θ_f) where θ_f = heading diff from lane
    float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
    float theta_f = acosf(fminf(fmaxf(cos_theta, -1.0f), 1.0f)); // Get |θ_f| from cos
    agent_log->lane_heading_aligned_rate += (cos_theta >= LANE_ALIGN_COS_THRESHOLD) ? 1.0f : 0.0f;

    // Rl-align: min(cos,0) + vel_align*min(cos*v,0) + 0.0025*(1-|θ|/(π/2))
    float against_lane_penalty = fminf(cos_theta, 0.0f); // negative when >90 degrees off
    float vel_aligned_penalty
        = agent->reward_coefs[REWARD_COEF_VEL_ALIGN] * fminf(cos_theta * agent->sim_speed_signed, 0.0f);
    float alignment_bonus = 0.0025f * (1.0f - theta_f / (M_PI / 2.0f));
    float lane_align_reward = agent->reward_coefs[REWARD_COEF_LANE_ALIGN] * env->dt
        * (against_lane_penalty + vel_aligned_penalty + alignment_bonus);
    env->rewards[i] += lane_align_reward;
    agent_log->reward_lane_align += lane_align_reward;

    // Rl-center: -α * dt * (|x_f - bias| - 0.05 / exp(|x_f - bias| - 0.5))
    float lane_center_distance = agent->metrics_array[LANE_DIST_IDX];
    float adjusted_dist = fabsf(lane_center_distance - agent->reward_coefs[REWARD_COEF_CENTER_BIAS]);
    float exp_decay = 0.05f / expf(adjusted_dist - 0.5f);
    float lane_center_reward
        = -agent->reward_coefs[REWARD_COEF_LANE_CENTER] * env->dt * ((cos_theta > 0.5f) * adjusted_dist - exp_decay);
    env->rewards[i] += lane_center_reward;
    agent_log->lane_center_rate += fabsf(lane_center_distance) < 0.5f ? 1.0f : 0.0f;
    agent_log->reward_lane_center += lane_center_reward;

    // Comfort reward
    float comfort_violations = agent->metrics_array[COMFORT_VIOLATION_IDX];
    float comfort_penalty = -agent->reward_coefs[REWARD_COEF_COMFORT] * comfort_violations;
    env->rewards[i] += comfort_penalty;
    agent_log->comfort_violation_count += comfort_violations;
    agent_log->reward_comfort += comfort_penalty;

    // Velocity reward
    float velocity_progress = agent->metrics_array[VELOCITY_PROGRESS_IDX];
    float velocity_reward = agent->reward_coefs[REWARD_COEF_VELOCITY] * env->dt * velocity_progress;
    env->rewards[i] += velocity_reward;
    agent_log->velocity_progress_sum += velocity_progress;
    agent_log->reward_velocity += velocity_reward;

    // Timestep reward
    float accel = sqrtf(agent->accel_long * agent->accel_long + agent->accel_lat * agent->accel_lat);
    if (agent->sim_speed > 0.01f || accel > 0.01f) {
        float timestep_penalty = -agent->reward_coefs[REWARD_COEF_TIMESTEP] * env->dt;
        env->rewards[i] += timestep_penalty;
        agent_log->reward_timestep += timestep_penalty;
    }

    // Reverse reward
    if (agent->sim_speed_signed < -0.01f) {
        float reverse_penalty = -agent->reward_coefs[REWARD_COEF_REVERSE] * env->dt;
        env->rewards[i] += reverse_penalty;
        agent_log->reward_reverse += reverse_penalty;
    }

    // Speed limit reward
    float speed_reward = -agent->reward_coefs[REWARD_COEF_OVERSPEED] * agent->metrics_array[SPEED_LIMIT_IDX];
    env->rewards[i] += speed_reward;
    agent_log->reward_overspeed += speed_reward;

    // ADE reward
    float current_ade = agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX];
    if (current_ade > 0.0f && env->reward_ade != 0.0f) {
        float ade_reward = env->reward_ade * current_ade;
        env->rewards[i] += ade_reward;
        agent_log->reward_ade += ade_reward;
    }
    agent_log->avg_displacement_error = current_ade;

    // Update episode return
    agent_log->episode_return += env->rewards[i];

    if (env->compute_eval_metrics) {
        float ml_time = agent_log->multi_lane_time;
        float ml_score = (ml_time <= MULTI_LANE_FULL_SCORE_TIME) ? 1.0f
            : (ml_time <= MULTI_LANE_HALF_SCORE_TIME)            ? 0.5f
                                                                 : 0.0f;
        agent_log->multi_lane_score = ml_score;
        agent->metrics_array[MULTI_LANE_TIME_IDX] = ml_time;
        agent->metrics_array[MULTI_LANE_SCORE_IDX] = ml_score;

        compute_agent_ttc(env, agent_idx);
        if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
            agent->metrics_array[TTC_IDX] = 0.0f;
            agent->metrics_array[DISTANCE_TO_COLLISION_IDX] = 0.0f;
        }
        float min_vehicle_ttc = agent->metrics_array[TTC_IDX];
        agent_log->ttc_samples += 1.0f;
        if (min_vehicle_ttc < TTC_VIOLATION_THRESHOLD) {
            agent_log->ttc_violations += 1.0f;
        }

        if (agent_log->ttc_samples > 0.0f) {
            agent_log->ttc_within_bound_rate = 1.0f - (agent_log->ttc_violations / agent_log->ttc_samples);
        } else {
            agent_log->ttc_within_bound_rate = 1.0f;
        }

        if (agent->metrics_array[COMFORT_VIOLATION_IDX] > 0.0f) {
            int comfort_window_idx = (int) (agent_log->episode_length * env->dt / METRIC_SCORE_WINDOW_SECONDS);
            if (comfort_window_idx != agent->comfort_violation_last_window_idx) {
                agent->comfort_violation_last_window_idx = comfort_window_idx;
                agent_log->comfort_violation_window_count += 1.0f;
            }
        }
        agent_log->comfort_score = calculate_duration_scaled_violation_score(
            agent_log->comfort_violation_window_count,
            agent_log->episode_length,
            env->dt);
    } else {
        agent->metrics_array[TTC_IDX] = DEFAULT_TTC;
        agent->metrics_array[DISTANCE_TO_COLLISION_IDX] = DEFAULT_DTC;
        agent->metrics_array[MULTI_LANE_TIME_IDX] = 0.0f;
        agent->metrics_array[MULTI_LANE_SCORE_IDX] = 0.0f;
        agent->metrics_array[AT_FAULT_COLLISION_IDX] = 0.0f;
    }

    // Update progress_ratio and puffer display fields during the episode
    float episode_duration_s = agent_log->episode_length * env->dt;
    float reference_progress_distance = PUFFER_PROGRESS_REFERENCE_SPEED * episode_duration_s;
    reference_progress_distance = fmaxf(reference_progress_distance, 1.0f);
    agent_log->progress_ratio = agent->distance_since_spawn / reference_progress_distance;

    calculate_puffer_score(agent_log, agent_log->episode_length, env->dt);
}

static int write_ego_obs(Drive *env, Agent *ego, float *obs, int obs_idx) {
    float perceived_margin = env->eval_mode ? 2.0f * env->eval_perceived_size_margin_m : 0.0f;
    obs[obs_idx++] = ego->sim_speed_signed / MAX_SPEED;
    obs[obs_idx++] = (ego->sim_width + perceived_margin) / env->obs_norm_veh_width_m;
    obs[obs_idx++] = (ego->sim_length + perceived_margin) / env->obs_norm_veh_length_m;
    obs[obs_idx++] = ego->steering_angle / STEERING_LIMIT;
    obs[obs_idx++] = ego->accel_long / fabsf(ACCEL_LONG_LIMIT[0]);
    obs[obs_idx++] = ego->accel_lat / ACCEL_LAT_LIMIT[1];
    obs[obs_idx++] = fmaxf(-1.0f, fminf(1.0f, ego->metrics_array[LANE_DIST_IDX] / LANE_DISTANCE_NORMALIZATION));
    obs[obs_idx++] = ego->metrics_array[LANE_ANGLE_IDX];
    float current_lane_speed_limit
        = (ego->current_lane_idx != -1) ? env->road_elements[ego->current_lane_idx].speed_limit : -1.0f;
    obs[obs_idx++] = current_lane_speed_limit / MAX_SPEED;
    obs[obs_idx++] = fminf(1.0f, ego->seconds_stopped / MAX_STOPPED_SECONDS);
    return obs_idx;
}

static int write_reward_target_obs(Drive *env, Agent *ego, float *obs, int obs_idx) {
    if (env->reward_conditioning) {
        for (int coef_idx = 0; coef_idx < NUM_REWARD_COEFS; coef_idx++) {
            float lo = REWARD_BOUNDS[coef_idx].min_val;
            float hi = REWARD_BOUNDS[coef_idx].max_val;
            float coef = ego->reward_coefs[coef_idx];
            float normalized_coef;
            if (REWARD_BOUNDS[coef_idx].log_scale) {
                // Match the log-uniform sampling so the conditioning signal stays even across [-1, 1].
                float clamped = fmaxf(lo, fminf(hi, coef));
                normalized_coef = (logf(clamped) - logf(lo)) / (logf(hi) - logf(lo));
            } else {
                normalized_coef = (coef - lo) / ((hi - lo) + 1e-8f);
            }
            float clamped_coef = fmaxf(0.0f, fminf(1.0f, normalized_coef));
            obs[obs_idx++] = 2.0f * clamped_coef - 1.0f;
        }
    }

    for (int goal_idx = 0; goal_idx < env->num_goals; goal_idx++) {
        if (goal_idx < ego->current_goal_idx || goal_idx >= ego->goal_count) {
            obs[obs_idx++] = 0.0f;
            obs[obs_idx++] = 0.0f;
            obs[obs_idx++] = 0.0f;
            continue;
        }
        float rel_goal_x, rel_goal_y;
        project_point_to_ego_frame(
            ego,
            ego->list_goal_x[goal_idx],
            ego->list_goal_y[goal_idx],
            &rel_goal_x,
            &rel_goal_y);
        obs[obs_idx++] = rel_goal_x / env->obs_norm_goal_offset_m;
        obs[obs_idx++] = rel_goal_y / env->obs_norm_goal_offset_m;
        obs[obs_idx++] = (ego->list_goal_z[goal_idx] - ego->sim_z) / env->obs_norm_z_m;
    }

    return obs_idx;
}

static int write_partner_obs(Drive *env, Agent *ego, int agent_idx, float *obs, int obs_idx, int *partner_count) {
    // Partner blindness: zero partner obs for the configured duration once triggered
    int partner_blindness_active = 0;
    if (ego->partner_blindness_counter > 0) {
        ego->partner_blindness_counter--;
        partner_blindness_active = 1;
    } else if (
        ego->is_blind_partner && env->partner_blindness_trigger_prob > 0.0f
        && sample_uniform(&env->rng_state, 0.0f, 1.0f) < env->partner_blindness_trigger_prob) {
        ego->partner_blindness_counter = env->partner_blindness_duration - 1;
        partner_blindness_active = 1;
    }
    if (partner_blindness_active) {
        int partner_obs_stride = env->obs_slots_partners_n * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, partner_obs_stride * sizeof(float));
        *partner_count = 0;
        return obs_idx + partner_obs_stride;
    }

    typedef struct {
        int index;
        float dist_sq;
        float dz;
    } AgentDistance;
    AgentDistance nearby_agents[env->num_agents];
    int nearby_count = 0;
    for (int j = 0; j < env->num_agents; j++) {
        int index = -1;
        if (j < env->active_agent_count) {
            index = env->active_agent_indices[j];
        } else if (j < env->num_agents) {
            index = env->static_agent_indices[j - env->active_agent_count];
        }
        if (index == env->active_agent_indices[agent_idx]) {
            continue; // Skip self, but don't increment obs_idx
        }
        Agent *other = &env->agents[index];
        float dx = other->sim_x - ego->sim_x;
        float dy = other->sim_y - ego->sim_y;
        float dz = other->sim_z - ego->sim_z;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq > env->obs_range_partner_m * env->obs_range_partner_m) {
            continue;
        }
        nearby_agents[nearby_count].index = index;
        nearby_agents[nearby_count].dist_sq = dist_sq;
        nearby_agents[nearby_count].dz = dz;
        nearby_count++;
    }

    int partners_written = 0;
    int partners_to_write = (nearby_count < env->obs_slots_partners_n) ? nearby_count : env->obs_slots_partners_n;
    for (int k = 0; k < partners_to_write; k++) {
        int nearest_idx = k;
        for (int j = k + 1; j < nearby_count; j++) {
            if (nearby_agents[j].dist_sq < nearby_agents[nearest_idx].dist_sq) {
                nearest_idx = j;
            }
        }
        if (nearest_idx != k) {
            AgentDistance tmp = nearby_agents[k];
            nearby_agents[k] = nearby_agents[nearest_idx];
            nearby_agents[nearest_idx] = tmp;
        }
    }

    for (int j = 0; j < partners_to_write; j++) {
        Agent *other = &env->agents[nearby_agents[j].index];
        float rel_x, rel_y, rel_heading_x, rel_heading_y, rel_vx, rel_vy;
        project_point_to_ego_frame(ego, other->sim_x, other->sim_y, &rel_x, &rel_y);
        project_point_to_local(
            other->sim_vx,
            other->sim_vy,
            ego->sim_vx,
            ego->sim_vy,
            ego->cos_heading,
            ego->sin_heading,
            &rel_vx,
            &rel_vy);
        project_vector_to_ego_frame(ego, other->cos_heading, other->sin_heading, &rel_heading_x, &rel_heading_y);
        obs[obs_idx++] = rel_x / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_y / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = nearby_agents[j].dz / env->obs_norm_z_m;
        obs[obs_idx++] = other->sim_length / env->obs_norm_veh_length_m;
        obs[obs_idx++] = other->sim_width / env->obs_norm_veh_width_m;
        obs[obs_idx++] = rel_heading_x;
        obs[obs_idx++] = rel_heading_y;
        obs[obs_idx++] = other->sim_speed_signed / MAX_SPEED;
        // TODO(hack): partner seconds_stopped is a temporary feature; remove later.
        obs[obs_idx++] = fminf(1.0f, other->seconds_stopped / MAX_STOPPED_SECONDS);
        partners_written++;
    }

    *partner_count = partners_written;
    return obs_idx + (env->obs_slots_partners_n - partners_written) * PARTNER_FEATURES;
}

static int write_road_obs(Drive *env, Agent *ego, float *obs, int obs_idx, int *lane_count, int *boundary_count) {
    int grid_idx = get_grid_index(env, ego->sim_x, ego->sim_y);
    int neighbor_count = 0;
    const GridMapEntity *neighbor_entities = NULL;
    if (!(grid_idx < 0 || grid_idx >= (env->grid_map->grid_cols * env->grid_map->grid_rows))) {
        if (env->use_neighbor_cache) {
            neighbor_count = env->grid_map->neighbor_cache_count[grid_idx];
            neighbor_entities = env->grid_map->neighbor_cache_entities[grid_idx];
        } else {
            // Same spiral order as the cache build, so obs are bit-identical to the cached path.
            neighbor_count = get_neighbors_entities(
                env,
                ego->sim_x,
                ego->sim_y,
                env->obs_neighbor_scratch,
                env->grid_map->total_entities,
                (const int (*)[2]) env->neighbor_offsets,
                env->grid_map->vision_range * env->grid_map->vision_range);
            neighbor_entities = env->obs_neighbor_scratch;
        }
    }

    // GPS lane-distance features
    int goal_graph_idx = -1;
    if (env->obs_goal_lane_distance && env->lane_graph.lane_to_graph_idx != NULL
        && ego->current_goal_idx < ego->goal_count) {
        int goal_lane = ego->list_goal_lane[ego->current_goal_idx];
        if (goal_lane >= 0 && goal_lane < env->num_road_elements) {
            goal_graph_idx = env->lane_graph.lane_to_graph_idx[goal_lane];
        }
    }
    // Ego's own lane->goal distance: reference for the relative column (delta vs ego lane).
    float ego_dist_to_goal_m = -1.0f; // <0 = no valid reference -> relative column stays 0
    if (goal_graph_idx >= 0) {        // implies obs_goal_lane_distance and a non-NULL lane_to_graph_idx
        int ego_lane = ego->current_lane_idx;
        if (ego_lane >= 0 && ego_lane < env->num_road_elements) {
            int ego_graph_idx = env->lane_graph.lane_to_graph_idx[ego_lane];
            if (ego_graph_idx >= 0) {
                float d = env->lane_graph.distances[ego_graph_idx * env->lane_graph.n_lanes + goal_graph_idx];
                // Map binaries store unreachable pairs as a negative sentinel or NaN; both clamp to max.
                ego_dist_to_goal_m = (!isfinite(d) || d < 0.0f) ? LANE_GRAPH_DISTANCE_NORM_M : d;
            }
        }
    }

    int lane_obs_idx = obs_idx;
    int boundary_obs_idx = lane_obs_idx + env->obs_slots_lane_kept * LANE_FEATURES;
    obs_idx = boundary_obs_idx + env->obs_slots_boundary_kept * BOUNDARY_FEATURES;

    float lanes_buffer[env->obs_slots_lane_n * LANE_FEATURES];
    float boundaries_buffer[env->obs_slots_boundary_n * BOUNDARY_FEATURES];
    float *lane_obs_dest = env->road_dropout_enabled ? lanes_buffer : &obs[lane_obs_idx];
    float *boundary_obs_dest = env->road_dropout_enabled ? boundaries_buffer : &obs[boundary_obs_idx];
    int lanes_found = 0;
    int boundaries_found = 0;

    for (int k = 0; k < neighbor_count; k++) {
        if (lanes_found >= env->obs_slots_lane_n && boundaries_found >= env->obs_slots_boundary_n) {
            break;
        }
        if (!neighbor_entities[k].valid_for_obs) {
            continue;
        }
        int entity_idx = neighbor_entities[k].entity_idx;
        int geometry_idx = neighbor_entities[k].geometry_idx;
        RoadMapElement *road_element = &env->road_elements[entity_idx];
        int is_lane = is_road_lane(road_element->type);
        int is_edge = is_road_edge(road_element->type);
        if (!is_lane && !is_edge) {
            continue;
        }

        float start_x = road_element->x[geometry_idx];
        float start_y = road_element->y[geometry_idx];
        float start_z = road_element->z[geometry_idx];
        float end_x = road_element->x[geometry_idx + 1];
        float end_y = road_element->y[geometry_idx + 1];
        float end_z = road_element->z[geometry_idx + 1];
        float mid_x = (start_x + end_x) / 2.0f;
        float mid_y = (start_y + end_y) / 2.0f;
        float mid_z = (start_z + end_z) / 2.0f;
        float rel_x, rel_y;
        float rel_z = mid_z - ego->sim_z;
        project_point_to_ego_frame(ego, mid_x, mid_y, &rel_x, &rel_y);
        if (rel_x < -env->obs_range_road_behind_m || rel_x > env->obs_range_road_front_m) {
            continue;
        }
        if (fabsf(rel_y) > env->obs_range_road_side_m) {
            continue;
        }

        float seg_dx = end_x - mid_x;
        float seg_dy = end_y - mid_y;
        float seg_half_len = sqrtf(seg_dx * seg_dx + seg_dy * seg_dy);
        float seg_dir_x = (seg_half_len > 0) ? seg_dx / seg_half_len : seg_dx;
        float seg_dir_y = (seg_half_len > 0) ? seg_dy / seg_half_len : seg_dy;
        float rel_seg_dir_x, rel_seg_dir_y;
        project_vector_to_ego_frame(ego, seg_dir_x, seg_dir_y, &rel_seg_dir_x, &rel_seg_dir_y);

        float *segment_dest = is_lane ? lane_obs_dest : boundary_obs_dest;
        int *segment_count = is_lane ? &lanes_found : &boundaries_found;
        int segment_cap = is_lane ? env->obs_slots_lane_n : env->obs_slots_boundary_n;
        int segment_features = is_lane ? LANE_FEATURES : BOUNDARY_FEATURES;
        if (*segment_count >= segment_cap) {
            continue;
        }
        int feature_base = (*segment_count)++ * segment_features;
        segment_dest[feature_base] = rel_x / env->obs_norm_xy_offset_m;
        segment_dest[feature_base + 1] = rel_y / env->obs_norm_xy_offset_m;
        segment_dest[feature_base + 2] = rel_z / env->obs_norm_z_m;
        segment_dest[feature_base + 3] = seg_half_len / env->obs_norm_road_seg_length_m;
        segment_dest[feature_base + 4] = LANE_WIDTH / env->obs_norm_road_seg_width_m;
        segment_dest[feature_base + 5] = rel_seg_dir_x;
        segment_dest[feature_base + 6] = rel_seg_dir_y;
        // Goal-distance features: absolute and relative to ego's lane->goal distance.
        if (is_lane) {
            float goal_dist_abs = 0.0f, goal_dist_rel = 0.0f; // 0 when flag off / unresolved
            if (env->obs_goal_lane_distance && goal_graph_idx >= 0 && entity_idx < env->num_road_elements) {
                int lane_graph_idx = env->lane_graph.lane_to_graph_idx[entity_idx];
                if (lane_graph_idx >= 0) {
                    float d = env->lane_graph.distances[lane_graph_idx * env->lane_graph.n_lanes + goal_graph_idx];
                    float d_m = (!isfinite(d) || d < 0.0f) ? LANE_GRAPH_DISTANCE_NORM_M : d; // unreachable/NaN -> max
                    goal_dist_abs = clip(d_m / LANE_GRAPH_DISTANCE_NORM_M, 0.0f, 1.0f);
                    if (ego_dist_to_goal_m >= 0.0f) {
                        goal_dist_rel = clip((d_m - ego_dist_to_goal_m) / LANE_GRAPH_DISTANCE_NORM_M, -1.0f, 1.0f);
                    }
                }
            }
            segment_dest[feature_base + 7] = goal_dist_abs;
            segment_dest[feature_base + 8] = goal_dist_rel;
        } else {
            // NOTE: Remove this with next model
            segment_dest[feature_base + 7] = 0.0f;
            segment_dest[feature_base + 8] = 0.0f;
        }
    }

    if (env->road_dropout_enabled) {
        int lanes_to_copy = (lanes_found < env->obs_slots_lane_kept) ? lanes_found : env->obs_slots_lane_kept;
        int boundaries_to_copy
            = (boundaries_found < env->obs_slots_boundary_kept) ? boundaries_found : env->obs_slots_boundary_kept;
        *lane_count = lanes_to_copy;
        *boundary_count = boundaries_to_copy;
        subsample_road_observation_rows(&env->rng_state, lanes_buffer, lanes_found, lanes_to_copy, LANE_FEATURES);
        subsample_road_observation_rows(
            &env->rng_state,
            boundaries_buffer,
            boundaries_found,
            boundaries_to_copy,
            BOUNDARY_FEATURES);
        memcpy(&obs[lane_obs_idx], lanes_buffer, lanes_to_copy * LANE_FEATURES * sizeof(float));
        memset(
            &obs[lane_obs_idx + lanes_to_copy * LANE_FEATURES],
            0,
            (env->obs_slots_lane_kept - lanes_to_copy) * LANE_FEATURES * sizeof(float));
        memcpy(&obs[boundary_obs_idx], boundaries_buffer, boundaries_to_copy * BOUNDARY_FEATURES * sizeof(float));
        memset(
            &obs[boundary_obs_idx + boundaries_to_copy * BOUNDARY_FEATURES],
            0,
            (env->obs_slots_boundary_kept - boundaries_to_copy) * BOUNDARY_FEATURES * sizeof(float));
        return obs_idx;
    }

    *lane_count = lanes_found;
    *boundary_count = boundaries_found;
    memset(
        &obs[lane_obs_idx + lanes_found * LANE_FEATURES],
        0,
        (env->obs_slots_lane_kept - lanes_found) * LANE_FEATURES * sizeof(float));
    memset(
        &obs[boundary_obs_idx + boundaries_found * BOUNDARY_FEATURES],
        0,
        (env->obs_slots_boundary_kept - boundaries_found) * BOUNDARY_FEATURES * sizeof(float));
    return obs_idx;
}

static int write_traffic_control_obs(Drive *env, Agent *ego, float *obs, int obs_idx, int *traffic_control_count) {
    typedef struct {
        int idx;
        float dist_sq;
    } TrafficControlDist;
    TrafficControlDist visible_controls[env->num_traffic_elements > 0 ? env->num_traffic_elements : 1];
    int visible_count = 0;

    for (int j = 0; j < env->num_traffic_elements; j++) {
        TrafficControlElement *tc = &env->traffic_elements[j];
        if (!traffic_control_in_scope(tc->type, env->traffic_control_scope)) {
            continue;
        }
        float mid_x = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
        float mid_y = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;
        float mid_z = (tc->stop_line[2] + tc->stop_line[5]) * 0.5f;
        float dx = mid_x - ego->sim_x;
        float dy = mid_y - ego->sim_y;
        float dz = mid_z - ego->sim_z;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq > env->obs_range_traffic_control_m * env->obs_range_traffic_control_m) {
            continue;
        }
        visible_controls[visible_count].idx = j;
        visible_controls[visible_count].dist_sq = dist_sq;
        visible_count++;
    }

    int controls_to_observe
        = (visible_count < env->obs_slots_traffic_controls_n) ? visible_count : env->obs_slots_traffic_controls_n;
    for (int k = 0; k < controls_to_observe; k++) {
        int nearest_idx = k;
        for (int j = k + 1; j < visible_count; j++) {
            if (visible_controls[j].dist_sq < visible_controls[nearest_idx].dist_sq) {
                nearest_idx = j;
            }
        }
        if (nearest_idx != k) {
            TrafficControlDist temp = visible_controls[k];
            visible_controls[k] = visible_controls[nearest_idx];
            visible_controls[nearest_idx] = temp;
        }
    }

    int controls_written = 0;
    for (int j = 0; j < controls_to_observe && controls_written < env->obs_slots_traffic_controls_n; j++) {
        TrafficControlElement *tc = &env->traffic_elements[visible_controls[j].idx];
        float rel_x1, rel_y1, rel_x2, rel_y2;
        project_point_to_ego_frame(ego, tc->stop_line[0], tc->stop_line[1], &rel_x1, &rel_y1);
        project_point_to_ego_frame(ego, tc->stop_line[3], tc->stop_line[4], &rel_x2, &rel_y2);
        float rel_z = (tc->stop_line[2] + tc->stop_line[5]) * 0.5f - ego->sim_z;
        int light_state = TRAFFIC_CONTROL_STATE_UNKNOWN;
        if (tc->type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            light_state = (env->timestep >= 0 && env->timestep < tc->state_size && tc->states != NULL)
                ? tc->states[env->timestep]
                : TRAFFIC_CONTROL_STATE_OFF;
        }

        obs[obs_idx++] = rel_x1 / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_y1 / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_x2 / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_y2 / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_z / env->obs_norm_z_m;
        obs[obs_idx++] = tc->type;
        obs[obs_idx++] = light_state;
        controls_written++;
    }

    *traffic_control_count = controls_written;
    return obs_idx + (env->obs_slots_traffic_controls_n - controls_written) * TRAFFIC_CONTROL_FEATURES;
}

static void compute_observations(Drive *env) {
    int obs_per_agent = compute_observation_size(env);
    memset(env->observations, 0, obs_per_agent * env->active_agent_count * sizeof(float));
    float (*obs_matrix)[obs_per_agent] = (float (*)[obs_per_agent]) env->observations;
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &obs_matrix[i][0];
        int agent_idx = env->active_agent_indices[i];
        Agent *ego = &env->agents[agent_idx];
        int partner_count = 0;
        int lane_count = 0;
        int boundary_count = 0;
        int traffic_control_count = 0;
        int obs_idx = 0;

        obs_idx = write_ego_obs(env, ego, obs, obs_idx);
        obs_idx = write_reward_target_obs(env, ego, obs, obs_idx);
        obs_idx = write_partner_obs(env, ego, i, obs, obs_idx, &partner_count);
        obs_idx = write_road_obs(env, ego, obs, obs_idx, &lane_count, &boundary_count);
        obs_idx = write_traffic_control_obs(env, ego, obs, obs_idx, &traffic_control_count);
        obs[obs_idx++] = (float) lane_count;
        obs[obs_idx++] = (float) boundary_count;
        obs[obs_idx++] = (float) partner_count;
        obs[obs_idx++] = (float) traffic_control_count;
        assert(obs_idx == obs_per_agent);
    }
}

static void move_dynamics(Drive *env, int action_idx, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    copy_pose_to_prev(agent);

    if (agent->removed) {
        invalidate_agent(agent);
        return;
    }
    if (agent->stopped) {
        clear_agent_motion(agent);
        agent->steering_angle = 0.0f;
        return;
    }

    // Phantom braking: override action with max braking
    int phantom_braking_active = 0;
    if (agent->phantom_braking_counter > 0) {
        agent->phantom_braking_counter--;
        phantom_braking_active = 1;
    } else if (
        agent->is_phantom_braker && env->phantom_braking_trigger_prob > 0.0f
        && sample_uniform(&env->rng_state, 0.0f, 1.0f) < env->phantom_braking_trigger_prob) {
        agent->phantom_braking_counter = env->phantom_braking_duration - 1;
        phantom_braking_active = 1;
    }

    if (env->dynamics_model == DYNAMICS_MODEL_CLASSIC) {
        // Classic dynamics model
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == ACTION_TYPE_DISCRETE) {
            // Interpret action as a single integer: a = accel_idx * num_steer + steer_idx
            int *action_array = (int *) env->actions;
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            int action_val = action_array[action_idx];
            int acceleration_index = action_val / num_steer;
            int steering_index = action_val % num_steer;
            acceleration = ACCELERATION_VALUES[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        } else if (env->action_type == ACTION_TYPE_CONTINUOUS) {
            float (*action_array_f)[2] = (float (*)[2]) env->actions;
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
        // If phantom braking is active, prevent speed from going negative
        if (phantom_braking_active) {
            if (speed < 0.0f) {
                speed = 0.0f;
                acceleration = 0.0f;
            }
        }
        speed = clip(speed, -MAX_SPEED, MAX_SPEED);
        // Compute yaw rate
        float beta = atanf(REAR_AXLE_RATIO * tanf(steering));
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
        agent->jerk_long = (new_a_long - agent->accel_long) / env->dt;
        agent->jerk_lat = (new_a_lat - agent->accel_lat) / env->dt;
        agent->accel_long = new_a_long;
        agent->accel_lat = new_a_lat;
    } else if (env->dynamics_model == DYNAMICS_MODEL_JERK) {
        // Extract jerk action components
        float j_long, j_lat;
        if (env->action_type == ACTION_TYPE_DISCRETE) {
            // Interpret action as a single integer: a = long_idx * num_lat + lat_idx
            int *action_array = (int *) env->actions;
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            int action_val = action_array[action_idx];
            int j_long_idx = action_val / num_lat;
            int j_lat_idx = action_val % num_lat;
            j_long = JERK_LONG[j_long_idx];
            j_lat = JERK_LAT[j_lat_idx];
        } else if (env->action_type == ACTION_TYPE_CONTINUOUS) {
            float (*action_array_f)[2] = (float (*)[2]) env->actions;
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
        float a_long_new = agent->accel_long + c_throttle * j_long * env->dt;

        // Zero-crossing: snap to 0 when crossing zero
        if (agent->accel_long * a_long_new < 0) {
            a_long_new = 0.0f;
        } else {
            a_long_new = clip(a_long_new, ACCEL_LONG_LIMIT[0], ACCEL_LONG_LIMIT[1] * c_acc);
        }

        // Calculate new lateral acceleration from jerk (Eq. 2 in paper)
        float a_lat_new = agent->accel_lat + c_steer * j_lat * env->dt;

        // Zero-crossing: snap to 0 when crossing zero
        if (agent->accel_lat * a_lat_new < 0) {
            a_lat_new = 0.0f;
        } else {
            a_lat_new = clip(a_lat_new, ACCEL_LAT_LIMIT[0], ACCEL_LAT_LIMIT[1]);
        }

        float heading_x = agent->cos_heading;
        float heading_y = agent->sin_heading;

        // Calculate new velocity using trapezoidal integration
        float v_dot_heading = agent->sim_vx * heading_x + agent->sim_vy * heading_y;
        float signed_v = copysignf(sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy), v_dot_heading);
        float v_new = signed_v + 0.5f * (a_long_new + agent->accel_long) * env->dt;

        // Zero-crossing: snap to 0 when crossing zero
        if (signed_v * v_new < 0) {
            v_new = 0.0f;
            if (env->reset_accel_on_stop) {
                a_long_new = 0.0f;
                a_lat_new = 0.0f;
            }
        } else {
            v_new = clip(v_new, -2.0f, MAX_SPEED);
        }

        // If phantom braking is active, prevent speed from going negative
        if (phantom_braking_active) {
            if (v_new < 0.0f) {
                v_new = 0.0f;
                a_long_new = 0.0f;
            }
        }

        // GIGAFLOW paper approach: accel_lat → curvature → steering
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
        agent->jerk_long = (a_long_new - agent->accel_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->accel_lat) / env->dt;
        agent->accel_long = a_long_new;
        agent->accel_lat = a_lat_new;
        agent->steering_angle = new_steering_angle;
    }

    update_agent_z(env, agent);

    return;
}

#include "idm.h"

void c_reset(Drive *env) {
    if (env->timestep == 0) {
        for (int i = 0; i < env->num_total_agents; i++) {
            copy_pose_to_prev(&env->agents[i]);
        }
        for (int x = 0; x < env->active_agent_count; x++) {
            env->logs[x] = (Log) {0};
            int agent_idx = env->active_agent_indices[x];
            sample_erratic_flags(env, &env->agents[agent_idx]);
            compute_metrics(env, agent_idx, x);
        }
        compute_observations(env);
        return;
    }

    env->timestep = env->init_step;

    begin_episode_rng(env);
    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
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
            printf(
                "[GIGAFLOW ERROR] -> Only respawned %d out of %d agents during reset\n",
                num_reset,
                env->active_agent_count);
        }

        // GIGAFLOW: spawn_agent already set positions, routes, paths, goals.
        // Only need to generate reward coefs and compute initial metrics.
        for (int x = 0; x < env->active_agent_count; x++) {
            env->logs[x] = (Log) {0};
            int agent_idx = env->active_agent_indices[x];
            Agent *agent = &env->agents[agent_idx];
            if (agent->removed) {
                continue;
            }
            reset_agent_metrics(env, agent_idx);
            reset_agent_state(agent);
            sample_erratic_flags(env, agent);
            generate_reward_coefs(env, agent);
            compute_metrics(env, agent_idx, x);
        }
        compute_observations(env);
        return;
    }

    set_start_position(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log) {0};
        int agent_idx = env->active_agent_indices[x];
        Agent *agent = &env->agents[agent_idx];

        // Common resets
        reset_agent_metrics(env, agent_idx);
        reset_agent_state(agent);
        sample_erratic_flags(env, agent);
        generate_reward_coefs(env, agent);

        if (env->goal_source == GOAL_SOURCE_GT) {
            int start = env->init_step > 0 ? env->init_step : 0;
            int remaining = agent->trajectory_size - 1 - start;
            if (remaining < 1) {
                remaining = 1;
            }
            int num_wp = env->num_goals;
            for (int g = 0; g < num_wp; g++) {
                int t = start + (g + 1) * remaining / num_wp;
                if (t >= agent->trajectory_size) {
                    t = agent->trajectory_size - 1;
                }
                agent->list_goal_x[g] = agent->log_trajectory_x[t];
                agent->list_goal_y[g] = agent->log_trajectory_y[t];
                agent->list_goal_z[g] = agent->log_trajectory_z[t];
                agent->list_goal_lane[g] = -1; // logged goals have no lane idx (no GPS lane-distance)
            }
            agent->goal_count = num_wp;
            agent->current_goal_idx = 0;
            agent->current_goal_x = agent->list_goal_x[0];
            agent->current_goal_y = agent->list_goal_y[0];
            agent->current_goal_z = agent->list_goal_z[0];
        } else {
            generate_new_goals_from_route(env, agent);
        }
        compute_metrics(env, agent_idx, x);
    }
    compute_observations(env);
}

void c_step(Drive *env) {
    // In eval, a scenario is evaluated once: hold the env after its episode
    // ended, so a short episode does not replay and re-emit within the window.
    if (env->eval_mode && env->eval_episode_done) {
        memset(env->rewards, 0, env->active_agent_count * sizeof(float));
        memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
        memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));
        return;
    }
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));

    // Update masks for stopped/removed agents
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *a = &env->agents[agent_idx];
        if (a->stopped || a->removed || a->is_blind_partner || a->is_phantom_braker) {
            env->masks[i] = 0;
        } else {
            env->masks[i] = 1;
        }
    }

    env->timestep++;

    // -> 1. Apply actions and move agents
    // Move static experts
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int background_idx = env->expert_static_agent_indices[i];
        Agent *agent = &env->agents[background_idx];
        if (agent->controller == CONTROLLER_IDM) {
            move_idm(env, background_idx);
        } else if (agent->controller == CONTROLLER_REPLAY && env->simulation_mode == SIMULATION_MODE_REPLAY) {
            move_expert(env, background_idx);
        }
    }
    // Move active agents with policy actions
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        if (agent->controller == CONTROLLER_POLICY) {
            move_dynamics(env, i, agent_idx);
        } else if (agent->controller == CONTROLLER_IDM) {
            move_idm(env, agent_idx);
        } else if (agent->controller == CONTROLLER_REPLAY && env->simulation_mode == SIMULATION_MODE_REPLAY) {
            move_expert(env, agent_idx);
        }
    }

    // Update stopped-duration for every agent (active + replayed/static), not
    // just policy-controlled ones, so the partner seconds_stopped observation is
    // populated even in control_sdc_only mode where only the ego is active.
    for (int j = 0; j < env->num_agents; j++) {
        int agent_idx = (j < env->active_agent_count) ? env->active_agent_indices[j]
                                                      : env->static_agent_indices[j - env->active_agent_count];
        Agent *agent = &env->agents[agent_idx];
        if (agent->sim_speed < AGENT_STOPPED_SPEED_THRESHOLD) {
            agent->seconds_stopped += env->dt;
        } else {
            agent->seconds_stopped = 0.0f;
        }
    }

    // -> 2. Compute metrics and rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->agents[agent_idx].stopped || env->agents[agent_idx].removed) {
            continue;
        }
        compute_metrics(env, agent_idx, i);
        compute_rewards(env, i);
    }

    // Mark terminals for stopped or removed agents
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->agents[agent_idx].stopped || env->agents[agent_idx].removed) {
            env->terminals[i] = 1;
        }
    }

    // -> 3. Check for episode truncation
    int early_reset = 0;
    if (env->termination_mode == 1) {
        int count_inactive = 0;
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            if (env->agents[agent_idx].removed || env->agents[agent_idx].stopped) {
                count_inactive++;
            }
        }
        float ratio_inactive = (float) count_inactive / (float) env->active_agent_count;
        if (ratio_inactive > env->inactive_agent_threshold) {
            early_reset = 1;
        }
    }

    if (env->terminate_on_goal == 1 && env->simulation_mode == SIMULATION_MODE_REPLAY
        && env->control_mode == CONTROL_MODE_SDC_ONLY) {
        for (int i = 0; i < env->active_agent_count; i++) {
            Agent *agent = &env->agents[env->active_agent_indices[i]];
            if (agent->metrics_array[REACHED_GOAL_IDX] > 0.0f && agent->current_goal_idx == env->num_goals) {
                early_reset = 1;
            }
        }
    }

    if (env->timestep == env->scenario_length || early_reset) {
        for (int i = 0; i < env->active_agent_count; i++) {
            env->truncations[i] = 1;
        }
        add_log(env);
        if (env->eval_mode) {
            env->eval_episode_done = 1;
            return;
        }
        c_reset(env);
        return;
    }

    // -> 4. Compute observations
    compute_observations(env);

    // -> 5. Update goals for agents that reached their goal
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        if (agent->metrics_array[REACHED_GOAL_IDX] == 0.0f) {
            continue;
        }
        if (env->goal_source == GOAL_SOURCE_GT) {
            // Replay mode: leave current_goal_idx saturated so the
            // reached-goal condition won't fire again. Re-generating
            // route-based goals on WOMD maps fails (removed=1).
            continue;
        }
        // Rolling slides the window forward by one goal; finite advances the alias to the next goal in the set.
        // Both fall back to a full regen: rolling on dead-end, finite when exhausted.
        bool regen;
        if (env->goal_regen_mode == GOAL_REGEN_ROLLING) {
            regen = !roll_goals(env, agent);
        } else if (agent->current_goal_idx == agent->goal_count) {
            regen = true;
        } else {
            agent->current_goal_x = agent->list_goal_x[agent->current_goal_idx];
            agent->current_goal_y = agent->list_goal_y[agent->current_goal_idx];
            agent->current_goal_z = agent->list_goal_z[agent->current_goal_idx];
            continue;
        }
        if (!regen) {
            continue;
        }
        // On regen failure remove the agent (route helper self-invalidates; the extra call is idempotent),
        // otherwise a stale current_goal_idx >= goal_count leaves an empty goal window and the agent freezes.
        bool regen_ok = (env->goal_source == GOAL_SOURCE_MAP) ? generate_new_goals_from_map(env, agent)
                                                              : generate_new_goals_from_route(env, agent);
        if (!regen_ok) {
            invalidate_agent(agent);
            agent->removed = 1;
        }
    }
}

#include "render.h"
