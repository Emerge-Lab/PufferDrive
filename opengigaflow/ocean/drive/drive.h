#include "datatypes.h"
#include "error.h"
#include "rng.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    float z_dis;
    float euclidean_dis;
    float z;
} DepthPoint;

typedef struct Drive Drive;
typedef struct Log Log;
typedef struct Agent Agent;
typedef struct ScriptedPedestrian ScriptedPedestrian;
typedef struct RoadMapElement RoadMapElement;
typedef struct TrafficControlElement TrafficControlElement;
typedef struct GridMapEntity GridMapEntity;
typedef struct GridMap GridMap;

struct Log {
    float n;
    float episode_return;
    float episode_length;
    float score;
    float offroad_rate;
    float collision_rate;
    float red_light_violation_rate;
    float stop_sign_violation_rate;
    float num_goals_reached;
    float comfort_violation_count;
    float comfort_violation_window_count;
    float velocity_progress_sum;
    float lane_center_rate;
    float dnf_rate;
    float avg_speed_per_agent;
    float at_fault_collision_rate;
    float driving_direction_score;
    float speed_limit_compliance;
    float making_progress_rate;
    float progress_ratio;
    float comfort_score;
    float wrong_way_distance;
    float speed_violation_sum;
    float multi_lane_time;
    float multi_lane_score;
    float total_distance_travelled;
    float total_infractions;
    float reward_collision;
    float reward_offroad;
    float reward_red_light;
    float reward_stop_sign;
    float reward_goal;
    float reward_lane_align;
    float reward_lane_center;
    float reward_comfort;
    float reward_velocity;
    float reward_timestep;
    float reward_reverse;
    float reward_overspeed;
    float pedestrian_collision_rate;
    float pedestrian_crossings;
    float pedestrian_bystanders;
};

struct GridMapEntity {
    int entity_idx;   // Index into the road_elements array
    int geometry_idx; // Index into element's geometry array
    int obs_kind;     // OBS_ENTITY_NONE / LANE / EDGE: how road observations sample this entity
    int pool_idx;     // Position in entity_pool and segment_pool
};

// One road-query candidate of a cell's 5x5 window: a road-edge segment, or a drivable lane with its listed segments.
typedef struct {
    int entity_idx;
    int seg_first;
    int seg_last;
} LaneQueryEntry;

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
    int *neighbor_cache_lane_count; // per cell: leading lane entries of its neighbor cache; edge entries follow
    int *grid_index_drivable;
    int num_drivable_grid_cell;
    int total_entities;
    GridMapEntity *entity_pool;           // one block; cells[i] point into it
    RoadObservationSegment *segment_pool; // parallel to entity_pool
    GridMapEntity **cells;
    LaneQueryEntry *lane_query_pool;    // per cell, window order, lanes deduplicated at first appearance
    int *lane_query_offset;             // cell count + 1
    int *window_entity_count;           // every grid entity in the cell's 5x5 window
    uint16_t **neighbor_cache_pool_idx; // per cell: entity_pool indices of its spiral neighborhood
};

// Read-only map geometry shared by the envs on one map file; only the process that built it (owner_pid) frees it
struct SharedMapData {
    char *map_name;
    RoadMapElement *road_elements;
    int num_road_elements;
    GridMap *grid_map;
    int *neighbor_offsets;
    struct LaneGraph lane_graph;
    int obs_lane_stride;
    int obs_boundary_stride;
    int obs_stride_merge_segments;
    int ref_count;
    pid_t owner_pid;
};

static struct SharedMapData **g_map_cache = NULL;
static int g_map_cache_count = 0;

struct Drive {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;
    unsigned char *masks;
    Agent *agents;
    int num_controllable_agents;
    int cosim_partner_slots;       // gigaflow co-sim: static slots owned by the external sim (never spawned/stepped)
    int scripted_pedestrian_slots; // gigaflow: scripted pedestrians appended after the policy agents
    ScriptedPedestrian *pedestrians;
    int active_agent_count;
    int *active_agent_indices;
    int num_total_agents;
    int num_max_agents;
    int num_agents;
    int static_agent_count;
    int *static_agent_indices;
    int expert_static_agent_count;
    int *expert_static_agent_indices;
    char *map_name;
    RoadMapElement *road_elements;
    int num_road_elements;
    int num_speed_zones;
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
    char scenario_id[128];
    char dataset_name[32];
    int scenario_length;
    int log_length;
    float log_dt;
    int num_objects_of_interest;
    int *objects_of_interest;
    int num_tracks_to_predict;
    int *tracks_to_predict;
    int timestep;
    int autoreset_pending;
    int init_step;
    int init_step_min_horizon;
    int stagger_first_episode;
    int first_reset_pending;
    int episode_end_timestep;
    float dt;
    float base_max_speed_mps;
    float max_speed_mps;
    float spawn_lateral_offset_max_frac;
    float spawn_heading_max_deg;
    float pose_noise_xy_m;
    float pose_noise_yaw_rad;
    float speed_limit_random_prob;
    float speed_limit_random_delta_mps;
    float speed_limit_random_min_mps;
    float speed_limit_random_max_mps;
    float *lane_speed_limit_mps;        // per-episode effective limit per road element (zone-randomized)
    float *speed_zone_offset_mps;       // num_speed_zones
    unsigned char *lane_limit_resolved; // scratch for junction-lane inheritance
    int init_mode;
    int control_mode;
    int collision_behavior;
    int offroad_behavior;
    int traffic_light_behavior;
    int disable_red_light_infractions;
    int disable_stop_sign_infractions;
    float traffic_light_green_min_s;  // training cycles: green ~ U(min, TL_DEFAULT_GREEN_DURATION)
    float traffic_light_yellow_min_s; // training cycles: yellow ~ U(min, max)
    float traffic_light_yellow_max_s;
    int stop_sign_behavior;
    int sdc_controller;
    int non_sdc_controller;
    int non_vehicle_controller;
    int replay_expert_agents;
    int simulation_mode;
    int termination_mode;
    float inactive_agent_threshold;
    int eval_mode;
    int eval_training_render;
    int compute_eval_metrics;
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
    int reward_conditioning;
    int reward_randomization;
    float goal_radius;
    float goal_speed;
    float min_goal_spacing;
    float max_goal_spacing;
    float goal_heading_max_deg;     // 0 disables the successive-waypoint heading constraint
    float conditioning_accel_scale; // eval C_acc: scales the positive accel cap (ACCEL_LONG_LIMIT[1]); 1.0 = paper eval
    float conditioning_speed_scale; // training C_vel ~ X(a): speed cap = base_max_speed_mps * [1/a, a]; 1.5 = paper
    int goal_reach_requires_speed;  // 1: final goal is consumed only below goal speed (paper semantics)
    int num_goals;
    int goal_regen_mode;
    int goal_source;
    int obs_slots_boundary_n;
    int obs_slots_lane_n;
    int obs_slots_partners_n;
    int obs_slots_traffic_controls_n;
    int obs_traffic_light_yellow_remaining; // row gains remaining yellow / traffic_light_yellow_max_s, clipped at 1
    int traffic_lights_enabled;
    int stop_signs_enabled;
    int obs_lane_stride;
    int obs_boundary_stride;
    int obs_stride_merge_segments;
    int obs_slots_lane_kept;
    int obs_slots_boundary_kept;
    int road_dropout_enabled;
    float obs_norm_speed_mps;
    float obs_norm_goal_offset_m;
    float obs_norm_xy_offset_m;
    float obs_norm_veh_length_m;
    float obs_norm_veh_width_m;
    float obs_norm_road_seg_length_m;
    float obs_norm_road_seg_width_m;
    float obs_norm_z_m;
    float eval_perceived_size_margin_m;
    float red_light_train_front_margin_m; // training-only gap the front edge must keep before a red stop line
    float obs_partner_split_length_m; // 0 disables; longer partners are observed as ceil(length / this) equal pieces
    float obs_lane_width_clip_min_m;  // observed lane width floor; 0 = no floor
    float obs_lane_width_clip_max_m;  // observed lane width ceiling; 0 disables both clips
    int cosim_eval_semantics;         // co-sim: apply the eval-only obs/action tweaks without native eval batching
    float eval_standstill_jerk_deadband_mps3; // 0 disables; eval-only standstill deadband on j_long
    float obs_range_traffic_control_m;
    float obs_range_partner_m;
    float obs_range_road_front_m;
    float obs_range_road_behind_m;
    float obs_range_road_side_m;
    float partner_blindness_prob;
    float partner_blindness_prob_min;
    float partner_blindness_trigger_prob;
    int partner_blindness_duration;
    float phantom_braking_prob;
    float phantom_braking_trigger_prob;
    int phantom_braking_duration;
    float episode_partner_blindness_prob;
    float episode_phantom_braking_prob;
    Log log;
    Log *logs;
    int logs_capacity;
    int eval_episode_done;
    int use_exact_episode_seed;
    Rng rng_state;
    Rng seed_stream_rng;
    uint64_t init_seed;
    uint64_t episode_seed;
    uint64_t log_episode_seed;
};

typedef struct {
    float min_val;
    float max_val;
} RewardBound;

static const RewardBound REWARD_BOUNDS[NUM_REWARD_COEFS] = {
    {2.0f, 12.0f},      // REWARD_COEF_GOAL_RADIUS     δ_goal ~ U(2, 12)
    {0.0f, 20.0f},      // REWARD_COEF_GOAL_SPEED      δ_goal-speed ~ U(0, 20)
    {0.0f, 3.0f},       // REWARD_COEF_COLLISION       α_collision ~ U(0, 3)
    {0.0f, 3.0f},       // REWARD_COEF_OFFROAD         α_boundary ~ U(0, 3)
    {0.0f, 0.1f},       // REWARD_COEF_COMFORT         α_comfort ~ U(0, 0.1)
    {2.5e-4f, 2.5e-2f}, // REWARD_COEF_LANE_ALIGN      α_l-align ~ U(2.5e-4, 2.5e-2)
    {0.0f, 1.0f},       // REWARD_COEF_VEL_ALIGN       α_vel-align ~ U(0, 1)
    {2.5e-4f, 7.5e-3f}, // REWARD_COEF_LANE_CENTER     α_l-center ~ U(2.5e-4, 7.5e-3)
    {-0.5f, 0.5f},      // REWARD_COEF_CENTER_BIAS     α_center-bias ~ U(-0.5, 0.5)
    {0.0f, 5e-3f},      // REWARD_COEF_VELOCITY        α_velocity ~ U(0, 5e-3f)
    {2.5e-4f, 7.5e-3f}, // REWARD_COEF_REVERSE         α_reverse ~ U(2.5e-4, 7.5e-3)
    {0.0f, 1.0f},       // REWARD_COEF_STOP_LINE       α_stop-line ~ U(0, 1)
    {0.0f, 5e-5f},      // REWARD_COEF_TIMESTEP        α_timestep ~ U(0, 5e-5f)
    {0.0f, 0.1f},       // REWARD_COEF_OVERSPEED       α_overspeed ~ U(0, 0.1)
    {0.8f, 1.25f},      // REWARD_COEF_THROTTLE        C_throttle
    {0.8f, 1.25f},      // REWARD_COEF_STEER           C_steer
    {0.666f, 1.5f},     // REWARD_COEF_ACC             C_acc
    {0.666f, 1.5f},     // REWARD_COEF_SPEED           C_vel
};

static float compute_euclidean_distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

static float compute_point_to_segment_distance(float px, float py, float x0, float y0, float x1, float y1) {
    // a zero-length segment measures to its start point
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
    if (!(fabsf(heading) < 2.0f * M_PI)) { // fmodf is the identity below one period
        heading = fmodf(heading, 2.0f * M_PI);
    }
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

static float compute_lane_curvature(RoadMapElement *element, int seg_idx) {
    int num_segments = element->segment_size - 1;
    if (num_segments < 2) {
        return 0.0f;
    }
    int next_seg_idx = (seg_idx < num_segments - 1) ? seg_idx + 1 : seg_idx;
    int prev_seg_idx = next_seg_idx - 1;
    float dx = element->x[next_seg_idx] - element->x[prev_seg_idx];
    float dy = element->y[next_seg_idx] - element->y[prev_seg_idx];
    float seg_length = sqrtf(dx * dx + dy * dy);
    if (seg_length < 1e-3f) {
        return 0.0f;
    }
    return compute_heading_diff(element->headings[next_seg_idx], element->headings[prev_seg_idx]) / seg_length;
}

static float sample_uniform(Rng *rng_state, float min_val, float max_val) {
    return min_val + rng_uniform_f32(rng_state) * (max_val - min_val);
}

static float sample_normal(Rng *rng_state, float std_dev) {
    float u1 = fmaxf(rng_uniform_f32(rng_state), 1e-7f);
    float u2 = rng_uniform_f32(rng_state);
    return std_dev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float) M_PI * u2);
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
    // Per-episode perturbation rates: uniform in [floor, configured max]; phantom braking has no floor knob
    env->episode_partner_blindness_prob
        = sample_uniform(&env->rng_state, env->partner_blindness_prob_min, env->partner_blindness_prob);
    env->episode_phantom_braking_prob = sample_uniform(&env->rng_state, 0.0f, env->phantom_braking_prob);
}

static inline void clear_agent_motion(Agent *agent) {
    agent->sim_vx = 0.0f;
    agent->sim_vy = 0.0f;
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
    agent->first_collision_partner_idx = -1;
    agent->stop_sign_target_idx = -1;
    agent->stop_sign_stop_completed = 0;
    agent->stop_sign_last_failed_idx = -1;
    agent->red_light_latched_idx = -1;
    agent->stop_sign_standstill_idx = -1;
    agent->current_lane_idx = -1;
    agent->current_route_idx = 0;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
    agent->steering_angle = 0.0f;
    agent->distance_since_spawn = 0.0f;
    agent->seconds_stopped = 0.0f;
    agent->lane_curvature = 0.0f;
    agent->lane_heading_error_rad = 0.0f;
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
    if (behavior == INFRACTION_BEHAVIOR_STOP) {
        agent->stopped = 1;
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

static void init_dynamics_state_from_log(Drive *env, Agent *agent) {
    // a mid-motion spawn starts from the logged accel and steering; log frames are log_dt apart, not dt
    int step = env->init_step;
    if (step >= agent->trajectory_size) {
        step = agent->trajectory_size - 1;
    }
    if (step < 0) {
        step = 0;
    }
    int next_step = step + 1;
    if (next_step < agent->trajectory_size && agent->log_valid[next_step] == 1) {
        float cos_heading = cosf(agent->log_heading[step]);
        float sin_heading = sinf(agent->log_heading[step]);
        float speed_now = agent->log_velocity_x[step] * cos_heading + agent->log_velocity_y[step] * sin_heading;
        float speed_next
            = agent->log_velocity_x[next_step] * cos_heading + agent->log_velocity_y[next_step] * sin_heading;
        agent->accel_long = clip((speed_next - speed_now) / env->log_dt, ACCEL_LONG_LIMIT[0], ACCEL_LONG_LIMIT[1]);
    }
    float log_yaw_rate = compute_log_yaw_rate(agent, step, env->log_dt);
    agent->accel_lat = clip(agent->sim_speed_signed * log_yaw_rate, ACCEL_LAT_LIMIT[0], ACCEL_LAT_LIMIT[1]);
    float v_eff = fmaxf(fabsf(agent->sim_speed_signed), 1.0f);
    float signed_curvature = log_yaw_rate / v_eff;
    agent->steering_angle
        = clip(atanf(signed_curvature * agent->wheelbase), -MAX_STEERING_ANGLE_RAD, MAX_STEERING_ANGLE_RAD);
}

static inline void project_vector_to_local(
    float world_vec_x,
    float world_vec_y,
    float cos_heading,
    float sin_heading,
    float *local_x,
    float *local_y) {
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

static float compute_lane_progress(
    RoadMapElement *lane,
    float pos_x,
    float pos_y,
    float cos_heading,
    float sin_heading,
    bool align_heading,
    float *out_dist_sq) {
    // arc length (m) of the closest lane point; pass 0 skips segments facing away (self-intersecting lanes)
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

            if (seg_len_sq <= 1e-6f || seg_s <= 1e-6f) {
                continue;
            }
            if (align_heading && pass == 0 && dx * cos_heading + dy * sin_heading < 0.0f) {
                continue;
            }

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
        if (!align_heading || best_dist_sq < 1e30f) {
            break;
        }
    }

    if (out_dist_sq != NULL) {
        *out_dist_sq = best_dist_sq;
    }
    return best_progress;
}

static void update_agent_z(Drive *env, Agent *agent) {
    static const int z_offsets[Z_QUERY_CELL_COUNT][2]
        = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {0, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * Z_QUERY_CELL_COUNT];
    int list_size = get_neighbors_entities(
        env,
        agent->sim_x,
        agent->sim_y,
        entity_list,
        MAX_ENTITIES_PER_CELL * Z_QUERY_CELL_COUNT,
        z_offsets,
        Z_QUERY_CELL_COUNT);
    if (list_size <= 0) {
        return;
    }

    DepthPoint road_neighbors[list_size];
    DepthPoint current_lane_neighbors[list_size];
    int valid_count = 0;
    int current_lane_count = 0;
    for (int i = 0; i < list_size; i++) {
        const RoadMapElement *entity = &env->road_elements[entity_list[i].entity_idx];
        int geometry_idx = entity_list[i].geometry_idx;
        float dz = agent->sim_z - entity->z[geometry_idx];
        if (!(fabsf(dz) < Z_BUFFER)) {
            continue;
        }
        float dx = agent->sim_x - entity->x[geometry_idx];
        float dy = agent->sim_y - entity->y[geometry_idx];
        DepthPoint point = {fabsf(dz), sqrtf(dx * dx + dy * dy + dz * dz), entity->z[geometry_idx]};
        road_neighbors[valid_count++] = point;
        if (entity_list[i].entity_idx == agent->current_lane_idx) {
            current_lane_neighbors[current_lane_count++] = point;
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
    float *out_s_on_lane,
    float *out_heading) {
    // cursor = route index, or lane index on a random forward walk (route NULL); false once the route or walk ends
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
            *out_x = lane->x[start_vertex_idx] + interp_frac * (lane->x[end_vertex_idx] - lane->x[start_vertex_idx]);
            *out_y = lane->y[start_vertex_idx] + interp_frac * (lane->y[end_vertex_idx] - lane->y[start_vertex_idx]);
            *out_z = lane->z[start_vertex_idx] + interp_frac * (lane->z[end_vertex_idx] - lane->z[start_vertex_idx]);
            *out_lane_idx = current_lane_idx;
            *out_cursor_idx = (route != NULL) ? cursor_idx : current_lane_idx;
            *out_s_on_lane = target_s_meters;
            *out_heading = lane->headings[start_vertex_idx];
            return true;
        }

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
        s_on_lane = 0.0f;
    }
    printf("[ERROR] -> route_point_at_distance: walk did not reach target within %d lanes\n", max_lane_hops);
    return false;
}

static bool walk_to_next_goal(
    Drive *env,
    const int *route,
    int route_length,
    int cursor_idx,
    float s_on_lane,
    float spacing_meters,
    float prev_heading,
    float *out_x,
    float *out_y,
    float *out_z,
    int *out_lane_idx,
    int *out_cursor_idx,
    float *out_s_on_lane,
    float *out_heading) {
    // free-roam with goal_heading_max_deg re-samples the spacing until the heading turn fits; the last try is kept
    bool constrain_heading = route == NULL && env->goal_heading_max_deg > 0.0f;
    float max_heading_delta_rad = env->goal_heading_max_deg * (float) M_PI / 180.0f;
    for (int attempt_idx = 0; attempt_idx < GOAL_HEADING_MAX_ATTEMPTS; attempt_idx++) {
        if (attempt_idx > 0) {
            spacing_meters = sample_uniform(&env->rng_state, env->min_goal_spacing, env->max_goal_spacing);
        }
        if (!route_point_at_distance(
                env,
                route,
                route_length,
                cursor_idx,
                s_on_lane,
                spacing_meters,
                out_x,
                out_y,
                out_z,
                out_lane_idx,
                out_cursor_idx,
                out_s_on_lane,
                out_heading)) {
            return false;
        }
        if (!constrain_heading) {
            return true;
        }
        if (fabsf(normalize_heading(*out_heading - prev_heading)) <= max_heading_delta_rad) {
            return true;
        }
    }
    return true;
}

// Goals chained forward from the start cursor (route == NULL: random lane walk); returns how many were placed
static int chain_goals(
    Drive *env,
    const int *route,
    int route_length,
    int start_lane_idx,
    float start_s_on_lane,
    float start_heading,
    const float *goal_spacings_meters,
    int requested_goal_count,
    float *out_goal_x,
    float *out_goal_y,
    float *out_goal_z,
    int *out_goal_lane) {
    int cursor_idx = start_lane_idx; // route index (route mode) or lane index (map mode) of the last landing
    float s_on_lane = start_s_on_lane;
    float heading = start_heading;
    for (int goal_idx = 0; goal_idx < requested_goal_count; goal_idx++) {
        if (!walk_to_next_goal(
                env,
                route,
                route_length,
                cursor_idx,
                s_on_lane,
                goal_spacings_meters[goal_idx],
                heading,
                &out_goal_x[goal_idx],
                &out_goal_y[goal_idx],
                &out_goal_z[goal_idx],
                &out_goal_lane[goal_idx],
                &cursor_idx,
                &s_on_lane,
                &heading)) {
            return goal_idx;
        }
    }
    return requested_goal_count;
}

static inline void set_current_goal(Agent *agent, int goal_idx) {
    agent->current_goal_idx = goal_idx;
    agent->current_goal_x = agent->list_goal_x[goal_idx];
    agent->current_goal_y = agent->list_goal_y[goal_idx];
    agent->current_goal_z = agent->list_goal_z[goal_idx];
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
    // placed goals from slot start_slot_idx on, the other slots zeroed with lane -1
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
    set_current_goal(agent, start_slot_idx);
}

static void compute_new_route(Drive *env, Agent *agent, int current_lane_idx) {
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

    while (route_distance_meters < ROUTE_TARGET_DISTANCE && route_length < MAX_ROUTE_LENGTH) {
        int chosen_exit_lane_idx
            = pick_random_exit_lane(&env->rng_state, road_elements, current_lane_idx, candidate_route, route_length);
        if (chosen_exit_lane_idx == -1) {
            break;
        }
        candidate_route[route_length++] = chosen_exit_lane_idx;
        route_distance_meters += road_elements[chosen_exit_lane_idx].length;
        current_lane_idx = chosen_exit_lane_idx;
    }

    free(agent->route);
    agent->route_length = route_length;
    agent->route = (int *) malloc(route_length * sizeof(int));
    memcpy(agent->route, candidate_route, route_length * sizeof(int));
    agent->current_route_idx = 0;
}

// Nearest route-aligned drivable lane to a goal and the goal's snap onto it; a crossing road's lane is rejected
static int find_goal_lane(
    Drive *env,
    float goal_x,
    float goal_y,
    float route_dir_x,
    float route_dir_y,
    float *out_snap_x,
    float *out_snap_y) {
    *out_snap_x = goal_x;
    *out_snap_y = goal_y;
    if (env->grid_map == NULL) {
        return -1;
    }
    float route_norm = sqrtf(route_dir_x * route_dir_x + route_dir_y * route_dir_y);
    int use_alignment_gate = route_norm > 1e-6f;
    if (use_alignment_gate) {
        route_dir_x /= route_norm;
        route_dir_y /= route_norm;
    }
    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int list_size = get_neighbors_entities(
        env,
        goal_x,
        goal_y,
        entity_list,
        ROAD_QUERY_ENTITY_COUNT,
        ROAD_OFFSETS,
        ROAD_QUERY_CELL_COUNT);
    int best_lane_idx = -1;
    float best_dist_sq = GOAL_LANE_SNAP_MAX_DIST_M * GOAL_LANE_SNAP_MAX_DIST_M;
    for (int i = 0; i < list_size; i++) {
        RoadMapElement *element = &env->road_elements[entity_list[i].entity_idx];
        if (!is_drivable_road_lane(element->type)) {
            continue;
        }
        int geometry_idx = entity_list[i].geometry_idx;
        float seg_start_x = element->x[geometry_idx];
        float seg_start_y = element->y[geometry_idx];
        float seg_dx = element->x[geometry_idx + 1] - seg_start_x;
        float seg_dy = element->y[geometry_idx + 1] - seg_start_y;
        float seg_length_sq = seg_dx * seg_dx + seg_dy * seg_dy;
        if (use_alignment_gate && seg_length_sq > 1e-6f) {
            float seg_length = sqrtf(seg_length_sq);
            float seg_dir_x = seg_dx / seg_length;
            float seg_dir_y = seg_dy / seg_length;
            float cross = route_dir_x * seg_dir_y - route_dir_y * seg_dir_x;
            if (fabsf(cross) > GOAL_LANE_ALIGN_SIN_LIMIT) {
                continue; // crossing road's lane, not the route's
            }
            float dot = route_dir_x * seg_dir_x + route_dir_y * seg_dir_y;
            if (dot < GOAL_LANE_ALIGN_COS_MIN) {
                continue; // oncoming lane (mirror-image direction of the route), not the route's
            }
        }
        float to_goal_x = goal_x - seg_start_x;
        float to_goal_y = goal_y - seg_start_y;
        float t = (seg_length_sq > 1e-6f) ? (to_goal_x * seg_dx + to_goal_y * seg_dy) / seg_length_sq : 0.0f;
        t = clip(t, 0.0f, 1.0f);
        float dx = to_goal_x - t * seg_dx;
        float dy = to_goal_y - t * seg_dy;
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_lane_idx = entity_list[i].entity_idx;
            *out_snap_x = seg_start_x + t * seg_dx;
            *out_snap_y = seg_start_y + t * seg_dy;
        }
    }
    return best_lane_idx;
}

static void generate_new_goals_from_log(Drive *env, Agent *agent) {
    // GT: logged points without a lane; GT_MAP: snapped onto a co-directional lane, hiding the expert's exact pose
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
        float goal_x = agent->log_trajectory_x[t];
        float goal_y = agent->log_trajectory_y[t];
        int goal_lane_idx = -1;
        if (env->goal_source == GOAL_SOURCE_GT_MAP) {
            float log_heading = agent->log_heading[t];
            goal_lane_idx = find_goal_lane(env, goal_x, goal_y, cosf(log_heading), sinf(log_heading), &goal_x, &goal_y);
        }
        agent->list_goal_x[g] = goal_x;
        agent->list_goal_y[g] = goal_y;
        agent->list_goal_z[g] = agent->log_trajectory_z[t];
        agent->list_goal_lane[g] = goal_lane_idx;
    }
    agent->goal_count = num_wp;
    set_current_goal(agent, 0);
}

static bool generate_new_goals_from_route(Drive *env, Agent *agent) {
    // goals by arc length along the route; gigaflow walks a fresh route when it runs out, replay pins its end
    if (agent->route == NULL || agent->route_length <= 0) {
        invalidate_agent(agent);
        agent->removed = 1;
        return false;
    }

    // base = route slot closest by position: current_lane_idx is often off-route and would place goals behind
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
        0.0f,
        goal_spacings_meters,
        env->num_goals,
        goal_x,
        goal_y,
        goal_z,
        goal_lane);

    if (placed_goal_count == env->num_goals) {
        commit_goals(env, agent, goal_x, goal_y, goal_z, goal_lane, placed_goal_count, 0);
        return true;
    }

    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        // Free-roam route source: random-walk a fresh route from the current lane, then retry once.
        int start_lane_idx = (agent->current_lane_idx != -1) ? agent->current_lane_idx : agent->route[base_route_idx];
        compute_new_route(env, agent, start_lane_idx);
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
            0.0f,
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

    // Replay: goals back-aligned so the last one is the route end, where the logged trajectory ends
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
    // 1..num_goals goals on a random lane walk from a uniform map lane; no route is stored
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

    GridMapEntity seed_entity = drivable_lane_candidates[rng_below(&env->rng_state, candidate_count)];
    int seed_lane_idx = seed_entity.entity_idx;
    RoadMapElement *seed_lane = &env->road_elements[seed_lane_idx];
    float seed_x = seed_lane->x[seed_entity.geometry_idx];
    float seed_y = seed_lane->y[seed_entity.geometry_idx];
    float seed_heading = seed_lane->headings[seed_entity.geometry_idx];
    float seed_s_on_lane
        = compute_lane_progress(seed_lane, seed_x, seed_y, cosf(seed_heading), sinf(seed_heading), true, NULL);

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
        seed_heading,
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
    // rolling window: drop the reached goal, append one past the last; 0 when the walk dead-ends
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

    // walk the route when the seed lane is on it, else free-roam
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
    float next_x, next_y, next_z, next_s_on_lane, next_heading;
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
            &next_s_on_lane,
            &next_heading)) {
        return 0;
    }

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
    set_current_goal(agent, 0);
    return 1;
}

static float compute_displacement_error(Agent *agent, int timestep) {
    if (timestep < 0 || timestep >= agent->trajectory_size || !agent->log_valid[timestep]
        || agent->log_trajectory_x[timestep] == INVALID_POSITION
        || agent->log_trajectory_y[timestep] == INVALID_POSITION) {
        return 0.0f;
    }
    return compute_euclidean_distance(
        agent->log_trajectory_x[timestep],
        agent->log_trajectory_y[timestep],
        agent->sim_x,
        agent->sim_y);
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

static inline void compute_agent_corners(const Agent *agent, float corners[4][2]) {
    compute_bounding_box_corners(
        agent->sim_x,
        agent->sim_y,
        agent->cos_heading,
        agent->sin_heading,
        agent->sim_length / 2.0f,
        agent->sim_width / 2.0f,
        corners);
}

static bool check_segment_intersects_aabb(float p0[2], float p1[2], float half_l, float half_w) {
    // Liang-Barsky clip against the origin-centered box [-half_l, half_l] x [-half_w, half_w]
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
    // AB in the box frame at both poses spans a quad; an edge of it hitting the box means the sweep crossed AB
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
    project_point_to_ego_frame(agent, ax, ay, &a_cur[0], &a_cur[1]);
    project_point_to_ego_frame(agent, bx, by, &b_cur[0], &b_cur[1]);
    // all four endpoint positions beyond one face of the box: the swept quad cannot reach it
    if (fminf(fminf(a_prev[0], b_prev[0]), fminf(a_cur[0], b_cur[0])) > half_length
        || fmaxf(fmaxf(a_prev[0], b_prev[0]), fmaxf(a_cur[0], b_cur[0])) < -half_length
        || fminf(fminf(a_prev[1], b_prev[1]), fminf(a_cur[1], b_cur[1])) > half_width
        || fmaxf(fmaxf(a_prev[1], b_prev[1]), fmaxf(a_cur[1], b_cur[1])) < -half_width) {
        return false;
    }
    if (check_segment_intersects_aabb(a_prev, b_prev, half_length, half_width)
        || check_segment_intersects_aabb(a_cur, b_cur, half_length, half_width)
        || check_segment_intersects_aabb(a_prev, a_cur, half_length, half_width)
        || check_segment_intersects_aabb(b_prev, b_cur, half_length, half_width)) {
        return true;
    }

    // Centre moved less than the inscribed radius: a segment over its path already hit the cur test above.
    float center_step_x = agent->sim_x - agent->prev_x;
    float center_step_y = agent->sim_y - agent->prev_y;
    float inscribed_radius = fminf(half_length, half_width);
    if (center_step_x * center_step_x + center_step_y * center_step_y < inscribed_radius * inscribed_radius) {
        return false;
    }

    // Edges all missed: the box is still crossed if its center lies inside the swept quad (one cross-product sign)
    float swept_quad[4][2]
        = {{a_prev[0], a_prev[1]}, {b_prev[0], b_prev[1]}, {b_cur[0], b_cur[1]}, {a_cur[0], a_cur[1]}};
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

// Road-edge segment within Z_BUFFER of the agent's height, crossed by its prev -> cur sweep
static bool road_edge_segment_crosses_agent(const RoadMapElement *edge, int geometry_idx, Agent *agent) {
    if (fabsf(edge->z[geometry_idx] - agent->sim_z) > Z_BUFFER) {
        return false;
    }
    return check_segment_crosses_moving_box(
        edge->x[geometry_idx],
        edge->y[geometry_idx],
        edge->x[geometry_idx + 1],
        edge->y[geometry_idx + 1],
        agent);
}

static bool check_corner_boxes_overlap(float corners_a[4][2], float corners_b[4][2], float axes[4][2]) {
    for (int i = 0; i < 4; i++) {
        float min_a = INFINITY, max_a = -INFINITY;
        float min_b = INFINITY, max_b = -INFINITY;
        for (int j = 0; j < 4; j++) {
            float proj_a = corners_a[j][0] * axes[i][0] + corners_a[j][1] * axes[i][1];
            min_a = fminf(min_a, proj_a);
            max_a = fmaxf(max_a, proj_a);
            float proj_b = corners_b[j][0] * axes[i][0] + corners_b[j][1] * axes[i][1];
            min_b = fminf(min_b, proj_b);
            max_b = fmaxf(max_b, proj_b);
        }
        if (max_a < min_b || min_a > max_b) {
            return false;
        }
    }
    return true;
}

typedef struct {
    float mid_x;
    float mid_y;
    float along_x; // unit travel direction, normal to the stop line
    float along_y;
    float half_len;
} StopLineFrame;

static bool compute_stop_line_frame(TrafficControlElement *tc, StopLineFrame *frame) {
    float line_dx = tc->stop_line[3] - tc->stop_line[0];
    float line_dy = tc->stop_line[4] - tc->stop_line[1];
    float line_len = sqrtf(line_dx * line_dx + line_dy * line_dy);
    if (line_len <= 0.0f) {
        return false;
    }
    float along_x = -line_dy / line_len;
    float along_y = line_dx / line_len;
    if (along_x * cosf(tc->heading) + along_y * sinf(tc->heading) < 0.0f) {
        along_x = -along_x;
        along_y = -along_y;
    }
    frame->mid_x = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
    frame->mid_y = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;
    frame->along_x = along_x;
    frame->along_y = along_y;
    frame->half_len = 0.5f * line_len;
    return true;
}

// directed segment a->b crosses the line in its travel direction (a before it, b at or past); |offset| along the line
static bool segment_crosses_stop_line_frame(
    const StopLineFrame *frame,
    float a_x,
    float a_y,
    float b_x,
    float b_y,
    float *abs_lateral_m) {
    float s_a = (a_x - frame->mid_x) * frame->along_x + (a_y - frame->mid_y) * frame->along_y;
    float s_b = (b_x - frame->mid_x) * frame->along_x + (b_y - frame->mid_y) * frame->along_y;
    if (!(s_a < 0.0f && s_b >= 0.0f)) {
        return false;
    }
    float crossing_frac = s_a / (s_a - s_b);
    float cross_x = a_x + crossing_frac * (b_x - a_x);
    float cross_y = a_y + crossing_frac * (b_y - a_y);
    *abs_lateral_m = fabsf((cross_x - frame->mid_x) * frame->along_y - (cross_y - frame->mid_y) * frame->along_x);
    return true;
}

static bool segment_crosses_stop_line(TrafficControlElement *tc, float a_x, float a_y, float b_x, float b_y) {
    StopLineFrame frame;
    float abs_lateral_m;
    return compute_stop_line_frame(tc, &frame)
        && segment_crosses_stop_line_frame(&frame, a_x, a_y, b_x, b_y, &abs_lateral_m)
        && abs_lateral_m <= frame.half_len + RED_LIGHT_LATERAL_EXTENSION_M;
}

// include_yellow: spawn probe, a yellow line has to be cleared before red just like a red one
static bool check_red_light_footprint(Drive *env, Agent *agent, bool include_yellow) {
    // On red the line may not lie between the tail clearance point and the front edge, moving or standing
    bool eval_semantics = (env->eval_mode && !env->eval_training_render) || env->cosim_eval_semantics;
    float front_offset_m = 0.5f * agent->sim_length + (eval_semantics ? 0.0f : env->red_light_train_front_margin_m);
    float rear_offset_m = 0.5f * agent->sim_length + RED_LIGHT_TAIL_CLEARANCE_M;
    float front_x = agent->sim_x + front_offset_m * agent->cos_heading;
    float front_y = agent->sim_y + front_offset_m * agent->sin_heading;
    float rear_x = agent->sim_x - rear_offset_m * agent->cos_heading;
    float rear_y = agent->sim_y - rear_offset_m * agent->sin_heading;
    // one violation per stay: nothing fires while the footprint is on the latched line, re-armed once it has left
    bool was_latched = agent->red_light_latched_idx != -1;
    if (was_latched) {
        TrafficControlElement *latched_light = &env->traffic_elements[agent->red_light_latched_idx];
        if (segment_crosses_stop_line(latched_light, rear_x, rear_y, front_x, front_y)) {
            return false;
        }
        agent->red_light_latched_idx = -1;
    }
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (env->timestep >= tc->state_size) {
            continue;
        }
        int light_state = tc->states[env->timestep];
        if (light_state != TRAFFIC_CONTROL_STATE_RED
            && !(include_yellow && light_state == TRAFFIC_CONTROL_STATE_YELLOW)) {
            continue;
        }
        float mid_x = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
        float mid_y = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;
        float dx = agent->sim_x - mid_x;
        float dy = agent->sim_y - mid_y;
        if (dx * dx + dy * dy > RED_LIGHT_TRIGGER_DIST_SQ) {
            continue;
        }
        float mid_z = (tc->stop_line[2] + tc->stop_line[5]) * 0.5f;
        if (fabsf(agent->sim_z - mid_z) > Z_BUFFER) {
            continue;
        }
        if (fabsf(compute_heading_diff(agent->sim_heading, tc->heading)) >= RED_LIGHT_ENTER_HEADING_THRESHOLD) {
            continue;
        }
        if (!segment_crosses_stop_line(tc, rear_x, rear_y, front_x, front_y)) {
            continue;
        }
        agent->red_light_latched_idx = i;
        return !was_latched; // an overlapping red line takes over the latch without a second violation
    }
    return false;
}

static bool check_red_light_violation(Drive *env, int agent_idx) {
    return check_red_light_footprint(env, &env->agents[agent_idx], false);
}

// RunStopSign2 acquisition: the path ahead reaches the scaled trigger box within 20 m, never while reversing
static int find_stop_sign_target(Drive *env, Agent *agent) {
    if (agent->sim_speed_signed < -STOP_SIGN_REVERSING_SPEED_MPS) {
        return -1;
    }
    int target_idx = -1;
    float target_dist_sq = STOP_SIGN_PROXIMITY_DIST_SQ;
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_STOP_SIGN || i == agent->stop_sign_last_failed_idx) {
            continue;
        }
        float mid_z = (tc->stop_line[2] + tc->stop_line[5]) * 0.5f;
        if (fabsf(agent->sim_z - mid_z) > Z_BUFFER) {
            continue;
        }
        StopLineFrame frame;
        if (!compute_stop_line_frame(tc, &frame)) {
            continue;
        }
        float dx = agent->sim_x - frame.mid_x;
        float dy = agent->sim_y - frame.mid_y;
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq >= target_dist_sq) {
            continue;
        }
        if (fabsf(compute_heading_diff(agent->sim_heading, tc->heading)) > STOP_SIGN_APPROACH_HEADING_THRESHOLD) {
            continue;
        }
        float along = dx * frame.along_x + dy * frame.along_y;
        float across = dx * frame.along_y - dy * frame.along_x;
        if (along > STOP_SIGN_AFFECTED_BOX_SCALE * STOP_SIGN_TRIGGER_HALF_DEPTH_M
            || fabsf(across) > STOP_SIGN_AFFECTED_BOX_SCALE * frame.half_len) {
            continue;
        }
        target_idx = i;
        target_dist_sq = dist_sq;
    }
    return target_idx;
}

static bool check_agent_in_stop_sign_box(Agent *agent, StopLineFrame *frame) {
    float agent_corners[4][2];
    compute_agent_corners(agent, agent_corners);
    float box_corners[4][2];
    compute_bounding_box_corners(
        frame->mid_x,
        frame->mid_y,
        frame->along_x,
        frame->along_y,
        STOP_SIGN_TRIGGER_HALF_DEPTH_M,
        frame->half_len,
        box_corners);
    float axes[4][2]
        = {{agent->cos_heading, agent->sin_heading},
           {-agent->sin_heading, agent->cos_heading},
           {frame->along_x, frame->along_y},
           {-frame->along_y, frame->along_x}};
    return check_corner_boxes_overlap(agent_corners, box_corners, axes);
}

static bool stop_sign_beyond_proximity(Drive *env, Agent *agent, int element_idx) {
    TrafficControlElement *tc = &env->traffic_elements[element_idx];
    float dx = agent->sim_x - (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
    float dy = agent->sim_y - (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;
    return dx * dx + dy * dy > STOP_SIGN_PROXIMITY_DIST_SQ;
}

// A standstill inside a sign's trigger box is remembered so it still counts once that sign becomes the target
static int find_stop_sign_standstill(Drive *env, Agent *agent) {
    if (agent->sim_speed >= STOP_SIGN_STOP_SPEED_MPS) {
        return -1;
    }
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_STOP_SIGN || stop_sign_beyond_proximity(env, agent, i)) {
            continue;
        }
        float mid_z = (tc->stop_line[2] + tc->stop_line[5]) * 0.5f;
        if (fabsf(agent->sim_z - mid_z) > Z_BUFFER) {
            continue;
        }
        StopLineFrame frame;
        if (compute_stop_line_frame(tc, &frame) && check_agent_in_stop_sign_box(agent, &frame)) {
            return i;
        }
    }
    return -1;
}

// True on the step the centre crosses the target's stop line without a completed stop.
static bool update_stop_sign_state(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    StopLineFrame frame = {0};
    if (agent->stop_sign_last_failed_idx >= 0
        && stop_sign_beyond_proximity(env, agent, agent->stop_sign_last_failed_idx)) {
        agent->stop_sign_last_failed_idx = -1;
    }
    if (agent->stop_sign_standstill_idx >= 0
        && stop_sign_beyond_proximity(env, agent, agent->stop_sign_standstill_idx)) {
        agent->stop_sign_standstill_idx = -1;
    }
    if (agent->stop_sign_target_idx >= 0 && stop_sign_beyond_proximity(env, agent, agent->stop_sign_target_idx)) {
        agent->stop_sign_target_idx = -1;
        agent->stop_sign_stop_completed = 0;
    }
    // an uncompleted target keeps competing: a curved approach latches the neighbour lane's sign first
    if (agent->stop_sign_target_idx < 0 || !agent->stop_sign_stop_completed) {
        int standstill_idx = find_stop_sign_standstill(env, agent);
        if (standstill_idx >= 0) {
            agent->stop_sign_standstill_idx = standstill_idx;
        }
        int target_idx = find_stop_sign_target(env, agent);
        if (target_idx >= 0 && target_idx != agent->stop_sign_target_idx) {
            agent->stop_sign_target_idx = target_idx;
            agent->stop_sign_stop_completed = target_idx == agent->stop_sign_standstill_idx;
        }
        if (agent->stop_sign_target_idx < 0) {
            return false;
        }
    }
    compute_stop_line_frame(&env->traffic_elements[agent->stop_sign_target_idx], &frame);
    if (!agent->stop_sign_stop_completed && agent->sim_speed < STOP_SIGN_STOP_SPEED_MPS
        && check_agent_in_stop_sign_box(agent, &frame)) {
        agent->stop_sign_stop_completed = 1;
    }
    if (agent->stop_sign_stop_completed) {
        return false;
    }
    float abs_lateral_m;
    if (!segment_crosses_stop_line_frame(
            &frame,
            agent->prev_x,
            agent->prev_y,
            agent->sim_x,
            agent->sim_y,
            &abs_lateral_m)
        || abs_lateral_m > frame.half_len + STOP_SIGN_LATERAL_EXTENSION_M) {
        return false;
    }
    agent->stop_sign_last_failed_idx = agent->stop_sign_target_idx;
    agent->stop_sign_target_idx = -1;
    return true;
}

static bool check_obb_collision(const Agent *car1, const Agent *car2) {
    // SAT on the four box axes; no epsilon, so exact boundary contact may flicker across steps
    float car1_top = car1->sim_z + car1->sim_height;
    float car2_top = car2->sim_z + car2->sim_height;
    if (car1_top < car2->sim_z || car2_top < car1->sim_z) {
        return false;
    }

    float car1_corners[4][2];
    compute_agent_corners(car1, car1_corners);
    float car2_corners[4][2];
    compute_agent_corners(car2, car2_corners);

    float axes[4][2]
        = {{car1->cos_heading, car1->sin_heading},
           {-car1->sin_heading, car1->cos_heading},
           {car2->cos_heading, car2->sin_heading},
           {-car2->sin_heading, car2->cos_heading}};
    return check_corner_boxes_overlap(car1_corners, car2_corners, axes);
}

static bool check_moving_obb_collision(Agent *a, Agent *b, float a_disp, float b_disp) {
    // swept prev -> cur boxes, tunnelling-safe
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

    // each box's corners swept prev -> cur in the other's frame, both ways round
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
        compute_agent_corners(other, oc_cur);
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
            project_point_to_ego_frame(ego, oc_cur[k][0], oc_cur[k][1], &q_cur[0], &q_cur[1]);
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

    float ego_disp = compute_euclidean_distance(agent->sim_x, agent->sim_y, agent->prev_x, agent->prev_y);

    // Linear over all actors; pair radius quick-check prunes before OBB SAT.
    for (int i = 0; i < env->num_agents; i++) {
        int index = (i < env->active_agent_count) ? env->active_agent_indices[i]
                                                  : env->static_agent_indices[i - env->active_agent_count];
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
            return index;
        }
    }
    return -1;
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
    compute_agent_corners(agent, agent_corners);

    // front bumper = corners 0 (front-left) and 1 (front-right), in other's frame against its box
    float front_left_local[2], front_right_local[2];
    project_point_to_ego_frame(
        other,
        agent_corners[0][0],
        agent_corners[0][1],
        &front_left_local[0],
        &front_left_local[1]);
    project_point_to_ego_frame(
        other,
        agent_corners[1][0],
        agent_corners[1][1],
        &front_right_local[0],
        &front_right_local[1]);
    if (check_segment_intersects_aabb(
            front_left_local,
            front_right_local,
            other->sim_length / 2.0f,
            other->sim_width / 2.0f)
        || agent->current_lane_idx == -1) {
        return true;
    }

    float edge_dist = fabsf(agent->metrics_array[LANE_DIST_IDX]) + 0.5f * agent->sim_width;
    return edge_dist > MULTI_LANE_THRESHOLD;
}

static inline void compute_pairwise_ttc(Agent *ego, Agent *other) {
    if (other->sim_x == INVALID_POSITION) {
        return;
    }

    float center_dx = other->sim_x - ego->sim_x;
    float center_dy = other->sim_y - ego->sim_y;
    float center_distance = sqrtf(center_dx * center_dx + center_dy * center_dy);
    if (center_distance <= 1e-6f || center_distance >= DEFAULT_DTC) {
        return;
    }

    float forward_cos = (center_dx * ego->cos_heading + center_dy * ego->sin_heading) / center_distance;
    float heading_align_cos = ego->cos_heading * other->cos_heading + ego->sin_heading * other->sin_heading;
    if (forward_cos < DTC_FRONT_CONE_COS_THRESHOLD || heading_align_cos <= DTC_OPPOSITE_HEADING_COS_THRESHOLD) {
        return;
    }

    float ego_corners[4][2];
    float other_corners[4][2];
    compute_agent_corners(ego, ego_corners);
    compute_agent_corners(other, other_corners);

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
        int other_idx = (j < env->active_agent_count) ? env->active_agent_indices[j]
                                                      : env->static_agent_indices[j - env->active_agent_count];
        if (other_idx == agent_idx) {
            continue;
        }
        compute_pairwise_ttc(agent, &env->agents[other_idx]);
    }
}

static float calculate_duration_scaled_violation_score(float violation_timestep_count, float duration_steps, float dt) {
    float duration_s = fmaxf(duration_steps * dt, dt);
    float windows = ceilf(duration_s / METRIC_SCORE_WINDOW_SECONDS);
    float safe_windows = fmaxf(windows, 1.0f);
    float score = 1.0f - (violation_timestep_count / safe_windows);

    return fmaxf(0.0f, fminf(1.0f, score));
}

// True when the first collisions of logged policy agent i and its partner are with each other.
static bool is_shared_policy_collision(Drive *env, int log_idx) {
    int agent_idx = env->active_agent_indices[log_idx];
    int partner_idx = env->agents[agent_idx].first_collision_partner_idx;
    if (partner_idx == -1 || env->agents[partner_idx].first_collision_partner_idx != agent_idx) {
        return false;
    }
    for (int j = 0; j < env->active_agent_count; j++) {
        if (env->active_agent_indices[j] != partner_idx) {
            continue;
        }
        Agent *partner = &env->agents[partner_idx];
        if (partner->is_blind_partner || partner->is_phantom_braker) {
            return false;
        }
        return env->logs[j].collision_rate > 0.0f;
    }
    return false;
}

static void add_log(Drive *env) {
    int safe_timestep = (env->timestep > 0) ? env->timestep : 1;
    Log episode_log = {0};
    for (int i = 0; i < env->active_agent_count; i++) {
        Agent *agent = &env->agents[env->active_agent_indices[i]];
        if (agent->is_blind_partner || agent->is_phantom_braker) {
            continue;
        }
        int offroad = env->logs[i].offroad_rate;
        episode_log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        episode_log.collision_rate += collided;
        int red_light_violations = env->logs[i].red_light_violation_rate;
        episode_log.red_light_violation_rate += red_light_violations;
        int stop_sign_violations = env->logs[i].stop_sign_violation_rate;
        episode_log.stop_sign_violation_rate += stop_sign_violations;
        float collision_share = collided ? 1.0f : 0.0f;
        if (collided && is_shared_policy_collision(env, i)) {
            collision_share = 0.5f;
        }
        float infraction_count = offroad + red_light_violations + stop_sign_violations + collision_share;
        float avg_speed_per_agent = env->logs[i].avg_speed_per_agent;
        episode_log.avg_speed_per_agent += avg_speed_per_agent / safe_timestep;
        int num_goals_reached = env->logs[i].num_goals_reached;
        episode_log.num_goals_reached += num_goals_reached;
        // Score: 1 per agent that reached its full goal set without being removed/stopped.
        if (num_goals_reached >= env->num_goals && !agent->removed && !agent->stopped) {
            episode_log.score += 1.0f;
        }
        if (!offroad && !collided && !red_light_violations && !stop_sign_violations && num_goals_reached < 1) {
            episode_log.dnf_rate += 1.0f;
        }
        episode_log.total_distance_travelled += agent->distance_since_spawn;
        episode_log.total_infractions += infraction_count;
        episode_log.episode_length += env->logs[i].episode_length;
        episode_log.episode_return += env->logs[i].episode_return;
        episode_log.pedestrian_collision_rate += env->logs[i].pedestrian_collision_rate;
        episode_log.pedestrian_crossings += env->logs[i].pedestrian_crossings;
        episode_log.pedestrian_bystanders += env->logs[i].pedestrian_bystanders;
        episode_log.reward_collision += env->logs[i].reward_collision;
        episode_log.reward_offroad += env->logs[i].reward_offroad;
        episode_log.reward_red_light += env->logs[i].reward_red_light;
        episode_log.reward_stop_sign += env->logs[i].reward_stop_sign;
        episode_log.reward_goal += env->logs[i].reward_goal;
        episode_log.reward_lane_align += env->logs[i].reward_lane_align;
        episode_log.reward_lane_center += env->logs[i].reward_lane_center;
        episode_log.reward_comfort += env->logs[i].reward_comfort;
        episode_log.reward_velocity += env->logs[i].reward_velocity;
        episode_log.reward_timestep += env->logs[i].reward_timestep;
        episode_log.reward_reverse += env->logs[i].reward_reverse;
        episode_log.reward_overspeed += env->logs[i].reward_overspeed;
        episode_log.comfort_violation_count += env->logs[i].comfort_violation_count / safe_timestep;
        episode_log.velocity_progress_sum += env->logs[i].velocity_progress_sum / safe_timestep;
        episode_log.lane_center_rate += env->logs[i].lane_center_rate / safe_timestep;
        if (env->compute_eval_metrics) {
            float episode_duration_s = env->logs[i].episode_length * env->dt;
            float reference_progress_distance = fmaxf(PROGRESS_REFERENCE_SPEED_MPS * episode_duration_s, 1.0f);
            env->logs[i].progress_ratio = agent->distance_since_spawn / reference_progress_distance;
            env->logs[i].comfort_score = calculate_duration_scaled_violation_score(
                env->logs[i].comfort_violation_window_count,
                env->logs[i].episode_length,
                env->dt);
            episode_log.at_fault_collision_rate += env->logs[i].at_fault_collision_rate;
            episode_log.progress_ratio += env->logs[i].progress_ratio;
            episode_log.comfort_score += env->logs[i].comfort_score;
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
        }
        episode_log.n += 1;
    }

    int num_log_fields = sizeof(Log) / sizeof(float);
    for (int field_idx = 0; field_idx < num_log_fields; field_idx++) {
        ((float *) &env->log)[field_idx] += ((float *) &episode_log)[field_idx];
    }
    env->log_episode_seed = env->episode_seed;
}

static inline void sample_erratic_flags(Drive *env, Agent *agent) {
    agent->is_blind_partner = env->episode_partner_blindness_prob > 0.0f
        && rng_uniform_f32(&env->rng_state) < env->episode_partner_blindness_prob;
    agent->is_phantom_braker = env->episode_phantom_braking_prob > 0.0f
        && rng_uniform_f32(&env->rng_state) < env->episode_phantom_braking_prob;
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
        for (int i = 0; i < (int) (sizeof(random_coefs) / sizeof(random_coefs[0])); i++) {
            int c = random_coefs[i];
            agent->reward_coefs[c]
                = sample_uniform(&env->rng_state, REWARD_BOUNDS[c].min_val, REWARD_BOUNDS[c].max_val);
        }
        // pinned (GIGAFLOW v_goal); its draw above stays so the RNG stream is unchanged
        agent->reward_coefs[REWARD_COEF_GOAL_SPEED] = env->goal_speed;
        agent->reward_coefs[REWARD_COEF_VELOCITY] = 2.5e-3f;
        agent->reward_coefs[REWARD_COEF_TIMESTEP] = 2.5e-5f;
        agent->reward_coefs[REWARD_COEF_THROTTLE]
            = sample_mixed_uniform(&env->rng_state, REWARD_BOUNDS[REWARD_COEF_THROTTLE].max_val);
        agent->reward_coefs[REWARD_COEF_STEER]
            = sample_mixed_uniform(&env->rng_state, REWARD_BOUNDS[REWARD_COEF_STEER].max_val);
        agent->reward_coefs[REWARD_COEF_ACC]
            = sample_mixed_uniform(&env->rng_state, REWARD_BOUNDS[REWARD_COEF_ACC].max_val);
        agent->reward_coefs[REWARD_COEF_SPEED] = sample_mixed_uniform(&env->rng_state, env->conditioning_speed_scale);
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
        agent->reward_coefs[REWARD_COEF_ACC] = env->conditioning_accel_scale;
        agent->reward_coefs[REWARD_COEF_SPEED] = env->max_speed_mps / env->base_max_speed_mps;
    }
}

typedef struct {
    int steps_green;
    int steps_yellow;
    int steps_red;
    int offset;
} TrafficLightCycle;

// The group removal draw is made once per junction, for the junction's first light.
static int traffic_light_junction_leader(Drive *env, int light_idx) {
    TrafficControlElement *tc = &env->traffic_elements[light_idx];
    for (int i = 0; i < light_idx; i++) {
        TrafficControlElement *other = &env->traffic_elements[i];
        if (other->type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT && other->junction_id == tc->junction_id) {
            return i;
        }
    }
    return light_idx;
}

// Rounded up: a phase never lasts less than its configured duration
static int duration_to_steps(float duration_seconds, float dt) {
    int steps = (int) ceilf(duration_seconds / dt);
    return steps < 1 ? 1 : steps;
}

static TrafficLightCycle sample_traffic_light_cycle(Drive *env, int training_mode) {
    float dur_green, dur_yellow, dur_red;
    if (!training_mode) {
        dur_green = TL_DEFAULT_GREEN_DURATION;
        dur_yellow = TL_DEFAULT_YELLOW_DURATION;
        dur_red = TL_DEFAULT_RED_DURATION;
    } else {
        dur_green = sample_uniform(&env->rng_state, env->traffic_light_green_min_s, TL_DEFAULT_GREEN_DURATION);
        dur_yellow = sample_uniform(&env->rng_state, env->traffic_light_yellow_min_s, env->traffic_light_yellow_max_s);
        dur_red = sample_uniform(&env->rng_state, 0.15f * TL_DEFAULT_RED_DURATION, 5.0f * TL_DEFAULT_RED_DURATION);
    }
    TrafficLightCycle cycle;
    cycle.steps_green = duration_to_steps(dur_green, env->dt);
    cycle.steps_yellow = duration_to_steps(dur_yellow, env->dt);
    cycle.steps_red = duration_to_steps(dur_red, env->dt);
    int cycle_length = cycle.steps_green + cycle.steps_yellow + cycle.steps_red;
    cycle.offset = rng_below(&env->rng_state, cycle_length);
    return cycle;
}

static void fill_traffic_light_states(TrafficControlElement *tc, const TrafficLightCycle *cycle, int fill_steps) {
    int cycle_length = cycle->steps_green + cycle->steps_yellow + cycle->steps_red;
    for (int t = 0; t < fill_steps; t++) {
        int cycle_step = (t + cycle->offset) % cycle_length;
        if (cycle_step < cycle->steps_green) {
            tc->states[t] = TRAFFIC_CONTROL_STATE_GREEN;
        } else if (cycle_step < cycle->steps_green + cycle->steps_yellow) {
            tc->states[t] = TRAFFIC_CONTROL_STATE_YELLOW;
        } else {
            tc->states[t] = TRAFFIC_CONTROL_STATE_RED;
        }
    }
}

static void sample_traffic_light_group_removal(Drive *env, int *group_removed, int enabled) {
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        group_removed[i] = 0;
        if (!enabled || tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || tc->junction_id < 0) {
            continue;
        }
        int leader_idx = traffic_light_junction_leader(env, i);
        if (leader_idx == i) {
            group_removed[i] = rng_uniform_f32(&env->rng_state) < TL_GROUP_REMOVE_PROB;
            continue;
        }
        group_removed[i] = group_removed[leader_idx];
    }
}

static float min_resolved_entry_limit(const Drive *env, const RoadMapElement *road) {
    float min_limit = INFINITY;
    for (int k = 0; k < road->num_entries; k++) {
        int entry_idx = road->entry_lanes[k];
        if (env->lane_limit_resolved[entry_idx]) {
            min_limit = fminf(min_limit, env->lane_speed_limit_mps[entry_idx]);
        }
    }
    return min_limit;
}

static int inherit_junction_speed_limits_pass(Drive *env) {
    int changed = 0;
    for (int i = 0; i < env->num_road_elements; i++) {
        const RoadMapElement *road = &env->road_elements[i];
        if (env->lane_limit_resolved[i] || !is_road_lane(road->type) || road->speed_limit <= 0.0f) {
            continue;
        }
        float min_limit = min_resolved_entry_limit(env, road);
        if (!isfinite(min_limit)) {
            continue;
        }
        env->lane_speed_limit_mps[i] = min_limit;
        env->lane_limit_resolved[i] = 1;
        changed = 1;
    }
    return changed;
}

// Per-episode: one additive offset per speed zone; lanes without a zone inherit the min over their entries.
static void sample_zone_speed_limits(Drive *env) {
    for (int i = 0; i < env->num_road_elements; i++) {
        const RoadMapElement *road = &env->road_elements[i];
        int limit_unknown = is_road_lane(road->type) && road->speed_limit <= 0.0f;
        env->lane_speed_limit_mps[i] = limit_unknown ? UNKNOWN_LANE_SPEED_LIMIT_MPS : road->speed_limit;
    }
    if (env->speed_limit_random_prob <= 0.0f || env->num_speed_zones <= 0) {
        return;
    }
    if (rng_uniform_f32(&env->rng_state) >= env->speed_limit_random_prob) {
        return;
    }
    float delta = env->speed_limit_random_delta_mps;
    for (int zone_idx = 0; zone_idx < env->num_speed_zones; zone_idx++) {
        env->speed_zone_offset_mps[zone_idx] = sample_uniform(&env->rng_state, -delta, delta);
    }
    for (int i = 0; i < env->num_road_elements; i++) {
        const RoadMapElement *road = &env->road_elements[i];
        int has_zone_limit = road->speed_zone_idx >= 0 && road->speed_limit > 0.0f;
        env->lane_limit_resolved[i] = (unsigned char) has_zone_limit;
        if (!has_zone_limit) {
            continue;
        }
        float limit = road->speed_limit + env->speed_zone_offset_mps[road->speed_zone_idx];
        env->lane_speed_limit_mps[i] = clip(limit, env->speed_limit_random_min_mps, env->speed_limit_random_max_mps);
    }
    for (int pass = 0; pass < SPEED_LIMIT_JUNCTION_INHERIT_PASSES; pass++) {
        if (!inherit_junction_speed_limits_pass(env)) {
            break;
        }
    }
}

static void generate_traffic_light_states(Drive *env) {
    int training_mode = !env->eval_mode || env->eval_training_render;
    int disable_all = training_mode && rng_uniform_f32(&env->rng_state) < TL_EPISODE_DISABLE_PROB;
    int element_capacity = env->num_traffic_elements > 0 ? env->num_traffic_elements : 1;
    int group_removed[element_capacity];
    sample_traffic_light_group_removal(env, group_removed, training_mode && !disable_all);
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || tc->states == NULL || tc->state_size <= 0) {
            continue;
        }
        TrafficLightCycle cycle = {0};
        bool removed = disable_all;
        if (!disable_all) {
            cycle = sample_traffic_light_cycle(env, training_mode);
            removed
                = training_mode && (group_removed[i] || rng_uniform_f32(&env->rng_state) < TL_INDIVIDUAL_REMOVE_PROB);
        }
        bool always_green = training_mode && !removed && rng_uniform_f32(&env->rng_state) < TL_ALWAYS_GREEN_PROB;
        if (!removed && !always_green) {
            fill_traffic_light_states(tc, &cycle, tc->state_size);
            continue;
        }
        int constant_state = removed ? TRAFFIC_CONTROL_STATE_OFF : TRAFFIC_CONTROL_STATE_GREEN;
        for (int t = 0; t < tc->state_size; t++) {
            tc->states[t] = constant_state;
        }
    }
}

static bool check_spawn_collision(Drive *env, Agent *tmp_agent) {
    // Inflate the candidate box so agents keep SPAWN_CLEARANCE_M of gap to every neighbor
    Agent inflated = *tmp_agent;
    inflated.sim_length += 2.0f * SPAWN_CLEARANCE_M;
    inflated.sim_width += 2.0f * SPAWN_CLEARANCE_M;
    update_agent_radius(&inflated);

    for (int i = 0; i < env->num_total_agents; i++) {
        Agent *other = &env->agents[i];

        if (other->sim_x == INVALID_POSITION || other->sim_valid != 1) {
            continue;
        }

        float dx = other->sim_x - tmp_agent->sim_x;
        float dy = other->sim_y - tmp_agent->sim_y;
        float max_overlap_dist = inflated.radius + other->radius;

        if (dx * dx + dy * dy > max_overlap_dist * max_overlap_dist) {
            continue;
        }
        if (check_obb_collision(&inflated, other)) {
            return true;
        }
    }

    return false;
}

static bool check_spawn_offroad(Drive *env, Agent *tmp_agent, float edge_clearance_m) {
    Agent scaled = *tmp_agent;
    scaled.sim_length = scaled.sim_length * SPAWN_OFFROAD_SCALE_FACTOR + 2.0f * edge_clearance_m;
    scaled.sim_width = scaled.sim_width * SPAWN_OFFROAD_SCALE_FACTOR + 2.0f * edge_clearance_m;

    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int list_size = get_neighbors_entities(
        env,
        tmp_agent->sim_x,
        tmp_agent->sim_y,
        entity_list,
        ROAD_QUERY_ENTITY_COUNT,
        ROAD_OFFSETS,
        ROAD_QUERY_CELL_COUNT);

    for (int i = 0; i < list_size; i++) {
        RoadMapElement *element = &env->road_elements[entity_list[i].entity_idx];
        if (is_road_edge(element->type)
            && road_edge_segment_crosses_agent(element, entity_list[i].geometry_idx, &scaled)) {
            return true;
        }
    }
    return false;
}

// Ray from (qx, qy) along the unit direction against segment AB; t_out = ray distance of the crossing.
static bool ray_hits_segment(
    float qx,
    float qy,
    float dir_x,
    float dir_y,
    float ax,
    float ay,
    float bx,
    float by,
    float *t_out) {
    float seg_dx = bx - ax;
    float seg_dy = by - ay;
    float denom = dir_x * seg_dy - dir_y * seg_dx;
    if (fabsf(denom) < 1e-6f) {
        return false;
    }
    float to_a_x = ax - qx;
    float to_a_y = ay - qy;
    float t = (to_a_x * seg_dy - to_a_y * seg_dx) / denom;
    float u = (to_a_x * dir_y - to_a_y * dir_x) / denom;
    if (t < 0.0f || u < 0.0f || u > 1.0f) {
        return false;
    }
    *t_out = t;
    return true;
}

// Nearest road-edge crossings of the lateral ray through a lane point: near kerb along +right, far kerb along -right.
static bool find_kerb_distances(
    Drive *env,
    float qx,
    float qy,
    float qz,
    float right_x,
    float right_y,
    float *near_m,
    float *far_m) {
    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int list_size = get_neighbors_entities(
        env,
        qx,
        qy,
        entity_list,
        ROAD_QUERY_ENTITY_COUNT,
        ROAD_OFFSETS,
        ROAD_QUERY_CELL_COUNT);
    *near_m = PEDESTRIAN_KERB_SEARCH_MAX_M;
    *far_m = PEDESTRIAN_KERB_SEARCH_MAX_M;
    for (int i = 0; i < list_size; i++) {
        RoadMapElement *element = &env->road_elements[entity_list[i].entity_idx];
        int geometry_idx = entity_list[i].geometry_idx;
        if (!is_road_edge(element->type) || fabsf(element->z[geometry_idx] - qz) > Z_BUFFER) {
            continue;
        }
        float ax = element->x[geometry_idx];
        float ay = element->y[geometry_idx];
        float bx = element->x[geometry_idx + 1];
        float by = element->y[geometry_idx + 1];
        float t;
        if (ray_hits_segment(qx, qy, right_x, right_y, ax, ay, bx, by, &t) && t < *near_m) {
            *near_m = t;
        }
        if (ray_hits_segment(qx, qy, -right_x, -right_y, ax, ay, bx, by, &t) && t < *far_m) {
            *far_m = t;
        }
    }
    return *near_m < PEDESTRIAN_KERB_SEARCH_MAX_M && *far_m < PEDESTRIAN_KERB_SEARCH_MAX_M;
}

// Lane point `ahead_m` beyond (lane_idx, lane_s_m) and its road edges; false at dead ends or with an edge out of reach.
static bool find_kerb_site(Drive *env, int lane_idx, float lane_s_m, float ahead_m, KerbSite *site) {
    int cursor_idx;
    float lane_heading;
    if (!route_point_at_distance(
            env,
            NULL,
            0,
            lane_idx,
            lane_s_m,
            ahead_m,
            &site->x,
            &site->y,
            &site->z,
            &site->lane_idx,
            &cursor_idx,
            &site->lane_s_m,
            &lane_heading)) {
        return false;
    }
    site->right_x = sinf(lane_heading);
    site->right_y = -cosf(lane_heading);
    return find_kerb_distances(
        env,
        site->x,
        site->y,
        site->z,
        site->right_x,
        site->right_y,
        &site->right_kerb_m,
        &site->left_kerb_m);
}

// Point `offset_m` outside the road edge of a site; side = +1 right kerb, -1 left kerb.
static void kerb_point(const KerbSite *site, float side, float offset_m, float *x, float *y) {
    float kerb_m = side > 0.0f ? site->right_kerb_m : site->left_kerb_m;
    *x = site->x + side * site->right_x * (kerb_m + offset_m);
    *y = site->y + side * site->right_y * (kerb_m + offset_m);
}

// Straight walk along the kerb: the chord between two kerb points of one lane, refused where the road bends.
static bool find_kerb_walk(
    Drive *env,
    ScriptedPedestrian *pedestrian,
    const KerbSite *site,
    float side,
    float kerb_offset_m,
    float *start_x,
    float *start_y,
    float *start_z) {
    float walk_m = sample_uniform(&env->rng_state, PEDESTRIAN_KERB_WALK_MIN_M, PEDESTRIAN_KERB_WALK_MAX_M);
    KerbSite mid_site, end_site;
    if (!find_kerb_site(env, site->lane_idx, site->lane_s_m, 0.5f * walk_m, &mid_site)
        || !find_kerb_site(env, site->lane_idx, site->lane_s_m, walk_m, &end_site)
        || end_site.lane_idx != site->lane_idx) {
        return false;
    }
    float near_x, near_y, mid_x, mid_y, far_x, far_y;
    kerb_point(site, side, kerb_offset_m, &near_x, &near_y);
    kerb_point(&mid_site, side, kerb_offset_m, &mid_x, &mid_y);
    kerb_point(&end_site, side, kerb_offset_m, &far_x, &far_y);
    float bend_m = hypotf(0.5f * (near_x + far_x) - mid_x, 0.5f * (near_y + far_y) - mid_y);
    if (bend_m > PEDESTRIAN_KERB_WALK_BEND_TOLERANCE_M) {
        return false;
    }
    float chord_m = hypotf(far_x - near_x, far_y - near_y);
    bool against_traffic = rng_uniform_f32(&env->rng_state) < PEDESTRIAN_KERB_WALK_AGAINST_TRAFFIC_PROB;
    float travel_sign = against_traffic ? -1.0f : 1.0f;
    *start_x = against_traffic ? far_x : near_x;
    *start_y = against_traffic ? far_y : near_y;
    *start_z = against_traffic ? end_site.z : site->z;
    pedestrian->dir_x = travel_sign * (far_x - near_x) / chord_m;
    pedestrian->dir_y = travel_sign * (far_y - near_y) / chord_m;
    pedestrian->path_m = chord_m;
    pedestrian->climb_per_m = travel_sign * (end_site.z - site->z) / chord_m;
    return true;
}

// Crossing from one kerb to past the other, up to PEDESTRIAN_SLOW_SKEW_MAX_RAD off the perpendicular along one lane.
static bool find_slow_crossing(
    Drive *env,
    ScriptedPedestrian *pedestrian,
    const KerbSite *site,
    float side,
    float kerb_offset_m,
    float *start_x,
    float *start_y,
    float *start_z) {
    float skew_rad = sample_uniform(&env->rng_state, 0.0f, PEDESTRIAN_SLOW_SKEW_MAX_RAD);
    float along_m = (site->right_kerb_m + site->left_kerb_m) * tanf(skew_rad);
    KerbSite ahead_site;
    if (!find_kerb_site(env, site->lane_idx, site->lane_s_m, along_m, &ahead_site)
        || ahead_site.lane_idx != site->lane_idx) {
        return false;
    }
    bool against_traffic = rng_uniform_f32(&env->rng_state) < PEDESTRIAN_SLOW_AGAINST_TRAFFIC_PROB;
    const KerbSite *start_site = against_traffic ? &ahead_site : site;
    const KerbSite *end_site = against_traffic ? site : &ahead_site;
    float end_x, end_y;
    kerb_point(start_site, side, kerb_offset_m, start_x, start_y);
    kerb_point(end_site, -side, PEDESTRIAN_FAR_KERB_OVERSHOOT_M, &end_x, &end_y);
    *start_z = start_site->z;
    pedestrian->path_m = hypotf(end_x - *start_x, end_y - *start_y);
    pedestrian->dir_x = (end_x - *start_x) / pedestrian->path_m;
    pedestrian->dir_y = (end_y - *start_y) / pedestrian->path_m;
    pedestrian->climb_per_m = (end_site->z - start_site->z) / pedestrian->path_m;
    return true;
}

static int sample_pedestrian_mode(Drive *env) {
    static const float mode_probs[PEDESTRIAN_MODE_KERB_STEP]
        = {PEDESTRIAN_POP_IN_CROSS_PROB, PEDESTRIAN_SLOW_CROSS_PROB, PEDESTRIAN_STAND_PROB, PEDESTRIAN_KERB_WALK_PROB};
    float draw = rng_uniform_f32(&env->rng_state);
    for (int mode = 0; mode < PEDESTRIAN_MODE_KERB_STEP; mode++) {
        if (draw < mode_probs[mode]) {
            return mode;
        }
        draw -= mode_probs[mode];
    }
    return PEDESTRIAN_MODE_KERB_STEP;
}

// Start point and leg of the pedestrian's mode at `site`; false where a kerb walk or skewed crossing leaves the lane.
static bool plan_pedestrian_leg(
    Drive *env,
    ScriptedPedestrian *pedestrian,
    const KerbSite *site,
    int lanes_to_kerb,
    float *start_x,
    float *start_y,
    float *start_z) {
    int mode = pedestrian->mode;
    bool is_scenario3 = mode == PEDESTRIAN_MODE_POP_IN_CROSS || mode == PEDESTRIAN_MODE_KERB_STEP;
    float side = 1.0f; // +1 right kerb, -1 left kerb
    float scenario3_speed_mps = 0.0f;
    if (is_scenario3) {
        scenario3_speed_mps = PEDESTRIAN_LB1_SPEED_BASE_MPS + PEDESTRIAN_LB1_SPEED_PER_LANE_MPS * (float) lanes_to_kerb
            + sample_uniform(&env->rng_state, -PEDESTRIAN_SPEED_JITTER_MPS, PEDESTRIAN_SPEED_JITTER_MPS);
    } else if (rng_uniform_f32(&env->rng_state) < PEDESTRIAN_LEFT_KERB_PROB) {
        side = -1.0f;
    }
    float kerb_offset_m = sample_uniform(&env->rng_state, PEDESTRIAN_KERB_OFFSET_MIN_M, PEDESTRIAN_KERB_OFFSET_MAX_M);
    kerb_point(site, side, kerb_offset_m, start_x, start_y);
    *start_z = site->z;
    float into_road_x = -side * site->right_x;
    float into_road_y = -side * site->right_y;
    float road_crossing_m = kerb_offset_m + site->right_kerb_m + site->left_kerb_m + PEDESTRIAN_FAR_KERB_OVERSHOOT_M;
    pedestrian->dir_x = into_road_x;
    pedestrian->dir_y = into_road_y;
    pedestrian->timer_s = 0.0f;
    pedestrian->climb_per_m = 0.0f;
    pedestrian->ramp_s = sample_uniform(&env->rng_state, PEDESTRIAN_RAMP_MIN_S, PEDESTRIAN_RAMP_MAX_S);
    pedestrian->stand_s = sample_uniform(&env->rng_state, PEDESTRIAN_LINGER_MIN_S, PEDESTRIAN_LINGER_MAX_S);

    if (mode == PEDESTRIAN_MODE_POP_IN_CROSS) {
        pedestrian->path_m = road_crossing_m;
        pedestrian->speed_mps = scenario3_speed_mps;
        pedestrian->timer_s = sample_uniform(&env->rng_state, PEDESTRIAN_WAIT_MIN_S, PEDESTRIAN_WAIT_MAX_S);
        return true;
    }
    if (mode == PEDESTRIAN_MODE_SLOW_CROSS) {
        pedestrian->speed_mps
            = sample_uniform(&env->rng_state, PEDESTRIAN_SLOW_SPEED_MIN_MPS, PEDESTRIAN_SLOW_SPEED_MAX_MPS);
        pedestrian->timer_s = sample_uniform(&env->rng_state, 0.0f, PEDESTRIAN_SLOW_WAIT_MAX_S);
        return find_slow_crossing(env, pedestrian, site, side, kerb_offset_m, start_x, start_y, start_z);
    }
    if (mode == PEDESTRIAN_MODE_STAND) {
        float facing_rad = sample_uniform(&env->rng_state, -(float) M_PI, (float) M_PI);
        pedestrian->dir_x = cosf(facing_rad);
        pedestrian->dir_y = sinf(facing_rad);
        pedestrian->path_m = 0.0f;
        pedestrian->speed_mps = 0.0f;
        pedestrian->stand_s
            = sample_uniform(&env->rng_state, PEDESTRIAN_BYSTANDER_STAND_MIN_S, PEDESTRIAN_BYSTANDER_STAND_MAX_S);
        return true;
    }
    if (mode == PEDESTRIAN_MODE_KERB_STEP) {
        float step_m = sample_uniform(&env->rng_state, PEDESTRIAN_KERB_STEP_MIN_M, PEDESTRIAN_KERB_STEP_MAX_M);
        if (rng_uniform_f32(&env->rng_state) < PEDESTRIAN_KERB_STEP_AWAY_PROB) {
            pedestrian->dir_x = -into_road_x;
            pedestrian->dir_y = -into_road_y;
        } else {
            step_m = fminf(step_m, fmaxf(0.0f, kerb_offset_m - PEDESTRIAN_KERB_STEP_EDGE_CLEARANCE_M));
        }
        pedestrian->path_m = step_m;
        pedestrian->speed_mps = scenario3_speed_mps;
        pedestrian->timer_s = sample_uniform(&env->rng_state, PEDESTRIAN_WAIT_MIN_S, PEDESTRIAN_WAIT_MAX_S);
        pedestrian->stand_s
            = sample_uniform(&env->rng_state, PEDESTRIAN_BYSTANDER_STAND_MIN_S, PEDESTRIAN_BYSTANDER_STAND_MAX_S);
        return true;
    }
    pedestrian->speed_mps
        = sample_uniform(&env->rng_state, PEDESTRIAN_KERB_WALK_SPEED_MIN_MPS, PEDESTRIAN_KERB_WALK_SPEED_MAX_MPS);
    return find_kerb_walk(env, pedestrian, site, side, kerb_offset_m, start_x, start_y, start_z);
}

static bool place_scripted_pedestrian(Drive *env, ScriptedPedestrian *pedestrian, const Agent *car) {
    RoadMapElement *car_lane = &env->road_elements[car->current_lane_idx];
    float car_s_m
        = compute_lane_progress(car_lane, car->sim_x, car->sim_y, car->cos_heading, car->sin_heading, true, NULL);
    KerbSite car_site, site;
    if (!find_kerb_site(env, car->current_lane_idx, car_s_m, 0.0f, &car_site)) {
        return false;
    }
    int mode = pedestrian->mode;
    int lanes_to_kerb = (int) (car_site.right_kerb_m / PEDESTRIAN_LB1_LANE_WIDTH_M + 1.0f);
    float ahead_m;
    if (mode == PEDESTRIAN_MODE_POP_IN_CROSS || mode == PEDESTRIAN_MODE_KERB_STEP) {
        ahead_m = PEDESTRIAN_LB1_APPEAR_BASE_M + PEDESTRIAN_LB1_APPEAR_PER_LANE_M * (float) (lanes_to_kerb - 1)
            + sample_uniform(&env->rng_state, -PEDESTRIAN_APPEAR_JITTER_M, PEDESTRIAN_APPEAR_JITTER_M);
    } else if (mode == PEDESTRIAN_MODE_SLOW_CROSS) {
        ahead_m = sample_uniform(&env->rng_state, PEDESTRIAN_SLOW_APPEAR_MIN_M, PEDESTRIAN_SLOW_APPEAR_MAX_M);
    } else {
        ahead_m = sample_uniform(&env->rng_state, PEDESTRIAN_FAR_APPEAR_MIN_M, PEDESTRIAN_FAR_APPEAR_MAX_M);
    }
    if (!find_kerb_site(env, car->current_lane_idx, car_s_m, ahead_m, &site)) {
        return false;
    }
    if (env->num_speed_zones > 0 && env->road_elements[site.lane_idx].speed_zone_idx < 0) {
        return false; // junction lane
    }
    float start_x, start_y, start_z;
    if (!plan_pedestrian_leg(env, pedestrian, &site, lanes_to_kerb, &start_x, &start_y, &start_z)) {
        return false;
    }

    Agent *agent = &env->agents[pedestrian->agent_idx];
    agent->sim_length = sample_uniform(&env->rng_state, PEDESTRIAN_SIZE_MIN_M, PEDESTRIAN_SIZE_MAX_M);
    agent->sim_width = sample_uniform(&env->rng_state, PEDESTRIAN_SIZE_MIN_M, PEDESTRIAN_SIZE_MAX_M);
    agent->sim_height = PEDESTRIAN_HEIGHT_M;
    update_agent_radius(agent);

    // The whole leg as one box: nothing may stand on it when the pedestrian appears
    Agent path_probe = *agent;
    path_probe.sim_x = start_x + pedestrian->dir_x * 0.5f * pedestrian->path_m;
    path_probe.sim_y = start_y + pedestrian->dir_y * 0.5f * pedestrian->path_m;
    path_probe.sim_z = start_z;
    path_probe.sim_heading = atan2f(pedestrian->dir_y, pedestrian->dir_x);
    path_probe.cos_heading = pedestrian->dir_x;
    path_probe.sin_heading = pedestrian->dir_y;
    path_probe.sim_length = fmaxf(pedestrian->path_m, agent->sim_length);
    update_agent_radius(&path_probe);
    copy_pose_to_prev(&path_probe);
    if (check_spawn_collision(env, &path_probe)) {
        return false;
    }

    agent->sim_x = start_x;
    agent->sim_y = start_y;
    agent->sim_z = start_z;
    agent->sim_heading = path_probe.sim_heading;
    agent->cos_heading = pedestrian->dir_x;
    agent->sin_heading = pedestrian->dir_y;
    copy_pose_to_prev(agent);
    clear_agent_motion(agent);
    agent->seconds_stopped = 0.0f;
    agent->sim_valid = 1;
    pedestrian->remaining_m = pedestrian->path_m;
    pedestrian->current_speed_mps = 0.0f;
    pedestrian->state = PEDESTRIAN_STATE_WALKING;
    if (mode == PEDESTRIAN_MODE_STAND) {
        // a stander has been there for a while
        agent->seconds_stopped = sample_uniform(&env->rng_state, 0.0f, MAX_STOPPED_SECONDS);
        pedestrian->state = PEDESTRIAN_STATE_STANDING;
        pedestrian->timer_s = pedestrian->stand_s;
    }
    if (mode == PEDESTRIAN_MODE_KERB_WALK) {
        // already walking when it appears
        pedestrian->current_speed_mps = pedestrian->speed_mps;
        agent->sim_vx = pedestrian->dir_x * pedestrian->speed_mps;
        agent->sim_vy = pedestrian->dir_y * pedestrian->speed_mps;
        update_agent_speed(agent);
    }
    return true;
}

static bool arm_scripted_pedestrian(Drive *env, ScriptedPedestrian *pedestrian) {
    if (env->active_agent_count == 0) {
        return false;
    }
    bool enters_road
        = pedestrian->mode == PEDESTRIAN_MODE_POP_IN_CROSS || pedestrian->mode == PEDESTRIAN_MODE_SLOW_CROSS;
    for (int attempt = 0; attempt < PEDESTRIAN_ARM_ATTEMPTS; attempt++) {
        int target_slot = rng_below(&env->rng_state, env->active_agent_count);
        int target_agent_idx = env->active_agent_indices[target_slot];
        Agent *car = &env->agents[target_agent_idx];
        bool car_eligible = car->sim_valid == 1 && !car->stopped && !car->removed && car->current_lane_idx != -1
            && car->sim_speed >= PEDESTRIAN_TARGET_MIN_SPEED_MPS
            && car->metrics_array[LANE_ANGLE_IDX] >= LANE_ALIGN_COS_THRESHOLD;
        if (!car_eligible || !place_scripted_pedestrian(env, pedestrian, car)) {
            continue;
        }
        pedestrian->target_agent_idx = target_agent_idx;
        env->logs[target_slot].pedestrian_crossings += enters_road ? 1.0f : 0.0f;
        env->logs[target_slot].pedestrian_bystanders += enters_road ? 0.0f : 1.0f;
        return true;
    }
    return false;
}

static void idle_scripted_pedestrian(Drive *env, ScriptedPedestrian *pedestrian) {
    Agent *agent = &env->agents[pedestrian->agent_idx];
    invalidate_agent(agent);
    copy_pose_to_prev(agent);
    pedestrian->target_agent_idx = -1;
    pedestrian->state = PEDESTRIAN_STATE_IDLE;
    pedestrian->mode = sample_pedestrian_mode(env);
    pedestrian->timer_s = sample_uniform(&env->rng_state, PEDESTRIAN_COOLDOWN_MIN_S, PEDESTRIAN_COOLDOWN_MAX_S);
    pedestrian->remaining_m = 0.0f;
}

static void reset_scripted_pedestrians(Drive *env) {
    for (int k = 0; k < env->scripted_pedestrian_slots; k++) {
        idle_scripted_pedestrian(env, &env->pedestrians[k]);
    }
}

static void step_pedestrian_walking(Drive *env, ScriptedPedestrian *pedestrian, Agent *agent) {
    if (pedestrian->timer_s > 0.0f) {
        pedestrian->timer_s -= env->dt; // standing where it appeared
        return;
    }
    float ramp_step_mps = pedestrian->speed_mps * env->dt / pedestrian->ramp_s;
    pedestrian->current_speed_mps = fminf(pedestrian->speed_mps, pedestrian->current_speed_mps + ramp_step_mps);
    agent->sim_vx = pedestrian->dir_x * pedestrian->current_speed_mps;
    agent->sim_vy = pedestrian->dir_y * pedestrian->current_speed_mps;
    update_agent_speed(agent);
    float step_m = fminf(pedestrian->current_speed_mps * env->dt, pedestrian->remaining_m);
    agent->sim_x += pedestrian->dir_x * step_m;
    agent->sim_y += pedestrian->dir_y * step_m;
    agent->sim_z += pedestrian->climb_per_m * step_m;
    pedestrian->remaining_m -= step_m;
    if (pedestrian->remaining_m > 0.0f) {
        return;
    }
    clear_agent_motion(agent);
    pedestrian->state = PEDESTRIAN_STATE_STANDING;
    pedestrian->timer_s = pedestrian->stand_s;
}

static void move_scripted_pedestrians(Drive *env) {
    for (int k = 0; k < env->scripted_pedestrian_slots; k++) {
        ScriptedPedestrian *pedestrian = &env->pedestrians[k];
        Agent *agent = &env->agents[pedestrian->agent_idx];
        copy_pose_to_prev(agent);
        if (pedestrian->state == PEDESTRIAN_STATE_WALKING) {
            step_pedestrian_walking(env, pedestrian, agent);
            continue;
        }
        pedestrian->timer_s -= env->dt;
        if (pedestrian->timer_s > 0.0f) {
            continue;
        }
        if (pedestrian->state == PEDESTRIAN_STATE_STANDING) {
            idle_scripted_pedestrian(env, pedestrian);
        } else {
            arm_scripted_pedestrian(env, pedestrian); // a failed attempt retries next step
        }
    }
}

static bool spawn_agent(Drive *env, int agent_idx, int num_agents) {
    Agent *agent = &env->agents[agent_idx];
    free(agent->route);
    agent->route = NULL;
    agent->id = num_agents;
    agent->type = VEHICLE;
    agent->active_agent = 1;
    agent->mark_as_expert = 0;

    float spawn_length, spawn_width, spawn_edge_clearance_m;
    if (env->eval_mode && !env->eval_training_render) {
        spawn_length = sample_uniform(&env->rng_state, EVAL_SPAWN_LENGTH_MIN_M, EVAL_SPAWN_LENGTH_MAX_M);
        spawn_width = sample_uniform(&env->rng_state, EVAL_SPAWN_WIDTH_MIN_M, EVAL_SPAWN_WIDTH_MAX_M);
        spawn_edge_clearance_m = EVAL_SPAWN_EDGE_CLEARANCE_M;
    } else {
        spawn_length = sample_uniform(&env->rng_state, TRAIN_SPAWN_LENGTH_MIN_M, TRAIN_SPAWN_LENGTH_MAX_M);
        spawn_width = sample_uniform(&env->rng_state, TRAIN_SPAWN_WIDTH_MIN_M, TRAIN_SPAWN_WIDTH_MAX_M);
        spawn_edge_clearance_m = 0.0f;
    }
    if (spawn_width > spawn_length) {
        spawn_width = spawn_length;
    }

    float spawn_x, spawn_y, spawn_z, spawn_heading;
    int start_lane_idx;
    bool is_agent_spawned = false;
    for (int attempt = 0; attempt < MAX_SPAWN_ATTEMPTS; attempt++) {
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
        start_lane_idx = chosen_entity.entity_idx;
        RoadMapElement *start_lane = &env->road_elements[start_lane_idx];
        float lane_heading = start_lane->headings[chosen_entity.geometry_idx];
        float half_lane_width_m = 0.5f * start_lane->widths[chosen_entity.geometry_idx];
        float lateral_offset_m
            = env->spawn_lateral_offset_max_frac * half_lane_width_m * sample_uniform(&env->rng_state, -1.0f, 1.0f);
        float heading_offset_rad
            = env->spawn_heading_max_deg * (float) M_PI / 180.0f * sample_uniform(&env->rng_state, -1.0f, 1.0f);
        spawn_x = start_lane->x[chosen_entity.geometry_idx] - sinf(lane_heading) * lateral_offset_m;
        spawn_y = start_lane->y[chosen_entity.geometry_idx] + cosf(lane_heading) * lateral_offset_m;
        spawn_z = start_lane->z[chosen_entity.geometry_idx];
        spawn_heading = normalize_heading(lane_heading + heading_offset_rad);

        Agent tmp_agent = {0};
        tmp_agent.sim_x = spawn_x;
        tmp_agent.sim_y = spawn_y;
        tmp_agent.sim_z = spawn_z;
        tmp_agent.sim_heading = spawn_heading;
        tmp_agent.cos_heading = cosf(spawn_heading);
        tmp_agent.sin_heading = sinf(spawn_heading);
        copy_pose_to_prev(&tmp_agent); // static spawn pose: the moving-box checks degenerate to static ones
        tmp_agent.sim_length = spawn_length;
        tmp_agent.sim_width = spawn_width;
        tmp_agent.sim_height = SPAWN_HEIGHT_M;
        update_agent_radius(&tmp_agent);
        tmp_agent.current_lane_idx = start_lane_idx;

        if (check_spawn_collision(env, &tmp_agent)) {
            continue;
        }

        if (check_spawn_offroad(env, &tmp_agent, spawn_edge_clearance_m)) {
            continue;
        }

        // the detector's own footprint test, on yellow too, so no spawn starts on a line it would have to clear
        Agent stop_line_probe = tmp_agent;
        stop_line_probe.red_light_latched_idx = -1;
        if (check_red_light_footprint(env, &stop_line_probe, true)) {
            continue;
        }

        is_agent_spawned = true;
        break;
    }

    if (!is_agent_spawned) {
        return false;
    }

    agent->sim_x = spawn_x;
    agent->sim_y = spawn_y;
    agent->sim_z = spawn_z;
    agent->sim_heading = spawn_heading;
    agent->cos_heading = cosf(spawn_heading);
    agent->sin_heading = sinf(spawn_heading);
    copy_pose_to_prev(agent);
    agent->sim_length = spawn_length;
    agent->sim_width = spawn_width;
    agent->sim_height = SPAWN_HEIGHT_M;
    update_agent_radius(agent);
    agent->sim_valid = 1;
    agent->wheelbase = WHEELBASE_LENGTH_RATIO * spawn_length;
    agent->current_lane_idx = start_lane_idx;
    agent->sim_vx = 0.0f;
    agent->sim_vy = 0.0f;
    update_agent_speed(agent);

    if (env->goal_source == GOAL_SOURCE_EXTERNAL) {
        agent->goal_count = 0;
        agent->current_goal_idx = 0;
        return true;
    }
    if (env->goal_source == GOAL_SOURCE_MAP) {
        if (!generate_new_goals_from_map(env, agent)) {
            printf("[GIGAFLOW WARNING] -> Failed to generate map goals for agent %d\n", agent_idx);
            return false;
        }
        return true;
    }

    compute_new_route(env, agent, start_lane_idx);
    return generate_new_goals_from_route(env, agent);
}

static void load_pose_from_log(Agent *agent, int step) {
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
    agent->wheelbase = WHEELBASE_LENGTH_RATIO * agent->sim_length;
    copy_pose_to_prev(agent);
}

// agent may be an inflated copy of self, which is skipped
static bool log_pose_overlaps_created_agent(
    Drive *env,
    Agent *agent,
    const Agent *self,
    const int *created_agent_indices,
    int created_agent_count) {
    for (int i = 0; i < created_agent_count; i++) {
        Agent *other = &env->agents[created_agent_indices[i]];
        if (other == self) {
            continue;
        }
        float dx = other->sim_x - agent->sim_x;
        float dy = other->sim_y - agent->sim_y;
        float max_overlap_dist = agent->radius + other->radius;
        if (dx * dx + dy * dy > max_overlap_dist * max_overlap_dist) {
            continue;
        }
        if (fabsf(other->sim_z - agent->sim_z) > Z_BUFFER) {
            continue;
        }
        if (check_obb_collision(agent, other)) {
            return true;
        }
    }
    return false;
}

static bool spawn_near_drivable_lane(Drive *env, Agent *agent) {
    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int list_size = get_neighbors_entities(
        env,
        agent->sim_x,
        agent->sim_y,
        entity_list,
        ROAD_QUERY_ENTITY_COUNT,
        ROAD_OFFSETS,
        ROAD_QUERY_CELL_COUNT);
    for (int i = 0; i < list_size; i++) {
        int entity_idx = entity_list[i].entity_idx;
        int geometry_idx = entity_list[i].geometry_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];
        if (!is_drivable_road_lane(element->type)) {
            continue;
        }
        if (fabsf(element->z[geometry_idx] - agent->sim_z) > Z_BUFFER) {
            continue;
        }
        float lane_distance = compute_point_to_segment_distance(
            agent->sim_x,
            agent->sim_y,
            element->x[geometry_idx],
            element->y[geometry_idx],
            element->x[geometry_idx + 1],
            element->y[geometry_idx + 1]);
        if (lane_distance <= REPLAY_SPAWN_MAX_LANE_DISTANCE_M) {
            return true;
        }
    }
    return false;
}

static bool replay_spawn_fit_for_control(
    Drive *env,
    Agent *agent,
    const int *created_agent_indices,
    int created_agent_count) {
    if (agent->sim_length > REPLAY_MAX_CONTROLLED_LENGTH_M) {
        return false;
    }
    if (!spawn_near_drivable_lane(env, agent)) {
        return false;
    }
    if (check_spawn_offroad(env, agent, REPLAY_SPAWN_EDGE_CLEARANCE_M)) {
        return false;
    }
    Agent inflated = *agent;
    inflated.sim_length += 2.0f * REPLAY_SPAWN_LONGITUDINAL_CLEARANCE_M;
    update_agent_radius(&inflated);
    return !log_pose_overlaps_created_agent(env, &inflated, agent, created_agent_indices, created_agent_count);
}

static void set_start_position(Drive *env) {
    for (int list_idx = 0; list_idx < env->num_agents; list_idx++) {
        int is_active = list_idx < env->active_agent_count;
        int i = is_active ? env->active_agent_indices[list_idx]
                          : env->static_agent_indices[list_idx - env->active_agent_count];
        Agent *agent = &env->agents[i];
        if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW && agent->removed) {
            continue; // failed init spawn: reset_agent_state would clear the removed latch
        }

        if (env->simulation_mode == SIMULATION_MODE_REPLAY) {
            int step = env->init_step;
            if (step >= agent->trajectory_size) {
                step = agent->trajectory_size - 1;
            }
            if (step < 0) {
                step = 0;
            }

            // invalid at init_step: move_expert places it once its log turns valid
            if (agent->log_valid[step] != 1) {
                invalidate_agent(agent);
                agent->sim_length = agent->log_length[step];
                agent->sim_width = agent->log_width[step];
                agent->sim_height = agent->log_height[step];
                continue;
            }

            load_pose_from_log(agent, step);

            if (agent->type == UNKNOWN) {
                continue;
            }

            if (is_active) {
                agent->sim_vx = agent->log_velocity_x[step];
                agent->sim_vy = agent->log_velocity_y[step];
                update_agent_speed(agent);
            } else {
                clear_agent_motion(agent);
            }
        }

        reset_agent_metrics(env, i);
        reset_agent_state(agent);
        generate_reward_coefs(env, agent);
        if (env->simulation_mode == SIMULATION_MODE_REPLAY && is_active) {
            init_dynamics_state_from_log(env, agent);
        }
    }
}

static bool should_control_agent(Drive *env, int agent_idx) {
    if (env->num_controllable_agents != 0 && env->active_agent_count >= env->num_controllable_agents) {
        return false;
    }

    Agent *agent = &env->agents[agent_idx];

    if (env->control_mode == CONTROL_MODE_SDC_ONLY) {
        return agent_idx == EGO_IDX && agent->route_length != 0;
    }

    if (agent->type != VEHICLE || (agent->mark_as_expert && env->replay_expert_agents)) {
        return false;
    }

    // In REPLAY mode without route data, control agents spawning far enough from their goal
    bool log_goal_source = env->goal_source == GOAL_SOURCE_GT || env->goal_source == GOAL_SOURCE_GT_MAP;
    if (log_goal_source && agent->route_length == 0) {
        float dx = agent->gt_goal_x - agent->log_trajectory_x[env->init_step];
        float dy = agent->gt_goal_y - agent->log_trajectory_y[env->init_step];
        float dz = agent->gt_goal_z - agent->log_trajectory_z[env->init_step];
        return sqrtf(dx * dx + dy * dy + dz * dz) > env->goal_radius;
    }

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
    env->active_agent_count = 0;
    env->static_agent_count = 0;
    env->expert_static_agent_count = 0;
    env->num_agents = 0;

    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        int num_agents_to_create = env->num_controllable_agents;
        int partner_slots = env->cosim_partner_slots;
        int pedestrian_slots = env->scripted_pedestrian_slots;
        int static_slots = partner_slots + pedestrian_slots;

        // the bin's logged agents are not used
        for (int i = 0; i < env->num_total_agents; i++) {
            free_agent(&env->agents[i]);
        }
        free(env->agents);
        env->agents = (Agent *) calloc(num_agents_to_create + static_slots, sizeof(Agent));
        env->num_total_agents = num_agents_to_create + static_slots;
        int successfully_created = 0;
        for (int i = 0; i < num_agents_to_create; i++) {
            if (spawn_agent(env, i, i)) {
                successfully_created++;
            } else {
                // A failed spawn keeps its policy slot, removed and masked until the next c_reset respawns it
                invalidate_agent(&env->agents[i]);
                reset_agent_state(&env->agents[i]);
                env->agents[i].removed = 1;
            }
        }

        env->active_agent_indices = (int *) malloc(num_agents_to_create * sizeof(int));
        env->static_agent_indices = static_slots > 0 ? (int *) malloc(static_slots * sizeof(int)) : NULL;
        env->expert_static_agent_indices = NULL;
        for (int k = 0; k < partner_slots; k++) {
            int partner_idx = num_agents_to_create + k;
            Agent *partner = &env->agents[partner_idx];
            partner->id = partner_idx;
            partner->type = VEHICLE;
            partner->controller = CONTROLLER_STATIC;
            partner->sim_length = COSIM_PARTNER_DEFAULT_LENGTH_M;
            partner->sim_width = COSIM_PARTNER_DEFAULT_WIDTH_M;
            partner->sim_height = COSIM_PARTNER_DEFAULT_HEIGHT_M;
            partner->wheelbase = WHEELBASE_LENGTH_RATIO * partner->sim_length;
            update_agent_radius(partner);
            invalidate_agent(partner);
            env->static_agent_indices[k] = partner_idx;
        }
        env->pedestrians
            = pedestrian_slots > 0 ? (ScriptedPedestrian *) calloc(pedestrian_slots, sizeof(ScriptedPedestrian)) : NULL;
        for (int k = 0; k < pedestrian_slots; k++) {
            int pedestrian_idx = num_agents_to_create + partner_slots + k;
            Agent *pedestrian = &env->agents[pedestrian_idx];
            pedestrian->id = pedestrian_idx;
            pedestrian->type = PEDESTRIAN;
            pedestrian->controller = CONTROLLER_STATIC;
            pedestrian->sim_length = PEDESTRIAN_SIZE_MIN_M;
            pedestrian->sim_width = PEDESTRIAN_SIZE_MIN_M;
            pedestrian->sim_height = PEDESTRIAN_HEIGHT_M;
            update_agent_radius(pedestrian);
            invalidate_agent(pedestrian);
            env->static_agent_indices[partner_slots + k] = pedestrian_idx;
            env->pedestrians[k].agent_idx = pedestrian_idx;
        }
        reset_scripted_pedestrians(env);
        env->static_agent_count = static_slots;

        for (int i = 0; i < num_agents_to_create; i++) {
            env->active_agent_indices[i] = i;
            env->agents[i].controller = resolve_agent_controller(env, i, 1, 0);
        }

        if (successfully_created == 0) {
            fprintf(
                stderr,
                "[ERROR] -> gigaflow spawn created 0 of %d agents on map %s (no drivable lanes or missing lane "
                "connectivity?)\n",
                num_agents_to_create,
                env->map_name);
        }
        env->active_agent_count = num_agents_to_create;
        env->num_agents = num_agents_to_create + static_slots;

        return;
    }

    bool is_log_replay = (env->control_mode == CONTROL_MODE_SDC_ONLY);
    // Eval and log-replay keep the whole scene; training stays capped at num_max_agents.
    int max_agents = (env->eval_mode || is_log_replay) ? env->num_total_agents : env->num_max_agents;

    env->active_agent_indices = (int *) malloc(max_agents * sizeof(int));
    env->static_agent_indices = (int *) malloc(max_agents * sizeof(int));
    env->expert_static_agent_indices = (int *) malloc(max_agents * sizeof(int));
    int *created_agent_indices = (int *) malloc(max_agents * sizeof(int));
    // Log-replay keeps every logged track; self-play drops duplicate and road-edge spawns.
    bool filter_spawns = !is_log_replay;

    int created_agent_count = 0;
    for (int i = 0; i < env->num_total_agents && created_agent_count < max_agents; i++) {
        Agent *agent = &env->agents[i];

        if (agent->log_valid[env->init_step] != 1 && !is_log_replay) {
            continue;
        }

        // create_controllable_types keeps the vehicles only
        bool should_create = is_log_replay || env->init_mode == INIT_MODE_CREATE_ALL_VALID || agent->type == VEHICLE;
        if (!should_create) {
            continue;
        }

        if (filter_spawns) {
            load_pose_from_log(agent, env->init_step);
        }
        if (filter_spawns
            && log_pose_overlaps_created_agent(env, agent, agent, created_agent_indices, created_agent_count)) {
            continue;
        }
        created_agent_indices[created_agent_count] = i;
        created_agent_count++;
    }

    // Control decisions need every created agent's pose, so they run after creation
    for (int created_idx = 0; created_idx < created_agent_count; created_idx++) {
        int i = created_agent_indices[created_idx];
        Agent *agent = &env->agents[i];
        bool is_controlled = should_control_agent(env, i);
        if (is_controlled && filter_spawns && agent->type == VEHICLE
            && !replay_spawn_fit_for_control(env, agent, created_agent_indices, created_agent_count)) {
            is_controlled = false;
        }

        if (is_controlled) {
            env->active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->agents[i].active_agent = 1;
            env->agents[i].controller = resolve_agent_controller(env, i, 1, 0);
        } else {
            env->static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->agents[i].active_agent = 0;
            int replay_by_default = is_log_replay || (env->agents[i].mark_as_expert == 1 && env->replay_expert_agents)
                || env->active_agent_count == env->num_max_agents;
            env->agents[i].controller = resolve_agent_controller(env, i, 0, replay_by_default);
            if (env->agents[i].controller == CONTROLLER_REPLAY) {
                env->expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->agents[i].mark_as_expert = 1;
            }
        }
    }

    env->num_agents = env->active_agent_count + env->static_agent_count;
    free(created_agent_indices);
    // Uncreated tracks must not linger as ghosts: set_start_position re-poses only created agents
    for (int i = 0; i < env->num_total_agents; i++) {
        invalidate_agent(&env->agents[i]);
    }

    if (env->num_controllable_agents > 0 && env->active_agent_count != env->num_controllable_agents) {
        printf(
            "ERROR Between my_shared and init : Mismatch in active agent count: %d vs %d\n",
            env->active_agent_count,
            env->num_controllable_agents);
    }

    return;
}

void move_expert(Drive *env, int agent_idx) {
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
        agent->wheelbase = WHEELBASE_LENGTH_RATIO * agent->sim_length;
    }
    agent->sim_vx = agent->log_velocity_x[t];
    agent->sim_vy = agent->log_velocity_y[t];
    update_agent_speed(agent);

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
    set_start_position(env);
    int collided_with_indices[env->active_agent_count];
    for (int i = 0; i < env->active_agent_count; ++i) {
        collided_with_indices[i] = -1;
    }
    // replay the logs: a static agent that a logged policy agent runs into starts invalid
    for (int t = 0; t < env->scenario_length; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            move_expert(env, env->active_agent_indices[i]);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->agents[expert_idx].sim_x == INVALID_POSITION) {
                continue;
            }
            move_expert(env, expert_idx);
        }
        for (int i = 0; i < env->active_agent_count; i++) {
            if (collided_with_indices[i] == -1) {
                collided_with_indices[i] = collision_check(env, env->active_agent_indices[i]);
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        int other_idx = collided_with_indices[i];
        if (other_idx == -1 || env->agents[other_idx].active_agent) {
            continue;
        }
        env->agents[other_idx].log_trajectory_x[0] = INVALID_POSITION;
        env->agents[other_idx].log_trajectory_y[0] = INVALID_POSITION;
        env->agents[other_idx].log_valid[0] = 0;
    }
    env->timestep = 0;
}

// 0 on success, -1 when the map cannot be loaded or is unusable (reason on stderr); the env must not be used then.
int init(Drive *env) {
    env->timestep = 0;
    env->first_reset_pending = 1;
    if (load_map_binary(env->map_name, env) != 0) {
        fprintf(stderr, "[ERROR] -> Failed to load map binary: %s\n", env->map_name);
        return -1;
    }
    struct SharedMapData *shared = env->use_map_cache ? map_cache_lookup(env) : NULL;
    if (shared != NULL) {
        // cache hit: keep the per-env agents and traffic controls, borrow the shared geometry
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
        if (init_grid_map(env) != 0) {
            fprintf(stderr, "[ERROR] -> Failed to build grid map for map: %s\n", env->map_name);
            return -1;
        }
        env->grid_map->vision_range = compute_vision_range(env);
        init_neighbor_offsets(env);
        if (env->use_map_cache) {
            env->shared_map = map_cache_store(env);
        }
    }
    if (env->simulation_mode == SIMULATION_MODE_REPLAY && (!isfinite(env->log_dt) || env->log_dt <= 0.0f)) {
        fprintf(stderr, "[ERROR] -> Replay map %s has invalid log_dt %f\n", env->map_name, (double) env->log_dt);
        return -1;
    }
    if (env->use_neighbor_cache && env->grid_map->neighbor_cache_pool_idx == NULL) {
        cache_neighbor_offsets(env);
    }
    if (!env->use_neighbor_cache) {
        env->obs_neighbor_scratch = (GridMapEntity *) malloc(env->grid_map->total_entities * sizeof(GridMapEntity));
    }
    env->road_dropout_enabled = (env->obs_slots_lane_kept < env->obs_slots_lane_n)
        || (env->obs_slots_boundary_kept < env->obs_slots_boundary_n);
    if (env->speed_limit_random_prob > 0.0f && env->num_speed_zones <= 0) {
        fprintf(stderr, "[ERROR] -> speed_limit_random_prob > 0 but map %s carries no speed zones\n", env->map_name);
        return -1;
    }
    env->lane_speed_limit_mps = (float *) malloc(env->num_road_elements * sizeof(float));
    env->lane_limit_resolved = (unsigned char *) calloc(env->num_road_elements, sizeof(unsigned char));
    env->speed_zone_offset_mps
        = (float *) malloc((env->num_speed_zones > 0 ? env->num_speed_zones : 1) * sizeof(float));
    begin_episode_rng(env);
    sample_zone_speed_limits(env);
    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        int steps
            = env->scenario_length + 1; // the terminal step's obs and red-light check read states[scenario_length]
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
                        fprintf(
                            stderr,
                            "[ERROR] -> Failed to allocate traffic light states for map: %s\n",
                            env->map_name);
                        return -1;
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

    for (int i = 0; i < env->active_agent_count; i++) {
        Agent *agent = &env->agents[env->active_agent_indices[i]];
        if (env->goal_source == GOAL_SOURCE_GT || env->goal_source == GOAL_SOURCE_GT_MAP) {
            generate_new_goals_from_log(env, agent);
        } else if (env->goal_source == GOAL_SOURCE_MAP) {
            generate_new_goals_from_map(env, agent);
        } else if (env->goal_source == GOAL_SOURCE_ROUTE) {
            generate_new_goals_from_route(env, agent);
        }
    }
    return 0;
}

void c_close(Drive *env) {
    free(env->active_agent_indices);
    free(env->logs);
    free(env->lane_speed_limit_mps);
    free(env->speed_zone_offset_mps);
    free(env->lane_limit_resolved);
    free(env->obs_neighbor_scratch);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->pedestrians);
    free_loaded_map_data(env);
}

static inline int traffic_control_feature_count(const Drive *env) {
    return TRAFFIC_CONTROL_FEATURES
        + (env->obs_traffic_light_yellow_remaining ? TRAFFIC_CONTROL_YELLOW_REMAINING_FEATURES : 0);
}

static int compute_observation_size(Drive *env) {
    return EGO_FEATURES + PARTNER_ROW_FEATURES * env->obs_slots_partners_n
        + LANE_ROW_FEATURES * env->obs_slots_lane_kept + BOUNDARY_FEATURES * env->obs_slots_boundary_kept
        + traffic_control_feature_count(env) * env->obs_slots_traffic_controls_n + OBS_VALID_COUNT_FEATURES
        + env->reward_conditioning * NUM_REWARD_COEFS + env->num_goals * GOAL_FEATURES;
}

int get_track_id_or_placeholder(Drive *env, int agent_idx) {
    for (int k = 0; k < env->num_tracks_to_predict; k++) {
        if (env->tracks_to_predict[k] == agent_idx) {
            return agent_idx;
        }
    }
    return -1;
}

// include_static: also write the static partner slots after the active agents (co-sim BEV/debug).
void c_get_global_agent_state(
    Drive *env,
    int include_static,
    float *x_out,
    float *y_out,
    float *z_out,
    float *heading_out,
    int *id_out,
    float *length_out,
    float *width_out) {
    int count = include_static ? env->num_agents : env->active_agent_count;
    for (int i = 0; i < count; i++) {
        int agent_idx = (i < env->active_agent_count) ? env->active_agent_indices[i]
                                                      : env->static_agent_indices[i - env->active_agent_count];
        Agent *agent = &env->agents[agent_idx];

        // World coordinates: add the world means back
        x_out[i] = agent->sim_x + env->world_mean_x;
        y_out[i] = agent->sim_y + env->world_mean_y;
        z_out[i] = agent->sim_z;
        heading_out[i] = agent->sim_heading;
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        length_out[i] = agent->sim_length;
        width_out[i] = agent->sim_width;
    }
}

// Squared distance from the agent to lane segment seg_idx; cross_out carries the left/right sign.
static inline float lane_segment_distance_sq(
    const RoadMapElement *element,
    int seg_idx,
    float agent_x,
    float agent_y,
    float *cross_out) {
    float seg_start_x = element->x[seg_idx];
    float seg_start_y = element->y[seg_idx];
    float seg_end_x = element->x[seg_idx + 1];
    float seg_end_y = element->y[seg_idx + 1];
    float seg_dx = seg_end_x - seg_start_x;
    float seg_dy = seg_end_y - seg_start_y;
    float seg_length_sq = seg_dx * seg_dx + seg_dy * seg_dy;
    float to_agent_x = agent_x - seg_start_x;
    float to_agent_y = agent_y - seg_start_y;
    float cross = seg_dx * to_agent_y - seg_dy * to_agent_x;
    *cross_out = cross;
    if (seg_length_sq <= 1e-6f) {
        return to_agent_x * to_agent_x + to_agent_y * to_agent_y;
    }
    float t = (to_agent_x * seg_dx + to_agent_y * seg_dy) / seg_length_sq;
    if (t <= 0.0f) {
        return to_agent_x * to_agent_x + to_agent_y * to_agent_y;
    }
    if (t >= 1.0f) {
        float dxe = agent_x - seg_end_x;
        float dye = agent_y - seg_end_y;
        return dxe * dxe + dye * dye;
    }
    return (cross * cross) / seg_length_sq;
}

// Closest segment in [seg_first, seg_last], lowest index on ties; min_dist_sq stays 1e18 if none passes the z filter.
static void scan_lane_segment_range(
    const RoadMapElement *element,
    const Agent *agent,
    int seg_first,
    int seg_last,
    int *closest_seg_idx,
    float *min_dist_sq,
    float *closest_cross) {
    *closest_seg_idx = 0;
    *min_dist_sq = 1e18f;
    *closest_cross = 0.0f;
    for (int seg_idx = seg_first; seg_idx <= seg_last; seg_idx++) {
        if (fabsf(element->z[seg_idx] - agent->sim_z) > Z_BUFFER) {
            continue;
        }
        float cross;
        float dist_sq = lane_segment_distance_sq(element, seg_idx, agent->sim_x, agent->sim_y, &cross);
        if (dist_sq < *min_dist_sq) {
            *min_dist_sq = dist_sq;
            *closest_seg_idx = seg_idx;
            *closest_cross = cross;
        }
    }
}

// Distance from the agent to the nearest outer edge of its 5x5 road query window, meters.
static float road_query_window_margin_m(const Drive *env, const Agent *agent) {
    float rel_x = agent->sim_x - env->grid_map->top_left_x;
    float rel_y = agent->sim_y - env->grid_map->bottom_right_y;
    float cell_x = (float) (int) (rel_x / GRID_CELL_SIZE);
    float cell_y = (float) (int) (rel_y / GRID_CELL_SIZE);
    float margin_x = fminf(rel_x - (cell_x - 2.0f) * GRID_CELL_SIZE, (cell_x + 3.0f) * GRID_CELL_SIZE - rel_x);
    float margin_y = fminf(rel_y - (cell_y - 2.0f) * GRID_CELL_SIZE, (cell_y + 3.0f) * GRID_CELL_SIZE - rel_y);
    return fminf(margin_x, margin_y);
}

// Unlisted segments lie beyond window_margin - half_length; returns false when all segments exceed max_distance_m.
static bool find_closest_lane_segment(
    const RoadMapElement *element,
    const Agent *agent,
    const LaneQueryEntry *entry,
    float window_margin_m,
    float max_distance_m,
    int *closest_seg_idx,
    float *min_dist_sq,
    float *closest_cross) {
    scan_lane_segment_range(
        element,
        agent,
        entry->seg_first,
        entry->seg_last,
        closest_seg_idx,
        min_dist_sq,
        closest_cross);
    float unlisted_reach_m = window_margin_m - element->max_segment_half_length_m - LANE_SCAN_GUARD_MARGIN_M;
    float listed_dist_m = *min_dist_sq < 1e18f ? sqrtf(*min_dist_sq) : 1e9f;
    if (listed_dist_m < unlisted_reach_m) {
        return true;
    }
    if (unlisted_reach_m > max_distance_m && listed_dist_m > max_distance_m) {
        return false;
    }
    scan_lane_segment_range(element, agent, 0, element->segment_size - 2, closest_seg_idx, min_dist_sq, closest_cross);
    return true;
}

// Road edges listed by one cell's 5x5 window against the agent's swept box.
static bool crosses_road_edge_in_window(Drive *env, Agent *agent, int cell_idx) {
    GridMap *grid = env->grid_map;
    const LaneQueryEntry *entries = &grid->lane_query_pool[grid->lane_query_offset[cell_idx]];
    int entry_count = grid->lane_query_offset[cell_idx + 1] - grid->lane_query_offset[cell_idx];
    for (int i = 0; i < entry_count; i++) {
        RoadMapElement *element = &env->road_elements[entries[i].entity_idx];
        if (is_road_edge(element->type) && road_edge_segment_crosses_agent(element, entries[i].seg_first, agent)) {
            return true;
        }
    }
    return false;
}

static void find_lane_and_offroad(
    Drive *env,
    Agent *agent,
    int cell_idx,
    bool *is_offroad_out,
    int *lane_idx_out,
    int *lane_seg_idx_out,
    float *signed_lane_distance_out,
    float *lane_heading_out) {
    bool is_offroad = false;

    float best_score = 1e9f;
    int lane_idx = -1;
    int lane_seg_idx = 0;
    float signed_lane_distance = 0.0f, lane_heading = 0.0f;

    GridMap *grid = env->grid_map;
    const LaneQueryEntry *entries = NULL;
    int entry_count = 0;
    float window_margin_m = 0.0f;
    if (grid->window_entity_count[cell_idx] <= 0) {
        is_offroad = true;
    } else {
        entries = &grid->lane_query_pool[grid->lane_query_offset[cell_idx]];
        entry_count = grid->lane_query_offset[cell_idx + 1] - grid->lane_query_offset[cell_idx];
        window_margin_m = road_query_window_margin_m(env, agent);
    }

    float max_distance_threshold = 3.0f * agent->sim_width;

    // Window candidates in grid order: road edges (offroad crossing) and drivable lanes (association)
    for (int i = 0; i < entry_count; i++) {
        const LaneQueryEntry *entry = &entries[i];
        int entity_idx = entry->entity_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];

        if (is_road_edge(element->type)) {
            is_offroad = road_edge_segment_crosses_agent(element, entry->seg_first, agent);
            if (is_offroad) {
                break;
            }
            continue;
        }

        int closest_seg_idx;
        float min_dist_sq, closest_cross;
        if (!find_closest_lane_segment(
                element,
                agent,
                entry,
                window_margin_m,
                max_distance_threshold,
                &closest_seg_idx,
                &min_dist_sq,
                &closest_cross)) {
            continue;
        }
        float abs_dist = sqrtf(min_dist_sq);
        if (abs_dist > max_distance_threshold) {
            continue;
        }
        float signed_dist = (closest_cross >= 0.0f) ? -abs_dist : abs_dist; // left of the lane is negative

        // Multi-segment lane heading (more weight on center segment)
        float avg_lane_heading = 0.0f;
        float total_weight = 0.0f;
        int seg_start = (closest_seg_idx > 0) ? (closest_seg_idx - 1) : closest_seg_idx;
        int seg_end
            = (closest_seg_idx < element->segment_size - 2) ? (closest_seg_idx + 1) : (element->segment_size - 2);
        for (int seg_idx = seg_start; seg_idx <= seg_end; seg_idx++) {
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
            lane_seg_idx = closest_seg_idx;
            signed_lane_distance = signed_dist;
            lane_heading = avg_lane_heading;
        }
    }

    // A step longer than the window margin sweeps road edges that only the previous cell's window lists
    float step_m = compute_euclidean_distance(agent->prev_x, agent->prev_y, agent->sim_x, agent->sim_y);
    if (!is_offroad && step_m + agent->radius > window_margin_m) {
        int prev_cell_idx = get_grid_index(env, agent->prev_x, agent->prev_y);
        is_offroad = prev_cell_idx != -1 && prev_cell_idx != cell_idx
            && crosses_road_edge_in_window(env, agent, prev_cell_idx);
    }

    *is_offroad_out = is_offroad;
    *lane_idx_out = lane_idx;
    *lane_seg_idx_out = lane_seg_idx;
    *signed_lane_distance_out = signed_lane_distance;
    *lane_heading_out = lane_heading;
}

// lane_idx -1 clears the association
static void set_lane_association(
    Drive *env,
    Agent *agent,
    int lane_idx,
    int lane_seg_idx,
    float signed_lane_distance,
    float lane_heading) {
    agent->current_lane_idx = lane_idx;
    if (lane_idx == -1) {
        agent->metrics_array[LANE_DIST_IDX] = LANE_DISTANCE_NORMALIZATION;
        agent->metrics_array[LANE_ANGLE_IDX] = 0.0f;
        agent->lane_heading_error_rad = 0.0f;
        agent->lane_curvature = 0.0f;
        return;
    }
    agent->metrics_array[LANE_DIST_IDX] = signed_lane_distance;
    float theta_f = compute_heading_diff(agent->sim_heading, lane_heading);
    agent->metrics_array[LANE_ANGLE_IDX] = cosf(theta_f);
    agent->lane_heading_error_rad = theta_f;
    agent->lane_curvature = compute_lane_curvature(&env->road_elements[lane_idx], lane_seg_idx);
}

// Co-sim: lane association for an externally set pose, without compute_metrics' infractions, goals and logs
static void refresh_lane_association(Drive *env, Agent *agent) {
    bool has_grid_pose = env->grid_map != NULL && agent->sim_x != INVALID_POSITION;
    int cell_idx = has_grid_pose ? get_grid_index(env, agent->sim_x, agent->sim_y) : -1;
    if (cell_idx == -1) {
        set_lane_association(env, agent, -1, 0, 0.0f, 0.0f);
        return;
    }
    bool is_offroad;
    int lane_idx, lane_seg_idx;
    float signed_lane_distance, lane_heading;
    find_lane_and_offroad(
        env,
        agent,
        cell_idx,
        &is_offroad,
        &lane_idx,
        &lane_seg_idx,
        &signed_lane_distance,
        &lane_heading);
    set_lane_association(env, agent, is_offroad ? -1 : lane_idx, lane_seg_idx, signed_lane_distance, lane_heading);
}

// Co-sim setters return 0 on success, -1 on invalid external input (the binding raises).
int c_set_agent_states(
    Drive *env,
    int count,
    const int *idx,
    const float *x,
    const float *y,
    const float *z,
    const float *heading,
    const float *vx,
    const float *vy,
    const float *yaw_rate,
    const float *accel_long,
    const float *seconds_stopped,
    const float *steering) {
    for (int k = 0; k < count; k++) {
        int agent_idx = idx[k];
        if (agent_idx < 0 || agent_idx >= env->num_total_agents) {
            return -1;
        }
        if (!isfinite(x[k]) || !isfinite(y[k]) || !isfinite(z[k]) || !isfinite(heading[k]) || !isfinite(vx[k])
            || !isfinite(vy[k])) {
            return -1;
        }
        if (seconds_stopped && !(seconds_stopped[k] >= 0.0f)) {
            return -1;
        }
        if (steering && !isfinite(steering[k])) {
            return -1;
        }
        Agent *agent = &env->agents[agent_idx];
        agent->sim_x = x[k] - env->world_mean_x;
        agent->sim_y = y[k] - env->world_mean_y;
        agent->sim_z = z[k];
        agent->sim_heading = heading[k];
        agent->cos_heading = cosf(agent->sim_heading);
        agent->sin_heading = sinf(agent->sim_heading);
        // teleport: prev follows, else the swept offroad and goal checks would sweep the jump
        copy_pose_to_prev(agent);
        agent->sim_vx = vx[k];
        agent->sim_vy = vy[k];
        update_agent_speed(agent);
        agent->sim_valid = 1;
        // the external sim owns this agent: a removed/stopped latch would make move_dynamics re-invalidate it
        agent->removed = 0;
        agent->stopped = 0;

        // yaw rate and accel come from the external sim: the agent's own state here is a throwaway dummy-action step
        if (steering) {
            agent->steering_angle = clip(steering[k], -STEERING_LIMIT, STEERING_LIMIT);
            agent->accel_lat
                = agent->sim_speed_signed * agent->sim_speed_signed * tanf(agent->steering_angle) / agent->wheelbase;
        } else {
            float speed_for_steering = fmaxf(fabsf(agent->sim_speed_signed), 1.0f);
            float derived_steering = atanf(yaw_rate[k] * agent->wheelbase / speed_for_steering);
            agent->steering_angle = clip(derived_steering, -STEERING_LIMIT, STEERING_LIMIT);
            agent->accel_lat = agent->sim_speed_signed * yaw_rate[k];
        }
        agent->accel_long = accel_long[k];
        refresh_lane_association(env, agent);

        // NULL keeps c_step's own accumulation (co-sim default); an array injects stopped-time as state.
        if (seconds_stopped) {
            agent->seconds_stopped = seconds_stopped[k];
        }
    }
    return 0;
}

int c_set_agent_sizes(Drive *env, int count, const int *idx, const float *length, const float *width) {
    for (int k = 0; k < count; k++) {
        int agent_idx = idx[k];
        if (agent_idx < 0 || agent_idx >= env->num_total_agents) {
            return -1;
        }
        if (!(length[k] > 0.0f) || !(width[k] > 0.0f)) {
            return -1;
        }
        Agent *agent = &env->agents[agent_idx];
        agent->sim_length = length[k];
        agent->sim_width = width[k];
        update_agent_radius(agent); // collision broad-phase uses this; stale radius under/over-detects hits
        agent->wheelbase = WHEELBASE_LENGTH_RATIO * agent->sim_length;
    }
    return 0;
}

// yellow_remaining_s (optional, per element): a yellow light stays yellow through ts + ceil(s / dt) - 1, then red
int c_set_traffic_light_states(Drive *env, const int *states, const float *yellow_remaining_s) {
    int ts = env->timestep;
    if (env->obs_traffic_light_yellow_remaining && yellow_remaining_s == NULL) {
        return -1;
    }
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (traffic->states == NULL) {
            continue; // element without a state schedule cannot hold a state
        }
        if (ts < 0 || ts >= traffic->state_size) {
            return -1;
        }
        if (states[i] < TRAFFIC_CONTROL_STATE_UNKNOWN || states[i] > TRAFFIC_CONTROL_STATE_OFF) {
            return -1;
        }
        if (yellow_remaining_s != NULL
            && (!isfinite(yellow_remaining_s[i]) || yellow_remaining_s[i] < 0.0f
                || yellow_remaining_s[i] > env->traffic_light_yellow_max_s)) {
            return -1;
        }
        traffic->states[ts] = states[i];
        // holds until the next sync: c_step's metrics and obs read timestep + 1 before that sync arrives
        if (ts + 1 < traffic->state_size) {
            traffic->states[ts + 1] = states[i];
        }
        if (yellow_remaining_s == NULL || states[i] != TRAFFIC_CONTROL_STATE_YELLOW) {
            continue;
        }
        int yellow_steps = (int) ceilf(yellow_remaining_s[i] / env->dt);
        if (yellow_steps < 2) {
            yellow_steps = 2; // the held step stays yellow
        }
        int yellow_end = ts + yellow_steps < traffic->state_size ? ts + yellow_steps : traffic->state_size;
        for (int t = ts + 2; t < yellow_end; t++) {
            traffic->states[t] = TRAFFIC_CONTROL_STATE_YELLOW;
        }
        if (ts + yellow_steps < traffic->state_size) {
            traffic->states[ts + yellow_steps] = TRAFFIC_CONTROL_STATE_RED;
        }
    }
    return 0;
}

// Co-sim: map stop signs retired in place, the external ones appended (lines: count x 6 world coords)
int c_set_stop_signs(Drive *env, int count, const float *lines, const float *headings) {
    if (count < 0 || (count > 0 && (lines == NULL || headings == NULL))) {
        return -1;
    }
    for (int v = 0; v < count * 6; v++) {
        if (!isfinite(lines[v])) {
            return -1;
        }
    }
    for (int k = 0; k < count; k++) {
        float line_dx = lines[k * 6 + 3] - lines[k * 6];
        float line_dy = lines[k * 6 + 4] - lines[k * 6 + 1];
        if (!isfinite(headings[k]) || line_dx * line_dx + line_dy * line_dy <= 0.0f) {
            return -1;
        }
    }
    int old_count = env->num_traffic_elements;
    int new_count = old_count + count;
    if (new_count > 0) {
        TrafficControlElement *elements
            = (TrafficControlElement *) realloc(env->traffic_elements, new_count * sizeof(TrafficControlElement));
        if (elements == NULL) {
            return -1;
        }
        env->traffic_elements = elements;
    }
    for (int i = 0; i < old_count; i++) {
        if (env->traffic_elements[i].type == TRAFFIC_CONTROL_TYPE_STOP_SIGN) {
            env->traffic_elements[i].type = TRAFFIC_CONTROL_TYPE_NONE;
        }
    }
    for (int k = 0; k < count; k++) {
        TrafficControlElement *tc = &env->traffic_elements[old_count + k];
        memset(tc, 0, sizeof(TrafficControlElement));
        tc->type = TRAFFIC_CONTROL_TYPE_STOP_SIGN;
        tc->stop_line[0] = lines[k * 6] - env->world_mean_x;
        tc->stop_line[1] = lines[k * 6 + 1] - env->world_mean_y;
        tc->stop_line[2] = lines[k * 6 + 2];
        tc->stop_line[3] = lines[k * 6 + 3] - env->world_mean_x;
        tc->stop_line[4] = lines[k * 6 + 4] - env->world_mean_y;
        tc->stop_line[5] = lines[k * 6 + 5];
        tc->heading = headings[k];
        tc->junction_id = -1;
    }
    env->num_traffic_elements = new_count;
    for (int i = 0; i < env->num_total_agents; i++) {
        env->agents[i].stop_sign_target_idx = -1;
        env->agents[i].stop_sign_stop_completed = 0;
        env->agents[i].stop_sign_last_failed_idx = -1;
        env->agents[i].stop_sign_standstill_idx = -1;
        env->agents[i].red_light_latched_idx = -1;
    }
    return 0;
}

// Co-sim: light stop lines moved to the ones the external red-light judge uses (lines: count x 6 world coords)
int c_set_traffic_light_lines(Drive *env, int count, const int *indices, const float *lines) {
    if (count < 0 || (count > 0 && (indices == NULL || lines == NULL))) {
        return -1;
    }
    for (int k = 0; k < count; k++) {
        int idx = indices[k];
        if (idx < 0 || idx >= env->num_traffic_elements
            || env->traffic_elements[idx].type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            return -1;
        }
        for (int v = 0; v < 6; v++) {
            if (!isfinite(lines[k * 6 + v])) {
                return -1;
            }
        }
        float line_dx = lines[k * 6 + 3] - lines[k * 6];
        float line_dy = lines[k * 6 + 4] - lines[k * 6 + 1];
        if (line_dx * line_dx + line_dy * line_dy <= 0.0f) {
            return -1;
        }
    }
    for (int k = 0; k < count; k++) {
        TrafficControlElement *tc = &env->traffic_elements[indices[k]];
        tc->stop_line[0] = lines[k * 6] - env->world_mean_x;
        tc->stop_line[1] = lines[k * 6 + 1] - env->world_mean_y;
        tc->stop_line[2] = lines[k * 6 + 2];
        tc->stop_line[3] = lines[k * 6 + 3] - env->world_mean_x;
        tc->stop_line[4] = lines[k * 6 + 4] - env->world_mean_y;
        tc->stop_line[5] = lines[k * 6 + 5];
    }
    return 0;
}

int c_set_agent_goals(
    Drive *env,
    int agent_idx,
    int num_wp,
    const float *gx,
    const float *gy,
    const float *gz,
    const float *gdir_x,
    const float *gdir_y) {
    if (agent_idx < 0 || agent_idx >= env->num_total_agents || num_wp < 1 || num_wp > MAX_GOALS) {
        return -1;
    }
    Agent *agent = &env->agents[agent_idx];
    for (int w = 0; w < num_wp; w++) {
        if (!isfinite(gx[w]) || !isfinite(gy[w])) {
            return -1;
        }
        agent->list_goal_x[w] = gx[w] - env->world_mean_x;
        agent->list_goal_y[w] = gy[w] - env->world_mean_y;
        agent->list_goal_z[w] = gz[w];
        // the lane snap keeps the GPS lane-distance features working for external routes
        float snap_x, snap_y; // external goals keep their pushed position, only the lane idx is used
        agent->list_goal_lane[w]
            = find_goal_lane(env, agent->list_goal_x[w], agent->list_goal_y[w], gdir_x[w], gdir_y[w], &snap_x, &snap_y);
    }
    agent->goal_count = num_wp;
    set_current_goal(agent, 0);
    return 0;
}

// Rows are num_timesteps wide starting at init_step; steps beyond the log read invalid.
void c_get_global_ground_truth_trajectories(
    Drive *env,
    int num_timesteps,
    float *x_out,
    float *y_out,
    float *z_out,
    float *heading_out,
    int *valid_out,
    int *id_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);

        for (int step_idx = 0; step_idx < num_timesteps; step_idx++) {
            int t = env->init_step + step_idx;
            int out_idx = i * num_timesteps + step_idx;
            if (t >= agent->trajectory_size) {
                valid_out[out_idx] = 0;
                continue;
            }
            // Add world means back to get original world coordinates
            x_out[out_idx] = agent->log_trajectory_x[t] + env->world_mean_x;
            y_out[out_idx] = agent->log_trajectory_y[t] + env->world_mean_y;
            z_out[out_idx] = agent->log_trajectory_z[t];
            heading_out[out_idx] = agent->log_heading[t];
            valid_out[out_idx] = agent->log_valid[t];
        }
    }
}

// Partial Fisher-Yates over row indices; gathering rows in this order equals swapping the rows themselves.
static void subsample_road_observation_rows(Rng *rng_state, int *row_order, int collected_count, int keep_count) {
    if (collected_count <= keep_count) {
        return;
    }
    for (int sample_idx = 0; sample_idx < keep_count; sample_idx++) {
        int swap_idx = sample_idx + rng_below(rng_state, collected_count - sample_idx);
        int sampled_row = row_order[sample_idx];
        row_order[sample_idx] = row_order[swap_idx];
        row_order[swap_idx] = sampled_row;
    }
}

static void compute_metrics(Drive *env, int agent_idx, int log_idx) {
    Agent *agent = &env->agents[agent_idx];
    Log *agent_log = &env->logs[log_idx];

    reset_agent_metrics(env, agent_idx);
    if (agent->sim_x == INVALID_POSITION) {
        return;
    }
    int cell_idx = get_grid_index(env, agent->sim_x, agent->sim_y);
    if (cell_idx == -1) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f; // off the map grid counts as offroad
        apply_infraction_behavior(agent, env->offroad_behavior);
        return;
    }

    if (env->simulation_mode == SIMULATION_MODE_REPLAY) {
        float displacement_error = compute_displacement_error(agent, env->timestep);
        if (displacement_error > 0.0f) { // Only count valid displacements
            agent->cumulative_displacement += displacement_error;
            agent->displacement_sample_count++;
            agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX]
                = agent->cumulative_displacement / agent->displacement_sample_count;
        }
    }

    bool is_offroad;
    int lane_idx, lane_seg_idx;
    float signed_lane_distance, lane_heading;
    find_lane_and_offroad(
        env,
        agent,
        cell_idx,
        &is_offroad,
        &lane_idx,
        &lane_seg_idx,
        &signed_lane_distance,
        &lane_heading);
    set_lane_association(env, agent, lane_idx, lane_seg_idx, signed_lane_distance, lane_heading);
    if (lane_idx != -1 && env->compute_eval_metrics && agent->sim_speed > 0.0f
        && fabsf(signed_lane_distance) + agent->sim_width / 2.0f > MULTI_LANE_THRESHOLD) {
        agent_log->multi_lane_time += env->dt;
    }

    agent->distance_since_spawn += agent->sim_speed * env->dt;
    agent_log->avg_speed_per_agent += agent->sim_speed;

    float target_speed = UNKNOWN_LANE_SPEED_LIMIT_MPS;
    int current_lane_idx = agent->current_lane_idx;
    if (current_lane_idx != -1 && env->lane_speed_limit_mps[current_lane_idx] > 0) {
        target_speed = env->lane_speed_limit_mps[current_lane_idx];
    }
    agent->metrics_array[SPEED_LIMIT_IDX] = (agent->sim_speed > target_speed) ? 1.0f : 0.0f;
    if (env->compute_eval_metrics) {
        agent_log->speed_violation_sum += fmaxf(agent->sim_speed - target_speed, 0.0f) * env->dt;
    }

    if (agent->sim_speed_signed > VELOCITY_PROGRESS_MIN_SPEED_MPS && lane_idx != -1) {
        float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = fmaxf(cos_theta, 0.0f);
        if (env->compute_eval_metrics && cos_theta < 0.0f) {
            agent_log->wrong_way_distance += agent->sim_speed_signed * env->dt;
        }
    }

    int accel_violation = (fabsf(agent->accel_long) > COMFORT_ACCEL_THRESHOLD_MPS2)
        + (fabsf(agent->accel_lat) > COMFORT_ACCEL_THRESHOLD_MPS2);
    int jerk_violation
        = fabsf(agent->jerk_long) > COMFORT_JERK_THRESHOLD_MPS3 || fabsf(agent->jerk_lat) > COMFORT_JERK_THRESHOLD_MPS3;
    agent->metrics_array[COMFORT_VIOLATION_IDX] = (float) (accel_violation + jerk_violation);

    // infraction flags are exclusive, by priority: offroad, collision, red light, stop sign
    if (is_offroad) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->offroad_behavior);
        return;
    }

    int car_collided_with_index = collision_check(env, agent_idx);
    if (car_collided_with_index != -1) {
        agent->metrics_array[COLLISION_IDX] = 1.0f;
        if (agent->first_collision_partner_idx == -1) {
            agent->first_collision_partner_idx = car_collided_with_index;
        }
        if (env->compute_eval_metrics && is_at_fault_collision(env, agent_idx, car_collided_with_index)) {
            agent_log->at_fault_collision_rate = 1.0f;
            agent->metrics_array[AT_FAULT_COLLISION_IDX] = 1.0f;
        }
        apply_infraction_behavior(agent, env->collision_behavior);
        return;
    }

    if (env->traffic_lights_enabled && env->obs_slots_traffic_controls_n && !env->disable_red_light_infractions
        && check_red_light_violation(env, agent_idx)) {
        agent->metrics_array[RED_LIGHT_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->traffic_light_behavior);
        return;
    }

    // the stop sign state updates even with its infractions disabled, so the sign obs stay identical
    if (env->stop_signs_enabled && env->obs_slots_traffic_controls_n && update_stop_sign_state(env, agent_idx)
        && !env->disable_stop_sign_infractions) {
        agent->metrics_array[STOP_SIGN_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->stop_sign_behavior);
        return;
    }

    // swept goal check along prev -> sim, so a large dt cannot jump over the goal disc
    float distance_to_goal = compute_point_to_segment_distance(
        agent->current_goal_x,
        agent->current_goal_y,
        agent->prev_x,
        agent->prev_y,
        agent->sim_x,
        agent->sim_y);
    float goal_z_dist = fabsf(agent->sim_z - agent->current_goal_z);
    bool final_goal_too_fast = env->goal_reach_requires_speed && agent->current_goal_idx == agent->goal_count - 1
        && agent->sim_speed > agent->reward_coefs[REWARD_COEF_GOAL_SPEED];
    if (agent->current_goal_idx < agent->goal_count && distance_to_goal < agent->reward_coefs[REWARD_COEF_GOAL_RADIUS]
        && goal_z_dist < Z_BUFFER && !final_goal_too_fast) {
        agent->metrics_array[REACHED_GOAL_IDX] = 1.0f;
        agent_log->num_goals_reached += 1;
        agent->current_goal_idx++;
        if (agent->current_goal_idx < agent->goal_count) {
            set_current_goal(agent, agent->current_goal_idx);
        }
    }
}

static void compute_rewards(Drive *env, int i) {
    int agent_idx = env->active_agent_indices[i];
    Agent *agent = &env->agents[agent_idx];
    Log *agent_log = &env->logs[i];

    if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
        // grows with speed (-2 extra at 20 m/s), so braking before an unavoidable collision pays
        float reward_collision = -(agent->reward_coefs[REWARD_COEF_COLLISION] + 0.1f * agent->sim_speed);
        env->rewards[i] += reward_collision;
        agent_log->collision_rate = 1.0f;
        agent_log->pedestrian_collision_rate
            = env->agents[agent->first_collision_partner_idx].type == PEDESTRIAN ? 1.0f : 0.0f;
        agent_log->reward_collision += reward_collision;
    }

    if (agent->metrics_array[OFFROAD_IDX] > 0.0f) {
        float reward_offroad = -agent->reward_coefs[REWARD_COEF_OFFROAD];
        env->rewards[i] += reward_offroad;
        agent_log->offroad_rate = 1.0f;
        agent_log->reward_offroad += reward_offroad;
    }

    if (agent->metrics_array[RED_LIGHT_IDX] > 0.0f) {
        float reward_red_light = -agent->reward_coefs[REWARD_COEF_STOP_LINE];
        env->rewards[i] += reward_red_light;
        agent_log->red_light_violation_rate = 1.0f;
        agent_log->reward_red_light += reward_red_light;
    }

    if (agent->metrics_array[STOP_SIGN_IDX] > 0.0f) {
        float reward_stop_sign = -agent->reward_coefs[REWARD_COEF_STOP_LINE];
        env->rewards[i] += reward_stop_sign;
        agent_log->stop_sign_violation_rate = 1.0f;
        agent_log->reward_stop_sign += reward_stop_sign;
    }

    if (agent->metrics_array[REACHED_GOAL_IDX] > 0.0f) {
        bool final_waypoint = (agent->current_goal_idx == agent->goal_count);
        bool speeding = (agent->sim_speed > agent->reward_coefs[REWARD_COEF_GOAL_SPEED]);
        float reward_goal = (final_waypoint && speeding) ? 0.0f : env->reward_goal;
        env->rewards[i] += reward_goal;
        agent_log->reward_goal += reward_goal;
    }

    float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
    float theta_f = acosf(fminf(fmaxf(cos_theta, -1.0f), 1.0f));

    // Rl-align: min(cos,0) + vel_align*min(cos*v,0) + 0.0025*(1-|θ|/(π/2))
    float against_lane_penalty = fminf(cos_theta, 0.0f);
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
    agent_log->lane_center_rate += adjusted_dist < 0.5f ? 1.0f : 0.0f;
    agent_log->reward_lane_center += lane_center_reward;

    float comfort_violations = agent->metrics_array[COMFORT_VIOLATION_IDX];
    float comfort_penalty = -agent->reward_coefs[REWARD_COEF_COMFORT] * comfort_violations;
    env->rewards[i] += comfort_penalty;
    agent_log->comfort_violation_count += comfort_violations;
    agent_log->reward_comfort += comfort_penalty;

    float velocity_progress = agent->metrics_array[VELOCITY_PROGRESS_IDX];
    float velocity_reward = agent->reward_coefs[REWARD_COEF_VELOCITY] * env->dt * velocity_progress;
    env->rewards[i] += velocity_reward;
    agent_log->velocity_progress_sum += velocity_progress;
    agent_log->reward_velocity += velocity_reward;

    float accel = sqrtf(agent->accel_long * agent->accel_long + agent->accel_lat * agent->accel_lat);
    if (agent->sim_speed > 0.01f || accel > 0.01f) {
        float timestep_penalty = -agent->reward_coefs[REWARD_COEF_TIMESTEP] * env->dt;
        env->rewards[i] += timestep_penalty;
        agent_log->reward_timestep += timestep_penalty;
    }

    if (agent->sim_speed_signed < -0.01f) {
        float reverse_penalty = -agent->reward_coefs[REWARD_COEF_REVERSE] * env->dt;
        env->rewards[i] += reverse_penalty;
        agent_log->reward_reverse += reverse_penalty;
    }

    float speed_reward = -agent->reward_coefs[REWARD_COEF_OVERSPEED] * env->dt * agent->metrics_array[SPEED_LIMIT_IDX];
    env->rewards[i] += speed_reward;
    agent_log->reward_overspeed += speed_reward;

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

        if (agent->metrics_array[COMFORT_VIOLATION_IDX] > 0.0f) {
            int comfort_window_idx = (int) (agent_log->episode_length * env->dt / METRIC_SCORE_WINDOW_SECONDS);
            if (comfort_window_idx != agent->comfort_violation_last_window_idx) {
                agent->comfort_violation_last_window_idx = comfort_window_idx;
                agent_log->comfort_violation_window_count += 1.0f;
            }
        }
    } else {
        agent->metrics_array[TTC_IDX] = DEFAULT_TTC;
        agent->metrics_array[DISTANCE_TO_COLLISION_IDX] = DEFAULT_DTC;
    }
}

static int write_ego_obs(Drive *env, Agent *ego, float *obs, int obs_idx) {
    bool perceived_margin_active = (env->eval_mode && !env->eval_training_render) || env->cosim_eval_semantics;
    float perceived_margin = perceived_margin_active ? 2.0f * env->eval_perceived_size_margin_m : 0.0f;
    obs[obs_idx++] = ego->sim_speed_signed / env->obs_norm_speed_mps;
    obs[obs_idx++] = (ego->sim_width + perceived_margin) / env->obs_norm_veh_width_m;
    obs[obs_idx++] = (ego->sim_length + perceived_margin) / env->obs_norm_veh_length_m;
    obs[obs_idx++] = ego->steering_angle / STEERING_LIMIT;
    obs[obs_idx++] = ego->accel_long / fabsf(ACCEL_LONG_LIMIT[0]);
    obs[obs_idx++] = ego->accel_lat / ACCEL_LAT_LIMIT[1];
    obs[obs_idx++] = fmaxf(-1.0f, fminf(1.0f, ego->metrics_array[LANE_DIST_IDX] / LANE_DISTANCE_NORMALIZATION));
    obs[obs_idx++] = ego->lane_heading_error_rad / (float) M_PI;
    float current_lane_speed_limit
        = (ego->current_lane_idx != -1) ? env->lane_speed_limit_mps[ego->current_lane_idx] : -1.0f;
    obs[obs_idx++] = current_lane_speed_limit / env->obs_norm_speed_mps;
    obs[obs_idx++] = fminf(1.0f, ego->seconds_stopped / MAX_STOPPED_SECONDS);
    obs[obs_idx++] = fmaxf(-1.0f, fminf(1.0f, ego->lane_curvature / LANE_CURVATURE_NORM));
    return obs_idx;
}

static int write_reward_target_obs(Drive *env, Agent *ego, float *obs, int obs_idx) {
    if (env->reward_conditioning) {
        for (int coef_idx = 0; coef_idx < NUM_REWARD_COEFS; coef_idx++) {
            float lo = REWARD_BOUNDS[coef_idx].min_val;
            float hi = REWARD_BOUNDS[coef_idx].max_val;
            if (coef_idx == REWARD_COEF_SPEED) {
                lo = 1.0f / env->conditioning_speed_scale;
                hi = env->conditioning_speed_scale;
            }
            float coef = ego->reward_coefs[coef_idx];
            float normalized_coef = (coef - lo) / ((hi - lo) + 1e-8f);
            float clamped_coef = fmaxf(0.0f, fminf(1.0f, normalized_coef));
            obs[obs_idx++] = 2.0f * clamped_coef - 1.0f;
        }
    }

    for (int goal_idx = 0; goal_idx < env->num_goals; goal_idx++) {
        if (goal_idx < ego->current_goal_idx || goal_idx >= ego->goal_count) {
            obs_idx += GOAL_FEATURES; // compute_observations zeroed the slot
            continue;
        }
        float rel_goal_x, rel_goal_y;
        project_point_to_ego_frame(
            ego,
            ego->list_goal_x[goal_idx],
            ego->list_goal_y[goal_idx],
            &rel_goal_x,
            &rel_goal_y);
        // Goals beyond obs_norm_goal_offset_m collapse to a unit direction vector (keeps obs in [-1, 1])
        float goal_distance_m = sqrtf(rel_goal_x * rel_goal_x + rel_goal_y * rel_goal_y);
        float goal_normalization_m = fmaxf(env->obs_norm_goal_offset_m, goal_distance_m);
        obs[obs_idx++] = rel_goal_x / goal_normalization_m;
        obs[obs_idx++] = rel_goal_y / goal_normalization_m;
        obs[obs_idx++] = (ego->list_goal_z[goal_idx] - ego->sim_z) / env->obs_norm_z_m;
    }

    return obs_idx;
}

typedef struct {
    float x;
    float y;
    float z;
    int agent_idx;
} PartnerScanEntry;

static int write_partner_obs(
    Drive *env,
    Agent *ego,
    int ego_agent_idx,
    const PartnerScanEntry *partner_scan,
    float *obs,
    int obs_idx,
    int *partner_count) {
    if (ego->partner_blindness_counter > 0) {
        ego->partner_blindness_counter--;
    }
    if (ego->partner_blindness_counter == 0 && ego->is_blind_partner && env->partner_blindness_trigger_prob > 0.0f
        && rng_uniform_f32(&env->rng_state) < env->partner_blindness_trigger_prob) {
        ego->partner_blindness_counter = env->partner_blindness_duration;
    }
    if (ego->partner_blindness_counter > 0) {
        *partner_count = 0;
        return obs_idx + env->obs_slots_partners_n * PARTNER_ROW_FEATURES;
    }

    typedef struct {
        int index;
        float dist_sq;
        float dz;
    } AgentDistance;
    AgentDistance nearby_agents[env->num_agents];
    int nearby_count = 0;
    float range_sq = env->obs_range_partner_m * env->obs_range_partner_m;
    for (int j = 0; j < env->num_agents; j++) {
        float dx = partner_scan[j].x - ego->sim_x;
        float dy = partner_scan[j].y - ego->sim_y;
        float dz = partner_scan[j].z - ego->sim_z;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        nearby_agents[nearby_count].index = partner_scan[j].agent_idx;
        nearby_agents[nearby_count].dist_sq = dist_sq;
        nearby_agents[nearby_count].dz = dz;
        // branchless append: self and out-of-range candidates are overwritten by the next one
        nearby_count += (partner_scan[j].agent_idx != ego_agent_idx) & !(dist_sq > range_sq);
    }

    int partners_written = 0;
    int partners_to_write = (nearby_count < env->obs_slots_partners_n) ? nearby_count : env->obs_slots_partners_n;
    for (int k = 0; k < partners_to_write; k++) {
        int nearest_idx = k;
        float nearest_dist_sq = nearby_agents[k].dist_sq;
        for (int j = k + 1; j < nearby_count; j++) {
            if (nearby_agents[j].dist_sq < nearest_dist_sq) {
                nearest_dist_sq = nearby_agents[j].dist_sq;
                nearest_idx = j;
            }
        }
        if (nearest_idx != k) {
            AgentDistance tmp = nearby_agents[k];
            nearby_agents[k] = nearby_agents[nearest_idx];
            nearby_agents[nearest_idx] = tmp;
        }
    }

    for (int j = 0; j < partners_to_write && partners_written < env->obs_slots_partners_n; j++) {
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
        // float count: agent lengths are not range-checked at load, an int cast could overflow
        float piece_count = 1.0f;
        if (env->obs_partner_split_length_m > 0.0f && other->sim_length > env->obs_partner_split_length_m) {
            piece_count = ceilf(other->sim_length / env->obs_partner_split_length_m);
        }
        float piece_length_m = other->sim_length / piece_count;
        float rear_piece_offset_m = 0.5f * (piece_length_m - other->sim_length);
        int row_start = obs_idx;
        obs[obs_idx++] = (rel_x + rear_piece_offset_m * rel_heading_x) / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = (rel_y + rear_piece_offset_m * rel_heading_y) / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = nearby_agents[j].dz / env->obs_norm_z_m;
        obs[obs_idx++] = piece_length_m / env->obs_norm_veh_length_m;
        obs[obs_idx++] = other->sim_width / env->obs_norm_veh_width_m;
        obs[obs_idx++] = rel_heading_x;
        obs[obs_idx++] = rel_heading_y;
        obs[obs_idx++] = other->sim_speed_signed / env->obs_norm_speed_mps;
        obs[obs_idx++] = fminf(1.0f, other->seconds_stopped / MAX_STOPPED_SECONDS);
        obs[obs_idx++] = rel_vx / env->obs_norm_speed_mps;
        obs[obs_idx++] = rel_vy / env->obs_norm_speed_mps;
        partners_written++;
        for (int piece_idx = 1; (float) piece_idx < piece_count && partners_written < env->obs_slots_partners_n;
             piece_idx++) {
            float piece_offset_m = rear_piece_offset_m + (float) piece_idx * piece_length_m;
            memcpy(&obs[obs_idx], &obs[row_start], PARTNER_ROW_FEATURES * sizeof(float));
            obs[obs_idx] = (rel_x + piece_offset_m * rel_heading_x) / env->obs_norm_xy_offset_m;
            obs[obs_idx + 1] = (rel_y + piece_offset_m * rel_heading_y) / env->obs_norm_xy_offset_m;
            obs_idx += PARTNER_ROW_FEATURES;
            partners_written++;
        }
    }

    *partner_count = partners_written;
    return obs_idx + (env->obs_slots_partners_n - partners_written) * PARTNER_ROW_FEATURES;
}

// Lane-graph distance from a lane to the goal lane: INFINITY = unreachable, negative = lane not in the graph.
static float lane_graph_distance_to_goal_m(Drive *env, int lane_idx, int goal_graph_idx) {
    if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
        return -1.0f;
    }
    int lane_graph_idx = env->lane_graph.lane_to_graph_idx[lane_idx];
    if (lane_graph_idx < 0) {
        return -1.0f;
    }
    float distance_m = env->lane_graph.distances[goal_graph_idx * env->lane_graph.n_lanes + lane_graph_idx];
    return (isfinite(distance_m) && distance_m >= 0.0f) ? distance_m : INFINITY;
}

// GPS columns [abs, rel]; an unreachable lane reads as far, and as worse than an ego lane that can reach the goal.
static void write_lane_goal_distance_obs(float lane_dist_m, float ego_dist_to_goal_m, float *dest) {
    dest[0] = 0.0f;
    dest[1] = 0.0f;
    if (lane_dist_m < 0.0f) {
        return;
    }
    bool lane_reachable = isfinite(lane_dist_m);
    dest[0] = lane_reachable ? clip(lane_dist_m / LANE_GRAPH_DISTANCE_NORM_M, 0.0f, 1.0f) : 1.0f;
    if (ego_dist_to_goal_m < 0.0f) {
        return;
    }
    bool ego_reachable = isfinite(ego_dist_to_goal_m);
    if (lane_reachable && ego_reachable) {
        dest[1] = clip((lane_dist_m - ego_dist_to_goal_m) / LANE_GRAPH_DISTANCE_NORM_M, -1.0f, 1.0f);
    } else if (lane_reachable) {
        dest[1] = -1.0f;
    } else if (ego_reachable) {
        dest[1] = 1.0f;
    }
}

// Writes up to slot_cap rows of the cursor's road obs kind around the ego; returns the rows written.
static int write_road_obs_rows(
    Drive *env,
    Agent *ego,
    NeighborCursor *neighbor_cursor,
    float *dest,
    int slot_cap,
    int goal_graph_idx,
    float ego_dist_to_goal_m) {
    int is_lane = neighbor_cursor->obs_kind == OBS_ENTITY_LANE;
    int segment_features = is_lane ? LANE_ROW_FEATURES : BOUNDARY_FEATURES;
    int rows_written = 0;
    float width_clip_min_m = env->obs_lane_width_clip_max_m > 0.0f ? env->obs_lane_width_clip_min_m : 0.0f;
    float width_clip_max_m = env->obs_lane_width_clip_max_m > 0.0f ? env->obs_lane_width_clip_max_m : INFINITY;

    for (const GridMapEntity *entity = neighbor_cursor_next(env, neighbor_cursor);
         entity != NULL && rows_written < slot_cap;
         entity = neighbor_cursor_next(env, neighbor_cursor)) {
        int entity_idx = entity->entity_idx;
        const RoadObservationSegment *segment = &env->grid_map->segment_pool[entity->pool_idx];
        float rel_x, rel_y;
        float rel_z = segment->mid_z_m - ego->sim_z;
        project_point_to_ego_frame(ego, segment->mid_x_m, segment->mid_y_m, &rel_x, &rel_y);
        if (rel_x < -env->obs_range_road_behind_m || rel_x > env->obs_range_road_front_m) {
            continue;
        }
        if (fabsf(rel_y) > env->obs_range_road_side_m) {
            continue;
        }

        float rel_seg_dir_x, rel_seg_dir_y;
        project_vector_to_ego_frame(ego, segment->direction_x, segment->direction_y, &rel_seg_dir_x, &rel_seg_dir_y);

        float *row = &dest[rows_written++ * segment_features];
        row[0] = rel_x / env->obs_norm_xy_offset_m;
        row[1] = rel_y / env->obs_norm_xy_offset_m;
        row[2] = rel_z / env->obs_norm_z_m;
        row[3] = segment->half_length_m / env->obs_norm_road_seg_length_m;
        row[4] = rel_seg_dir_x;
        row[5] = rel_seg_dir_y;
        // Goal-distance features: absolute and relative to ego's lane->goal distance.
        if (is_lane) {
            row[6] = clip(segment->width_m, width_clip_min_m, width_clip_max_m) / env->obs_norm_road_seg_width_m;
            float lane_dist_m
                = goal_graph_idx >= 0 ? lane_graph_distance_to_goal_m(env, entity_idx, goal_graph_idx) : -1.0f;
            write_lane_goal_distance_obs(lane_dist_m, ego_dist_to_goal_m, &row[7]);
            row[LANE_FEATURES] = env->lane_speed_limit_mps[entity_idx] / env->obs_norm_speed_mps;
        }
    }
    return rows_written;
}

// Rows of the cursor's kind; with road dropout a random kept_slot_count of the up to slot_count rows found
static int write_road_obs_kind(
    Drive *env,
    Agent *ego,
    NeighborCursor *neighbor_cursor,
    float *dest,
    int slot_count,
    int kept_slot_count,
    int goal_graph_idx,
    float ego_dist_to_goal_m) {
    if (!env->road_dropout_enabled) {
        return write_road_obs_rows(env, ego, neighbor_cursor, dest, slot_count, goal_graph_idx, ego_dist_to_goal_m);
    }
    int row_features = neighbor_cursor->obs_kind == OBS_ENTITY_LANE ? LANE_ROW_FEATURES : BOUNDARY_FEATURES;
    float rows_buffer[slot_count * row_features];
    int rows_found
        = write_road_obs_rows(env, ego, neighbor_cursor, rows_buffer, slot_count, goal_graph_idx, ego_dist_to_goal_m);
    int rows_kept = (rows_found < kept_slot_count) ? rows_found : kept_slot_count;
    int row_order[slot_count];
    for (int k = 0; k < rows_found; k++) {
        row_order[k] = k;
    }
    subsample_road_observation_rows(&env->rng_state, row_order, rows_found, rows_kept);
    for (int k = 0; k < rows_kept; k++) {
        memcpy(&dest[k * row_features], &rows_buffer[row_order[k] * row_features], row_features * sizeof(float));
    }
    return rows_kept;
}

// Lane rows (and their dropout draws) come before boundary rows
static int write_road_obs(Drive *env, Agent *ego, float *obs, int obs_idx, int *lane_count, int *boundary_count) {
    int goal_graph_idx = -1;
    if (env->lane_graph.lane_to_graph_idx != NULL && ego->current_goal_idx < ego->goal_count) {
        int goal_lane = ego->list_goal_lane[ego->current_goal_idx];
        if (goal_lane >= 0) {
            goal_graph_idx = env->lane_graph.lane_to_graph_idx[goal_lane];
        }
    }
    float ego_dist_to_goal_m = -1.0f; // <0 = no valid reference -> relative column stays 0
    if (goal_graph_idx >= 0) {
        ego_dist_to_goal_m = lane_graph_distance_to_goal_m(env, ego->current_lane_idx, goal_graph_idx);
    }

    int boundary_obs_idx = obs_idx + env->obs_slots_lane_kept * LANE_ROW_FEATURES;
    int scratch_count = fill_neighbor_scratch(env, ego->sim_x, ego->sim_y);
    NeighborCursor lane_cursor = neighbor_cursor_begin(env, ego->sim_x, ego->sim_y, OBS_ENTITY_LANE, scratch_count);
    NeighborCursor edge_cursor = neighbor_cursor_begin(env, ego->sim_x, ego->sim_y, OBS_ENTITY_EDGE, scratch_count);
    *lane_count = write_road_obs_kind(
        env,
        ego,
        &lane_cursor,
        &obs[obs_idx],
        env->obs_slots_lane_n,
        env->obs_slots_lane_kept,
        goal_graph_idx,
        ego_dist_to_goal_m);
    *boundary_count = write_road_obs_kind(
        env,
        ego,
        &edge_cursor,
        &obs[boundary_obs_idx],
        env->obs_slots_boundary_n,
        env->obs_slots_boundary_kept,
        goal_graph_idx,
        ego_dist_to_goal_m);
    return boundary_obs_idx + env->obs_slots_boundary_kept * BOUNDARY_FEATURES;
}

// Consecutive yellow steps from timestep on, at most max_steps (the schedule's end truncates)
static int yellow_remaining_steps(const TrafficControlElement *tc, int timestep, int max_steps) {
    int scan_end = timestep + max_steps < tc->state_size ? timestep + max_steps : tc->state_size;
    int remaining_steps = 0;
    for (int t = timestep; t < scan_end; t++) {
        if (tc->states[t] != TRAFFIC_CONTROL_STATE_YELLOW) {
            break;
        }
        remaining_steps++;
    }
    return remaining_steps;
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
        if (tc->type == TRAFFIC_CONTROL_TYPE_NONE
            || (tc->type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT && !env->traffic_lights_enabled)
            || (tc->type == TRAFFIC_CONTROL_TYPE_STOP_SIGN && !env->stop_signs_enabled)
            || tc->type == TRAFFIC_CONTROL_TYPE_YIELD_SIGN) {
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
        float nearest_dist_sq = visible_controls[k].dist_sq;
        for (int j = k + 1; j < visible_count; j++) {
            if (visible_controls[j].dist_sq < nearest_dist_sq) {
                nearest_dist_sq = visible_controls[j].dist_sq;
                nearest_idx = j;
            }
        }
        if (nearest_idx != k) {
            TrafficControlDist temp = visible_controls[k];
            visible_controls[k] = visible_controls[nearest_idx];
            visible_controls[nearest_idx] = temp;
        }
    }

    int yellow_scan_steps = (int) ceilf(env->traffic_light_yellow_max_s / env->dt);
    int controls_written = 0;
    for (int j = 0; j < controls_to_observe && controls_written < env->obs_slots_traffic_controls_n; j++) {
        TrafficControlElement *tc = &env->traffic_elements[visible_controls[j].idx];
        float rel_x1, rel_y1, rel_x2, rel_y2;
        project_point_to_ego_frame(ego, tc->stop_line[0], tc->stop_line[1], &rel_x1, &rel_y1);
        project_point_to_ego_frame(ego, tc->stop_line[3], tc->stop_line[4], &rel_x2, &rel_y2);
        float rel_z = (tc->stop_line[2] + tc->stop_line[5]) * 0.5f - ego->sim_z;
        int light_state = TRAFFIC_CONTROL_STATE_UNKNOWN;
        if (tc->type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            light_state = (env->timestep >= 0 && env->timestep < tc->state_size) ? tc->states[env->timestep]
                                                                                 : TRAFFIC_CONTROL_STATE_OFF;
        } else if (tc->type == TRAFFIC_CONTROL_TYPE_STOP_SIGN && visible_controls[j].idx == ego->stop_sign_target_idx) {
            // the ego's own stop sign reads red until it stood still in the trigger box, then green
            light_state = ego->stop_sign_stop_completed ? TRAFFIC_CONTROL_STATE_GREEN : TRAFFIC_CONTROL_STATE_RED;
        }

        obs[obs_idx++] = rel_x1 / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_y1 / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_x2 / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_y2 / env->obs_norm_xy_offset_m;
        obs[obs_idx++] = rel_z / env->obs_norm_z_m;
        if (env->obs_traffic_light_yellow_remaining) {
            float yellow_remaining_s = light_state == TRAFFIC_CONTROL_STATE_YELLOW
                ? yellow_remaining_steps(tc, env->timestep, yellow_scan_steps) * env->dt
                : 0.0f;
            obs[obs_idx++]
                = fminf(yellow_remaining_s, env->traffic_light_yellow_max_s) / env->traffic_light_yellow_max_s;
        }
        obs[obs_idx++] = tc->type;
        obs[obs_idx++] = light_state;
        controls_written++;
    }

    *traffic_control_count = controls_written;
    return obs_idx + (env->obs_slots_traffic_controls_n - controls_written) * traffic_control_feature_count(env);
}

static void compute_observations(Drive *env) {
    int obs_per_agent = compute_observation_size(env);
    memset(env->observations, 0, obs_per_agent * env->active_agent_count * sizeof(float));
    PartnerScanEntry partner_scan[env->num_agents];
    for (int j = 0; j < env->num_agents; j++) {
        int index = (j < env->active_agent_count) ? env->active_agent_indices[j]
                                                  : env->static_agent_indices[j - env->active_agent_count];
        Agent *agent = &env->agents[index];
        partner_scan[j] = (PartnerScanEntry) {agent->sim_x, agent->sim_y, agent->sim_z, index};
    }
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &env->observations[i * obs_per_agent];
        int agent_idx = env->active_agent_indices[i];
        Agent *ego = &env->agents[agent_idx];
        int partner_count = 0;
        int lane_count = 0;
        int boundary_count = 0;
        int traffic_control_count = 0;
        int obs_idx = write_ego_obs(env, ego, obs, 0);
        obs_idx = write_reward_target_obs(env, ego, obs, obs_idx);
        obs_idx = write_partner_obs(env, ego, agent_idx, partner_scan, obs, obs_idx, &partner_count);
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
    if (agent->phantom_braking_counter > 0) {
        agent->phantom_braking_counter--;
    }
    if (agent->phantom_braking_counter == 0 && agent->is_phantom_braker && env->phantom_braking_trigger_prob > 0.0f
        && rng_uniform_f32(&env->rng_state) < env->phantom_braking_trigger_prob) {
        agent->phantom_braking_counter = env->phantom_braking_duration;
    }

    // action in [-1, 1]: braking scales to -JERK_LONG[0], throttle to JERK_LONG[3]
    float (*action_array_f)[2] = (float (*)[2]) env->actions;
    float j_long_action = action_array_f[action_idx][0];
    float j_long = j_long_action * (j_long_action < 0.0f ? -JERK_LONG[0] : JERK_LONG[3]);
    float j_lat = action_array_f[action_idx][1] * JERK_LAT[2];
    if (agent->phantom_braking_counter > 0) {
        j_long = JERK_LONG[0];
    }
    if ((env->eval_mode || env->cosim_eval_semantics) && agent->sim_speed < STANDSTILL_SPEED_EPSILON_MPS
        && fabsf(j_long) < env->eval_standstill_jerk_deadband_mps3) {
        j_long = 0.0f;
    }

    float c_throttle = agent->reward_coefs[REWARD_COEF_THROTTLE];
    float c_steer = agent->reward_coefs[REWARD_COEF_STEER];
    float c_acc = agent->reward_coefs[REWARD_COEF_ACC];

    // paper Eq. 1; a sign change snaps the acceleration to 0
    float a_long_new = agent->accel_long + c_throttle * j_long * env->dt;
    if (agent->accel_long * a_long_new < 0) {
        a_long_new = 0.0f;
    } else {
        a_long_new = clip(a_long_new, ACCEL_LONG_LIMIT[0], ACCEL_LONG_LIMIT[1] * c_acc);
    }
    // Comfort scores the driver's jerk, so take it before the stop resets below zero the stored acceleration
    float jerk_long_before_stop_reset = (a_long_new - agent->accel_long) / env->dt;

    // paper Eq. 2
    float a_lat_new = agent->accel_lat + c_steer * j_lat * env->dt;
    if (agent->accel_lat * a_lat_new < 0) {
        a_lat_new = 0.0f;
    } else {
        a_lat_new = clip(a_lat_new, ACCEL_LAT_LIMIT[0], ACCEL_LAT_LIMIT[1]);
    }

    float heading_x = agent->cos_heading;
    float heading_y = agent->sin_heading;

    // trapezoidal integration of the speed
    float v_dot_heading = agent->sim_vx * heading_x + agent->sim_vy * heading_y;
    float signed_v = copysignf(sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy), v_dot_heading);
    float v_new = signed_v + 0.5f * (a_long_new + agent->accel_long) * env->dt;
    if (signed_v * v_new < 0) {
        v_new = 0.0f;
        a_long_new = 0.0f;
        a_lat_new = 0.0f;
    } else {
        v_new = clip(v_new, MAX_BACKWARD_SPEED, env->base_max_speed_mps * agent->reward_coefs[REWARD_COEF_SPEED]);
    }
    if (agent->phantom_braking_counter > 0 && v_new < 0.0f) {
        v_new = 0.0f;
        a_long_new = 0.0f;
    }

    // accel_lat -> curvature -> steering, with the speed floored at 1 m/s
    float v_eff = fmaxf(fabsf(v_new), 1.0f);
    float signed_curvature = a_lat_new / (v_eff * v_eff);
    float steering_angle = atanf(signed_curvature * agent->wheelbase);
    float delta_steer = clip(
        steering_angle - agent->steering_angle,
        -MAX_STEERING_RATE_RADPS * env->dt,
        MAX_STEERING_RATE_RADPS * env->dt);
    float new_steering_angle
        = clip(agent->steering_angle + delta_steer, -MAX_STEERING_ANGLE_RAD, MAX_STEERING_ANGLE_RAD);
    signed_curvature = tanf(new_steering_angle) / agent->wheelbase;
    a_lat_new = v_new * v_new * signed_curvature;

    // bicycle model: arc of length d along the limited curvature
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

    agent->sim_x += dx;
    agent->sim_y += dy;
    agent->sim_heading = normalize_heading(agent->sim_heading + theta);
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->sim_vx = v_new * agent->cos_heading;
    agent->sim_vy = v_new * agent->sin_heading;
    update_agent_speed(agent);
    agent->jerk_long = jerk_long_before_stop_reset;
    agent->jerk_lat = (a_lat_new - agent->accel_lat) / env->dt;
    agent->accel_long = a_long_new;
    agent->accel_lat = a_lat_new;
    agent->steering_angle = new_steering_angle;
    update_agent_z(env, agent);
}

static void apply_pose_noise(Drive *env, Agent *agent) {
    // GIGAFLOW dynamics randomization: brownian buffeting applied to the pose, not integrated through speed.
    agent->sim_x += sample_normal(&env->rng_state, env->pose_noise_xy_m);
    agent->sim_y += sample_normal(&env->rng_state, env->pose_noise_xy_m);
    agent->sim_heading
        = normalize_heading(agent->sim_heading + sample_normal(&env->rng_state, env->pose_noise_yaw_rad));
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
}

#include "idm.h"

static void update_rollout_masks(Drive *env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Agent *a = &env->agents[env->active_agent_indices[i]];
        env->masks[i] = !(a->stopped || a->removed || a->is_blind_partner || a->is_phantom_braker);
    }
}

// c_reset scores no goals: one inside the radius at spawn stays for the first step to consume, reward and renew
static void restore_goal_reached_at_reset(Agent *agent, Log *agent_log) {
    if (agent->metrics_array[REACHED_GOAL_IDX] == 0.0f) {
        return;
    }
    agent->metrics_array[REACHED_GOAL_IDX] = 0.0f;
    agent_log->num_goals_reached -= 1;
    set_current_goal(agent, agent->current_goal_idx - 1);
}

void c_reset(Drive *env) {
    env->autoreset_pending = 0;
    env->episode_end_timestep = env->scenario_length;
    if (env->first_reset_pending) {
        env->first_reset_pending = 0;
        // first episode after creation ends early so sub-envs do not run their episodes in lockstep
        if (env->stagger_first_episode && !env->eval_mode) {
            int offset_range = env->scenario_length - env->init_step - env->init_step_min_horizon;
            env->episode_end_timestep -= rng_below(&env->rng_state, offset_range + 1);
        }
    }
    if (env->timestep == 0) {
        for (int i = 0; i < env->num_total_agents; i++) {
            copy_pose_to_prev(&env->agents[i]);
        }
        for (int x = 0; x < env->active_agent_count; x++) {
            env->logs[x] = (Log) {0};
            int agent_idx = env->active_agent_indices[x];
            sample_erratic_flags(env, &env->agents[agent_idx]);
            compute_metrics(env, agent_idx, x);
            restore_goal_reached_at_reset(&env->agents[agent_idx], &env->logs[x]);
        }
        update_rollout_masks(env);
        compute_observations(env);
        return;
    }

    env->timestep = env->init_step;

    begin_episode_rng(env);
    sample_zone_speed_limits(env);
    if (env->simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        generate_traffic_light_states(env);
        // stale poses must neither block nor escape the spawn overlap check
        for (int x = 0; x < env->active_agent_count; x++) {
            invalidate_agent(&env->agents[env->active_agent_indices[x]]);
        }
        reset_scripted_pedestrians(env);
        int num_reset = 0;
        for (int x = 0; x < env->active_agent_count; x++) {
            int agent_idx = env->active_agent_indices[x];
            env->logs[x] = (Log) {0};

            if (spawn_agent(env, agent_idx, num_reset)) {
                env->agents[agent_idx].removed = 0; // a removal from an earlier episode must not outlive a respawn
                num_reset++;
            } else {
                invalidate_agent(&env->agents[agent_idx]);
                env->agents[agent_idx].removed = 1;
            }
        }

        // spawn_agent already set poses, routes and goals
        for (int x = 0; x < env->active_agent_count; x++) {
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
            restore_goal_reached_at_reset(agent, &env->logs[x]);
        }
        update_rollout_masks(env);
        compute_observations(env);
        return;
    }

    set_start_position(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log) {0};
        int agent_idx = env->active_agent_indices[x];
        Agent *agent = &env->agents[agent_idx];

        reset_agent_metrics(env, agent_idx);
        reset_agent_state(agent);
        sample_erratic_flags(env, agent);
        generate_reward_coefs(env, agent);
        if (agent->sim_valid == 1) {
            init_dynamics_state_from_log(env, agent);
        }

        if (env->goal_source == GOAL_SOURCE_GT || env->goal_source == GOAL_SOURCE_GT_MAP) {
            generate_new_goals_from_log(env, agent);
        } else {
            generate_new_goals_from_route(env, agent);
        }
        compute_metrics(env, agent_idx, x);
        restore_goal_reached_at_reset(agent, &env->logs[x]);
    }
    update_rollout_masks(env);
    compute_observations(env);
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));
    // eval scores a scenario once: the env holds after its episode so it is not replayed within the window
    if (env->eval_episode_done) {
        return;
    }
    if (env->autoreset_pending) {
        c_reset(env);
        return;
    }

    env->timestep++;

    for (int i = 0; i < env->expert_static_agent_count; i++) {
        move_expert(env, env->expert_static_agent_indices[i]);
    }
    move_scripted_pedestrians(env);
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        if (agent->controller == CONTROLLER_POLICY) {
            move_dynamics(env, i, agent_idx);
            if (!agent->stopped && !agent->removed && (env->pose_noise_xy_m > 0.0f || env->pose_noise_yaw_rad > 0.0f)
                && (!env->eval_mode || env->eval_training_render)) {
                apply_pose_noise(env, agent);
            }
        } else if (agent->controller == CONTROLLER_IDM) {
            move_idm(env, agent_idx);
        } else if (agent->controller == CONTROLLER_REPLAY && env->simulation_mode == SIMULATION_MODE_REPLAY) {
            move_expert(env, agent_idx);
        }
    }

    // every agent, not only policy ones: partners' seconds_stopped is observed
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

    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->agents[agent_idx].stopped || env->agents[agent_idx].removed) {
            continue;
        }
        compute_metrics(env, agent_idx, i);
        compute_rewards(env, i);
    }

    // Masks describe the next action's eligibility; terminal rewards belong to the previous action.
    update_rollout_masks(env);
    int inactive_count = 0;
    for (int i = 0; i < env->active_agent_count; i++) {
        Agent *agent = &env->agents[env->active_agent_indices[i]];
        if (agent->stopped || agent->removed) {
            env->terminals[i] = 1;
            inactive_count++;
        }
    }
    bool early_reset = env->termination_mode == 1
        && (float) inactive_count / (float) env->active_agent_count > env->inactive_agent_threshold;
    if (env->timestep == env->episode_end_timestep || early_reset) {
        for (int i = 0; i < env->active_agent_count; i++) {
            env->truncations[i] = 1;
        }
        add_log(env);
        if (env->eval_mode) {
            env->eval_episode_done = 1;
            return;
        }
        // Expose the terminal state for the value bootstrap; the next c_step resets without applying its action.
        compute_observations(env);
        memset(env->masks, 0, env->active_agent_count * sizeof(unsigned char));
        env->autoreset_pending = 1;
        return;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        // stopped and removed agents skipped compute_metrics above, so their goal flag is from an earlier step
        if (agent->stopped || agent->removed || agent->metrics_array[REACHED_GOAL_IDX] == 0.0f) {
            continue;
        }
        if (env->goal_source == GOAL_SOURCE_GT || env->goal_source == GOAL_SOURCE_GT_MAP
            || env->goal_source == GOAL_SOURCE_EXTERNAL) {
            // the goal index stays saturated: the next window comes from the log or c_set_agent_goals
            continue;
        }
        // full regen: rolling when its walk dead-ends, finite once the goal set is used up
        bool regen = (env->goal_regen_mode == GOAL_REGEN_ROLLING) ? !roll_goals(env, agent)
                                                                  : agent->current_goal_idx == agent->goal_count;
        if (!regen) {
            continue;
        }
        // a failed regen would leave an empty goal window and freeze the agent, so it is removed
        bool regen_ok = (env->goal_source == GOAL_SOURCE_MAP) ? generate_new_goals_from_map(env, agent)
                                                              : generate_new_goals_from_route(env, agent);
        if (!regen_ok) {
            invalidate_agent(agent);
            agent->removed = 1;
            env->masks[i] = 0;
            env->terminals[i] = 1;
        }
    }

    // observations describe the goals the next action is taken for
    compute_observations(env);
}
