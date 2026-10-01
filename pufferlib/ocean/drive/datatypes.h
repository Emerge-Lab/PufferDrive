#include "constants.h"

#include <stdlib.h>

static inline int is_road_lane(int type) {
    return (type >= 0 && type <= 9);
}

static inline int is_drivable_road_lane(int type) {
    return (type == LANE_FREEWAY || type == LANE_SURFACE_STREET);
}

static inline int is_road_line(int type) {
    return (type >= 10 && type <= 19);
}

static inline int is_road_edge(int type) {
    return (type >= 20 && type <= 29);
}

static inline int is_misc_road(int type) {
    return type >= MISC_UNKNOWN;
}

static inline int is_road(int type) {
    return is_road_lane(type) || is_road_line(type) || is_road_edge(type);
}

static inline int is_road_grid_candidate(int type) {
    return is_road_lane(type) || is_road_edge(type);
}

static inline int is_controllable_agent(int type) {
    return (type == VEHICLE || type == PEDESTRIAN || type == CYCLIST);
}

struct Agent {
    int id;
    int type;

    // Log trajectory
    int trajectory_size;
    float *log_trajectory_x;
    float *log_trajectory_y;
    float *log_trajectory_z;
    float *log_heading;
    float *log_velocity_x;
    float *log_velocity_y;
    float *log_length;
    float *log_width;
    float *log_height;
    int *log_valid;

    // Simulation state
    float sim_x;       // Bounding box center x
    float sim_y;       // Bounding box center y
    float sim_z;       // Bounding box center z
    float sim_heading; // Bounding box heading
    float cos_heading;
    float sin_heading;
    float sim_vx;           // Bounding box velocity x
    float sim_vy;           // Bounding box velocity y
    float yaw_rate;         // Angular velocity used to convert between rear and center velocity
    float sim_speed;        // Bounding box center speed magnitude
    float sim_speed_signed; // Bounding box center signed longitudinal speed
    float sim_length;       // Bounding box length
    float sim_width;        // Bounding box width
    float sim_height;       // Bounding box height
    float radius;           // Circumradius (smallest enclosing circle) -> 0.5*sqrt(L^2+W^2)
    float prev_x;
    float prev_y;
    float prev_cos_heading;
    float prev_sin_heading;
    int sim_valid;

    // Route information
    int route_length;
    int *route;
    int route_gt_len;      // Number of leading route lanes supported by GT before extension
    int current_route_idx; // Tracks progress through route array

    // Metrics and status tracking (size must match NUM_METRICS in drive.h)
    float metrics_array[NUM_METRICS]; // [collision, offroad, red_light, stop_sign, reached_goal, lane_dist, lane_angle,
                                      // comfort_violation, velocity_progress, speed_limit, avg_displacement_error,
                                      // progression, at_fault_collision, ttc, distance_to_collision, progress_ratio,
                                      // multi_lane_time, multi_lane_score]
    int current_lane_idx;
    int previous_lane_idx;
    int current_lane_geometry_idx;
    int reached_goal_this_episode;
    int num_goals_reached;
    int active_agent;
    int mark_as_expert;
    int controller;
    float cumulative_displacement;
    int displacement_sample_count;
    float distance_since_spawn;
    float seconds_stopped;
    int stop_sign_stopped_timestep_count;

    // Goal positions
    float list_goal_x[MAX_GOALS];
    float list_goal_y[MAX_GOALS];
    float list_goal_z[MAX_GOALS];
    int list_goal_lane[MAX_GOALS]; // lane idx of each goal (for GPS lookup); -1 if none
    float current_goal_x;          // alias = list_goal_x[current_goal_idx]
    float current_goal_y;          // alias = list_goal_y[current_goal_idx]
    float current_goal_z;          // alias = list_goal_z[current_goal_idx]
    int current_goal_idx;          // index of next goal to reach (0..N-1)
    int goal_count;                // number of active goals (<= num_goals)
    float gt_goal_x;               // Last valid ground-truth goal position x
    float gt_goal_y;               // Last valid ground-truth goal position y
    float gt_goal_z;               // Last valid ground-truth goal position z

    int stopped; // 0/1 -> freeze if set
    int removed; // 0/1 -> remove from sim if set

    // Jerk dynamics
    float accel_long;
    float accel_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;

    // Spline terminal-boundary dynamics (trajectory_training). curr plus a ring of past
    // curves store the full quintic c0..c5 per global axis, so compute_rewards can compare
    // this step's curve against the last k over their overlap.
    float spline_coefs_x[6];
    float spline_coefs_y[6];
    float spline_history_coefs_x[SPLINE_CONSISTENCY_MAX_LAG][6];
    float spline_history_coefs_y[SPLINE_CONSISTENCY_MAX_LAG][6];
    // ring entry at lag j was solved exactly j steps ago; head is where the next curve goes
    int spline_history_count;
    int spline_history_head;
    float spline_intent[SPLINE_INTENT_FEATURES]; // last emitted action, raw [-1, 1], fed back as obs

    // Reward conditioning coefficients (per-agent, randomized at spawn)
    float reward_coefs[NUM_REWARD_COEFS];

    int phantom_braking_counter;     // >0 means currently phantom braking
    int partner_blindness_counter;   // >0 means currently blind to partners
    unsigned char is_blind_partner;  // episode-level flag: agent sees no other agents
    unsigned char is_phantom_braker; // episode-level flag: agent may phantom-brake
};

struct RoadMapElement {
    int type;

    int segment_size;
    float *x;
    float *y;
    float *z;
    float *headings; // Pre-computed heading for each segment

    // Lane specific info
    int num_entries;
    int *entry_lanes;
    int num_exits;
    int *exit_lanes;
    float speed_limit;
    float length;
    float *cum_lengths;
};

struct TrafficControlElement {
    int type;

    int state_size;
    int *states;
    float stop_line[6]; // Two 3D endpoints: [x1,y1,z1, x2,y2,z2]
    float heading;
    int num_controlled_lanes;
    int *controlled_lanes;
};

struct LaneGraph {
    int n_lanes;
    int *lane_ids;
    float *distances;       // n_lanes * n_lanes row-major (row = from, col = to)
    int *lane_to_graph_idx; // road-element idx -> graph idx (-1 if lane absent from graph), sized num_road_elements
};

// Lattice menus and derived action layout; units in field names, cells enumerate duration-major.
struct LatticeConfig {
    int lat_offset_count;
    float lat_offsets_m[LATTICE_MAX_LAT_OFFSETS];
    int lat_duration_count;
    float lat_durations_s[LATTICE_MAX_LAT_DURATIONS];
    float low_speed_distances_m[LATTICE_MAX_LAT_DURATIONS];
    int lon_speed_count;
    float lon_speeds_mps[LATTICE_MAX_LON_SPEEDS];
    int lon_duration_count;
    float lon_durations_s[LATTICE_MAX_LON_DURATIONS];
    int stop_distance_count;
    float stop_distances_m[LATTICE_MAX_STOP_DISTANCES];
    int backup_distance_count;
    float backup_distances_m[LATTICE_MAX_BACKUP_DISTANCES];
    float low_speed_mps;
    float decision_period_s;
    int exit_mode;
    int decision_period_steps;
    int lat_duration_steps[LATTICE_MAX_LAT_DURATIONS];
    int lon_duration_steps[LATTICE_MAX_LON_DURATIONS];
    int lat_cell_count;
    int lon_speed_cell_count;
    int lon_stop_cell_base;
    int lon_stop_line_cell;
    int lon_emergency_cell;
    int lon_backup_cell_base;
    int lon_cell_count;
    int mask_feature_count;
    int nvec[LATTICE_ACTION_FACTORS];
};

// Per road element, built once per env; exits are the pruned drivable successors in slot order.
struct LatticeLaneInfo {
    int exit_count;
    int exit_slots[LATTICE_EXIT_SLOTS];
    float exit_turn_rad[LATTICE_EXIT_SLOTS];
    int predecessor_count;
    int predecessors[LATTICE_MAX_PREDECESSORS];
    int is_connector;
    int traffic_light_idx;
    float length_m;
    float max_curvature;
    int profile_offset;
    int profile_count;
    int cum_offset;
    int point_count;
};

// Lane cross-section every LATTICE_PROFILE_SPACING_M; offsets signed, left positive.
struct LatticeProfileSample {
    float edge_left_m;
    float edge_right_m;
    int neighbour_lane[2];
    float neighbour_offset_m[2];
    float neighbour_arc_m[2];
};

struct LatticeVertex {
    float x;
    float y;
    float lane_arc_m;
    int chain_slot;
};

// Per-env buffers for building one rail; never shared between agents mid-build.
struct LatticeBuildScratch {
    struct LatticeVertex vertices[LATTICE_MAX_RAIL_VERTICES];
    float raw_x[LATTICE_MAX_RAW_SAMPLES];
    float raw_y[LATTICE_MAX_RAW_SAMPLES];
    float raw_lane_arc_m[LATTICE_MAX_RAW_SAMPLES];
    unsigned char raw_slot[LATTICE_MAX_RAW_SAMPLES];
    float filtered_x[LATTICE_MAX_RAW_SAMPLES];
    float filtered_y[LATTICE_MAX_RAW_SAMPLES];
};

// Reference rail: smoothed chain samples; s of sample j is s_start_m + j * LATTICE_RAIL_SPACING_M.
struct LatticeRail {
    int sample_count;
    float s_start_m;
    float x[LATTICE_RAIL_SAMPLES];
    float y[LATTICE_RAIL_SAMPLES];
    float heading[LATTICE_RAIL_SAMPLES];
    float curvature[LATTICE_RAIL_SAMPLES];
    float v_env[LATTICE_RAIL_SAMPLES];
    float edge_left_m[LATTICE_RAIL_SAMPLES];
    float edge_right_m[LATTICE_RAIL_SAMPLES];
    float lane_arc_m[LATTICE_RAIL_SAMPLES];
    unsigned char chain_slot[LATTICE_RAIL_SAMPLES];
    int lane_count;
    int lanes[LATTICE_CHAIN_MAX_LANES];
    float lane_start_s_m[LATTICE_CHAIN_MAX_LANES];
    float lane_end_s_m[LATTICE_CHAIN_MAX_LANES];
    int exit_decided[LATTICE_CHAIN_MAX_LANES];
    float chain_start_arc_m;
    int chain_is_dead_end;
    int chain_is_complete;
    int is_straight_fallback;
};

// Lateral plan: quintic in time (u = t - t0) or in signed rail distance (u = dir * (s - s0)); holds beyond its end.
struct LatticeLatPlan {
    int mode;
    int kind;
    double coefs[6];
    float horizon;
    int start_step;
    int end_step;
    float start_s_m;
    int dir;
    float target_d_m;
    int reindexed_low_speed;
    int reindexed_stop;
};

// Longitudinal plan in distance driven sigma and real signed speed; speed plans hold their end speed.
struct LatticeLonPlan {
    int kind;
    double coefs[6];
    float horizon_s;
    int start_step;
    int end_step;
    int cell;
    float target_speed_mps;
    double target_sigma_m;
    int gear;
    int release_latched;
    int two_step_stage;
};

struct LatticeCounters {
    float steps;
    float decisions;
    float lat_new;
    float lon_new;
    float invalid_actions;
    float decode_rejects;
    float auto_replans;
    float no_reference_steps;
    float lost;
    float chosen_changes;
    float drift_changes;
    float tracking_error_m;
    float speed_error_mps;
    float jerk_clip_long;
    float jerk_clip_lat;
    float steer_rate_saturated;
    float dist_mode_steps;
    float emergency_steps;
    float unfollowable_steps;
    float exit_decisions;
    float exit_nonstraight;
    float late_exit_decisions;
    float backups;
    float backup_m;
    float moving_steps;
    float first_motion_step;
    float rail_regens;
};

struct LatticeAgent {
    struct LatticeRail rail;
    struct LatticeLatPlan lat;
    struct LatticeLonPlan lon;
    int gear;
    double sigma_m;
    int lane_change_active;
    int rail_changed_flag;
    int has_reference;
    int projection_hint;
    int context_step;
    int live_split_slot;
    int late_exit_pending;
    int phantom_was_active;
    int neighbour_lane[2];
    float neighbour_offset_m[2];
    float neighbour_arc_m[2];
    struct LatticeRail neighbour_check_rail[2];
    int neighbour_check_lane[2];
    int neighbour_check_hint[2];
    float backup_duration_s[LATTICE_MAX_BACKUP_DISTANCES];
    float reverse_emergency_stop_m;
    short lon_cell_steps[LATTICE_MAX_LON_CELLS];
    float stop_line_distance_m;
    float preview_world_xy[AGENT_F32_PATH_SAMPLES][2];
    unsigned char mask[LATTICE_MAX_MASK_FEATURES];
    struct LatticeCounters counters;
};

void free_agent(struct Agent *agent) {
    free(agent->log_trajectory_x);
    free(agent->log_trajectory_y);
    free(agent->log_trajectory_z);
    free(agent->log_heading);
    free(agent->log_velocity_x);
    free(agent->log_velocity_y);
    free(agent->log_length);
    free(agent->log_width);
    free(agent->log_height);
    free(agent->log_valid);
    free(agent->route);
}

void free_road_element(struct RoadMapElement *element) {
    free(element->x);
    free(element->y);
    free(element->z);
    free(element->headings);
    free(element->entry_lanes);
    free(element->exit_lanes);
    free(element->cum_lengths);
}

void free_traffic_element(struct TrafficControlElement *element) {
    free(element->states);
    free(element->controlled_lanes);
}

void free_lane_graph(struct LaneGraph *graph) {
    free(graph->lane_ids);
    free(graph->distances);
    free(graph->lane_to_graph_idx);
}
