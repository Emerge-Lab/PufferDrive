#define UNKNOWN 0

// -- REWARD CONDITIONING COEFFICIENTS
#define REWARD_COEF_GOAL_RADIUS 0
#define REWARD_COEF_COLLISION 1
#define REWARD_COEF_OFFROAD 2
#define REWARD_COEF_COMFORT 3
#define REWARD_COEF_LANE_ALIGN 4
#define REWARD_COEF_VEL_ALIGN 5
#define REWARD_COEF_LANE_CENTER 6
#define REWARD_COEF_CENTER_BIAS 7
#define REWARD_COEF_VELOCITY 8
#define REWARD_COEF_REVERSE 9
#define REWARD_COEF_TRAFFIC_LIGHT 10
#define REWARD_COEF_TIMESTEP 11
#define REWARD_COEF_OVERSPEED 12
// Dynamic conditioning coefficients
#define REWARD_COEF_THROTTLE 13
#define REWARD_COEF_STEER 14
#define REWARD_COEF_ACC 15
#define NUM_REWARD_COEFS 16

// -- AGENT TYPE
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define OTHER 4

// -- ROAD TYPE
#define LANE_FREEWAY 1
#define LANE_SURFACE_STREET 2
#define LANE_BIKE_LANE 3

#define ROAD_LINE_UNKNOWN 10
#define ROAD_LINE_BROKEN_SINGLE_WHITE 11
#define ROAD_LINE_SOLID_SINGLE_WHITE 12
#define ROAD_LINE_SOLID_DOUBLE_WHITE 13
#define ROAD_LINE_BROKEN_SINGLE_YELLOW 14
#define ROAD_LINE_BROKEN_DOUBLE_YELLOW 15
#define ROAD_LINE_SOLID_SINGLE_YELLOW 16
#define ROAD_LINE_SOLID_DOUBLE_YELLOW 17
#define ROAD_LINE_PASSING_DOUBLE_YELLOW 18

#define ROAD_EDGE_UNKNOWN 20
#define ROAD_EDGE_BOUNDARY 21
#define ROAD_EDGE_MEDIAN 22
#define ROAD_EDGE_SIDEWALK 23

#define CROSSWALK 31
#define DRIVEWAY 32

// -- TRAFFIC CONTROL TYPE
#define TRAFFIC_LIGHT 1
#define STOP_SIGN 2
#define YIELD_SIGN 3
#define SPEED_LIMIT_SIGN 4

#define TL_STATE_DISABLED 0
#define TL_STATE_YELLOW 2
#define TL_STATE_GREEN 3
#define TL_STATE_RED 4
// Number of normalized traffic light states: disabled, red, yellow, green
#define NUM_TRAFFIC_LIGHT_STATES 4

// Metrics array indices
#define NUM_METRICS 18
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define RED_LIGHT_IDX 2
#define STOP_SIGN_IDX 3
#define REACHED_GOAL_IDX 4
#define LANE_DIST_IDX 5
#define LANE_ANGLE_IDX 6
#define COMFORT_VIOLATION_IDX 7
#define VELOCITY_PROGRESS_IDX 8
#define SPEED_LIMIT_IDX 9
#define AVG_DISPLACEMENT_ERROR_IDX 10
#define PROGRESSION_IDX 11
// Evaluation metrics
#define AT_FAULT_COLLISION_IDX 12
#define TTC_IDX 13
#define TTC_TFL_IDX 14
#define PROGRESS_RATIO_IDX 15
#define MULTI_LANE_TIME_IDX 16
#define MULTI_LANE_SCORE_IDX 17

// Path
#define MAX_NUM_WP_PATH 200
#define MAX_TARGET_WAYPOINTS 20

struct Waypoint {
    float s;           // Arc length (cumulative distance from the start) - init position
    float x;           // Global x-coordinate
    float y;           // Global y-coordinate
    float z;           // Global z-coordinate
    float heading;     // Global heading (tangent angle) in radians
    float cos_heading; // Cached cosf(heading) - set in build_path
    float sin_heading; // Cached sinf(heading) - set in build_path
    float kappa;       // Curvature at this point
    int lane_id;       // Lane id of the waypoint
};
struct Path {
    struct Waypoint waypoints[MAX_NUM_WP_PATH];
    int num_waypoints;
};

struct ttc_result {
    float min_ttc;
    int other_idx;
    float distance_to_collision;
    float closing_speed;
};

static inline int is_road_lane(int type) { return (type >= 1 && type <= 9); }

static inline int is_drivable_road_lane(int type) { return (type == LANE_FREEWAY || type == LANE_SURFACE_STREET); }

static inline int is_road_line(int type) { return (type >= 10 && type <= 19); }

static inline int is_road_edge(int type) { return (type >= 20 && type <= 29); }

static inline int is_road(int type) { return is_road_lane(type) || is_road_line(type) || is_road_edge(type); }

static inline int is_controllable_agent(int type) { return (type == VEHICLE || type == PEDESTRIAN || type == CYCLIST); }

static inline int is_traffic_light_red(int state) { return state == 1 || state == TL_STATE_RED || state == 7; }

static inline int is_traffic_light_green(int state) { return state == TL_STATE_GREEN || state == 6 || state == 9; }

static inline int is_traffic_light_yellow(int state) { return state == TL_STATE_YELLOW || state == 5 || state == 8; }

static inline int normalize_road_type(int type) {
    if (is_road_lane(type)) {
        return 0;
    } else if (is_road_line(type)) {
        return 1;
    } else if (is_road_edge(type)) {
        return 2;
    } else {
        return -1;
    }
}

static inline int unnormalize_road_type(int norm_type) {
    if (norm_type == 0) {
        return LANE_SURFACE_STREET;
    } else if (norm_type == 1) {
        return ROAD_LINE_BROKEN_SINGLE_WHITE;
    } else if (norm_type == 2) {
        return ROAD_EDGE_BOUNDARY;
    } else {
        return -1; // Invalid
    }
}

static inline int normalize_traffic_light_state(int state) {
    if (is_traffic_light_red(state)) {
        return 1;
    } else if (is_traffic_light_yellow(state)) {
        return 2;
    } else if (is_traffic_light_green(state)) {
        return 3;
    } else {
        return TL_STATE_DISABLED; // Invalid or disabled
    }
}

static inline int unnormalize_traffic_light_state(int norm_state) {
    if (norm_state == 1) {
        return TL_STATE_RED;
    } else if (norm_state == 2) {
        return TL_STATE_YELLOW;
    } else if (norm_state == 3) {
        return TL_STATE_GREEN;
    } else {
        return TL_STATE_DISABLED; // Invalid or disabled
    }
}

struct Agent {
    int id;
    int type;

    // Log trajectory
    int trajectory_length;
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

    // Simulation state (always center-based, even when trajectory control uses the rear axle)
    float sim_x; // Vehicle center x-coordinate
    float sim_y; // Vehicle center y-coordinate
    float sim_z;
    float sim_heading;
    float cos_heading; // Cached cosf(sim_heading) - updated in move_dynamics
    float sin_heading; // Cached sinf(sim_heading) - updated in move_dynamics
    float sim_vx;      // Vehicle center velocity x-component
    float sim_vy;      // Vehicle center velocity y-component
    float yaw_rate;    // Angular velocity used to derive rear-axle velocity
    float sim_speed;
    float sim_speed_signed;
    float sim_length;
    float sim_width;
    float sim_height;
    int sim_valid;

    // Route information
    int route_length;
    int *route;
    int route_gt_len;        // Number of leading route lanes supported by GT before extension
    int current_route_index; // Tracks progress through route array

    // Path
    struct Path *path;
    int closest_path_idx_wp;

    // Metrics and status tracking (size must match NUM_METRICS in drive.h)
    float metrics_array[NUM_METRICS]; // [collision, offroad, red_light, stop_sign, reached_goal, lane_dist, lane_angle,
                                      // comfort_violation, velocity_progress, speed_limit, avg_displacement_error,
                                      // progression, at_fault_collision, ttc, ttc_tfl, progress_ratio,
                                      // multi_lane_time, multi_lane_score]
    int current_lane_index;
    int previous_lane_index;
    int current_lane_geometry_idx;
    int reached_goal_this_episode;
    int num_waypoints_reached;
    int num_goals_reached;
    int active_agent;
    int mark_as_expert;
    float cumulative_displacement;
    int displacement_sample_count;
    float path_progression;
    float distance_since_spawn;

    // Goal positions (N sequential waypoints)
    float goal_positions_x[MAX_TARGET_WAYPOINTS];
    float goal_positions_y[MAX_TARGET_WAYPOINTS];
    float goal_positions_z[MAX_TARGET_WAYPOINTS];
    float goal_position_x; // alias = goal_positions_x[current_goal_idx]
    float goal_position_y; // alias = goal_positions_y[current_goal_idx]
    float goal_position_z; // alias = goal_positions_z[current_goal_idx]
    int current_goal_idx;  // index of next goal to reach (0..N-1)

    int stopped; // 0/1 -> freeze if set
    int removed; // 0/1 -> remove from sim if set

    // Jerk dynamics
    float a_long;
    float a_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;

    // Reward conditioning coefficients (per-agent, randomized at spawn)
    float reward_coefs[NUM_REWARD_COEFS];

    // Puffer score tracking (per-episode accumulators)
    struct ttc_result cached_ttc; // Filled once per step before reward computation
    float wrong_way_distance;     // Accumulated wrong-way distance
    float speed_violation_sum;    // For nuPlan speed compliance formula
    int ttc_violations;           // Count of TTC < 0.95s violations
    int ttc_samples;              // Total TTC samples for rate
    int at_fault_collision;       // 1 if at-fault collision occurred this episode
    float multi_lane_time;        // Accumulated time (s) on multiple lanes
};

struct RoadMapElement {
    int id;
    int type;

    int segment_length;
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
};

struct TrafficControlElement {
    int id;
    int type;

    int state_length;
    int *states;
    float stop_line[6]; // Two 3D endpoints: [x1,y1,z1, x2,y2,z2]
    float heading;
    int num_controlled_lanes;
    int *controlled_lanes;
};

typedef struct {
    float z_dis;
    float euclidean_dis;
    float z;
} DepthPoint;

struct LaneGraph {
    int n_lanes;
    int *lane_ids;
    float *lane_lengths;
    float *distances; // n_lanes * n_lanes row-major
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
    free(agent->path);
}

void free_road_element(struct RoadMapElement *element) {
    free(element->x);
    free(element->y);
    free(element->z);
    free(element->headings);
    free(element->entry_lanes);
    free(element->exit_lanes);
}

void free_traffic_element(struct TrafficControlElement *element) {
    free(element->states);
    free(element->controlled_lanes);
}

void free_lane_graph(struct LaneGraph *graph) {
    free(graph->lane_ids);
    free(graph->lane_lengths);
    free(graph->distances);
}
