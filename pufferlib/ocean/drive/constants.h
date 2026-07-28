#ifndef PUFFERLIB_OCEAN_DRIVE_CONSTANTS_H
#define PUFFERLIB_OCEAN_DRIVE_CONSTANTS_H

#include <math.h>

// -- REWARD CONDITIONING COEFFICIENTS
#define REWARD_COEF_GOAL_RADIUS 0
#define REWARD_COEF_GOAL_SPEED 1
#define REWARD_COEF_COLLISION 2
#define REWARD_COEF_OFFROAD 3
#define REWARD_COEF_COMFORT 4
#define REWARD_COEF_LANE_ALIGN 5
#define REWARD_COEF_VEL_ALIGN 6
#define REWARD_COEF_LANE_CENTER 7
#define REWARD_COEF_CENTER_BIAS 8
#define REWARD_COEF_VELOCITY 9
#define REWARD_COEF_REVERSE 10
#define REWARD_COEF_STOP_LINE 11
#define REWARD_COEF_TIMESTEP 12
#define REWARD_COEF_OVERSPEED 13
// Dynamic conditioning coefficients
#define REWARD_COEF_THROTTLE 14
#define REWARD_COEF_STEER 15
#define REWARD_COEF_ACC 16
#define NUM_REWARD_COEFS 17

// -- AGENT TYPE
#define UNKNOWN 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3

// -- ROAD TYPE
#define LANE_UNKNOWN 0
#define LANE_FREEWAY 1
#define LANE_SURFACE_STREET 2
#define LANE_BIKE_LANE 3
#define LANE_BUS_LANE 4

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

#define MISC_UNKNOWN 30
#define MISC_CROSSWALK 31
#define MISC_SPEED_BUMP 32

// -- TRAFFIC CONTROL TYPE
#define TRAFFIC_CONTROL_TYPE_NONE 0
#define TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT 1
#define TRAFFIC_CONTROL_TYPE_STOP_SIGN 2
#define TRAFFIC_CONTROL_TYPE_YIELD_SIGN 3
#define NUM_TRAFFIC_CONTROL_TYPES 4
// -- TRAFFIC CONTROL STATE
#define TRAFFIC_CONTROL_STATE_UNKNOWN 0
#define TRAFFIC_CONTROL_STATE_RED 1
#define TRAFFIC_CONTROL_STATE_YELLOW 2
#define TRAFFIC_CONTROL_STATE_GREEN 3
#define TRAFFIC_CONTROL_STATE_OFF 4
#define NUM_TRAFFIC_CONTROL_STATES 5

#define TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS 0
#define TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS_STOP_SIGN 1
#define TRAFFIC_CONTROL_SCOPE_ALL 2

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
#define DISTANCE_TO_COLLISION_IDX 14
#define PROGRESS_RATIO_IDX 15
#define MULTI_LANE_TIME_IDX 16
#define MULTI_LANE_SCORE_IDX 17

#define MAX_GOALS 20

// obs_html_frame array field counts
#define AGENT_F32_FIELDS                                                                                               \
    12 // sim_x/y/z, heading, length, width, speed, steering, accel_long, accel_lat, jerk_long, jerk_lat
#define AGENT_I32_FIELDS 8             // id, type, sim_valid, active_agent, stopped, removed, lane_idx, active_idx
#define METRICS_F32_FIELDS NUM_METRICS // must equal NUM_METRICS
#define SCORE_F32_FIELDS 15            // Log struct fields: puffer_score .. weighted_average
#define TRAFFIC_I16_FIELDS 3           // is_valid, type, state

// Render modes (selected by drive.py's render_mode kwarg, plumbed via binding.c)
#define RENDER_WINDOW 0
#define RENDER_HEADLESS 1

#define INVALID_POSITION -10000.0f
// Ego agent is always at index 0 in the agents array
#define EGO_IDX 0

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
#define EVAL_PERCEIVED_SIZE_MARGIN_M 0.1f // inflate ego's perceived size by this per side for safety clearance
#define COLLISION_SKIP_DISP_M 0.1f
#define COLLISION_PAIR_MARGIN_M 0.5f // Extra slack on the radius+displacement quick-check before OBB SAT
#define MAX_CHECKED_LANES 32
#define AGENT_STOPPED_SPEED_THRESHOLD 0.2f
#define MAX_STOPPED_SECONDS 60.0f
#define DTC_FRONT_CONE_COS_THRESHOLD -0.90f       // 340 degree cone centered on ego heading
#define DTC_OPPOSITE_HEADING_COS_THRESHOLD -0.90f // Exclude near-perfect opposite directions
#define DEFAULT_DTC 50.0f                         // Ignore candidates beyond this range
#define STOP_LINE_DIST_SQ (10.0f * 10.0f)
#define STOP_LINE_EXTENSION_FACTOR 1.5f
#define STOP_LINE_HEADING_THRESHOLD (M_PI / 4.0f)
#define BEHIND_COS_THRESHOLD -0.8660254f // cos(150 degrees)

// TTC default value when no vehicle ahead
#define DEFAULT_TTC 5.0f
// TTC violation threshold for "within bound" rate
#define TTC_VIOLATION_THRESHOLD 0.95f
#define METRIC_SCORE_WINDOW_SECONDS 10.0f
#define PUFFER_PROGRESS_REFERENCE_SPEED 10.0f
// Multi-lane detection thresholds
#define LANE_WIDTH 3.7f
#define LANE_MARGIN 0.2f
#define MULTI_LANE_THRESHOLD (LANE_WIDTH / 2.0f + LANE_MARGIN) // 2.05m
#define MULTI_LANE_FULL_SCORE_TIME 3.4f                        // seconds
#define MULTI_LANE_HALF_SCORE_TIME 5.7f                        // seconds

// Collision/Infraction behaviors
#define STOP_AGENT 1
#define REMOVE_AGENT 2

#define MAX_SPEED 40.0f

// Grid cell size
#define GRID_CELL_SIZE 5.0f
// Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution)
// => For each entity type in gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2
#define MAX_ENTITIES_PER_CELL 30
// Heading deviation since last kept point that forces a keep when obs stride > 1 (~15 degrees).
#define OBS_STRIDE_HEADING_THRESHOLD 0.2618f
#define ROAD_QUERY_ENTITY_COUNT (MAX_ENTITIES_PER_CELL * 25)

// GOAL_REGEN modes
#define GOAL_REGEN_FINITE 0  // regenerate the full goal set once all are reached
#define GOAL_REGEN_ROLLING 1 // slide window: drop reached goal, append one at the frontier
// GOAL_SOURCE modes
#define GOAL_SOURCE_ROUTE 0               // seed from the agent's own forward route
#define GOAL_SOURCE_MAP 1                 // seed from a uniformly sampled map lane
#define GOAL_SOURCE_GT 2                  // seed directly from the logged ground-truth trajectory
#define LANE_GRAPH_DISTANCE_NORM_M 500.0f // normalization for the GPS lane-distance feature
#define MAX_ROUTE_LENGTH 64
#define ROUTE_TARGET_DISTANCE 1000.0f
#define ROUTE_EXIT_MAX_CANDIDATES 5
#define GT_GOAL_RADIUS_M 6.0f

// Observation feature counts
#define EGO_FEATURES 10
#define LANE_FEATURES 9
#define BOUNDARY_FEATURES 9
#define PARTNER_FEATURES 9
#define TRAFFIC_CONTROL_FEATURES 7
#define GOAL_FEATURES 3
#define OBS_VALID_COUNT_FEATURES 4

// Traffic light generation
#define TL_DEFAULT_RED_DURATION 2.0f
#define TL_DEFAULT_YELLOW_DURATION 3.0f
#define TL_DEFAULT_GREEN_DURATION 10.0f
#define TL_EPISODE_DISABLE_PROB 0.20f
#define TL_INDIVIDUAL_REMOVE_PROB 0.20f
#define TL_ALWAYS_GREEN_PROB 0.05f

// 2.5D Z estimation
#define Z_BUFFER 4.0f
#define Z_NUM_PT_AVG 30

static const int ROAD_OFFSETS[25][2]
    = {{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1},
       {2, -1},  {-2, 0},  {-1, 0}, {0, 0},  {1, 0},  {2, 0},   {-2, 1},  {-1, 1}, {0, 1},
       {1, 1},   {2, 1},   {-2, 2}, {-1, 2}, {0, 2},  {1, 2},   {2, 2}};

// Dynamics Models
#define CLASSIC 0
#define JERK 1

static const float ACCEL_LONG_LIMIT[2] = {-5.0f, 2.5f};
static const float ACCEL_LAT_LIMIT[2] = {-4.0f, 4.0f};
#define STEERING_LIMIT 0.667f
static const float REAR_AXLE_RATIO = 0.5f;

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space (for CLASSIC dynamics model)
static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
static const float STEERING_VALUES[9] = {-0.667f, -0.500f, -0.333f, -0.167f, 0.000f, 0.167f, 0.333f, 0.500f, 0.667f};

#define COMPLETED_EPISODE_QUEUE_CAPACITY 16

#endif
