#ifndef PUFFERLIB_OCEAN_DRIVE_CONSTANTS_H
#define PUFFERLIB_OCEAN_DRIVE_CONSTANTS_H

#include <math.h>

// Every simulation constant lives here. Two kinds, kept separate within each section:
//   - enum-like values: a closed set of modes/types selected by config or map data
//   - tuning values: physical units, thresholds, capacities
//
//  1. Entity types          agent + road element + traffic control types
//  2. Modes                 config-selected behavior switches
//  3. Dynamics              action spaces, actuator limits, speed
//  4. Geometry & collision  lane fitting, OBB checks, stop lines
//  5. Spatial grid          cell sizing and neighborhood queries
//  6. Routing & goals       route search, goal seeding, lane graph
//  7. Traffic lights        procedural light generation
//  8. Observations          per-entity feature counts
//  9. Rewards               conditioning coefficient indices
// 10. Metrics & scoring     metrics_array indices and scoring windows
// 11. Telemetry             html frame field counts, episode queue
// 12. Lattice               spline_werling action factors, rail, tracking and mask tuning

// =====================================================================================
// 1. ENTITY TYPES
// =====================================================================================

// -- AGENT TYPE
#define UNKNOWN 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3

// -- ROAD TYPE
// Ranges are load-bearing: is_road_lane/is_road_line/is_road_edge in datatypes.h
// classify by decade (0-9 lanes, 10-19 lines, 20-29 edges, 30+ misc).
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

// =====================================================================================
// 2. MODES
// =====================================================================================

// Which agents get initialized at reset
#define INIT_MODE_CREATE_ALL_VALID 0
#define INIT_MODE_CREATE_ONLY_CONTROLLED 1

// Which initialized agents the policy controls
#define CONTROL_MODE_VEHICLES 0
#define CONTROL_MODE_AGENTS 1
#define CONTROL_MODE_WOSAC 2
#define CONTROL_MODE_SDC_ONLY 3

// Per-agent controller
#define CONTROLLER_STATIC 0
#define CONTROLLER_POLICY 1
#define CONTROLLER_REPLAY 2
#define CONTROLLER_IDM 3

// Episode generation
#define SIMULATION_MODE_GIGAFLOW 0
#define SIMULATION_MODE_REPLAY 1

// What happens to an agent on collision/infraction
#define INFRACTION_BEHAVIOR_IGNORE 0
#define INFRACTION_BEHAVIOR_STOP 1
#define INFRACTION_BEHAVIOR_REMOVE 2

// Goal set refresh policy
#define GOAL_REGEN_FINITE 0  // regenerate the full goal set once all are reached
#define GOAL_REGEN_ROLLING 1 // slide window: drop reached goal, append one at the frontier

// Where goals are sampled from
#define GOAL_SOURCE_ROUTE 0 // seed from the agent's own forward route
#define GOAL_SOURCE_MAP 1   // seed from a uniformly sampled map lane
#define GOAL_SOURCE_GT 2    // seed directly from the logged ground-truth trajectory

// Dynamics model
#define DYNAMICS_MODEL_CLASSIC 0
#define DYNAMICS_MODEL_JERK 1
#define DYNAMICS_MODEL_SPLINE 2 // car follows the spline decoded from jerk-channel actions exactly
#define DYNAMICS_MODEL_SPLINE_WERLING 3 // committed Werling Frenet plans tracked through the jerk integrator

// Action representation
#define ACTION_TYPE_DISCRETE 0
#define ACTION_TYPE_CONTINUOUS 1
#define ACTION_TYPE_SPLINE 2
#define ACTION_TYPE_LATTICE 3

// Widest per-agent float action layout across all action_type modes (spline: p1/v1/a1 per
// axis). Used only by the C test fixture's allocate()-equivalent, which must size env->actions
// generously since tests set action_type after allocation, matching every existing test's
// calling convention — the real Python runtime path owns and sizes its own actions buffer.
#define MAX_ACTION_STRIDE_FLOATS 6

// Cap on the intent-consistency reward term's sampling loop (Phase B), derived
// from spline_horizon_seconds/dt at env-init and validated against this bound
// in config_schema.py before it ever reaches C.
#define SPLINE_CONSISTENCY_MAX_SAMPLES 64

// Depth of the per-agent spline history ring, bounding spline_consistency_lag_count.
#define SPLINE_CONSISTENCY_MAX_LAG 25

// =====================================================================================
// 3. DYNAMICS
// =====================================================================================

#define MAX_BACKWARD_SPEED -2.0f
#define STEERING_LIMIT 0.667f
static const float REAR_AXLE_RATIO = 0.5f;
static const float ACCEL_LONG_LIMIT[2] = {-5.0f, 2.5f};
static const float ACCEL_LAT_LIMIT[2] = {-4.0f, 4.0f};

#define NUM_JERK_LONG_ACTIONS 4
#define NUM_JERK_LAT_ACTIONS 3
static const float JERK_LONG[NUM_JERK_LONG_ACTIONS] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[NUM_JERK_LAT_ACTIONS] = {-4.0f, 0.0f, 4.0f};

// Spline dynamics: heading turns at most arc_length * tan(STEERING_ANGLE_LIMIT) / wheelbase per step
#define STEERING_ANGLE_LIMIT 0.55f
#define SPLINE_MIN_SPEED_FOR_HEADING_MPS 1e-3f
#define SPLINE_MIN_ARC_M 1e-4f
// Spline dynamics action per ego axis: (accel change over dt, jerk at T/2, jerk at T)
#define SPLINE_CHANNELS_PER_AXIS 3
// Baseline fit check: allowed float error as a fraction of the summed quintic term magnitudes
#define TRAJECTORY_FIT_RELATIVE_TOLERANCE 1e-4f

// Discrete action space, DYNAMICS_MODEL_CLASSIC
#define NUM_ACCELERATION_ACTIONS 7
#define NUM_STEERING_ACTIONS 9
static const float ACCELERATION_VALUES[NUM_ACCELERATION_ACTIONS]
    = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
static const float STEERING_VALUES[NUM_STEERING_ACTIONS]
    = {-0.667f, -0.500f, -0.333f, -0.167f, 0.000f, 0.167f, 0.333f, 0.500f, 0.667f};

// =====================================================================================
// 4. GEOMETRY & COLLISION
// =====================================================================================

#define INVALID_POSITION -10000.0f
#define EGO_IDX 0 // Ego agent is always at index 0 in the agents array

// Lane fitting: score = distance_weight * dist_term + heading_weight * heading_term
#define LANE_SELECTION_DISTANCE_WEIGHT 0.7f
#define LANE_SELECTION_HEADING_WEIGHT 0.3f
#define LANE_DISTANCE_NORMALIZATION 4.0f
#define LANE_SWITCH_THRESHOLD 0.05f // Hysteresis: new lane must be 5% better to switch
#define LANE_ALIGN_COS_THRESHOLD 0.5f
#define MAX_CHECKED_LANES 32
#define LANE_WIDTH 3.7f
#define LANE_MARGIN 0.2f

// Agent-agent collision
#define COLLISION_SKIP_DISP_M 0.1f
#define COLLISION_PAIR_MARGIN_M 0.5f // Extra slack on the radius+displacement quick-check before OBB SAT

// Stop line geometry
#define STOP_LINE_DIST_SQ (10.0f * 10.0f)
#define STOP_LINE_EXTENSION_FACTOR 1.5f
#define STOP_LINE_HEADING_THRESHOLD (M_PI / 4.0f)
#define STOP_SIGN_HEADING_TOLERANCE_RADIANS 1e-5f
#define STOP_SIGN_REQUIRED_STOP_DURATION_SECONDS 0.5f

#define BEHIND_COS_THRESHOLD -0.8660254f // cos(150 degrees)

// =====================================================================================
// 5. SPATIAL GRID
// =====================================================================================

#define GRID_CELL_SIZE 5.0f
#define MAX_GRID_CELL_COUNT 100000000
// Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution)
// => For each entity type in gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2
#define MAX_ENTITIES_PER_CELL 30
#define ROAD_QUERY_ENTITY_COUNT (MAX_ENTITIES_PER_CELL * 25) // 5x5 cell neighborhood

// 5x5 cell neighborhood swept by a road query, centered on the agent's cell
static const int ROAD_OFFSETS[25][2]
    = {{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1},
       {2, -1},  {-2, 0},  {-1, 0}, {0, 0},  {1, 0},  {2, 0},   {-2, 1},  {-1, 1}, {0, 1},
       {1, 1},   {2, 1},   {-2, 2}, {-1, 2}, {0, 2},  {1, 2},   {2, 2}};

// 2.5D Z estimation
#define Z_BUFFER 4.0f
#define Z_NUM_PT_AVG 30

// =====================================================================================
// 6. ROUTING & GOALS
// =====================================================================================

#define MAX_GOALS 20
#define MAX_ROUTE_LENGTH 64
#define ROUTE_TARGET_DISTANCE 1000.0f
#define ROUTE_EXIT_MAX_CANDIDATES 5
#define GT_GOAL_RADIUS_M 6.0f
#define LANE_GRAPH_DISTANCE_NORM_M 500.0f // normalization for the GPS lane-distance feature

// =====================================================================================
// 7. TRAFFIC LIGHTS
// =====================================================================================

#define TL_DEFAULT_RED_DURATION 2.0f
#define TL_DEFAULT_YELLOW_DURATION 3.0f
#define TL_DEFAULT_GREEN_DURATION 10.0f
#define TL_EPISODE_DISABLE_PROB 0.20f
#define TL_INDIVIDUAL_REMOVE_PROB 0.20f
#define TL_ALWAYS_GREEN_PROB 0.05f

// =====================================================================================
// 8. OBSERVATIONS
// =====================================================================================

#define EGO_FEATURES 10
#define LANE_FEATURES 9
#define BOUNDARY_FEATURES 9
#define PARTNER_FEATURES 9
#define TRAFFIC_CONTROL_FEATURES 7
#define GOAL_FEATURES 3
#define OBS_VALID_COUNT_FEATURES 4
// (p, v, a) x (fwd, left) of the spline target the policy last committed to; spline mode only.
#define SPLINE_INTENT_FEATURES 6
// Heading deviation since last kept point that forces a keep when obs stride > 1 (~15 degrees).
#define OBS_STRIDE_HEADING_THRESHOLD 0.2618f

// =====================================================================================
// 9. REWARDS
// =====================================================================================

// Indices into Agent.reward_coefs, randomized per-agent at spawn
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
#define REWARD_COEF_SPEED 17
#define REWARD_COEF_TRAJECTORY_CONSISTENCY 18
#define NUM_REWARD_COEFS 19

// =====================================================================================
// 10. METRICS & SCORING
// =====================================================================================

// Indices into Agent.metrics_array; NUM_METRICS must stay equal to the index count
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

// Time to collision
#define DEFAULT_TTC 5.0f              // when no vehicle ahead
#define TTC_VIOLATION_THRESHOLD 0.95f // "within bound" rate

// Distance to collision
#define DTC_FRONT_CONE_COS_THRESHOLD -0.90f       // 340 degree cone centered on ego heading
#define DTC_OPPOSITE_HEADING_COS_THRESHOLD -0.90f // Exclude near-perfect opposite directions
#define DEFAULT_DTC 50.0f                         // Ignore candidates beyond this range

// Multi-lane occupancy
#define MULTI_LANE_THRESHOLD (LANE_WIDTH / 2.0f + LANE_MARGIN) // 2.05m
#define MULTI_LANE_FULL_SCORE_TIME 3.4f                        // seconds
#define MULTI_LANE_HALF_SCORE_TIME 5.7f                        // seconds

// Stopped-agent detection
#define AGENT_STOPPED_SPEED_THRESHOLD 0.2f
#define MAX_STOPPED_SECONDS 60.0f

#define METRIC_SCORE_WINDOW_SECONDS 10.0f
#define PUFFER_PROGRESS_REFERENCE_SPEED 10.0f

// =====================================================================================
// 11. TELEMETRY
// =====================================================================================

// obs_html_frame array field counts
#define AGENT_F32_GOAL_RADIUS_IDX 12
// Predicted-path samples, world-frame (x,y) pairs: ego-only for spline/baseline, every policy agent for lattice.
#define AGENT_F32_PATH_BASE_IDX 13
#define AGENT_F32_PATH_SAMPLES 16
#define AGENT_F32_FIELDS                                                                                               \
    (13 + 2 * AGENT_F32_PATH_SAMPLES) // sim_x/y/z, heading, length, width, speed, steering, accel_long, accel_lat,
                                       // jerk_long, jerk_lat, goal_radius, then AGENT_F32_PATH_SAMPLES (x,y) pairs
#define AGENT_I32_FIELDS                                                                                               \
    10 // id, type, sim_valid, active_agent, stopped, removed, lane_idx, active_idx, blindness_active, braking_active
#define GOAL_XY_FIELDS 2               // goal x, y per goal slot; reached slots stay zeroed
#define METRICS_F32_FIELDS NUM_METRICS // must equal NUM_METRICS
#define SCORE_F32_FIELDS 15            // Log struct fields: puffer_score .. weighted_average
#define TRAFFIC_I16_FIELDS 3           // is_valid, type, state
#define REWARD_F32_EPISODE_RETURN_IDX 0
#define REWARD_F32_COLLISION_IDX 1
#define REWARD_F32_OFFROAD_IDX 2
#define REWARD_F32_RED_LIGHT_IDX 3
#define REWARD_F32_STOP_SIGN_IDX 4
#define REWARD_F32_GOAL_IDX 5
#define REWARD_F32_LANE_ALIGN_IDX 6
#define REWARD_F32_LANE_CENTER_IDX 7
#define REWARD_F32_COMFORT_IDX 8
#define REWARD_F32_VELOCITY_IDX 9
#define REWARD_F32_TIMESTEP_IDX 10
#define REWARD_F32_REVERSE_IDX 11
#define REWARD_F32_OVERSPEED_IDX 12
#define REWARD_F32_ADE_IDX 13
#define REWARD_F32_FIELDS 14

// =====================================================================================
// 12. LATTICE (DYNAMICS_MODEL_SPLINE_WERLING + ACTION_TYPE_LATTICE)
// =====================================================================================

// -- Action factors: [lat gate, lat cell, lon gate, lon cell, exit slot]; index 0 of every factor is neutral
#define LATTICE_ACTION_FACTORS 5
#define LATTICE_FACTOR_LAT_GATE 0
#define LATTICE_FACTOR_LAT_CELL 1
#define LATTICE_FACTOR_LON_GATE 2
#define LATTICE_FACTOR_LON_CELL 3
#define LATTICE_FACTOR_EXIT 4
#define LATTICE_GATE_KEEP 0
#define LATTICE_GATE_NEW 1
#define LATTICE_GATE_COUNT 2
#define LATTICE_EXIT_SLOTS 5

// -- Lateral plan modes and kinds
#define LATTICE_LAT_MODE_TIME 0
#define LATTICE_LAT_MODE_DIST 1
#define LATTICE_LAT_KIND_OFFSET 0
#define LATTICE_LAT_KIND_LANE_CHANGE 1
#define LATTICE_LAT_KIND_HOLD 2

// -- Longitudinal plan kinds
#define LATTICE_LON_KIND_SPEED 0
#define LATTICE_LON_KIND_STOP 1
#define LATTICE_LON_KIND_STOP_LINE 2
#define LATTICE_LON_KIND_BACKUP 3
#define LATTICE_LON_KIND_EMERGENCY 4

// -- Exit mode
#define LATTICE_EXIT_MODE_POLICY 0
#define LATTICE_EXIT_MODE_GOAL 1

// -- Rail change kinds
#define LATTICE_RAIL_CHANGE_NONE 0
#define LATTICE_RAIL_CHANGE_CHOSEN 1
#define LATTICE_RAIL_CHANGE_DRIFT 2
#define LATTICE_RAIL_CHANGE_LOST 3
#define LATTICE_RAIL_CHANGE_REBUILD 4

// -- Menu capacities (config lists are validated against these)
#define LATTICE_MAX_LAT_OFFSETS 5
#define LATTICE_MAX_LAT_DURATIONS 6
#define LATTICE_MAX_LON_SPEEDS 12
#define LATTICE_MAX_LON_DURATIONS 6
#define LATTICE_MAX_STOP_DISTANCES 6
#define LATTICE_MAX_BACKUP_DISTANCES 6
#define LATTICE_MAX_LAT_CELLS ((LATTICE_MAX_LAT_OFFSETS + 3) * LATTICE_MAX_LAT_DURATIONS)
#define LATTICE_MAX_LON_CELLS                                                                                          \
    (LATTICE_MAX_LON_SPEEDS * LATTICE_MAX_LON_DURATIONS + LATTICE_MAX_STOP_DISTANCES + 2 + LATTICE_MAX_BACKUP_DISTANCES)
#define LATTICE_MAX_MASK_FEATURES                                                                                      \
    (2 * LATTICE_GATE_COUNT + LATTICE_MAX_LAT_CELLS + LATTICE_MAX_LON_CELLS + LATTICE_EXIT_SLOTS)
#define LATTICE_DURATION_TOLERANCE 1e-4f // |T/dt - lround(T/dt)| allowed for every menu duration
#define LATTICE_MAX_PLAN_STEPS 400

// -- Observation layout
#define LATTICE_PLAN_FEATURES 60
#define LATTICE_PREVIEW_POINTS 6
#define LATTICE_PREVIEW_SUBSTEP_S 0.2f
#define LATTICE_PREVIEW_SUBSTEPS 30
#define LATTICE_PREVIEW_OBS_STRIDE 5    // substeps per observed preview point (1 s)
#define LATTICE_PREVIEW_RENDER_STRIDE 2 // substeps per rendered path sample (0.4 s)
#define LATTICE_OBS_DIST_NORM_M 50.0f
#define LATTICE_OBS_TIME_NORM_S 6.0f
#define LATTICE_OBS_SPEED_NORM_MPS 20.0f
#define LATTICE_OBS_CURVATURE_SCALE_M 10.0f
#define LATTICE_OBS_PREVIEW_NORM_M 100.0f
#define LATTICE_OBS_ROAD_AHEAD_NEAR_M 10.0f
#define LATTICE_OBS_ROAD_AHEAD_MID_M 25.0f
#define LATTICE_OBS_ROAD_AHEAD_FAR_M 50.0f
#define LATTICE_OBS_MARGIN_WINDOW_M 20.0f
#define LATTICE_OBS_ENVELOPE_WINDOW_M 50.0f
#define LATTICE_OBS_STOP_LINE_NORM_M 100.0f
#define LATTICE_OBS_ROUTE_GAP_NORM_M 200.0f // extra route over the best exit that saturates the feature
#define LATTICE_OBS_ROUTE_LOG_SCALE_M 50.0f
#define LATTICE_OBS_ROUTE_LOG_MAX_M 5000.0f
#define LATTICE_CONSISTENCY_SAMPLES 5 // first render-path samples compared after a plan change (0.6 s apart at dt 0.3)
#define LATTICE_ROUTE_PROGRESS_CAP_M 1000.0f // route distance beyond this is flat in the progress potential
#define LATTICE_WAIT_FULL_SPEED_MPS 1.0f     // the waiting penalty fades out linearly up to this speed

// -- Rail (per agent): samples at LATTICE_RAIL_SPACING_M, heading box filter over +-half window
#define LATTICE_RAIL_SAMPLES 512
#define LATTICE_RAIL_SPACING_M 0.5f
#define LATTICE_SMOOTH_HALF_WINDOW 4
#define LATTICE_CHAIN_MAX_LANES 24
#define LATTICE_TRAIL_BEHIND_M 20.0f
#define LATTICE_TRAIL_KEEP_M 30.0f
#define LATTICE_EXIT_FREEZE_M 160.0f
#define LATTICE_LOOKAHEAD_FREEZE_PAD_M 40.0f
#define LATTICE_LOOKAHEAD_MIN_M 60.0f
#define LATTICE_LOOKAHEAD_HORIZON_S 6.0f
#define LATTICE_LOOKAHEAD_SPEED_PAD_M 20.0f
#define LATTICE_LOOKAHEAD_HYSTERESIS_M 20.0f
#define LATTICE_JOIN_GAP_MAX_M 0.5f
#define LATTICE_MIN_SEGMENT_M 1e-3f
#define LATTICE_MAX_PREDECESSORS 8
#define LATTICE_BASE_LANE_COS 0.5f
#define LATTICE_BASE_LANE_RANGE_M (2.0f * LANE_WIDTH)
#define LATTICE_PROJECTION_WINDOW_SAMPLES 24
#define LATTICE_MAX_RAIL_VERTICES 4096
#define LATTICE_MAX_RAW_SAMPLES 1024
#define LATTICE_BUILD_PAD_M 10.0f
#define LATTICE_CHECK_RAIL_BEHIND_M 10.0f
#define LATTICE_CHECK_RAIL_PAD_M 65.0f // speed gain within the 6 s keep horizon at 2.5 m/s^2, plus margin
#define LATTICE_CHECK_RAIL_REUSE_SLACK_M 20.0f
#define LATTICE_STOP_PROFILE_SAMPLES 20
#define LATTICE_STOP_FALLBACK_CANDIDATES 2

// -- Map profiles (per lane, every LATTICE_PROFILE_SPACING_M)
#define LATTICE_PROFILE_SPACING_M 2.0f
#define LATTICE_EDGE_SEARCH_M 12.0f
#define LATTICE_NEIGHBOUR_MIN_M 2.5f
#define LATTICE_NEIGHBOUR_MAX_M 4.5f
#define LATTICE_NEIGHBOUR_COS 0.9f
#define LATTICE_NEIGHBOUR_TIEBREAK_M 15.0f
#define LATTICE_MAX_PROFILE_SAMPLES_PER_LANE 4096

// -- Borrowing the oncoming lane (one lane per direction): offered, kept and forced back by the lane ahead
#define LATTICE_BORROW_FRACTION 0.5f // past this share of the oncoming lane's offset the car is in that lane
#define LATTICE_BORROW_MIN_D_M (LATTICE_BORROW_FRACTION * LATTICE_NEIGHBOUR_MIN_M)
#define LATTICE_ONCOMING_START_MIN_M 40.0f
#define LATTICE_ONCOMING_START_S 6.0f
#define LATTICE_ONCOMING_KEEP_MIN_M 20.0f
#define LATTICE_ONCOMING_KEEP_S 4.0f

// -- Rail changes
#define LATTICE_DRIFT_MARGIN_M 0.3f
#define LATTICE_LOST_OFFSET_M (2.0f * LANE_WIDTH)
#define LATTICE_LOST_COS 0.5f

// -- Tracking (feedforward + clamped PD, then implied jerk into the jerk integrator)
#define LATTICE_KP_LAT 2.25f
#define LATTICE_KD_LAT 3.0f
#define LATTICE_E_D_MAX_M 0.5f
#define LATTICE_E_DD_MAX_MPS 1.0f
#define LATTICE_K_SPEED 1.0f
#define LATTICE_K_SIGMA 0.25f
#define LATTICE_E_SIGMA_MAX_M 2.0f
#define LATTICE_LOW_SPEED_MPS 3.0f
#define LATTICE_FEEDBACK_FULL_SPEED_MPS 1.0f
#define LATTICE_K_D_SPATIAL 0.3f
#define LATTICE_K_HEADING_SPATIAL 1.0f
#define LATTICE_WERLING_TOL_D_M 0.3f
#define LATTICE_WERLING_TOL_DD_MPS 0.5f
#define LATTICE_WERLING_TOL_V_MPS 0.5f
#define LATTICE_WERLING_TOL_SIGMA_M 0.3f
#define LATTICE_REPLAN_D_M 0.5f
#define LATTICE_REPLAN_DD_MPS 1.0f
#define LATTICE_REPLAN_ACCEL_MARGIN 0.85f
#define LATTICE_REPLAN_T_STEP_S 0.3f
#define LATTICE_REPLAN_T_MAX_S 12.0f
#define LATTICE_RELEASE_SPEED_MPS 0.2f
#define LATTICE_EMERGENCY_JERK_STEP 0.25f
#define LATTICE_EMERGENCY_MAX_RELEASE_STEPS 200
#define LATTICE_EMERGENCY_MIN_DIST_M 5.0f
#define LATTICE_STOPPED_SPEED_MPS 1e-4f
#define LATTICE_STOPPED_ACCEL_MPS2 1e-4f

// -- Plan menus: durations searched for stop / back-up / automatic plans
#define LATTICE_STOP_T_GRID_S 1.2f
#define LATTICE_STOP_T_MAX_S 12.0f
#define LATTICE_BACKUP_T_GRID_S 0.3f
#define LATTICE_BACKUP_T_MAX_S 12.0f
#define LATTICE_COMFORT_ACCEL_MPS2 3.0f
#define LATTICE_COMFORT_BRAKE_JERK -5.0f
#define LATTICE_LOW_SPEED_CURVATURE_PEAK 5.77f // max |d''| * S^2 / D of the rest-to-rest quintic

// -- Masks
#define LATTICE_ENVELOPE_MARGIN 0.85f
#define LATTICE_ENVELOPE_BRAKE_MPS2 2.5f
#define LATTICE_ENVELOPE_CURVATURE_HALF_WINDOW 8
#define LATTICE_ENVELOPE_TOLERANCE_MPS 0.05f
#define LATTICE_MASK_SAMPLES 10
#define LATTICE_KEEP_CHECK_S 6.0f
#define LATTICE_UNFOLLOWABLE_TOLERANCE 0.02f
#define LATTICE_MIN_FRENET_FACTOR 0.2f
#define LATTICE_NOMINAL_LAUNCH_SPEED_MPS 5.0f
#define LATTICE_NOMINAL_LAUNCH_T_S 3.6f
#define LATTICE_STEER_RATE_RPS 0.6f
#define LATTICE_STOP_LINE_MARGIN_M 1.0f
#define LATTICE_BACKUP_LANDING_M 0.2f
#define LATTICE_REVERSE_EMERGENCY_MAX_M 2.93f // from -2 m/s at a = -1.5 m/s^2, weakest c
#define LATTICE_SPAWN_MAX_LENGTH_M 7.0f       // gigaflow spawn length maximum (drive.h spawn sampling)
#define LATTICE_CHECK_EPS 1e-6f
#define LATTICE_GEOMETRY_EPS 1e-6f
#define LATTICE_MIN_RATE_MPS 1e-3f
#define LATTICE_REVERSAL_TOLERANCE_MPS 1e-6f
#define LATTICE_TWO_STEP_EPS 1e-9f
#define LATTICE_FILLET_MIN_TURN_RAD 1e-4f
#define LATTICE_FILLET_MAX_TURN_RAD 3.1f
#define LATTICE_MIN_BACKUP_DISTANCE_M 0.1f

#endif
