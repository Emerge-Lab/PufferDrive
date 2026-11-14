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
#define INIT_MODE_CREATE_CONTROLLABLE_TYPES 1

// Which initialized agents the policy controls
#define CONTROL_MODE_VEHICLES 0
#define CONTROL_MODE_SDC_ONLY 1

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

// Goal set refresh policy
#define GOAL_REGEN_FINITE 0  // regenerate the full goal set once all are reached
#define GOAL_REGEN_ROLLING 1 // slide window: drop reached goal, append one at the frontier

// Where goals are sampled from
#define GOAL_SOURCE_ROUTE 0    // seed from the agent's own forward route
#define GOAL_SOURCE_MAP 1      // seed from a uniformly sampled map lane
#define GOAL_SOURCE_GT 2       // seed directly from the logged ground-truth trajectory
#define GOAL_SOURCE_EXTERNAL 3 // co-sim: goal windows pushed by the external sim (c_set_agent_goals)
#define GOAL_SOURCE_GT_MAP 4   // logged-trajectory goals projected onto the nearest co-directional lane center

// =====================================================================================
// 3. DYNAMICS
// =====================================================================================

#define MAX_BACKWARD_SPEED -2.0f
#define STEERING_LIMIT 0.667f
static const float ACCEL_LONG_LIMIT[2] = {-5.0f, 2.5f};
static const float ACCEL_LAT_LIMIT[2] = {-4.0f, 4.0f};

#define NUM_JERK_LONG_ACTIONS 4
#define NUM_JERK_LAT_ACTIONS 3
static const float JERK_LONG[NUM_JERK_LONG_ACTIONS] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[NUM_JERK_LAT_ACTIONS] = {-4.0f, 0.0f, 4.0f};

// exported for alpasim-training, which reads the steering obs normalizer as STEERING_VALUES[-1]
#define NUM_STEERING_ACTIONS 9
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
#define LANE_CURVATURE_NORM 0.2f       // 1/m; a 5 m turn radius saturates the obs
#define LANE_SCAN_GUARD_MARGIN_M 0.01f // float slack so an unlisted segment never ties the cached nearest one
#define LANE_WIDTH 3.7f                // fallback when the map carries no per-point widths
#define MAX_LANE_WIDTH_M 50.0f
#define LANE_MARGIN 0.2f

// Agent-agent collision
#define COLLISION_SKIP_DISP_M 0.1f
#define COLLISION_PAIR_MARGIN_M 0.5f        // Extra slack on the radius+displacement quick-check before OBB SAT
#define SPAWN_CLEARANCE_M 1.5f              // Min box-to-box gap between agents at spawn
#define COSIM_PARTNER_DEFAULT_LENGTH_M 4.5f // co-sim partner slot box until the external sim sets sizes
#define COSIM_PARTNER_DEFAULT_WIDTH_M 2.0f
#define COSIM_PARTNER_DEFAULT_HEIGHT_M 1.5f
#define SPAWN_OFFROAD_SCALE_FACTOR 1.1f
#define EVAL_SPAWN_EDGE_CLEARANCE_M 0.5f // Rear swing on a first full-lock turn must not clip the road edge
// Replay self-play: logged vehicles failing these are created static instead of policy-controlled
#define REPLAY_SPAWN_EDGE_CLEARANCE_M 0.5f                      // Car box must stay this far from the road edge
#define REPLAY_SPAWN_LONGITUDINAL_CLEARANCE_M SPAWN_CLEARANCE_M // Gap needed to other cars in front of and behind
#define REPLAY_MAX_CONTROLLED_LENGTH_M 6.0f                     // Longer vehicles (trucks, buses) are made static
#define REPLAY_SPAWN_MAX_LANE_DISTANCE_M 2.5f                   // Car must be this close to a lane center line

// Scripted pedestrians: one behaviour per appearance, drawn with the PEDESTRIAN_*_PROB shares
#define PEDESTRIAN_STATE_IDLE 0        // hidden, cooldown running
#define PEDESTRIAN_STATE_WALKING 1     // stands where it appeared, then walks its leg
#define PEDESTRIAN_STATE_STANDING 2    // leg done: stands until it vanishes
#define PEDESTRIAN_MODE_POP_IN_CROSS 0 // CARLA Scenario3: pops in at the right kerb close ahead, crosses fast
#define PEDESTRIAN_MODE_SLOW_CROSS 1   // crosses at walking speed from either kerb, visible from afar
#define PEDESTRIAN_MODE_STAND 2        // stands beside the road, never enters it
#define PEDESTRIAN_MODE_KERB_WALK 3    // walks along the kerb, never enters the road
#define PEDESTRIAN_MODE_KERB_STEP 4    // pops in like a crosser, steps toward or away from the road, then stands
#define PEDESTRIAN_POP_IN_CROSS_PROB 0.40f
#define PEDESTRIAN_SLOW_CROSS_PROB 0.15f
#define PEDESTRIAN_STAND_PROB 0.20f
#define PEDESTRIAN_KERB_WALK_PROB 0.15f // the rest: PEDESTRIAN_MODE_KERB_STEP
// Leaderboard 1.0 Scenario3 is the centre of every LB1 range; lanes = driving lanes from the car to the kerb, >= 1
#define PEDESTRIAN_LB1_LANE_WIDTH_M 3.5f
#define PEDESTRIAN_LB1_APPEAR_BASE_M 12.0f // walker pops in 12 m + 1.5 m per extra lane ahead of the car
#define PEDESTRIAN_LB1_APPEAR_PER_LANE_M 1.5f
#define PEDESTRIAN_APPEAR_JITTER_M 2.0f
#define PEDESTRIAN_KERB_OFFSET_MIN_M 0.75f // LB1 stands half a lane (1.75 m) outside the road edge
#define PEDESTRIAN_KERB_OFFSET_MAX_M 2.75f
#define PEDESTRIAN_KERB_SEARCH_MAX_M 10.0f   // lateral reach per side; wider roads get no crossing
#define PEDESTRIAN_FAR_KERB_OVERSHOOT_M 1.0f // walk this far past the far road edge
#define PEDESTRIAN_LB1_SPEED_BASE_MPS 3.4f   // LB1 walks at 3 + 0.4 m/s x (lanes + 1)
#define PEDESTRIAN_LB1_SPEED_PER_LANE_MPS 0.4f
#define PEDESTRIAN_SPEED_JITTER_MPS 1.0f
#define PEDESTRIAN_WAIT_MIN_S 0.0f // LB1 walker stands ~0.25 s after popping in
#define PEDESTRIAN_WAIT_MAX_S 0.6f
#define PEDESTRIAN_RAMP_MIN_S 0.3f // LB1 walker reaches its speed ~0.5 s after setting off
#define PEDESTRIAN_RAMP_MAX_S 0.7f
#define PEDESTRIAN_SIZE_MIN_M 1.0f // CARLA walker box (0.4-0.5 m) plus the 0.8 m eval VRU pad
#define PEDESTRIAN_SIZE_MAX_M 1.5f
#define PEDESTRIAN_HEIGHT_M 1.8f
#define PEDESTRIAN_LINGER_MIN_S 1.0f // crossers and kerb walkers at the end of their leg
#define PEDESTRIAN_LINGER_MAX_S 3.0f
#define PEDESTRIAN_FAR_APPEAR_MIN_M 15.0f // standers and kerb walkers appear this far ahead of the car
#define PEDESTRIAN_FAR_APPEAR_MAX_M 60.0f
#define PEDESTRIAN_SLOW_APPEAR_MIN_M 25.0f // slow crossers: far enough to yield without an emergency stop
#define PEDESTRIAN_SLOW_APPEAR_MAX_M 70.0f
#define PEDESTRIAN_SLOW_SPEED_MIN_MPS 0.8f // nuPlan pedestrians cross at 0.6-1.7 m/s
#define PEDESTRIAN_SLOW_SPEED_MAX_MPS 2.0f
#define PEDESTRIAN_SLOW_WAIT_MAX_S 5.0f           // standstill at the kerb before setting off
#define PEDESTRIAN_SLOW_SKEW_MAX_RAD 0.6f         // crossing direction off the perpendicular
#define PEDESTRIAN_SLOW_AGAINST_TRAFFIC_PROB 0.5f // which way a skewed crossing leans
#define PEDESTRIAN_KERB_WALK_SPEED_MIN_MPS 0.7f
#define PEDESTRIAN_KERB_WALK_SPEED_MAX_MPS 1.7f
#define PEDESTRIAN_KERB_WALK_MIN_M 8.0f
#define PEDESTRIAN_KERB_WALK_MAX_M 25.0f
#define PEDESTRIAN_KERB_WALK_BEND_TOLERANCE_M 0.5f // the straight walk may leave the kerb line by this much mid-way
#define PEDESTRIAN_KERB_WALK_AGAINST_TRAFFIC_PROB 0.5f
#define PEDESTRIAN_KERB_STEP_MIN_M 0.5f // CARLA's stuck Scenario3 walkers move 0.7-1.4 m and stand
#define PEDESTRIAN_KERB_STEP_MAX_M 1.5f
#define PEDESTRIAN_KERB_STEP_EDGE_CLEARANCE_M 0.3f // a step toward the road ends this far outside the road edge
#define PEDESTRIAN_KERB_STEP_AWAY_PROB 0.5f
#define PEDESTRIAN_BYSTANDER_STAND_MIN_S 8.0f // standers and kerb steppers before they vanish
#define PEDESTRIAN_BYSTANDER_STAND_MAX_S 25.0f
#define PEDESTRIAN_LEFT_KERB_PROB 0.5f // every mode but the two Scenario3 ones
#define PEDESTRIAN_COOLDOWN_MIN_S 5.0f
#define PEDESTRIAN_COOLDOWN_MAX_S 20.0f
#define PEDESTRIAN_TARGET_MIN_SPEED_MPS 1.0f // only moving cars get a crossing
#define PEDESTRIAN_ARM_ATTEMPTS 4            // target cars tried per step once the cooldown ran out

// Stop line geometry (spawn overlap check and IDM braking)
#define STOP_LINE_DIST_SQ (10.0f * 10.0f)
#define STOP_LINE_EXTENSION_FACTOR 1.5f
#define STOP_LINE_HEADING_THRESHOLD (M_PI / 4.0f)

// Red light violation detection
#define RED_LIGHT_TRIGGER_DIST_SQ (30.0f * 30.0f)
#define RED_LIGHT_LATERAL_EXTENSION_M 15.0f // Beyond each stop line endpoint, so the line cannot be driven around
#define RED_LIGHT_ENTER_HEADING_THRESHOLD (M_PI / 2.0f)
#define RED_LIGHT_TAIL_CLEARANCE_M 1.0f // CARLA's RunningRedLightTest scores the tail down to 1 m behind the rear edge

// Stop sign violation detection (CaRL RunStopSign2 semantics, trigger box centred on the stop line)
#define STOP_SIGN_PROXIMITY_DIST_SQ (20.0f * 20.0f)
#define STOP_SIGN_TRIGGER_HALF_DEPTH_M 1.7f // CARLA traffic.stop trigger volume half extent along the lane
#define STOP_SIGN_AFFECTED_BOX_SCALE 1.2f
#define STOP_SIGN_STOP_SPEED_MPS 0.1f
#define STOP_SIGN_REVERSING_SPEED_MPS 0.17f
#define STOP_SIGN_APPROACH_HEADING_THRESHOLD (M_PI / 4.0f)
#define STOP_SIGN_LATERAL_EXTENSION_M 17.5f // Five lane widths beyond each endpoint, cannot be driven around

#define BEHIND_COS_THRESHOLD -0.8660254f // cos(150 degrees)

// =====================================================================================
// 5. SPATIAL GRID
// =====================================================================================

#define GRID_CELL_SIZE 5.0f
#define MAX_GRID_CELL_COUNT 100000000
// Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution)
// => For each entity type in gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2
#define MAX_ENTITIES_PER_CELL 64
#define ROAD_QUERY_CELL_COUNT 25 // 5x5 cell neighborhood
#define ROAD_QUERY_HALF_CELLS 2  // cells on each side of the window's centre cell
#define ROAD_QUERY_ENTITY_COUNT (MAX_ENTITIES_PER_CELL * ROAD_QUERY_CELL_COUNT)
#define OBS_ENTITY_NONE 0 // grid entity not sampled for road observations
#define OBS_ENTITY_LANE 1
#define OBS_ENTITY_EDGE 2

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
#define GOAL_HEADING_MAX_ATTEMPTS 8
#define GT_GOAL_RADIUS_M 6.0f
// Max snap distance from an externally-set (co-sim) goal waypoint to its lane; route goals are
// already lane-centered upstream, so anything farther is off the drivable network -> no lane.
#define GOAL_LANE_SNAP_MAX_DIST_M 6.0f
#define LANE_GRAPH_DISTANCE_NORM_M 500.0f // normalization for the GPS lane-distance feature

// =====================================================================================
// 7. TRAFFIC LIGHTS
// =====================================================================================

#define TL_DEFAULT_RED_DURATION 2.0f
#define TL_DEFAULT_YELLOW_DURATION 3.0f
#define TL_DEFAULT_GREEN_DURATION 10.0f
#define TL_EPISODE_DISABLE_PROB 0.20f
// Junction lanes inherit the min limit over their entry lanes; passes bound chained connectors
#define SPEED_LIMIT_JUNCTION_INHERIT_PASSES 4
#define UNKNOWN_LANE_SPEED_LIMIT_MPS 15.0f // lanes without a map limit (nuPlan Boston/Singapore); CaRL's IDM fallback
#define TL_INDIVIDUAL_REMOVE_PROB 0.20f
#define TL_GROUP_REMOVE_PROB 0.20f
#define TL_ALWAYS_GREEN_PROB 0.05f

// =====================================================================================
// 8. OBSERVATIONS
// =====================================================================================

#define EGO_FEATURES 11
#define LANE_FEATURES 9
#define LANE_SPEED_LIMIT_FEATURES 1 // the lane's speed limit, after the LANE_FEATURES columns
#define LANE_ROW_FEATURES (LANE_FEATURES + LANE_SPEED_LIMIT_FEATURES)
#define BOUNDARY_FEATURES 6
#define PARTNER_FEATURES 9
#define PARTNER_RELATIVE_VELOCITY_FEATURES 2 // velocity relative to ego, after the PARTNER_FEATURES columns
#define PARTNER_ROW_FEATURES (PARTNER_FEATURES + PARTNER_RELATIVE_VELOCITY_FEATURES)
#define TRAFFIC_CONTROL_FEATURES 7
#define TRAFFIC_CONTROL_YELLOW_REMAINING_FEATURES 1 // before type/state when obs_traffic_light_yellow_remaining is set
#define GOAL_FEATURES 3
#define OBS_VALID_COUNT_FEATURES 4
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
#define NUM_REWARD_COEFS 18

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
#define STANDSTILL_SPEED_EPSILON_MPS 1e-4f
#define MAX_STOPPED_SECONDS 60.0f

#define METRIC_SCORE_WINDOW_SECONDS 10.0f
#define PUFFER_PROGRESS_REFERENCE_SPEED 10.0f

// =====================================================================================
// 11. TELEMETRY
// =====================================================================================

// obs_html_frame array field counts
#define AGENT_F32_GOAL_RADIUS_IDX 12
#define AGENT_F32_FIELDS                                                                                               \
    13 // sim_x/y/z, heading, length, width, speed, steering, accel_long, accel_lat, jerk_long, jerk_lat, goal_radius
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
#define REWARD_F32_FIELDS 13

#endif
