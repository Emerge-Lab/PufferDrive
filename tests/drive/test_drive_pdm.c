#include "include/drive_fixture.h"
#include "include/test.h"

#define WORLD_LANE_COUNT 3
#define WORLD_LANE_LENGTH_M 100.0f
#define WORLD_SPEED_LIMIT_MPS 10.0f
#define WORLD_MARGIN_M 50.0f
#define EGO 0
#define OTHER 1
#define PDM_TOWN_AGENT_COUNT 50

// Straight chained lanes along +x on an empty grid, so nothing is ever off the road
typedef struct {
    Drive env;
    RoadMapElement lanes[WORLD_LANE_COUNT];
    float lane_x[WORLD_LANE_COUNT][2];
    float lane_y[WORLD_LANE_COUNT][2];
    float lane_z[WORLD_LANE_COUNT][2];
    float lane_headings[WORLD_LANE_COUNT][1];
    int lane_exits[WORLD_LANE_COUNT][1];
    float lane_limits[WORLD_LANE_COUNT];
    GridMap grid;
    Agent agents[2];
    int active_indices[2];
    TrafficControlElement light;
    int light_state[1];
    int light_lanes[1];
} PdmWorld;

static void world_init(PdmWorld *world, int simulation_mode, int agent_count) {
    memset(world, 0, sizeof(*world));
    for (int i = 0; i < WORLD_LANE_COUNT; i++) {
        RoadMapElement *lane = &world->lanes[i];
        world->lane_x[i][0] = i * WORLD_LANE_LENGTH_M;
        world->lane_x[i][1] = (i + 1) * WORLD_LANE_LENGTH_M;
        world->lane_exits[i][0] = i + 1;
        world->lane_limits[i] = WORLD_SPEED_LIMIT_MPS;
        lane->type = LANE_SURFACE_STREET;
        lane->segment_size = 2;
        lane->x = world->lane_x[i];
        lane->y = world->lane_y[i];
        lane->z = world->lane_z[i];
        lane->headings = world->lane_headings[i];
        lane->length = WORLD_LANE_LENGTH_M;
        lane->num_exits = (i < WORLD_LANE_COUNT - 1) ? 1 : 0;
        lane->exit_lanes = world->lane_exits[i];
    }

    GridMap *grid = &world->grid;
    grid->top_left_x = -WORLD_MARGIN_M;
    grid->bottom_right_x = WORLD_LANE_COUNT * WORLD_LANE_LENGTH_M + WORLD_MARGIN_M;
    grid->bottom_right_y = -WORLD_MARGIN_M;
    grid->top_left_y = WORLD_MARGIN_M;
    grid->grid_cols = (int) ((grid->bottom_right_x - grid->top_left_x) / GRID_CELL_SIZE);
    grid->grid_rows = (int) ((grid->top_left_y - grid->bottom_right_y) / GRID_CELL_SIZE);
    grid->cell_entities_count = (int *) calloc(grid->grid_cols * grid->grid_rows, sizeof(int));
    grid->cells = (GridMapEntity **) calloc(grid->grid_cols * grid->grid_rows, sizeof(GridMapEntity *));

    Drive *env = &world->env;
    env->simulation_mode = simulation_mode;
    env->road_elements = world->lanes;
    env->num_road_elements = WORLD_LANE_COUNT;
    env->lane_speed_limit_mps = world->lane_limits;
    env->grid_map = grid;
    env->agents = world->agents;
    env->active_agent_indices = world->active_indices;
    env->active_agent_count = agent_count;
    env->num_agents = agent_count;
    env->dt = 0.1f;
    env->base_max_speed_mps = 20.0f;
    env->pdm_horizon_seconds = 4.0f;
    rng_seed(&env->rng_state, 7);
    for (int i = 0; i < agent_count; i++) {
        world->active_indices[i] = i;
    }
}

static void world_free(PdmWorld *world) {
    free(world->grid.cell_entities_count);
    free(world->grid.cells);
    free(world->agents[EGO].route);
    free(world->agents[OTHER].route);
}

static void world_place_agent(PdmWorld *world, int agent_idx, float x, float y, float speed_mps, int route_lane_count) {
    Agent *agent = &world->agents[agent_idx];
    *agent = drive_test_agent(x, y, 0.0f);
    agent->current_lane_idx = 0;
    agent->route = (int *) malloc(route_lane_count * sizeof(int));
    agent->route_length = route_lane_count;
    for (int i = 0; i < route_lane_count; i++) {
        agent->route[i] = i;
    }
    agent->sim_vx = speed_mps;
    update_agent_speed(agent);
}

// A light across lane 0 at light_x, in the state it keeps for the test
static void world_add_light(PdmWorld *world, float light_x, int state) {
    TrafficControlElement *light = &world->light;
    world->light_state[0] = state;
    world->light_lanes[0] = 0;
    light->type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    light->state_size = 1;
    light->states = world->light_state;
    light->stop_line[0] = light_x;
    light->stop_line[1] = -2.0f;
    light->stop_line[3] = light_x;
    light->stop_line[4] = 2.0f;
    light->num_controlled_lanes = 1;
    light->controlled_lanes = world->light_lanes;
    world->env.traffic_elements = light;
    world->env.num_traffic_elements = 1;
}

static int test_route_pose_follows_lanes_and_offset(void) {
    PdmWorld world;
    world_init(&world, SIMULATION_MODE_REPLAY, 1);
    world_place_agent(&world, EGO, 40.0f, 0.0f, 0.0f, 2);
    Agent *ego = &world.agents[EGO];
    IDMLaneProjection projection = idm_project_to_route_lanes(&world.env, ego);
    EXPECT_EQ_INT(projection.valid, 1);

    PDMRoutePose pose;
    EXPECT_EQ_INT(pdm_sample_route_pose(&world.env, ego, projection, 25.0f, 0.0f, &pose), 1);
    EXPECT_NEAR(pose.x, 65.0f, 1e-4f);
    EXPECT_NEAR(pose.y, 0.0f, 1e-4f);
    EXPECT_EQ_INT(pose.route_idx, 0);

    EXPECT_EQ_INT(pdm_sample_route_pose(&world.env, ego, projection, 75.0f, 1.0f, &pose), 1);
    EXPECT_NEAR(pose.x, 115.0f, 1e-4f);
    EXPECT_NEAR(pose.y, 1.0f, 1e-4f); // a positive offset is to the left of the lane heading
    EXPECT_EQ_INT(pose.lane_idx, 1);
    EXPECT_EQ_INT(pose.route_idx, 1);

    EXPECT_NEAR(pdm_remaining_route_distance(&world.env, ego, projection), 160.0f, 1e-3f);
    EXPECT_EQ_INT(pdm_sample_route_pose(&world.env, ego, projection, 161.0f, 0.0f, &pose), 0);
    world_free(&world);
    return 0;
}

static int test_speed_profile_ramps_then_holds(void) {
    float speed_mps, distance_meters;
    pdm_speed_profile(0.0f, 2.0f, 4.0f, 1.0f, 20.0f, &speed_mps, &distance_meters);
    EXPECT_NEAR(speed_mps, 2.0f, 1e-5f);
    EXPECT_NEAR(distance_meters, 1.0f, 1e-5f);
    pdm_speed_profile(0.0f, 2.0f, 4.0f, 3.0f, 20.0f, &speed_mps, &distance_meters);
    EXPECT_NEAR(speed_mps, 4.0f, 1e-5f);
    EXPECT_NEAR(distance_meters, 8.0f, 1e-5f);
    // braking to a stop never rolls backward
    pdm_speed_profile(4.0f, -2.0f, 0.0f, 5.0f, 20.0f, &speed_mps, &distance_meters);
    EXPECT_NEAR(speed_mps, 0.0f, 1e-5f);
    EXPECT_NEAR(distance_meters, 4.0f, 1e-5f);
    return 0;
}

static int test_free_road_settles_at_speed_limit_on_lane_center(void) {
    PdmWorld world;
    world_init(&world, SIMULATION_MODE_REPLAY, 1);
    world_place_agent(&world, EGO, 5.0f, 0.6f, 0.0f, WORLD_LANE_COUNT);
    Agent *ego = &world.agents[EGO];
    for (int step = 0; step < 150; step++) {
        move_pdm(&world.env, EGO);
        EXPECT_TRUE(ego->sim_speed <= WORLD_SPEED_LIMIT_MPS + 1e-3f);
    }
    EXPECT_TRUE(ego->sim_speed > 0.9f * WORLD_SPEED_LIMIT_MPS);
    EXPECT_TRUE(ego->sim_x > 100.0f);
    EXPECT_NEAR(ego->sim_y, 0.0f, 0.1f);
    EXPECT_NEAR(ego->sim_heading, 0.0f, 0.02f);
    EXPECT_EQ_INT(ego->current_route_idx, 1);
    world_free(&world);
    return 0;
}

static int test_stops_behind_standing_vehicle(void) {
    PdmWorld world;
    world_init(&world, SIMULATION_MODE_REPLAY, 2);
    world_place_agent(&world, EGO, 5.0f, 0.0f, 8.0f, WORLD_LANE_COUNT);
    world_place_agent(&world, OTHER, 35.0f, 0.0f, 0.0f, WORLD_LANE_COUNT);
    world.agents[OTHER].stopped = 1;
    Agent *ego = &world.agents[EGO];
    Agent *other = &world.agents[OTHER];

    IDMLaneProjection projection = idm_project_to_route_lanes(&world.env, ego);
    IDMLeader leader = pdm_find_leader(&world.env, EGO, projection, 0.0f);
    EXPECT_EQ_INT(leader.has_leader, 1);
    EXPECT_EQ_INT(leader.leader_agent_idx, OTHER);
    EXPECT_FALSE(leader.is_traffic_light);

    for (int step = 0; step < 200; step++) {
        move_pdm(&world.env, EGO);
        EXPECT_FALSE(check_obb_collision(ego, other));
    }
    float bumper_gap_m = other->sim_x - ego->sim_x - 0.5f * (ego->sim_length + other->sim_length);
    EXPECT_NEAR(ego->sim_speed, 0.0f, 0.05f);
    EXPECT_TRUE(bumper_gap_m > 0.5f && bumper_gap_m < 6.0f);
    world_free(&world);
    return 0;
}

// Lateral offsets are the way out when braking alone no longer avoids a box that reaches into the lane
static int test_swerves_past_obstacle_inside_braking_distance(void) {
    PdmWorld world;
    world_init(&world, SIMULATION_MODE_REPLAY, 2);
    world_place_agent(&world, EGO, 5.0f, 0.0f, 10.0f, WORLD_LANE_COUNT);
    world_place_agent(&world, OTHER, 17.0f, -1.7f, 0.0f, WORLD_LANE_COUNT);
    world.agents[OTHER].stopped = 1;
    Agent *ego = &world.agents[EGO];
    Agent *other = &world.agents[OTHER];

    float lateral_at_obstacle_m = 0.0f;
    for (int step = 0; step < 100; step++) {
        move_pdm(&world.env, EGO);
        EXPECT_FALSE(check_obb_collision(ego, other));
        if (fabsf(ego->sim_x - other->sim_x) < 0.6f) {
            lateral_at_obstacle_m = ego->sim_y;
        }
    }
    EXPECT_TRUE(lateral_at_obstacle_m > 0.3f);
    EXPECT_TRUE(ego->sim_x > other->sim_x + 20.0f);
    EXPECT_NEAR(ego->sim_y, 0.0f, 0.1f);
    world_free(&world);
    return 0;
}

static int test_waits_at_red_light_and_leaves_on_green(void) {
    PdmWorld world;
    world_init(&world, SIMULATION_MODE_REPLAY, 1);
    world_place_agent(&world, EGO, 5.0f, 0.0f, 8.0f, WORLD_LANE_COUNT);
    world_add_light(&world, 40.0f, TRAFFIC_CONTROL_STATE_RED);
    Agent *ego = &world.agents[EGO];

    IDMLaneProjection projection = idm_project_to_route_lanes(&world.env, ego);
    IDMLeader leader = pdm_find_leader(&world.env, EGO, projection, 0.0f);
    EXPECT_EQ_INT(leader.has_leader, 1);
    EXPECT_TRUE(leader.is_traffic_light);

    for (int step = 0; step < 200; step++) {
        move_pdm(&world.env, EGO);
    }
    float front_edge_x = ego->sim_x + 0.5f * ego->sim_length;
    EXPECT_NEAR(ego->sim_speed, 0.0f, 0.05f);
    EXPECT_TRUE(front_edge_x < 40.0f && front_edge_x > 30.0f);

    world.light_state[0] = TRAFFIC_CONTROL_STATE_GREEN;
    for (int step = 0; step < 100; step++) {
        move_pdm(&world.env, EGO);
    }
    EXPECT_TRUE(ego->sim_x > 60.0f);
    world_free(&world);
    return 0;
}

static int test_gigaflow_route_is_extended_then_ends_at_dead_end(void) {
    PdmWorld world;
    world_init(&world, SIMULATION_MODE_GIGAFLOW, 1);
    world_place_agent(&world, EGO, 60.0f, 0.0f, 8.0f, 1);
    Agent *ego = &world.agents[EGO];

    // 40 m of route left, less than the horizon needs: the walk continues through the lane exits
    move_pdm(&world.env, EGO);
    EXPECT_EQ_INT(ego->route_length, WORLD_LANE_COUNT);
    EXPECT_EQ_INT(ego->route[0], 0);
    EXPECT_EQ_INT(ego->route[WORLD_LANE_COUNT - 1], WORLD_LANE_COUNT - 1);
    EXPECT_EQ_INT(ego->current_route_idx, 0);
    EXPECT_TRUE(ego->sim_x > 60.0f);

    // lane 2 has no exit: the agent comes to rest before the end of the road
    for (int step = 0; step < 600; step++) {
        move_pdm(&world.env, EGO);
        EXPECT_FINITE(ego->sim_x);
    }
    EXPECT_EQ_INT(ego->route_length, WORLD_LANE_COUNT);
    EXPECT_TRUE(ego->sim_x > 2.0f * WORLD_LANE_LENGTH_M);
    EXPECT_TRUE(ego->sim_x + 0.5f * ego->sim_length < WORLD_LANE_COUNT * WORLD_LANE_LENGTH_M);
    EXPECT_NEAR(ego->sim_speed, 0.0f, 0.3f);
    world_free(&world);
    return 0;
}

static int test_agent_without_route_brakes_to_a_stop(void) {
    PdmWorld world;
    world_init(&world, SIMULATION_MODE_GIGAFLOW, 1);
    world_place_agent(&world, EGO, 5.0f, 0.0f, 8.0f, 1);
    Agent *ego = &world.agents[EGO];
    free(ego->route);
    ego->route = NULL;
    ego->route_length = 0;
    for (int step = 0; step < 30; step++) {
        move_pdm(&world.env, EGO);
    }
    EXPECT_NEAR(ego->sim_speed, 0.0f, 1e-4f);
    EXPECT_TRUE(ego->sim_x > 5.0f && ego->sim_x < 15.0f);
    EXPECT_NEAR(ego->sim_y, 0.0f, 1e-4f);
    world_free(&world);
    return 0;
}

// 15 Hz is the self-play benchmark's step: rollouts on a coarser grid cut the inner kerb of tight turns
static int test_pdm_agents_drive_a_carla_town_at_15hz_without_infractions(void) {
    const char *town03 = "./opengigaflow/resources/carla/opendrive__Town03.bin";
    Drive env = drive_test_env_config(town03, SIMULATION_MODE_GIGAFLOW, PDM_TOWN_AGENT_COUNT, 0);
    env.sdc_controller = CONTROLLER_PDM;
    env.non_sdc_controller = CONTROLLER_PDM;
    env.dt = 0.0667f;
    env.traffic_lights_enabled = 0;
    env.scenario_length = 400;
    env.init_seed = 11;
    allocate(&env);
    c_reset(&env);

    float start_x[PDM_TOWN_AGENT_COUNT], start_y[PDM_TOWN_AGENT_COUNT];
    for (int i = 0; i < env.active_agent_count; i++) {
        start_x[i] = env.agents[env.active_agent_indices[i]].sim_x;
        start_y[i] = env.agents[env.active_agent_indices[i]].sim_y;
    }
    int infraction_count = 0;
    for (int step = 0; step < 300; step++) {
        c_step(&env);
        for (int i = 0; i < env.active_agent_count; i++) {
            Agent *agent = &env.agents[env.active_agent_indices[i]];
            EXPECT_EQ_INT(agent->controller, CONTROLLER_PDM);
            infraction_count += agent->metrics_array[COLLISION_IDX] > 0.0f || agent->metrics_array[OFFROAD_IDX] > 0.0f;
        }
    }
    int moved_agent_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        moved_agent_count += compute_euclidean_distance(start_x[i], start_y[i], agent->sim_x, agent->sim_y) > 20.0f;
    }
    EXPECT_EQ_INT(infraction_count, 0);
    EXPECT_TRUE(moved_agent_count >= 40);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_route_pose_follows_lanes_and_offset);
    RUN_TEST(test_speed_profile_ramps_then_holds);
    RUN_TEST(test_free_road_settles_at_speed_limit_on_lane_center);
    RUN_TEST(test_stops_behind_standing_vehicle);
    RUN_TEST(test_swerves_past_obstacle_inside_braking_distance);
    RUN_TEST(test_waits_at_red_light_and_leaves_on_green);
    RUN_TEST(test_gigaflow_route_is_extended_then_ends_at_dead_end);
    RUN_TEST(test_agent_without_route_brakes_to_a_stop);
    RUN_TEST(test_pdm_agents_drive_a_carla_town_at_15hz_without_infractions);
    return test_summary(failures);
}
