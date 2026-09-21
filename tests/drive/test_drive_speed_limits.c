#include "include/drive_fixture.h"
#include "include/test.h"

// Synthetic map: two zone-0 lanes, one zone-1 lane, a chained pair of junction lanes, a road edge,
// a zone-2 lane near the clip floor and a dangling junction lane without entries.
#define ELEMENT_COUNT 8
#define ZONE_COUNT 3

static int entries_of_3[2] = {0, 2};
static int entries_of_4[1] = {3};

static void make_element(RoadMapElement *road, int type, float speed_limit, int zone_idx, int *entries, int entry_count) {
    *road = (RoadMapElement) {0};
    road->type = type;
    road->speed_limit = speed_limit;
    road->speed_zone_idx = zone_idx;
    road->entry_lanes = entries;
    road->num_entries = entry_count;
}

typedef struct {
    RoadMapElement roads[ELEMENT_COUNT];
    float limits[ELEMENT_COUNT];
    float offsets[ZONE_COUNT];
    unsigned char resolved[ELEMENT_COUNT];
} SpeedLimitFixture;

static Drive make_env(SpeedLimitFixture *fixture, float prob, float delta_mps, uint64_t seed) {
    make_element(&fixture->roads[0], LANE_SURFACE_STREET, 10.0f, 0, NULL, 0);
    make_element(&fixture->roads[1], LANE_SURFACE_STREET, 10.0f, 0, NULL, 0);
    make_element(&fixture->roads[2], LANE_SURFACE_STREET, 25.0f, 1, NULL, 0);
    make_element(&fixture->roads[3], LANE_SURFACE_STREET, 10.0f, -1, entries_of_3, 2);
    make_element(&fixture->roads[4], LANE_SURFACE_STREET, 10.0f, -1, entries_of_4, 1);
    make_element(&fixture->roads[5], ROAD_EDGE_BOUNDARY, 0.0f, -1, NULL, 0);
    make_element(&fixture->roads[6], LANE_SURFACE_STREET, 2.5f, 2, NULL, 0);
    make_element(&fixture->roads[7], LANE_SURFACE_STREET, 10.0f, -1, NULL, 0);
    Drive env = {0};
    env.road_elements = fixture->roads;
    env.num_road_elements = ELEMENT_COUNT;
    env.num_speed_zones = ZONE_COUNT;
    env.lane_speed_limit_mps = fixture->limits;
    env.speed_zone_offset_mps = fixture->offsets;
    env.lane_limit_resolved = fixture->resolved;
    env.speed_limit_random_prob = prob;
    env.speed_limit_random_delta_mps = delta_mps;
    env.speed_limit_random_min_mps = 2.0f;
    env.speed_limit_random_max_mps = 30.0f;
    rng_seed(&env.rng_state, seed);
    return env;
}

static int test_disabled_keeps_original_limits_and_rng(void) {
    SpeedLimitFixture fixture;
    Drive env = make_env(&fixture, 0.0f, 5.0f, 7);
    Rng rng_before = env.rng_state;
    sample_zone_speed_limits(&env);
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        EXPECT_NEAR(env.lane_speed_limit_mps[i], fixture.roads[i].speed_limit, 1e-6f);
    }
    EXPECT_TRUE(memcmp(&rng_before, &env.rng_state, sizeof(Rng)) == 0);
    return 0;
}

static int test_zone_offsets_apply_and_junctions_inherit(void) {
    SpeedLimitFixture fixture;
    Drive env = make_env(&fixture, 1.0f, 5.0f, 7);
    sample_zone_speed_limits(&env);
    float *limits = env.lane_speed_limit_mps;
    EXPECT_NEAR(limits[0], limits[1], 1e-6f);
    EXPECT_TRUE(fabsf(limits[0] - 10.0f) <= 5.0f + 1e-5f);
    EXPECT_TRUE(fabsf(limits[2] - 25.0f) <= 5.0f + 1e-5f);
    EXPECT_NEAR(limits[3], fminf(limits[0], limits[2]), 1e-6f);
    EXPECT_NEAR(limits[4], limits[3], 1e-6f);
    EXPECT_NEAR(limits[5], 0.0f, 1e-6f);
    EXPECT_TRUE(limits[6] >= 2.0f);
    EXPECT_NEAR(limits[7], 10.0f, 1e-6f);
    return 0;
}

static int test_clip_bounds_hold_for_large_delta(void) {
    SpeedLimitFixture fixture;
    Drive env = make_env(&fixture, 1.0f, 100.0f, 3);
    sample_zone_speed_limits(&env);
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        if (fixture.roads[i].speed_zone_idx < 0) {
            continue;
        }
        EXPECT_TRUE(env.lane_speed_limit_mps[i] >= 2.0f);
        EXPECT_TRUE(env.lane_speed_limit_mps[i] <= 30.0f);
    }
    return 0;
}

static int test_same_seed_reproduces_and_other_seed_differs(void) {
    SpeedLimitFixture fixture_a, fixture_b, fixture_c;
    Drive env_a = make_env(&fixture_a, 1.0f, 5.0f, 11);
    Drive env_b = make_env(&fixture_b, 1.0f, 5.0f, 11);
    Drive env_c = make_env(&fixture_c, 1.0f, 5.0f, 12);
    sample_zone_speed_limits(&env_a);
    sample_zone_speed_limits(&env_b);
    sample_zone_speed_limits(&env_c);
    int differs = 0;
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        EXPECT_NEAR(env_a.lane_speed_limit_mps[i], env_b.lane_speed_limit_mps[i], 1e-6f);
        differs |= fabsf(env_a.lane_speed_limit_mps[i] - env_c.lane_speed_limit_mps[i]) > 1e-4f;
    }
    EXPECT_TRUE(differs);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_disabled_keeps_original_limits_and_rng);
    RUN_TEST(test_zone_offsets_apply_and_junctions_inherit);
    RUN_TEST(test_clip_bounds_hold_for_large_delta);
    RUN_TEST(test_same_seed_reproduces_and_other_seed_differs);
    return test_summary(failures);
}
