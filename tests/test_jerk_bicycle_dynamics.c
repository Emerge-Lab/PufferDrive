#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../pufferlib/ocean/drive/drive.h"

#define TEST_EPSILON 0.001f
#define DT 0.1f

// Bicycle Jerk Constants from paper
#define COEFF_THROTTLE_STEER_MIN 0.8f
#define COEFF_THROTTLE_STEER_MAX 1.25f
#define COEFF_ACC_VEL_MIN 0.666f
#define COEFF_ACC_VEL_MAX 1.5f

#define VEHICLE_LENGTH 4.5f
#define VEHICLE_WIDTH 2.0f

#define ACCEL_MAX_POSITIVE 2.5f
#define ACCEL_MAX_NEGATIVE -5.0f
#define VELOCITY_MAX 20
#define STEERING_ANGLE_MAX 0.55f
#define STEERING_RATE_MAX 0.6f

#define JERK_LONG_MAX_NEG -15.0f
#define JERK_LONG_MEDIUM_NEG -4.0f
#define JERK_ZERO 0
#define JERK_LONG_MEDIUM_POS 4.0f
#define JERK_LAT_NEG -4.0f
#define JERK_LAT_POS 4.0f

#define EXPECTED_ACCEL_INTEGRATION 0.4f
#define SIGN_CHANGE_THRESHOLD 0.1f

#define NUM_AGENTS 1
#define NUM_ENTITIES 1
#define ACTION_ARRAY_SIZE 2

// Action indices for readability
#define ACTION_LONG_MAX_NEG 0
#define ACTION_LONG_MED_NEG 1
#define ACTION_ZERO 2
#define ACTION_LONG_MED_POS 3
#define ACTION_LAT_NEG 0
#define ACTION_LAT_ZERO 1
#define ACTION_LAT_POS 2

// Helper functions
int float_equals(float a, float b) {
    return fabs(a - b) < TEST_EPSILON;
}


Drive* create_test_env() {
    Drive* env = (Drive*)calloc(1, sizeof(Drive));
    env->dynamics_model = JERK_BICYCLE;
    env->dt = DT;
    env->num_agents = NUM_AGENTS;
    env->active_agent_count = NUM_AGENTS;
    env->timestep = 0;
    
    // Create a test entity
    env->num_entities = NUM_ENTITIES;
    env->entities = (Entity*)calloc(NUM_ENTITIES, sizeof(Entity));
    env->entities[0].type = VEHICLE;
    env->entities[0].length = VEHICLE_LENGTH;
    env->entities[0].width = VEHICLE_WIDTH;
    env->entities[0].active_agent = 1;
    
    env->active_agent_indices = (int*)malloc(sizeof(int));
    env->active_agent_indices[0] = 0;
    
    // Allocate action array for Bicycle Jerk ([4,3])
    env->actions = (int*)calloc(ACTION_ARRAY_SIZE, sizeof(int));
    
    return env;
}

void cleanup_test_env(Drive* env) {
    free(env->entities);
    free(env->active_agent_indices);
    free(env->actions);
    free(env);
}

// Helper to set basic moving state
void set_basic_state(Entity* agent, float vx, float vy) {
    agent->x = 0;
    agent->y = 0;
    agent->heading = 0;
    agent->heading_x = 1;
    agent->heading_y = 0;
    agent->vx = vx;
    agent->vy = vy;
    agent->prev_v = sqrtf(vx * vx + vy * vy);
    agent->a_long = 0;
    agent->a_lat = 0;
    agent->prev_a_long = 0;
    agent->prev_a_lat = 0;
    agent->steering_angle = 0;
    agent->prev_steering_angle = 0;
}

// Helper to set actions
void set_actions(Drive* env, int long_action, int lat_action) {
    env->actions[0] = long_action;
    env->actions[1] = lat_action;
}

// Test 3: Verify jerk values match paper specification
void test_jerk_values() {
    printf("\nTest 3: Verifying jerk values...\n");
    
    assert(float_equals(JERK_BICYCLE_LONG[0], JERK_LONG_MAX_NEG));
    assert(float_equals(JERK_BICYCLE_LONG[1], JERK_LONG_MEDIUM_NEG));
    assert(float_equals(JERK_BICYCLE_LONG[2], JERK_ZERO));
    assert(float_equals(JERK_BICYCLE_LONG[3], JERK_LONG_MEDIUM_POS));
    
    assert(float_equals(JERK_BICYCLE_LAT[0], JERK_LAT_NEG));
    assert(float_equals(JERK_BICYCLE_LAT[1], JERK_ZERO));
    assert(float_equals(JERK_BICYCLE_LAT[2], JERK_LAT_POS));
    
    printf("Jerk values verification tests passed\n");
}

// Test 4: Test acceleration integration using actual move_jerk_bicycle_dynamics
void test_acceleration_integration() {
    printf("\nTest 4: Testing acceleration integration from jerk...\n");
    
    Drive* env = create_test_env();
    Entity* agent = &env->entities[0];

    set_basic_state(agent, 5.0f, 0);
    
    // Test case 1: Apply positive longitudinal jerk
     // 4.0 m/s longitudinal jerk, 0 laterals
    set_actions(env, ACTION_LONG_MED_POS, ACTION_LAT_ZERO);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    // Expected: a_long = 0 + 1.0 * 4.0 * 0.1 = 0.4
    assert(float_equals(agent->a_long, EXPECTED_ACCEL_INTEGRATION));
    assert(float_equals(agent->a_lat, JERK_ZERO));
    
    // Test case 2: Apply negative longitudinal jerk
    agent->prev_a_long = agent->a_long;
    // -15.0 m/s longitudinal jerk, 0 lateral
    set_actions(env, ACTION_LONG_MAX_NEG, ACTION_LAT_ZERO);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    // Expected: a_long = 0.4 + 1.0 * (-15.0) * 0.1 = -1.1 but there is sign switch so should be 0
    assert(float_equals(agent->a_long, 0));
    
    cleanup_test_env(env);
    printf("Acceleration integration tests passed\n");
}

// Test 5: Test acceleration clipping using actual dynamics
void test_acceleration_clipping() {
    printf("\nTest 5: Testing acceleration clipping...\n");
    
    Drive* env = create_test_env();
    Entity* agent = &env->entities[0];
    
    set_basic_state(agent, 10, 0);
    
    // Test max positive acceleration clipping
    agent->a_long = 2.0f;
    agent->prev_a_long = 2.0f;
    
    set_actions(env, ACTION_LONG_MED_POS, ACTION_LAT_ZERO);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    // Should be clamped to 2.5 * c_acc = 2.5
    assert(agent->a_long <= ACCEL_MAX_POSITIVE);
    
    // Test max negative acceleration clipping
    agent->a_long = -4.0f;
    agent->prev_a_long = -4.0f;
    
     // Max negative jerk (-15.0)
    set_actions(env, ACTION_LONG_MAX_NEG, ACTION_LAT_ZERO);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    // Should be clamped to -5.0
    assert(agent->a_long >= ACCEL_MAX_NEGATIVE);
    
    // Test with randomization coefficient
    agent->a_long = 2.0f;
    agent->prev_a_long = 2.0f;
    
    set_actions(env, ACTION_LONG_MED_POS, ACTION_LAT_ZERO);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    // Should be clamped to 2.5 * 1.5 = 3.75
    assert(agent->a_long <= ACCEL_MAX_POSITIVE * COEFF_ACC_VEL_MAX);
    
    cleanup_test_env(env);
    printf("Acceleration clipping tests passed\n");
}

// Test 6: Test velocity integration through dynamics
void test_velocity_integration() {
    printf("\nTest 6: Testing velocity integration...\n");
    
    Drive* env = create_test_env();
    Entity* agent = &env->entities[0];
    
    set_basic_state(agent, 10, 0);
    agent->a_long = 1;
    agent->prev_a_long = 1;
    
    float initial_speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    
    // Apply acceleration to test velocity update
    set_actions(env, ACTION_ZERO, ACTION_LAT_ZERO);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    float final_speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    
    // Velocity should have increased
    assert(final_speed > initial_speed);
    
    // Test sign change behavior
    set_basic_state(agent, 0.5f, 0);
    agent->a_long = -10;
    agent->prev_a_long = -10;
    
    set_actions(env, ACTION_LONG_MED_NEG, ACTION_LAT_ZERO);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    
    // Speed should be set to near 0 on sign change
    assert(speed < SIGN_CHANGE_THRESHOLD || float_equals(speed, JERK_ZERO));
    
    cleanup_test_env(env);
    printf("Velocity integration tests passed\n");
}

// Test 7: Test velocity clipping through dynamics
void test_velocity_clipping() {
    printf("\nTest 7: Testing velocity clipping...\n");
    
    Drive* env = create_test_env();
    Entity* agent = &env->entities[0];
    
    set_basic_state(agent, 30, 0);
    
    set_actions(env, ACTION_ZERO, ACTION_LAT_ZERO);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    assert(speed <= VELOCITY_MAX);
    
    // Test with randomization
    set_basic_state(agent, 25.0f, 0);
    
    move_jerk_bicycle_dynamics(env, 0, 0);
    speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    assert(speed <= VELOCITY_MAX * COEFF_ACC_VEL_MAX);
    
    cleanup_test_env(env);
    printf("Velocity clipping tests passed\n");
}

// Test 2: Test steering angle calculation through dynamics
void test_steering_angle_calculation() {
    printf("\nTest 2: Testing steering angle calculation...\n");
    
    Drive* env = create_test_env();
    Entity* agent = &env->entities[0];
    
    set_basic_state(agent, 10, 0);
    
    // Apply lateral jerk
    set_actions(env, ACTION_ZERO, ACTION_LAT_POS);
    move_jerk_bicycle_dynamics(env, 0, 0);
    
    // Steering angle should be positive and within limits
    assert(agent->steering_angle > JERK_ZERO);
    assert(agent->steering_angle <= STEERING_ANGLE_MAX);
    
    cleanup_test_env(env);
    printf("Steering angle calculation tests passed\n");
}

// Test 1: Test steering rate limiting through dynamics
void test_steering_rate_limiting() {
    printf("\nTest 1: Testing steering angle rate limiting...\n");

    Drive* env = create_test_env();
    Entity* agent = &env->entities[0];

    set_basic_state(agent, 10, 0);

    // Set initial steering state
    agent->steering_angle = 0.2f;
    agent->prev_steering_angle = 0.2f;
    agent->a_lat = 2.0f;
    agent->prev_a_lat = 2.0f;

    float initial_steering_angle = agent->steering_angle;

    // Apply strong lateral jerk
    set_actions(env, ACTION_ZERO, ACTION_LAT_POS);
    move_jerk_bicycle_dynamics(env, 0, 0);

    float delta_steering_angle = agent->steering_angle - initial_steering_angle;
    float max_delta = STEERING_RATE_MAX * DT;

    // Change should be limited by rate
    assert(fabs(delta_steering_angle) <= max_delta + TEST_EPSILON);

    cleanup_test_env(env);
    printf("Steering rate limiting tests passed\n");
}


// Main test runner
int main() {
    printf("=== Bicycle Jerk Dynamics Test Suite ===\n\n");

    test_steering_rate_limiting();
    test_steering_angle_calculation();
    test_jerk_values();
    test_acceleration_integration();
    test_acceleration_clipping();
    test_velocity_integration();
    test_velocity_clipping();

    printf("\n=== All tests passed! ===\n");
    
    return 0;
}