// Maths of the demo-only renderer. Links raylib for its types but never opens a window: every
// raylib draw call render.h makes is redirected to a capture stub below, so draw_cockpit_wheel can
// be called and its geometry inspected on a headless machine.
#include "pufferlib/ocean/drive/drive.h"
#include "raylib.h"
#include "rlgl.h"

#include <stdarg.h>

#define CAPTURE_RECT_MAX 16
#define CAPTURE_TEXT_MAX 8
#define CAPTURE_TEXT_LEN 256
#define CAPTURE_ROTATE_MAX 16

static int captured_rotate_count = 0;
static float captured_rotate_deg[CAPTURE_ROTATE_MAX];
static Vector3 captured_rotate_axis[CAPTURE_ROTATE_MAX];
static int captured_rect_count = 0;
static float captured_rect_rotation_deg[CAPTURE_RECT_MAX];
static int captured_ring_count = 0;
static Vector2 captured_ring_center;
static float captured_ring_inner_radius_px = 0.0f;
static float captured_ring_outer_radius_px = 0.0f;
static int captured_circle_count = 0;
static Vector2 captured_circle_center[CAPTURE_RECT_MAX];
static float captured_circle_radius_px[CAPTURE_RECT_MAX];
static int captured_text_count = 0;
static char captured_text[CAPTURE_TEXT_MAX][CAPTURE_TEXT_LEN];
static char format_buffer[CAPTURE_TEXT_LEN];

static void capture_rectangle(int pos_x, int pos_y, int width, int height, Color color) {
    (void) pos_x;
    (void) pos_y;
    (void) width;
    (void) height;
    (void) color;
}

static void capture_rectangle_pro(Rectangle rect, Vector2 origin, float rotation_deg, Color color) {
    (void) rect;
    (void) origin;
    (void) color;
    if (captured_rect_count < CAPTURE_RECT_MAX) {
        captured_rect_rotation_deg[captured_rect_count] = rotation_deg;
    }
    captured_rect_count++;
}

static void capture_ring(
    Vector2 center,
    float inner_radius_px,
    float outer_radius_px,
    float start_angle_deg,
    float end_angle_deg,
    int segments,
    Color color) {
    (void) start_angle_deg;
    (void) end_angle_deg;
    (void) segments;
    (void) color;
    captured_ring_center = center;
    captured_ring_inner_radius_px = inner_radius_px;
    captured_ring_outer_radius_px = outer_radius_px;
    captured_ring_count++;
}

// The dashboard cowl. Captured only so render.h's draw calls never reach real raylib headless.
static void capture_ellipse(int center_x, int center_y, float radius_h_px, float radius_v_px, Color color) {
    (void) center_x;
    (void) center_y;
    (void) radius_h_px;
    (void) radius_v_px;
    (void) color;
}

static void capture_circle(Vector2 center, float radius_px, Color color) {
    (void) color;
    if (captured_circle_count < CAPTURE_RECT_MAX) {
        captured_circle_center[captured_circle_count] = center;
        captured_circle_radius_px[captured_circle_count] = radius_px;
    }
    captured_circle_count++;
}

static void capture_text(const char *text, int pos_x, int pos_y, int font_size, Color color) {
    (void) pos_x;
    (void) pos_y;
    (void) font_size;
    (void) color;
    if (captured_text_count < CAPTURE_TEXT_MAX) {
        snprintf(captured_text[captured_text_count], CAPTURE_TEXT_LEN, "%s", text);
    }
    captured_text_count++;
}

static const char *capture_text_format(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(format_buffer, sizeof(format_buffer), format, args);
    va_end(args);
    return format_buffer;
}

static void capture_rotatef(float angle_deg, float axis_x, float axis_y, float axis_z) {
    if (captured_rotate_count < CAPTURE_ROTATE_MAX) {
        captured_rotate_deg[captured_rotate_count] = angle_deg;
        captured_rotate_axis[captured_rotate_count] = (Vector3) {axis_x, axis_y, axis_z};
    }
    captured_rotate_count++;
}

static void capture_push_matrix(void) {
}

static void capture_pop_matrix(void) {
}

static void capture_translatef(float x, float y, float z) {
    (void) x;
    (void) y;
    (void) z;
}

static void capture_cube(Vector3 position, float width, float height, float length, Color color) {
    (void) position;
    (void) width;
    (void) height;
    (void) length;
    (void) color;
}

static void capture_model_ex(
    Model model,
    Vector3 position,
    Vector3 rotation_axis,
    float rotation_deg,
    Vector3 scale,
    Color tint) {
    (void) model;
    (void) position;
    (void) rotation_axis;
    (void) rotation_deg;
    (void) scale;
    (void) tint;
}

static void capture_cylinder_ex(
    Vector3 start_position,
    Vector3 end_position,
    float start_radius_m,
    float end_radius_m,
    int sides,
    Color color) {
    (void) start_position;
    (void) end_position;
    (void) start_radius_m;
    (void) end_radius_m;
    (void) sides;
    (void) color;
}

static void capture_fps(int pos_x, int pos_y) {
    (void) pos_x;
    (void) pos_y;
}

static void reset_capture(void) {
    captured_rect_count = 0;
    captured_ring_count = 0;
    captured_circle_count = 0;
    captured_text_count = 0;
    captured_rotate_count = 0;
}

#define DrawRectangle capture_rectangle
#define DrawRectanglePro capture_rectangle_pro
#define DrawRing capture_ring
#define DrawCircleV capture_circle
#define DrawEllipse capture_ellipse
#define DrawText capture_text
#define TextFormat capture_text_format
#define DrawCube capture_cube
#define DrawModelEx capture_model_ex
#define DrawCylinderEx capture_cylinder_ex
#define DrawFPS capture_fps
#define rlPushMatrix capture_push_matrix
#define rlPopMatrix capture_pop_matrix
#define rlTranslatef capture_translatef
#define rlRotatef capture_rotatef

#include "include/drive_fixture.h"
#include "include/test.h"
#include "pufferlib/ocean/drive/render/render.h"

#define RENDER_TEST_SCREEN_WIDTH 1280
#define RENDER_TEST_SCREEN_HEIGHT 720
#define POSITION_TOLERANCE_M 1e-4f
#define ANGLE_TOLERANCE_DEG 1e-3f
#define UNIT_TOLERANCE 1e-5f
#define ARC_TOLERANCE_M 1e-4f
#define HEADING_WRAP_DELTA_RAD 1e-4f
#define FULL_LOCK_HAND_WHEEL_DEG 450.0f
// sim convention: positive steering_angle turns the car LEFT (positive curvature).
#define FULL_RIGHT_LOCK_STEERING_RAD (-HUMAN_STEERING_LIMIT_RAD)
#define FULL_LEFT_LOCK_STEERING_RAD HUMAN_STEERING_LIMIT_RAD
#define SPOKE_STEP_DEG (360.0f / (float) COCKPIT_WHEEL_SPOKE_COUNT)
#define WHEEL_LEVER_PX (COCKPIT_WHEEL_RADIUS_PX - COCKPIT_WHEEL_RIM_PX * 0.5f)

// Headings include both sides of the +-pi wrap so a camera that secretly normalised the heading
// instead of using cos/sin would show up as a discontinuity.
static const float TEST_HEADINGS_RAD[]
    = {0.0f,
       0.7f,
       1.5707963f,
       2.5f,
       3.1415926f - HEADING_WRAP_DELTA_RAD,
       -3.1415926f + HEADING_WRAP_DELTA_RAD,
       -2.5f,
       -1.5707963f,
       -0.7f};
#define TEST_HEADING_COUNT ((int) (sizeof(TEST_HEADINGS_RAD) / sizeof(TEST_HEADINGS_RAD[0])))

static Agent render_test_ego(float heading_rad) {
    Agent agent = drive_test_agent(12.5f, -37.25f, heading_rad);
    agent.sim_z = 3.75f;
    agent.id = 4;
    agent.active_agent = 1;
    return agent;
}

static Vector3 vector3_sub(Vector3 a, Vector3 b) {
    return (Vector3) {a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vector3 vector3_cross(Vector3 a, Vector3 b) {
    return (Vector3) {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static float vector3_dot(Vector3 a, Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float vector3_length(Vector3 a) {
    return sqrtf(vector3_dot(a, a));
}

static int expect_vector3_near(Vector3 actual, Vector3 expected, float tolerance, const char *label) {
    if (fabsf(actual.x - expected.x) > tolerance || fabsf(actual.y - expected.y) > tolerance
        || fabsf(actual.z - expected.z) > tolerance) {
        printf(
            "FAIL %s:%d: %s == (%f, %f, %f), expected (%f, %f, %f)\n",
            __FILE__,
            __LINE__,
            label,
            actual.x,
            actual.y,
            actual.z,
            expected.x,
            expected.y,
            expected.z);
        return 1;
    }
    return 0;
}

static int test_camera_chase(void) {
    g_renderer.camera_mode = CAMERA_MODE_CHASE;
    for (int heading_idx = 0; heading_idx < TEST_HEADING_COUNT; heading_idx++) {
        Agent ego = render_test_ego(TEST_HEADINGS_RAD[heading_idx]);
        float forward_x = cosf(ego.sim_heading);
        float forward_y = sinf(ego.sim_heading);
        set_camera(&ego);
        Camera3D camera = g_renderer.camera;

        EXPECT_EQ_INT(camera.projection, CAMERA_PERSPECTIVE);
        EXPECT_NEAR(camera.fovy, CHASE_FOV_DEG, 0.0f);
        EXPECT_TRUE(expect_vector3_near(camera.up, (Vector3) {0.0f, 0.0f, 1.0f}, UNIT_TOLERANCE, "chase up") == 0);
        EXPECT_TRUE(
            expect_vector3_near(
                camera.position,
                (Vector3) {ego.sim_x - forward_x * CHASE_BACK_M,
                           ego.sim_y - forward_y * CHASE_BACK_M,
                           ego.sim_z + CHASE_EYE_HEIGHT_M},
                POSITION_TOLERANCE_M,
                "chase position")
            == 0);
        EXPECT_TRUE(
            expect_vector3_near(
                camera.target,
                (Vector3) {ego.sim_x + forward_x * CHASE_AHEAD_M,
                           ego.sim_y + forward_y * CHASE_AHEAD_M,
                           ego.sim_z + CHASE_TARGET_HEIGHT_M},
                POSITION_TOLERANCE_M,
                "chase target")
            == 0);

        // Stated as a distance and a direction rather than a formula: the eye is exactly
        // CHASE_BACK_M behind, CHASE_EYE_HEIGHT_M up, and the target CHASE_AHEAD_M ahead.
        Vector3 eye_offset = vector3_sub(camera.position, (Vector3) {ego.sim_x, ego.sim_y, ego.sim_z});
        Vector3 target_offset = vector3_sub(camera.target, (Vector3) {ego.sim_x, ego.sim_y, ego.sim_z});
        EXPECT_NEAR(eye_offset.x * forward_x + eye_offset.y * forward_y, -CHASE_BACK_M, POSITION_TOLERANCE_M);
        EXPECT_NEAR(-eye_offset.x * forward_y + eye_offset.y * forward_x, 0.0f, POSITION_TOLERANCE_M);
        EXPECT_NEAR(eye_offset.z, CHASE_EYE_HEIGHT_M, POSITION_TOLERANCE_M);
        EXPECT_NEAR(target_offset.x * forward_x + target_offset.y * forward_y, CHASE_AHEAD_M, POSITION_TOLERANCE_M);
        EXPECT_NEAR(-target_offset.x * forward_y + target_offset.y * forward_x, 0.0f, POSITION_TOLERANCE_M);
        EXPECT_NEAR(
            sqrtf(eye_offset.x * eye_offset.x + eye_offset.y * eye_offset.y),
            CHASE_BACK_M,
            POSITION_TOLERANCE_M);
    }
    return 0;
}

static int test_camera_driver(void) {
    g_renderer.camera_mode = CAMERA_MODE_DRIVER;
    for (int heading_idx = 0; heading_idx < TEST_HEADING_COUNT; heading_idx++) {
        Agent ego = render_test_ego(TEST_HEADINGS_RAD[heading_idx]);
        float forward_x = cosf(ego.sim_heading);
        float forward_y = sinf(ego.sim_heading);
        // Left of the car's heading, the side the driver sits on.
        float left_x = -sinf(ego.sim_heading);
        float left_y = cosf(ego.sim_heading);
        set_camera(&ego);
        Camera3D camera = g_renderer.camera;

        EXPECT_EQ_INT(camera.projection, CAMERA_PERSPECTIVE);
        EXPECT_NEAR(camera.fovy, DRIVER_FOV_DEG, 0.0f);
        EXPECT_TRUE(expect_vector3_near(camera.up, (Vector3) {0.0f, 0.0f, 1.0f}, UNIT_TOLERANCE, "driver up") == 0);
        EXPECT_TRUE(
            expect_vector3_near(
                camera.position,
                (Vector3) {ego.sim_x + forward_x * DRIVER_SEAT_FORWARD_M + left_x * DRIVER_SEAT_LEFT_M,
                           ego.sim_y + forward_y * DRIVER_SEAT_FORWARD_M + left_y * DRIVER_SEAT_LEFT_M,
                           ego.sim_z + DRIVER_EYE_HEIGHT_M},
                POSITION_TOLERANCE_M,
                "driver position")
            == 0);
        EXPECT_TRUE(
            expect_vector3_near(
                camera.target,
                (Vector3) {
                    ego.sim_x + forward_x * (DRIVER_SEAT_FORWARD_M + DRIVER_AHEAD_M) + left_x * DRIVER_SEAT_LEFT_M,
                    ego.sim_y + forward_y * (DRIVER_SEAT_FORWARD_M + DRIVER_AHEAD_M) + left_y * DRIVER_SEAT_LEFT_M,
                    ego.sim_z + DRIVER_TARGET_HEIGHT_M},
                POSITION_TOLERANCE_M,
                "driver target")
            == 0);

        // The eye sits in the driver's seat: ahead of centre, and to the LEFT for left-hand drive.
        Vector3 seat = vector3_sub(camera.position, (Vector3) {ego.sim_x, ego.sim_y, ego.sim_z});
        EXPECT_NEAR(seat.x * forward_x + seat.y * forward_y, DRIVER_SEAT_FORWARD_M, POSITION_TOLERANCE_M);
        EXPECT_NEAR(seat.x * left_x + seat.y * left_y, DRIVER_SEAT_LEFT_M, POSITION_TOLERANCE_M);
        EXPECT_TRUE(DRIVER_SEAT_LEFT_M > 0.0f);
        Vector3 look = vector3_sub(camera.target, camera.position);
        EXPECT_NEAR(sqrtf(look.x * look.x + look.y * look.y), DRIVER_AHEAD_M, POSITION_TOLERANCE_M);
        EXPECT_NEAR(look.x * forward_x + look.y * forward_y, DRIVER_AHEAD_M, POSITION_TOLERANCE_M);

        // The eye sits inside the ego's box, which is why demo_render_frame hides the ego in
        // this mode: drawn, the car's own body would fill the view.
        Vector3 eye_offset = vector3_sub(camera.position, (Vector3) {ego.sim_x, ego.sim_y, ego.sim_z});
        EXPECT_TRUE(fabsf(eye_offset.x * forward_x + eye_offset.y * forward_y) <= 0.5f * ego.sim_length);
        EXPECT_TRUE(fabsf(-eye_offset.x * forward_y + eye_offset.y * forward_x) <= 0.5f * ego.sim_width);
        EXPECT_TRUE(eye_offset.z <= ego.sim_height);
    }
    return 0;
}

static int test_camera_agent(void) {
    g_renderer.camera_mode = CAMERA_MODE_AGENT;
    for (int heading_idx = 0; heading_idx < TEST_HEADING_COUNT; heading_idx++) {
        Agent ego = render_test_ego(TEST_HEADINGS_RAD[heading_idx]);
        float forward_x = cosf(ego.sim_heading);
        float forward_y = sinf(ego.sim_heading);
        set_camera(&ego);
        Camera3D camera = g_renderer.camera;

        EXPECT_EQ_INT(camera.projection, CAMERA_ORTHOGRAPHIC);
        EXPECT_NEAR(camera.fovy, AGENT_VIEW_SPAN_M, 0.0f);
        EXPECT_TRUE(
            expect_vector3_near(
                camera.position,
                (Vector3) {ego.sim_x, ego.sim_y, ego.sim_z + AGENT_VIEW_HEIGHT_M},
                POSITION_TOLERANCE_M,
                "agent position")
            == 0);
        EXPECT_TRUE(
            expect_vector3_near(
                camera.target,
                (Vector3) {ego.sim_x, ego.sim_y, ego.sim_z},
                POSITION_TOLERANCE_M,
                "agent target")
            == 0);
        EXPECT_TRUE(
            expect_vector3_near(camera.up, (Vector3) {forward_x, forward_y, 0.0f}, UNIT_TOLERANCE, "agent up") == 0);
        EXPECT_NEAR(vector3_length(camera.up), 1.0f, UNIT_TOLERANCE);

        // The view basis raylib's MatrixLookAt builds, recomputed here: screen-up must be the
        // ego's forward vector and screen-right the ego's right-hand side, i.e. not mirrored.
        Vector3 view_z = vector3_sub(camera.position, camera.target);
        float view_z_length = vector3_length(view_z);
        EXPECT_NEAR(view_z_length, AGENT_VIEW_HEIGHT_M, POSITION_TOLERANCE_M);
        view_z = (Vector3) {view_z.x / view_z_length, view_z.y / view_z_length, view_z.z / view_z_length};
        Vector3 view_x = vector3_cross(camera.up, view_z);
        float view_x_length = vector3_length(view_x);
        EXPECT_NEAR(view_x_length, 1.0f, UNIT_TOLERANCE);
        view_x = (Vector3) {view_x.x / view_x_length, view_x.y / view_x_length, view_x.z / view_x_length};
        Vector3 view_y = vector3_cross(view_z, view_x);

        EXPECT_TRUE(expect_vector3_near(view_z, (Vector3) {0.0f, 0.0f, 1.0f}, UNIT_TOLERANCE, "agent view_z") == 0);
        EXPECT_TRUE(
            expect_vector3_near(view_y, (Vector3) {forward_x, forward_y, 0.0f}, UNIT_TOLERANCE, "agent screen up")
            == 0);
        EXPECT_TRUE(
            expect_vector3_near(view_x, (Vector3) {forward_y, -forward_x, 0.0f}, UNIT_TOLERANCE, "agent screen right")
            == 0);
        EXPECT_TRUE(
            expect_vector3_near(vector3_cross(view_x, view_y), view_z, UNIT_TOLERANCE, "agent basis handedness") == 0);
    }
    return 0;
}

static int test_camera_continuous_across_heading_wrap(void) {
    for (int mode = 0; mode < CAMERA_MODE_COUNT; mode++) {
        g_renderer.camera_mode = mode;
        Agent below_pi = render_test_ego(3.1415926f - HEADING_WRAP_DELTA_RAD);
        set_camera(&below_pi);
        Camera3D camera_below = g_renderer.camera;

        Agent above_pi = render_test_ego(-3.1415926f + HEADING_WRAP_DELTA_RAD);
        set_camera(&above_pi);
        Camera3D camera_above = g_renderer.camera;

        EXPECT_EQ_INT(camera_below.projection, camera_above.projection);
        EXPECT_NEAR(camera_below.fovy, camera_above.fovy, 0.0f);
        EXPECT_TRUE(expect_vector3_near(camera_below.position, camera_above.position, 1e-2f, "wrap position") == 0);
        EXPECT_TRUE(expect_vector3_near(camera_below.target, camera_above.target, 1e-2f, "wrap target") == 0);
        EXPECT_TRUE(expect_vector3_near(camera_below.up, camera_above.up, 1e-3f, "wrap up") == 0);
    }
    return 0;
}

// The presets are tuned to match the offline renders, so the live view and recorded clips agree.
// Pinned as literals: the formula tests above are all written in terms of these symbols.
static int test_camera_presets_match_the_tuned_values(void) {
    EXPECT_NEAR(CHASE_BACK_M, 9.0f, 0.0f);
    EXPECT_NEAR(CHASE_EYE_HEIGHT_M, 3.6f, 0.0f);
    EXPECT_NEAR(CHASE_AHEAD_M, 14.0f, 0.0f);
    EXPECT_NEAR(CHASE_FOV_DEG, 45.0f, 0.0f);
    EXPECT_NEAR(DRIVER_EYE_HEIGHT_M, 1.26f, 0.0f);
    EXPECT_NEAR(DRIVER_AHEAD_M, 30.0f, 0.0f);
    EXPECT_NEAR(DRIVER_FOV_DEG, 60.0f, 0.0f);
    EXPECT_EQ_INT(CAMERA_MODE_CHASE, 0);
    EXPECT_EQ_INT(CAMERA_MODE_DRIVER, 1);
    EXPECT_EQ_INT(CAMERA_MODE_AGENT, 2);
    EXPECT_EQ_INT(CAMERA_MODE_COUNT, 3);
    EXPECT_EQ_INT(CAR_MODEL_COUNT, 6);
    EXPECT_EQ_INT(NORMAL_CAR_MODEL_FIRST, 1);
    EXPECT_EQ_INT(NORMAL_CAR_MODEL_COUNT, 4);
    EXPECT_EQ_INT(COLLIDED_CAR_MODEL_IDX, 0);
    EXPECT_EQ_INT(STATIC_CAR_MODEL_IDX, 5);
    EXPECT_NEAR(HUMAN_WHEEL_RANGE_DEG, 900.0f, 0.0f);
    EXPECT_NEAR(HUMAN_STEERING_LIMIT_RAD, 0.55f, 0.0f);
    return 0;
}

static int test_camera_mode_setter_rejects_out_of_range(void) {
    g_renderer.camera_mode = CAMERA_MODE_CHASE;
    demo_render_set_camera_mode(CAMERA_MODE_AGENT);
    EXPECT_EQ_INT(g_renderer.camera_mode, CAMERA_MODE_AGENT);
    demo_render_set_camera_mode(-1);
    EXPECT_EQ_INT(g_renderer.camera_mode, CAMERA_MODE_AGENT);
    demo_render_set_camera_mode(CAMERA_MODE_COUNT);
    EXPECT_EQ_INT(g_renderer.camera_mode, CAMERA_MODE_AGENT);
    demo_render_set_camera_mode(CAMERA_MODE_CHASE);
    EXPECT_EQ_INT(g_renderer.camera_mode, CAMERA_MODE_CHASE);

    demo_render_set_night_mode(1);
    EXPECT_EQ_INT(g_renderer.night_mode, 1);
    demo_render_set_night_mode(7);
    EXPECT_EQ_INT(g_renderer.night_mode, 1);
    demo_render_set_night_mode(0);
    EXPECT_EQ_INT(g_renderer.night_mode, 0);
    return 0;
}

static int is_normal_car_model(int model_idx) {
    return model_idx >= NORMAL_CAR_MODEL_FIRST && model_idx < NORMAL_CAR_MODEL_FIRST + NORMAL_CAR_MODEL_COUNT;
}

static int test_car_model_for_agent(void) {
    // The sentinels must not double as ordinary models, or "red car" stops meaning "collided".
    EXPECT_TRUE(NORMAL_CAR_MODEL_FIRST + NORMAL_CAR_MODEL_COUNT <= CAR_MODEL_COUNT);
    EXPECT_FALSE(is_normal_car_model(COLLIDED_CAR_MODEL_IDX));
    EXPECT_FALSE(is_normal_car_model(STATIC_CAR_MODEL_IDX));

    for (int agent_id = 0; agent_id < 4 * CAR_MODEL_COUNT; agent_id++) {
        Agent agent = render_test_ego(0.0f);
        agent.id = agent_id;
        int model_idx = car_model_for_agent(&agent);
        EXPECT_TRUE(model_idx >= 0 && model_idx < CAR_MODEL_COUNT);
        // An active, uncollided agent never draws as a collided or a static car.
        EXPECT_TRUE(is_normal_car_model(model_idx));
        EXPECT_TRUE(model_idx != COLLIDED_CAR_MODEL_IDX && model_idx != STATIC_CAR_MODEL_IDX);
        EXPECT_EQ_INT(model_idx, NORMAL_CAR_MODEL_FIRST + agent_id % NORMAL_CAR_MODEL_COUNT);
        EXPECT_EQ_INT(car_model_for_agent(&agent), model_idx);

        // A different pose and size must not change the car you are looking at.
        agent.sim_x += 123.0f;
        agent.sim_heading = 2.0f;
        agent.sim_length = 6.0f;
        EXPECT_EQ_INT(car_model_for_agent(&agent), model_idx);
    }

    // Every ordinary model is reachable, so no colour is quietly unused.
    int model_seen[CAR_MODEL_COUNT] = {0};
    for (int agent_id = 0; agent_id < 4 * CAR_MODEL_COUNT; agent_id++) {
        Agent agent = render_test_ego(0.0f);
        agent.id = agent_id;
        model_seen[car_model_for_agent(&agent)] = 1;
    }
    for (int model_idx = 0; model_idx < CAR_MODEL_COUNT; model_idx++) {
        EXPECT_EQ_INT(model_seen[model_idx], is_normal_car_model(model_idx));
    }

    // Two ids NORMAL_CAR_MODEL_COUNT apart share a model; adjacent ids do not.
    for (int agent_id = 0; agent_id < 3 * CAR_MODEL_COUNT; agent_id++) {
        Agent agent = render_test_ego(0.0f);
        Agent next_agent = render_test_ego(0.0f);
        agent.id = agent_id;
        next_agent.id = agent_id + NORMAL_CAR_MODEL_COUNT;
        EXPECT_EQ_INT(car_model_for_agent(&agent), car_model_for_agent(&next_agent));
        next_agent.id = agent_id + 1;
        EXPECT_TRUE(car_model_for_agent(&agent) != car_model_for_agent(&next_agent));
    }

    for (int agent_id = -1; agent_id > -4 * CAR_MODEL_COUNT; agent_id--) {
        Agent agent = render_test_ego(0.0f);
        agent.id = agent_id;
        int model_idx = car_model_for_agent(&agent);
        EXPECT_TRUE(is_normal_car_model(model_idx));
        EXPECT_EQ_INT(model_idx, NORMAL_CAR_MODEL_FIRST + (-agent_id) % NORMAL_CAR_MODEL_COUNT);
    }

    for (int agent_id = 0; agent_id < CAR_MODEL_COUNT; agent_id++) {
        Agent collided = render_test_ego(0.0f);
        collided.id = agent_id;
        collided.metrics_array[COLLISION_IDX] = 1.0f;
        EXPECT_EQ_INT(car_model_for_agent(&collided), COLLIDED_CAR_MODEL_IDX);

        Agent inactive = render_test_ego(0.0f);
        inactive.id = agent_id;
        inactive.active_agent = 0;
        EXPECT_EQ_INT(car_model_for_agent(&inactive), STATIC_CAR_MODEL_IDX);

        // Collision wins over inactive: a parked car you hit still reads as collided.
        Agent both = render_test_ego(0.0f);
        both.id = agent_id;
        both.active_agent = 0;
        both.metrics_array[COLLISION_IDX] = 1.0f;
        EXPECT_EQ_INT(car_model_for_agent(&both), COLLIDED_CAR_MODEL_IDX);
    }

    // The collision flag only fires strictly above zero.
    Agent clean = render_test_ego(0.0f);
    clean.id = 2;
    clean.metrics_array[COLLISION_IDX] = 0.0f;
    EXPECT_EQ_INT(car_model_for_agent(&clean), NORMAL_CAR_MODEL_FIRST + 2 % NORMAL_CAR_MODEL_COUNT);
    return 0;
}

// Reads the hand-wheel angle back out of what draw_cockpit_wheel actually drew, rather than
// recomputing it, so dropping HUMAN_STEERING_RATIO from the renderer fails here.
static int drawn_hand_wheel_deg(float steering_angle_rad, float *hand_wheel_deg_out) {
    Agent ego = render_test_ego(0.0f);
    ego.steering_angle = steering_angle_rad;
    g_renderer.screen_width = RENDER_TEST_SCREEN_WIDTH;
    g_renderer.screen_height = RENDER_TEST_SCREEN_HEIGHT;
    reset_capture();
    draw_cockpit_wheel(&ego);

    if (captured_rect_count != COCKPIT_WHEEL_SPOKE_COUNT) {
        printf(
            "FAIL %s:%d: drew %d spokes, expected %d\n",
            __FILE__,
            __LINE__,
            captured_rect_count,
            COCKPIT_WHEEL_SPOKE_COUNT);
        return 1;
    }
    *hand_wheel_deg_out = captured_rect_rotation_deg[0] - 90.0f;
    for (int spoke = 1; spoke < COCKPIT_WHEEL_SPOKE_COUNT; spoke++) {
        float expected_deg = captured_rect_rotation_deg[0] + (float) spoke * SPOKE_STEP_DEG;
        if (fabsf(captured_rect_rotation_deg[spoke] - expected_deg) > ANGLE_TOLERANCE_DEG) {
            printf(
                "FAIL %s:%d: spoke %d at %f deg, expected %f deg\n",
                __FILE__,
                __LINE__,
                spoke,
                captured_rect_rotation_deg[spoke],
                expected_deg);
            return 1;
        }
    }
    return 0;
}

static int test_cockpit_wheel_full_lock_is_half_the_wheel_range(void) {
    // The whole calibration story: road-wheel lock x steering ratio == half the hand wheel's travel.
    EXPECT_NEAR(
        HUMAN_STEERING_LIMIT_RAD * HUMAN_STEERING_RATIO * RAD2DEG,
        0.5f * HUMAN_WHEEL_RANGE_DEG,
        ANGLE_TOLERANCE_DEG);
    EXPECT_NEAR(0.5f * HUMAN_WHEEL_RANGE_DEG, FULL_LOCK_HAND_WHEEL_DEG, 0.0f);

    // sim steering_angle is positive-left; the driver's surfaces are positive-right.
    float hand_wheel_deg = 0.0f;
    EXPECT_EQ_INT(drawn_hand_wheel_deg(FULL_RIGHT_LOCK_STEERING_RAD, &hand_wheel_deg), 0);
    EXPECT_NEAR(hand_wheel_deg, FULL_LOCK_HAND_WHEEL_DEG, ANGLE_TOLERANCE_DEG);
    EXPECT_TRUE(captured_text_count == 1);
    EXPECT_TRUE(strcmp(captured_text[0], "+450 deg") == 0);

    EXPECT_EQ_INT(drawn_hand_wheel_deg(FULL_LEFT_LOCK_STEERING_RAD, &hand_wheel_deg), 0);
    EXPECT_NEAR(hand_wheel_deg, -FULL_LOCK_HAND_WHEEL_DEG, ANGLE_TOLERANCE_DEG);
    EXPECT_TRUE(strcmp(captured_text[0], "-450 deg") == 0);

    // Not the road-wheel angle: at full lock the two differ by more than an order of magnitude.
    EXPECT_TRUE(fabsf(HUMAN_STEERING_LIMIT_RAD * RAD2DEG) < 0.1f * FULL_LOCK_HAND_WHEEL_DEG);
    return 0;
}

static int test_cockpit_wheel_is_linear_and_sign_correct(void) {
    const int sample_count = 21;
    for (int sample = 0; sample < sample_count; sample++) {
        float fraction = -1.0f + 2.0f * (float) sample / (float) (sample_count - 1);
        float steering_angle_rad = fraction * FULL_RIGHT_LOCK_STEERING_RAD;
        float hand_wheel_deg = 0.0f;
        EXPECT_EQ_INT(drawn_hand_wheel_deg(steering_angle_rad, &hand_wheel_deg), 0);
        EXPECT_NEAR(hand_wheel_deg, fraction * FULL_LOCK_HAND_WHEEL_DEG, ANGLE_TOLERANCE_DEG);
    }

    float at_zero = 0.0f;
    EXPECT_EQ_INT(drawn_hand_wheel_deg(0.0f, &at_zero), 0);
    EXPECT_NEAR(at_zero, 0.0f, ANGLE_TOLERANCE_DEG);
    // Index mark at twelve o'clock when centred (raylib screen y grows downwards).
    EXPECT_TRUE(captured_circle_count == 2);
    EXPECT_NEAR(captured_circle_center[0].x, captured_ring_center.x, 1e-3f);
    EXPECT_NEAR(captured_circle_center[0].y, captured_ring_center.y - WHEEL_LEVER_PX, 1e-3f);
    EXPECT_NEAR(captured_circle_radius_px[0], COCKPIT_WHEEL_INDEX_PX, 0.0f);
    EXPECT_NEAR(captured_circle_center[1].x, captured_ring_center.x, 0.0f);
    EXPECT_NEAR(captured_circle_center[1].y, captured_ring_center.y, 0.0f);
    EXPECT_NEAR(captured_circle_radius_px[1], COCKPIT_WHEEL_HUB_PX, 0.0f);
    EXPECT_TRUE(captured_ring_count == 1);
    EXPECT_NEAR(captured_ring_outer_radius_px, COCKPIT_WHEEL_RADIUS_PX, 0.0f);
    EXPECT_NEAR(captured_ring_inner_radius_px, COCKPIT_WHEEL_RADIUS_PX - COCKPIT_WHEEL_RIM_PX, 0.0f);
    EXPECT_NEAR(captured_ring_center.x, (float) RENDER_TEST_SCREEN_WIDTH * COCKPIT_WHEEL_CENTER_X_FRACTION, 1e-3f);
    // Left-hand drive: the wheel sits on the driver's side, not in the middle of the screen.
    EXPECT_TRUE(captured_ring_center.x < 0.5f * (float) RENDER_TEST_SCREEN_WIDTH - COCKPIT_WHEEL_RADIUS_PX);
    EXPECT_NEAR(captured_ring_center.y, (float) RENDER_TEST_SCREEN_HEIGHT - COCKPIT_WHEEL_BOTTOM_MARGIN_PX, 1e-3f);

    // A quarter turn of the hand wheel to the right puts the index mark on the screen's right.
    float quarter_turn_steering_rad = -90.0f * DEG2RAD / HUMAN_STEERING_RATIO;
    float quarter_turn_deg = 0.0f;
    EXPECT_EQ_INT(drawn_hand_wheel_deg(quarter_turn_steering_rad, &quarter_turn_deg), 0);
    EXPECT_NEAR(quarter_turn_deg, 90.0f, ANGLE_TOLERANCE_DEG);
    EXPECT_NEAR(captured_circle_center[0].x, captured_ring_center.x + WHEEL_LEVER_PX, 1e-3f);
    EXPECT_NEAR(captured_circle_center[0].y, captured_ring_center.y, 1e-3f);

    EXPECT_EQ_INT(drawn_hand_wheel_deg(-quarter_turn_steering_rad, &quarter_turn_deg), 0);
    EXPECT_NEAR(quarter_turn_deg, -90.0f, ANGLE_TOLERANCE_DEG);
    EXPECT_NEAR(captured_circle_center[0].x, captured_ring_center.x - WHEEL_LEVER_PX, 1e-3f);
    EXPECT_NEAR(captured_circle_center[0].y, captured_ring_center.y, 1e-3f);
    return 0;
}

static Drive render_test_env = {0};

// World-frame rotation the 3D front wheels are drawn with, about +Z (positive = counter-clockwise
// seen from above = pointing left).
static int drawn_front_wheel_deg(float steering_angle_rad, float *front_wheel_deg_out) {
    Agent agent = render_test_ego(0.6f);
    agent.steering_angle = steering_angle_rad;
    for (int model_idx = 0; model_idx < CAR_MODEL_COUNT; model_idx++) {
        g_renderer.car_model_size[model_idx] = (Vector3) {1.0f, 1.0f, 1.0f};
    }
    g_renderer.night_mode = 0;
    reset_capture();
    draw_vehicle(&render_test_env, &agent, agent.sim_x, agent.sim_y, agent.sim_heading, 1);

    // One rotation for the body heading, then one per front wheel.
    if (captured_rotate_count != 3) {
        printf("FAIL %s:%d: %d rotations, expected 3\n", __FILE__, __LINE__, captured_rotate_count);
        return 1;
    }
    for (int rotate_idx = 0; rotate_idx < 3; rotate_idx++) {
        if (expect_vector3_near(
                captured_rotate_axis[rotate_idx],
                (Vector3) {0.0f, 0.0f, 1.0f},
                UNIT_TOLERANCE,
                "rotation axis")
            != 0) {
            return 1;
        }
    }
    if (fabsf(captured_rotate_deg[1] - captured_rotate_deg[2]) > ANGLE_TOLERANCE_DEG) {
        printf(
            "FAIL %s:%d: front wheels disagree: %f vs %f\n",
            __FILE__,
            __LINE__,
            captured_rotate_deg[1],
            captured_rotate_deg[2]);
        return 1;
    }
    if (fabsf(captured_rotate_deg[0] - agent.sim_heading * RAD2DEG) > ANGLE_TOLERANCE_DEG) {
        printf(
            "FAIL %s:%d: body drawn at %f deg, expected %f deg\n",
            __FILE__,
            __LINE__,
            captured_rotate_deg[0],
            agent.sim_heading * RAD2DEG);
        return 1;
    }
    *front_wheel_deg_out = captured_rotate_deg[1];
    return 0;
}

// The HUD's "wheel ... -> road ..." line, which must read the same way round as the cockpit.
static int hud_steering_numbers(float steering_angle_rad, float *wheel_deg_out, float *road_deg_out) {
    Agent ego = render_test_ego(0.0f);
    ego.steering_angle = steering_angle_rad;
    g_renderer.screen_width = RENDER_TEST_SCREEN_WIDTH;
    g_renderer.screen_height = RENDER_TEST_SCREEN_HEIGHT;
    reset_capture();
    draw_hud(&render_test_env, &ego);

    for (int text_idx = 0; text_idx < captured_text_count && text_idx < CAPTURE_TEXT_MAX; text_idx++) {
        if (sscanf(captured_text[text_idx], "wheel %f deg -> road %f deg", wheel_deg_out, road_deg_out) == 2) {
            return 0;
        }
    }
    printf("FAIL %s:%d: no 'wheel ... -> road ...' line among %d HUD lines\n", __FILE__, __LINE__, captured_text_count);
    return 1;
}

// Finding from the first pass: the cockpit wheel is drawn in 2D screen space (y down, positive is
// clockwise) and the front wheels in the world frame (z up, positive is counter-clockwise), from
// steering angles of opposite sign. Both must still agree on which way is right.
static int test_driver_surfaces_agree_on_which_way_is_right(void) {
    float hand_wheel_deg = 0.0f, front_wheel_deg = 0.0f, hud_wheel_deg = 0.0f, hud_road_deg = 0.0f;

    EXPECT_EQ_INT(drawn_hand_wheel_deg(FULL_RIGHT_LOCK_STEERING_RAD, &hand_wheel_deg), 0);
    float right_lock_index_x = captured_circle_center[0].x;
    float right_lock_center_x = captured_ring_center.x;
    EXPECT_EQ_INT(drawn_front_wheel_deg(FULL_RIGHT_LOCK_STEERING_RAD, &front_wheel_deg), 0);
    EXPECT_EQ_INT(hud_steering_numbers(FULL_RIGHT_LOCK_STEERING_RAD, &hud_wheel_deg, &hud_road_deg), 0);

    // Cockpit: clockwise, so the index mark leaves twelve o'clock towards the screen's right.
    EXPECT_TRUE(hand_wheel_deg > 0.0f);
    EXPECT_TRUE(right_lock_index_x > right_lock_center_x);
    // Front wheels: clockwise seen from above, which about +Z is a negative rotation.
    EXPECT_TRUE(front_wheel_deg < 0.0f);
    EXPECT_NEAR(front_wheel_deg, FULL_RIGHT_LOCK_STEERING_RAD * RAD2DEG, ANGLE_TOLERANCE_DEG);
    // HUD: positive, and the same number the cockpit draws.
    EXPECT_TRUE(hud_wheel_deg > 0.0f && hud_road_deg > 0.0f);
    EXPECT_NEAR(hud_wheel_deg, hand_wheel_deg, 0.1f);
    EXPECT_NEAR(hud_wheel_deg, hud_road_deg * HUMAN_STEERING_RATIO, 0.5f);

    EXPECT_EQ_INT(drawn_hand_wheel_deg(FULL_LEFT_LOCK_STEERING_RAD, &hand_wheel_deg), 0);
    float left_lock_index_x = captured_circle_center[0].x;
    float left_lock_center_x = captured_ring_center.x;
    EXPECT_EQ_INT(drawn_front_wheel_deg(FULL_LEFT_LOCK_STEERING_RAD, &front_wheel_deg), 0);
    EXPECT_EQ_INT(hud_steering_numbers(FULL_LEFT_LOCK_STEERING_RAD, &hud_wheel_deg, &hud_road_deg), 0);

    EXPECT_TRUE(hand_wheel_deg < 0.0f);
    EXPECT_TRUE(left_lock_index_x < left_lock_center_x);
    EXPECT_TRUE(front_wheel_deg > 0.0f);
    EXPECT_NEAR(front_wheel_deg, FULL_LEFT_LOCK_STEERING_RAD * RAD2DEG, ANGLE_TOLERANCE_DEG);
    EXPECT_TRUE(hud_wheel_deg < 0.0f && hud_road_deg < 0.0f);
    EXPECT_NEAR(hud_wheel_deg, hand_wheel_deg, 0.1f);

    // Centred is centred on every surface.
    EXPECT_EQ_INT(drawn_hand_wheel_deg(0.0f, &hand_wheel_deg), 0);
    EXPECT_EQ_INT(drawn_front_wheel_deg(0.0f, &front_wheel_deg), 0);
    EXPECT_EQ_INT(hud_steering_numbers(0.0f, &hud_wheel_deg, &hud_road_deg), 0);
    EXPECT_NEAR(hand_wheel_deg, 0.0f, ANGLE_TOLERANCE_DEG);
    EXPECT_NEAR(front_wheel_deg, 0.0f, ANGLE_TOLERANCE_DEG);
    EXPECT_NEAR(hud_wheel_deg, 0.0f, 0.05f);
    EXPECT_NEAR(hud_road_deg, 0.0f, 0.05f);
    return 0;
}

// Partner cars carry no steering state, so only the human's wheels are drawn turned.
static int test_only_the_human_car_draws_front_wheels(void) {
    Agent agent = render_test_ego(0.6f);
    agent.steering_angle = FULL_RIGHT_LOCK_STEERING_RAD;
    for (int model_idx = 0; model_idx < CAR_MODEL_COUNT; model_idx++) {
        g_renderer.car_model_size[model_idx] = (Vector3) {1.0f, 1.0f, 1.0f};
    }
    g_renderer.night_mode = 0;
    reset_capture();
    draw_vehicle(&render_test_env, &agent, agent.sim_x, agent.sim_y, agent.sim_heading, 0);
    EXPECT_EQ_INT(captured_rotate_count, 1);
    EXPECT_NEAR(captured_rotate_deg[0], agent.sim_heading * RAD2DEG, ANGLE_TOLERANCE_DEG);
    return 0;
}

static Agent predict_test_agent(float heading_rad, float speed_mps, float yaw_rate_rps) {
    Agent agent = drive_test_agent(40.0f, -15.0f, heading_rad);
    agent.sim_speed_signed = speed_mps;
    agent.yaw_rate = yaw_rate_rps;
    return agent;
}

static int test_predict_pose_identity_and_straight_line(void) {
    for (int heading_idx = 0; heading_idx < TEST_HEADING_COUNT; heading_idx++) {
        float heading_rad = TEST_HEADINGS_RAD[heading_idx];
        Agent agent = predict_test_agent(heading_rad, 13.0f, 0.4f);
        float x_m = 0.0f, y_m = 0.0f, heading_out_rad = 0.0f;
        predict_pose(&agent, 0.0f, &x_m, &y_m, &heading_out_rad);
        EXPECT_NEAR(x_m, agent.sim_x, 0.0f);
        EXPECT_NEAR(y_m, agent.sim_y, 0.0f);
        EXPECT_NEAR(heading_out_rad, agent.sim_heading, 1e-6f);

        Agent straight = predict_test_agent(heading_rad, 13.0f, 0.0f);
        for (int tenth = 1; tenth <= 10; tenth++) {
            float elapsed_seconds = 0.03f * (float) tenth;
            predict_pose(&straight, elapsed_seconds, &x_m, &y_m, &heading_out_rad);
            EXPECT_NEAR(x_m, straight.sim_x + 13.0f * elapsed_seconds * cosf(heading_rad), ARC_TOLERANCE_M);
            EXPECT_NEAR(y_m, straight.sim_y + 13.0f * elapsed_seconds * sinf(heading_rad), ARC_TOLERANCE_M);
            EXPECT_NEAR(heading_out_rad, heading_rad, 1e-6f);
        }

        // Reversing holds the heading and walks backwards along it.
        Agent reversing = predict_test_agent(heading_rad, -2.5f, 0.0f);
        predict_pose(&reversing, 0.5f, &x_m, &y_m, &heading_out_rad);
        EXPECT_NEAR(x_m, reversing.sim_x - 1.25f * cosf(heading_rad), ARC_TOLERANCE_M);
        EXPECT_NEAR(y_m, reversing.sim_y - 1.25f * sinf(heading_rad), ARC_TOLERANCE_M);
        EXPECT_NEAR(heading_out_rad, heading_rad, 1e-6f);

        // Standing still stays put even with a yaw rate on the books.
        Agent parked = predict_test_agent(heading_rad, 0.0f, 0.3f);
        predict_pose(&parked, 0.2f, &x_m, &y_m, &heading_out_rad);
        EXPECT_NEAR(x_m, parked.sim_x, 0.0f);
        EXPECT_NEAR(y_m, parked.sim_y, 0.0f);
    }
    return 0;
}

static int test_predict_pose_traces_the_arc(void) {
    const float speed_mps = 11.0f;
    const float yaw_rates_rps[] = {0.35f, -0.35f, 0.08f, -0.6f};
    for (int rate_idx = 0; rate_idx < 4; rate_idx++) {
        float yaw_rate_rps = yaw_rates_rps[rate_idx];
        for (int heading_idx = 0; heading_idx < TEST_HEADING_COUNT; heading_idx++) {
            float heading_rad = TEST_HEADINGS_RAD[heading_idx];
            Agent agent = predict_test_agent(heading_rad, speed_mps, yaw_rate_rps);
            // Independent circle: radius speed/yaw_rate about a centre abeam the car.
            float radius_m = speed_mps / yaw_rate_rps;
            float center_x = agent.sim_x - radius_m * sinf(heading_rad);
            float center_y = agent.sim_y + radius_m * cosf(heading_rad);

            for (int tenth = 1; tenth <= 10; tenth++) {
                float elapsed_seconds = 0.03f * (float) tenth;
                float swept_rad = yaw_rate_rps * elapsed_seconds;
                float x_m = 0.0f, y_m = 0.0f, heading_out_rad = 0.0f;
                predict_pose(&agent, elapsed_seconds, &x_m, &y_m, &heading_out_rad);
                EXPECT_NEAR(x_m, center_x + radius_m * sinf(heading_rad + swept_rad), ARC_TOLERANCE_M);
                EXPECT_NEAR(y_m, center_y - radius_m * cosf(heading_rad + swept_rad), ARC_TOLERANCE_M);
                EXPECT_NEAR(
                    normalize_heading(heading_out_rad - normalize_heading(heading_rad + swept_rad)),
                    0.0f,
                    1e-5f);
                // Constant speed means a constant distance from the circle's centre.
                float dx = x_m - center_x;
                float dy = y_m - center_y;
                EXPECT_NEAR(sqrtf(dx * dx + dy * dy), fabsf(radius_m), ARC_TOLERANCE_M);
            }
        }
    }
    return 0;
}

static int test_predict_pose_never_mutates_the_agent(void) {
    Agent agent = predict_test_agent(0.9f, 14.0f, 0.5f);
    Agent snapshot;
    memcpy(&snapshot, &agent, sizeof(Agent));
    float x_m = 0.0f, y_m = 0.0f, heading_out_rad = 0.0f;
    for (int tenth = 0; tenth <= 20; tenth++) {
        predict_pose(&agent, 0.05f * (float) tenth, &x_m, &y_m, &heading_out_rad);
        EXPECT_EQ_INT(memcmp(&agent, &snapshot, sizeof(Agent)), 0);
    }
    // A prediction that silently became the real pose would show up here.
    EXPECT_NEAR(agent.sim_x, snapshot.sim_x, 0.0f);
    EXPECT_NEAR(agent.sim_heading, snapshot.sim_heading, 0.0f);
    EXPECT_TRUE(fabsf(x_m - agent.sim_x) > 1.0f);
    return 0;
}

static int test_renderer_starts_closed(void) {
    // Nothing in this file may open a window; the suite is meaningless if one leaked open.
    EXPECT_EQ_INT(g_renderer.window_open, 0);
    EXPECT_EQ_INT(g_renderer.models_loaded, 0);
    EXPECT_EQ_INT(demo_render_should_close(), 1);
    demo_render_close();
    EXPECT_EQ_INT(g_renderer.window_open, 0);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_camera_chase);
    RUN_TEST(test_camera_driver);
    RUN_TEST(test_camera_agent);
    RUN_TEST(test_camera_continuous_across_heading_wrap);
    RUN_TEST(test_camera_presets_match_the_tuned_values);
    RUN_TEST(test_camera_mode_setter_rejects_out_of_range);
    RUN_TEST(test_car_model_for_agent);
    RUN_TEST(test_cockpit_wheel_full_lock_is_half_the_wheel_range);
    RUN_TEST(test_cockpit_wheel_is_linear_and_sign_correct);
    RUN_TEST(test_driver_surfaces_agree_on_which_way_is_right);
    RUN_TEST(test_only_the_human_car_draws_front_wheels);
    RUN_TEST(test_predict_pose_identity_and_straight_line);
    RUN_TEST(test_predict_pose_traces_the_arc);
    RUN_TEST(test_predict_pose_never_mutates_the_agent);
    RUN_TEST(test_renderer_starts_closed);
    return test_summary(failures);
}
