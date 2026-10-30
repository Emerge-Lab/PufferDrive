// Demo-only renderer for the wheel demo. Deliberately NOT part of the training extension: it
// lives in its own shared object so pufferlib.ocean.drive.binding stays free of raylib, which is
// what #593 removed it for.
//
// World convention: Z-up, sim coordinates drawn directly, heading is a rotation about Z. The .glb
// car models are authored Y-up, so each is tipped 90 degrees about X -- same as the pre-#593
// renderer these models came from.
#ifndef PUFFERLIB_OCEAN_DRIVE_RENDER_H
#define PUFFERLIB_OCEAN_DRIVE_RENDER_H

#include "pufferlib/ocean/drive/drive.h"
#include "raylib.h"
#include "rlgl.h"

#include <math.h>
#include <stdio.h>

#define CAMERA_MODE_CHASE 0
#define CAMERA_MODE_DRIVER 1
#define CAMERA_MODE_AGENT 2
#define CAMERA_MODE_COUNT 3

// Matches the offline renderer's tuned presets so the live view and recorded clips agree.
#define CHASE_BACK_M 25.0f
#define CHASE_EYE_HEIGHT_M 15.0f
#define CHASE_AHEAD_M 40.0f
#define CHASE_FOV_DEG 45.0f
#define DRIVER_EYE_HEIGHT_M 1.4f
#define DRIVER_AHEAD_M 30.0f
#define DRIVER_TARGET_HEIGHT_M 1.0f
#define DRIVER_FOV_DEG 60.0f
#define AGENT_VIEW_SPAN_M 120.0f
#define AGENT_VIEW_HEIGHT_M 100.0f

#define RENDER_CULL_RANGE_M 220.0f
#define GROUND_SPAN_M 1400.0f
#define ROAD_SURFACE_HEIGHT_M 0.01f
#define EGO_MARKER_HEIGHT_M 4.2f
#define EGO_MARKER_RADIUS_M 0.9f

// Cockpit wheel. It is rotated by the HAND-wheel angle, not the road-wheel angle, so the thing on
// screen turns exactly as far as the wheel in your hands: full lock is +-450 deg, not +-31.5.
#define COCKPIT_WHEEL_RADIUS_PX 138.0f
#define COCKPIT_WHEEL_RIM_PX 17.0f
#define COCKPIT_WHEEL_SPOKE_COUNT 3
#define COCKPIT_WHEEL_SPOKE_WIDTH_PX 14.0f
#define COCKPIT_WHEEL_HUB_PX 24.0f
#define COCKPIT_WHEEL_INDEX_PX 12.0f
#define COCKPIT_WHEEL_BOTTOM_MARGIN_PX 126.0f
#define COCKPIT_WHEEL_CENTER_X_FRACTION 0.27f // left-hand drive: the wheel is on the driver's side
#define COCKPIT_DASH_HEIGHT_PX 112.0f
#define ROAD_EDGE_HEIGHT_M 0.12f
#define LANE_LINE_HEIGHT_M 0.02f
#define STOP_LINE_HEIGHT_M 0.05f
#define GOAL_MARKER_RADIUS_M 1.6f
#define VEHICLE_HEIGHT_M 1.5f
#define MODEL_TIP_DEG 90.0f

#define CAR_MODEL_COUNT 6
#define COLLIDED_CAR_MODEL_IDX 0
#define STATIC_CAR_MODEL_IDX 5
// The two sentinels are reserved, so an ordinary car never draws as crashed or parked.
#define NORMAL_CAR_MODEL_FIRST 1
#define NORMAL_CAR_MODEL_COUNT 4

// Lamps, as small emissive boxes on the body. Sizes are in metres.
#define LAMP_SIZE_M 0.18f
#define LAMP_LATERAL_FRACTION 0.36f
#define LAMP_HEIGHT_M 0.62f
#define BRAKE_LIGHT_ACCEL_MPS2 (-0.6f) // below this the brake lamps come on
#define HEADLIGHT_BEAM_LENGTH_M 18.0f
#define HEADLIGHT_BEAM_RADIUS_M 2.2f

static const Color COLOR_SKY_DAY = {207, 224, 240, 255};
static const Color COLOR_SKY_NIGHT = {10, 12, 22, 255};
static const Color COLOR_GROUND_DAY = {220, 223, 214, 255};
static const Color COLOR_GROUND_NIGHT = {16, 18, 20, 255};
static const Color COLOR_ROAD_EDGE_DAY = {42, 46, 53, 255};
static const Color COLOR_ROAD_EDGE_NIGHT = {96, 100, 110, 255};
static const Color COLOR_LANE_DAY = {242, 242, 242, 255};
static const Color COLOR_LANE_NIGHT = {120, 122, 128, 255};
static const Color COLOR_ROAD_SURFACE_DAY = {154, 161, 171, 255};
static const Color COLOR_ROAD_SURFACE_NIGHT = {30, 31, 35, 255};
static const Color COLOR_EGO_MARKER = {60, 200, 255, 255};
static const Color COLOR_WHEEL_RIM = {96, 102, 115, 255};
static const Color COLOR_WHEEL_SPOKE = {126, 133, 148, 255};
static const Color COLOR_WHEEL_INDEX = {235, 60, 55, 255};
static const Color COLOR_WHEEL_HUB = {58, 62, 72, 255};
static const Color COLOR_DASH = {32, 34, 39, 255};
static const Color COLOR_DASH_EDGE = {58, 62, 70, 255};
static const Color COLOR_GOAL = {235, 70, 215, 255};
static const Color COLOR_HUD = {245, 245, 245, 255};
static const Color COLOR_HUD_PANEL = {12, 14, 18, 200};
static const Color COLOR_WARN = {240, 90, 70, 255};
static const Color COLOR_BRAKE_ON = {255, 40, 30, 255};
static const Color COLOR_BRAKE_OFF = {150, 38, 32, 255};
static const Color COLOR_HEADLIGHT = {255, 246, 214, 255};
static const Color COLOR_HEADLIGHT_BEAM = {255, 244, 200, 40};
static const Color COLOR_NIGHT_TINT = {178, 186, 205, 255};

static const char *CAR_MODEL_FILES[CAR_MODEL_COUNT]
    = {"RedCar.glb", "WhiteCar.glb", "BlueCar.glb", "YellowCar.glb", "GreenCar.glb", "GreyCar.glb"};

typedef struct {
    int window_open;
    int models_loaded;
    int camera_mode;
    int night_mode;
    int screen_width;
    int screen_height;
    Camera3D camera;
    Model cars[CAR_MODEL_COUNT];
    Vector3 car_model_size[CAR_MODEL_COUNT];
} DemoRenderer;

static DemoRenderer g_renderer = {0};

static Color renderer_sky(void) {
    return g_renderer.night_mode ? COLOR_SKY_NIGHT : COLOR_SKY_DAY;
}

static Color renderer_ground(void) {
    return g_renderer.night_mode ? COLOR_GROUND_NIGHT : COLOR_GROUND_DAY;
}

// One model per vehicle, stable for the agent's lifetime so cars do not change colour as you drive.
static int car_model_for_agent(const Agent *agent) {
    int model_idx = NORMAL_CAR_MODEL_FIRST + (agent->id < 0 ? -agent->id : agent->id) % NORMAL_CAR_MODEL_COUNT;
    if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
        return COLLIDED_CAR_MODEL_IDX;
    }
    if (!agent->active_agent) {
        return STATIC_CAR_MODEL_IDX;
    }
    return model_idx;
}

static void draw_lamp(float local_x, float local_y, Color color) {
    rlPushMatrix();
    rlTranslatef(local_x, local_y, LAMP_HEIGHT_M);
    DrawCube((Vector3) {0.0f, 0.0f, 0.0f}, LAMP_SIZE_M, LAMP_SIZE_M, LAMP_SIZE_M, color);
    rlPopMatrix();
}

// Brake and head lamps, drawn in the car's local frame inside its transform block.
static void draw_vehicle_lamps(const Agent *agent, int is_braking) {
    float half_length_m = 0.5f * agent->sim_length;
    float lateral_m = LAMP_LATERAL_FRACTION * agent->sim_width;
    Color brake_color = is_braking ? COLOR_BRAKE_ON : COLOR_BRAKE_OFF;

    draw_lamp(-half_length_m, lateral_m, brake_color);
    draw_lamp(-half_length_m, -lateral_m, brake_color);

    if (!g_renderer.night_mode) {
        return;
    }
    draw_lamp(half_length_m, lateral_m, COLOR_HEADLIGHT);
    draw_lamp(half_length_m, -lateral_m, COLOR_HEADLIGHT);
    DrawCylinderEx(
        (Vector3) {half_length_m, 0.0f, LAMP_HEIGHT_M},
        (Vector3) {half_length_m + HEADLIGHT_BEAM_LENGTH_M, 0.0f, 0.1f},
        0.25f,
        HEADLIGHT_BEAM_RADIUS_M,
        12,
        COLOR_HEADLIGHT_BEAM);
}

static void draw_vehicle(Drive *env, const Agent *agent, float draw_x, float draw_y, float draw_heading, int is_human) {
    int model_idx = car_model_for_agent(agent);
    Vector3 model_size = g_renderer.car_model_size[model_idx];
    Vector3 scale
        = {agent->sim_length / model_size.x,
           agent->sim_height > 0.0f ? agent->sim_height / model_size.y : VEHICLE_HEIGHT_M / model_size.y,
           agent->sim_width / model_size.z};
    Color tint = g_renderer.night_mode ? COLOR_NIGHT_TINT : WHITE;

    rlPushMatrix();
    rlTranslatef(draw_x, draw_y, agent->sim_z);
    rlRotatef(draw_heading * RAD2DEG, 0.0f, 0.0f, 1.0f);
    DrawModelEx(
        g_renderer.cars[model_idx],
        (Vector3) {0.0f, 0.0f, 0.0f},
        (Vector3) {1.0f, 0.0f, 0.0f},
        MODEL_TIP_DEG,
        scale,
        tint);
    draw_vehicle_lamps(agent, agent->accel_long < BRAKE_LIGHT_ACCEL_MPS2);
    if (is_human) {
        // Front wheels at the agent's ACTUAL steering angle, so the screen and the wheel in your
        // hands always show the same number.
        float front_offset_m = 0.5f * agent->wheelbase;
        float half_axle_m = 0.5f * agent->sim_width;
        for (int side = -1; side <= 1; side += 2) {
            rlPushMatrix();
            rlTranslatef(front_offset_m, (float) side * half_axle_m, 0.30f);
            rlRotatef(agent->steering_angle * RAD2DEG, 0.0f, 0.0f, 1.0f);
            DrawCube((Vector3) {0.0f, 0.0f, 0.0f}, 0.64f, 0.22f, 0.60f, (Color) {22, 24, 28, 255});
            rlPopMatrix();
        }
    }
    rlPopMatrix();
    (void) env;
}

// Lane centrelines carry a per-point width, so the drivable surface is a quad strip along each
// lane. Without it the world reads as a wireframe and you cannot see where the road is.
static void draw_road_surface(Drive *env, float focus_x, float focus_y) {
    float cull_sq = RENDER_CULL_RANGE_M * RENDER_CULL_RANGE_M;
    Color surface = g_renderer.night_mode ? COLOR_ROAD_SURFACE_NIGHT : COLOR_ROAD_SURFACE_DAY;
    rlDisableBackfaceCulling();
    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *road = &env->road_elements[i];
        if (!is_drivable_road_lane(road->type)) {
            continue;
        }
        for (int k = 0; k + 1 < road->segment_size; k++) {
            float dx = road->x[k] - focus_x;
            float dy = road->y[k] - focus_y;
            if (dx * dx + dy * dy > cull_sq) {
                continue;
            }
            float sx = road->x[k + 1] - road->x[k];
            float sy = road->y[k + 1] - road->y[k];
            float length_m = sqrtf(sx * sx + sy * sy);
            if (length_m < 1e-4f) {
                continue;
            }
            float half_a = 0.5f * (road->widths != NULL ? road->widths[k] : LANE_WIDTH);
            float half_b = 0.5f * (road->widths != NULL ? road->widths[k + 1] : LANE_WIDTH);
            float nx = -sy / length_m;
            float ny = sx / length_m;
            float z0 = road->z[k] + ROAD_SURFACE_HEIGHT_M;
            float z1 = road->z[k + 1] + ROAD_SURFACE_HEIGHT_M;
            Vector3 a = {road->x[k] + nx * half_a, road->y[k] + ny * half_a, z0};
            Vector3 b = {road->x[k] - nx * half_a, road->y[k] - ny * half_a, z0};
            Vector3 c = {road->x[k + 1] - nx * half_b, road->y[k + 1] - ny * half_b, z1};
            Vector3 d = {road->x[k + 1] + nx * half_b, road->y[k + 1] + ny * half_b, z1};
            DrawTriangle3D(a, b, c, surface);
            DrawTriangle3D(a, c, d, surface);
        }
    }
    rlEnableBackfaceCulling();
}

static void draw_road(Drive *env, float focus_x, float focus_y) {
    float cull_sq = RENDER_CULL_RANGE_M * RENDER_CULL_RANGE_M;
    Color edge_color = g_renderer.night_mode ? COLOR_ROAD_EDGE_NIGHT : COLOR_ROAD_EDGE_DAY;
    Color lane_color = g_renderer.night_mode ? COLOR_LANE_NIGHT : COLOR_LANE_DAY;
    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *road = &env->road_elements[i];
        int is_edge = is_road_edge(road->type);
        if (!is_edge && !is_drivable_road_lane(road->type)) {
            continue;
        }
        float height_m = is_edge ? ROAD_EDGE_HEIGHT_M : LANE_LINE_HEIGHT_M;
        Color color = is_edge ? edge_color : lane_color;
        for (int k = 0; k + 1 < road->segment_size; k++) {
            float dx = road->x[k] - focus_x;
            float dy = road->y[k] - focus_y;
            if (dx * dx + dy * dy > cull_sq) {
                continue;
            }
            DrawLine3D(
                (Vector3) {road->x[k], road->y[k], road->z[k] + height_m},
                (Vector3) {road->x[k + 1], road->y[k + 1], road->z[k + 1] + height_m},
                color);
        }
    }
}

static Color traffic_control_color(Drive *env, TrafficControlElement *element) {
    if (element->type == TRAFFIC_CONTROL_TYPE_STOP_SIGN) {
        return (Color) {210, 60, 50, 255};
    }
    int state = (element->state_size > 0) ? element->states[env->timestep % element->state_size]
                                          : TRAFFIC_CONTROL_STATE_UNKNOWN;
    if (state == TRAFFIC_CONTROL_STATE_GREEN) {
        return (Color) {60, 200, 90, 255};
    }
    if (state == TRAFFIC_CONTROL_STATE_YELLOW) {
        return (Color) {235, 195, 60, 255};
    }
    if (state == TRAFFIC_CONTROL_STATE_RED) {
        return (Color) {235, 70, 60, 255};
    }
    return (Color) {140, 140, 140, 255};
}

static void draw_traffic_controls(Drive *env, float focus_x, float focus_y) {
    float cull_sq = RENDER_CULL_RANGE_M * RENDER_CULL_RANGE_M;
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *element = &env->traffic_elements[i];
        float dx = element->stop_line[0] - focus_x;
        float dy = element->stop_line[1] - focus_y;
        if (dx * dx + dy * dy > cull_sq) {
            continue;
        }
        DrawLine3D(
            (Vector3) {element->stop_line[0], element->stop_line[1], element->stop_line[2] + STOP_LINE_HEIGHT_M},
            (Vector3) {element->stop_line[3], element->stop_line[4], element->stop_line[5] + STOP_LINE_HEIGHT_M},
            traffic_control_color(env, element));
    }
}

static void draw_goals(const Agent *agent) {
    for (int goal_idx = agent->current_goal_idx; goal_idx < agent->goal_count; goal_idx++) {
        DrawCircle3D(
            (Vector3) {agent->list_goal_x[goal_idx], agent->list_goal_y[goal_idx], agent->list_goal_z[goal_idx] + 0.1f},
            GOAL_MARKER_RADIUS_M,
            (Vector3) {0.0f, 0.0f, 1.0f},
            0.0f,
            COLOR_GOAL);
    }
}

static void draw_agents(
    Drive *env,
    const float *pose_x,
    const float *pose_y,
    const float *pose_heading,
    float focus_x,
    float focus_y,
    int hide_human) {
    float cull_sq = RENDER_CULL_RANGE_M * RENDER_CULL_RANGE_M;
    for (int j = 0; j < env->num_agents; j++) {
        int agent_idx = (j < env->active_agent_count) ? env->active_agent_indices[j]
                                                      : env->static_agent_indices[j - env->active_agent_count];
        Agent *agent = &env->agents[agent_idx];
        if (agent->sim_valid != 1 || agent->removed) {
            continue;
        }
        int is_human = (agent_idx == env->human_agent_idx);
        if (is_human && hide_human) {
            continue;
        }
        float draw_x = pose_x[j] - env->world_mean_x;
        float draw_y = pose_y[j] - env->world_mean_y;
        float dx = draw_x - focus_x;
        float dy = draw_y - focus_y;
        if (dx * dx + dy * dy > cull_sq) {
            continue;
        }
        draw_vehicle(env, agent, draw_x, draw_y, pose_heading[j], is_human);
        if (is_human) {
            DrawCylinderEx(
                (Vector3) {draw_x, draw_y, agent->sim_z + EGO_MARKER_HEIGHT_M},
                (Vector3) {draw_x, draw_y, agent->sim_z + EGO_MARKER_HEIGHT_M + 1.6f},
                EGO_MARKER_RADIUS_M,
                0.0f,
                10,
                COLOR_EGO_MARKER);
        }
    }
}

static void set_camera(const Agent *ego) {
    Camera3D *camera = &g_renderer.camera;
    float forward_x = ego->cos_heading;
    float forward_y = ego->sin_heading;
    camera->up = (Vector3) {0.0f, 0.0f, 1.0f};

    if (g_renderer.camera_mode == CAMERA_MODE_AGENT) {
        camera->projection = CAMERA_ORTHOGRAPHIC;
        camera->fovy = AGENT_VIEW_SPAN_M;
        camera->position = (Vector3) {ego->sim_x, ego->sim_y, ego->sim_z + AGENT_VIEW_HEIGHT_M};
        camera->target = (Vector3) {ego->sim_x, ego->sim_y, ego->sim_z};
        camera->up = (Vector3) {forward_x, forward_y, 0.0f};
        return;
    }

    camera->projection = CAMERA_PERSPECTIVE;
    if (g_renderer.camera_mode == CAMERA_MODE_DRIVER) {
        camera->fovy = DRIVER_FOV_DEG;
        camera->position = (Vector3) {ego->sim_x, ego->sim_y, ego->sim_z + DRIVER_EYE_HEIGHT_M};
        camera->target = (Vector3) {ego->sim_x + forward_x * DRIVER_AHEAD_M,
                                    ego->sim_y + forward_y * DRIVER_AHEAD_M,
                                    ego->sim_z + DRIVER_TARGET_HEIGHT_M};
        return;
    }

    camera->fovy = CHASE_FOV_DEG;
    camera->position = (Vector3) {ego->sim_x - forward_x * CHASE_BACK_M,
                                  ego->sim_y - forward_y * CHASE_BACK_M,
                                  ego->sim_z + CHASE_EYE_HEIGHT_M};
    camera->target
        = (Vector3) {ego->sim_x + forward_x * CHASE_AHEAD_M, ego->sim_y + forward_y * CHASE_AHEAD_M, ego->sim_z + 1.0f};
}

static const char *camera_mode_name(int camera_mode) {
    if (camera_mode == CAMERA_MODE_DRIVER) {
        return "driver";
    }
    if (camera_mode == CAMERA_MODE_AGENT) {
        return "agent";
    }
    return "chase";
}

static const char *gear_label(int gear) {
    if (gear == GEAR_REVERSE) {
        return "R";
    }
    if (gear == GEAR_NEUTRAL) {
        return "N";
    }
    return TextFormat("%d", gear);
}

// A steering wheel that is faithful to the physical one: HUMAN_STEERING_RATIO converts the car's
// road-wheel angle back into the hand-wheel angle the driver actually applied.
// Driver-facing sign: positive is a RIGHT turn, matching the wheel in your hands. The sim's own
// steering_angle is positive-left, so everything shown to the driver is negated.
static float driver_steering_deg(const Agent *ego) {
    return -ego->steering_angle * RAD2DEG;
}

static void draw_cockpit_wheel(const Agent *ego) {
    float hand_wheel_deg = driver_steering_deg(ego) * HUMAN_STEERING_RATIO;
    Vector2 center
        = {(float) g_renderer.screen_width * COCKPIT_WHEEL_CENTER_X_FRACTION,
           (float) g_renderer.screen_height - COCKPIT_WHEEL_BOTTOM_MARGIN_PX};

    int dash_top = g_renderer.screen_height - (int) COCKPIT_DASH_HEIGHT_PX;
    DrawRectangle(0, dash_top, g_renderer.screen_width, (int) COCKPIT_DASH_HEIGHT_PX, COLOR_DASH);
    DrawRectangle(0, dash_top, g_renderer.screen_width, 3, COLOR_DASH_EDGE);

    DrawRing(
        center,
        COCKPIT_WHEEL_RADIUS_PX - COCKPIT_WHEEL_RIM_PX,
        COCKPIT_WHEEL_RADIUS_PX,
        0.0f,
        360.0f,
        64,
        COLOR_WHEEL_RIM);

    for (int spoke = 0; spoke < COCKPIT_WHEEL_SPOKE_COUNT; spoke++) {
        float spoke_deg = hand_wheel_deg + 90.0f + (float) spoke * (360.0f / (float) COCKPIT_WHEEL_SPOKE_COUNT);
        Rectangle bar
            = {center.x, center.y, COCKPIT_WHEEL_RADIUS_PX - COCKPIT_WHEEL_RIM_PX * 0.5f, COCKPIT_WHEEL_SPOKE_WIDTH_PX};
        DrawRectanglePro(bar, (Vector2) {0.0f, COCKPIT_WHEEL_SPOKE_WIDTH_PX * 0.5f}, spoke_deg, COLOR_WHEEL_SPOKE);
    }

    // Index mark at the wheel's twelve o'clock, so rotation past half a turn stays readable.
    float index_rad = (hand_wheel_deg - 90.0f) * DEG2RAD;
    Vector2 index
        = {center.x + cosf(index_rad) * (COCKPIT_WHEEL_RADIUS_PX - COCKPIT_WHEEL_RIM_PX * 0.5f),
           center.y + sinf(index_rad) * (COCKPIT_WHEEL_RADIUS_PX - COCKPIT_WHEEL_RIM_PX * 0.5f)};
    DrawCircleV(index, COCKPIT_WHEEL_INDEX_PX, COLOR_WHEEL_INDEX);
    DrawCircleV(center, COCKPIT_WHEEL_HUB_PX, COLOR_WHEEL_HUB);

    const char *label = TextFormat("%+.0f deg", hand_wheel_deg);
    DrawText(label, (int) center.x + (int) COCKPIT_WHEEL_RADIUS_PX + 18, dash_top - 24, 20, COLOR_HUD);
}

static void draw_hud(Drive *env, const Agent *ego) {
    const int panel_height = 104;
    DrawRectangle(0, g_renderer.screen_height - panel_height, g_renderer.screen_width, panel_height, COLOR_HUD_PANEL);
    int line_y = g_renderer.screen_height - panel_height + 10;

    DrawText(
        TextFormat(
            "%5.1f km/h   gear %s   %5.0f rpm%s",
            ego->sim_speed_signed * 3.6f,
            gear_label(env->human_gear),
            env->human_engine_rpm,
            env->human_engine_stalled ? "   STALLED" : ""),
        16,
        line_y,
        26,
        env->human_engine_stalled ? COLOR_WARN : COLOR_HUD);
    line_y += 32;

    DrawText(
        TextFormat(
            "wheel %+6.1f deg -> road %+5.2f deg   a_long %+5.2f   a_lat %+6.2f   %s",
            driver_steering_deg(ego) * HUMAN_STEERING_RATIO,
            driver_steering_deg(ego),
            ego->accel_long,
            ego->accel_lat,
            env->human_gearbox_mode == GEARBOX_MODE_MANUAL ? "manual" : "auto"),
        16,
        line_y,
        18,
        COLOR_HUD);
    line_y += 24;

    DrawText(
        TextFormat(
            "agent %d   camera %s   %s   %s   [1/2/3] camera  [n] night  [tab] next car  [r] restart",
            env->human_agent_idx,
            camera_mode_name(g_renderer.camera_mode),
            g_renderer.night_mode ? "night" : "day",
            env->substep_mode == SUBSTEP_MODE_ALL ? "substep: all" : "substep: human only"),
        16,
        line_y,
        18,
        COLOR_HUD);

    const char *infraction = NULL;
    if (ego->metrics_array[COLLISION_IDX] > 0.0f) {
        infraction = "COLLISION";
    } else if (ego->metrics_array[OFFROAD_IDX] > 0.0f) {
        infraction = "OFF ROAD";
    } else if (ego->metrics_array[RED_LIGHT_IDX] > 0.0f) {
        infraction = "RED LIGHT";
    } else if (ego->metrics_array[STOP_SIGN_IDX] > 0.0f) {
        infraction = "STOP SIGN";
    }
    if (infraction != NULL) {
        DrawText(
            infraction,
            g_renderer.screen_width - 210,
            g_renderer.screen_height - panel_height + 16,
            28,
            COLOR_WARN);
    }
    DrawFPS(g_renderer.screen_width - 90, 10);
}

static int load_car_models(const char *resource_dir) {
    char path[1024];
    for (int i = 0; i < CAR_MODEL_COUNT; i++) {
        snprintf(path, sizeof(path), "%s/%s", resource_dir, CAR_MODEL_FILES[i]);
        g_renderer.cars[i] = LoadModel(path);
        if (g_renderer.cars[i].meshCount == 0) {
            fprintf(stderr, "[ERROR] -> wheel demo could not load car model: %s\n", path);
            return -1;
        }
        BoundingBox bounds = GetModelBoundingBox(g_renderer.cars[i]);
        g_renderer.car_model_size[i]
            = (Vector3) {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z};
    }
    g_renderer.models_loaded = 1;
    return 0;
}

static int demo_render_init(
    const char *resource_dir,
    int screen_width,
    int screen_height,
    int camera_mode,
    int night_mode) {
    if (g_renderer.window_open) {
        return 0;
    }
    SetTraceLogLevel(LOG_WARNING);
    InitWindow(screen_width, screen_height, "PufferDrive - wheel demo");
    if (!IsWindowReady()) {
        return -1;
    }
    SetTargetFPS(60);
    g_renderer.window_open = 1;
    g_renderer.screen_width = screen_width;
    g_renderer.screen_height = screen_height;
    g_renderer.camera_mode = camera_mode;
    g_renderer.night_mode = night_mode;
    g_renderer.camera.up = (Vector3) {0.0f, 0.0f, 1.0f};
    g_renderer.camera.projection = CAMERA_PERSPECTIVE;
    g_renderer.camera.fovy = CHASE_FOV_DEG;
    if (load_car_models(resource_dir) != 0) {
        CloseWindow();
        g_renderer.window_open = 0;
        return -1;
    }
    return 0;
}

static void demo_render_frame(Drive *env, const float *pose_x, const float *pose_y, const float *pose_heading) {
    if (!g_renderer.window_open || env->human_agent_idx < 0) {
        return;
    }
    Agent *ego = &env->agents[env->human_agent_idx];
    set_camera(ego);

    BeginDrawing();
    ClearBackground(renderer_sky());
    BeginMode3D(g_renderer.camera);
    DrawCube(
        (Vector3) {ego->sim_x, ego->sim_y, ego->sim_z - 0.10f},
        GROUND_SPAN_M,
        GROUND_SPAN_M,
        0.02f,
        renderer_ground());
    draw_road_surface(env, ego->sim_x, ego->sim_y);
    draw_road(env, ego->sim_x, ego->sim_y);
    draw_traffic_controls(env, ego->sim_x, ego->sim_y);
    draw_goals(ego);
    draw_agents(
        env,
        pose_x,
        pose_y,
        pose_heading,
        ego->sim_x,
        ego->sim_y,
        g_renderer.camera_mode == CAMERA_MODE_DRIVER);
    EndMode3D();
    if (g_renderer.camera_mode == CAMERA_MODE_DRIVER) {
        draw_cockpit_wheel(ego);
    }
    draw_hud(env, ego);
    EndDrawing();
}

static void demo_render_set_camera_mode(int camera_mode) {
    if (camera_mode >= 0 && camera_mode < CAMERA_MODE_COUNT) {
        g_renderer.camera_mode = camera_mode;
    }
}

static void demo_render_set_night_mode(int night_mode) {
    g_renderer.night_mode = night_mode ? 1 : 0;
}

static int demo_render_should_close(void) {
    return g_renderer.window_open ? WindowShouldClose() : 1;
}

static void demo_render_close(void) {
    if (!g_renderer.window_open) {
        return;
    }
    if (g_renderer.models_loaded) {
        for (int i = 0; i < CAR_MODEL_COUNT; i++) {
            UnloadModel(g_renderer.cars[i]);
        }
        g_renderer.models_loaded = 0;
    }
    CloseWindow();
    g_renderer.window_open = 0;
}

#endif
