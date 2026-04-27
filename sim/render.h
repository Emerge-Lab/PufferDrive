#ifndef SIM_RENDER_H
#define SIM_RENDER_H

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <math.h>
#include <stdlib.h>

const Color STONE_GRAY = (Color) {80, 80, 80, 255};
const Color PUFF_RED = (Color) {187, 0, 0, 255};
const Color PUFF_CYAN = (Color) {0, 187, 187, 255};
const Color PUFF_WHITE = (Color) {241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color) {6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color) {18, 72, 72, 255};
const Color ROAD_COLOR = (Color) {35, 35, 37, 255};
const Color LANE_CENTER_COLOR = (Color) {25, 170, 220, 255};
const Color ROAD_LINE_WHITE = (Color) {230, 230, 220, 255};
const Color ROAD_LINE_YELLOW = (Color) {230, 190, 40, 255};
const Color CROSSWALK_COLOR = (Color) {245, 245, 230, 255};

// Observation scaling factors
#define OBS_GOAL_SCALE 0.005f
#define OBS_SPEED_SCALE 0.01f
#define OBS_POSITION_SCALE 0.02f
#define OBS_DIST_SQ (100.0f * 100.0f)

struct Client {
    float width;
    float height;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client *make_client() {
    Client *client = (Client *) calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferDrive");
    SetTargetFPS(30);
    Vector3 target_pos = {0, 0, 1};

    client->default_camera_position = (Vector3) {0, 120.0f, 175.0f};
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3) {0.0f, -1.0f, 0.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

void handle_camera_controls(Client *client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta
            = {(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
               -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed};

        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;
        prev_mouse_pos = current_mouse_pos;
    }

    float wheel = GetMouseWheelMove();
    if (wheel == 0) {
        return;
    }

    float zoom_factor = 1.0f - (wheel * 0.1f);
    Vector3 direction
        = {client->camera.position.x - client->camera.target.x,
           client->camera.position.y - client->camera.target.y,
           client->camera.position.z - client->camera.target.z};

    direction.x *= zoom_factor;
    direction.y *= zoom_factor;
    direction.z *= zoom_factor;
    client->camera.position.x = client->camera.target.x + direction.x;
    client->camera.position.y = client->camera.target.y + direction.y;
    client->camera.position.z = client->camera.target.z + direction.z;
}

static inline float render_z(float z) {
    return z + 1.0f;
}

static inline Color agent_type_color(int type) {
    if (type == VEHICLE) {
        return (Color) {30, 140, 230, 255};
    }
    if (type == PEDESTRIAN) {
        return (Color) {235, 215, 60, 255};
    }
    if (type == CYCLIST) {
        return (Color) {45, 205, 120, 255};
    }
    return STONE_GRAY;
}

static inline Color control_state_color(int control_state) {
    if (control_state == CONTROL_STATE_ACTIVE) {
        return PUFF_WHITE;
    }
    if (control_state == CONTROL_STATE_MOVING) {
        return PUFF_CYAN;
    }
    if (control_state == CONTROL_STATE_STATIC) {
        return STONE_GRAY;
    }
    return PUFF_RED;
}

static inline Color traffic_state_color(int state) {
    if (state == TRAFFIC_CONTROL_STATE_RED) {
        return RED;
    }
    if (state == TRAFFIC_CONTROL_STATE_YELLOW) {
        return YELLOW;
    }
    if (state == TRAFFIC_CONTROL_STATE_GREEN) {
        return GREEN;
    }
    if (state == TRAFFIC_CONTROL_STATE_OFF) {
        return STONE_GRAY;
    }
    return PUFF_WHITE;
}

static inline Color road_line_color(int type) {
    if (type == ROAD_LINE_BROKEN_SINGLE_YELLOW || type == ROAD_LINE_BROKEN_DOUBLE_YELLOW
        || type == ROAD_LINE_SOLID_SINGLE_YELLOW || type == ROAD_LINE_SOLID_DOUBLE_YELLOW
        || type == ROAD_LINE_PASSING_DOUBLE_YELLOW) {
        return ROAD_LINE_YELLOW;
    }
    return ROAD_LINE_WHITE;
}

static inline int road_line_is_broken(int type) {
    return type == ROAD_LINE_BROKEN_SINGLE_WHITE || type == ROAD_LINE_BROKEN_SINGLE_YELLOW
        || type == ROAD_LINE_BROKEN_DOUBLE_YELLOW;
}

static inline int current_traffic_state(Drive *env, TrafficControlElement *traffic) {
    if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
        return TRAFFIC_CONTROL_STATE_UNKNOWN;
    }
    if (env->timestep < 0 || env->timestep >= traffic->state_length || traffic->states == NULL) {
        return TRAFFIC_CONTROL_STATE_OFF;
    }
    return traffic->states[env->timestep];
}

void draw_local_box(
    float x,
    float y,
    float z,
    float length,
    float width,
    float height,
    float heading,
    Color fill,
    Color outline) {
    Vector3 size = {length, width, height};
    rlPushMatrix();
    rlTranslatef(x, y, z);
    rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);
    DrawCubeV((Vector3) {0, 0, height * 0.5f}, size, fill);
    DrawCubeWiresV((Vector3) {0, 0, height * 0.5f}, size, outline);
    DrawLine3D((Vector3) {0, 0, height + 0.1f}, (Vector3) {length * 0.5f, 0, height + 0.1f}, outline);
    rlPopMatrix();
}

void draw_agent_box(Agent *agent, Color fill, Color outline) {
    draw_local_box(
        agent->sim_x,
        agent->sim_y,
        render_z(agent->sim_z),
        agent->sim_length,
        agent->sim_width,
        agent->sim_height,
        agent->sim_heading,
        fill,
        outline);
}

void draw_polyline(RoadMapElement *element, Color color, float z_offset, int dashed) {
    for (int j = 0; j < element->segment_length - 1; j++) {
        if (dashed && j % 2 == 1) {
            continue;
        }
        DrawLine3D(
            (Vector3) {element->x[j], element->y[j], render_z(element->z[j]) + z_offset},
            (Vector3) {element->x[j + 1], element->y[j + 1], render_z(element->z[j + 1]) + z_offset},
            color);
    }
}

void draw_crosswalk(RoadMapElement *element) {
    if (element->segment_length < 2) {
        return;
    }

    for (int j = 0; j < element->segment_length; j++) {
        int next = (j + 1) % element->segment_length;
        DrawLine3D(
            (Vector3) {element->x[j], element->y[j], render_z(element->z[j]) + 0.08f},
            (Vector3) {element->x[next], element->y[next], render_z(element->z[next]) + 0.08f},
            CROSSWALK_COLOR);
    }

    if (element->segment_length < 4) {
        return;
    }

    Vector3 a0 = {element->x[0], element->y[0], render_z(element->z[0]) + 0.09f};
    Vector3 a1 = {element->x[1], element->y[1], render_z(element->z[1]) + 0.09f};
    Vector3 b0 = {element->x[3], element->y[3], render_z(element->z[3]) + 0.09f};
    Vector3 b1 = {element->x[2], element->y[2], render_z(element->z[2]) + 0.09f};
    for (int k = 1; k < 8; k += 2) {
        float t = k / 8.0f;
        Vector3 p0 = {a0.x + (a1.x - a0.x) * t, a0.y + (a1.y - a0.y) * t, a0.z + (a1.z - a0.z) * t};
        Vector3 p1 = {b0.x + (b1.x - b0.x) * t, b0.y + (b1.y - b0.y) * t, b0.z + (b1.z - b0.z) * t};
        DrawLine3D(p0, p1, CROSSWALK_COLOR);
    }
}

void draw_traffic_control(Drive *env, TrafficControlElement *traffic) {
    Vector3 a = {traffic->stop_line[0], traffic->stop_line[1], render_z(traffic->stop_line[2]) + 0.15f};
    Vector3 b = {traffic->stop_line[3], traffic->stop_line[4], render_z(traffic->stop_line[5]) + 0.15f};
    int state = current_traffic_state(env, traffic);
    Color color = traffic_state_color(state);
    if (traffic->type == TRAFFIC_CONTROL_TYPE_STOP_SIGN) {
        color = PUFF_RED;
    } else if (traffic->type == TRAFFIC_CONTROL_TYPE_YIELD_SIGN) {
        color = ROAD_LINE_YELLOW;
    }

    DrawLine3D(a, b, color);
    Vector3 mid = {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f, (a.z + b.z) * 0.5f};
    DrawLine3D(mid, (Vector3) {mid.x, mid.y, mid.z + 4.0f}, color);
    if (traffic->type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
        DrawSphere((Vector3) {mid.x, mid.y, mid.z + 4.5f}, 0.45f, color);
    } else {
        DrawCube((Vector3) {mid.x, mid.y, mid.z + 4.0f}, 0.8f, 0.8f, 0.8f, color);
    }

    for (int i = 0; i < traffic->num_controlled_lanes; i++) {
        int lane_idx = traffic->controlled_lanes[i];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            continue;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_length <= 0) {
            continue;
        }
        DrawLine3D(
            mid,
            (Vector3) {lane->x[0], lane->y[0], render_z(lane->z[0]) + 0.2f},
            (Color) {color.r, color.g, color.b, 120});
    }
}

void draw_scene_map(Drive *env) {
    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *element = &env->road_elements[i];
        if (is_road_lane(element->type)) {
            draw_polyline(element, LANE_CENTER_COLOR, 0.03f, false);
        } else if (is_road_line(element->type)) {
            draw_polyline(element, road_line_color(element->type), 0.07f, road_line_is_broken(element->type));
        } else if (element->type == MISC_CROSSWALK) {
            draw_crosswalk(element);
        }
    }

    for (int i = 0; i < env->num_traffic_elements; i++) {
        draw_traffic_control(env, &env->traffic_elements[i]);
    }
}

void draw_agent_obs(Drive *env, int agent_index) {
    float diamond_height = 3.0f;
    float diamond_width = 1.5f;
    float diamond_z = 8.0f;

    Vector3 top = {0, 0, diamond_z + diamond_height / 2};
    Vector3 bot = {0, 0, diamond_z - diamond_height / 2};
    Vector3 fwd = {diamond_width / 2, 0, diamond_z};
    Vector3 bck = {-diamond_width / 2, 0, diamond_z};
    Vector3 lft = {0, diamond_width / 2, diamond_z};
    Vector3 rgt = {0, -diamond_width / 2, diamond_z};

    DrawTriangle3D(top, fwd, rgt, PUFF_CYAN);
    DrawTriangle3D(top, rgt, bck, PUFF_CYAN);
    DrawTriangle3D(top, bck, lft, PUFF_CYAN);
    DrawTriangle3D(top, lft, fwd, PUFF_CYAN);
    DrawTriangle3D(bot, rgt, fwd, PUFF_CYAN);
    DrawTriangle3D(bot, bck, rgt, PUFF_CYAN);
    DrawTriangle3D(bot, lft, bck, PUFF_CYAN);
    DrawTriangle3D(bot, fwd, lft, PUFF_CYAN);

    if (!IsKeyDown(KEY_LEFT_CONTROL) && !IsKeyDown(KEY_RIGHT_CONTROL)) {
        return;
    }

    float vision_radius = sqrtf(OBS_DIST_SQ);
    for (int k = 0; k < 48; k++) {
        float a0 = (2.0f * M_PI * k) / 48.0f;
        float a1 = (2.0f * M_PI * (k + 1)) / 48.0f;
        DrawLine3D(
            (Vector3) {vision_radius * cosf(a0), vision_radius * sinf(a0), 1.1f},
            (Vector3) {vision_radius * cosf(a1), vision_radius * sinf(a1), 1.1f},
            PUFF_BACKGROUND2);
    }

    float *agent_obs = ((float *) env->observations) + (size_t) agent_index * env->obs_size;
    float goal_x = agent_obs[0] / OBS_GOAL_SCALE;
    float goal_y = agent_obs[1] / OBS_GOAL_SCALE;
    DrawSphere((Vector3) {goal_x, goal_y, 1}, 0.5f, GREEN);

    int counts_idx = env->obs_size - OBS_COUNT_FEATURES;
    int lane_count = (int) agent_obs[counts_idx];
    int boundary_count = (int) agent_obs[counts_idx + 1];
    int partner_count = (int) agent_obs[counts_idx + 2];
    if (partner_count > env->obs_slots_partners) {
        partner_count = env->obs_slots_partners;
    }
    if (lane_count > env->obs_slots_lane) {
        lane_count = env->obs_slots_lane;
    }
    if (boundary_count > env->obs_slots_boundary) {
        boundary_count = env->obs_slots_boundary;
    }

    int obs_idx = EGO_FEATURES;
    for (int j = 0; j < partner_count; j++) {
        float x = agent_obs[obs_idx] / OBS_POSITION_SCALE;
        float y = agent_obs[obs_idx + 1] / OBS_POSITION_SCALE;
        float z = 1.0f + agent_obs[obs_idx + 2] * Z_BUFFER;
        float width = fmaxf(agent_obs[obs_idx + 3] * env->norm_vehicle_width_m, 0.4f);
        float length = fmaxf(agent_obs[obs_idx + 4] * env->norm_vehicle_length_m, 0.4f);

        float theta_x = agent_obs[obs_idx + 5];
        float theta_y = agent_obs[obs_idx + 6];
        float angle = atan2f(theta_y, theta_x);
        draw_local_box(x, y, z, length, width, 1.5f, angle, (Color) {255, 140, 0, 130}, ORANGE);

        float arrow_length = 7.5f;
        float ax = x + arrow_length * cosf(angle);
        float ay = y + arrow_length * sinf(angle);
        DrawLine3D((Vector3) {x, y, z + 1.8f}, (Vector3) {ax, ay, z + 1.8f}, PUFF_WHITE);

        float arrow_size = 2.0f;
        float dx = ax - x;
        float dy = ay - y;
        float len = sqrtf(dx * dx + dy * dy);
        if (len > 0) {
            dx /= len;
            dy /= len;
            float px = -dy * arrow_size;
            float py = dx * arrow_size;
            DrawLine3D(
                (Vector3) {ax, ay, z + 1.8f},
                (Vector3) {ax - dx * arrow_size + px, ay - dy * arrow_size + py, z + 1.8f},
                PUFF_WHITE);
            DrawLine3D(
                (Vector3) {ax, ay, z + 1.8f},
                (Vector3) {ax - dx * arrow_size - px, ay - dy * arrow_size - py, z + 1.8f},
                PUFF_WHITE);
        }
        obs_idx += PARTNER_FEATURES;
    }

    int lane_start_idx = EGO_FEATURES + PARTNER_FEATURES * env->obs_slots_partners;
    for (int k = 0; k < lane_count; k++) {
        int idx = lane_start_idx + k * ROAD_FEATURES;
        float x_mid = agent_obs[idx] / OBS_POSITION_SCALE;
        float y_mid = agent_obs[idx + 1] / OBS_POSITION_SCALE;
        float z_mid = 1.0f + agent_obs[idx + 2] * Z_BUFFER;
        float rel_angle = atan2f(agent_obs[idx + 6], agent_obs[idx + 5]);
        float half_len = 0.5f * agent_obs[idx + 3] * env->norm_road_segment_length_m;
        float x_start = x_mid - half_len * cosf(rel_angle);
        float y_start = y_mid - half_len * sinf(rel_angle);
        float x_end = x_mid + half_len * cosf(rel_angle);
        float y_end = y_mid + half_len * sinf(rel_angle);

        DrawCube((Vector3) {x_mid, y_mid, z_mid}, 0.5f, 0.5f, 0.5f, PUFF_CYAN);
        DrawLine3D((Vector3) {x_start, y_start, z_mid}, (Vector3) {x_end, y_end, z_mid}, LANE_CENTER_COLOR);
    }

    int boundary_start_idx = lane_start_idx + env->obs_slots_lane * ROAD_FEATURES;
    for (int k = 0; k < boundary_count; k++) {
        int idx = boundary_start_idx + k * ROAD_FEATURES;
        float x_mid = agent_obs[idx] / OBS_POSITION_SCALE;
        float y_mid = agent_obs[idx + 1] / OBS_POSITION_SCALE;
        float z_mid = 1.0f + agent_obs[idx + 2] * Z_BUFFER;
        float rel_angle = atan2f(agent_obs[idx + 6], agent_obs[idx + 5]);
        float half_len = 0.5f * agent_obs[idx + 3] * env->norm_road_segment_length_m;
        float x_start = x_mid - half_len * cosf(rel_angle);
        float y_start = y_mid - half_len * sinf(rel_angle);
        float x_end = x_mid + half_len * cosf(rel_angle);
        float y_end = y_mid + half_len * sinf(rel_angle);

        DrawCube((Vector3) {x_mid, y_mid, z_mid}, 0.5f, 0.5f, 0.5f, PUFF_CYAN);
        DrawLine3D((Vector3) {x_start, y_start, z_mid}, (Vector3) {x_end, y_end, z_mid}, PUFF_WHITE);
    }
}

void draw_road_edge(float start_x, float start_y, float start_z, float end_x, float end_y, float end_z) {
    Color CURB_TOP = (Color) {220, 220, 220, 255};
    Color CURB_SIDE = (Color) {180, 180, 180, 255};
    Color CURB_BOTTOM = (Color) {160, 160, 160, 255};
    float curb_height = 0.5f;
    float curb_width = 0.3f;

    Vector3 direction = {end_x - start_x, end_y - start_y, 0};
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);
    if (length <= 0) {
        return;
    }
    Vector3 nd = {direction.x / length, direction.y / length, 0};
    Vector3 perp = {-nd.y, nd.x, 0};

    Vector3 b1 = {start_x - perp.x * curb_width / 2, start_y - perp.y * curb_width / 2, render_z(start_z)};
    Vector3 b2 = {start_x + perp.x * curb_width / 2, start_y + perp.y * curb_width / 2, render_z(start_z)};
    Vector3 b3 = {end_x + perp.x * curb_width / 2, end_y + perp.y * curb_width / 2, render_z(end_z)};
    Vector3 b4 = {end_x - perp.x * curb_width / 2, end_y - perp.y * curb_width / 2, render_z(end_z)};

    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void c_render(Drive *env) {
    if (env->client == NULL) {
        env->client = make_client();
    }
    Client *client = env->client;
    BeginDrawing();
    ClearBackground(ROAD_COLOR);
    BeginMode3D(client->camera);
    handle_camera_controls(client);
    bool draw_obs = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);

    DrawLine3D(
        (Vector3) {env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0},
        (Vector3) {env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0},
        PUFF_CYAN);
    DrawLine3D(
        (Vector3) {env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0},
        (Vector3) {env->grid_map->top_left_x, env->grid_map->top_left_y, 0},
        PUFF_CYAN);
    DrawLine3D(
        (Vector3) {env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0},
        (Vector3) {env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0},
        PUFF_CYAN);
    DrawLine3D(
        (Vector3) {env->grid_map->top_left_x, env->grid_map->top_left_y, 0},
        (Vector3) {env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0},
        PUFF_CYAN);

    if (!draw_obs) {
        draw_scene_map(env);
    }

    for (int i = 0; i < env->num_sim_agents; i++) {
        bool is_active_agent = i < env->num_agents;
        if (env->agents[i].removed) {
            continue;
        }
        if (draw_obs && i != EGO_IDX) {
            continue;
        }

        Agent *agent = &env->agents[i];
        Vector3 position = {agent->sim_x, agent->sim_y, render_z(agent->sim_z)};
        float heading = agent->sim_heading;
        Color fill = agent_type_color(agent->type);
        Color outline = control_state_color(agent->control_state);
        if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
            outline = PUFF_RED;
        }

        draw_agent_box(agent, fill, outline);
        if (i == EGO_IDX) {
            DrawSphere((Vector3) {position.x, position.y, position.z + agent->sim_height + 0.5f}, 0.35f, PUFF_WHITE);
        }
        if (i == EGO_IDX) {
            rlPushMatrix();
            rlTranslatef(position.x, position.y, position.z);
            rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);
            draw_agent_obs(env, i);
            rlPopMatrix();
        }

        if (IsKeyDown(KEY_SPACE) && i == EGO_IDX) {
            client->camera.position
                = (Vector3) {position.x - 25.0f * cosf(heading), position.y - 25.0f * sinf(heading), position.z + 15};
            client->camera.target
                = (Vector3) {position.x + 40.0f * cosf(heading), position.y + 40.0f * sinf(heading), position.z - 5.0f};
            client->camera.up = (Vector3) {0, 0, 1};
        }
        if (IsKeyReleased(KEY_SPACE)) {
            client->camera.position = client->default_camera_position;
            client->camera.target = client->default_camera_target;
            client->camera.up = (Vector3) {0, 0, 1};
        }

        if (!is_active_agent || env->agents[i].sim_valid == 0) {
            continue;
        }
        if (!draw_obs) {
            DrawSphere(
                (Vector3) {agent->goal_position_x, agent->goal_position_y, render_z(agent->goal_position_z)},
                0.5f,
                DARKGREEN);
        }
    }

    for (int i = 0; i < env->num_road_elements; i++) {
        if (draw_obs) {
            break;
        }
        if (!is_road_edge(env->road_elements[i].type)) {
            continue;
        }

        for (int j = 0; j < env->road_elements[i].segment_length - 1; j++) {
            draw_road_edge(
                env->road_elements[i].x[j],
                env->road_elements[i].y[j],
                env->road_elements[i].z[j],
                env->road_elements[i].x[j + 1],
                env->road_elements[i].y[j + 1],
                env->road_elements[i].z[j + 1]);
        }
    }

    float grid_start_x = env->grid_map->top_left_x;
    float grid_start_y = env->grid_map->bottom_right_y;
    for (int i = 0; i < env->grid_map->grid_cols; i++) {
        for (int j = 0; j < env->grid_map->grid_rows; j++) {
            float x = grid_start_x + i * GRID_CELL_SIZE;
            float y = grid_start_y + j * GRID_CELL_SIZE;
            DrawCubeWires(
                (Vector3) {x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 1},
                GRID_CELL_SIZE,
                GRID_CELL_SIZE,
                0.1f,
                PUFF_BACKGROUND2);
        }
    }
    EndMode3D();

    DrawText(
        TextFormat(
            "Camera Position: (%.2f, %.2f, %.2f)",
            client->camera.position.x,
            client->camera.position.y,
            client->camera.position.z),
        10,
        10,
        20,
        PUFF_WHITE);
    DrawText(
        TextFormat(
            "Camera Target: (%.2f, %.2f, %.2f)",
            client->camera.target.x,
            client->camera.target.y,
            client->camera.target.z),
        10,
        30,
        20,
        PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", EGO_IDX), 10, 90, 20, PUFF_WHITE);
    DrawText(
        "Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent",
        10,
        client->height - 30,
        20,
        PUFF_WHITE);
    if (env->action_type == DISCRETE) {
        int action = (int) env->actions[EGO_IDX];
        if (env->dynamics_model == CLASSIC) {
            int num_steer = (int) (sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]));
            DrawText(TextFormat("Acceleration: %d", action / num_steer), 10, 110, 20, PUFF_WHITE);
            DrawText(TextFormat("Steering: %d", action % num_steer), 10, 130, 20, PUFF_WHITE);
        } else {
            int num_lat = (int) (sizeof(JERK_LAT) / sizeof(JERK_LAT[0]));
            DrawText(TextFormat("Longitudinal: %d", action / num_lat), 10, 110, 20, PUFF_WHITE);
            DrawText(TextFormat("Lateral: %d", action % num_lat), 10, 130, 20, PUFF_WHITE);
        }
    } else {
        DrawText(TextFormat("Acceleration: %.2f", env->actions[EGO_IDX * 2]), 10, 110, 20, PUFF_WHITE);
        DrawText(TextFormat("Steering: %.2f", env->actions[EGO_IDX * 2 + 1]), 10, 130, 20, PUFF_WHITE);
    }
    DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, 170, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client *client) {
    CloseWindow();
    free(client);
}

#endif
