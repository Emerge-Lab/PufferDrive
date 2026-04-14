#ifndef SIM_RENDER_H
#define SIM_RENDER_H

#include <stdlib.h>
#include <math.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

const Color STONE_GRAY       = (Color){80, 80, 80, 255};
const Color PUFF_RED         = (Color){187, 0, 0, 255};
const Color PUFF_CYAN        = (Color){0, 187, 187, 255};
const Color PUFF_WHITE       = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND  = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color ROAD_COLOR       = (Color){35, 35, 37, 255};

struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    int car_assignments[MAX_AGENTS];
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client* make_client(Drive* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferLib Ray GPU Drive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand_r(&env->rng) % 4) + 1;
    }
    Vector3 target_pos = {0, 0, 1};

    client->default_camera_position = (Vector3){0, 120.0f, 175.0f};
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

void handle_camera_controls(Client* client) {
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
        Vector2 delta = {
            (current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;
        prev_mouse_pos = current_mouse_pos;
    }

    float wheel = GetMouseWheelMove();
    if (wheel == 0) return;

    float zoom_factor = 1.0f - (wheel * 0.1f);
    Vector3 direction = {
        client->camera.position.x - client->camera.target.x,
        client->camera.position.y - client->camera.target.y,
        client->camera.position.z - client->camera.target.z
    };

    direction.x *= zoom_factor;
    direction.y *= zoom_factor;
    direction.z *= zoom_factor;
    client->camera.position.x = client->camera.target.x + direction.x;
    client->camera.position.y = client->camera.target.y + direction.y;
    client->camera.position.z = client->camera.target.z + direction.z;
}

void draw_agent_obs(Drive* env, int agent_index) {
    float diamond_height = 3.0f;
    float diamond_width = 1.5f;
    float diamond_z = 8.0f;

    Vector3 top = {0, 0, diamond_z + diamond_height / 2};
    Vector3 bot = {0, 0, diamond_z - diamond_height / 2};
    Vector3 fwd = {0, diamond_width / 2, diamond_z};
    Vector3 bck = {0, -diamond_width / 2, diamond_z};
    Vector3 lft = {-diamond_width / 2, 0, diamond_z};
    Vector3 rgt = {diamond_width / 2, 0, diamond_z};

    DrawTriangle3D(top, fwd, rgt, PUFF_CYAN);
    DrawTriangle3D(top, rgt, bck, PUFF_CYAN);
    DrawTriangle3D(top, bck, lft, PUFF_CYAN);
    DrawTriangle3D(top, lft, fwd, PUFF_CYAN);
    DrawTriangle3D(bot, rgt, fwd, PUFF_CYAN);
    DrawTriangle3D(bot, bck, rgt, PUFF_CYAN);
    DrawTriangle3D(bot, lft, bck, PUFF_CYAN);
    DrawTriangle3D(bot, fwd, lft, PUFF_CYAN);

    if (!IsKeyDown(KEY_LEFT_CONTROL)) return;

    float (*observations)[OBS_SIZE] = (float(*)[OBS_SIZE])env->observations;
    float* agent_obs = &observations[agent_index][0];
    float goal_x = agent_obs[0] / OBS_GOAL_SCALE;
    float goal_y = agent_obs[1] / OBS_GOAL_SCALE;
    DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, GREEN);

    int obs_idx = EGO_FEATURES;
    for (int j = 0; j < MAX_AGENTS - 1; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += PARTNER_FEATURES;
            continue;
        }

        float x = agent_obs[obs_idx] / OBS_POSITION_SCALE;
        float y = agent_obs[obs_idx + 1] / OBS_POSITION_SCALE;
        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, 1}, ORANGE);

        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float angle = atan2f(theta_y, theta_x);
        float arrow_length = 7.5f;
        float ax = x + arrow_length * cosf(angle);
        float ay = y + arrow_length * sinf(angle);
        DrawLine3D((Vector3){x, y, 1}, (Vector3){ax, ay, 1}, PUFF_WHITE);

        float arrow_size = 2.0f;
        float dx = ax - x;
        float dy = ay - y;
        float len = sqrtf(dx * dx + dy * dy);
        if (len > 0) {
            dx /= len;
            dy /= len;
            float px = -dy * arrow_size;
            float py = dx * arrow_size;
            DrawLine3D((Vector3){ax, ay, 1}, (Vector3){ax - dx * arrow_size + px, ay - dy * arrow_size + py, 1}, PUFF_WHITE);
            DrawLine3D((Vector3){ax, ay, 1}, (Vector3){ax - dx * arrow_size - px, ay - dy * arrow_size - py, 1}, PUFF_WHITE);
        }
        obs_idx += PARTNER_FEATURES;
    }

    int map_start_idx = EGO_FEATURES + PARTNER_FEATURES * (MAX_AGENTS - 1);
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
        int idx = map_start_idx + k * ROAD_FEATURES;
        if (agent_obs[idx] == 0 && agent_obs[idx + 1] == 0) continue;

        int entity_type = (int)agent_obs[idx + 6];
        if (entity_type + ROAD_LANE != ROAD_EDGE) continue;

        float x_mid = agent_obs[idx] / OBS_POSITION_SCALE;
        float y_mid = agent_obs[idx + 1] / OBS_POSITION_SCALE;
        float rel_angle = atan2f(agent_obs[idx + 5], agent_obs[idx + 4]);
        float seg_len = agent_obs[idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        float x_start = x_mid - seg_len * cosf(rel_angle);
        float y_start = y_mid - seg_len * sinf(rel_angle);
        float x_end = x_mid + seg_len * cosf(rel_angle);
        float y_end = y_mid + seg_len * sinf(rel_angle);

        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_mid, y_mid, 1}, PUFF_CYAN);
        DrawCube((Vector3){x_mid, y_mid, 1}, 0.5f, 0.5f, 0.5f, PUFF_CYAN);
        DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
    }
}

void draw_road_edge(Drive* env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};
    Color CURB_SIDE = (Color){180, 180, 180, 255};
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    float curb_height = 0.5f;
    float curb_width = 0.3f;

    Vector3 direction = {end_x - start_x, end_y - start_y, 0};
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);
    Vector3 nd = {direction.x / length, direction.y / length, 0};
    Vector3 perp = {-nd.y, nd.x, 0};

    Vector3 b1 = {start_x - perp.x * curb_width / 2, start_y - perp.y * curb_width / 2, 1.0f};
    Vector3 b2 = {start_x + perp.x * curb_width / 2, start_y + perp.y * curb_width / 2, 1.0f};
    Vector3 b3 = {end_x + perp.x * curb_width / 2, end_y + perp.y * curb_width / 2, 1.0f};
    Vector3 b4 = {end_x - perp.x * curb_width / 2, end_y - perp.y * curb_width / 2, 1.0f};

    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    DrawTriangle3D(b1, t1, b2, CURB_SIDE); DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE); DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE); DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE); DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void c_render(Drive* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;
    BeginDrawing();
    ClearBackground(ROAD_COLOR);
    BeginMode3D(client->camera);
    handle_camera_controls(client);

    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[1], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[0], env->map_corners[3], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[2], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[3], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, PUFF_CYAN);

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN) {
            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            if ((!is_active_agent && !is_static_agent) || env->entities[i].respawn_timestep != -1) {
                continue;
            }

            Vector3 position = {env->entities[i].x, env->entities[i].y, 1};
            float heading = env->entities[i].heading;
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            rlPushMatrix();
            rlTranslatef(position.x, position.y, position.z);
            rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);

            Model car_model = client->cars[5];
            if (is_active_agent) {
                car_model = client->cars[client->car_assignments[i % MAX_AGENTS]];
            }
            if (is_active_agent && env->entities[i].collision_state > NO_COLLISION) {
                car_model = client->cars[0];
            }

            if (agent_index == env->human_agent_idx && !env->entities[agent_index].reached_goal) {
                draw_agent_obs(env, agent_index);
            }

            BoundingBox bounds = GetModelBoundingBox(car_model);
            Vector3 model_size = {
                bounds.max.x - bounds.min.x,
                bounds.max.y - bounds.min.y,
                bounds.max.z - bounds.min.z
            };
            Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};
            DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
            rlPopMatrix();

            float cos_h = env->entities[i].heading_x;
            float sin_h = env->entities[i].heading_y;
            float hl = env->entities[i].length * 0.5f;
            float hw = env->entities[i].width * 0.5f;
            Vector3 corners[4] = {
                {position.x + (hl * cos_h - hw * sin_h), position.y + (hl * sin_h + hw * cos_h), position.z},
                {position.x + (hl * cos_h + hw * sin_h), position.y + (hl * sin_h - hw * cos_h), position.z},
                {position.x + (-hl * cos_h - hw * sin_h), position.y + (-hl * sin_h + hw * cos_h), position.z},
                {position.x + (-hl * cos_h + hw * sin_h), position.y + (-hl * sin_h - hw * cos_h), position.z}
            };
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], PURPLE);
            }

            if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index) {
                if (env->entities[agent_index].reached_goal) {
                    env->human_agent_idx = rand_r(&env->rng) % env->active_agent_count;
                }
                client->camera.position = (Vector3){
                    position.x - 25.0f * cosf(heading),
                    position.y - 25.0f * sinf(heading),
                    position.z + 15
                };
                client->camera.target = (Vector3){
                    position.x + 40.0f * cosf(heading),
                    position.y + 40.0f * sinf(heading),
                    position.z - 5.0f
                };
                client->camera.up = (Vector3){0, 0, 1};
            }
            if (IsKeyReleased(KEY_SPACE)) {
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }

            if (!is_active_agent || env->entities[i].valid == 0) continue;
            if (!IsKeyDown(KEY_LEFT_CONTROL)) {
                DrawSphere((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, 1}, 0.5f, DARKGREEN);
            }
        }

        if (env->entities[i].type < ROAD_LANE || env->entities[i].type > ROAD_EDGE) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            if (env->entities[i].type != ROAD_EDGE) continue;
            if (!IsKeyDown(KEY_LEFT_CONTROL)) {
                draw_road_edge(env,
                    env->entities[i].traj_x[j], env->entities[i].traj_y[j],
                    env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1]);
            }
        }
    }

    float grid_start_x = env->map_corners[0];
    float grid_start_y = env->map_corners[1];
    for (int i = 0; i < env->grid_cols; i++) {
        for (int j = 0; j < env->grid_rows; j++) {
            float x = grid_start_x + i * GRID_CELL_SIZE;
            float y = grid_start_y + j * GRID_CELL_SIZE;
            DrawCubeWires((Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 1},
                GRID_CELL_SIZE, GRID_CELL_SIZE, 0.1f, PUFF_BACKGROUND2);
        }
    }
    EndMode3D();

    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)",
        client->camera.position.x, client->camera.position.y, client->camera.position.z), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)",
        client->camera.target.x, client->camera.target.y, client->camera.target.z), 10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);
    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);
    DrawText("Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent",
        10, client->height - 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Acceleration: %d", env->actions[env->human_agent_idx * 2]), 10, 110, 20, PUFF_WHITE);
    DrawText(TextFormat("Steering: %d", env->actions[env->human_agent_idx * 2 + 1]), 10, 130, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Rows: %d", env->grid_rows), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_cols), 10, 170, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client* client) {
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}

#endif
