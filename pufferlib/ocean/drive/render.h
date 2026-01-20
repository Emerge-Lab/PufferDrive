#ifndef RENDER_H
#define RENDER_H

#include <raylib.h>
#include "rlgl.h"

#define MAX_AGENTS 64

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color LIGHTGREEN = (Color){152, 255, 152, 255};

typedef struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    Model cyclist;
    Model pedestrian;
    ModelAnimation *cycle_anim;
    int car_assignments[MAX_AGENTS]; // To keep car model assignments consistent per vehicle
    Vector3 default_camera_position;
    Vector3 default_camera_target;
} Client;

Client *make_client(Drive *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));
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
    client->cyclist = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");
    int animCountCyc = 0;
    client->cycle_anim = LoadModelAnimations("resources/drive/cyclist.glb", &animCountCyc);
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand() % 4) + 1;
    }
    // Get initial target position from first active agent
    Vector3 target_pos = {
        0,
        0, // Y is up
        1  // Z is depth
    };

    // Set up camera to look at target from above and behind
    client->default_camera_position = (Vector3){
        0,      // Same X as target
        120.0f, // 20 units above target
        175.0f  // 20 units behind target
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f}; // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

// Camera control functions
void handle_camera_controls(Client *client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
                         -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed};

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;

        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {client->camera.position.x - client->camera.target.x,
                             client->camera.position.y - client->camera.target.y,
                             client->camera.position.z - client->camera.target.z};

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void draw_agent_obs(Drive *env, int agent_index, int mode, int obs_only, int lasers) {
    // Diamond dimensions
    float diamond_height = 3.0f; // Total height of diamond
    float diamond_width = 1.5f;  // Width of diamond
    float diamond_z = 8.0f;      // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height / 2};    // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height / 2}; // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width / 2, diamond_z};          // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width / 2, diamond_z};          // Back point
    Vector3 left_point = (Vector3){-diamond_width / 2, 0.0f, diamond_z};          // Left point
    Vector3 right_point = (Vector3){diamond_width / 2, 0.0f, diamond_z};          // Right point

    // Draw the diamond faces
    // Top pyramid

    if (mode == 0) {
        DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);  // Front-left face

        // Bottom pyramid
        DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    }
    if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
        return;
    }

    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_obs =
        ego_dim + PARTNER_FEATURES * (MAX_AGENTS_OBSERVATIONS) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    float *agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self = env->agents[active_idx].sim_heading;
    float heading_self_x = cosf(heading_self);
    float heading_self_y = sinf(heading_self);
    float px = env->agents[active_idx].sim_x;
    float py = env->agents[active_idx].sim_y;
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;
    if (mode == 0) {
        DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x, goal_y, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(LIGHTGREEN, 0.3f));
    }

    if (mode == 1) {
        float goal_x_world = px + (goal_x * heading_self_x - goal_y * heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y * heading_self_x);
        DrawSphere((Vector3){goal_x_world, goal_y_world, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(LIGHTGREEN, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = ego_dim; // Start after ego obs
    for (int j = 0; j < MAX_AGENTS - 1; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += 7; // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, 1}, ORANGE);
        }

        float partner_x = px + (x * heading_self_x - y * heading_self_y);
        float partner_y = py + (x * heading_self_y + y * heading_self_x);
        if (lasers && mode == 1) {
            DrawLine3D((Vector3){px, py, 1}, (Vector3){partner_x, partner_y, 1}, ORANGE);
        }

        float half_width = 0.5 * agent_obs[obs_idx + 2] * MAX_VEH_WIDTH;
        float half_len = 0.5 * agent_obs[obs_idx + 3] * MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){x + (half_len * cos_heading - half_width * sin_heading),
                      y + (half_len * sin_heading + half_width * cos_heading), 1},
            (Vector3){x + (half_len * cos_heading + half_width * sin_heading),
                      y + (half_len * sin_heading - half_width * cos_heading), 1},
            (Vector3){x + (-half_len * cos_heading + half_width * sin_heading),
                      y + (-half_len * sin_heading - half_width * cos_heading), 1},
            (Vector3){x + (-half_len * cos_heading - half_width * sin_heading),
                      y + (-half_len * sin_heading + half_width * cos_heading), 1},
        };

        if (mode == 0) {
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], ORANGE);
            }
        }

        if (mode == 1) {
            Vector3 world_corners[4];
            for (int j = 0; j < 4; j++) {
                float lx = corners[j].x;
                float ly = corners[j].y;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = 1;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j + 1) % 4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 2.5f;
        float arrow_x = x + arrow_length * cosf(partner_angle);
        float arrow_y = y + arrow_length * sinf(partner_angle);
        float arrow_x_world;
        float arrow_y_world;
        if (mode == 0) {
            DrawLine3D((Vector3){x, y, 1}, (Vector3){arrow_x, arrow_y, 1}, PUFF_WHITE);
        }
        if (mode == 1) {
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y * heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y * heading_self_x);
            DrawLine3D((Vector3){partner_x, partner_y, 1}, (Vector3){arrow_x_world, arrow_y_world, 1}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 0.3f; // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx * dx + dy * dy);
        if (length > 0) {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector
            float perp_x = -dy * arrow_size;
            float perp_y = dx * arrow_size;

            float arrow_x_end1 = arrow_x - dx * arrow_size + perp_x;
            float arrow_y_end1 = arrow_y - dy * arrow_size + perp_y;
            float arrow_x_end2 = arrow_x - dx * arrow_size - perp_x;
            float arrow_y_end2 = arrow_y - dy * arrow_size - perp_y;

            // Draw the two lines forming the arrow head
            if (mode == 0) {
                DrawLine3D((Vector3){arrow_x, arrow_y, 1}, (Vector3){arrow_x_end1, arrow_y_end1, 1}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x, arrow_y, 1}, (Vector3){arrow_x_end2, arrow_y_end2, 1}, PUFF_WHITE);
            }

            if (mode == 1) {
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1 * heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1 * heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2 * heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2 * heading_self_x);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 1},
                           (Vector3){arrow_x_end1_world, arrow_y_end1_world, 1}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 1},
                           (Vector3){arrow_x_end2_world, arrow_y_end2_world, 1}, PUFF_WHITE);
            }
        }

        obs_idx += 7; // Move to next agent observation (7 values per agent)
    }
    // Then draw map observations
    int map_start_idx = 7 + 7 * (MAX_AGENTS - 1);             // Start after agent observations
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) { // Loop through potential map entities
        int entity_idx = map_start_idx + k * 7;
        if (agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0) {
            continue;
        }
        Color lineColor = BLUE; // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        int unnormalized_type = unnormalize_road_type(entity_type);
        if (!is_road_edge(unnormalized_type))
            continue;

        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 4]);
        float rel_angle_y = (agent_obs[entity_idx + 5]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        }

        if (mode == 1) {
            float x_middle_world = px + (x_middle * heading_self_x - y_middle * heading_self_y);
            float y_middle_world = py + (x_middle * heading_self_y + y_middle * heading_self_x);
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, 1}, (Vector3){x_end_world, y_end_world, 1}, BLUE);
            if (lasers)
                DrawLine3D((Vector3){px, py, 1}, (Vector3){x_middle_world, y_middle_world, 1}, lineColor);
        }
        if (mode == 0) {
            DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
        }
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    // Calculate curb dimensions
    float curb_height = 0.5f; // Height of the curb
    float curb_width = 0.3f;  // Width/thickness of the curb
    float road_z = 0.2f;      // Ensure z-level for roads is below agents

    // Calculate direction vector between start and end
    Vector3 direction = {end_x - start_x, end_y - start_y, 0.0f};

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {direction.x / length, direction.y / length, 0.0f};

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {-normalized_dir.y, normalized_dir.x, 0.0f};

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {start_x - perpendicular.x * curb_width / 2, start_y - perpendicular.y * curb_width / 2, road_z};
    Vector3 b2 = {start_x + perpendicular.x * curb_width / 2, start_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b3 = {end_x + perpendicular.x * curb_width / 2, end_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b4 = {end_x - perpendicular.x * curb_width / 2, end_y - perpendicular.y * curb_width / 2, road_z};

    // Draw the curb faces
    // Bottom face
    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    // Top face (raised by curb_height)
    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    // Side faces
    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void draw_scene(Drive *env, Client *client, int mode, int obs_only, int lasers, int show_grid) {
    // Draw a grid to help with orientation
    // DrawGrid(20, 1.0f);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->top_left_y, 0},
               (Vector3){env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0},
               (Vector3){env->grid_map->top_left_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0},
               (Vector3){env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0},
               (Vector3){env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0}, PUFF_CYAN);

    for (int i = 0; i < env->num_total_agents; i++) {
        Agent *agent = &env->agents[i];
        // Draw objects
        // Check if this vehicle is an active agent
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
        // HIDE CARS ON RESPAWN - IMPORTANT TO KNOW VISUAL SETTING
        if ((!is_active_agent && !is_static_agent) || agent->respawn_timestep != -1) {
            continue;
        }
        Vector3 position;
        float heading;
        position = (Vector3){agent->sim_x, agent->sim_y, 1};
        heading = agent->sim_heading;
        // Create size vector
        Vector3 size = {agent->sim_length, agent->sim_width, agent->sim_height};

        bool is_expert = (!is_active_agent) && (agent->mark_as_expert == 1);

        // Save current transform
        if (mode == 1) {
            float cos_heading = cosf(heading);
            float sin_heading = sinf(heading);

            // Calculate half dimensions
            float half_len = agent->sim_length * 0.5f;
            float half_width = agent->sim_width * 0.5f;

            // Calculate the four corners of the collision box
            Vector3 corners[4] = {
                (Vector3){position.x + (half_len * cos_heading - half_width * sin_heading),
                          position.y + (half_len * sin_heading + half_width * cos_heading), position.z},

                (Vector3){position.x + (half_len * cos_heading + half_width * sin_heading),
                          position.y + (half_len * sin_heading - half_width * cos_heading), position.z},
                (Vector3){position.x + (-half_len * cos_heading + half_width * sin_heading),
                          position.y + (-half_len * sin_heading - half_width * cos_heading), position.z},
                (Vector3){position.x + (-half_len * cos_heading - half_width * sin_heading),
                          position.y + (-half_len * sin_heading + half_width * cos_heading), position.z},

            };

            if (agent_index == env->human_agent_idx && !agent->metrics_array[REACHED_GOAL_IDX]) {
                draw_agent_obs(env, agent_index, mode, obs_only, lasers);
            }
            if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                continue;
            }

            // --- Draw the car  ---
            Color car_color = GRAY; // default for static
            if (is_expert)
                car_color = GOLD; // expert replay
            if (is_active_agent)
                car_color = BLUE; // policy-controlled
            if (is_active_agent && (agent->metrics_array[COLLISION_IDX] > 0 || agent->metrics_array[OFFROAD_IDX] > 0 ||
                                    agent->metrics_array[RED_LIGHT_IDX] > 0))
                car_color = RED;
            rlSetLineWidth(3.0f);
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], car_color);
            }
            // --- Draw a heading arrow pointing forward ---
            Vector3 arrowStart = position;
            Vector3 arrowEnd = {position.x + cos_heading * half_len * 1.5f, // extend arrow beyond car
                                position.y + sin_heading * half_len * 1.5f, position.z};

            DrawLine3D(arrowStart, arrowEnd, car_color);
            DrawSphere(arrowEnd, 0.2f, car_color); // arrow tip

        } else {
            rlPushMatrix();
            // Translate to position, rotate around Y axis, then draw
            rlTranslatef(position.x, position.y, position.z);
            rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f); // Convert radians to degrees
            // Determine color based on status
            Model car_model = client->cars[5];
            if (is_active_agent) {
                car_model = client->cars[client->car_assignments[i % 64]];
            }
            if (is_active_agent && (agent->metrics_array[COLLISION_IDX] > 0 || agent->metrics_array[OFFROAD_IDX] > 0 ||
                                    agent->metrics_array[RED_LIGHT_IDX] > 0)) {
                car_model = client->cars[0]; // Collided agent
            }
            // Draw obs for human selected agent
            if (agent_index == env->human_agent_idx && !agent->metrics_array[REACHED_GOAL_IDX]) {
                draw_agent_obs(env, agent_index, mode, obs_only, lasers);
            }
            // Draw cube for cars static and active
            // Calculate scale factors based on desired size and model dimensions

            BoundingBox bounds = GetModelBoundingBox(car_model);
            Vector3 model_size = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                                  bounds.max.z - bounds.min.z};
            Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};
            if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                rlPopMatrix();
                continue;
            }

            DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
            {
                float cos_heading = cosf(heading);
                float sin_heading = sinf(heading);
                float half_len = agent->sim_length * 0.5f;
                float half_width = agent->sim_width * 0.5f;
                Vector3 corners[4] = {
                    (Vector3){0 + (half_len * cos_heading - half_width * sin_heading),
                              0 + (half_len * sin_heading + half_width * cos_heading), 0},
                    (Vector3){0 + (half_len * cos_heading + half_width * sin_heading),
                              0 + (half_len * sin_heading - half_width * cos_heading), 0},
                    (Vector3){0 + (-half_len * cos_heading + half_width * sin_heading),
                              0 + (-half_len * sin_heading - half_width * cos_heading), 0},
                    (Vector3){0 + (-half_len * cos_heading - half_width * sin_heading),
                              0 + (-half_len * sin_heading + half_width * cos_heading), 0},
                };
                Color wire_color = GRAY; // static
                if (!is_active_agent && agent->mark_as_expert == 1)
                    wire_color = GOLD; // expert replay
                if (is_active_agent)
                    wire_color = BLUE; // policy
                if (is_active_agent &&
                    (agent->metrics_array[COLLISION_IDX] > 0 || agent->metrics_array[OFFROAD_IDX] > 0 ||
                     agent->metrics_array[RED_LIGHT_IDX] > 0))
                    wire_color = RED;
                rlSetLineWidth(2.0f);
                for (int j = 0; j < 4; j++) {
                    DrawLine3D(corners[j], corners[(j + 1) % 4], wire_color);
                }
            }
            rlPopMatrix();
        }

        // FPV Camera Control
        if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index) {
            if (agent->metrics_array[REACHED_GOAL_IDX]) {
                env->human_agent_idx = rand() % env->active_agent_count;
            }
            Vector3 camera_position =
                (Vector3){position.x - (25.0f * cosf(heading)), position.y - (25.0f * sinf(heading)), position.z + 15};

            Vector3 camera_target =
                (Vector3){position.x + 40.0f * cosf(heading), position.y + 40.0f * sinf(heading), position.z - 5.0f};
            client->camera.position = camera_position;
            client->camera.target = camera_target;
            client->camera.up = (Vector3){0, 0, 1};
        }
        if (IsKeyReleased(KEY_SPACE)) {
            client->camera.position = client->default_camera_position;
            client->camera.target = client->default_camera_target;
            client->camera.up = (Vector3){0, 0, 1};
        }
        // Draw goal position for active agents

        if (!is_active_agent || agent->sim_valid == 0) {
            continue;
        }
        if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
            DrawSphere((Vector3){agent->goal_position_x, agent->goal_position_y, 1}, 0.5f, DARKGREEN);

            DrawCircle3D((Vector3){agent->goal_position_x, agent->goal_position_y, 0.1f}, env->goal_radius,
                         (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.3f));
        }
    }
    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *element = &env->road_elements[i];

        for (int j = 0; j < element->segment_length - 1; j++) {
            Vector3 start = {element->x[j], element->y[j], 1};
            Vector3 end = {element->x[j + 1], element->y[j + 1], 1};
            Color lineColor = GRAY;

            if (is_road_lane(element->type))
                lineColor = GRAY;
            else if (is_road_line(element->type))
                lineColor = BLUE;
            else if (is_road_edge(element->type))
                lineColor = WHITE;
            else if (element->type == DRIVEWAY)
                lineColor = RED;
            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                draw_road_edge(env, start.x, start.y, end.x, end.y);
            }
        }
    }
    if (show_grid) {
        // Draw grid cells using the stored bounds
        float grid_start_x = env->grid_map->top_left_x;
        float grid_start_y = env->grid_map->bottom_right_y;
        for (int i = 0; i < env->grid_map->grid_cols; i++) {
            for (int j = 0; j < env->grid_map->grid_rows; j++) {
                float x = grid_start_x + i * GRID_CELL_SIZE;
                float y = grid_start_y + j * GRID_CELL_SIZE;
                DrawCubeWires((Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 1}, GRID_CELL_SIZE,
                              GRID_CELL_SIZE, 0.1f, PUFF_BACKGROUND2);
            }
        }
    }

    EndMode3D();

    // Draw track indices for the tracks to predict
    if (mode == 1 && env->control_mode == CONTROL_WOSAC) {
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;
        float pixels_per_world_unit = client->height / map_height;

        for (int i = 0; i < env->active_agent_count; i++) {
            // Ignore respawned agents
            if (env->agents[i].respawn_timestep != -1) {
                continue;
            }
            int agent_idx = env->active_agent_indices[i];
            int womd_track_idx = env->tracks_to_predict[i];

            float raw_x = -env->agents[agent_idx].sim_x * pixels_per_world_unit;
            float raw_y = env->agents[agent_idx].sim_y * pixels_per_world_unit;

            int screen_x = (int)raw_x + client->width / 2 + 20;
            int screen_y = (int)raw_y + client->height / 2 - 25;

            if (screen_x >= 0 && screen_x <= client->width && screen_y >= 0 && screen_y <= client->height) {
                char text[32];
                snprintf(text, sizeof(text), "%d", womd_track_idx);
                int text_width = MeasureText(text, 20);
                DrawText(text, screen_x - text_width / 2, screen_y, 20, PUFF_WHITE);
            }
        }
    }
}

void saveTopDownImage(Drive *env, Client *client, const char *filename, RenderTexture2D target, int map_height, int obs,
                      int lasers, int trajectories, int frame_count, float *path, int log_trajectories, int show_grid) {
    // Top-down orthographic camera
    Camera3D camera = {0};
    camera.position = (Vector3){0.0f, 0.0f, 500.0f}; // above the scene
    camera.target = (Vector3){0.0f, 0.0f, 0.0f};     // look at origin
    camera.up = (Vector3){0.0f, -1.0f, 0.0f};
    camera.fovy = map_height;
    camera.projection = CAMERA_ORTHOGRAPHIC;
    Color road = (Color){35, 35, 37, 255};

    BeginTextureMode(target);
    ClearBackground(road);
    BeginMode3D(camera);
    rlEnableDepthTest();

    // Draw log trajectories FIRST (in background at lower Z-level)
    if (log_trajectories) {
        for (int i = 0; i < env->num_total_agents; i++) {
            Agent *agent = &env->agents[i];
            for (int j = 0; j < agent->trajectory_length; j++) {
                float x = agent->log_trajectory_x[j];
                float y = agent->log_trajectory_y[j];
                float valid = agent->log_valid[j];
                if (!valid)
                    continue;
                DrawSphere((Vector3){x, y, 0.5f}, 0.3f, Fade(LIGHTGREEN, 0.6f));
            }
        }
    }

    // Draw current path trajectories SECOND (slightly higher than log trajectories)
    if (trajectories) {
        for (int i = 0; i < frame_count; i++) {
            DrawSphere((Vector3){path[i * 2], path[i * 2 + 1], 0.8f}, 0.5f, YELLOW);
        }
    }

    // Draw main scene LAST (on top)
    draw_scene(env, client, 1, obs, lasers, show_grid);

    EndMode3D();
    EndTextureMode();

    // save to file
    Image img = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&img);
    ExportImage(img, filename);
    UnloadImage(img);
}

void saveAgentViewImage(Drive *env, Client *client, const char *filename, RenderTexture2D target, int map_height,
                        int obs_only, int lasers, int show_grid) {
    // Agent perspective camera following the human agent
    int agent_idx = env->active_agent_indices[env->human_agent_idx];
    Agent *agent = &env->agents[agent_idx];

    Camera3D camera = {0};
    // Position camera behind and above the agent
    camera.position = (Vector3){agent->sim_x - (25.0f * cosf(agent->sim_heading)),
                                agent->sim_y - (25.0f * sinf(agent->sim_heading)), 15.0f};
    camera.target = (Vector3){agent->sim_x + 40.0f * cosf(agent->sim_heading),
                              agent->sim_y + 40.0f * sinf(agent->sim_heading), 1.0f};
    camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Color road = (Color){35, 35, 37, 255};

    BeginTextureMode(target);
    ClearBackground(road);
    BeginMode3D(camera);
    rlEnableDepthTest();
    draw_scene(env, client, 0, obs_only, lasers, show_grid); // mode=0 for agent view
    EndMode3D();
    EndTextureMode();

    // Save to file
    Image img = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&img);
    ExportImage(img, filename);
    UnloadImage(img);
}

void c_render(Drive *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client *client = env->client;
    BeginDrawing();
    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(client->camera);
    handle_camera_controls(env->client);
    draw_scene(env, client, 0, 0, 0, 0);
    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)", client->camera.position.x, client->camera.position.y,
                        client->camera.position.z),
             10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)", client->camera.target.x, client->camera.target.y,
                        client->camera.target.z),
             10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);
    // acceleration & steering
    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);
    // Controls help
    DrawText("Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent", 10, client->height - 30, 20,
             PUFF_WHITE);
    // acceleration & steering
    if (env->action_type == 1) { // continuous (float)
        float (*action_array_f)[2] = (float (*)[2])env->actions;
        DrawText(TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]), 10, 110, 20, PUFF_WHITE);
        DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, PUFF_WHITE);
    } else { // discrete (int)
        int (*action_array)[2] = (int (*)[2])env->actions;
        DrawText(TextFormat("Acceleration: %d", action_array[env->human_agent_idx][0]), 10, 110, 20, PUFF_WHITE);
        DrawText(TextFormat("Steering: %d", action_array[env->human_agent_idx][1]), 10, 130, 20, PUFF_WHITE);
    }
    DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, 170, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client *client) {
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}

#endif // RENDER_H
