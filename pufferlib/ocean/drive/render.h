#ifndef RENDER_H
#define RENDER_H

// View modes for c_render in headless EGL recording. Mirrors 3.0 drive.h
// VIEW_MODE_* enum so callers can pick which camera is used per render.
//   DEFAULT (0)        — preserves the legacy fixed bird's-eye-ish
//                        perspective camera stored on client->camera.
//                        This is what every render before the BEV port did.
//   BEV_AGENT_OBS (1)  — top-down orthographic camera centered on the
//                        ego (env->human_agent_idx) at vision-range zoom.
//                        Matches 3.0's VIEW_MODE_BEV_AGENT_OBS branch.
#define VIEW_MODE_DEFAULT 0
#define VIEW_MODE_BEV_AGENT_OBS 1
#define VIEW_MODE_TOPDOWN_SIM 2

#include "rlgl.h"

#include <raylib.h>

#define MAX_AGENTS 64

const Color STONE_GRAY = (Color) {80, 80, 80, 255};
const Color PUFF_RED = (Color) {187, 0, 0, 255};
const Color PUFF_CYAN = (Color) {0, 187, 187, 255};
const Color PUFF_WHITE = (Color) {241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color) {6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color) {18, 72, 72, 255};
const Color LIGHTGREEN = (Color) {152, 255, 152, 255};

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
    // Headless batch recording state (populated when env->render_mode == RENDER_HEADLESS).
    // recorder_pipefd[1] is the write end of the pipe to the forked ffmpeg child.
    int recorder_pipefd[2];
    pid_t recorder_pid;
    int recorder_active; // 1 once ffmpeg fork + pipe are set up
    // PBO double-buffer for async glReadPixels → CPU DMA. Hides readback
    // latency behind the next frame's draw.
    unsigned int pbo[2];
    int pbo_index;       // which PBO to read INTO this frame (0 or 1)
    int pbo_frame_count; // total frames rendered so far (0 = first frame, nothing to flush)
    // 1 once the EGL GPU context switch has succeeded for this process. Set
    // in make_client; subsequent clients reuse the persistent EGL display.
    int egl_mode;
    // Cached static road geometry. Rebuilt once per Client in build_road_cache
    // (called lazily from c_render in headless mode). Curb triangles are uploaded
    // to a GPU Mesh VBO and drawn with a single DrawMesh call per frame, replacing
    // thousands of per-segment draw_road_edge calls. Lane/road lines are a much
    // smaller set — kept as a flat vertex array and drawn via rlgl.
    Mesh road_tri_mesh;
    Material road_material;
    float *road_line_verts;
    unsigned char *road_line_colors;
    int road_line_count;
    int road_cache_valid;
} Client;

// Precompute static road geometry once per Client. Curbs become a GPU-resident
// Mesh (single DrawMesh call per frame). Lane/line segments become a flat
// vertex array drawn via rlgl. Turns what used to be thousands of per-segment
// per-frame draw calls into a single VBO submission + a tight rlgl loop.
void build_road_cache(Drive *env, Client *client) {
    int tri_count = 0, line_count = 0;
    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *road = &env->road_elements[i];
        int segs = road->segment_length - 1;
        if (segs <= 0) {
            continue;
        }
        if (is_road_edge(road->type)) {
            tri_count += segs * 14; // 7 faces of a rectangular curb, 2 tris each
        } else if (is_road_lane(road->type) || is_road_line(road->type)) {
            line_count += segs;
        }
    }

    int num_tri_verts = tri_count * 3;
    float *tri_verts = (float *) RL_CALLOC(num_tri_verts * 3, sizeof(float));
    unsigned char *tri_colors = (unsigned char *) RL_CALLOC(num_tri_verts * 4, sizeof(unsigned char));
    free(client->road_line_verts);
    free(client->road_line_colors);
    client->road_line_verts = (float *) malloc(line_count * 2 * 3 * sizeof(float));
    client->road_line_colors = (unsigned char *) malloc(line_count * 2 * 4);
    int actual_tri_count = 0;
    client->road_line_count = 0;

#define PUSH_TRI(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, cr, cg, cb, ca)                                          \
    do {                                                                                                               \
        int _ti = actual_tri_count * 9;                                                                                \
        int _ci = actual_tri_count * 12;                                                                               \
        tri_verts[_ti + 0] = vx1;                                                                                      \
        tri_verts[_ti + 1] = vy1;                                                                                      \
        tri_verts[_ti + 2] = vz1;                                                                                      \
        tri_verts[_ti + 3] = vx2;                                                                                      \
        tri_verts[_ti + 4] = vy2;                                                                                      \
        tri_verts[_ti + 5] = vz2;                                                                                      \
        tri_verts[_ti + 6] = vx3;                                                                                      \
        tri_verts[_ti + 7] = vy3;                                                                                      \
        tri_verts[_ti + 8] = vz3;                                                                                      \
        for (int _v = 0; _v < 3; _v++) {                                                                               \
            tri_colors[_ci + _v * 4 + 0] = cr;                                                                         \
            tri_colors[_ci + _v * 4 + 1] = cg;                                                                         \
            tri_colors[_ci + _v * 4 + 2] = cb;                                                                         \
            tri_colors[_ci + _v * 4 + 3] = ca;                                                                         \
        }                                                                                                              \
        actual_tri_count++;                                                                                            \
    } while (0)

#define PUSH_LINE(vx1, vy1, vz1, vx2, vy2, vz2, cr, cg, cb, ca)                                                        \
    do {                                                                                                               \
        int _li = client->road_line_count * 6;                                                                         \
        int _ci = client->road_line_count * 8;                                                                         \
        client->road_line_verts[_li + 0] = vx1;                                                                        \
        client->road_line_verts[_li + 1] = vy1;                                                                        \
        client->road_line_verts[_li + 2] = vz1;                                                                        \
        client->road_line_verts[_li + 3] = vx2;                                                                        \
        client->road_line_verts[_li + 4] = vy2;                                                                        \
        client->road_line_verts[_li + 5] = vz2;                                                                        \
        for (int _v = 0; _v < 2; _v++) {                                                                               \
            client->road_line_colors[_ci + _v * 4 + 0] = cr;                                                           \
            client->road_line_colors[_ci + _v * 4 + 1] = cg;                                                           \
            client->road_line_colors[_ci + _v * 4 + 2] = cb;                                                           \
            client->road_line_colors[_ci + _v * 4 + 3] = ca;                                                           \
        }                                                                                                              \
        client->road_line_count++;                                                                                     \
    } while (0)

    for (int i = 0; i < env->num_road_elements; i++) {
        RoadMapElement *road = &env->road_elements[i];
        for (int j = 0; j < road->segment_length - 1; j++) {
            float sx = road->x[j], sy = road->y[j], sz = road->z[j];
            float ex = road->x[j + 1], ey = road->y[j + 1], ez = road->z[j + 1];

            if (is_road_edge(road->type)) {
                float curb_height = 0.5f, curb_width = 0.3f;
                float dx = ex - sx, dy = ey - sy;
                float len = sqrtf(dx * dx + dy * dy);
                if (len < 1e-6f) {
                    continue;
                }
                float nx = -dy / len, ny = dx / len;
                float hw = curb_width / 2;

                float b1x = sx - nx * hw, b1y = sy - ny * hw;
                float b2x = sx + nx * hw, b2y = sy + ny * hw;
                float b3x = ex + nx * hw, b3y = ey + ny * hw;
                float b4x = ex - nx * hw, b4y = ey - ny * hw;
                float t1z = sz + curb_height;
                float t2z = sz + curb_height;
                float t3z = ez + curb_height;
                float t4z = ez + curb_height;

                PUSH_TRI(b1x, b1y, sz, b2x, b2y, sz, b3x, b3y, ez, 160, 160, 160, 255);
                PUSH_TRI(b1x, b1y, sz, b3x, b3y, ez, b4x, b4y, ez, 160, 160, 160, 255);
                PUSH_TRI(b1x, b1y, t1z, b3x, b3y, t3z, b2x, b2y, t2z, 220, 220, 220, 255);
                PUSH_TRI(b1x, b1y, t1z, b4x, b4y, t4z, b3x, b3y, t3z, 220, 220, 220, 255);
                PUSH_TRI(b1x, b1y, sz, b1x, b1y, t1z, b2x, b2y, sz, 180, 180, 180, 255);
                PUSH_TRI(b1x, b1y, t1z, b2x, b2y, t2z, b2x, b2y, sz, 180, 180, 180, 255);
                PUSH_TRI(b2x, b2y, sz, b2x, b2y, t2z, b3x, b3y, ez, 180, 180, 180, 255);
                PUSH_TRI(b2x, b2y, t2z, b3x, b3y, t3z, b3x, b3y, ez, 180, 180, 180, 255);
                PUSH_TRI(b3x, b3y, ez, b3x, b3y, t3z, b4x, b4y, ez, 180, 180, 180, 255);
                PUSH_TRI(b3x, b3y, t3z, b4x, b4y, t4z, b4x, b4y, ez, 180, 180, 180, 255);
                PUSH_TRI(b4x, b4y, ez, b4x, b4y, t4z, b1x, b1y, sz, 180, 180, 180, 255);
                PUSH_TRI(b4x, b4y, t4z, b1x, b1y, t1z, b1x, b1y, sz, 180, 180, 180, 255);
            } else if (is_road_lane(road->type)) {
                // soft yellow, alpha 64
                PUSH_LINE(sx, sy, sz + 0.01f, ex, ey, ez + 0.01f, 230, 200, 90, 64);
            } else if (is_road_line(road->type)) {
                PUSH_LINE(sx, sy, sz + 0.01f, ex, ey, ez + 0.01f, 255, 255, 255, 255);
            }
        }
    }
#undef PUSH_TRI
#undef PUSH_LINE

    Mesh mesh = {0};
    mesh.vertexCount = actual_tri_count * 3;
    mesh.triangleCount = actual_tri_count;
    mesh.vertices = tri_verts;
    mesh.colors = tri_colors;
    UploadMesh(&mesh, false); // static draw — GPU owns it, CPU arrays kept for UnloadMesh
    client->road_tri_mesh = mesh;
    client->road_material = LoadMaterialDefault();

    client->road_cache_valid = 1;
    fprintf(
        stderr,
        "[drive] Road cache: %d triangles (VBO), %d lines (rlgl)\n",
        actual_tri_count,
        client->road_line_count);
}

// Draw cached road geometry: curbs via the GPU VBO, lane/road lines via rlgl.
static inline void draw_road_cached(Client *client) {
    if (client->road_tri_mesh.vertexCount > 0) {
        DrawMesh(client->road_tri_mesh, client->road_material, MatrixIdentity());
    }
    if (client->road_line_count > 0) {
        rlSetLineWidth(2.0f);
        rlBegin(RL_LINES);
        int nv = client->road_line_count * 2;
        for (int i = 0; i < nv; i++) {
            rlColor4ub(
                client->road_line_colors[i * 4],
                client->road_line_colors[i * 4 + 1],
                client->road_line_colors[i * 4 + 2],
                client->road_line_colors[i * 4 + 3]);
            rlVertex3f(
                client->road_line_verts[i * 3],
                client->road_line_verts[i * 3 + 1],
                client->road_line_verts[i * 3 + 2]);
        }
        rlEnd();
    }
}

// Persistent Xvfb state. Xvfb + GLFW are initialized once per process and
// reused across successive headless Clients. Starting a fresh Xvfb per client
// leaks display sockets; calling glfwTerminate prevents re-init on the same
// thread. Keeping them as statics mirrors 3.0's pattern in drive.h.
static pid_t g_xvfb_pid = 0;
static int g_xvfb_display_num = 0;
// GLFW/Xvfb init is a per-process one-shot. Subsequent make_client
// invocations reuse the existing window + GL context; the EGL pbuffer
// is resized instead. Matches 3.0 drive.h g_glfw_ready pattern.
static int g_glfw_ready = 0;

Client *make_client(Drive *env) {
    Client *client = (Client *) calloc(1, sizeof(Client));
    // Fixed 1920x1080 pbuffer for headless (roughly 3x the pixel area of
    // the old 1280x704). MUST be identical across all envs because the
    // EGL pbuffer is a single global resource shared across every env's
    // client. 3.0's dynamic per-map sizing works there because each map
    // is its own process/env; for our multi-env-per-Drive setup a shared
    // pbuffer with per-env dims would corrupt the frame stream.
    if (env->render_mode == RENDER_HEADLESS) {
        client->width = 1920;
        client->height = 1080;
    } else {
        client->width = 1280;
        client->height = 704;
    }
    if (env->render_mode == RENDER_HEADLESS && !g_glfw_ready) {
        // Headless one-time init: hide window, fork Xvfb if needed, and let
        // InitWindow load glad. Subsequent make_client calls reuse this
        // window + GL context (EGL pbuffer is resized in place).
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
        SetTargetFPS(6000);
        SetTraceLogLevel(LOG_WARNING);

        // Auto-spawn Xvfb when running on a compute node with no X server.
        // Ported from 3.0's drive.h:3633–3672. We only do this when DISPLAY
        // is unset and we haven't already forked an Xvfb for this process.
        if (getenv("DISPLAY") == NULL && g_xvfb_pid == 0) {
            g_xvfb_display_num = 100 + (getpid() % 900);
            char lock_file[32], socket_file[32], display_str[16];
            snprintf(display_str, sizeof(display_str), ":%d", g_xvfb_display_num);
            snprintf(lock_file, sizeof(lock_file), "/tmp/.X%d-lock", g_xvfb_display_num);
            snprintf(socket_file, sizeof(socket_file), "/tmp/.X11-unix/X%d", g_xvfb_display_num);

            // Clean up a stale lockfile from a dead Xvfb on this display number.
            FILE *f = fopen(lock_file, "r");
            if (f) {
                pid_t pid = -1;
                fscanf(f, "%d", &pid);
                fclose(f);
                if (pid > 0 && kill(pid, 0) != 0) {
                    unlink(lock_file);
                    unlink(socket_file);
                }
            }

            g_xvfb_pid = fork();
            if (g_xvfb_pid == 0) {
                close(STDOUT_FILENO);
                close(STDERR_FILENO);
                execlp(
                    "Xvfb",
                    "Xvfb",
                    display_str,
                    "-screen",
                    "0",
                    "1280x720x24",
                    "+extension",
                    "GLX",
                    "-ac",
                    "-noreset",
                    NULL);
                _exit(1);
            }
            setenv("DISPLAY", display_str, 1);
            // Wait up to 2 seconds for the lockfile (Xvfb ready signal).
            for (int i = 0; i < 20 && access(lock_file, F_OK) != 0; i++) {
                usleep(100000);
            }
            usleep(200000);
        }
        InitWindow(client->width, client->height, "PufferLib Ray GPU Drive");
        g_glfw_ready = 1;
    } else if (env->render_mode != RENDER_HEADLESS) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        SetTargetFPS(30);
        InitWindow(client->width, client->height, "PufferLib Ray GPU Drive");
    }
#ifdef DRIVE_HAS_EGL
    // EGL GPU context switch: after InitWindow loads glad/rlgl on the
    // Xvfb/Mesa context, create an NVIDIA GPU context via EGL and switch
    // the thread's active GL context to it. rlgl state (render batches,
    // default textures) has to be re-created on the new context via
    // rlglClose + rlglInit, otherwise subsequent draw calls crash.
    static int egl_ready = 0;
    if (env->render_mode == RENDER_HEADLESS && !egl_ready) {
        if (egl_headless_init((int) client->width, (int) client->height)) {
            if (egl_switch_to_gpu()) {
                // Do NOT call rlglClose here. rlglClose triggers glDelete*
                // on IDs from the previous (GLX/Mesa) context, which forces
                // an XSync on the GLX connection and surfaces queued X errors
                // from the glXMakeCurrent(dpy, 0, NULL) release. 3.0 avoids
                // this by just calling rlglInit on the fresh EGL context.
                rlglInit((int) client->width, (int) client->height);
                rlViewport(0, 0, (int) client->width, (int) client->height);
                rlEnableDepthTest();
                egl_ready = 1;
            }
        }
        if (!egl_ready) {
            fprintf(stderr, "[drive] EGL GPU unavailable, falling back to Xvfb/Mesa software rendering\n");
        }
    } else if (env->render_mode == RENDER_HEADLESS && egl_ready) {
        // Subsequent headless clients reuse the persistent EGL context. The
        // pbuffer may need to grow if this env's map is larger than the one
        // the first client sized for.
        egl_headless_resize((int) client->width, (int) client->height);
        rlViewport(0, 0, (int) client->width, (int) client->height);
    }
    if (egl_ready) {
        client->egl_mode = 1;
    }
#endif
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
    // Get initial target position from first valid agent (active or expert_static).
    // Agents may have large raw world coordinates (e.g. WOMD), so center on them.
    float scene_cx = 0.0f, scene_cy = 0.0f, scene_cz = 0.0f;
    for (int i = 0; i < env->num_total_agents; i++) {
        Agent *a = &env->agents[i];
        if (a->sim_valid && a->sim_x != INVALID_POSITION && a->sim_y != INVALID_POSITION) {
            scene_cx = a->sim_x;
            scene_cy = a->sim_y;
            scene_cz = a->sim_z;
            break;
        }
    }
    // Fallback: use first road element midpoint if no valid agent found
    if (scene_cx == 0.0f && scene_cy == 0.0f && env->num_road_elements > 0) {
        RoadMapElement *r0 = &env->road_elements[0];
        if (r0->segment_length > 0) {
            scene_cx = r0->x[0];
            scene_cy = r0->y[0];
            scene_cz = r0->z[0];
        }
    }

    Vector3 target_pos = {scene_cx, scene_cy, scene_cz};

    // Top-down bird's-eye view: camera directly above target in Z.
    // World convention: X=east, Y=north, Z=altitude.
    // This puts roads and agents in the same X-Y plane as seen from above.
    client->default_camera_position = (Vector3) {scene_cx, scene_cy, scene_cz + 200.0f};
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3) {0.0f, 1.0f, 0.0f}; // +Y (north) is screen-up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;

    // Headless batch recording: fork an ffmpeg child and open a pipe to
    // stream raw RGBA frames into libx264. The pipe write end lives in the
    // parent; the read end is dup'd to stdin in the child. One child per
    // Client; the child runs for the lifetime of the Client and exits when
    // close_client closes the pipe.
    if (env->render_mode == RENDER_HEADLESS) {
        if (pipe(client->recorder_pipefd) == -1) {
            fprintf(stderr, "[drive] pipe() failed: %s\n", strerror(errno));
            return client;
        }

        char size_str[64];
        snprintf(size_str, sizeof(size_str), "%dx%d", (int) client->width, (int) client->height);

        char filename[320];
        const char *stem = env->scenario_id[0] ? env->scenario_id : "pufferdrive";
        if (env->video_suffix[0]) {
            snprintf(filename, sizeof(filename), "%s%s.mp4", stem, env->video_suffix);
        } else {
            snprintf(filename, sizeof(filename), "%s.mp4", stem);
        }

        client->recorder_pid = fork();
        if (client->recorder_pid == -1) {
            fprintf(stderr, "[drive] fork() failed: %s\n", strerror(errno));
            close(client->recorder_pipefd[0]);
            close(client->recorder_pipefd[1]);
            return client;
        }

        if (client->recorder_pid == 0) { // ffmpeg child
            close(client->recorder_pipefd[1]);
            dup2(client->recorder_pipefd[0], STDIN_FILENO);
            close(client->recorder_pipefd[0]);
            for (int fd = 3; fd < 256; fd++) {
                close(fd);
            }
            // -threads 4: cap libx264's internal thread pool. x264 autodetects
            // from the physical node (~96+ cores on H100/H200) and spawns ~24
            // encode threads + 4 lookahead threads. Under a 16-core SLURM
            // cgroup that's ~2x oversubscription that burst-preempts the render
            // producer and causes eval renders to hang / SIGABRT. 4 threads is
            // already far more than needed for ultrafast encoding (~500 fps
            // encode vs <200 fps producer), and leaves cores untouched for the
            // main thread / env / render producer.
            execlp(
                "ffmpeg",
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s",
                size_str,
                "-r",
                "30",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-threads",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                filename,
                NULL);
            fprintf(stderr, "[drive] execlp ffmpeg failed: %s\n", strerror(errno));
            _exit(1);
        }
        close(client->recorder_pipefd[0]);
        fprintf(
            stderr,
            "[drive] ffmpeg forked: pid=%d pipe_write_fd=%d size=%s file=%s egl=%d\n",
            client->recorder_pid,
            client->recorder_pipefd[1],
            size_str,
            filename,
            client->egl_mode);

        // Grow the pipe buffer so one frame fits without blocking the writer.
#ifdef F_SETPIPE_SZ
        // 1920x1080x4 bytes = 8.3 MB per frame; need a pipe buffer big enough
        // for at least one frame so the writev from client_record_frame doesn't
        // block the producer on every frame. 16 MB leaves headroom.
        int pipe_sz = fcntl(client->recorder_pipefd[1], F_SETPIPE_SZ, 16 * 1024 * 1024);
        if (pipe_sz > 0) {
            fprintf(stderr, "[drive] Pipe buffer set to %d bytes\n", pipe_sz);
        }
#endif
        client->recorder_active = 1;
    }

    return client;
}

// Async PBO readback + writev pipe write. Called once per frame AFTER the
// draw batch has been flushed (after EndDrawing / rlDrawRenderBatchActive).
// Double-buffers two PBOs so glReadPixels returns immediately and the
// previous frame's mapped PBO is written to ffmpeg while the current draw
// runs. glReadPixels returns rows bottom-up but ffmpeg expects top-down —
// we row-reverse via a writev iovec instead of memcpy.
static inline void client_record_frame(Client *client) {
    if (!client->recorder_active) {
        return;
    }
#ifdef DRIVE_HAS_EGL
    if (!client->egl_mode) {
        // Xvfb/Mesa fallback: synchronous rlReadScreenPixels + write.
        int w = (int) client->width, h = (int) client->height;
        int frame_bytes = w * h * 4;
        unsigned char *screen_data = rlReadScreenPixels(w, h);
        if (!screen_data) {
            return;
        }
        size_t remaining = (size_t) frame_bytes;
        unsigned char *p = screen_data;
        while (remaining > 0) {
            ssize_t written = write(client->recorder_pipefd[1], p, remaining);
            if (written < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            p += written;
            remaining -= (size_t) written;
        }
        RL_FREE(screen_data);
        client->pbo_frame_count++;
        return;
    }

    int w = (int) client->width, h = (int) client->height;
    int frame_bytes = w * h * 4;

    // Lazily allocate the PBO pair on first frame.
    if (client->pbo[0] == 0) {
        glGenBuffers(2, client->pbo);
        for (int i = 0; i < 2; i++) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[i]);
            glBufferData(GL_PIXEL_PACK_BUFFER, frame_bytes, NULL, GL_STREAM_READ);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    int curr = client->pbo_index;
    int prev = 1 - curr;

    // Kick off async DMA for the current frame's pixels.
    glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[curr]);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // Write the previous frame (map + writev). Skip the very first frame
    // because there's no prior data yet.
    if (client->pbo_frame_count > 0) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[prev]);
        unsigned char *ptr = (unsigned char *) glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
        if (ptr) {
            int row_bytes = w * 4;
            // IOV_MAX on Linux is 1024, so frames taller than 1024 rows need
            // multiple writev calls. Split the top-down flip into chunks of
            // at most IOV_MAX rows. Within each chunk, loop to handle short
            // writes (EINTR / partial pipe buffer) by shrinking the head
            // iovec and retrying.
            int iov_max = 1024;
            int rows_remaining = h;
            int row_top = 0;
            int io_error = 0;
            while (rows_remaining > 0 && !io_error) {
                int chunk = rows_remaining < iov_max ? rows_remaining : iov_max;
                struct iovec iov[1024];
                size_t chunk_bytes = 0;
                for (int i = 0; i < chunk; i++) {
                    int src_row = h - 1 - (row_top + i);
                    iov[i].iov_base = ptr + (size_t) src_row * row_bytes;
                    iov[i].iov_len = row_bytes;
                    chunk_bytes += row_bytes;
                }
                struct iovec *cur = iov;
                int cur_cnt = chunk;
                size_t cur_remaining = chunk_bytes;
                while (cur_remaining > 0) {
                    ssize_t written = writev(client->recorder_pipefd[1], cur, cur_cnt);
                    if (written < 0) {
                        if (errno == EINTR) {
                            continue;
                        }
                        fprintf(
                            stderr,
                            "[drive-pbo] frame=%d writev chunk=%d failed errno=%d(%s)\n",
                            client->pbo_frame_count,
                            cur_cnt,
                            errno,
                            strerror(errno));
                        io_error = 1;
                        break;
                    }
                    cur_remaining -= (size_t) written;
                    size_t consumed = (size_t) written;
                    while (cur_cnt > 0 && consumed >= cur[0].iov_len) {
                        consumed -= cur[0].iov_len;
                        cur++;
                        cur_cnt--;
                    }
                    if (cur_cnt > 0 && consumed > 0) {
                        cur[0].iov_base = (unsigned char *) cur[0].iov_base + consumed;
                        cur[0].iov_len -= consumed;
                    }
                }
                row_top += chunk;
                rows_remaining -= chunk;
            }
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        } else {
            fprintf(
                stderr,
                "[drive-pbo] frame=%d glMapBuffer returned NULL! GL error=0x%x\n",
                client->pbo_frame_count,
                glGetError());
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    client->pbo_index = prev;
    client->pbo_frame_count++;
#else
    (void) client;
#endif
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
        Vector2 delta
            = {(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
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
        Vector3 direction
            = {client->camera.position.x - client->camera.target.x,
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
    Vector3 top_point = (Vector3) {0.0f, 0.0f, diamond_z + diamond_height / 2};    // Top point
    Vector3 bottom_point = (Vector3) {0.0f, 0.0f, diamond_z - diamond_height / 2}; // Bottom point
    Vector3 front_point = (Vector3) {0.0f, diamond_width / 2, diamond_z};          // Front point
    Vector3 back_point = (Vector3) {0.0f, -diamond_width / 2, diamond_z};          // Back point
    Vector3 left_point = (Vector3) {-diamond_width / 2, 0.0f, diamond_z};          // Left point
    Vector3 right_point = (Vector3) {diamond_width / 2, 0.0f, diamond_z};          // Right point

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

    int ego_dim = EGO_FEATURES;
    int num_reward_coefs = env->reward_conditioning ? NUM_REWARD_COEFS : 0;
    int target_features = (env->target_type == TARGET_STATIC) ? env->num_target_waypoints * STATIC_TARGET_FEATURES
                                                              : env->num_target_waypoints * DYNAMIC_TARGET_FEATURES;
    int max_obs = compute_observation_size(env);
    float (*observations)[max_obs] = (float (*)[max_obs]) env->observations;
    float *agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self = env->agents[active_idx].sim_heading;
    float heading_self_x = cosf(heading_self);
    float heading_self_y = sinf(heading_self);
    float px = env->agents[active_idx].sim_x;
    float py = env->agents[active_idx].sim_y;
    float pz = env->agents[active_idx].sim_z;
    // draw goal (first target waypoint, in ego frame)
    if (env->num_target_waypoints > 0) {
        int goal_obs_idx = ego_dim + num_reward_coefs;
        float goal_x = agent_obs[goal_obs_idx] * env->obs_norm_goal_offset_m;
        float goal_y = agent_obs[goal_obs_idx + 1] * env->obs_norm_goal_offset_m;
        if (mode == 0) {
            DrawSphere((Vector3) {goal_x, goal_y, 1}, 0.5f, LIGHTGREEN);
            DrawCircle3D(
                (Vector3) {goal_x, goal_y, 0.1f},
                env->goal_radius,
                (Vector3) {0, 0, 1},
                90.0f,
                Fade(LIGHTGREEN, 0.3f));
        }

        if (mode == 1) {
            float goal_x_world = px + (goal_x * heading_self_x - goal_y * heading_self_y);
            float goal_y_world = py + (goal_x * heading_self_y + goal_y * heading_self_x);
            DrawSphere((Vector3) {goal_x_world, goal_y_world, 1}, 0.5f, LIGHTGREEN);
            DrawCircle3D(
                (Vector3) {goal_x_world, goal_y_world, 0.1f},
                env->goal_radius,
                (Vector3) {0, 0, 1},
                90.0f,
                Fade(LIGHTGREEN, 0.3f));
        }
    }
    // First draw other agent observations
    int obs_idx = ego_dim + num_reward_coefs + target_features; // Start after ego, conditioning, and target obs
    for (int j = 0; j < env->obs_slots_partners; j++) {
        bool is_empty = true;
        for (int k = 0; k < PARTNER_FEATURES; k++) {
            if (agent_obs[obs_idx + k] != 0.0f) {
                is_empty = false;
                break;
            }
        }
        if (is_empty) {
            obs_idx += PARTNER_FEATURES;
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * env->obs_norm_xy_offset_m;
        float y = agent_obs[obs_idx + 1] * env->obs_norm_xy_offset_m;
        float z = agent_obs[obs_idx + 2] * env->obs_norm_xy_offset_m;
        if (lasers && mode == 0) {
            DrawLine3D((Vector3) {0, 0, 0}, (Vector3) {x, y, z + 1}, ORANGE);
        }

        float partner_x = px + (x * heading_self_x - y * heading_self_y);
        float partner_y = py + (x * heading_self_y + y * heading_self_x);
        float partner_z = pz + z;
        if (lasers && mode == 1) {
            DrawLine3D((Vector3) {px, py, pz + 1}, (Vector3) {partner_x, partner_y, partner_z + 1}, ORANGE);
        }

        float half_len = 0.5f * agent_obs[obs_idx + 3] * env->obs_norm_veh_length_m;
        float half_width = 0.5f * agent_obs[obs_idx + 4] * env->obs_norm_veh_width_m;
        float theta_x = agent_obs[obs_idx + 5];
        float theta_y = agent_obs[obs_idx + 6];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3) {x + (half_len * cos_heading - half_width * sin_heading),
                       y + (half_len * sin_heading + half_width * cos_heading),
                       z + 1},
            (Vector3) {x + (half_len * cos_heading + half_width * sin_heading),
                       y + (half_len * sin_heading - half_width * cos_heading),
                       z + 1},
            (Vector3) {x + (-half_len * cos_heading + half_width * sin_heading),
                       y + (-half_len * sin_heading - half_width * cos_heading),
                       z + 1},
            (Vector3) {x + (-half_len * cos_heading - half_width * sin_heading),
                       y + (-half_len * sin_heading + half_width * cos_heading),
                       z + 1},
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
                world_corners[j].z = pz + z + 1;
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
            DrawLine3D((Vector3) {x, y, z + 1}, (Vector3) {arrow_x, arrow_y, z + 1}, PUFF_WHITE);
        }
        if (mode == 1) {
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y * heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y * heading_self_x);
            DrawLine3D(
                (Vector3) {partner_x, partner_y, partner_z + 1},
                (Vector3) {arrow_x_world, arrow_y_world, partner_z + 1},
                PUFF_WHITE);
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
                DrawLine3D(
                    (Vector3) {arrow_x, arrow_y, z + 1},
                    (Vector3) {arrow_x_end1, arrow_y_end1, z + 1},
                    PUFF_WHITE);
                DrawLine3D(
                    (Vector3) {arrow_x, arrow_y, z + 1},
                    (Vector3) {arrow_x_end2, arrow_y_end2, z + 1},
                    PUFF_WHITE);
            }

            if (mode == 1) {
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1 * heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1 * heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2 * heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2 * heading_self_x);
                DrawLine3D(
                    (Vector3) {arrow_x_world, arrow_y_world, partner_z + 1},
                    (Vector3) {arrow_x_end1_world, arrow_y_end1_world, partner_z + 1},
                    PUFF_WHITE);
                DrawLine3D(
                    (Vector3) {arrow_x_world, arrow_y_world, partner_z + 1},
                    (Vector3) {arrow_x_end2_world, arrow_y_end2_world, partner_z + 1},
                    PUFF_WHITE);
            }
        }

        obs_idx += PARTNER_FEATURES;
    }
    // Then draw lane segment observations (obs_idx is now at lane obs start after partner loop)
    int lane_obs_start = obs_idx;
    for (int k = 0; k < env->obs_slots_lane_kept; k++) {
        int entity_idx = lane_obs_start + k * ROAD_FEATURES;
        bool is_empty = true;
        for (int j = 0; j < ROAD_FEATURES; j++) {
            if (agent_obs[entity_idx + j] != 0.0f) {
                is_empty = false;
                break;
            }
        }
        if (is_empty) {
            continue;
        }
        Color lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * env->obs_norm_xy_offset_m;
        float y_middle = agent_obs[entity_idx + 1] * env->obs_norm_xy_offset_m;
        float z_middle = agent_obs[entity_idx + 2] * env->obs_norm_xy_offset_m;
        float rel_angle_x = (agent_obs[entity_idx + 5]);
        float rel_angle_y = (agent_obs[entity_idx + 6]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 3] * env->obs_norm_road_seg_length_m;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (lasers && mode == 0) {
            DrawLine3D((Vector3) {0, 0, 0}, (Vector3) {x_middle, y_middle, z_middle + 1}, lineColor);
        }

        if (mode == 1) {
            float x_middle_world = px + (x_middle * heading_self_x - y_middle * heading_self_y);
            float y_middle_world = py + (x_middle * heading_self_y + y_middle * heading_self_x);
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            float z_world = pz + z_middle + 1;
            DrawCube((Vector3) {x_middle_world, y_middle_world, z_world}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D(
                (Vector3) {x_start_world, y_start_world, z_world},
                (Vector3) {x_end_world, y_end_world, z_world},
                BLUE);
            if (lasers) {
                DrawLine3D((Vector3) {px, py, pz + 1}, (Vector3) {x_middle_world, y_middle_world, z_world}, lineColor);
            }
        }
        if (mode == 0) {
            DrawCube((Vector3) {x_middle, y_middle, z_middle + 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3) {x_start, y_start, z_middle + 1}, (Vector3) {x_end, y_end, z_middle + 1}, BLUE);
        }
    }
    // Draw boundary/edge segment observations in red (immediately after lane obs)
    int boundary_obs_start = lane_obs_start + env->obs_slots_lane_kept * ROAD_FEATURES;
    for (int k = 0; k < env->obs_slots_boundary_kept; k++) {
        int entity_idx = boundary_obs_start + k * ROAD_FEATURES;
        bool is_empty = true;
        for (int j = 0; j < ROAD_FEATURES; j++) {
            if (agent_obs[entity_idx + j] != 0.0f) {
                is_empty = false;
                break;
            }
        }
        if (is_empty) {
            continue;
        }
        float x_middle = agent_obs[entity_idx] * env->obs_norm_xy_offset_m;
        float y_middle = agent_obs[entity_idx + 1] * env->obs_norm_xy_offset_m;
        float z_middle = agent_obs[entity_idx + 2] * env->obs_norm_xy_offset_m;
        float rel_angle_x = agent_obs[entity_idx + 5];
        float rel_angle_y = agent_obs[entity_idx + 6];
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 3] * env->obs_norm_road_seg_length_m;
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (mode == 1) {
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            float z_world = pz + z_middle + 1;
            DrawLine3D(
                (Vector3) {x_start_world, y_start_world, z_world},
                (Vector3) {x_end_world, y_end_world, z_world},
                RED);
        }
        if (mode == 0) {
            DrawLine3D((Vector3) {x_start, y_start, z_middle + 1}, (Vector3) {x_end, y_end, z_middle + 1}, RED);
        }
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y, Color lineColor) {
    Color CURB_TOP = (Color) {220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color) {180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color) {160, 160, 160, 255};
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
    DrawLine3D(
        (Vector3) {env->grid_map->top_left_x, env->grid_map->top_left_y, 0},
        (Vector3) {env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0},
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
        (Vector3) {env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0},
        (Vector3) {env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0},
        PUFF_CYAN);

    // ==== Traffic lights ====================================================
    // For each TRAFFIC_LIGHT element, query states[env->timestep] and draw the
    // stop-line segment + a marker cube at its midpoint colored by current
    // state. Drawn BEFORE the agent loop so cars can occlude overlapping markers.
    // Non-TRAFFIC_LIGHT types (stop signs, yield signs) don't have a per-step
    // dynamic state, so we leave them out for now (they're a follow-up).
    for (int tl_i = 0; tl_i < env->num_traffic_elements; tl_i++) {
        TrafficControlElement *traffic = &env->traffic_elements[tl_i];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (traffic->states == NULL || traffic->state_length <= 0) {
            continue;
        }

        int state_idx = env->timestep;
        if (state_idx < 0) {
            state_idx = 0;
        }
        if (state_idx >= traffic->state_length) {
            state_idx = traffic->state_length - 1;
        }
        int tl_state = traffic->states[state_idx];

        Color tl_color;
        switch (tl_state) {
        case TRAFFIC_CONTROL_STATE_RED:
            tl_color = RED;
            break;
        case TRAFFIC_CONTROL_STATE_YELLOW:
            tl_color = YELLOW;
            break;
        case TRAFFIC_CONTROL_STATE_GREEN:
            tl_color = GREEN;
            break;
        default:
            tl_color = GRAY;
            break; // OFF / UNKNOWN
        }

        // Lift the stop-line endpoints a little above the road surface so the
        // line and marker render on top of the road tri mesh instead of
        // z-fighting with it.
        Vector3 sl_p1 = (Vector3) {traffic->stop_line[0], traffic->stop_line[1], traffic->stop_line[2] + 0.3f};
        Vector3 sl_p2 = (Vector3) {traffic->stop_line[3], traffic->stop_line[4], traffic->stop_line[5] + 0.3f};
        Vector3 sl_mid = (Vector3) {
            (traffic->stop_line[0] + traffic->stop_line[3]) * 0.5f,
            (traffic->stop_line[1] + traffic->stop_line[4]) * 0.5f,
            (traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f + 0.5f,
        };

        DrawLine3D(sl_p1, sl_p2, tl_color);
        // Marker cube at the midpoint. 2 x 2 x 1 m is large enough to read in a
        // ~210 m BEV viewport (~1% of view height) and small enough to not
        // dominate the perspective sim_state camera.
        DrawCube(sl_mid, 2.0f, 2.0f, 1.0f, tl_color);
    }

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
        if (!is_active_agent && !is_static_agent) {
            continue;
        }
        Vector3 position;
        float heading;
        // Use sim_z so cars on elevated map sections (hills, overpasses)
        // are drawn at their actual world z, matching 3.0's draw_scene.
        // With a hardcoded z=1 the car's world pos diverges from the road
        // surface (which comes from road->x/y/z in build_road_cache), so
        // on maps with elevation the car appears off the road surface.
        position = (Vector3) {agent->sim_x, agent->sim_y, agent->sim_z};
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
                (Vector3) {position.x + (half_len * cos_heading - half_width * sin_heading),
                           position.y + (half_len * sin_heading + half_width * cos_heading),
                           position.z},

                (Vector3) {position.x + (half_len * cos_heading + half_width * sin_heading),
                           position.y + (half_len * sin_heading - half_width * cos_heading),
                           position.z},
                (Vector3) {position.x + (-half_len * cos_heading + half_width * sin_heading),
                           position.y + (-half_len * sin_heading - half_width * cos_heading),
                           position.z},
                (Vector3) {position.x + (-half_len * cos_heading - half_width * sin_heading),
                           position.y + (-half_len * sin_heading + half_width * cos_heading),
                           position.z},

            };

            if (agent_index == env->human_agent_idx) {
                draw_agent_obs(env, agent_index, mode, obs_only, lasers);
            }
            if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                continue;
            }

            // --- Draw the car  ---
            Color car_color = GRAY; // default for static
            if (is_expert) {
                car_color = GOLD; // expert replay
            }
            if (is_active_agent) {
                car_color = GREEN; // policy-controlled SDC
            }
            if (is_active_agent
                && (agent->metrics_array[COLLISION_IDX] > 0 || agent->metrics_array[OFFROAD_IDX] > 0
                    || agent->metrics_array[RED_LIGHT_IDX] > 0)) {
                car_color = RED;
            }
            if (is_active_agent && car_color.r == GREEN.r && car_color.g == GREEN.g) {
                // Filled quad for the SDC
                DrawTriangle3D(corners[0], corners[1], corners[2], Fade(GREEN, 0.7f));
                DrawTriangle3D(corners[0], corners[2], corners[3], Fade(GREEN, 0.7f));
            }
            rlSetLineWidth(3.0f);
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], car_color);
            }
            // --- Draw a heading arrow pointing forward ---
            Vector3 arrowStart = position;
            Vector3 arrowEnd
                = {position.x + cos_heading * half_len * 1.5f, // extend arrow beyond car
                   position.y + sin_heading * half_len * 1.5f,
                   position.z};

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
            if (is_active_agent
                && (agent->metrics_array[COLLISION_IDX] > 0 || agent->metrics_array[OFFROAD_IDX] > 0
                    || agent->metrics_array[RED_LIGHT_IDX] > 0)) {
                car_model = client->cars[0]; // Collided agent
            }
            // Draw obs for human selected agent
            if (agent_index == env->human_agent_idx) {
                draw_agent_obs(env, agent_index, mode, obs_only, lasers);
            }
            // Draw cube for cars static and active
            // Calculate scale factors based on desired size and model dimensions

            BoundingBox bounds = GetModelBoundingBox(car_model);
            Vector3 model_size
                = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z};
            Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};
            if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                rlPopMatrix();
                continue;
            }

            DrawModelEx(car_model, (Vector3) {0, 0, 0}, (Vector3) {1, 0, 0}, 90.0f, scale, WHITE);
            {
                // Corners in LOCAL space: the enclosing rlPushMatrix +
                // rlTranslatef(position) + rlRotatef(heading) block already
                // rotates and translates everything drawn after it, so we
                // must NOT pre-rotate the corners here. The previous code
                // applied cos_heading/sin_heading to the corners AND let the
                // matrix rotate them again -- double rotation, boxes appeared
                // visibly offset from the car models. Matches 3.0 drive.h.
                float half_len = agent->sim_length * 0.5f;
                float half_width = agent->sim_width * 0.5f;
                Vector3 corners[4] = {
                    (Vector3) {half_len, -half_width, 0},  // front-left
                    (Vector3) {half_len, half_width, 0},   // front-right
                    (Vector3) {-half_len, half_width, 0},  // rear-right
                    (Vector3) {-half_len, -half_width, 0}, // rear-left
                };
                Color wire_color = GRAY; // static
                if (!is_active_agent && agent->mark_as_expert == 1) {
                    wire_color = GOLD; // expert replay
                }
                if (is_active_agent) {
                    wire_color = GREEN; // policy-controlled SDC
                }
                if (is_active_agent
                    && (agent->metrics_array[COLLISION_IDX] > 0 || agent->metrics_array[OFFROAD_IDX] > 0
                        || agent->metrics_array[RED_LIGHT_IDX] > 0)) {
                    wire_color = RED;
                }
                if (is_active_agent && wire_color.r == GREEN.r && wire_color.g == GREEN.g) {
                    // Filled quad for the SDC
                    rlBegin(RL_TRIANGLES);
                    rlColor4ub(0, 228, 48, 180);
                    rlVertex3f(corners[0].x, corners[0].y, corners[0].z);
                    rlVertex3f(corners[1].x, corners[1].y, corners[1].z);
                    rlVertex3f(corners[2].x, corners[2].y, corners[2].z);
                    rlVertex3f(corners[0].x, corners[0].y, corners[0].z);
                    rlVertex3f(corners[2].x, corners[2].y, corners[2].z);
                    rlVertex3f(corners[3].x, corners[3].y, corners[3].z);
                    rlEnd();
                }
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
            Vector3 camera_position = (Vector3) {position.x - (25.0f * cosf(heading)),
                                                 position.y - (25.0f * sinf(heading)),
                                                 position.z + 15};

            Vector3 camera_target
                = (Vector3) {position.x + 40.0f * cosf(heading), position.y + 40.0f * sinf(heading), position.z - 5.0f};
            client->camera.position = camera_position;
            client->camera.target = camera_target;
            client->camera.up = (Vector3) {0, 0, 1};
        }
        if (IsKeyReleased(KEY_SPACE)) {
            client->camera.position = client->default_camera_position;
            client->camera.target = client->default_camera_target;
            client->camera.up = (Vector3) {0, 0, 1};
        }
        // Draw goal position for active agents

        if (!is_active_agent || agent->sim_valid == 0) {
            continue;
        }
        if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
            // Draw all target waypoints: brightest (first) to darkest (last)
            int num_wp = env->num_target_waypoints;
            if (num_wp > MAX_TARGET_WAYPOINTS) {
                num_wp = MAX_TARGET_WAYPOINTS;
            }
            for (int wp = 0; wp < num_wp; wp++) {
                if (wp < agent->current_goal_idx) {
                    continue; // already reached
                }
                float wx = agent->goal_positions_x[wp];
                float wy = agent->goal_positions_y[wp];
                float wz = agent->goal_positions_z[wp];
                // Brightness: first=1.0, last=0.3
                float alpha = 1.0f - 0.7f * (float) wp / (float) (num_wp > 1 ? num_wp - 1 : 1);
                float radius = 1.5f - 0.5f * (float) wp / (float) (num_wp > 1 ? num_wp - 1 : 1);
                Color wp_color = Fade(LIME, alpha);
                DrawSphere((Vector3) {wx, wy, wz + 1.0f}, radius, wp_color);
                DrawCircle3D(
                    (Vector3) {wx, wy, wz + 0.1f},
                    env->goal_radius,
                    (Vector3) {0, 0, 1},
                    90.0f,
                    Fade(LIME, alpha * 0.3f));
            }
        }
    }
    // Per-frame road geometry — skipped entirely when the static road cache is
    // valid (headless batch path draws it from a VBO via draw_road_cached).
    if (!client->road_cache_valid) {
        if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
            rlSetLineWidth(1.5f);
            for (int i = 0; i < env->num_road_elements; i++) {
                RoadMapElement *element = &env->road_elements[i];
                for (int j = 0; j < element->segment_length - 1; j++) {
                    float sx = element->x[j], sy = element->y[j], sz = element->z[j];
                    float ex = element->x[j + 1], ey = element->y[j + 1], ez = element->z[j + 1];
                    if (is_road_lane(element->type)) {
                        // Lane centerline: semi-transparent yellow line (matches headless cache)
                        DrawLine3D(
                            (Vector3) {sx, sy, sz + 0.05f},
                            (Vector3) {ex, ey, ez + 0.05f},
                            (Color) {230, 200, 90, 100});
                    } else if (is_road_line(element->type)) {
                        // Road marking: solid white line
                        DrawLine3D((Vector3) {sx, sy, sz + 0.05f}, (Vector3) {ex, ey, ez + 0.05f}, WHITE);
                    } else if (is_road_edge(element->type)) {
                        // Road edge / curb: flat white line at same level as road markings
                        DrawLine3D(
                            (Vector3) {sx, sy, sz + 0.05f},
                            (Vector3) {ex, ey, ez + 0.05f},
                            (Color) {255, 255, 255, 200});
                    } else if (is_misc_road(element->type)) {
                        DrawLine3D(
                            (Vector3) {sx, sy, sz + 0.05f},
                            (Vector3) {ex, ey, ez + 0.05f},
                            (Color) {255, 100, 100, 180});
                    }
                }
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
                DrawCubeWires(
                    (Vector3) {x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 1},
                    GRID_CELL_SIZE,
                    GRID_CELL_SIZE,
                    0.1f,
                    PUFF_BACKGROUND2);
            }
        }
    }

    EndMode3D();

    // Draw track indices for the tracks to predict
    if (mode == 1 && env->control_mode == CONTROL_WOSAC) {
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;
        float pixels_per_world_unit = client->height / map_height;

        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int womd_track_idx = env->tracks_to_predict[i];

            float raw_x = -env->agents[agent_idx].sim_x * pixels_per_world_unit;
            float raw_y = env->agents[agent_idx].sim_y * pixels_per_world_unit;

            int screen_x = (int) raw_x + client->width / 2 + 20;
            int screen_y = (int) raw_y + client->height / 2 - 25;

            if (screen_x >= 0 && screen_x <= client->width && screen_y >= 0 && screen_y <= client->height) {
                char text[32];
                snprintf(text, sizeof(text), "%d", womd_track_idx);
                int text_width = MeasureText(text, 20);
                DrawText(text, screen_x - text_width / 2, screen_y, 20, PUFF_WHITE);
            }
        }
    }
}

void saveTopDownImage(
    Drive *env,
    Client *client,
    const char *filename,
    RenderTexture2D target,
    int map_height,
    int obs,
    int lasers,
    int trajectories,
    int frame_count,
    float *path,
    int log_trajectories,
    int show_grid) {
    // Top-down orthographic camera
    Camera3D camera = {0};
    camera.position = (Vector3) {0.0f, 0.0f, 500.0f}; // above the scene
    camera.target = (Vector3) {0.0f, 0.0f, 0.0f};     // look at origin
    camera.up = (Vector3) {0.0f, -1.0f, 0.0f};
    camera.fovy = map_height;
    camera.projection = CAMERA_ORTHOGRAPHIC;
    Color road = (Color) {35, 35, 37, 255};

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
                if (!valid) {
                    continue;
                }
                DrawSphere((Vector3) {x, y, 0.5f}, 0.3f, Fade(LIGHTGREEN, 0.6f));
            }
        }
    }

    // Draw current path trajectories SECOND (slightly higher than log trajectories)
    if (trajectories) {
        for (int i = 0; i < frame_count; i++) {
            DrawSphere((Vector3) {path[i * 2], path[i * 2 + 1], 0.8f}, 0.5f, YELLOW);
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

void saveAgentViewImage(
    Drive *env,
    Client *client,
    const char *filename,
    RenderTexture2D target,
    int map_height,
    int obs_only,
    int lasers,
    int show_grid) {
    // Agent perspective camera following the human agent
    int agent_idx = env->active_agent_indices[env->human_agent_idx];
    Agent *agent = &env->agents[agent_idx];

    Camera3D camera = {0};
    // Position camera behind and above the agent
    camera.position = (Vector3) {agent->sim_x - (25.0f * cosf(agent->sim_heading)),
                                 agent->sim_y - (25.0f * sinf(agent->sim_heading)),
                                 15.0f};
    camera.target = (Vector3) {agent->sim_x + 40.0f * cosf(agent->sim_heading),
                               agent->sim_y + 40.0f * sinf(agent->sim_heading),
                               1.0f};
    camera.up = (Vector3) {0.0f, 0.0f, 1.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Color road = (Color) {35, 35, 37, 255};

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

void c_render(Drive *env, int view_mode) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client *client = env->client;

    // Build the static road cache once per headless client. Interactive mode
    // keeps using draw_scene's per-frame draw_road_edge so hot-editing the
    // viewer still reflects live road state.
    if (env->render_mode == RENDER_HEADLESS && !client->road_cache_valid) {
        build_road_cache(env, client);
    }

    // Pick the camera based on view_mode. Default is the legacy fixed
    // perspective camera stored on the client; BEV_AGENT_OBS rebuilds an
    // ortho camera every frame so it stays centered on the ego agent as
    // the agent moves through the map. Mirrors 3.0 drive.h's c_render.
    Camera3D render_camera;
    if (env->render_mode == RENDER_HEADLESS && view_mode == VIEW_MODE_TOPDOWN_SIM) {
        render_camera = (Camera3D) {0};
        float cx = (env->grid_map->top_left_x + env->grid_map->bottom_right_x) * 0.5f;
        float cy = (env->grid_map->top_left_y + env->grid_map->bottom_right_y) * 0.5f;
        render_camera.position = (Vector3) {cx, cy, 400.0f};
        render_camera.target = (Vector3) {cx, cy, 0.0f};
        render_camera.up = (Vector3) {0.0f, -1.0f, 0.0f};
        render_camera.projection = CAMERA_ORTHOGRAPHIC;
        float map_w = fabsf(env->grid_map->bottom_right_x - env->grid_map->top_left_x);
        float map_h = fabsf(env->grid_map->bottom_right_y - env->grid_map->top_left_y);
        render_camera.fovy = fmaxf(map_w, map_h) * 1.05f;
    } else if (
        env->render_mode == RENDER_HEADLESS && view_mode == VIEW_MODE_BEV_AGENT_OBS && env->active_agent_count > 0) {
        int agent_idx = env->active_agent_indices[env->human_agent_idx];
        Agent *agent = &env->agents[agent_idx];
        render_camera = (Camera3D) {0};
        render_camera.position = (Vector3) {agent->sim_x, agent->sim_y, agent->sim_z + 400.0f};
        render_camera.target = (Vector3) {agent->sim_x, agent->sim_y, agent->sim_z};
        render_camera.up = (Vector3) {0.0f, -1.0f, 0.0f};
        render_camera.projection = CAMERA_ORTHOGRAPHIC;
        float _bev_obs_window
            = fmaxf(fmaxf(env->obs_range_road_front_m, env->obs_range_road_behind_m), env->obs_range_road_side_m);
        int _bev_vrange = (int) ceilf(_bev_obs_window / GRID_CELL_SIZE) + 1;
        render_camera.fovy = (float) _bev_vrange * GRID_CELL_SIZE * 2.0f;
    } else {
        render_camera = client->camera;
    }

    BeginDrawing();
    Color road = (Color) {35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(render_camera);
    if (env->render_mode != RENDER_HEADLESS) {
        handle_camera_controls(env->client);
    }
    if (client->road_cache_valid) {
        draw_road_cached(client);
    }
    // Per 3.0's c_render: BEV uses draw_scene(mode=1, obs_only=1) — the
    // flat wireframe-box agent path that fits the ortho top-down camera.
    // Default view keeps mode=0 (3D car model path) since that matches
    // the legacy fixed perspective camera.
    if (view_mode == VIEW_MODE_BEV_AGENT_OBS) {
        draw_scene(env, client, 1, 1, 0, 0);
    } else {
        draw_scene(env, client, 0, 0, 0, 0);
    }
    if (env->render_mode != RENDER_HEADLESS) {
        // Debug overlay — only meaningful in the interactive viewer.
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
        int human_idx = env->active_agent_indices[env->human_agent_idx];
        DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
        DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);
        DrawText(
            "Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent",
            10,
            client->height - 30,
            20,
            PUFF_WHITE);
        if (env->action_type == 1) { // continuous (float)
            float (*action_array_f)[2] = (float (*)[2]) env->actions;
            DrawText(
                TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]),
                10,
                110,
                20,
                PUFF_WHITE);
            DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, PUFF_WHITE);
        } else { // discrete (int)
            int (*action_array)[2] = (int (*)[2]) env->actions;
            DrawText(TextFormat("Acceleration: %d", action_array[env->human_agent_idx][0]), 10, 110, 20, PUFF_WHITE);
            DrawText(TextFormat("Steering: %d", action_array[env->human_agent_idx][1]), 10, 130, 20, PUFF_WHITE);
        }
        DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, 150, 20, PUFF_WHITE);
        DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, 170, 20, PUFF_WHITE);
    }
#ifdef DRIVE_HAS_EGL
    if (client->egl_mode) {
        // EGL headless: flush the rlgl batch directly. Skip EndDrawing's
        // glfwSwapBuffers + glfwPollEvents — there's no window to swap.
        rlDrawRenderBatchActive();
    } else
#endif
    {
        EndDrawing();
    }

    if (env->render_mode == RENDER_HEADLESS) {
        client_record_frame(client);
    }
}

void close_client(Client *client) {
    // Flush the trailing PBO frame to the ffmpeg pipe before tearing down.
    // The recorder loop only writes frame N when frame N+1 is kicked off, so
    // the last produced frame is still sitting in a mapped PBO when we get
    // here. Without this flush, the output mp4 ends one frame short.
    if (client->recorder_active) {
#ifdef DRIVE_HAS_EGL
        if (client->egl_mode && client->pbo_frame_count > 0) {
            int prev = 1 - client->pbo_index;
            int w = (int) client->width, h = (int) client->height;
            int row_bytes = w * 4;
            glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[prev]);
            unsigned char *ptr = (unsigned char *) glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
            if (ptr) {
                int iov_max = 1024;
                int rows_remaining = h;
                int row_top = 0;
                int io_error = 0;
                while (rows_remaining > 0 && !io_error) {
                    int chunk = rows_remaining < iov_max ? rows_remaining : iov_max;
                    struct iovec iov[1024];
                    size_t chunk_bytes = 0;
                    for (int i = 0; i < chunk; i++) {
                        int src_row = h - 1 - (row_top + i);
                        iov[i].iov_base = ptr + (size_t) src_row * row_bytes;
                        iov[i].iov_len = row_bytes;
                        chunk_bytes += row_bytes;
                    }
                    struct iovec *cur = iov;
                    int cur_cnt = chunk;
                    size_t cur_remaining = chunk_bytes;
                    while (cur_remaining > 0) {
                        ssize_t written = writev(client->recorder_pipefd[1], cur, cur_cnt);
                        if (written < 0) {
                            if (errno == EINTR) {
                                continue;
                            }
                            io_error = 1;
                            break;
                        }
                        cur_remaining -= (size_t) written;
                        size_t consumed = (size_t) written;
                        while (cur_cnt > 0 && consumed >= cur[0].iov_len) {
                            consumed -= cur[0].iov_len;
                            cur++;
                            cur_cnt--;
                        }
                        if (cur_cnt > 0 && consumed > 0) {
                            cur[0].iov_base = (unsigned char *) cur[0].iov_base + consumed;
                            cur[0].iov_len -= consumed;
                        }
                    }
                    row_top += chunk;
                    rows_remaining -= chunk;
                }
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glDeleteBuffers(2, client->pbo);
            client->pbo[0] = 0;
            client->pbo[1] = 0;
        }
#endif
        close(client->recorder_pipefd[1]);
        if (client->recorder_pid > 0) {
            int status = 0;
            waitpid(client->recorder_pid, &status, 0);
            fprintf(stderr, "[drive] ffmpeg child %d exited (status=0x%x)\n", client->recorder_pid, status);
        }
        client->recorder_active = 0;
    }
    if (client->road_cache_valid) {
        UnloadMesh(client->road_tri_mesh);
        client->road_cache_valid = 0;
    }
    free(client->road_line_verts);
    client->road_line_verts = NULL;
    free(client->road_line_colors);
    client->road_line_colors = NULL;
    client->road_line_count = 0;
    // Always unload models — they hold GPU-side VBOs/VAOs/textures tracked
    // by rlgl. Leaking them corrupts rlgl's internal bookkeeping and causes
    // segfaults when the next make_client loads fresh models on the same GL
    // context. Cyclist + pedestrian were being leaked until this fix (3.0's
    // close_client unloads all of them).
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadModel(client->cyclist);
    UnloadModel(client->pedestrian);
    UnloadTexture(client->puffers);

    // CRITICAL: skip CloseWindow + egl_headless_cleanup in egl_mode.
    //
    // CloseWindow tears down GLFW, which on Linux calls glXMakeCurrent(dpy,
    // None, NULL) + glXDestroyContext on the GLFW-owned GLX context. That
    // glXMakeCurrent flushes the GLX connection, which surfaces the queued
    // async BadAccess from egl_switch_to_gpu's earlier release call and
    // crashes the process. It also kills the global GLFW window that the
    // next make_client is supposed to reuse (via the g_glfw_ready guard).
    //
    // egl_headless_cleanup destroys g_egl_ctx.surface/context. The next
    // make_client hits the `egl_ready` static flag (still set) and goes
    // to egl_headless_resize, which operates on a destroyed context =>
    // crash on the second batch of scenarios.
    //
    // 3.0 keeps GLFW + EGL persistent across render envs — see drive.h
    // close_client (the "EGL mode: don't touch GLFW/Xvfb/EGL" comment).
    if (!client->egl_mode) {
        CloseWindow();
#ifdef DRIVE_HAS_EGL
        egl_headless_cleanup();
#endif
    }
    free(client);
}

#endif // RENDER_H
