/*
 * test_main.c — standalone smoke test for the trajviz C side.
 *
 * Builds a synthetic scene (road grid + a few moving agents), calls
 * trajviz_render_episode, and exits. No npz, no Python, no map parsing —
 * the whole point is to validate the Vulkan + ffmpeg path in isolation
 * before wiring it up to anything else.
 *
 * Build (after libvulkan-dev + glslang-tools are installed and shaders
 * have been compiled by shaders/build_shaders.sh):
 *
 *   cd pufferlib/ocean/drive/trajviz
 *   bash shaders/build_shaders.sh
 *   cc -O2 -Wall -Wextra -I. \
 *      tests/test_main.c trajviz.c vk_context.c vk_pipeline.c \
 *      vk_renderer.c ffmpeg_pipe.c shaders.c \
 *      -lvulkan -lm -lpthread -o tests/trajviz_test
 *   ./tests/trajviz_test
 *
 * Outputs test_topdown.mp4 and test_bev.mp4 in the current directory.
 *
 * Open them and check:
 *   - Top-down: a 200x200 m square road grid with a horizontal line of
 *     blue cars moving rightward. One car is orange (the ego) and stays
 *     in the middle of the line.
 *   - BEV: the orange ego car at the center, facing up, with the road
 *     grid sliding past from top to bottom as the car moves +x.
 */

#include "../trajviz.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 1280
#define HEIGHT 720
#define NUM_STEPS 90 /* 3 seconds at 30 fps */
#define NUM_AGENTS 8
#define FPS 30

/* Build a simple road network: a grid of horizontal and vertical lines
 * spanning [-100, +100]^2 meters at 25m spacing. Each grid line is one
 * polyline with 2 vertices (a single segment). */
static void build_grid(float **out_xy, uint32_t **out_offsets, uint32_t **out_types, uint32_t *out_num_polys,
                       uint32_t *out_num_verts) {
    const float extent = 100.0f;
    const float step = 25.0f;
    const int n = (int)(2 * extent / step) + 1; /* 9 lines per axis */
    const int total = n * 2;                    /* 18 polylines */

    *out_num_polys = (uint32_t)total;
    *out_num_verts = (uint32_t)(total * 2);

    *out_xy = (float *)calloc((size_t)total * 2 * 2, sizeof(float));
    *out_offsets = (uint32_t *)calloc((size_t)total + 1, sizeof(uint32_t));
    *out_types = (uint32_t *)calloc((size_t)total, sizeof(uint32_t));

    int p = 0;
    int v = 0;
    /* Horizontal lines (constant y) */
    for (int i = 0; i < n; ++i) {
        float y = -extent + i * step;
        (*out_xy)[v * 2 + 0] = -extent;
        (*out_xy)[v * 2 + 1] = y;
        v++;
        (*out_xy)[v * 2 + 0] = extent;
        (*out_xy)[v * 2 + 1] = y;
        v++;
        (*out_offsets)[p + 1] = (uint32_t)v;
        (*out_types)[p] = TVZ_ROAD_EDGE;
        p++;
    }
    /* Vertical lines (constant x) */
    for (int i = 0; i < n; ++i) {
        float x = -extent + i * step;
        (*out_xy)[v * 2 + 0] = x;
        (*out_xy)[v * 2 + 1] = -extent;
        v++;
        (*out_xy)[v * 2 + 0] = x;
        (*out_xy)[v * 2 + 1] = extent;
        v++;
        (*out_offsets)[p + 1] = (uint32_t)v;
        (*out_types)[p] = TVZ_ROAD_LANE;
        p++;
    }
}

/* Build NUM_STEPS frames of a horizontal line of agents moving in +x at
 * 10 m/s. Stored step-major: traj[step*NA*3 + a*3 + {0,1,2}]. */
static float *build_trajectory(void) {
    float *traj = (float *)calloc((size_t)NUM_STEPS * NUM_AGENTS * 3, sizeof(float));
    const float vx_per_step = 10.0f / FPS; /* 10 m/s */
    for (int step = 0; step < NUM_STEPS; ++step) {
        for (int a = 0; a < NUM_AGENTS; ++a) {
            float base_x = -40.0f + a * 12.0f;
            float x = base_x + step * vx_per_step;
            float y = 0.0f;
            float h = 0.0f; /* facing +x */
            size_t off = ((size_t)step * NUM_AGENTS + a) * 3;
            traj[off + 0] = x;
            traj[off + 1] = y;
            traj[off + 2] = h;
        }
    }
    return traj;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    TrajvizCtx *ctx = trajviz_init(WIDTH, HEIGHT);
    if (!ctx) {
        fprintf(stderr, "trajviz_init failed: %s\n", trajviz_last_error(NULL));
        return 1;
    }

    float *road_xy = NULL;
    uint32_t *road_off = NULL;
    uint32_t *road_typ = NULL;
    uint32_t num_polys = 0, num_verts = 0;
    build_grid(&road_xy, &road_off, &road_typ, &num_polys, &num_verts);

    float *traj = build_trajectory();

    /* All agents are valid for the full episode. */
    int32_t lengths[NUM_AGENTS];
    for (int i = 0; i < NUM_AGENTS; ++i)
        lengths[i] = NUM_STEPS;

    /* Default agent dimensions for all. */
    int rc = trajviz_render_episode(ctx, road_xy, road_off, road_typ, num_polys, traj, NUM_STEPS, NUM_AGENTS,
                                    NULL,       /* agent_dims = default */
                                    lengths, 4, /* ego_idx — middle of the line */
                                    FPS, "test_topdown.mp4", "test_bev.mp4");

    if (rc != TRAJVIZ_OK) {
        fprintf(stderr, "trajviz_render_episode failed (%d): %s\n", rc, trajviz_last_error(ctx));
    } else {
        fprintf(stderr, "wrote test_topdown.mp4 and test_bev.mp4\n");
    }

    free(road_xy);
    free(road_off);
    free(road_typ);
    free(traj);
    trajviz_close(ctx);
    return rc == TRAJVIZ_OK ? 0 : 1;
}
