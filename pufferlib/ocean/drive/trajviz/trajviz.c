/*
 * trajviz.c — public C API: orchestrates Vulkan + ffmpeg per episode.
 *
 * The public functions live here. They compose vk_context, vk_pipeline,
 * vk_renderer, and ffmpeg_pipe into a single "render this episode to
 * these mp4 paths" call. The CPython extension and the standalone test
 * harness both call into this and nothing else.
 *
 * Per-frame loop:
 *   1. For each agent in (0, num_agents):
 *        - Skip if agent_lengths is set and step >= length.
 *        - Pull (x, y, heading) from traj_xyh[step][agent].
 *        - Pull (length, width) from agent_dims (or defaults).
 *        - Color: ego = orange, others = teal.
 *        - Append to instances[].
 *   2. Build top-down camera (fits the road AABB) — same matrix every
 *      frame, computed once outside the loop.
 *   3. Build BEV camera from ego pose at this step.
 *   4. vk_renderer_render_frame.
 *   5. fwrite each readback buffer into the corresponding ffmpeg pipe.
 *
 * The road AABB is computed once on the road verts before the per-frame
 * loop. The BEV window is hardcoded to 50 m half-extent (matching the
 * drive sim's observation_window_size of 100 m total).
 */

#include "trajviz.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include "vk_renderer.h"
#include "vk_batch_renderer.h"
#include "ffmpeg_pipe.h"
#include "vk_math.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* The opaque ctx struct exposed via trajviz.h. Holds everything that
 * lives across episodes. */
struct TrajvizCtx {
    VkCtx vk;
    Pipelines pipelines;
    Renderer renderer;

    /* Lazily-allocated batched renderer. Created on first call to
     * render_episodes_batch and reused if subsequent batches request
     * the same batch_size. If a different batch_size is requested,
     * the existing one is destroyed and recreated. */
    BatchRenderer batch;
    int batch_initialized;
    int batch_size_cur;

    /* Reusable scratch buffer for per-frame instance arrays. Grows on
     * demand and stays at the high water mark for the ctx lifetime. */
    AgentInstance *scratch_instances;
    uint32_t scratch_capacity;

    /* Mirror of the last error string from the underlying VkCtx so the
     * caller can read it via trajviz_last_error even if vk has been
     * partially torn down. */
    char last_error[TRAJVIZ_ERROR_BUF];
};

/* Global last-error slot for init failures (when there's no ctx yet). */
static char g_init_error[TRAJVIZ_ERROR_BUF];

const char *trajviz_last_error(const TrajvizCtx *ctx) {
    if (!ctx)
        return g_init_error;
    return ctx->last_error;
}

static void copy_error(TrajvizCtx *ctx) {
    if (!ctx)
        return;
    snprintf(ctx->last_error, TRAJVIZ_ERROR_BUF, "%s", ctx->vk.last_error);
}

/* Forward declarations of helpers used by both single-episode and
 * batched paths. They live near the bottom of the file. */
static void compute_road_aabb(const float *road_xy, uint32_t num_verts, float aabb[4]);
static int32_t resolve_ego(int32_t requested, uint32_t num_agents, const int32_t *lengths);
static int ensure_scratch(TrajvizCtx *ctx, uint32_t num);

TrajvizCtx *trajviz_init(int width, int height) {
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192) {
        snprintf(g_init_error, sizeof(g_init_error), "invalid dimensions %dx%d", width, height);
        return NULL;
    }
    TrajvizCtx *ctx = (TrajvizCtx *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        snprintf(g_init_error, sizeof(g_init_error), "out of memory");
        return NULL;
    }

    int rc = vk_ctx_init(&ctx->vk);
    if (rc != 0) {
        /* %.480s caps the source string so the total snprintf output
         * can't exceed g_init_error's 512-byte capacity. */
        snprintf(g_init_error, sizeof(g_init_error), "vk_ctx_init: %.480s", ctx->vk.last_error);
        free(ctx);
        return NULL;
    }

    rc = vk_pipelines_init(&ctx->vk, &ctx->pipelines, VK_FORMAT_R8G8B8A8_UNORM);
    if (rc != 0) {
        snprintf(g_init_error, sizeof(g_init_error), "vk_pipelines_init: %.480s", ctx->vk.last_error);
        vk_ctx_destroy(&ctx->vk);
        free(ctx);
        return NULL;
    }

    rc = vk_renderer_init(&ctx->vk, &ctx->pipelines, &ctx->renderer, (uint32_t)width, (uint32_t)height);
    if (rc != 0) {
        snprintf(g_init_error, sizeof(g_init_error), "vk_renderer_init: %.480s", ctx->vk.last_error);
        vk_pipelines_destroy(&ctx->vk, &ctx->pipelines);
        vk_ctx_destroy(&ctx->vk);
        free(ctx);
        return NULL;
    }

    fprintf(stderr, "[trajviz] using device: %s\n", ctx->vk.device_name);
    return ctx;
}

void trajviz_close(TrajvizCtx *ctx) {
    if (!ctx)
        return;
    if (ctx->batch_initialized) {
        vk_batch_renderer_destroy(&ctx->vk, &ctx->batch);
        ctx->batch_initialized = 0;
    }
    vk_renderer_destroy(&ctx->vk, &ctx->renderer);
    vk_pipelines_destroy(&ctx->vk, &ctx->pipelines);
    vk_ctx_destroy(&ctx->vk);
    free(ctx->scratch_instances);
    free(ctx);
}

/* Compute the AABB of the road verts. Used to fit the top-down camera. */
static void compute_road_aabb(const float *road_xy, uint32_t num_verts, float aabb[4]) {
    if (num_verts == 0) {
        aabb[0] = -50.0f;
        aabb[1] = -50.0f;
        aabb[2] = 50.0f;
        aabb[3] = 50.0f;
        return;
    }
    float xmin = road_xy[0], xmax = road_xy[0];
    float ymin = road_xy[1], ymax = road_xy[1];
    for (uint32_t i = 1; i < num_verts; ++i) {
        float x = road_xy[i * 2 + 0];
        float y = road_xy[i * 2 + 1];
        if (x < xmin)
            xmin = x;
        if (x > xmax)
            xmax = x;
        if (y < ymin)
            ymin = y;
        if (y > ymax)
            ymax = y;
    }
    aabb[0] = xmin;
    aabb[1] = ymin;
    aabb[2] = xmax;
    aabb[3] = ymax;
}

/* Determine the ego agent for the BEV view: the requested index, or the
 * first agent with at least 2 valid steps if -1. */
static int32_t resolve_ego(int32_t requested, uint32_t num_agents, const int32_t *lengths) {
    if (requested >= 0 && (uint32_t)requested < num_agents)
        return requested;
    if (!lengths)
        return 0;
    for (uint32_t a = 0; a < num_agents; ++a) {
        if (lengths[a] >= 2)
            return (int32_t)a;
    }
    return 0;
}

static int ensure_scratch(TrajvizCtx *ctx, uint32_t num) {
    if (num <= ctx->scratch_capacity)
        return 0;
    uint32_t cap = 16;
    while (cap < num)
        cap <<= 1;
    AgentInstance *p = (AgentInstance *)realloc(ctx->scratch_instances, cap * sizeof(AgentInstance));
    if (!p) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "out of memory for scratch instances (%u)", cap);
        return -1;
    }
    ctx->scratch_instances = p;
    ctx->scratch_capacity = cap;
    return 0;
}

/* Default agent dimensions if the caller doesn't supply per-agent ones.
 * Roughly the median car size in Waymo Open. */
#define DEFAULT_AGENT_LENGTH 5.0f
#define DEFAULT_AGENT_WIDTH 2.0f

int trajviz_render_episode(TrajvizCtx *ctx, const float *road_xy, const uint32_t *road_offsets,
                           const uint32_t *road_types, uint32_t num_road_polys, const float *traj_xyh,
                           uint32_t num_steps, uint32_t num_agents, const float *agent_dims,
                           const int32_t *agent_lengths, int32_t ego_idx, int fps, const char *out_topdown_mp4,
                           const char *out_bev_mp4) {
    if (!ctx)
        return TRAJVIZ_ERR_BAD_ARG;
    if (!traj_xyh || num_steps == 0 || num_agents == 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "empty trajectory (steps=%u agents=%u)", num_steps,
                 num_agents);
        return TRAJVIZ_ERR_BAD_ARG;
    }
    if (!out_topdown_mp4 && !out_bev_mp4) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "no output paths supplied — at least one of topdown/bev must be set");
        return TRAJVIZ_ERR_BAD_ARG;
    }
    if (fps <= 0)
        fps = 30;

    /* Number of road verts is the last entry of road_offsets, by CSR
     * convention. Allow zero polylines (and therefore zero verts). */
    uint32_t num_road_verts = (num_road_polys > 0) ? road_offsets[num_road_polys] : 0;

    int rc = vk_renderer_set_roads(&ctx->vk, &ctx->renderer, road_xy, num_road_verts, road_offsets, road_types,
                                   num_road_polys);
    if (rc != 0) {
        copy_error(ctx);
        return TRAJVIZ_ERR_VK_DEVICE;
    }

    if (ensure_scratch(ctx, num_agents) != 0) {
        return TRAJVIZ_ERR_VK_OOM;
    }

    /* Pre-compute the road AABB and the static top-down camera. */
    float aabb[4];
    compute_road_aabb(road_xy, num_road_verts, aabb);
    Mat4 mvp_topdown =
        mat4_fit_aabb(aabb[0], aabb[1], aabb[2], aabb[3], (int)ctx->renderer.width, (int)ctx->renderer.height, 0.05f);

    int32_t ego = resolve_ego(ego_idx, num_agents, agent_lengths);

    /* Episode length: the longest agent's lifetime, capped at num_steps.
     * Without lengths we use the full traj. */
    uint32_t ep_len = num_steps;
    if (agent_lengths) {
        uint32_t maxlen = 0;
        for (uint32_t a = 0; a < num_agents; ++a) {
            uint32_t l = (agent_lengths[a] < 0) ? 0 : (uint32_t)agent_lengths[a];
            if (l > maxlen)
                maxlen = l;
        }
        if (maxlen < ep_len)
            ep_len = maxlen;
        if (ep_len == 0)
            ep_len = 1; /* always render at least one frame */
    }

    /* Open ffmpeg pipes for whichever views were requested. */
    FfmpegPipe pipe_td = {0}, pipe_bev = {0};
    int has_td = (out_topdown_mp4 != NULL);
    int has_bev = (out_bev_mp4 != NULL);
    if (has_td) {
        if (ffmpeg_pipe_open(&pipe_td, (int)ctx->renderer.width, (int)ctx->renderer.height, fps, out_topdown_mp4) !=
            0) {
            snprintf(ctx->last_error, sizeof(ctx->last_error), "ffmpeg_pipe_open failed for top-down view (%s)",
                     out_topdown_mp4);
            return TRAJVIZ_ERR_FFMPEG_SPAWN;
        }
    }
    if (has_bev) {
        if (ffmpeg_pipe_open(&pipe_bev, (int)ctx->renderer.width, (int)ctx->renderer.height, fps, out_bev_mp4) != 0) {
            snprintf(ctx->last_error, sizeof(ctx->last_error), "ffmpeg_pipe_open failed for bev view (%s)",
                     out_bev_mp4);
            if (has_td)
                ffmpeg_pipe_close(&pipe_td);
            return TRAJVIZ_ERR_FFMPEG_SPAWN;
        }
    }

    /* Hand the pipes to the renderer for the duration of the episode.
     * It will fwrite to them internally as the frames-in-flight ring
     * drains older slots. */
    vk_renderer_episode_begin(&ctx->renderer, has_td ? &pipe_td : NULL, has_bev ? &pipe_bev : NULL);

    int err = TRAJVIZ_OK;
    for (uint32_t step = 0; step < ep_len; ++step) {
        /* Build instance array for this frame. */
        uint32_t n_inst = 0;
        for (uint32_t a = 0; a < num_agents; ++a) {
            if (agent_lengths && step >= (uint32_t)((agent_lengths[a] < 0) ? 0 : agent_lengths[a])) {
                continue;
            }
            const float *ph = &traj_xyh[((size_t)step * num_agents + a) * 3];
            if (ph[0] == 0.0f && ph[1] == 0.0f)
                continue;

            AgentInstance *ai = &ctx->scratch_instances[n_inst++];
            ai->pose[0] = ph[0];
            ai->pose[1] = ph[1];
            ai->pose[2] = ph[2];
            ai->pose[3] = 0.0f;
            if (agent_dims) {
                ai->size[0] = agent_dims[a * 2 + 0];
                ai->size[1] = agent_dims[a * 2 + 1];
            } else {
                ai->size[0] = DEFAULT_AGENT_LENGTH;
                ai->size[1] = DEFAULT_AGENT_WIDTH;
            }
            if ((int32_t)a == ego) {
                ai->color[0] = 1.00f;
                ai->color[1] = 0.55f;
                ai->color[2] = 0.10f;
                ai->color[3] = 1.0f;
            } else {
                ai->color[0] = 0.20f;
                ai->color[1] = 0.75f;
                ai->color[2] = 0.85f;
                ai->color[3] = 1.0f;
            }
        }

        /* BEV camera follows ego at this step. If the ego has terminated
         * (length exceeded), keep the camera at its last valid position
         * by clamping the step index used for the lookup. */
        Mat4 mvp_bev;
        if (has_bev) {
            uint32_t bev_step = step;
            if (agent_lengths && bev_step >= (uint32_t)((agent_lengths[ego] < 0) ? 0 : agent_lengths[ego])) {
                bev_step = (uint32_t)((agent_lengths[ego] <= 0) ? 0 : agent_lengths[ego] - 1);
            }
            const float *ph = &traj_xyh[((size_t)bev_step * num_agents + ego) * 3];
            mvp_bev = mat4_bev_camera(ph[0], ph[1], ph[2], 50.0f, (int)ctx->renderer.width, (int)ctx->renderer.height);
        }

        rc = vk_renderer_submit_frame(&ctx->vk, &ctx->renderer, ctx->scratch_instances, n_inst,
                                      has_td ? &mvp_topdown : NULL, has_bev ? &mvp_bev : NULL);
        if (rc != 0) {
            copy_error(ctx);
            err = (rc == -1) ? TRAJVIZ_ERR_FFMPEG_WRITE : TRAJVIZ_ERR_VK_DEVICE;
            break;
        }
    }

    /* Drain the FRAMES_IN_FLIGHT - 1 slots still pending after the loop. */
    if (err == TRAJVIZ_OK) {
        rc = vk_renderer_episode_end(&ctx->vk, &ctx->renderer);
        if (rc != 0) {
            copy_error(ctx);
            err = (rc == -1) ? TRAJVIZ_ERR_FFMPEG_WRITE : TRAJVIZ_ERR_VK_DEVICE;
        }
    } else {
        /* Best-effort drain so we don't leave the renderer in a half-state. */
        vk_renderer_episode_end(&ctx->vk, &ctx->renderer);
    }

    if (has_td)
        ffmpeg_pipe_close(&pipe_td);
    if (has_bev)
        ffmpeg_pipe_close(&pipe_bev);
    return err;
}

/* ============================================================================
 * Batched multi-episode rendering
 * ============================================================================
 *
 * Renders N episodes simultaneously by tiling them into a per-view atlas
 * image and recording all N tiles into one command-buffer per frame. The
 * BatchRenderer is held in TrajvizCtx and reused across calls; if a
 * subsequent call requests a different batch_size, we destroy and
 * recreate (init cost ~20 ms paid once per unique size).
 *
 * Atlas tile dimensions match ctx->renderer.width/height — i.e. the same
 * resolution as a single-episode render. The atlas is tile_w × (N*tile_h)
 * with tiles stacked vertically so each tile's bytes are contiguous in
 * the host readback buffer (one fwrite per tile per frame, no row stitching).
 */

#define TRAJVIZ_BATCH_MAX 16

int trajviz_render_episodes_batch(TrajvizCtx *ctx, int batch_size, uint32_t num_steps, uint32_t max_agents,
                                  const float *all_road_xy, const uint32_t *vert_offsets,
                                  const uint32_t *all_road_offsets, const uint32_t *poly_meta_offsets,
                                  const uint32_t *all_road_types, const uint32_t *poly_type_offsets,
                                  const float *traj_xyh, const int32_t *agent_lengths, const int32_t *ego_idx_per_ep,
                                  int fps, const char **out_topdown_paths, const char **out_bev_paths) {
    if (!ctx)
        return TRAJVIZ_ERR_BAD_ARG;
    if (batch_size <= 0 || batch_size > TRAJVIZ_BATCH_MAX) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "batch_size %d out of range [1, %d]", batch_size,
                 TRAJVIZ_BATCH_MAX);
        return TRAJVIZ_ERR_BAD_ARG;
    }
    if (num_steps == 0 || max_agents == 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "empty trajectory (steps=%u agents=%u)", num_steps,
                 max_agents);
        return TRAJVIZ_ERR_BAD_ARG;
    }
    if (!traj_xyh || !vert_offsets || !poly_meta_offsets || !poly_type_offsets || !agent_lengths) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "null required pointer to render_episodes_batch");
        return TRAJVIZ_ERR_BAD_ARG;
    }
    if (fps <= 0)
        fps = 30;

    /* Lazily allocate or recreate the BatchRenderer when the requested
     * size doesn't match the current one. */
    if (ctx->batch_initialized && ctx->batch_size_cur != batch_size) {
        vk_batch_renderer_destroy(&ctx->vk, &ctx->batch);
        ctx->batch_initialized = 0;
    }
    if (!ctx->batch_initialized) {
        int rc = vk_batch_renderer_init(&ctx->vk, &ctx->pipelines, &ctx->batch, batch_size, ctx->renderer.width,
                                        ctx->renderer.height);
        if (rc != 0) {
            copy_error(ctx);
            return TRAJVIZ_ERR_VK_DEVICE;
        }
        ctx->batch_initialized = 1;
        ctx->batch_size_cur = batch_size;
    }

    /* Per-episode local state. All malloc'd so we can goto a single
     * cleanup label on failure without VLAs. */
    Mat4 *topdown_cams = (Mat4 *)calloc((size_t)batch_size, sizeof(Mat4));
    int32_t *effective_lengths = (int32_t *)calloc((size_t)batch_size, sizeof(int32_t));
    int32_t *resolved_egos = (int32_t *)calloc((size_t)batch_size, sizeof(int32_t));
    FfmpegPipe *pipes_td = (FfmpegPipe *)calloc((size_t)batch_size, sizeof(FfmpegPipe));
    FfmpegPipe *pipes_bev = (FfmpegPipe *)calloc((size_t)batch_size, sizeof(FfmpegPipe));
    int *has_pipe_td = (int *)calloc((size_t)batch_size, sizeof(int));
    int *has_pipe_bev = (int *)calloc((size_t)batch_size, sizeof(int));
    if (!topdown_cams || !effective_lengths || !resolved_egos || !pipes_td || !pipes_bev || !has_pipe_td ||
        !has_pipe_bev) {
        free(topdown_cams);
        free(effective_lengths);
        free(resolved_egos);
        free(pipes_td);
        free(pipes_bev);
        free(has_pipe_td);
        free(has_pipe_bev);
        snprintf(ctx->last_error, sizeof(ctx->last_error), "out of memory allocating batch state");
        return TRAJVIZ_ERR_VK_OOM;
    }

    int err = TRAJVIZ_OK;
    int max_eff_length = 0;

    /* Per-episode setup pass: open pipes, upload roads, compute the
     * frame-invariant top-down camera matrix, resolve ego index. */
    for (int s = 0; s < batch_size; ++s) {
        const char *out_td_s = out_topdown_paths ? out_topdown_paths[s] : NULL;
        const char *out_bev_s = out_bev_paths ? out_bev_paths[s] : NULL;
        if (!out_td_s && !out_bev_s) {
            /* No outputs for this slot — skip entirely. */
            continue;
        }

        /* Slice road data for episode s out of the concatenated arrays. */
        uint32_t v_start = vert_offsets[s];
        uint32_t v_end = vert_offsets[s + 1];
        uint32_t num_verts_s = v_end - v_start;
        const float *xy_s = all_road_xy + (size_t)v_start * 2;

        uint32_t pm_start = poly_meta_offsets[s];
        uint32_t pm_end = poly_meta_offsets[s + 1];
        uint32_t num_polys_plus_1 = (pm_end > pm_start) ? (pm_end - pm_start) : 0;
        uint32_t num_polys_s = (num_polys_plus_1 > 0) ? num_polys_plus_1 - 1 : 0;
        const uint32_t *off_s = (all_road_offsets && num_polys_plus_1 > 0) ? (all_road_offsets + pm_start) : NULL;

        uint32_t pt_start = poly_type_offsets[s];
        const uint32_t *typ_s = (all_road_types && num_polys_s > 0) ? (all_road_types + pt_start) : NULL;

        /* Open ffmpeg pipes for this slot. */
        if (out_td_s) {
            if (ffmpeg_pipe_open(&pipes_td[s], (int)ctx->batch.tile_w, (int)ctx->batch.tile_h, fps, out_td_s) != 0) {
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                         "ffmpeg_pipe_open failed for episode %d top-down (%s)", s, out_td_s);
                err = TRAJVIZ_ERR_FFMPEG_SPAWN;
                goto cleanup;
            }
            has_pipe_td[s] = 1;
        }
        if (out_bev_s) {
            if (ffmpeg_pipe_open(&pipes_bev[s], (int)ctx->batch.tile_w, (int)ctx->batch.tile_h, fps, out_bev_s) != 0) {
                snprintf(ctx->last_error, sizeof(ctx->last_error), "ffmpeg_pipe_open failed for episode %d bev (%s)", s,
                         out_bev_s);
                err = TRAJVIZ_ERR_FFMPEG_SPAWN;
                goto cleanup;
            }
            has_pipe_bev[s] = 1;
        }

        int rc =
            vk_batch_renderer_set_episode(&ctx->vk, &ctx->batch, s, xy_s, num_verts_s, off_s, typ_s, num_polys_s,
                                          has_pipe_td[s] ? &pipes_td[s] : NULL, has_pipe_bev[s] ? &pipes_bev[s] : NULL);
        if (rc != 0) {
            copy_error(ctx);
            err = TRAJVIZ_ERR_VK_DEVICE;
            goto cleanup;
        }

        /* Top-down camera = fit road AABB to tile (frame-invariant). */
        float aabb[4];
        compute_road_aabb(xy_s, num_verts_s, aabb);
        topdown_cams[s] =
            mat4_fit_aabb(aabb[0], aabb[1], aabb[2], aabb[3], (int)ctx->batch.tile_w, (int)ctx->batch.tile_h, 0.05f);

        /* Effective episode length = max valid agent_lengths in this slot. */
        const int32_t *lens_s = agent_lengths + (size_t)s * max_agents;
        int32_t maxlen = 0;
        for (uint32_t a = 0; a < max_agents; ++a) {
            int32_t l = lens_s[a];
            if (l < 0)
                l = 0;
            if ((uint32_t)l > num_steps)
                l = (int32_t)num_steps;
            if (l > maxlen)
                maxlen = l;
        }
        effective_lengths[s] = maxlen;
        if (maxlen > max_eff_length)
            max_eff_length = maxlen;

        /* Resolve ego index per episode. */
        int32_t requested_ego = ego_idx_per_ep ? ego_idx_per_ep[s] : -1;
        resolved_egos[s] = resolve_ego(requested_ego, max_agents, lens_s);
    }

    if (max_eff_length == 0)
        max_eff_length = 1;

    /* Make sure the per-frame instance scratch buffer can hold the
     * widest episode's agent count. */
    if (ensure_scratch(ctx, max_agents) != 0) {
        err = TRAJVIZ_ERR_VK_OOM;
        goto cleanup;
    }

    /* Per-frame loop. For each frame, populate every active slot's
     * per-frame state (instances + camera matrices), then submit one
     * batched frame. The submit_frame call internally fwrites every
     * slot's tile to its ffmpeg pipes. */
    for (uint32_t frame = 0; frame < (uint32_t)max_eff_length && err == TRAJVIZ_OK; ++frame) {
        for (int s = 0; s < batch_size; ++s) {
            if (!has_pipe_td[s] && !has_pipe_bev[s])
                continue;

            /* Episode finished? Skip this slot for this frame. */
            if ((int32_t)frame >= effective_lengths[s]) {
                vk_batch_renderer_set_frame(&ctx->vk, &ctx->batch, s, NULL, 0, NULL, NULL);
                continue;
            }

            const int32_t *lens_s = agent_lengths + (size_t)s * max_agents;
            int32_t ego = resolved_egos[s];

            /* Build instance array for this slot for this frame. */
            uint32_t n_inst = 0;
            for (uint32_t a = 0; a < max_agents; ++a) {
                int32_t l = lens_s[a];
                if (l < 0)
                    l = 0;
                if ((int32_t)frame >= l)
                    continue;
                size_t off = (((size_t)s * num_steps + (size_t)frame) * (size_t)max_agents + (size_t)a) * 3;
                const float *ph = &traj_xyh[off];
                if (ph[0] == 0.0f && ph[1] == 0.0f)
                    continue;

                AgentInstance *ai = &ctx->scratch_instances[n_inst++];
                ai->pose[0] = ph[0];
                ai->pose[1] = ph[1];
                ai->pose[2] = ph[2];
                ai->pose[3] = 0.0f;
                ai->size[0] = DEFAULT_AGENT_LENGTH;
                ai->size[1] = DEFAULT_AGENT_WIDTH;
                if ((int32_t)a == ego) {
                    ai->color[0] = 1.00f;
                    ai->color[1] = 0.55f;
                    ai->color[2] = 0.10f;
                    ai->color[3] = 1.0f;
                } else {
                    ai->color[0] = 0.20f;
                    ai->color[1] = 0.75f;
                    ai->color[2] = 0.85f;
                    ai->color[3] = 1.0f;
                }
            }

            /* BEV camera follows the slot's ego at this frame, clamped
             * to the ego's last valid step if it has terminated. */
            Mat4 bev_cam;
            int has_bev = has_pipe_bev[s];
            if (has_bev) {
                uint32_t bev_step = frame;
                int32_t ego_len = lens_s[ego];
                if (ego_len <= 0)
                    ego_len = 1;
                if ((int32_t)bev_step >= ego_len)
                    bev_step = (uint32_t)(ego_len - 1);
                size_t off = (((size_t)s * num_steps + (size_t)bev_step) * (size_t)max_agents + (size_t)ego) * 3;
                const float *ph = &traj_xyh[off];
                bev_cam = mat4_bev_camera(ph[0], ph[1], ph[2], 50.0f, (int)ctx->batch.tile_w, (int)ctx->batch.tile_h);
            }

            int rc = vk_batch_renderer_set_frame(&ctx->vk, &ctx->batch, s, ctx->scratch_instances, n_inst,
                                                 has_pipe_td[s] ? &topdown_cams[s] : NULL, has_bev ? &bev_cam : NULL);
            if (rc != 0) {
                copy_error(ctx);
                err = TRAJVIZ_ERR_VK_DEVICE;
                break;
            }
        }
        if (err != TRAJVIZ_OK)
            break;

        int rc = vk_batch_renderer_submit_frame(&ctx->vk, &ctx->batch);
        if (rc != 0) {
            copy_error(ctx);
            err = (rc == -1) ? TRAJVIZ_ERR_FFMPEG_WRITE : TRAJVIZ_ERR_VK_DEVICE;
            break;
        }
    }

cleanup:
    for (int s = 0; s < batch_size; ++s) {
        if (has_pipe_td[s])
            ffmpeg_pipe_close(&pipes_td[s]);
        if (has_pipe_bev[s])
            ffmpeg_pipe_close(&pipes_bev[s]);
        vk_batch_renderer_close_episode(&ctx->batch, s);
    }
    free(topdown_cams);
    free(effective_lengths);
    free(resolved_egos);
    free(pipes_td);
    free(pipes_bev);
    free(has_pipe_td);
    free(has_pipe_bev);
    return err;
}
