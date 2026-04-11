/*
 * vk_batch_renderer.h — multi-episode batched renderer.
 *
 * Renders N episodes in lockstep into a single tiled atlas per view, so
 * one queue submit and one fence wait per frame covers N episodes'
 * worth of work. The per-frame Vulkan + ffmpeg overhead that dominated
 * the single-episode path gets amortized across the batch.
 *
 * Atlas layout: tiles are stacked **vertically** in a single 2D image
 * sized (tile_w, batch_size * tile_h). Vertical stacking means each
 * tile's pixel rows are contiguous in memory — slot i's MP4 frame is
 * a single (tile_w * tile_h * 4)-byte block at offset
 * (i * tile_w * tile_h * 4) in the readback buffer, so we can fwrite
 * each tile to its ffmpeg pipe in one syscall with no row stitching.
 *
 * Per-frame command buffer:
 *   1. Top-down atlas: barrier UNDEFINED → COLOR_ATTACHMENT
 *   2. vkCmdBeginRendering on the full atlas
 *   3. For each active slot: set viewport+scissor to its tile rect,
 *      push its top-down camera matrix, draw its road polylines + agent
 *      boxes
 *   4. vkCmdEndRendering
 *   5. Barrier → TRANSFER_SRC, vkCmdCopyImageToBuffer → host readback
 *   6. Same six steps for the BEV atlas
 *   7. Memory barrier → HOST
 *   8. End cmd buffer, submit, wait
 *   9. fwrite each slot's tile to its ffmpeg pipe
 *
 * The orchestrator (trajviz_render_episodes_batch) handles assembling
 * per-frame instance arrays + camera matrices from the per-episode
 * input data, opening/closing ffmpeg pipes, and calling the lifecycle
 * functions below.
 *
 * Lifecycle:
 *   br = vk_batch_renderer_init(ctx, p, batch_n, tile_w, tile_h);
 *   for batch in batches_of_episodes:
 *       for slot in active_slots:
 *           vk_batch_renderer_set_episode(br, slot, roads..., pipes...);
 *       for frame in 0..max_episode_length:
 *           for slot in active_slots:
 *               vk_batch_renderer_set_frame(br, slot, instances, n_inst,
 *                                            mvp_topdown, mvp_bev);
 *           vk_batch_renderer_submit_frame(ctx, br);
 *       for slot in active_slots:
 *           vk_batch_renderer_close_episode(br, slot);
 *   vk_batch_renderer_destroy(ctx, br);
 */

#ifndef VK_BATCH_RENDERER_H
#define VK_BATCH_RENDERER_H

#include "vk_context.h"
#include "vk_pipeline.h"
#include "vk_math.h"
#include "vk_renderer.h" /* for VkBufferM, VkImageM, AgentInstance */
#include "ffmpeg_pipe.h"

#include <vulkan/vulkan.h>

/* Per-episode state held by one slot in the batch. */
typedef struct BatchSlot {
    int active; /* 1 if this slot is currently rendering an episode */

    /* Per-episode static geometry (set by set_episode) */
    VkBufferM road_vb;
    uint32_t road_vb_capacity;
    uint32_t *road_offsets; /* (num_polys+1,) host copy */
    uint32_t *road_types;   /* (num_polys,) */
    uint32_t num_polys;
    uint32_t road_meta_capacity;

    /* Per-frame agent instance buffer (resized as needed). */
    VkBufferM agent_inst_vb;
    uint32_t agent_inst_capacity;
    uint32_t current_n_instances;

    /* Per-frame camera matrices (set by set_frame). NULL pointer = skip. */
    Mat4 mvp_topdown;
    Mat4 mvp_bev;
    int has_topdown_this_frame;
    int has_bev_this_frame;

    /* Per-episode ffmpeg pipes (borrowed from the orchestrator). */
    FfmpegPipe *pipe_topdown; /* may be NULL */
    FfmpegPipe *pipe_bev;     /* may be NULL */
} BatchSlot;

typedef struct BatchRenderer {
    Pipelines *pipelines;    /* borrowed */
    int batch_n;             /* number of slots */
    uint32_t tile_w, tile_h; /* per-tile pixel dimensions */

    BatchSlot *slots; /* batch_n entries */

    /* Tiled atlases — one per view. Width = tile_w, height = batch_n * tile_h.
     * Slot i occupies y in [i * tile_h, (i+1) * tile_h). */
    VkImageM atlas_topdown;
    VkImageM atlas_bev;

    /* Host-visible readback buffers (persistently mapped). One per atlas.
     * Size = tile_w * batch_n * tile_h * 4. Each slot's tile starts at
     * offset i * (tile_w * tile_h * 4) and is contiguous. */
    VkBufferM readback_topdown;
    VkBufferM readback_bev;

    /* Static unit-quad geometry shared across all slots. */
    VkBufferM unit_quad_vb;
    VkBufferM unit_quad_ib;

    /* One command buffer + one fence — single-frame-in-flight is fine
     * once we're batching, since each frame already does N episodes of
     * work in one submit. */
    VkCommandBuffer cmd;
    VkFence fence;
} BatchRenderer;

int vk_batch_renderer_init(VkCtx *ctx, Pipelines *p, BatchRenderer *br, int batch_n, uint32_t tile_w, uint32_t tile_h);
void vk_batch_renderer_destroy(VkCtx *ctx, BatchRenderer *br);

/* Bind an episode to a slot. Copies road geometry into device memory and
 * stores ffmpeg pipe pointers (which the orchestrator must keep alive
 * for the duration of this slot's episode). Either pipe may be NULL. */
int vk_batch_renderer_set_episode(VkCtx *ctx, BatchRenderer *br, int slot, const float *road_xy, uint32_t num_verts,
                                  const uint32_t *road_offsets, const uint32_t *road_types, uint32_t num_polys,
                                  FfmpegPipe *pipe_topdown, FfmpegPipe *pipe_bev);

/* Update per-frame state for one slot: agent instance array + camera
 * matrices. Either MVP pointer may be NULL to skip that view this frame
 * (e.g. the slot's episode has terminated and shouldn't draw anything
 * new). Must be called for every active slot before submit_frame. */
int vk_batch_renderer_set_frame(VkCtx *ctx, BatchRenderer *br, int slot, const AgentInstance *instances,
                                uint32_t num_instances, const Mat4 *mvp_topdown, const Mat4 *mvp_bev);

/* Submit one batched frame: records all slots' draws into one command
 * buffer, submits, waits for the fence, and fwrites each tile to its
 * slot's ffmpeg pipes. */
int vk_batch_renderer_submit_frame(VkCtx *ctx, BatchRenderer *br);

/* Mark a slot as inactive (the episode finished). Does not free its
 * device buffers — they get reused if a future set_episode lands here. */
void vk_batch_renderer_close_episode(BatchRenderer *br, int slot);

#endif
