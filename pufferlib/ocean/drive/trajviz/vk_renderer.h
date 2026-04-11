/*
 * vk_renderer.h — pipelined per-episode render state.
 *
 * The renderer keeps a small ring of frame slots — each one a complete
 * snapshot of "everything the GPU needs to render one frame": its own
 * command buffer, fence, render-target images, host readback buffers,
 * and per-frame instance vertex buffer. The CPU can record frame N+1
 * into slot S+1 while the GPU runs frame N on slot S and the host reads
 * the readback for frame N-1 from slot S-1, so per-submit latency
 * (vkQueueSubmit + vkWaitForFences scheduler wakeup) is amortized
 * across FRAMES_IN_FLIGHT frames.
 *
 * Static, episode-level data (road geometry, polyline metadata) lives
 * outside the slots — it's read by every slot and never written during
 * the loop, so no synchronization is needed.
 *
 * Episode lifecycle:
 *
 *   vk_renderer_episode_begin(r, ffmpeg_td, ffmpeg_bev);
 *   for each frame:
 *       vk_renderer_submit_frame(r, instances, n, mvp_td, mvp_bev);
 *       // returns immediately after submitting; may have internally
 *       // drained an older slot and fwritten its readback to ffmpeg
 *   vk_renderer_episode_end(r);
 *       // drains the remaining FRAMES_IN_FLIGHT - 1 in-flight frames
 *
 * The renderer holds the ffmpeg pipe pointers for the duration of an
 * episode so submit_frame and episode_end can write directly to them
 * during the drain phase, without going back through the orchestrator.
 */

#ifndef VK_RENDERER_H
#define VK_RENDERER_H

#include "vk_context.h"
#include "vk_pipeline.h"
#include "vk_math.h"
#include "ffmpeg_pipe.h"

#include <vulkan/vulkan.h>

/* The frames-in-flight ring is currently a no-op (=1) — empirical
 * timings showed that on this Vulkan path, neither the per-fence wait
 * nor the per-submit latency is the dominant cost (we tested up to 16
 * slots and saw no improvement). The episode_begin/submit_frame/end
 * API shape is preserved because the batched renderer in
 * vk_batch_renderer.{h,c} still relies on a coordinated drain phase. */
#define FRAMES_IN_FLIGHT 1

/* Buffer + memory pair, optionally persistently mapped. */
typedef struct VkBufferM {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    VkDeviceSize size;
} VkBufferM;

/* Image + memory + view bundle. */
typedef struct VkImageM {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    uint32_t width, height;
    VkFormat format;
} VkImageM;

/* One render target = one rendered view. The readback buffer is
 * persistently mapped so we fwrite directly from VRAM-staged DMA. */
typedef struct RenderTarget {
    VkImageM color;
    VkBufferM readback;
    size_t row_pitch_bytes;
} RenderTarget;

/* One slot in the frames-in-flight ring. Holds everything that varies
 * per-frame, so frame N+1 doesn't stomp on data the GPU is still
 * reading for frame N. */
typedef struct FrameSlot {
    VkCommandBuffer cmd;
    VkFence fence; /* signals when this slot's GPU work is done */
    RenderTarget rt_topdown;
    RenderTarget rt_bev;
    VkBufferM agent_inst_vb;
    uint32_t agent_inst_capacity; /* in instances */
    int pending;                  /* 1 if a submit on this slot is in flight */
    int rendered_topdown;         /* did we draw the topdown view this frame? */
    int rendered_bev;
} FrameSlot;

typedef struct Renderer {
    Pipelines *pipelines; /* borrowed */
    uint32_t width, height;

    /* Static geometry, set up once at init. */
    VkBufferM unit_quad_vb;
    VkBufferM unit_quad_ib;

    /* Per-episode geometry (constant across all frames in an episode).
     * Read by every slot's command buffer; never written during the
     * pipelined loop, so no per-slot duplication needed. */
    VkBufferM road_vb;
    uint32_t road_vb_capacity;
    uint32_t *road_offsets; /* (num_polys+1,) host copy */
    uint32_t *road_types;   /* (num_polys,)  */
    uint32_t num_polys;
    uint32_t road_meta_capacity;

    /* Frames-in-flight ring + FIFO indices. */
    FrameSlot slots[FRAMES_IN_FLIGHT];
    int head; /* next slot to drain */
    int tail; /* next slot to write */
    int n_in_flight;

    /* Ffmpeg pipes for the current episode. Owned by the caller; the
     * renderer just borrows the pointers between episode_begin and
     * episode_end. NULL = view disabled for this episode. */
    FfmpegPipe *ep_pipe_topdown;
    FfmpegPipe *ep_pipe_bev;
} Renderer;

int vk_renderer_init(VkCtx *ctx, Pipelines *p, Renderer *r, uint32_t width, uint32_t height);
void vk_renderer_destroy(VkCtx *ctx, Renderer *r);

/* Upload road geometry for a new episode. The data is copied; pointers
 * are not retained past this call. */
int vk_renderer_set_roads(VkCtx *ctx, Renderer *r, const float *road_xy, uint32_t num_verts,
                          const uint32_t *road_offsets, const uint32_t *road_types, uint32_t num_polys);

/* Begin an episode. Stores the ffmpeg pipe pointers and resets the
 * frames-in-flight FIFO. Either pipe may be NULL to disable that view. */
void vk_renderer_episode_begin(Renderer *r, FfmpegPipe *pipe_topdown, FfmpegPipe *pipe_bev);

/* Submit one frame. May internally wait on the oldest pending slot and
 * fwrite its readback buffers to the ffmpeg pipes set in episode_begin
 * before reusing it for this frame. Returns immediately after the
 * submit completes (does not wait on the just-submitted frame). */
int vk_renderer_submit_frame(VkCtx *ctx, Renderer *r, const AgentInstance *instances, uint32_t num_instances,
                             const Mat4 *mvp_topdown, const Mat4 *mvp_bev);

/* Drain remaining in-flight frames at end of episode, fwriting each to
 * the ffmpeg pipes in submission order. */
int vk_renderer_episode_end(VkCtx *ctx, Renderer *r);

#endif
