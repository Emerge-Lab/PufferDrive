/*
 * trajviz.h — public C API for the trajviz Vulkan renderer.
 *
 * This header is consumed by:
 *   - _native.c       (CPython extension shell, the production caller)
 *   - tools/test_main.c (standalone test harness, no Python needed)
 *
 * The shape is deliberate: every function takes raw pointers and shapes,
 * never a file path or a numpy object. The Python wrapper is responsible
 * for loading .npz / .bin files and slicing them into per-episode arrays;
 * this layer only knows about geometry and rendering. Keeps the C side
 * focused, and means the same code works behind a CPython extension or a
 * test harness without changes.
 *
 * Coordinate frame: all positions (road_xy, traj_xyh) are in *mean-centered
 * sim frame*, the same frame the trajectories live in. The renderer never
 * sees world_mean — the Python wrapper has already subtracted it from the
 * road geometry by the time pointers reach this layer.
 *
 * Heading convention: radians, math convention (0 = +x, pi/2 = +y, CCW).
 *
 * Lifecycle:
 *   ctx = trajviz_init(W, H);
 *   for each episode:
 *       trajviz_render_episode(ctx, ...);   // blocks until both MP4s closed
 *   trajviz_close(ctx);
 *
 * The Vulkan context (instance, device, queues, pipelines, shaders) is
 * created in init() and reused across all render_episode calls. This is
 * the whole point of having a stateful API: a single trajviz_init pays
 * the ~50ms Vulkan startup cost once for an entire batch of episodes.
 */

#ifndef TRAJVIZ_H
#define TRAJVIZ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque renderer handle. Holds the Vulkan instance, device, command pool,
 * pipelines, and reusable per-frame buffers. Created by trajviz_init,
 * destroyed by trajviz_close. Not thread-safe — one ctx per thread. */
typedef struct TrajvizCtx TrajvizCtx;

/* Return codes for render_episode. 0 = success, negative = error class. */
#define TRAJVIZ_OK 0
#define TRAJVIZ_ERR_BAD_ARG -1
#define TRAJVIZ_ERR_VK_DEVICE -2 /* a vulkan device call failed */
#define TRAJVIZ_ERR_VK_OOM -3
#define TRAJVIZ_ERR_FFMPEG_SPAWN -4 /* could not popen ffmpeg */
#define TRAJVIZ_ERR_FFMPEG_WRITE -5 /* fwrite to ffmpeg pipe failed */
#define TRAJVIZ_ERR_NO_DEVICE -6    /* no vulkan-capable physical device */

/* Get the last error message set by a failed call. The returned string is
 * owned by the ctx (or by global state if ctx is NULL — for init failures)
 * and stays valid until the next trajviz_* call on the same ctx. */
const char *trajviz_last_error(const TrajvizCtx *ctx);

/* Create a renderer.
 *
 * width / height: pixel dimensions of each output video frame. Both views
 *   (top-down and BEV) render at the same size — the BEV could be rendered
 *   smaller, but matching sizes keeps the GPU pipeline state count low and
 *   makes side-by-side video composition trivial.
 *
 * Returns NULL on failure; call trajviz_last_error(NULL) to get a message. */
TrajvizCtx *trajviz_init(int width, int height);

/* Render one episode.
 *
 * Geometry inputs (all read-only, never retained past this call):
 *   road_xy:        (num_road_verts, 2) float32, packed xy pairs
 *   road_offsets:   (num_road_polys + 1,) uint32 — CSR-style; polyline i's
 *                   vertices are road_xy[road_offsets[i]..road_offsets[i+1]]
 *   road_types:     (num_road_polys,) uint32 — TVZ_ROAD_* type ids; the
 *                   renderer maps these to colors
 *   num_road_polys: number of polylines
 *
 *   traj_xyh:       (num_steps, num_agents, 3) float32, step-major.
 *                   Per-frame: traj_xyh[t * num_agents * 3 + a * 3 + {0,1,2}]
 *                   = (x, y, heading) of agent a at step t. Step-major
 *                   layout means one frame's worth is contiguous, which
 *                   is exactly what the per-frame upload wants.
 *
 *   agent_dims:     (num_agents, 2) float32 — (length, width) per agent.
 *                   If NULL, the renderer uses default car dimensions
 *                   (5.0 x 2.0 m).
 *   agent_lengths:  (num_agents,) int32 — valid step count per agent. The
 *                   renderer skips drawing agents past their length and
 *                   ends the episode at max(agent_lengths). NULL = treat
 *                   all agents as fully valid.
 *   ego_idx:        which agent the BEV view follows. Negative values
 *                   pick the first agent with length >= 2.
 *
 * Output paths (must be writable):
 *   out_topdown_mp4: full-map ortho top-down view, NULL to skip
 *   out_bev_mp4:     agent-centric BEV (RenderView.BEV_AGENT_OBS), NULL to skip
 *
 * Other:
 *   fps:             output video framerate, e.g. 30
 *
 * Returns TRAJVIZ_OK or a negative error code; call trajviz_last_error(ctx)
 * for the message. */
int trajviz_render_episode(TrajvizCtx *ctx, const float *road_xy, const uint32_t *road_offsets,
                           const uint32_t *road_types, uint32_t num_road_polys, const float *traj_xyh,
                           uint32_t num_steps, uint32_t num_agents, const float *agent_dims,
                           const int32_t *agent_lengths, int32_t ego_idx, int fps, const char *out_topdown_mp4,
                           const char *out_bev_mp4);

/* Tear down. Idempotent; passing NULL is a no-op. */
void trajviz_close(TrajvizCtx *ctx);

/* Render a batch of episodes simultaneously. All episodes are tiled
 * into a per-view atlas image and drawn in one command-buffer per
 * frame; one queue submit + one fence wait per frame covers
 * batch_size episodes' worth of work. Per-episode wall time should
 * drop by roughly batch_size× compared to calling render_episode
 * batch_size times sequentially (until the GPU saturates).
 *
 * For v1 the batch requires uniform num_steps and max_agents across
 * all episodes — pad shorter trajectories with zeros and use
 * agent_lengths to mark valid steps. Roads are ragged: each episode
 * has its own road geometry packed end-to-end with CSR-style offsets.
 *
 * Concatenation layout for road data:
 *   all_road_xy:        (V_total, 2) float32 — V_total = sum of vert counts
 *   vert_offsets:       (batch_size + 1,) uint32 — episode i's verts are
 *                       all_road_xy[vert_offsets[i] : vert_offsets[i+1]]
 *   all_road_offsets:   (P_total + batch_size,) uint32 — episode i's CSR
 *                       offsets are all_road_offsets[poly_meta_offsets[i] :
 *                       poly_meta_offsets[i] + num_polys_i + 1] — these
 *                       offsets are RELATIVE to episode i's vert range
 *                       (i.e. they index into the slice of all_road_xy)
 *   poly_meta_offsets:  (batch_size + 1,) uint32 — episode i's poly count
 *                       is poly_meta_offsets[i+1] - poly_meta_offsets[i] - 1
 *                       (one extra entry per ep for the closing offset)
 *   all_road_types:     (P_total,) uint32 — type ids, packed without padding
 *   poly_type_offsets:  (batch_size + 1,) uint32 — index into all_road_types
 *
 * Trajectories are uniform shape: (batch_size, num_steps, max_agents, 3).
 *
 * Output paths are an array of C strings; NULL entries skip that view
 * for that episode.
 *
 * Returns TRAJVIZ_OK or a negative error code. */
int trajviz_render_episodes_batch(TrajvizCtx *ctx, int batch_size, uint32_t num_steps, uint32_t max_agents,
                                  const float *all_road_xy, const uint32_t *vert_offsets,
                                  const uint32_t *all_road_offsets, const uint32_t *poly_meta_offsets,
                                  const uint32_t *all_road_types, const uint32_t *poly_type_offsets,
                                  const float *traj_xyh, const int32_t *agent_lengths, const int32_t *ego_idx_per_ep,
                                  int fps, const char **out_topdown_paths, const char **out_bev_paths);

/* Road type ids — copied from drive.h. The renderer hardcodes a color for
 * each. Unknown types render in a default gray. */
#define TVZ_ROAD_LANE 4
#define TVZ_ROAD_LINE 5
#define TVZ_ROAD_EDGE 6
#define TVZ_ROAD_DRIVEWAY 10

#ifdef __cplusplus
}
#endif

#endif /* TRAJVIZ_H */
