/*
 * vk_pipeline.h — graphics pipelines for trajviz.
 *
 * Two pipelines, one shared pipeline layout:
 *
 *   - line_pipeline: VK_PRIMITIVE_TOPOLOGY_LINE_LIST. Binding 0 = vec2
 *     per vertex (8 bytes stride, per-vertex rate). Used for road
 *     polylines and (eventually) trajectory traces. The vertex buffer is
 *     a flat array of (x, y) pairs with per-polyline runs delimited by
 *     vkCmdDraw calls (one draw per polyline) — no index buffer.
 *
 *   - box_pipeline: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST. Binding 0 = the
 *     per-vertex unit quad (4 verts, never changes). Binding 1 =
 *     per-instance AgentInstance (40 bytes stride, per-instance rate).
 *     Drawn with vkCmdDrawIndexed using a 6-index quad index buffer
 *     and instance count = number of active agents.
 *
 * Pipeline layout: 0 descriptor sets, 1 push-constant range (PushConstants
 * struct, 80 bytes, vertex+fragment stages). All per-frame state goes
 * through push constants — no descriptor pool, no UBO juggling. This is
 * fine because we only ever push (mat4 mvp + vec4 tint) per draw, well
 * under the 128-byte minimum guaranteed limit.
 *
 * Color attachment format is captured at pipeline-creation time via
 * VkPipelineRenderingCreateInfo (the dynamic_rendering equivalent of a
 * VkRenderPass). The renderer's color image MUST match this format.
 */

#ifndef VK_PIPELINE_H
#define VK_PIPELINE_H

#include "vk_context.h"
#include <vulkan/vulkan.h>

/* Push constant block — must match the GLSL Push struct layout in
 * polyline.vert / agent_box.vert. std430 is implicit for push constants. */
typedef struct PushConstants {
    float mvp[16];  /* column-major mat4 */
    float color[4]; /* polyline: line color; agent: per-view tint */
} PushConstants;

/* Per-instance attributes for agent boxes — must match the vertex input
 * layout below and the location 1..3 attributes in agent_box.vert. */
typedef struct AgentInstance {
    float pose[4];  /* (x, y, heading_rad, _pad) */
    float size[2];  /* (length, width) meters */
    float color[4]; /* (r, g, b, a) */
} AgentInstance;

typedef struct Pipelines {
    VkPipelineLayout layout;
    VkPipeline line_pipeline;
    VkPipeline box_pipeline;
    VkFormat color_format;
} Pipelines;

/* Build both pipelines targeting the given color attachment format and
 * viewport size. The viewport is dynamic state, so width/height are only
 * advisory at this stage — set them per-frame via vkCmdSetViewport. */
int vk_pipelines_init(VkCtx *ctx, Pipelines *p, VkFormat color_format);

void vk_pipelines_destroy(VkCtx *ctx, Pipelines *p);

#endif
