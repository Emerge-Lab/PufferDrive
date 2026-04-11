/*
 * vk_pipeline.c — graphics pipeline construction.
 *
 * Both pipelines share the same VkPipelineLayout (one push-constant range,
 * no descriptor sets) but differ in vertex input state and primitive
 * topology. Most state is identical and described once in helper structs.
 */

#include "vk_pipeline.h"
#include "shaders.h"

#include <stdlib.h>
#include <string.h>

static int create_shader_module(VkCtx *ctx, const uint32_t *code, size_t size_bytes, VkShaderModule *out) {
    VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size_bytes,
        .pCode = code,
    };
    VK_CHECK(vkCreateShaderModule(ctx->device, &ci, NULL, out));
    return 0;
}

static int create_pipeline_layout(VkCtx *ctx, Pipelines *p) {
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    VkPipelineLayoutCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    VK_CHECK(vkCreatePipelineLayout(ctx->device, &ci, NULL, &p->layout));
    return 0;
}

/* Common pipeline state shared between line and box pipelines. */
typedef struct PipelineDefaults {
    VkPipelineInputAssemblyStateCreateInfo ia;
    VkPipelineViewportStateCreateInfo vp;
    VkPipelineRasterizationStateCreateInfo rs;
    VkPipelineMultisampleStateCreateInfo ms;
    VkPipelineDepthStencilStateCreateInfo ds;
    VkPipelineColorBlendAttachmentState blend_att;
    VkPipelineColorBlendStateCreateInfo blend;
    VkPipelineDynamicStateCreateInfo dyn;
    VkDynamicState dyn_states[2];
} PipelineDefaults;

static void fill_defaults(PipelineDefaults *d, VkPrimitiveTopology topology) {
    memset(d, 0, sizeof(*d));

    d->ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    d->ia.topology = topology;
    d->ia.primitiveRestartEnable = VK_FALSE;

    /* Viewport + scissor are dynamic state — actual values come from
     * vkCmdSetViewport / vkCmdSetScissor at record time. The struct still
     * needs viewportCount/scissorCount = 1 here. */
    d->vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    d->vp.viewportCount = 1;
    d->vp.scissorCount = 1;

    d->rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    d->rs.polygonMode = VK_POLYGON_MODE_FILL;
    d->rs.cullMode = VK_CULL_MODE_NONE;
    d->rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    d->rs.lineWidth = 1.5f; /* used for line topology only; ignored for tris */

    d->ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    d->ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    d->ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    d->ds.depthTestEnable = VK_FALSE;
    d->ds.depthWriteEnable = VK_FALSE;

    /* Standard alpha-blend over the existing color, so trace overlays and
     * agent boxes with alpha < 1 fade nicely. */
    d->blend_att.blendEnable = VK_TRUE;
    d->blend_att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    d->blend_att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    d->blend_att.colorBlendOp = VK_BLEND_OP_ADD;
    d->blend_att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    d->blend_att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    d->blend_att.alphaBlendOp = VK_BLEND_OP_ADD;
    d->blend_att.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    d->blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    d->blend.attachmentCount = 1;
    d->blend.pAttachments = &d->blend_att;

    d->dyn_states[0] = VK_DYNAMIC_STATE_VIEWPORT;
    d->dyn_states[1] = VK_DYNAMIC_STATE_SCISSOR;
    d->dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    d->dyn.dynamicStateCount = 2;
    d->dyn.pDynamicStates = d->dyn_states;
}

static int create_line_pipeline(VkCtx *ctx, Pipelines *p, VkShaderModule vs, VkShaderModule fs) {
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs,
         .pName = "main"},
    };

    VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(float) * 2,
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    VkVertexInputAttributeDescription attr = {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = 0,
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &attr,
    };

    PipelineDefaults d;
    /* LINE_STRIP lets us draw a polyline of N verts with one vkCmdDraw
     * call (N verts → N-1 connected segments). Previously LINE_LIST
     * forced one draw per segment, which dominated CPU command-recording
     * cost on real maps with 200+ polylines. */
    fill_defaults(&d, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);

    VkPipelineRenderingCreateInfo rci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &p->color_format,
    };

    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rci,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &d.ia,
        .pViewportState = &d.vp,
        .pRasterizationState = &d.rs,
        .pMultisampleState = &d.ms,
        .pDepthStencilState = &d.ds,
        .pColorBlendState = &d.blend,
        .pDynamicState = &d.dyn,
        .layout = p->layout,
    };
    VK_CHECK(vkCreateGraphicsPipelines(ctx->device, VK_NULL_HANDLE, 1, &gci, NULL, &p->line_pipeline));
    return 0;
}

static int create_box_pipeline(VkCtx *ctx, Pipelines *p, VkShaderModule vs, VkShaderModule fs) {
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs,
         .pName = "main"},
    };

    /* Two vertex bindings:
     *   binding 0 — per-vertex unit quad corner (vec2)
     *   binding 1 — per-instance AgentInstance (40 bytes) */
    VkVertexInputBindingDescription bindings[2] = {
        {.binding = 0, .stride = sizeof(float) * 2, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX},
        {.binding = 1, .stride = sizeof(AgentInstance), .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE},
    };
    VkVertexInputAttributeDescription attrs[4] = {
        {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 0},
        {.location = 1, .binding = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = offsetof(AgentInstance, pose)},
        {.location = 2, .binding = 1, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(AgentInstance, size)},
        {.location = 3,
         .binding = 1,
         .format = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset = offsetof(AgentInstance, color)},
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 2,
        .pVertexBindingDescriptions = bindings,
        .vertexAttributeDescriptionCount = 4,
        .pVertexAttributeDescriptions = attrs,
    };

    PipelineDefaults d;
    fill_defaults(&d, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    VkPipelineRenderingCreateInfo rci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &p->color_format,
    };

    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rci,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &d.ia,
        .pViewportState = &d.vp,
        .pRasterizationState = &d.rs,
        .pMultisampleState = &d.ms,
        .pDepthStencilState = &d.ds,
        .pColorBlendState = &d.blend,
        .pDynamicState = &d.dyn,
        .layout = p->layout,
    };
    VK_CHECK(vkCreateGraphicsPipelines(ctx->device, VK_NULL_HANDLE, 1, &gci, NULL, &p->box_pipeline));
    return 0;
}

int vk_pipelines_init(VkCtx *ctx, Pipelines *p, VkFormat color_format) {
    memset(p, 0, sizeof(*p));
    p->color_format = color_format;

    int r = create_pipeline_layout(ctx, p);
    if (r != 0)
        return r;

    VkShaderModule line_vs = VK_NULL_HANDLE, line_fs = VK_NULL_HANDLE;
    VkShaderModule box_vs = VK_NULL_HANDLE, box_fs = VK_NULL_HANDLE;

    r = create_shader_module(ctx, polyline_vert_spv, polyline_vert_spv_size, &line_vs);
    if (r != 0)
        goto cleanup;
    r = create_shader_module(ctx, polyline_frag_spv, polyline_frag_spv_size, &line_fs);
    if (r != 0)
        goto cleanup;
    r = create_shader_module(ctx, agent_box_vert_spv, agent_box_vert_spv_size, &box_vs);
    if (r != 0)
        goto cleanup;
    r = create_shader_module(ctx, agent_box_frag_spv, agent_box_frag_spv_size, &box_fs);
    if (r != 0)
        goto cleanup;

    r = create_line_pipeline(ctx, p, line_vs, line_fs);
    if (r != 0)
        goto cleanup;
    r = create_box_pipeline(ctx, p, box_vs, box_fs);

cleanup:
    /* Shader modules can be destroyed as soon as the pipelines are built —
     * the pipeline keeps its own reference. */
    if (line_vs)
        vkDestroyShaderModule(ctx->device, line_vs, NULL);
    if (line_fs)
        vkDestroyShaderModule(ctx->device, line_fs, NULL);
    if (box_vs)
        vkDestroyShaderModule(ctx->device, box_vs, NULL);
    if (box_fs)
        vkDestroyShaderModule(ctx->device, box_fs, NULL);
    if (r != 0)
        vk_pipelines_destroy(ctx, p);
    return r;
}

void vk_pipelines_destroy(VkCtx *ctx, Pipelines *p) {
    if (!p)
        return;
    if (p->line_pipeline) {
        vkDestroyPipeline(ctx->device, p->line_pipeline, NULL);
        p->line_pipeline = VK_NULL_HANDLE;
    }
    if (p->box_pipeline) {
        vkDestroyPipeline(ctx->device, p->box_pipeline, NULL);
        p->box_pipeline = VK_NULL_HANDLE;
    }
    if (p->layout) {
        vkDestroyPipelineLayout(ctx->device, p->layout, NULL);
        p->layout = VK_NULL_HANDLE;
    }
}
