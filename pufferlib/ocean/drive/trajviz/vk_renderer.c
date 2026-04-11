/*
 * vk_renderer.c — pipelined per-frame rendering with frames-in-flight ring.
 *
 * The orchestrator (trajviz.c) calls episode_begin → submit_frame×N →
 * episode_end. Inside, the renderer keeps a small ring of FrameSlot
 * structs and walks it as a FIFO: the CPU records the next slot, the
 * GPU runs the previous one(s), and the host reads back from whichever
 * slot is now signaled. This amortizes the per-submit + per-wait
 * scheduler latency (which is the dominant cost on this path) across
 * FRAMES_IN_FLIGHT frames.
 *
 * The actual command-buffer recording (record_view at the bottom of
 * this file) is unchanged from the synchronous version — it draws roads
 * with LINE_STRIP topology and instanced agent boxes.
 */

#include "vk_renderer.h"
#include "shaders.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ----------------------------- buffer helpers ----------------------------- */

static int create_buffer(VkCtx *ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props,
                         int map_persistent, VkBufferM *out) {
    memset(out, 0, sizeof(*out));
    out->size = size;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(ctx->device, &bci, NULL, &out->buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx->device, out->buffer, &req);

    uint32_t mem_idx = vk_find_memory_type(ctx, req.memoryTypeBits, mem_props);
    if (mem_idx == UINT32_MAX) {
        vk_ctx_set_error(ctx, "no memory type matches buffer requirements (props=0x%x)", (unsigned)mem_props);
        return -1;
    }

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mem_idx,
    };
    VK_CHECK(vkAllocateMemory(ctx->device, &mai, NULL, &out->memory));
    VK_CHECK(vkBindBufferMemory(ctx->device, out->buffer, out->memory, 0));

    if (map_persistent) {
        VK_CHECK(vkMapMemory(ctx->device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->mapped));
    }
    return 0;
}

static void destroy_buffer(VkCtx *ctx, VkBufferM *b) {
    if (!b || !ctx)
        return;
    if (b->mapped && b->memory) {
        vkUnmapMemory(ctx->device, b->memory);
        b->mapped = NULL;
    }
    if (b->buffer) {
        vkDestroyBuffer(ctx->device, b->buffer, NULL);
        b->buffer = VK_NULL_HANDLE;
    }
    if (b->memory) {
        vkFreeMemory(ctx->device, b->memory, NULL);
        b->memory = VK_NULL_HANDLE;
    }
    b->size = 0;
}

/* ------------------------------ image helpers ------------------------------ */

static int create_image(VkCtx *ctx, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImageM *out) {
    memset(out, 0, sizeof(*out));
    out->width = w;
    out->height = h;
    out->format = format;

    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK_CHECK(vkCreateImage(ctx->device, &ici, NULL, &out->image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx->device, out->image, &req);

    uint32_t mem_idx = vk_find_memory_type(ctx, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_idx == UINT32_MAX) {
        vk_ctx_set_error(ctx, "no DEVICE_LOCAL memory type for color image");
        return -1;
    }

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mem_idx,
    };
    VK_CHECK(vkAllocateMemory(ctx->device, &mai, NULL, &out->memory));
    VK_CHECK(vkBindImageMemory(ctx->device, out->image, out->memory, 0));

    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = out->image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
    VK_CHECK(vkCreateImageView(ctx->device, &vci, NULL, &out->view));
    return 0;
}

static void destroy_image(VkCtx *ctx, VkImageM *im) {
    if (!im || !ctx)
        return;
    if (im->view) {
        vkDestroyImageView(ctx->device, im->view, NULL);
        im->view = VK_NULL_HANDLE;
    }
    if (im->image) {
        vkDestroyImage(ctx->device, im->image, NULL);
        im->image = VK_NULL_HANDLE;
    }
    if (im->memory) {
        vkFreeMemory(ctx->device, im->memory, NULL);
        im->memory = VK_NULL_HANDLE;
    }
}

/* --------------------------- render target helpers -------------------------- */

static int create_render_target(VkCtx *ctx, uint32_t w, uint32_t h, RenderTarget *rt) {
    memset(rt, 0, sizeof(*rt));
    int r = create_image(ctx, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &rt->color);
    if (r != 0)
        return r;

    /* Prefer HOST_CACHED so the CPU reads the readback at full RAM
     * bandwidth instead of going over uncached PCIe BAR (~250 MB/s on
     * NVIDIA, vs >5 GB/s cached). Fall back to plain HOST_COHERENT if
     * the device doesn't expose a cached host-visible memory type. */
    VkDeviceSize buf_size = (VkDeviceSize)w * (VkDeviceSize)h * 4;
    VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    r = create_buffer(ctx, buf_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, want, 1, &rt->readback);
    if (r != 0) {
        r = create_buffer(ctx, buf_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, &rt->readback);
        if (r != 0)
            return r;
    }
    rt->row_pitch_bytes = (size_t)w * 4;
    return 0;
}

static void destroy_render_target(VkCtx *ctx, RenderTarget *rt) {
    if (!rt)
        return;
    destroy_image(ctx, &rt->color);
    destroy_buffer(ctx, &rt->readback);
}

/* ------------------------------ static geometry ----------------------------- */

static int upload_static_geometry(VkCtx *ctx, Renderer *r) {
    /* Unit quad: 4 vec2 corners spanning [-1, +1]^2. Order: BL, BR, TR, TL. */
    const float quad[8] = {
        -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    };
    const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};

    int rc =
        create_buffer(ctx, sizeof(quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, &r->unit_quad_vb);
    if (rc != 0)
        return rc;
    memcpy(r->unit_quad_vb.mapped, quad, sizeof(quad));

    rc = create_buffer(ctx, sizeof(idx), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, &r->unit_quad_ib);
    if (rc != 0)
        return rc;
    memcpy(r->unit_quad_ib.mapped, idx, sizeof(idx));
    return 0;
}

/* --------------------------- frame slot lifecycle --------------------------- */

static int slot_init(VkCtx *ctx, FrameSlot *s, uint32_t width, uint32_t height) {
    memset(s, 0, sizeof(*s));
    int rc;
    if ((rc = create_render_target(ctx, width, height, &s->rt_topdown)) != 0)
        return rc;
    if ((rc = create_render_target(ctx, width, height, &s->rt_bev)) != 0)
        return rc;

    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &cai, &s->cmd));

    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = 0,
    };
    VK_CHECK(vkCreateFence(ctx->device, &fci, NULL, &s->fence));
    return 0;
}

static void slot_destroy(VkCtx *ctx, FrameSlot *s) {
    if (!s || !ctx)
        return;
    if (s->fence) {
        vkDestroyFence(ctx->device, s->fence, NULL);
        s->fence = VK_NULL_HANDLE;
    }
    if (s->cmd) {
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &s->cmd);
        s->cmd = VK_NULL_HANDLE;
    }
    destroy_render_target(ctx, &s->rt_topdown);
    destroy_render_target(ctx, &s->rt_bev);
    destroy_buffer(ctx, &s->agent_inst_vb);
}

/* --------------------------- init / destroy / set_roads --------------------- */

int vk_renderer_init(VkCtx *ctx, Pipelines *p, Renderer *r, uint32_t width, uint32_t height) {
    memset(r, 0, sizeof(*r));
    r->pipelines = p;
    r->width = width;
    r->height = height;

    int rc;
    if ((rc = upload_static_geometry(ctx, r)) != 0)
        goto fail;
    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        if ((rc = slot_init(ctx, &r->slots[i], width, height)) != 0)
            goto fail;
    }
    return 0;

fail:
    vk_renderer_destroy(ctx, r);
    return rc;
}

void vk_renderer_destroy(VkCtx *ctx, Renderer *r) {
    if (!r || !ctx)
        return;
    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        slot_destroy(ctx, &r->slots[i]);
    }
    destroy_buffer(ctx, &r->unit_quad_vb);
    destroy_buffer(ctx, &r->unit_quad_ib);
    destroy_buffer(ctx, &r->road_vb);
    free(r->road_offsets);
    r->road_offsets = NULL;
    free(r->road_types);
    r->road_types = NULL;
    r->num_polys = 0;
    r->road_meta_capacity = 0;
    r->road_vb_capacity = 0;
    r->head = r->tail = r->n_in_flight = 0;
}

static int ensure_buffer_capacity(VkCtx *ctx, VkBufferM *b, VkDeviceSize required, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags mem_props) {
    if (b->size >= required)
        return 0;
    destroy_buffer(ctx, b);
    VkDeviceSize cap = 256;
    while (cap < required)
        cap <<= 1;
    return create_buffer(ctx, cap, usage, mem_props, 1, b);
}

int vk_renderer_set_roads(VkCtx *ctx, Renderer *r, const float *road_xy, uint32_t num_verts,
                          const uint32_t *road_offsets, const uint32_t *road_types, uint32_t num_polys) {
    /* Before re-uploading the road buffer, make sure no slot is still
     * reading from it. The simplest correct path is to drain everything
     * pending — set_roads is called once per episode, before the loop,
     * so this is essentially free in steady state. */
    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        if (r->slots[i].pending) {
            vkWaitForFences(ctx->device, 1, &r->slots[i].fence, VK_TRUE, UINT64_MAX);
            vkResetFences(ctx->device, 1, &r->slots[i].fence);
            r->slots[i].pending = 0;
        }
    }
    r->head = r->tail = r->n_in_flight = 0;

    VkDeviceSize required = (VkDeviceSize)num_verts * sizeof(float) * 2;
    if (required == 0)
        required = sizeof(float) * 2;
    int rc = ensure_buffer_capacity(ctx, &r->road_vb, required, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (rc != 0)
        return rc;
    if (num_verts > 0) {
        memcpy(r->road_vb.mapped, road_xy, (size_t)num_verts * sizeof(float) * 2);
    }
    r->road_vb_capacity = num_verts;

    if (num_polys + 1 > r->road_meta_capacity) {
        free(r->road_offsets);
        free(r->road_types);
        r->road_meta_capacity = num_polys + 1;
        r->road_offsets = (uint32_t *)malloc(sizeof(uint32_t) * (num_polys + 1));
        r->road_types = (uint32_t *)malloc(sizeof(uint32_t) * num_polys);
        if (!r->road_offsets || !r->road_types) {
            vk_ctx_set_error(ctx, "out of host memory for road metadata");
            return -1;
        }
    }
    if (num_polys > 0) {
        memcpy(r->road_offsets, road_offsets, sizeof(uint32_t) * (num_polys + 1));
        memcpy(r->road_types, road_types, sizeof(uint32_t) * num_polys);
    }
    r->num_polys = num_polys;
    return 0;
}

/* ------------------------------- per-frame draw ----------------------------- */

static void color_for_road_type(uint32_t type, float out[4]) {
    out[3] = 1.0f;
    switch (type) {
    case 6: /* ROAD_EDGE */
        out[0] = 0.55f;
        out[1] = 0.55f;
        out[2] = 0.55f;
        break;
    case 4: /* ROAD_LANE */
        out[0] = 0.85f;
        out[1] = 0.78f;
        out[2] = 0.30f;
        out[3] = 0.6f;
        break;
    case 5: /* ROAD_LINE */
        out[0] = 0.95f;
        out[1] = 0.95f;
        out[2] = 0.95f;
        out[3] = 0.5f;
        break;
    case 10: /* DRIVEWAY */
        out[0] = 0.40f;
        out[1] = 0.40f;
        out[2] = 0.55f;
        out[3] = 0.7f;
        break;
    default:
        out[0] = 0.45f;
        out[1] = 0.45f;
        out[2] = 0.45f;
        break;
    }
}

static int ensure_slot_agent_capacity(VkCtx *ctx, FrameSlot *s, uint32_t num_instances) {
    VkDeviceSize required = (VkDeviceSize)num_instances * sizeof(AgentInstance);
    if (required == 0)
        required = sizeof(AgentInstance);
    int rc = ensure_buffer_capacity(ctx, &s->agent_inst_vb, required, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (rc != 0)
        return rc;
    s->agent_inst_capacity = num_instances;
    return 0;
}

static void barrier_image(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                          VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 imb = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
    VkDependencyInfo di = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &imb,
    };
    vkCmdPipelineBarrier2(cmd, &di);
}

/* Record one view's draws into the slot's command buffer. */
static void record_view(VkCommandBuffer cmd, Renderer *r, FrameSlot *slot, RenderTarget *rt, const Mat4 *mvp,
                        uint32_t num_instances) {
    barrier_image(cmd, rt->color.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkClearValue clear = {.color = {.float32 = {0.05f, 0.05f, 0.08f, 1.0f}}};
    VkRenderingAttachmentInfo att = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = rt->color.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear,
    };
    VkRenderingInfo ri = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = {r->width, r->height}},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &att,
    };
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)r->width,
        .height = (float)r->height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    VkRect2D sc = {.offset = {0, 0}, .extent = {r->width, r->height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    if (r->num_polys > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipelines->line_pipeline);
        VkDeviceSize voff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &r->road_vb.buffer, &voff);

        PushConstants pc;
        memcpy(pc.mvp, mvp->m, sizeof(pc.mvp));

        for (uint32_t i = 0; i < r->num_polys; ++i) {
            uint32_t start = r->road_offsets[i];
            uint32_t end = r->road_offsets[i + 1];
            if (end <= start + 1)
                continue;

            color_for_road_type(r->road_types[i], pc.color);
            vkCmdPushConstants(cmd, r->pipelines->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(pc), &pc);
            vkCmdDraw(cmd, end - start, 1, start, 0);
        }
    }

    if (num_instances > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipelines->box_pipeline);
        VkBuffer vbufs[2] = {r->unit_quad_vb.buffer, slot->agent_inst_vb.buffer};
        VkDeviceSize voffs[2] = {0, 0};
        vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffs);
        vkCmdBindIndexBuffer(cmd, r->unit_quad_ib.buffer, 0, VK_INDEX_TYPE_UINT16);

        PushConstants pc;
        memcpy(pc.mvp, mvp->m, sizeof(pc.mvp));
        pc.color[0] = pc.color[1] = pc.color[2] = pc.color[3] = 1.0f;
        vkCmdPushConstants(cmd, r->pipelines->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pc), &pc);

        vkCmdDrawIndexed(cmd, 6, num_instances, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);

    barrier_image(cmd, rt->color.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {r->width, r->height, 1},
    };
    vkCmdCopyImageToBuffer(cmd, rt->color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rt->readback.buffer, 1, &region);
}

/* ------------------------------ episode lifecycle ------------------------------ */

void vk_renderer_episode_begin(Renderer *r, FfmpegPipe *pipe_topdown, FfmpegPipe *pipe_bev) {
    r->ep_pipe_topdown = pipe_topdown;
    r->ep_pipe_bev = pipe_bev;
    /* head/tail/n_in_flight were already reset by set_roads, but be
     * defensive in case episode_begin is called without it. */
    r->head = r->tail = r->n_in_flight = 0;
    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        r->slots[i].pending = 0;
        r->slots[i].rendered_topdown = 0;
        r->slots[i].rendered_bev = 0;
    }
}

/* Wait on the slot at head, fwrite its readback buffers to ffmpeg, and
 * advance head. Returns 0 on success, non-zero on ffmpeg failure. */
static int drain_head(VkCtx *ctx, Renderer *r) {
    FrameSlot *s = &r->slots[r->head];
    if (!s->pending) {
        /* Defensive: shouldn't happen if n_in_flight is accurate. */
        return 0;
    }
    VK_CHECK(vkWaitForFences(ctx->device, 1, &s->fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(ctx->device, 1, &s->fence));

    if (s->rendered_topdown && r->ep_pipe_topdown) {
        if (ffmpeg_pipe_write_frame(r->ep_pipe_topdown, s->rt_topdown.readback.mapped) != 0) {
            vk_ctx_set_error(ctx, "ffmpeg write failed (top-down) at slot %d", r->head);
            return -1;
        }
    }
    if (s->rendered_bev && r->ep_pipe_bev) {
        if (ffmpeg_pipe_write_frame(r->ep_pipe_bev, s->rt_bev.readback.mapped) != 0) {
            vk_ctx_set_error(ctx, "ffmpeg write failed (bev) at slot %d", r->head);
            return -1;
        }
    }

    s->pending = 0;
    s->rendered_topdown = 0;
    s->rendered_bev = 0;
    r->head = (r->head + 1) % FRAMES_IN_FLIGHT;
    r->n_in_flight--;
    return 0;
}

int vk_renderer_submit_frame(VkCtx *ctx, Renderer *r, const AgentInstance *instances, uint32_t num_instances,
                             const Mat4 *mvp_topdown, const Mat4 *mvp_bev) {
    /* If the ring is full, drain the oldest before reusing its slot. */
    if (r->n_in_flight == FRAMES_IN_FLIGHT) {
        int rc = drain_head(ctx, r);
        if (rc != 0)
            return rc;
    }

    FrameSlot *s = &r->slots[r->tail];

    /* Upload agent instances into THIS slot's buffer (not a shared one),
     * so the GPU executing the previous frame on a different slot is not
     * disturbed. */
    if (num_instances > 0) {
        int rc = ensure_slot_agent_capacity(ctx, s, num_instances);
        if (rc != 0)
            return rc;
        memcpy(s->agent_inst_vb.mapped, instances, (size_t)num_instances * sizeof(AgentInstance));
    }

    VK_CHECK(vkResetCommandBuffer(s->cmd, 0));
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(s->cmd, &bi));

    if (mvp_topdown) {
        record_view(s->cmd, r, s, &s->rt_topdown, mvp_topdown, num_instances);
        s->rendered_topdown = 1;
    }
    if (mvp_bev) {
        record_view(s->cmd, r, s, &s->rt_bev, mvp_bev, num_instances);
        s->rendered_bev = 1;
    }

    /* Memory barrier so the host can safely read the readback buffers
     * once the fence signals. */
    VkMemoryBarrier2 mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
    };
    VkDependencyInfo di = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &mb,
    };
    vkCmdPipelineBarrier2(s->cmd, &di);

    VK_CHECK(vkEndCommandBuffer(s->cmd));

    VkCommandBufferSubmitInfo csi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = s->cmd,
    };
    VkSubmitInfo2 si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &csi,
    };
    VK_CHECK(vkQueueSubmit2(ctx->graphics_queue, 1, &si, s->fence));

    s->pending = 1;
    r->tail = (r->tail + 1) % FRAMES_IN_FLIGHT;
    r->n_in_flight++;
    return 0;
}

int vk_renderer_episode_end(VkCtx *ctx, Renderer *r) {
    while (r->n_in_flight > 0) {
        int rc = drain_head(ctx, r);
        if (rc != 0)
            return rc;
    }
    r->ep_pipe_topdown = NULL;
    r->ep_pipe_bev = NULL;
    return 0;
}
