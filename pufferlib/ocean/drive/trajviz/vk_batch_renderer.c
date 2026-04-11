/*
 * vk_batch_renderer.c — multi-episode batched renderer.
 *
 * Tiles N episodes into a single atlas per view, draws them all in one
 * command-buffer recording, submits once per frame. The dominant per-
 * frame Vulkan + ffmpeg overhead in the single-episode path was per-
 * submit / per-pipe latency, so this should drop per-episode wall time
 * by roughly 1/N as long as the GPU isn't already saturated.
 *
 * Buffer + image helper plumbing is duplicated from vk_renderer.c to
 * keep this module self-contained — they're small (~50 lines apiece),
 * have stable signatures, and the alternative is plumbing yet another
 * shared header. The duplicates can be merged later if either side
 * grows complex.
 */

#include "vk_batch_renderer.h"
#include "shaders.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ----------------------------- buffer helpers ----------------------------- */

static int br_create_buffer(VkCtx *ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props,
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

static void br_destroy_buffer(VkCtx *ctx, VkBufferM *b) {
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

static int br_ensure_buffer_capacity(VkCtx *ctx, VkBufferM *b, VkDeviceSize required, VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags mem_props) {
    if (b->size >= required)
        return 0;
    br_destroy_buffer(ctx, b);
    VkDeviceSize cap = 256;
    while (cap < required)
        cap <<= 1;
    return br_create_buffer(ctx, cap, usage, mem_props, 1, b);
}

/* ------------------------------ image helpers ------------------------------ */

static int br_create_image(VkCtx *ctx, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
                           VkImageM *out) {
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
        vk_ctx_set_error(ctx, "no DEVICE_LOCAL memory type for atlas image");
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

static void br_destroy_image(VkCtx *ctx, VkImageM *im) {
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

/* --------------------------- color lookup (shared) -------------------------- */

static void color_for_road_type(uint32_t type, float out[4]) {
    out[3] = 1.0f;
    switch (type) {
    case 6:
        out[0] = 0.55f;
        out[1] = 0.55f;
        out[2] = 0.55f;
        break;
    case 4:
        out[0] = 0.85f;
        out[1] = 0.78f;
        out[2] = 0.30f;
        out[3] = 0.6f;
        break;
    case 5:
        out[0] = 0.95f;
        out[1] = 0.95f;
        out[2] = 0.95f;
        out[3] = 0.5f;
        break;
    case 10:
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

/* --------------------------- init / destroy --------------------------- */

static int upload_static_quad(VkCtx *ctx, BatchRenderer *br) {
    const float quad[8] = {-1, -1, 1, -1, 1, 1, -1, 1};
    const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
    int rc;

    rc = br_create_buffer(ctx, sizeof(quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1,
                          &br->unit_quad_vb);
    if (rc != 0)
        return rc;
    memcpy(br->unit_quad_vb.mapped, quad, sizeof(quad));

    rc = br_create_buffer(ctx, sizeof(idx), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1,
                          &br->unit_quad_ib);
    if (rc != 0)
        return rc;
    memcpy(br->unit_quad_ib.mapped, idx, sizeof(idx));
    return 0;
}

int vk_batch_renderer_init(VkCtx *ctx, Pipelines *p, BatchRenderer *br, int batch_n, uint32_t tile_w, uint32_t tile_h) {
    if (batch_n <= 0 || tile_w == 0 || tile_h == 0) {
        vk_ctx_set_error(ctx, "vk_batch_renderer_init: invalid args (n=%d w=%u h=%u)", batch_n, tile_w, tile_h);
        return -1;
    }

    memset(br, 0, sizeof(*br));
    br->pipelines = p;
    br->batch_n = batch_n;
    br->tile_w = tile_w;
    br->tile_h = tile_h;

    br->slots = (BatchSlot *)calloc((size_t)batch_n, sizeof(BatchSlot));
    if (!br->slots) {
        vk_ctx_set_error(ctx, "out of memory allocating %d batch slots", batch_n);
        return -1;
    }

    int rc;
    if ((rc = upload_static_quad(ctx, br)) != 0)
        goto fail;

    /* Atlas dimensions: tile_w wide, batch_n * tile_h tall (vertical
     * stacking → contiguous tile bytes in the readback buffer). Vulkan's
     * maxImageDimension2D is at least 4096 on every spec-compliant device
     * and typically 16384+ on real GPUs, so batch_n up to ~22 at 720p
     * height is safe before we need to fall back to multiple passes. */
    uint32_t atlas_h = (uint32_t)batch_n * tile_h;

    if ((rc = br_create_image(ctx, tile_w, atlas_h, VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              &br->atlas_topdown)) != 0)
        goto fail;
    if ((rc = br_create_image(ctx, tile_w, atlas_h, VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &br->atlas_bev)) !=
        0)
        goto fail;

    /* Readback buffers: prefer HOST_CACHED so the CPU can read them at
     * full RAM bandwidth. The default HOST_VISIBLE+HOST_COHERENT path
     * on NVIDIA picks a write-combined memory type — fast for GPU
     * writes but ~250 MB/s for CPU reads (uncached PCIe BAR), which is
     * by far the dominant cost when piping frames to ffmpeg. With
     * HOST_CACHED, reads hit RAM at >5 GB/s. */
    VkDeviceSize readback_size = (VkDeviceSize)tile_w * (VkDeviceSize)atlas_h * 4;
    VkMemoryPropertyFlags readback_props =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    if ((rc = br_create_buffer(ctx, readback_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback_props, 1,
                               &br->readback_topdown)) != 0) {
        /* Fall back without HOST_CACHED if the device doesn't expose it. */
        readback_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((rc = br_create_buffer(ctx, readback_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback_props, 1,
                                   &br->readback_topdown)) != 0)
            goto fail;
    }
    if ((rc = br_create_buffer(ctx, readback_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback_props, 1,
                               &br->readback_bev)) != 0)
        goto fail;

    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &cai, &br->cmd));

    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = 0,
    };
    VK_CHECK(vkCreateFence(ctx->device, &fci, NULL, &br->fence));

    return 0;

fail:
    vk_batch_renderer_destroy(ctx, br);
    return rc;
}

void vk_batch_renderer_destroy(VkCtx *ctx, BatchRenderer *br) {
    if (!br || !ctx)
        return;
    if (br->fence) {
        vkDestroyFence(ctx->device, br->fence, NULL);
        br->fence = VK_NULL_HANDLE;
    }
    if (br->cmd) {
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &br->cmd);
        br->cmd = VK_NULL_HANDLE;
    }
    br_destroy_image(ctx, &br->atlas_topdown);
    br_destroy_image(ctx, &br->atlas_bev);
    br_destroy_buffer(ctx, &br->readback_topdown);
    br_destroy_buffer(ctx, &br->readback_bev);
    br_destroy_buffer(ctx, &br->unit_quad_vb);
    br_destroy_buffer(ctx, &br->unit_quad_ib);
    if (br->slots) {
        for (int i = 0; i < br->batch_n; ++i) {
            br_destroy_buffer(ctx, &br->slots[i].road_vb);
            br_destroy_buffer(ctx, &br->slots[i].agent_inst_vb);
            free(br->slots[i].road_offsets);
            free(br->slots[i].road_types);
        }
        free(br->slots);
        br->slots = NULL;
    }
    br->batch_n = 0;
}

/* --------------------------- per-slot configuration --------------------------- */

int vk_batch_renderer_set_episode(VkCtx *ctx, BatchRenderer *br, int slot, const float *road_xy, uint32_t num_verts,
                                  const uint32_t *road_offsets, const uint32_t *road_types, uint32_t num_polys,
                                  FfmpegPipe *pipe_topdown, FfmpegPipe *pipe_bev) {
    if (slot < 0 || slot >= br->batch_n) {
        vk_ctx_set_error(ctx, "set_episode: slot %d out of range [0, %d)", slot, br->batch_n);
        return -1;
    }
    BatchSlot *s = &br->slots[slot];

    /* Resize road vb if needed and upload. */
    VkDeviceSize required = (VkDeviceSize)num_verts * sizeof(float) * 2;
    if (required == 0)
        required = sizeof(float) * 2;
    int rc = br_ensure_buffer_capacity(ctx, &s->road_vb, required, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (rc != 0)
        return rc;
    if (num_verts > 0) {
        memcpy(s->road_vb.mapped, road_xy, (size_t)num_verts * sizeof(float) * 2);
    }
    s->road_vb_capacity = num_verts;

    /* Host-side metadata copy. */
    if (num_polys + 1 > s->road_meta_capacity) {
        free(s->road_offsets);
        free(s->road_types);
        s->road_meta_capacity = num_polys + 1;
        s->road_offsets = (uint32_t *)malloc(sizeof(uint32_t) * (num_polys + 1));
        s->road_types = (uint32_t *)malloc(sizeof(uint32_t) * num_polys);
        if (!s->road_offsets || !s->road_types) {
            vk_ctx_set_error(ctx, "out of host memory for road metadata in slot %d", slot);
            return -1;
        }
    }
    if (num_polys > 0) {
        memcpy(s->road_offsets, road_offsets, sizeof(uint32_t) * (num_polys + 1));
        memcpy(s->road_types, road_types, sizeof(uint32_t) * num_polys);
    }
    s->num_polys = num_polys;

    s->pipe_topdown = pipe_topdown;
    s->pipe_bev = pipe_bev;
    s->active = 1;
    s->current_n_instances = 0;
    s->has_topdown_this_frame = 0;
    s->has_bev_this_frame = 0;
    return 0;
}

int vk_batch_renderer_set_frame(VkCtx *ctx, BatchRenderer *br, int slot, const AgentInstance *instances,
                                uint32_t num_instances, const Mat4 *mvp_topdown, const Mat4 *mvp_bev) {
    if (slot < 0 || slot >= br->batch_n) {
        vk_ctx_set_error(ctx, "set_frame: slot %d out of range", slot);
        return -1;
    }
    BatchSlot *s = &br->slots[slot];
    if (!s->active)
        return 0;

    if (num_instances > 0) {
        VkDeviceSize required = (VkDeviceSize)num_instances * sizeof(AgentInstance);
        int rc = br_ensure_buffer_capacity(ctx, &s->agent_inst_vb, required, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (rc != 0)
            return rc;
        memcpy(s->agent_inst_vb.mapped, instances, (size_t)num_instances * sizeof(AgentInstance));
        s->agent_inst_capacity = num_instances;
    }
    s->current_n_instances = num_instances;

    s->has_topdown_this_frame = (mvp_topdown != NULL) && (s->pipe_topdown != NULL);
    s->has_bev_this_frame = (mvp_bev != NULL) && (s->pipe_bev != NULL);
    if (s->has_topdown_this_frame)
        s->mvp_topdown = *mvp_topdown;
    if (s->has_bev_this_frame)
        s->mvp_bev = *mvp_bev;
    return 0;
}

void vk_batch_renderer_close_episode(BatchRenderer *br, int slot) {
    if (slot < 0 || slot >= br->batch_n)
        return;
    BatchSlot *s = &br->slots[slot];
    s->active = 0;
    s->pipe_topdown = NULL;
    s->pipe_bev = NULL;
    s->has_topdown_this_frame = 0;
    s->has_bev_this_frame = 0;
}

/* --------------------------- per-frame submit --------------------------- */

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

/* Record one full atlas pass: render every active slot's tile, then
 * copy the atlas image to its host-visible readback buffer. */
static void record_atlas_pass(VkCommandBuffer cmd, BatchRenderer *br, VkImage atlas_image, VkImageView atlas_view,
                              VkBuffer readback_buffer, int is_bev_view) {
    barrier_image(cmd, atlas_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkClearValue clear = {.color = {.float32 = {0.05f, 0.05f, 0.08f, 1.0f}}};
    VkRenderingAttachmentInfo att = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = atlas_view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear,
    };
    uint32_t atlas_h = (uint32_t)br->batch_n * br->tile_h;
    VkRenderingInfo ri = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = {br->tile_w, atlas_h}},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &att,
    };
    vkCmdBeginRendering(cmd, &ri);

    /* Render each active slot into its tile. */
    for (int i = 0; i < br->batch_n; ++i) {
        BatchSlot *s = &br->slots[i];
        if (!s->active)
            continue;

        int has_view = is_bev_view ? s->has_bev_this_frame : s->has_topdown_this_frame;
        if (!has_view)
            continue;

        /* Tile rect: full width, slice [i*tile_h, (i+1)*tile_h) vertically. */
        VkViewport vp = {
            .x = 0.0f,
            .y = (float)(i * br->tile_h),
            .width = (float)br->tile_w,
            .height = (float)br->tile_h,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        VkRect2D sc = {
            .offset = {0, (int32_t)(i * br->tile_h)},
            .extent = {br->tile_w, br->tile_h},
        };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        const Mat4 *mvp = is_bev_view ? &s->mvp_bev : &s->mvp_topdown;

        /* Road polylines for this slot */
        if (s->num_polys > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, br->pipelines->line_pipeline);
            VkDeviceSize voff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &s->road_vb.buffer, &voff);

            PushConstants pc;
            memcpy(pc.mvp, mvp->m, sizeof(pc.mvp));

            for (uint32_t j = 0; j < s->num_polys; ++j) {
                uint32_t start = s->road_offsets[j];
                uint32_t end = s->road_offsets[j + 1];
                if (end <= start + 1)
                    continue;

                color_for_road_type(s->road_types[j], pc.color);
                vkCmdPushConstants(cmd, br->pipelines->layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                vkCmdDraw(cmd, end - start, 1, start, 0);
            }
        }

        /* Agent boxes for this slot */
        if (s->current_n_instances > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, br->pipelines->box_pipeline);
            VkBuffer vbufs[2] = {br->unit_quad_vb.buffer, s->agent_inst_vb.buffer};
            VkDeviceSize voffs[2] = {0, 0};
            vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffs);
            vkCmdBindIndexBuffer(cmd, br->unit_quad_ib.buffer, 0, VK_INDEX_TYPE_UINT16);

            PushConstants pc;
            memcpy(pc.mvp, mvp->m, sizeof(pc.mvp));
            pc.color[0] = pc.color[1] = pc.color[2] = pc.color[3] = 1.0f;
            vkCmdPushConstants(cmd, br->pipelines->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(pc), &pc);

            vkCmdDrawIndexed(cmd, 6, s->current_n_instances, 0, 0, 0);
        }
    }

    vkCmdEndRendering(cmd);

    /* Atlas → TRANSFER_SRC, copy to readback. */
    barrier_image(cmd, atlas_image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
        .imageExtent = {br->tile_w, atlas_h, 1},
    };
    vkCmdCopyImageToBuffer(cmd, atlas_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_buffer, 1, &region);
}

int vk_batch_renderer_submit_frame(VkCtx *ctx, BatchRenderer *br) {
    /* Quick exit: nothing to render this frame. */
    int any_topdown = 0, any_bev = 0;
    for (int i = 0; i < br->batch_n; ++i) {
        if (!br->slots[i].active)
            continue;
        if (br->slots[i].has_topdown_this_frame)
            any_topdown = 1;
        if (br->slots[i].has_bev_this_frame)
            any_bev = 1;
    }
    if (!any_topdown && !any_bev)
        return 0;

    VK_CHECK(vkResetCommandBuffer(br->cmd, 0));
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(br->cmd, &bi));

    if (any_topdown) {
        record_atlas_pass(br->cmd, br, br->atlas_topdown.image, br->atlas_topdown.view, br->readback_topdown.buffer,
                          /*is_bev_view=*/0);
    }
    if (any_bev) {
        record_atlas_pass(br->cmd, br, br->atlas_bev.image, br->atlas_bev.view, br->readback_bev.buffer,
                          /*is_bev_view=*/1);
    }

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
    vkCmdPipelineBarrier2(br->cmd, &di);

    VK_CHECK(vkEndCommandBuffer(br->cmd));

    VkCommandBufferSubmitInfo csi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = br->cmd,
    };
    VkSubmitInfo2 si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &csi,
    };
    VK_CHECK(vkQueueSubmit2(ctx->graphics_queue, 1, &si, br->fence));
    VK_CHECK(vkWaitForFences(ctx->device, 1, &br->fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(ctx->device, 1, &br->fence));

    /* Fan out each slot's tile to its ffmpeg pipes' writer threads in
     * parallel, then wait for all of them. Each pipe has its own
     * background thread (see ffmpeg_pipe.c), so the wall time of this
     * phase is max(single fwrite) instead of sum(fwrites) — which is
     * a ~Nx win for batch_size N when the per-slot write is the
     * dominant cost.
     *
     * Tile bytes are row-contiguous in the readback buffer thanks to
     * vertical stacking, so each tile is one (tile_w * tile_h * 4)-
     * byte slab at offset (i * tile_bytes). */
    size_t tile_bytes = (size_t)br->tile_w * (size_t)br->tile_h * 4;

    /* Phase 1: submit all writes (returns immediately for each pipe). */
    for (int i = 0; i < br->batch_n; ++i) {
        BatchSlot *s = &br->slots[i];
        if (!s->active)
            continue;

        if (s->has_topdown_this_frame && s->pipe_topdown) {
            const uint8_t *p = (const uint8_t *)br->readback_topdown.mapped + (size_t)i * tile_bytes;
            ffmpeg_pipe_submit_frame(s->pipe_topdown, p);
        }
        if (s->has_bev_this_frame && s->pipe_bev) {
            const uint8_t *p = (const uint8_t *)br->readback_bev.mapped + (size_t)i * tile_bytes;
            ffmpeg_pipe_submit_frame(s->pipe_bev, p);
        }
    }

    /* Phase 2: wait for all writes to complete. The readback buffer is
     * about to be reused for the next frame's render so we cannot
     * proceed until every writer has consumed its tile. */
    int err = 0;
    for (int i = 0; i < br->batch_n; ++i) {
        BatchSlot *s = &br->slots[i];
        if (!s->active)
            continue;

        if (s->has_topdown_this_frame && s->pipe_topdown) {
            if (ffmpeg_pipe_wait(s->pipe_topdown) != 0) {
                vk_ctx_set_error(ctx, "ffmpeg topdown write failed at slot %d", i);
                err = -1;
            }
        }
        if (s->has_bev_this_frame && s->pipe_bev) {
            if (ffmpeg_pipe_wait(s->pipe_bev) != 0) {
                vk_ctx_set_error(ctx, "ffmpeg bev write failed at slot %d", i);
                err = -1;
            }
        }
        s->has_topdown_this_frame = 0;
        s->has_bev_this_frame = 0;
    }

    return err;
}
