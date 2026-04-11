/*
 * vk_context.h — Vulkan instance/device/queue lifecycle for trajviz.
 *
 * Pure headless: no surface, no swapchain, no window system. We render
 * into VkImages allocated from device memory and copy them back to host
 * buffers for ffmpeg encoding. This means trajviz works on a cluster
 * node with no display server, which is half the point of using Vulkan
 * over raylib.
 *
 * The VkCtx struct holds everything that lives for the entire renderer
 * lifetime — instance, device, queue, command pool, debug messenger.
 * Per-episode state (images, framebuffers, ffmpeg pipes) lives in
 * vk_renderer.h's RenderTargets, not here.
 *
 * One ctx per thread. The Vulkan spec allows concurrent submits to a
 * queue from multiple threads with external synchronization, but we
 * don't need that for v1 — single-threaded inside an episode, multi-
 * episode parallelism happens at a higher level (one ctx per worker).
 */

#ifndef VK_CONTEXT_H
#define VK_CONTEXT_H

#include <vulkan/vulkan.h>
#include <stdint.h>

#define TRAJVIZ_ERROR_BUF 512

typedef struct VkCtx {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t graphics_family;
    VkQueue graphics_queue;
    VkCommandPool command_pool;

    /* Cached physical device properties used by other modules. */
    VkPhysicalDeviceMemoryProperties mem_props;
    char device_name[256];

    /* Optional validation messenger; only created in debug builds. */
    VkDebugUtilsMessengerEXT debug_messenger;
    int debug_enabled;

    /* Last error message — populated by failing functions for the
     * caller to surface via trajviz_last_error(). */
    char last_error[TRAJVIZ_ERROR_BUF];
} VkCtx;

/* Initialize the Vulkan context. Returns 0 on success, non-zero on
 * failure. On failure, the last_error field contains a human-readable
 * message and any partially-created handles have been destroyed.
 *
 * If TRAJVIZ_DEBUG is defined at compile time, validation layers are
 * enabled and a debug messenger is registered. */
int vk_ctx_init(VkCtx *ctx);

/* Destroy the context. Idempotent. Safe to call after a failed init. */
void vk_ctx_destroy(VkCtx *ctx);

/* Find a memory type index that matches both the type bits (from a
 * VkMemoryRequirements query) and the requested property flags. Returns
 * UINT32_MAX if none. */
uint32_t vk_find_memory_type(const VkCtx *ctx, uint32_t type_bits, VkMemoryPropertyFlags properties);

/* Helper to set ctx->last_error from a printf-style message. Used by
 * vk_ctx and other modules. */
void vk_ctx_set_error(VkCtx *ctx, const char *fmt, ...);

/* VK_CHECK is the verbose-but-correct error path. On failure it sets
 * ctx->last_error and returns the result code from the enclosing
 * function. Use only inside functions that return int and have a VkCtx
 * *ctx in scope. */
#define VK_CHECK(expr)                                                                                                 \
    do {                                                                                                               \
        VkResult _r = (expr);                                                                                          \
        if (_r != VK_SUCCESS) {                                                                                        \
            vk_ctx_set_error(ctx, "%s failed at %s:%d (VkResult=%d)", #expr, __FILE__, __LINE__, (int)_r);             \
            return (int)_r;                                                                                            \
        }                                                                                                              \
    } while (0)

#endif /* VK_CONTEXT_H */
