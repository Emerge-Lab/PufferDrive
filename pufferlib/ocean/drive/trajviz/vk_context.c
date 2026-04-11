/*
 * vk_context.c — Vulkan instance, device, and queue setup for trajviz.
 *
 * The init order is:
 *   1. vkCreateInstance (request 1.3, optional debug utils)
 *   2. enumerate physical devices, prefer discrete GPU
 *   3. find a graphics queue family
 *   4. vkCreateDevice with dynamic_rendering + synchronization2 enabled
 *      (both core in 1.3, but the feature struct still has to be in the
 *       create-info chain)
 *   5. vkGetDeviceQueue
 *   6. vkCreateCommandPool with RESET_COMMAND_BUFFER (per-frame command
 *      buffers are short-lived; resetting the whole pool is cleaner than
 *      individual buffer resets)
 *
 * Cleanup is reverse order. vk_ctx_destroy can be called on a partially-
 * initialized context (init failure path) — every handle is checked for
 * VK_NULL_HANDLE before destruction.
 */

#include "vk_context.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vk_ctx_set_error(VkCtx *ctx, const char *fmt, ...) {
    if (!ctx)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->last_error, TRAJVIZ_ERROR_BUF, fmt, ap);
    va_end(ap);
}

uint32_t vk_find_memory_type(const VkCtx *ctx, uint32_t type_bits, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < ctx->mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (ctx->mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

#ifdef TRAJVIZ_DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                               VkDebugUtilsMessageTypeFlagsEXT type,
                                               const VkDebugUtilsMessengerCallbackDataEXT *data, void *user_data) {
    (void)type;
    (void)user_data;
    /* Only print warnings and errors — info/verbose are too noisy. */
    if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)) {
        fprintf(stderr, "[vk] %s\n", data->pMessage);
    }
    return VK_FALSE;
}
#endif

static int create_instance(VkCtx *ctx) {
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "trajviz",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "trajviz",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    const char *exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    uint32_t num_layers = 0;
    uint32_t num_exts = 0;

#ifdef TRAJVIZ_DEBUG
    /* Verify the validation layer is actually available; otherwise the
     * instance creation fails outright instead of degrading gracefully. */
    uint32_t avail_layer_count = 0;
    vkEnumerateInstanceLayerProperties(&avail_layer_count, NULL);
    VkLayerProperties *avail_layers = calloc(avail_layer_count, sizeof(*avail_layers));
    vkEnumerateInstanceLayerProperties(&avail_layer_count, avail_layers);
    for (uint32_t i = 0; i < avail_layer_count; ++i) {
        if (strcmp(avail_layers[i].layerName, layers[0]) == 0) {
            num_layers = 1;
            num_exts = 1;
            ctx->debug_enabled = 1;
            break;
        }
    }
    free(avail_layers);
    if (!num_layers) {
        fprintf(stderr, "[trajviz] validation layer not available, continuing without\n");
    }
#endif

    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledLayerCount = num_layers,
        .ppEnabledLayerNames = layers,
        .enabledExtensionCount = num_exts,
        .ppEnabledExtensionNames = exts,
    };
    VK_CHECK(vkCreateInstance(&ci, NULL, &ctx->instance));

#ifdef TRAJVIZ_DEBUG
    if (ctx->debug_enabled) {
        VkDebugUtilsMessengerCreateInfoEXT dci = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_cb,
        };
        PFN_vkCreateDebugUtilsMessengerEXT create_msgr =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(ctx->instance, "vkCreateDebugUtilsMessengerEXT");
        if (create_msgr) {
            create_msgr(ctx->instance, &dci, NULL, &ctx->debug_messenger);
        }
    }
#endif

    return 0;
}

static int pick_physical_device(VkCtx *ctx) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &count, NULL);
    if (count == 0) {
        vk_ctx_set_error(ctx, "no Vulkan-capable physical device found");
        return -1;
    }
    VkPhysicalDevice *devs = calloc(count, sizeof(*devs));
    vkEnumeratePhysicalDevices(ctx->instance, &count, devs);

    /* Prefer the first discrete GPU; fall back to the first device of any
     * type. This handles the typical "RTX + iGPU" workstation case where
     * we want the discrete card, and the "headless server" case where the
     * only device might be lavapipe (CPU rasterizer) or a virtio GPU. */
    VkPhysicalDevice picked = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            picked = devs[i];
            break;
        }
    }
    if (picked == VK_NULL_HANDLE) {
        picked = devs[0];
    }

    ctx->physical_device = picked;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(picked, &props);
    snprintf(ctx->device_name, sizeof(ctx->device_name), "%s", props.deviceName);
    vkGetPhysicalDeviceMemoryProperties(picked, &ctx->mem_props);

    free(devs);
    return 0;
}

static int find_graphics_queue(VkCtx *ctx) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &count, NULL);
    VkQueueFamilyProperties *fams = calloc(count, sizeof(*fams));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &count, fams);

    uint32_t found = UINT32_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            found = i;
            break;
        }
    }
    free(fams);
    if (found == UINT32_MAX) {
        vk_ctx_set_error(ctx, "no graphics queue family on device %s", ctx->device_name);
        return -1;
    }
    ctx->graphics_family = found;
    return 0;
}

static int create_device(VkCtx *ctx) {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->graphics_family,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };

    /* Vulkan 1.3 features struct chain. We need:
     *   - dynamicRendering: lets us draw without setting up VkRenderPass /
     *     VkFramebuffer objects. Cleaner code, no behavioral difference.
     *   - synchronization2: nicer image-barrier API (single struct, no
     *     dst-stage-mask juggling). Worth the one extra line.
     * Both are core in 1.3 but you still have to flip the bits in the
     * features chain to use them. */
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .dynamicRendering = VK_TRUE,
        .synchronization2 = VK_TRUE,
    };

    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VK_CHECK(vkCreateDevice(ctx->physical_device, &dci, NULL, &ctx->device));
    vkGetDeviceQueue(ctx->device, ctx->graphics_family, 0, &ctx->graphics_queue);
    return 0;
}

static int create_command_pool(VkCtx *ctx) {
    VkCommandPoolCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->graphics_family,
    };
    VK_CHECK(vkCreateCommandPool(ctx->device, &ci, NULL, &ctx->command_pool));
    return 0;
}

int vk_ctx_init(VkCtx *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    int r;
    if ((r = create_instance(ctx)) != 0)
        goto fail;
    if ((r = pick_physical_device(ctx)) != 0)
        goto fail;
    if ((r = find_graphics_queue(ctx)) != 0)
        goto fail;
    if ((r = create_device(ctx)) != 0)
        goto fail;
    if ((r = create_command_pool(ctx)) != 0)
        goto fail;
    return 0;
fail:
    vk_ctx_destroy(ctx);
    return r;
}

void vk_ctx_destroy(VkCtx *ctx) {
    if (!ctx)
        return;
    if (ctx->command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
        ctx->command_pool = VK_NULL_HANDLE;
    }
    if (ctx->device != VK_NULL_HANDLE) {
        vkDestroyDevice(ctx->device, NULL);
        ctx->device = VK_NULL_HANDLE;
    }
#ifdef TRAJVIZ_DEBUG
    if (ctx->debug_messenger != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugUtilsMessengerEXT destroy_msgr = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            ctx->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy_msgr) {
            destroy_msgr(ctx->instance, ctx->debug_messenger, NULL);
        }
        ctx->debug_messenger = VK_NULL_HANDLE;
    }
#endif
    if (ctx->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(ctx->instance, NULL);
        ctx->instance = VK_NULL_HANDLE;
    }
}
