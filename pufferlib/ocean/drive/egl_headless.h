// Headless EGL GPU rendering for PufferDrive.
//
// Strategy: InitWindow on Xvfb loads glad (GL function pointers) and initializes rlgl.
// Then we create an EGL context on the NVIDIA GPU and switch to it. The GL function
// pointers loaded by glad/GLFW remain valid (they're dlsym addresses, not context-specific).
// We re-init rlgl on the new GPU context so render batches and default textures are
// allocated on GPU memory. All subsequent rlgl/raylib draw calls execute on the GPU.
//
// Usage in make_client:
//   1. Fork Xvfb + InitWindow as normal (loads glad, creates rlgl state on Mesa)
//   2. Call egl_headless_init(width, height) to create GPU context
//   3. Call egl_switch_to_gpu() to activate GPU context
//   4. Call rlglClose() + rlglInit() to re-create rlgl state on GPU
//   5. Rendering now happens on GPU; rlReadScreenPixels still works

#ifndef EGL_HEADLESS_H
#define EGL_HEADLESS_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;
    int width;
    int height;
    int active;
} EGLHeadlessContext;

static EGLHeadlessContext g_egl_ctx = {0};

// Create an EGL context on an NVIDIA GPU. Does NOT make it current yet.
static int egl_headless_init(int width, int height) {
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT) {
        fprintf(stderr, "[egl_headless] EGL device extensions not available\n");
        return 0;
    }

    EGLDeviceEXT devices[8];
    EGLint numDevices = 0;
    eglQueryDevicesEXT(8, devices, &numDevices);
    if (numDevices == 0) {
        fprintf(stderr, "[egl_headless] No EGL devices found\n");
        return 0;
    }

    EGLDisplay display = EGL_NO_DISPLAY;
    for (int i = 0; i < numDevices; i++) {
        display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[i], NULL);
        if (display != EGL_NO_DISPLAY) {
            EGLint major, minor;
            if (eglInitialize(display, &major, &minor)) {
                const char *vendor = eglQueryString(display, EGL_VENDOR);
                if (vendor && strstr(vendor, "NVIDIA")) {
                    fprintf(stderr, "[egl_headless] Using NVIDIA EGL device %d (EGL %d.%d)\n", i, major, minor);
                    break;
                }
                eglTerminate(display);
                display = EGL_NO_DISPLAY;
            }
        }
    }

    if (display == EGL_NO_DISPLAY) {
        display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], NULL);
        EGLint major, minor;
        if (!eglInitialize(display, &major, &minor)) {
            fprintf(stderr, "[egl_headless] Failed to initialize any EGL device\n");
            return 0;
        }
        fprintf(stderr, "[egl_headless] Using fallback EGL device 0 (EGL %d.%d)\n", major, minor);
    }

    EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                              EGL_PBUFFER_BIT,
                              EGL_RENDERABLE_TYPE,
                              EGL_OPENGL_BIT,
                              EGL_RED_SIZE,
                              8,
                              EGL_GREEN_SIZE,
                              8,
                              EGL_BLUE_SIZE,
                              8,
                              EGL_ALPHA_SIZE,
                              8,
                              EGL_DEPTH_SIZE,
                              24,
                              EGL_NONE};
    EGLConfig config;
    EGLint numConfigs;
    eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);
    if (numConfigs == 0) {
        fprintf(stderr, "[egl_headless] No suitable EGL config\n");
        eglTerminate(display);
        return 0;
    }

    EGLint pbufferAttribs[] = {EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE};
    EGLSurface surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
    if (surface == EGL_NO_SURFACE) {
        fprintf(stderr, "[egl_headless] Failed to create pbuffer: 0x%x\n", eglGetError());
        eglTerminate(display);
        return 0;
    }

    eglBindAPI(EGL_OPENGL_API);
    EGLint contextAttribs[] = {EGL_CONTEXT_MAJOR_VERSION,
                               3,
                               EGL_CONTEXT_MINOR_VERSION,
                               3,
                               EGL_CONTEXT_OPENGL_PROFILE_MASK,
                               EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                               EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "[egl_headless] Failed to create GL context: 0x%x\n", eglGetError());
        eglDestroySurface(display, surface);
        eglTerminate(display);
        return 0;
    }

    g_egl_ctx.display = display;
    g_egl_ctx.context = context;
    g_egl_ctx.surface = surface;
    g_egl_ctx.width = width;
    g_egl_ctx.height = height;
    fprintf(stderr, "[egl_headless] GPU context created (%dx%d), ready to activate\n", width, height);
    return 1;
}

// Switch the current GL context from Xvfb/Mesa to the EGL/NVIDIA GPU.
// Call this AFTER InitWindow + rlglInit have loaded glad and initial state.
// We must first release the GLX context that GLFW/InitWindow created.
static int egl_switch_to_gpu(void) {
    // Release the GLX context by using glXMakeCurrent via dlsym
    // (we don't have glx.h here, so go through the dynamic linker)
    typedef int (*glXMakeCurrentFunc)(void *, unsigned long, void *);
    void *libgl = dlopen("libGL.so.1", RTLD_LAZY);
    if (libgl) {
        glXMakeCurrentFunc glXMC = (glXMakeCurrentFunc)dlsym(libgl, "glXMakeCurrent");
        // Get the current X display from the DISPLAY env var via XOpenDisplay
        typedef void *(*XOpenDisplayFunc)(const char *);
        void *libx11 = dlopen("libX11.so.6", RTLD_LAZY);
        if (libx11 && glXMC) {
            XOpenDisplayFunc XOD = (XOpenDisplayFunc)dlsym(libx11, "XOpenDisplay");
            if (XOD) {
                void *dpy = XOD(NULL);
                if (dpy) {
                    glXMC(dpy, 0, NULL); // Release GLX context on this thread
                    fprintf(stderr, "[egl_headless] Released GLX context\n");
                }
            }
        }
    }

    if (!eglMakeCurrent(g_egl_ctx.display, g_egl_ctx.surface, g_egl_ctx.surface, g_egl_ctx.context)) {
        fprintf(stderr, "[egl_headless] eglMakeCurrent failed: 0x%x\n", eglGetError());
        return 0;
    }
    g_egl_ctx.active = 1;
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    const char *version = (const char *)glGetString(GL_VERSION);
    fprintf(stderr, "[egl_headless] GPU active: %s (%s)\n", renderer ? renderer : "unknown", version ? version : "?");
    return 1;
}

static void egl_headless_cleanup(void) {
    if (g_egl_ctx.context) {
        eglMakeCurrent(g_egl_ctx.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(g_egl_ctx.display, g_egl_ctx.context);
    }
    if (g_egl_ctx.surface)
        eglDestroySurface(g_egl_ctx.display, g_egl_ctx.surface);
    if (g_egl_ctx.display)
        eglTerminate(g_egl_ctx.display);
    memset(&g_egl_ctx, 0, sizeof(g_egl_ctx));
}

#endif // EGL_HEADLESS_H
