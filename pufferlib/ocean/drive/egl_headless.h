// Headless EGL GPU rendering for PufferDrive.
//
// InitWindow on Xvfb loads raylib/glad and initializes rlgl. We then create an
// EGL OpenGL context on a GPU, switch to it, and reinitialize rlgl resources on
// that context. If EGL is unavailable, callers keep using the Xvfb/Mesa path.

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
    EGLConfig config;
    int width;
    int height;
    int active;
} EGLHeadlessContext;

static EGLHeadlessContext g_egl_ctx = {0};

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
        if (display == EGL_NO_DISPLAY)
            continue;

        EGLint major = 0, minor = 0;
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

    if (display == EGL_NO_DISPLAY) {
        display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], NULL);
        EGLint major = 0, minor = 0;
        if (!eglInitialize(display, &major, &minor)) {
            fprintf(stderr, "[egl_headless] Failed to initialize any EGL device\n");
            return 0;
        }
        fprintf(stderr, "[egl_headless] Using fallback EGL device 0 (EGL %d.%d)\n", major, minor);
    }

    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE,
    };
    EGLConfig config;
    EGLint numConfigs = 0;
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
    EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION,
        3,
        EGL_CONTEXT_MINOR_VERSION,
        3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK,
        EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
        EGL_NONE,
    };
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
    g_egl_ctx.config = config;
    g_egl_ctx.width = width;
    g_egl_ctx.height = height;
    fprintf(stderr, "[egl_headless] GPU context created (%dx%d), ready to activate\n", width, height);
    return 1;
}

static int egl_headless_resize(int width, int height) {
    if (!g_egl_ctx.active || g_egl_ctx.surface == EGL_NO_SURFACE)
        return 0;
    if (width <= g_egl_ctx.width && height <= g_egl_ctx.height)
        return 1;

    int new_w = width > g_egl_ctx.width ? width : g_egl_ctx.width;
    int new_h = height > g_egl_ctx.height ? height : g_egl_ctx.height;

    if (!eglMakeCurrent(g_egl_ctx.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)) {
        fprintf(stderr, "[egl_headless] resize unbind failed: 0x%x\n", eglGetError());
        return 0;
    }
    eglDestroySurface(g_egl_ctx.display, g_egl_ctx.surface);

    EGLint pbufferAttribs[] = {EGL_WIDTH, new_w, EGL_HEIGHT, new_h, EGL_NONE};
    EGLSurface new_surface = eglCreatePbufferSurface(g_egl_ctx.display, g_egl_ctx.config, pbufferAttribs);
    if (new_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "[egl_headless] resize create pbuffer failed: 0x%x\n", eglGetError());
        return 0;
    }
    if (!eglMakeCurrent(g_egl_ctx.display, new_surface, new_surface, g_egl_ctx.context)) {
        fprintf(stderr, "[egl_headless] resize rebind failed: 0x%x\n", eglGetError());
        eglDestroySurface(g_egl_ctx.display, new_surface);
        return 0;
    }

    g_egl_ctx.surface = new_surface;
    g_egl_ctx.width = new_w;
    g_egl_ctx.height = new_h;
    fprintf(stderr, "[egl_headless] pbuffer resized to %dx%d\n", new_w, new_h);
    return 1;
}

static int egl_switch_to_gpu(void) {
    typedef int (*glXMakeCurrentFunc)(void *, unsigned long, void *);
    typedef void *(*glXGetCurrentDisplayFunc)(void);
    typedef void *(*glXGetCurrentContextFunc)(void);

    void *libgl = dlopen("libGL.so.1", RTLD_LAZY);
    if (libgl) {
        glXMakeCurrentFunc glXMC = (glXMakeCurrentFunc)dlsym(libgl, "glXMakeCurrent");
        glXGetCurrentDisplayFunc glXGCD = (glXGetCurrentDisplayFunc)dlsym(libgl, "glXGetCurrentDisplay");
        glXGetCurrentContextFunc glXGCC = (glXGetCurrentContextFunc)dlsym(libgl, "glXGetCurrentContext");
        if (glXMC && glXGCD && glXGCC) {
            void *current_dpy = glXGCD();
            void *current_ctx = glXGCC();
            if (current_ctx && current_dpy) {
                if (glXMC(current_dpy, 0, NULL))
                    fprintf(stderr, "[egl_headless] Released GLX context\n");
                else
                    fprintf(stderr, "[egl_headless] Failed to release current GLX context\n");
            }
        }
        dlclose(libgl);
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

    // Do not call eglTerminate here. NVIDIA EGL and CUDA can share driver state.
    g_egl_ctx.context = NULL;
    g_egl_ctx.surface = NULL;
    g_egl_ctx.active = 0;
}

#endif
