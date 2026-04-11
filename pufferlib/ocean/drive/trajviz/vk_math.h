/*
 * vk_math.h — minimal column-major mat4 helpers for trajviz.
 *
 * Header-only on purpose: this is ~30 lines of arithmetic, not worth a
 * separate translation unit, and it gets inlined into the four call sites
 * (top-down camera, BEV camera, push-constant upload, agent box vertex
 * shader uniforms).
 *
 * Layout: column-major, 4x4 floats, m[col][row]. Matches Vulkan's expected
 * uniform layout (std140) when uploaded as 16 floats. Multiplication is
 * standard: M*v transforms a column vector v.
 *
 * Coordinate convention: world is right-handed, +x right, +y up. The
 * mat4_ortho helper bakes the Vulkan y-flip (clip +y is "down") into the
 * projection so that world +y appears at the TOP of the rendered frame —
 * the natural orientation for top-down maps.
 */

#ifndef VK_MATH_H
#define VK_MATH_H

#include <math.h>
#include <string.h>

typedef struct {
    float m[16];
} Mat4; /* column-major: m[col*4 + row] */

static inline Mat4 mat4_identity(void) {
    Mat4 r = {{0}};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

/* Vulkan-style orthographic projection with y-flip baked in.
 * Maps world (left..right, bottom..top, near..far) → clip (-1..1, +1..-1, 0..1).
 * Note the y range flip: world bottom→clip +1, world top→clip -1, so that
 * world +y appears upward on the rendered image.
 *
 * The +1..-1 on y is the Vulkan convention difference vs OpenGL.
 * The 0..1 on z is also Vulkan-specific (vs OpenGL's -1..1). */
static inline Mat4 mat4_ortho(float l, float r, float b, float t, float n, float f) {
    Mat4 m = {{0}};
    m.m[0] = 2.0f / (r - l);
    m.m[5] = -2.0f / (t - b); /* y-flip */
    m.m[10] = 1.0f / (f - n);
    m.m[12] = -(r + l) / (r - l);
    m.m[13] = (t + b) / (t - b); /* y-flip */
    m.m[14] = -n / (f - n);
    m.m[15] = 1.0f;
    return m;
}

static inline Mat4 mat4_translate(float tx, float ty, float tz) {
    Mat4 r = mat4_identity();
    r.m[12] = tx;
    r.m[13] = ty;
    r.m[14] = tz;
    return r;
}

static inline Mat4 mat4_rotate_z(float angle_rad) {
    Mat4 r = mat4_identity();
    float c = cosf(angle_rad);
    float s = sinf(angle_rad);
    r.m[0] = c;
    r.m[4] = -s;
    r.m[1] = s;
    r.m[5] = c;
    return r;
}

static inline Mat4 mat4_mul(Mat4 a, Mat4 b) {
    Mat4 r = {{0}};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) {
                s += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            r.m[col * 4 + row] = s;
        }
    }
    return r;
}

/* Fit a world-space AABB into a viewport of (vp_w, vp_h) pixels with the
 * given fractional padding (e.g. 0.05 = 5% margin on each side), preserving
 * aspect ratio. Returns an ortho projection that maps the world AABB to the
 * full viewport with letterbox/pillarbox as needed. */
static inline Mat4 mat4_fit_aabb(float xmin, float ymin, float xmax, float ymax, int vp_w, int vp_h, float pad_frac) {
    float w = xmax - xmin;
    float h = ymax - ymin;
    if (w <= 0.0f)
        w = 1.0f;
    if (h <= 0.0f)
        h = 1.0f;
    float cx = 0.5f * (xmin + xmax);
    float cy = 0.5f * (ymin + ymax);

    /* Choose half-extent so the world AABB fits inside the viewport with
     * aspect preserved. The viewport has aspect vp_w/vp_h; the world has
     * aspect w/h. Pick whichever side is the binding constraint. */
    float vp_aspect = (float)vp_w / (float)vp_h;
    float world_aspect = w / h;
    float half_w, half_h;
    if (world_aspect > vp_aspect) {
        /* World is wider than viewport: width-bound, height grows */
        half_w = 0.5f * w;
        half_h = half_w / vp_aspect;
    } else {
        /* World is taller than viewport: height-bound, width grows */
        half_h = 0.5f * h;
        half_w = half_h * vp_aspect;
    }
    half_w *= (1.0f + pad_frac);
    half_h *= (1.0f + pad_frac);

    return mat4_ortho(cx - half_w, cx + half_w, cy - half_h, cy + half_h, -1.0f, 1.0f);
}

/* Build the BEV camera matrix that puts the ego at the origin and rotates
 * the world so the ego's heading vector points to clip "up" (world +y after
 * the rotation). window_m is the half-extent in meters (e.g. 50 for a
 * 100m × 100m view). */
static inline Mat4 mat4_bev_camera(float ego_x, float ego_y, float ego_heading_rad, float window_m, int vp_w,
                                   int vp_h) {
    /* Translate so ego is at origin */
    Mat4 T = mat4_translate(-ego_x, -ego_y, 0.0f);
    /* Rotate by (pi/2 - heading) so the heading vector aligns with +y */
    Mat4 R = mat4_rotate_z(1.5707963f - ego_heading_rad);
    /* Aspect-corrected ortho window centered on origin */
    float vp_aspect = (float)vp_w / (float)vp_h;
    float half_h = window_m;
    float half_w = window_m * vp_aspect;
    Mat4 P = mat4_ortho(-half_w, half_w, -half_h, half_h, -1.0f, 1.0f);
    /* M = P * R * T (apply T first, then R, then P) */
    return mat4_mul(P, mat4_mul(R, T));
}

#endif /* VK_MATH_H */
