#version 450

// Agent box vertex shader — instanced unit-quad expansion.
//
// Per-vertex (location 0): vec2 quad corner in [-1, +1]^2.
//   The renderer binds a static 4-vert vertex buffer with the unit quad
//   corners and a 6-index index buffer (two triangles).
//
// Per-instance (locations 1..3): one AgentInstance struct per agent for
// the current frame.
//   loc 1 (pose):    vec4 (x, y, heading_rad, _pad)   — world position + angle
//   loc 2 (size):    vec2 (length, width)              — meters
//   loc 3 (color):   vec4 (r, g, b, a)                 — base color
//
// The vertex shader:
//   1. Scales the unit quad by half-(length, width).
//   2. Rotates it by heading.
//   3. Translates it by world (x, y).
//   4. Applies the camera mvp.

layout(location = 0) in vec2 in_corner;     // unit quad corner

layout(location = 1) in vec4 in_pose;       // (x, y, heading, _pad)
layout(location = 2) in vec2 in_size;       // (length, width)
layout(location = 3) in vec4 in_color;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 tint;   // multiplied with in_color; used for view-specific tinting
} pc;

layout(location = 0) out vec4 v_color;

void main() {
    // Scale local corner by half-extents (length is along agent forward = local x,
    // width is across = local y).
    vec2 local = in_corner * (in_size * 0.5);

    // Rotate by heading.
    float c = cos(in_pose.z);
    float s = sin(in_pose.z);
    vec2 rotated = vec2(c * local.x - s * local.y,
                        s * local.x + c * local.y);

    // Translate to world position.
    vec2 world = rotated + in_pose.xy;

    gl_Position = pc.mvp * vec4(world, 0.0, 1.0);
    v_color = in_color * pc.tint;
}
