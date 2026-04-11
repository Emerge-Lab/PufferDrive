#version 450

// Polyline vertex shader.
// Input:  vec2 position in world (mean-centered sim) coordinates.
// Push:   mat4 mvp — world → clip space, supplied per-frame per-view.
// Output: clip-space vec4 to gl_Position.
//
// Used for road polylines (line list) and optional trace overlays. The
// fragment shader pulls the color from a separate push-constant member,
// so this stage doesn't carry color attributes — keeps the vertex buffer
// to 8 bytes per vert.

layout(location = 0) in vec2 in_pos;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 color;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(in_pos, 0.0, 1.0);
}
