#version 450

// Polyline fragment shader — flat color from push constant.

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 color;
} pc;

void main() {
    out_color = pc.color;
}
