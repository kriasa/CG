#version 450

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push {
	mat4 model;
	vec4 color;
} push;

void main() {
	// Constant color, no lighting
	out_color = vec4(push.color.rgb, 1.0);
}
