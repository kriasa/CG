#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// One uniform block per object, each bound through its own descriptor set.
layout(set = 0, binding = 0, std140) uniform ObjectUniforms {
	mat4 mvp;
	vec4 tint;   // rgb: color chosen in the UI
	vec4 params; // x: how strongly procedural vertex colors are applied (0..1)
} object;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = object.mvp * vec4(in_position, 1.0);

	// UI color is multiplied by the (optionally faded) procedural vertex color.
	out_color = mix(vec3(1.0), in_color, object.params.x) * object.tint.rgb;
}
