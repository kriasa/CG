#version 450

layout(location = 0) in vec3 in_position;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 dir_direction;
	vec4 dir_color;
	uvec4 counts;
} scene;

layout(push_constant) uniform Push {
	mat4 model;
	vec4 color;
} push;

void main() {
	gl_Position = scene.view_projection * push.model * vec4(in_position, 1.0);
}
