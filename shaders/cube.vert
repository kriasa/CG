#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 dir_direction;
	vec4 dir_color;
	uvec4 counts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;   // rgb: tint, a: UV tiling
	vec4 specular; // rgb: color, a: shininess
} object;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec2 out_uv;

void main() {
	const vec4 world_position = object.model * vec4(in_position, 1.0);

	out_world_position = world_position.xyz;
	out_world_normal = mat3(object.normal_matrix) * in_normal;
	out_uv = in_uv * object.albedo.w;

	gl_Position = scene.view_projection * world_position;
}
