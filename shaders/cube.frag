#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec3 in_vertex_color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 dir_direction;
	vec4 dir_color;
	uvec4 counts;
} scene;

struct PointLight {
	vec4 position;
	vec4 color;
};

struct SpotLight {
	vec4 position;
	vec4 direction;
	vec4 color; 
	vec4 cone;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights {
	PointLight lights[];
} point_lights;

layout(set = 0, binding = 2, std430) readonly buffer SpotLights {
	SpotLight lights[];
} spot_lights;

layout(set = 1, binding = 0, std140) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} object;

// Blinn-Phong: L points from the surface to the light.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
	const float n_dot_l = max(dot(N, L), 0.0);
	if (n_dot_l <= 0.0) {
		return vec3(0.0);
	}

	const vec3 H = normalize(L + V);
	const float spec = pow(max(dot(N, H), 0.0), max(object.specular.w, 1.0));

	return radiance * (albedo * n_dot_l + object.specular.rgb * spec);
}

void main() {
	const vec3 N = normalize(in_world_normal);
	const vec3 V = normalize(scene.camera_position.xyz - in_world_position);

	const vec3 albedo = object.albedo.rgb * mix(vec3(1.0), in_vertex_color, object.albedo.w);

	vec3 color = albedo * scene.ambient.rgb;

	// Directional light
	color += blinnPhong(N, V, -normalize(scene.dir_direction.xyz), scene.dir_color.rgb, albedo);

	// Point lights: inverse square attenuation
	for (uint i = 0u; i < scene.counts.x; ++i) {
		const PointLight light = point_lights.lights[i];

		const vec3 to_light = light.position.xyz - in_world_position;
		const float dist2 = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light * inversesqrt(dist2);

		color += blinnPhong(N, V, L, light.color.rgb / dist2, albedo);
	}

	// Spot lights: inverse square attenuation + soft edge between inner and outer cone
	for (uint i = 0u; i < scene.counts.y; ++i) {
		const SpotLight light = spot_lights.lights[i];

		const vec3 to_light = light.position.xyz - in_world_position;
		const float dist2 = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light * inversesqrt(dist2);

		const float cos_theta = dot(-L, normalize(light.direction.xyz));
		const float cos_inner = light.cone.x;
		const float cos_outer = light.cone.y;

		float t = clamp((cos_theta - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
		t = t * t * (3.0 - 2.0 * t);

		color += blinnPhong(N, V, L, light.color.rgb * (t / dist2), albedo);
	}

	out_color = vec4(color, 1.0);
}
