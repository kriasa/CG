#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec2 in_uv;

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
	vec4 albedo;   // rgb: tint, a: UV tiling
	vec4 specular; // rgb: color, a: shininess
} object;

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;
layout(set = 1, binding = 2) uniform sampler2D specular_texture;
layout(set = 1, binding = 3) uniform sampler2D emissive_texture;

layout(push_constant) uniform Push {
	uint flags;
	float time;
	float emissive_strength;
	float split_amount;      // UV offset of the red/blue samples
	float wave_amount;       // amplitude of the UV distortion
	float modulation_amount; // 0 = original colors, 1 = full color modulation
	float speed;
	float _padding;
} push;

const uint FLAG_MODULATE    = 1u; 
const uint FLAG_MULTISAMPLE = 2u;
const uint FLAG_WAVE        = 4u;

// Blinn-Phong: L points from the surface to the light.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, vec3 specular_color) {
	const float n_dot_l = max(dot(N, L), 0.0);
	if (n_dot_l <= 0.0) {
		return vec3(0.0);
	}

	const vec3 H = normalize(L + V);
	const float spec = pow(max(dot(N, H), 0.0), max(object.specular.w, 1.0));

	return radiance * (albedo * n_dot_l + specular_color * spec);
}

void main() {
	const vec3 N = normalize(in_world_normal);
	const vec3 V = normalize(scene.camera_position.xyz - in_world_position);

	const float t = push.time * push.speed;

	// Texture coordinates: optionally distorted by waves that travel over time
	vec2 uv = in_uv;
	if ((push.flags & FLAG_WAVE) != 0u) {
		uv += push.wave_amount * vec2(sin(in_uv.y * 14.0 + t * 2.0),
		                              sin(in_uv.x * 11.0 - t * 1.6));
	}

	// Albedo (sRGB image -> already linear here)
	vec3 sampled;
	if ((push.flags & FLAG_MULTISAMPLE) != 0u) {
		// The same image is sampled with three different coordinates; each sample
		// provides one color channel, so the channels drift apart (anaglyph look).
		const vec2 offset = push.split_amount * vec2(cos(t * 1.3), sin(t * 1.3));
		sampled = vec3(texture(albedo_texture, uv + offset).r,
		               texture(albedo_texture, uv).g,
		               texture(albedo_texture, uv - offset).b);
	} else {
		sampled = texture(albedo_texture, uv).rgb;
	}

	if ((push.flags & FLAG_MODULATE) != 0u) {
		const vec3 palette = 0.6 + 0.5 * cos(6.28318 * (vec3(0.0, 0.33, 0.67) + t * 0.25));
		sampled *= mix(vec3(1.0), palette, push.modulation_amount);
	}

	const vec3 albedo = sampled * object.albedo.rgb;

	const vec3 specular_color = object.specular.rgb * texture(specular_texture, uv).rgb;
	const vec3 emissive = texture(emissive_texture, uv).rgb * push.emissive_strength;

	vec3 color = albedo * scene.ambient.rgb;

	color += blinnPhong(N, V, -normalize(scene.dir_direction.xyz), scene.dir_color.rgb, albedo, specular_color);

	for (uint i = 0u; i < scene.counts.x; ++i) {
		const PointLight light = point_lights.lights[i];

		const vec3 to_light = light.position.xyz - in_world_position;
		const float dist2 = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light * inversesqrt(dist2);

		color += blinnPhong(N, V, L, light.color.rgb / dist2, albedo, specular_color);
	}

	for (uint i = 0u; i < scene.counts.y; ++i) {
		const SpotLight light = spot_lights.lights[i];

		const vec3 to_light = light.position.xyz - in_world_position;
		const float dist2 = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light * inversesqrt(dist2);

		const float cos_theta = dot(-L, normalize(light.direction.xyz));
		const float cos_inner = light.cone.x;
		const float cos_outer = light.cone.y;

		float spot = clamp((cos_theta - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
		spot = spot * spot * (3.0 - 2.0 * spot);

		color += blinnPhong(N, V, L, light.color.rgb * (spot / dist2), albedo, specular_color);
	}

	// Emissive bypasses the Blinn-Phong calculation and is simply added to the final color
	out_color = vec4(color + emissive, 1.0);
}
