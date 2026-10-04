#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#include "math3d.hpp"

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace application {

namespace {

using graphics::internal::context;
using math3d::Mat4;
using math3d::Vec3;


struct Vertex {
	float position[3];
	float color[3];
};

constexpr uint32_t cube_vertex_count = 8;
constexpr uint32_t cube_index_count = 36;

std::array<Vertex, cube_vertex_count> makeCubeVertices() {
	std::array<Vertex, cube_vertex_count> vertices{};

	for (uint32_t i = 0; i < cube_vertex_count; ++i) {
		const float x = (i & 1u) ? 0.5f : -0.5f;
		const float y = (i & 2u) ? 0.5f : -0.5f;
		const float z = (i & 4u) ? 0.5f : -0.5f;

		vertices[i] = {
			{ x, y, z },
			{ x + 0.5f, y + 0.5f, z + 0.5f },
		};
	}

	return vertices;
}

constexpr std::array<uint16_t, cube_index_count> cube_indices = {
	1, 3, 7,  1, 7, 5, // +X
	0, 4, 6,  0, 6, 2, // -X
	2, 6, 7,  2, 7, 3, // +Y
	0, 1, 5,  0, 5, 4, // -Y
	4, 5, 7,  4, 7, 6, // +Z
	0, 2, 3,  0, 3, 1, // -Z
};

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
};

struct ObjectUniforms {
	Mat4 mvp;
	float tint[4];
	float params[4];
};
static_assert(sizeof(ObjectUniforms) == 96, "ObjectUniforms must match the std140 layout in cube.vert");

constexpr uint32_t object_count = 2;

struct GpuObject {
	Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
	ObjectUniforms uniforms{};
};

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

Buffer vertex_buffer;
Buffer index_buffer;

std::array<GpuObject, object_count> gpu_objects;

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result_info{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, &result_info) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	out.mapped = result_info.pMappedData;
	return true;
}

void writeBuffer(const Buffer& buffer, const void* data, size_t size) {
	std::memcpy(buffer.mapped, data, size);
	vmaFlushAllocation(context.allocator, buffer.allocation, 0, VK_WHOLE_SIZE);
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

bool createShaderModule(const std::string& path, VkShaderModule& out) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file: " << path << '\n';
		return false;
	}

	const std::streamsize size = file.tellg();
	if (size <= 0 || size % 4 != 0) {
		std::cerr << "Invalid SPIR-V file: " << path << '\n';
		return false;
	}

	std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);

	const VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = static_cast<size_t>(size),
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(context.device, &module_info, nullptr, &out) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return false;
	}

	return true;
}

bool createPipeline() {
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo set_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &set_layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout\n";
		return false;
	}

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	VkShaderModule vertex_module = VK_NULL_HANDLE;
	VkShaderModule fragment_module = VK_NULL_HANDLE;

	const std::string shader_dir = SHADER_DIR;
	if (!createShaderModule(shader_dir + "/cube.vert.spv", vertex_module) ||
	    !createShaderModule(shader_dir + "/cube.frag.spv", fragment_module)) {
		if (vertex_module != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, vertex_module, nullptr);
		}
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_module,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);

	vkDestroyShaderModule(context.device, vertex_module, nullptr);
	vkDestroyShaderModule(context.device, fragment_module, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		return false;
	}

	return true;
}

bool createGeometry() {
	const auto vertices = makeCubeVertices();

	if (!createBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||!createBuffer(sizeof(cube_indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
		return false;
	}

	writeBuffer(vertex_buffer, vertices.data(), sizeof(vertices));
	writeBuffer(index_buffer, cube_indices.data(), sizeof(cube_indices));

	return true;
}

bool createObjectResources() {
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = object_count,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = object_count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	std::array<VkDescriptorSetLayout, object_count> layouts;
	layouts.fill(descriptor_set_layout);

	std::array<VkDescriptorSet, object_count> sets{};

	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = object_count,
		.pSetLayouts = layouts.data(),
	};

	if (vkAllocateDescriptorSets(context.device, &allocate_info, sets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor sets\n";
		return false;
	}

	for (uint32_t i = 0; i < object_count; ++i) {
		GpuObject& object = gpu_objects[i];
		object.descriptor_set = sets[i];

		if (!createBuffer(sizeof(ObjectUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  object.uniform_buffer)) {
			return false;
		}

		const VkDescriptorBufferInfo buffer_info = {
			.buffer = object.uniform_buffer.buffer,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};

		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = object.descriptor_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};

		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}

	return true;
}

void destroyResources() {
	for (GpuObject& object : gpu_objects) {
		destroyBuffer(object.uniform_buffer);
		object.descriptor_set = VK_NULL_HANDLE;
	}

	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
	}
	if (pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, pipeline, nullptr);
		pipeline = VK_NULL_HANDLE;
	}
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
	}
	if (descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
		descriptor_set_layout = VK_NULL_HANDLE;
	}
}

enum class ProjectionType : int {
	Perspective = 0,
	Orthographic = 1,
};

struct Camera {
	ProjectionType projection = ProjectionType::Perspective;
	float distance = 6.0f;
	float fov_deg = 60.0f;
	float ortho_height = 7.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
};

struct Animation {
	bool paused = false;
	float speed = 1.0f;
	double time = 0.0;

	bool trajectory_enabled = true;
	float radius = 1.5f;
	float height_amplitude = 0.8f;
	float height_frequency = 3.0f;
	float orbit_radius = 1.8f;
	float orbit_speed = 2.5f;
	float orbit_tilt_deg = 25.0f;
};

struct ObjectState {
	const char* name;
	bool visible = true;
	Vec3 position;
	Vec3 rotation_deg;
	Vec3 scale{ 1.0f, 1.0f, 1.0f };
	bool uniform_scale = true;
	Vec3 spin_deg_per_sec;
	float tint[3] = { 1.0f, 1.0f, 1.0f };
};

Camera camera;
Animation animation;
float vertex_color_amount = 1.0f;

std::array<ObjectState, object_count> objects;

ObjectState makeDefaultObject(uint32_t index) {
	if (index == 0) {
		ObjectState object{ .name = "Cube A (main)" };
		object.rotation_deg = { 20.0f, 30.0f, 0.0f };
		object.spin_deg_per_sec = { 20.0f, 45.0f, 0.0f };
		return object;
	}

	ObjectState object{ .name = "Cube B (satellite)" };
	object.scale = { 0.4f, 0.4f, 0.4f };
	object.spin_deg_per_sec = { 0.0f, -90.0f, 30.0f };
	object.tint[0] = 1.0f;
	object.tint[1] = 0.8f;
	object.tint[2] = 0.5f;
	return object;
}

void resetScene() {
	camera = {};
	animation = {};
	vertex_color_amount = 1.0f;
	for (uint32_t i = 0; i < object_count; ++i) {
		objects[i] = makeDefaultObject(i);
	}
}

void drawCameraControls() {
	if (!ImGui::CollapsingHeader("Camera / projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}

	int projection = static_cast<int>(camera.projection);
	ImGui::RadioButton("Perspective", &projection, static_cast<int>(ProjectionType::Perspective));
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &projection, static_cast<int>(ProjectionType::Orthographic));
	camera.projection = static_cast<ProjectionType>(projection);

	ImGui::SliderFloat("Camera distance", &camera.distance, 2.0f, 20.0f);

	if (camera.projection == ProjectionType::Perspective) {
		ImGui::SliderFloat("FOV (deg)", &camera.fov_deg, 20.0f, 120.0f);
	} else {
		ImGui::SliderFloat("View height", &camera.ortho_height, 1.0f, 30.0f);
	}

	ImGui::SliderFloat("Near plane", &camera.near_plane, 0.01f, 5.0f);
	ImGui::SliderFloat("Far plane", &camera.far_plane, 10.0f, 500.0f);
	camera.near_plane = std::min(camera.near_plane, camera.far_plane - 1.0f);
}

void drawAnimationControls() {
	if (!ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}

	if (ImGui::Button(animation.paused ? "Play" : "Pause")) {
		animation.paused = !animation.paused;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		animation.time = 0.0;
	}
	ImGui::SameLine();
	ImGui::TextDisabled("(Space toggles)");

	ImGui::SliderFloat("Speed", &animation.speed, 0.0f, 5.0f);

	ImGui::SeparatorText("Cube A trajectory");
	ImGui::Checkbox("Move along trajectory", &animation.trajectory_enabled);
	ImGui::SliderFloat("Radius", &animation.radius, 0.0f, 4.0f);
	ImGui::SliderFloat("Height amplitude", &animation.height_amplitude, 0.0f, 2.0f);
	ImGui::SliderFloat("Height frequency", &animation.height_frequency, 1.0f, 8.0f);

	ImGui::SeparatorText("Cube B orbit");
	ImGui::SliderFloat("Orbit radius", &animation.orbit_radius, 0.5f, 4.0f);
	ImGui::SliderFloat("Orbit speed", &animation.orbit_speed, -6.0f, 6.0f);
	ImGui::SliderFloat("Orbit tilt (deg)", &animation.orbit_tilt_deg, -90.0f, 90.0f);
}

void drawObjectControls(uint32_t index) {
	ObjectState& object = objects[index];

	ImGui::PushID(static_cast<int>(index));

	if (ImGui::CollapsingHeader(object.name, ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Visible", &object.visible);

		ImGui::DragFloat3(index == 0 ? "Position" : "Offset", &object.position.x, 0.02f, -10.0f, 10.0f);
		ImGui::SliderFloat3("Rotation (deg)", &object.rotation_deg.x, -180.0f, 180.0f);

		if (object.uniform_scale) {
			float scale = object.scale.x;
			if (ImGui::DragFloat("Scale", &scale, 0.01f, 0.05f, 5.0f)) {
				object.scale = { scale, scale, scale };
			}
		} else {
			ImGui::DragFloat3("Scale", &object.scale.x, 0.01f, 0.05f, 5.0f);
		}
		ImGui::Checkbox("Uniform scale", &object.uniform_scale);

		ImGui::DragFloat3("Spin (deg/s)", &object.spin_deg_per_sec.x, 1.0f, -360.0f, 360.0f);
		ImGui::ColorEdit3("Color", object.tint);

		if (ImGui::Button("Reset object")) {
			object = makeDefaultObject(index);
		}
	}

	ImGui::PopID();
}

void drawUi() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 640.0f), ImGuiCond_FirstUseEver);

	if (ImGui::Begin("Lab 1: Hexahedron")) {
		ImGui::Text("%.1f FPS", static_cast<double>(ImGui::GetIO().Framerate));

		drawCameraControls();
		drawAnimationControls();

		if (ImGui::CollapsingHeader("Shading", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::SliderFloat("Vertex colors", &vertex_color_amount, 0.0f, 1.0f);
		}

		for (uint32_t i = 0; i < object_count; ++i) {
			drawObjectControls(i);
		}

		ImGui::Separator();
		if (ImGui::Button("Reset everything")) {
			resetScene();
		}
	}
	ImGui::End();
}

Mat4 makeModelMatrix(const ObjectState& object, const Vec3& world_position) {
	const float time = static_cast<float>(animation.time);

	const Vec3 angles = {
		math3d::radians(object.rotation_deg.x + object.spin_deg_per_sec.x * time),
		math3d::radians(object.rotation_deg.y + object.spin_deg_per_sec.y * time),
		math3d::radians(object.rotation_deg.z + object.spin_deg_per_sec.z * time),
	};

	return math3d::translation(world_position) *
	       math3d::rotationXYZ(angles) *
	       math3d::scaling(object.scale);
}

Mat4 makeViewProjection() {
	const float width = static_cast<float>(context.swapchain_extent.width);
	const float height = static_cast<float>(std::max(context.swapchain_extent.height, 1u));
	const float aspect = width / height;

	const Mat4 view = math3d::translation({ 0.0f, 0.0f, -camera.distance });

	Mat4 projection;
	if (camera.projection == ProjectionType::Perspective) {
		projection = math3d::perspective(math3d::radians(camera.fov_deg), aspect,
		                                 camera.near_plane, camera.far_plane);
	} else {
		projection = math3d::orthographic(camera.ortho_height * aspect, camera.ortho_height,
		                                  camera.near_plane, camera.far_plane);
	}

	return projection * view;
}

void updateUniforms() {
	const Mat4 view_projection = makeViewProjection();
	const float time = static_cast<float>(animation.time);

	Vec3 position_a = objects[0].position;
	if (animation.trajectory_enabled) {
		const float phase = time;
		position_a.x += animation.radius * std::cos(phase);
		position_a.y += animation.height_amplitude * std::sin(animation.height_frequency * phase);
		position_a.z += animation.radius * std::sin(phase);
	}

	const float angle = animation.orbit_speed * time;
	const float tilt = math3d::radians(animation.orbit_tilt_deg);
	const float orbit_x = animation.orbit_radius * std::cos(angle);
	const float orbit_z = animation.orbit_radius * std::sin(angle);

	Vec3 position_b = {
		position_a.x + objects[1].position.x + orbit_x,
		position_a.y + objects[1].position.y - orbit_z * std::sin(tilt),
		position_a.z + objects[1].position.z + orbit_z * std::cos(tilt),
	};

	const Vec3 positions[object_count] = { position_a, position_b };

	for (uint32_t i = 0; i < object_count; ++i) {
		const ObjectState& object = objects[i];
		ObjectUniforms& uniforms = gpu_objects[i].uniforms;

		uniforms.mvp = view_projection * makeModelMatrix(object, positions[i]);
		uniforms.tint[0] = object.tint[0];
		uniforms.tint[1] = object.tint[1];
		uniforms.tint[2] = object.tint[2];
		uniforms.tint[3] = 1.0f;
		uniforms.params[0] = vertex_color_amount;
		uniforms.params[1] = uniforms.params[2] = uniforms.params[3] = 0.0f;
	}
}

}

bool initialize() {
	resetScene();

	if (!createPipeline() || !createGeometry() || !createObjectResources()) {
		destroyResources();
		return false;
	}

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);
	destroyResources();
}

void update(double time) {
	static double last_time = time;
	const double delta = std::clamp(time - last_time, 0.0, 0.1);
	last_time = time;

	if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
		animation.paused = !animation.paused;
	}

	if (!animation.paused) {
		animation.time += delta * static_cast<double>(animation.speed);
	}

	drawUi();
	updateUniforms();
}

void render(const graphics::internal::FrameData& fd) {
	VkCommandBuffer cmd = fd.command_buffer;
	if (cmd == VK_NULL_HANDLE) {
		return;
	}

	for (uint32_t i = 0; i < object_count; ++i) {
		if (objects[i].visible) {
			writeBuffer(gpu_objects[i].uniform_buffer, &gpu_objects[i].uniforms, sizeof(ObjectUniforms));
		}
	}

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_values[2]{};
	clear_values[0].color = { { 0.08f, 0.09f, 0.11f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .offset = { 0, 0 }, .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(cmd, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .offset = { 0, 0 }, .extent = context.swapchain_extent };

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	for (uint32_t i = 0; i < object_count; ++i) {
		if (!objects[i].visible) {
			continue;
		}

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
		                        &gpu_objects[i].descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(cmd, cube_index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application