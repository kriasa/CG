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
	float normal[3];
};

constexpr uint32_t cube_vertex_count = 24; 
constexpr uint32_t cube_index_count = 36;

struct CubeMesh {
	std::array<Vertex, cube_vertex_count> vertices;
	std::array<uint16_t, cube_index_count> indices;
};

CubeMesh makeCubeMesh() {
	constexpr std::array<std::array<uint32_t, 4>, 6> faces = {{
		{ 1, 3, 7, 5 }, // +X
		{ 0, 4, 6, 2 }, // -X
		{ 2, 6, 7, 3 }, // +Y
		{ 0, 1, 5, 4 }, // -Y
		{ 4, 5, 7, 6 }, // +Z
		{ 0, 2, 3, 1 }, // -Z
	}};

	const auto corner = [](uint32_t i) {
		return Vec3{
			(i & 1u) ? 0.5f : -0.5f,
			(i & 2u) ? 0.5f : -0.5f,
			(i & 4u) ? 0.5f : -0.5f,
		};
	};

	CubeMesh mesh{};

	for (uint32_t f = 0; f < faces.size(); ++f) {
		const Vec3 a = corner(faces[f][0]);
		const Vec3 b = corner(faces[f][1]);
		const Vec3 c = corner(faces[f][2]);

		Vec3 normal = math3d::normalize(math3d::cross(b - a, c - a));
		if (math3d::dot(normal, a) < 0.0f) {
			normal = normal * -1.0f;
		}

		for (uint32_t v = 0; v < 4; ++v) {
			const Vec3 p = corner(faces[f][v]);
			mesh.vertices[f * 4 + v] = {
				{ p.x, p.y, p.z },
				{ p.x + 0.5f, p.y + 0.5f, p.z + 0.5f },
				{ normal.x, normal.y, normal.z },
			};
		}

		const uint16_t base = static_cast<uint16_t>(f * 4);
		const uint16_t quad[6] = { 0, 1, 2, 0, 2, 3 };
		for (uint32_t i = 0; i < 6; ++i) {
			mesh.indices[f * 6 + i] = static_cast<uint16_t>(base + quad[i]);
		}
	}

	return mesh;
}

std::vector<std::array<float, 3>> makeConeLines() {
	constexpr int segments = 32;
	constexpr int spokes = 8;

	const auto rim = [](int i) {
		const float angle = 2.0f * math3d::pi * static_cast<float>(i) / static_cast<float>(segments);
		return std::array<float, 3>{ std::cos(angle), std::sin(angle), 1.0f };
	};

	std::vector<std::array<float, 3>> lines;

	for (int i = 0; i < segments; ++i) {
		lines.push_back(rim(i));
		lines.push_back(rim(i + 1));
	}
	for (int s = 0; s < spokes; ++s) {
		lines.push_back({ 0.0f, 0.0f, 0.0f });
		lines.push_back(rim(s * segments / spokes));
	}

	// Axis: shows the light direction.
	lines.push_back({ 0.0f, 0.0f, 0.0f });
	lines.push_back({ 0.0f, 0.0f, 1.0f });

	return lines;
}

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
};

struct SceneUniforms {
	Mat4 view_projection;
	float camera_position[4];
	float ambient[4]; 
	float dir_direction[4];
	float dir_color[4]; 
	uint32_t counts[4];  
};
static_assert(sizeof(SceneUniforms) == 144, "SceneUniforms must match the std140 layout in the shaders");

struct PointLightGpu {
	float position[4];
	float color[4];
};
static_assert(sizeof(PointLightGpu) == 32, "PointLightGpu must match the std430 layout in cube.frag");

struct SpotLightGpu {
	float position[4];
	float direction[4]; 
	float color[4];   
	float cone[4]; 
};
static_assert(sizeof(SpotLightGpu) == 64, "SpotLightGpu must match the std430 layout in cube.frag");

struct ModelUniforms {
	Mat4 model;
	Mat4 normal_matrix;
	float albedo[4]; 
	float specular[4];
};
static_assert(sizeof(ModelUniforms) == 160, "ModelUniforms must match the std140 layout in the shaders");

struct VisPush {
	Mat4 model;
	float color[4];
};
static_assert(sizeof(VisPush) == 80, "VisPush must match the push constant block in vis.vert/vis.frag");

constexpr uint32_t max_point_lights = 128;
constexpr uint32_t max_spot_lights = 128;

constexpr uint32_t object_count = 3;

struct GpuObject {
	Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
	ModelUniforms uniforms{};
};

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

VkPipelineLayout lit_pipeline_layout = VK_NULL_HANDLE;
VkPipelineLayout vis_pipeline_layout = VK_NULL_HANDLE;

VkPipeline lit_pipeline = VK_NULL_HANDLE;
VkPipeline vis_solid_pipeline = VK_NULL_HANDLE; 
VkPipeline vis_line_pipeline = VK_NULL_HANDLE;  

Buffer vertex_buffer;
Buffer index_buffer;
Buffer cone_buffer;
uint32_t cone_vertex_count = 0;

Buffer scene_buffer;
Buffer point_light_buffer;
Buffer spot_light_buffer;
VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;

std::array<GpuObject, object_count> gpu_objects;

SceneUniforms scene_uniforms{};
std::vector<PointLightGpu> gpu_point_lights;
std::vector<SpotLightGpu> gpu_spot_lights;


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

struct PipelineDesc {
	const char* vertex_shader;
	const char* fragment_shader;
	VkPipelineLayout layout;
	uint32_t vertex_stride;
	const VkVertexInputAttributeDescription* attributes;
	uint32_t attribute_count;
	VkPrimitiveTopology topology;
	VkCullModeFlags cull_mode;
};

bool createGraphicsPipeline(const PipelineDesc& desc, VkPipeline& out) {
	VkShaderModule vertex_module = VK_NULL_HANDLE;
	VkShaderModule fragment_module = VK_NULL_HANDLE;

	const std::string shader_dir = SHADER_DIR;
	const bool shaders_ok =
		createShaderModule(shader_dir + "/" + desc.vertex_shader, vertex_module) &&
		createShaderModule(shader_dir + "/" + desc.fragment_shader, fragment_module);

	if (!shaders_ok) {
		if (vertex_module != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, vertex_module, nullptr);
		}
		if (fragment_module != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, fragment_module, nullptr);
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
		.stride = desc.vertex_stride,
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = desc.attribute_count,
		.pVertexAttributeDescriptions = desc.attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = desc.topology,
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
		.cullMode = desc.cull_mode,
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
		.layout = desc.layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &out);

	vkDestroyShaderModule(context.device, vertex_module, nullptr);
	vkDestroyShaderModule(context.device, fragment_module, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline (" << desc.vertex_shader << ")\n";
		return false;
	}

	return true;
}

bool createPipelines() {
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for the scene\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for models\n";
		return false;
	}

	const VkDescriptorSetLayout lit_set_layouts[] = { scene_set_layout, model_set_layout };

	const VkPipelineLayoutCreateInfo lit_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(lit_set_layouts) / sizeof(lit_set_layouts[0]),
		.pSetLayouts = lit_set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &lit_layout_info, nullptr,
	                           &lit_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	const VkPushConstantRange vis_push_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(VisPush),
	};

	const VkPipelineLayoutCreateInfo vis_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &scene_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &vis_push_range,
	};

	if (vkCreatePipelineLayout(context.device, &vis_layout_info, nullptr,
	                           &vis_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout for light visualization\n";
		return false;
	}

	const VkVertexInputAttributeDescription lit_attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
	};

	const VkVertexInputAttributeDescription marker_attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
	};

	const VkVertexInputAttributeDescription line_attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
	};

	const PipelineDesc lit_desc = {
		.vertex_shader = "cube.vert.spv",
		.fragment_shader = "cube.frag.spv",
		.layout = lit_pipeline_layout,
		.vertex_stride = sizeof(Vertex),
		.attributes = lit_attributes,
		.attribute_count = sizeof(lit_attributes) / sizeof(lit_attributes[0]),
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.cull_mode = VK_CULL_MODE_BACK_BIT,
	};

	const PipelineDesc marker_desc = {
		.vertex_shader = "vis.vert.spv",
		.fragment_shader = "vis.frag.spv",
		.layout = vis_pipeline_layout,
		.vertex_stride = sizeof(Vertex),
		.attributes = marker_attributes,
		.attribute_count = sizeof(marker_attributes) / sizeof(marker_attributes[0]),
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.cull_mode = VK_CULL_MODE_BACK_BIT,
	};

	const PipelineDesc line_desc = {
		.vertex_shader = "vis.vert.spv",
		.fragment_shader = "vis.frag.spv",
		.layout = vis_pipeline_layout,
		.vertex_stride = sizeof(float) * 3,
		.attributes = line_attributes,
		.attribute_count = sizeof(line_attributes) / sizeof(line_attributes[0]),
		.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		.cull_mode = VK_CULL_MODE_NONE,
	};

	return createGraphicsPipeline(lit_desc, lit_pipeline) &&
	       createGraphicsPipeline(marker_desc, vis_solid_pipeline) &&
	       createGraphicsPipeline(line_desc, vis_line_pipeline);
}

bool createGeometry() {
	const CubeMesh cube = makeCubeMesh();
	const auto cone = makeConeLines();
	cone_vertex_count = static_cast<uint32_t>(cone.size());

	if (!createBuffer(sizeof(cube.vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createBuffer(sizeof(cube.indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer) ||
	    !createBuffer(cone.size() * sizeof(cone[0]), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cone_buffer)) {
		return false;
	}

	writeBuffer(vertex_buffer, cube.vertices.data(), sizeof(cube.vertices));
	writeBuffer(index_buffer, cube.indices.data(), sizeof(cube.indices));
	writeBuffer(cone_buffer, cone.data(), cone.size() * sizeof(cone[0]));

	return true;
}

bool createDescriptors() {
	const VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 + object_count,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + object_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer) ||
	    !createBuffer(max_point_lights * sizeof(PointLightGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_light_buffer) ||
	    !createBuffer(max_spot_lights * sizeof(SpotLightGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_light_buffer)) {
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_allocate_info, &scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor set for the scene\n";
		return false;
	}

	const VkDescriptorBufferInfo scene_infos[] = {
		{ .buffer = scene_buffer.buffer, .offset = 0, .range = sizeof(SceneUniforms) },
		{ .buffer = point_light_buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
		{ .buffer = spot_light_buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
	};

	const VkWriteDescriptorSet scene_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_infos[0],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &scene_infos[1],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &scene_infos[2],
		},
	};

	vkUpdateDescriptorSets(context.device, sizeof(scene_writes) / sizeof(scene_writes[0]),
	                       scene_writes, 0, nullptr);

	std::array<VkDescriptorSetLayout, object_count> layouts;
	layouts.fill(model_set_layout);

	std::array<VkDescriptorSet, object_count> sets{};

	const VkDescriptorSetAllocateInfo model_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = object_count,
		.pSetLayouts = layouts.data(),
	};

	if (vkAllocateDescriptorSets(context.device, &model_allocate_info, sets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor sets for models\n";
		return false;
	}

	for (uint32_t i = 0; i < object_count; ++i) {
		GpuObject& object = gpu_objects[i];
		object.descriptor_set = sets[i];

		if (!createBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  object.uniform_buffer)) {
			return false;
		}

		const VkDescriptorBufferInfo buffer_info = {
			.buffer = object.uniform_buffer.buffer,
			.offset = 0,
			.range = sizeof(ModelUniforms),
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

	destroyBuffer(spot_light_buffer);
	destroyBuffer(point_light_buffer);
	destroyBuffer(scene_buffer);
	scene_descriptor_set = VK_NULL_HANDLE;

	destroyBuffer(cone_buffer);
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
	}

	for (VkPipeline* pipeline : { &lit_pipeline, &vis_solid_pipeline, &vis_line_pipeline }) {
		if (*pipeline != VK_NULL_HANDLE) {
			vkDestroyPipeline(context.device, *pipeline, nullptr);
			*pipeline = VK_NULL_HANDLE;
		}
	}
	for (VkPipelineLayout* layout : { &lit_pipeline_layout, &vis_pipeline_layout }) {
		if (*layout != VK_NULL_HANDLE) {
			vkDestroyPipelineLayout(context.device, *layout, nullptr);
			*layout = VK_NULL_HANDLE;
		}
	}
	for (VkDescriptorSetLayout* layout : { &scene_set_layout, &model_set_layout }) {
		if (*layout != VK_NULL_HANDLE) {
			vkDestroyDescriptorSetLayout(context.device, *layout, nullptr);
			*layout = VK_NULL_HANDLE;
		}
	}
}


enum class ProjectionType : int {
	Perspective = 0,
	Orthographic = 1,
};

struct Camera {
	ProjectionType projection = ProjectionType::Perspective;
	Vec3 position{ 0.0f, 1.5f, 8.0f };
	float yaw_deg = 0.0f;
	float pitch_deg = -12.0f; 
	float fov_deg = 60.0f;
	float ortho_height = 8.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float move_speed = 4.0f;
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

struct Material {
	float albedo[3] = { 1.0f, 1.0f, 1.0f };
	float specular[3] = { 0.6f, 0.6f, 0.6f };
	float shininess = 48.0f;
	float vertex_color_amount = 1.0f;
};

struct ObjectState {
	const char* name;
	bool visible = true;
	Vec3 position;
	Vec3 rotation_deg;
	Vec3 scale{ 1.0f, 1.0f, 1.0f };
	bool uniform_scale = true;
	Vec3 spin_deg_per_sec;
	Material material;
};

struct AmbientLight {
	float color[3] = { 1.0f, 1.0f, 1.0f };
	float intensity = 0.12f;
};

struct DirectionalLight {
	Vec3 direction{ -0.4f, -1.0f, -0.5f };
	float color[3] = { 1.0f, 1.0f, 1.0f };
	float intensity = 0.8f;
};

struct PointLightState {
	Vec3 position{ 3.0f, 2.0f, 3.0f };
	float color[3] = { 1.0f, 0.75f, 0.45f };
	float intensity = 14.0f;
};

struct SpotLightState {
	Vec3 position{ 0.0f, 4.0f, 0.0f };
	Vec3 direction{ 0.0f, -1.0f, 0.0f };
	float color[3] = { 0.55f, 0.75f, 1.0f };
	float intensity = 35.0f;
	float inner_deg = 15.0f;
	float outer_deg = 28.0f;
};

constexpr float spot_visual_length = 2.0f;
constexpr float light_marker_size = 0.15f;

Camera camera;
Animation animation;
AmbientLight ambient_light;
DirectionalLight directional_light;
std::vector<PointLightState> point_lights;
std::vector<SpotLightState> spot_lights;
bool show_light_visuals = true;

std::array<ObjectState, object_count> objects;

ObjectState makeDefaultObject(uint32_t index) {
	if (index == 0) {
		ObjectState object{ .name = "Cube A (main)" };
		object.rotation_deg = { 20.0f, 30.0f, 0.0f };
		object.spin_deg_per_sec = { 20.0f, 45.0f, 0.0f };
		return object;
	}

	if (index == 1) {
		ObjectState object{ .name = "Cube B (satellite)" };
		object.scale = { 0.4f, 0.4f, 0.4f };
		object.spin_deg_per_sec = { 0.0f, -90.0f, 30.0f };
		object.material.albedo[0] = 1.0f;
		object.material.albedo[1] = 0.8f;
		object.material.albedo[2] = 0.5f;
		return object;
	}

	ObjectState object{ .name = "Floor" };
	object.position = { 0.0f, -2.5f, 0.0f };
	object.scale = { 16.0f, 0.2f, 16.0f };
	object.uniform_scale = false;
	object.material.albedo[0] = 0.55f;
	object.material.albedo[1] = 0.56f;
	object.material.albedo[2] = 0.6f;
	object.material.specular[0] = object.material.specular[1] = object.material.specular[2] = 0.25f;
	object.material.shininess = 16.0f;
	object.material.vertex_color_amount = 0.0f;
	return object;
}

void resetScene() {
	camera = {};
	animation = {};
	ambient_light = {};
	directional_light = {};
	point_lights.assign(1, PointLightState{});
	spot_lights.assign(1, SpotLightState{});
	show_light_visuals = true;
	for (uint32_t i = 0; i < object_count; ++i) {
		objects[i] = makeDefaultObject(i);
	}
}

Vec3 safeNormalize(const Vec3& v, const Vec3& fallback) {
	const Vec3 n = math3d::normalize(v);
	return math3d::length(n) > 0.5f ? n : fallback;
}

Vec3 cameraForward() {
	const float yaw = math3d::radians(camera.yaw_deg);
	const float pitch = math3d::radians(camera.pitch_deg);
	return {
		std::sin(yaw) * std::cos(pitch),
		std::sin(pitch),
		-std::cos(yaw) * std::cos(pitch),
	};
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

	ImGui::DragFloat3("Position", &camera.position.x, 0.05f, -50.0f, 50.0f);
	ImGui::SliderFloat("Yaw (deg)", &camera.yaw_deg, -180.0f, 180.0f);
	ImGui::SliderFloat("Pitch (deg)", &camera.pitch_deg, -89.0f, 89.0f);

	if (camera.projection == ProjectionType::Perspective) {
		ImGui::SliderFloat("FOV (deg)", &camera.fov_deg, 20.0f, 120.0f);
	} else {
		ImGui::SliderFloat("View height", &camera.ortho_height, 1.0f, 30.0f);
	}

	ImGui::SliderFloat("Near plane", &camera.near_plane, 0.01f, 5.0f);
	ImGui::SliderFloat("Far plane", &camera.far_plane, 10.0f, 500.0f);
	camera.near_plane = std::min(camera.near_plane, camera.far_plane - 1.0f);

	ImGui::SliderFloat("Move speed", &camera.move_speed, 0.5f, 20.0f);
	ImGui::TextDisabled("WASD/QE: move (Shift: faster), hold RMB: look around");
}

void drawGlobalLightControls() {
	if (!ImGui::CollapsingHeader("Ambient / directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}

	ImGui::SeparatorText("Ambient");
	ImGui::ColorEdit3("Ambient color", ambient_light.color);
	ImGui::SliderFloat("Ambient intensity", &ambient_light.intensity, 0.0f, 1.0f);

	ImGui::SeparatorText("Directional light");
	ImGui::DragFloat3("Direction", &directional_light.direction.x, 0.01f, -1.0f, 1.0f);
	ImGui::ColorEdit3("Color##dir", directional_light.color);
	ImGui::SliderFloat("Intensity##dir", &directional_light.intensity, 0.0f, 5.0f);
}

void drawPointLightControls() {
	if (!ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}

	ImGui::BeginDisabled(point_lights.size() >= max_point_lights);
	if (ImGui::Button("Add point light")) {
		point_lights.emplace_back();
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	ImGui::Text("Count: %d", static_cast<int>(point_lights.size()));

	int remove_index = -1;

	for (int i = 0; i < static_cast<int>(point_lights.size()); ++i) {
		PointLightState& light = point_lights[static_cast<size_t>(i)];

		ImGui::PushID(i);
		if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Point light %d", i)) {
			ImGui::DragFloat3("Position", &light.position.x, 0.05f, -30.0f, 30.0f);
			ImGui::ColorEdit3("Color", light.color);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 200.0f, "%.2f", ImGuiSliderFlags_Logarithmic);

			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	if (remove_index >= 0) {
		point_lights.erase(point_lights.begin() + remove_index);
	}
}

void drawSpotLightControls() {
	if (!ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}

	ImGui::BeginDisabled(spot_lights.size() >= max_spot_lights);
	if (ImGui::Button("Add spot light")) {
		spot_lights.emplace_back();
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	ImGui::Text("Count: %d", static_cast<int>(spot_lights.size()));

	int remove_index = -1;

	for (int i = 0; i < static_cast<int>(spot_lights.size()); ++i) {
		SpotLightState& light = spot_lights[static_cast<size_t>(i)];

		ImGui::PushID(i);
		if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Spot light %d", i)) {
			ImGui::DragFloat3("Position", &light.position.x, 0.05f, -30.0f, 30.0f);
			ImGui::DragFloat3("Direction", &light.direction.x, 0.01f, -1.0f, 1.0f);
			ImGui::ColorEdit3("Color", light.color);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 400.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
			ImGui::SliderFloat("Inner angle (deg)", &light.inner_deg, 0.0f, 80.0f);
			ImGui::SliderFloat("Outer angle (deg)", &light.outer_deg, 1.0f, 80.0f);
			light.inner_deg = std::min(light.inner_deg, light.outer_deg);

			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	if (remove_index >= 0) {
		spot_lights.erase(spot_lights.begin() + remove_index);
	}
}

void drawAnimationControls() {
	if (!ImGui::CollapsingHeader("Animation")) {
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

	if (ImGui::CollapsingHeader(object.name)) {
		ImGui::Checkbox("Visible", &object.visible);

		ImGui::DragFloat3(index == 1 ? "Offset" : "Position", &object.position.x, 0.02f, -20.0f, 20.0f);
		ImGui::SliderFloat3("Rotation (deg)", &object.rotation_deg.x, -180.0f, 180.0f);

		if (object.uniform_scale) {
			float scale = object.scale.x;
			if (ImGui::DragFloat("Scale", &scale, 0.01f, 0.05f, 20.0f)) {
				object.scale = { scale, scale, scale };
			}
		} else {
			ImGui::DragFloat3("Scale", &object.scale.x, 0.01f, 0.05f, 20.0f);
		}
		ImGui::Checkbox("Uniform scale", &object.uniform_scale);

		ImGui::DragFloat3("Spin (deg/s)", &object.spin_deg_per_sec.x, 1.0f, -360.0f, 360.0f);

		ImGui::SeparatorText("Material (Blinn-Phong)");
		ImGui::ColorEdit3("Albedo", object.material.albedo);
		ImGui::ColorEdit3("Specular color", object.material.specular);
		ImGui::SliderFloat("Shininess", &object.material.shininess, 1.0f, 256.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
		ImGui::SliderFloat("Vertex colors", &object.material.vertex_color_amount, 0.0f, 1.0f);

		if (ImGui::Button("Reset object")) {
			object = makeDefaultObject(index);
		}
	}

	ImGui::PopID();
}

void drawUi() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400.0f, 700.0f), ImGuiCond_FirstUseEver);

	if (ImGui::Begin("Lab 2: Camera and lighting")) {
		ImGui::Text("%.1f FPS", static_cast<double>(ImGui::GetIO().Framerate));

		drawCameraControls();
		drawGlobalLightControls();
		drawPointLightControls();
		drawSpotLightControls();

		if (ImGui::CollapsingHeader("Light visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Show light sources", &show_light_visuals);
		}

		drawAnimationControls();

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

void updateCamera(float delta) {
	ImGuiIO& io = ImGui::GetIO();

	static bool looking = false;
	if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !io.WantCaptureMouse) {
		looking = true;
	}
	if (!ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		looking = false;
	}

	if (looking) {
		constexpr float sensitivity = 0.15f;
		camera.yaw_deg += io.MouseDelta.x * sensitivity;
		camera.pitch_deg = std::clamp(camera.pitch_deg - io.MouseDelta.y * sensitivity, -89.0f, 89.0f);
	}

	camera.yaw_deg = std::remainder(camera.yaw_deg, 360.0f);

	if (!io.WantTextInput) {
		const Vec3 world_up{ 0.0f, 1.0f, 0.0f };
		const Vec3 forward = cameraForward();
		const Vec3 right = math3d::normalize(math3d::cross(forward, world_up));

		Vec3 move;
		if (ImGui::IsKeyDown(ImGuiKey_W)) move = move + forward;
		if (ImGui::IsKeyDown(ImGuiKey_S)) move = move - forward;
		if (ImGui::IsKeyDown(ImGuiKey_D)) move = move + right;
		if (ImGui::IsKeyDown(ImGuiKey_A)) move = move - right;
		if (ImGui::IsKeyDown(ImGuiKey_E)) move = move + world_up;
		if (ImGui::IsKeyDown(ImGuiKey_Q)) move = move - world_up;

		const float speed = camera.move_speed * (io.KeyShift ? 3.0f : 1.0f);
		camera.position = camera.position + math3d::normalize(move) * (speed * delta);
	}
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

Mat4 makeNormalMatrix(const ObjectState& object) {
	const float time = static_cast<float>(animation.time);

	const Vec3 angles = {
		math3d::radians(object.rotation_deg.x + object.spin_deg_per_sec.x * time),
		math3d::radians(object.rotation_deg.y + object.spin_deg_per_sec.y * time),
		math3d::radians(object.rotation_deg.z + object.spin_deg_per_sec.z * time),
	};

	const Vec3 inverse_scale = {
		1.0f / object.scale.x,
		1.0f / object.scale.y,
		1.0f / object.scale.z,
	};

	return math3d::rotationXYZ(angles) * math3d::scaling(inverse_scale);
}

Mat4 makeViewProjection() {
	const float width = static_cast<float>(context.swapchain_extent.width);
	const float height = static_cast<float>(std::max(context.swapchain_extent.height, 1u));
	const float aspect = width / height;

	const Mat4 view = math3d::lookAlong(camera.position, cameraForward(), { 0.0f, 1.0f, 0.0f });

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

void updateScene() {
	const Vec3 light_dir = safeNormalize(directional_light.direction, { 0.0f, -1.0f, 0.0f });

	scene_uniforms.view_projection = makeViewProjection();
	scene_uniforms.camera_position[0] = camera.position.x;
	scene_uniforms.camera_position[1] = camera.position.y;
	scene_uniforms.camera_position[2] = camera.position.z;
	scene_uniforms.camera_position[3] = 1.0f;
	for (int c = 0; c < 3; ++c) {
		scene_uniforms.ambient[c] = ambient_light.color[c] * ambient_light.intensity;
		scene_uniforms.dir_color[c] = directional_light.color[c] * directional_light.intensity;
	}
	scene_uniforms.ambient[3] = 0.0f;
	scene_uniforms.dir_color[3] = 0.0f;
	scene_uniforms.dir_direction[0] = light_dir.x;
	scene_uniforms.dir_direction[1] = light_dir.y;
	scene_uniforms.dir_direction[2] = light_dir.z;
	scene_uniforms.dir_direction[3] = 0.0f;

	gpu_point_lights.clear();
	for (const PointLightState& light : point_lights) {
		PointLightGpu gpu{};
		gpu.position[0] = light.position.x;
		gpu.position[1] = light.position.y;
		gpu.position[2] = light.position.z;
		for (int c = 0; c < 3; ++c) {
			gpu.color[c] = light.color[c] * light.intensity;
		}
		gpu_point_lights.push_back(gpu);
	}

	gpu_spot_lights.clear();
	for (const SpotLightState& light : spot_lights) {
		const Vec3 direction = safeNormalize(light.direction, { 0.0f, -1.0f, 0.0f });

		SpotLightGpu gpu{};
		gpu.position[0] = light.position.x;
		gpu.position[1] = light.position.y;
		gpu.position[2] = light.position.z;
		gpu.direction[0] = direction.x;
		gpu.direction[1] = direction.y;
		gpu.direction[2] = direction.z;
		for (int c = 0; c < 3; ++c) {
			gpu.color[c] = light.color[c] * light.intensity;
		}
		gpu.cone[0] = std::cos(math3d::radians(light.inner_deg));
		gpu.cone[1] = std::cos(math3d::radians(light.outer_deg));
		gpu_spot_lights.push_back(gpu);
	}

	scene_uniforms.counts[0] = static_cast<uint32_t>(gpu_point_lights.size());
	scene_uniforms.counts[1] = static_cast<uint32_t>(gpu_spot_lights.size());
	scene_uniforms.counts[2] = scene_uniforms.counts[3] = 0;

	const float time = static_cast<float>(animation.time);

	Vec3 position_a = objects[0].position;
	if (animation.trajectory_enabled) {
		const float phase = time;
		position_a.x += animation.radius * std::sin(phase);
		position_a.y += animation.height_amplitude * std::sin(animation.height_frequency * phase);
		position_a.z += animation.radius * std::sin(2.0f * phase);
	}

	const float angle = animation.orbit_speed * time;
	const float tilt = math3d::radians(animation.orbit_tilt_deg);
	const float orbit_x = animation.orbit_radius * std::cos(angle);
	const float orbit_z = animation.orbit_radius * std::sin(angle);

	const Vec3 position_b = {
		position_a.x + objects[1].position.x + orbit_x,
		position_a.y + objects[1].position.y - orbit_z * std::sin(tilt),
		position_a.z + objects[1].position.z + orbit_z * std::cos(tilt),
	};

	const Vec3 positions[object_count] = { position_a, position_b, objects[2].position };

	for (uint32_t i = 0; i < object_count; ++i) {
		const ObjectState& object = objects[i];
		ModelUniforms& uniforms = gpu_objects[i].uniforms;

		uniforms.model = makeModelMatrix(object, positions[i]);
		uniforms.normal_matrix = makeNormalMatrix(object);

		uniforms.albedo[0] = object.material.albedo[0];
		uniforms.albedo[1] = object.material.albedo[1];
		uniforms.albedo[2] = object.material.albedo[2];
		uniforms.albedo[3] = object.material.vertex_color_amount;

		uniforms.specular[0] = object.material.specular[0];
		uniforms.specular[1] = object.material.specular[1];
		uniforms.specular[2] = object.material.specular[2];
		uniforms.specular[3] = object.material.shininess;
	}
}

void drawLightVisuals(VkCommandBuffer cmd) {
	if (point_lights.empty() && spot_lights.empty()) {
		return;
	}

	const VkDeviceSize vertex_offset = 0;

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vis_solid_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vis_pipeline_layout, 0, 1,
	                        &scene_descriptor_set, 0, nullptr);
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	const auto draw_marker = [&](const Vec3& position, const float color[3]) {
		VisPush push{};
		push.model = math3d::translation(position) *
		             math3d::scaling({ light_marker_size, light_marker_size, light_marker_size });
		push.color[0] = color[0];
		push.color[1] = color[1];
		push.color[2] = color[2];
		push.color[3] = 1.0f;

		vkCmdPushConstants(cmd, vis_pipeline_layout,
		                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(push), &push);
		vkCmdDrawIndexed(cmd, cube_index_count, 1, 0, 0, 0);
	};

	for (const PointLightState& light : point_lights) {
		draw_marker(light.position, light.color);
	}
	for (const SpotLightState& light : spot_lights) {
		draw_marker(light.position, light.color);
	}

	if (spot_lights.empty()) {
		return;
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vis_line_pipeline);
	vkCmdBindVertexBuffers(cmd, 0, 1, &cone_buffer.buffer, &vertex_offset);

	for (const SpotLightState& light : spot_lights) {
		const Vec3 z_axis = safeNormalize(light.direction, { 0.0f, -1.0f, 0.0f });
		const Vec3 up_reference = std::fabs(z_axis.y) > 0.99f ? Vec3{ 1.0f, 0.0f, 0.0f } : Vec3{ 0.0f, 1.0f, 0.0f };
		const Vec3 x_axis = math3d::normalize(math3d::cross(up_reference, z_axis));
		const Vec3 y_axis = math3d::cross(z_axis, x_axis);

		const float radius = spot_visual_length * std::tan(math3d::radians(light.outer_deg));

		VisPush push{};
		push.model = math3d::translation(light.position) *
		             math3d::basis(x_axis, y_axis, z_axis) *
		             math3d::scaling({ radius, radius, spot_visual_length });
		push.color[0] = push.color[1] = push.color[2] = push.color[3] = 1.0f;

		vkCmdPushConstants(cmd, vis_pipeline_layout,
		                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(push), &push);
		vkCmdDraw(cmd, cone_vertex_count, 1, 0, 0);
	}
}

}


bool initialize() {
	resetScene();

	if (!createPipelines() || !createGeometry() || !createDescriptors()) {
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

	updateCamera(static_cast<float>(delta));
	drawUi();
	updateScene();
}

void render(const graphics::internal::FrameData& fd) {
	VkCommandBuffer cmd = fd.command_buffer;
	if (cmd == VK_NULL_HANDLE) {
		return;
	}

	writeBuffer(scene_buffer, &scene_uniforms, sizeof(SceneUniforms));
	if (!gpu_point_lights.empty()) {
		writeBuffer(point_light_buffer, gpu_point_lights.data(),
		            gpu_point_lights.size() * sizeof(PointLightGpu));
	}
	if (!gpu_spot_lights.empty()) {
		writeBuffer(spot_light_buffer, gpu_spot_lights.data(),
		            gpu_spot_lights.size() * sizeof(SpotLightGpu));
	}
	for (uint32_t i = 0; i < object_count; ++i) {
		if (objects[i].visible) {
			writeBuffer(gpu_objects[i].uniform_buffer, &gpu_objects[i].uniforms, sizeof(ModelUniforms));
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

	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline_layout, 0, 1,
	                        &scene_descriptor_set, 0, nullptr);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	for (uint32_t i = 0; i < object_count; ++i) {
		if (!objects[i].visible) {
			continue;
		}

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline_layout, 1, 1,
		                        &gpu_objects[i].descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(cmd, cube_index_count, 1, 0, 0, 0);
	}

	if (show_light_visuals) {
		drawLightVisuals(cmd);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application