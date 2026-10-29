// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The rays of a picture traced with Vulkan on Linux: the shader is
// render_vulkan.comp, compiled to SPIR-V. The Vulkan loader is opened when
// first used, so mdapy needs no Vulkan to load; the build needs its headers.

#include "visualization/render_gpu.h"

#ifdef MDAPY_VULKAN

#define VK_NO_PROTOTYPES
#include <dlfcn.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// kSpirv: the bytes of render_vulkan.spv, written by CMake
#include "visualization/render_vulkan_spirv.h"

namespace mdapy::visualization::gpu {
namespace {

// Push constants: where each array of the scene starts, in words after the
// parameters, and the rows of one dispatch; the shader declares the same.
struct Tile {
    std::uint32_t arrays[13];
    std::uint32_t first_row, rows;
};

// The functions of Vulkan used here, found in the loader.
struct Functions {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance CreateInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateDevice CreateDevice = nullptr;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
    PFN_vkGetDeviceQueue GetDeviceQueue = nullptr;
    PFN_vkCreateShaderModule CreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout CreatePipelineLayout = nullptr;
    PFN_vkCreateComputePipelines CreateComputePipelines = nullptr;
    PFN_vkCreateBuffer CreateBuffer = nullptr;
    PFN_vkDestroyBuffer DestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkBindBufferMemory BindBufferMemory = nullptr;
    PFN_vkMapMemory MapMemory = nullptr;
    PFN_vkCreateDescriptorPool CreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;
    PFN_vkCreateCommandPool CreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkCmdCopyBuffer CmdCopyBuffer = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants CmdPushConstants = nullptr;
    PFN_vkCmdDispatch CmdDispatch = nullptr;
    PFN_vkCreateFence CreateFence = nullptr;
    PFN_vkDestroyFence DestroyFence = nullptr;
    PFN_vkResetFences ResetFences = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
};

// A function of Vulkan, or an error when the loader or driver has none.
template <typename F>
void find(F& function, PFN_vkVoidFunction found, const char* name) {
    if (found == nullptr) {
        throw std::runtime_error(std::string("Vulkan has no ") + name);
    }
    function = reinterpret_cast<F>(found);
}

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed with VkResult " + std::to_string(result));
    }
}

// The GPU, its queue and the compiled shader, made once and kept for the
// life of the process; one picture at a time on its queue.
struct Device {
    Functions vk;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    VkDeviceSize max_range = 0;
    std::uint32_t family = 0;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::string error;
    std::mutex mutex;
};

#define MDAPY_INSTANCE_FUNCTION(name) \
    find(vk.name, vk.GetInstanceProcAddr(d.instance, "vk" #name), "vk" #name)
#define MDAPY_DEVICE_FUNCTION(name) find(vk.name, vk.GetDeviceProcAddr(d.device, "vk" #name), "vk" #name)

void open(Device& d) {
    Functions& vk = d.vk;
    void* library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        throw std::runtime_error("no Vulkan loader (libvulkan.so.1)");
    }
    vk.GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
    if (vk.GetInstanceProcAddr == nullptr) {
        throw std::runtime_error("the Vulkan loader has no vkGetInstanceProcAddr");
    }
    MDAPY_INSTANCE_FUNCTION(CreateInstance);
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "mdapy";
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    check(vk.CreateInstance(&instance_info, nullptr, &d.instance), "vkCreateInstance");
    MDAPY_INSTANCE_FUNCTION(EnumeratePhysicalDevices);
    MDAPY_INSTANCE_FUNCTION(GetPhysicalDeviceProperties);
    MDAPY_INSTANCE_FUNCTION(GetPhysicalDeviceQueueFamilyProperties);
    MDAPY_INSTANCE_FUNCTION(GetPhysicalDeviceMemoryProperties);
    MDAPY_INSTANCE_FUNCTION(CreateDevice);
    MDAPY_INSTANCE_FUNCTION(GetDeviceProcAddr);

    // a GPU with a compute queue, discrete ones first; never the processor,
    // where mdapy's own threads trace faster
    std::uint32_t count = 0;
    check(vk.EnumeratePhysicalDevices(d.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> physicals(count);
    check(vk.EnumeratePhysicalDevices(d.instance, &count, physicals.data()), "vkEnumeratePhysicalDevices");
    const auto rank = [&](VkPhysicalDevice p) {
        VkPhysicalDeviceProperties properties{};
        vk.GetPhysicalDeviceProperties(p, &properties);
        switch (properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return 0;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return 1;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return 2;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return 4;
            default:
                return 3;
        }
    };
    std::stable_sort(physicals.begin(), physicals.end(),
                     [&](VkPhysicalDevice a, VkPhysicalDevice b) { return rank(a) < rank(b); });
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    for (VkPhysicalDevice p : physicals) {
        if (rank(p) == 4) {
            break;
        }
        std::uint32_t families = 0;
        vk.GetPhysicalDeviceQueueFamilyProperties(p, &families, nullptr);
        std::vector<VkQueueFamilyProperties> properties(families);
        vk.GetPhysicalDeviceQueueFamilyProperties(p, &families, properties.data());
        for (std::uint32_t f = 0; f < families && physical == VK_NULL_HANDLE; ++f) {
            if ((properties[f].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                physical = p;
                d.family = f;
            }
        }
        if (physical != VK_NULL_HANDLE) {
            break;
        }
    }
    if (physical == VK_NULL_HANDLE) {
        throw std::runtime_error("no Vulkan GPU");
    }
    VkPhysicalDeviceProperties properties{};
    vk.GetPhysicalDeviceProperties(physical, &properties);
    d.max_range = properties.limits.maxStorageBufferRange;
    vk.GetPhysicalDeviceMemoryProperties(physical, &d.memory);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = d.family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vk.CreateDevice(physical, &device_info, nullptr, &d.device), "vkCreateDevice");
    MDAPY_DEVICE_FUNCTION(GetDeviceQueue);
    MDAPY_DEVICE_FUNCTION(CreateShaderModule);
    MDAPY_DEVICE_FUNCTION(DestroyShaderModule);
    MDAPY_DEVICE_FUNCTION(CreateDescriptorSetLayout);
    MDAPY_DEVICE_FUNCTION(CreatePipelineLayout);
    MDAPY_DEVICE_FUNCTION(CreateComputePipelines);
    MDAPY_DEVICE_FUNCTION(CreateBuffer);
    MDAPY_DEVICE_FUNCTION(DestroyBuffer);
    MDAPY_DEVICE_FUNCTION(GetBufferMemoryRequirements);
    MDAPY_DEVICE_FUNCTION(AllocateMemory);
    MDAPY_DEVICE_FUNCTION(FreeMemory);
    MDAPY_DEVICE_FUNCTION(BindBufferMemory);
    MDAPY_DEVICE_FUNCTION(MapMemory);
    MDAPY_DEVICE_FUNCTION(CreateDescriptorPool);
    MDAPY_DEVICE_FUNCTION(DestroyDescriptorPool);
    MDAPY_DEVICE_FUNCTION(AllocateDescriptorSets);
    MDAPY_DEVICE_FUNCTION(UpdateDescriptorSets);
    MDAPY_DEVICE_FUNCTION(CreateCommandPool);
    MDAPY_DEVICE_FUNCTION(DestroyCommandPool);
    MDAPY_DEVICE_FUNCTION(AllocateCommandBuffers);
    MDAPY_DEVICE_FUNCTION(BeginCommandBuffer);
    MDAPY_DEVICE_FUNCTION(EndCommandBuffer);
    MDAPY_DEVICE_FUNCTION(CmdCopyBuffer);
    MDAPY_DEVICE_FUNCTION(CmdPipelineBarrier);
    MDAPY_DEVICE_FUNCTION(CmdBindPipeline);
    MDAPY_DEVICE_FUNCTION(CmdBindDescriptorSets);
    MDAPY_DEVICE_FUNCTION(CmdPushConstants);
    MDAPY_DEVICE_FUNCTION(CmdDispatch);
    MDAPY_DEVICE_FUNCTION(CreateFence);
    MDAPY_DEVICE_FUNCTION(DestroyFence);
    MDAPY_DEVICE_FUNCTION(ResetFences);
    MDAPY_DEVICE_FUNCTION(WaitForFences);
    MDAPY_DEVICE_FUNCTION(QueueSubmit);
    vk.GetDeviceQueue(d.device, d.family, 0, &d.queue);

    // the scene, three outputs and the nodes of the hierarchy
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (std::uint32_t b = 0; b < 5; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = 5;
    set_info.pBindings = bindings;
    check(vk.CreateDescriptorSetLayout(d.device, &set_info, nullptr, &d.set_layout), "vkCreateDescriptorSetLayout");
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(Tile);
    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &d.set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    check(vk.CreatePipelineLayout(d.device, &layout_info, nullptr, &d.layout), "vkCreatePipelineLayout");

    // the words of the SPIR-V, aligned as Vulkan wants them
    std::vector<std::uint32_t> code(sizeof(kSpirv) / 4);
    std::memcpy(code.data(), kSpirv, 4 * code.size());
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = 4 * code.size();
    module_info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check(vk.CreateShaderModule(d.device, &module_info, nullptr, &module), "vkCreateShaderModule");
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = d.layout;
    const VkResult made = vk.CreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &d.pipeline);
    vk.DestroyShaderModule(d.device, module, nullptr);
    check(made, "vkCreateComputePipelines");
}

#undef MDAPY_INSTANCE_FUNCTION
#undef MDAPY_DEVICE_FUNCTION

Device& shared() {
    static Device d;
    static std::once_flag once;
    std::call_once(once, [] {
        try {
            open(d);
        } catch (const std::exception& e) {
            d.error = e.what();
            d.pipeline = VK_NULL_HANDLE;
        }
    });
    return d;
}

// The resources of one picture, freed when it is done or fails: the GPU is
// idle then, as every submission is waited for.
class Picture {
public:
    explicit Picture(Device& d) : d_(d) {}
    Picture(const Picture&) = delete;
    Picture& operator=(const Picture&) = delete;
    ~Picture() {
        const Functions& vk = d_.vk;
        if (fence_ != VK_NULL_HANDLE) {
            vk.DestroyFence(d_.device, fence_, nullptr);
        }
        if (commands_ != VK_NULL_HANDLE) {
            vk.DestroyCommandPool(d_.device, commands_, nullptr);
        }
        if (descriptors_ != VK_NULL_HANDLE) {
            vk.DestroyDescriptorPool(d_.device, descriptors_, nullptr);
        }
        for (const Buffer& b : buffers_) {
            vk.DestroyBuffer(d_.device, b.buffer, nullptr);
            vk.FreeMemory(d_.device, b.memory, nullptr);
        }
    }

    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
    };

    // A buffer of the bytes given, in memory with the properties needed,
    // and those wanted when there is such; mapped when visible to the host.
    Buffer buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags need,
                  VkMemoryPropertyFlags want) {
        const Functions& vk = d_.vk;
        Buffer b;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vk.CreateBuffer(d_.device, &info, nullptr, &b.buffer), "vkCreateBuffer");
        buffers_.push_back(b);
        VkMemoryRequirements requirements{};
        vk.GetBufferMemoryRequirements(d_.device, b.buffer, &requirements);
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type(requirements.memoryTypeBits, need, want);
        if (vk.AllocateMemory(d_.device, &allocation, nullptr, &b.memory) != VK_SUCCESS) {
            throw std::runtime_error("no GPU memory for the picture");
        }
        buffers_.back().memory = b.memory;
        check(vk.BindBufferMemory(d_.device, b.buffer, b.memory, 0), "vkBindBufferMemory");
        if ((need & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            check(vk.MapMemory(d_.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
        }
        return b;
    }

    VkDescriptorSet descriptors(const Buffer* buffers) {
        const Functions& vk = d_.vk;
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &size;
        check(vk.CreateDescriptorPool(d_.device, &pool_info, nullptr, &descriptors_), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_info.descriptorPool = descriptors_;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &d_.set_layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        check(vk.AllocateDescriptorSets(d_.device, &set_info, &set), "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo infos[5]{};
        VkWriteDescriptorSet writes[5]{};
        for (std::uint32_t b = 0; b < 5; ++b) {
            infos[b].buffer = buffers[b].buffer;
            infos[b].range = VK_WHOLE_SIZE;
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        vk.UpdateDescriptorSets(d_.device, 5, writes, 0, nullptr);
        return set;
    }

    VkCommandBuffer command_buffer() {
        const Functions& vk = d_.vk;
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = d_.family;
        check(vk.CreateCommandPool(d_.device, &pool_info, nullptr, &commands_), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = commands_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer buffer = VK_NULL_HANDLE;
        check(vk.AllocateCommandBuffers(d_.device, &info, &buffer), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check(vk.CreateFence(d_.device, &fence_info, nullptr, &fence_), "vkCreateFence");
        return buffer;
    }

    // Runs the commands recorded and waits for them.
    void submit(VkCommandBuffer buffer) {
        const Functions& vk = d_.vk;
        VkSubmitInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &buffer;
        check(vk.QueueSubmit(d_.queue, 1, &info, fence_), "the GPU (vkQueueSubmit)");
        check(vk.WaitForFences(d_.device, 1, &fence_, VK_TRUE, UINT64_MAX), "the GPU (vkWaitForFences)");
        check(vk.ResetFences(d_.device, 1, &fence_), "vkResetFences");
    }

private:
    std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags need, VkMemoryPropertyFlags want) const {
        for (const VkMemoryPropertyFlags flags : {need | want, need}) {
            for (std::uint32_t i = 0; i < d_.memory.memoryTypeCount; ++i) {
                if ((bits & (1U << i)) != 0 && (d_.memory.memoryTypes[i].propertyFlags & flags) == flags) {
                    return i;
                }
            }
        }
        throw std::runtime_error("no GPU memory for the picture");
    }

    Device& d_;
    std::vector<Buffer> buffers_;
    VkDescriptorPool descriptors_ = VK_NULL_HANDLE;
    VkCommandPool commands_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

void barrier(const Functions& vk, VkCommandBuffer commands, VkPipelineStageFlags from, VkAccessFlags written,
             VkPipelineStageFlags to, VkAccessFlags read) {
    VkMemoryBarrier memory{};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory.srcAccessMask = written;
    memory.dstAccessMask = read;
    vk.CmdPipelineBarrier(commands, from, to, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

}  // namespace

bool available() { return shared().pipeline != VK_NULL_HANDLE; }

void trace(const Geometry& g, const Params& params, float* rgba, float* depth, std::int64_t* ids,
           const Control& control) {
    Device& d = shared();
    if (d.pipeline == VK_NULL_HANDLE) {
        throw std::runtime_error("no GPU: " + d.error);
    }
    const std::lock_guard<std::mutex> lock(d.mutex);
    const Functions& vk = d.vk;
    const auto width = static_cast<std::size_t>(params.width);
    const auto height = static_cast<std::size_t>(params.height);

    // the scene in words, the parameters first, and the nodes of the
    // hierarchy apart, which the shader reads in quads
    struct Part {
        const void* data;
        std::size_t words;
    };
    const Part parts[] = {
        {g.sphere_center, 3 * g.spheres},     {g.sphere_radius, g.spheres},
        {g.sphere_color, 3 * g.spheres},      {g.sphere_id, g.spheres},
        {g.cylinder_end, 6 * g.cylinders},    {g.cylinder_radius, g.cylinders},
        {g.cylinder_color, 3 * g.cylinders},  {g.cylinder_id, g.cylinders},
        {g.triangle_vertex, 9 * g.triangles}, {g.triangle_color, 3 * g.triangles},
        {g.triangle_alpha, g.triangles},      {g.triangle_id, g.triangles},
        {g.refs, params.refs},
    };
    Tile tile{};
    static_assert(sizeof(tile.arrays) == sizeof(parts) / sizeof(Part) * 4, "an offset for each array");
    std::size_t words = 0;
    for (std::size_t k = 0; k < sizeof(parts) / sizeof(Part); ++k) {
        tile.arrays[k] = static_cast<std::uint32_t>(words);
        words += parts[k].words;
    }
    const std::size_t scene_bytes = sizeof(Params) + 4 * std::max<std::size_t>(words, 1);
    const std::size_t node_bytes = 32 * std::max<std::size_t>(g.node_count, 1);
    if (scene_bytes > d.max_range || node_bytes > d.max_range) {
        throw std::runtime_error("the scene is too large for the GPU's buffers");
    }

    // rows in tiles: first of some million paths, then twice as many rows
    // after a tile shorter than 0.1 s and half as many after one longer than
    // 0.4 s, so the GPU is busy yet each dispatch short, with cancellation and
    // progress between them; at most 2^22 subpixels, whose colors fit in the
    // smallest storage buffers of Vulkan
    const std::size_t paths = width * static_cast<std::size_t>(std::max(1, params.samples)) *
                              static_cast<std::size_t>(1 + params.bounces);
    const std::size_t most = std::min(height, std::max<std::size_t>(1, (std::size_t{1} << 22) / width));
    std::size_t rows = std::min(most, std::max<std::size_t>(1, (std::size_t{1} << 21) / paths));

    Picture picture(d);
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const Picture::Buffer staging = picture.buffer(scene_bytes + node_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, 0);
    auto* bytes = static_cast<unsigned char*>(staging.mapped);
    std::memcpy(bytes, &params, sizeof(Params));
    for (std::size_t k = 0; k < sizeof(parts) / sizeof(Part); ++k) {
        if (parts[k].words > 0) {
            std::memcpy(bytes + sizeof(Params) + 4 * std::size_t{tile.arrays[k]}, parts[k].data, 4 * parts[k].words);
        }
    }
    if (g.node_count > 0) {
        std::memcpy(bytes + scene_bytes, g.nodes, 32 * g.node_count);
    }
    const std::size_t subpixels = most * width;
    const Picture::Buffer buffers[5] = {
        picture.buffer(scene_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0),
        picture.buffer(16 * subpixels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
        picture.buffer(4 * subpixels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
        picture.buffer(8 * subpixels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
        picture.buffer(node_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0),
    };
    const VkDescriptorSet set = picture.descriptors(buffers);
    const VkCommandBuffer commands = picture.command_buffer();
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    check(vk.BeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
    const VkBufferCopy copy{0, 0, scene_bytes};
    vk.CmdCopyBuffer(commands, staging.buffer, buffers[0].buffer, 1, &copy);
    const VkBufferCopy node_copy{scene_bytes, 0, node_bytes};
    vk.CmdCopyBuffer(commands, staging.buffer, buffers[4].buffer, 1, &node_copy);
    barrier(vk, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    check(vk.EndCommandBuffer(commands), "vkEndCommandBuffer");
    picture.submit(commands);

    for (std::size_t row = 0; row < height;) {
        if (control.cancel != nullptr && control.cancel->cancelled()) {
            throw Cancelled();
        }
        const std::size_t count = std::min(rows, height - row);
        tile.first_row = static_cast<std::uint32_t>(row);
        tile.rows = static_cast<std::uint32_t>(count);
        const auto started = std::chrono::steady_clock::now();
        check(vk.BeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
        vk.CmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
        vk.CmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, d.layout, 0, 1, &set, 0, nullptr);
        vk.CmdPushConstants(commands, d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Tile), &tile);
        vk.CmdDispatch(commands, static_cast<std::uint32_t>((width + 7) / 8),
                       static_cast<std::uint32_t>((count + 7) / 8), 1);
        barrier(vk, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        check(vk.EndCommandBuffer(commands), "vkEndCommandBuffer");
        picture.submit(commands);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const std::size_t first = row * width, traced = count * width;
        std::memcpy(rgba + 4 * first, buffers[1].mapped, 16 * traced);
        std::memcpy(depth + first, buffers[2].mapped, 4 * traced);
        const auto* pairs = static_cast<const std::int32_t*>(buffers[3].mapped);
        for (std::size_t p = 0; p < traced; ++p) {
            const std::int32_t kind = pairs[2 * p];
            ids[first + p] = kind < 0 ? -1 : (static_cast<std::int64_t>(kind) << 32) + pairs[2 * p + 1];
        }
        row += count;
        if (control.monitor &&
            !control.monitor(static_cast<std::int64_t>(row), static_cast<std::int64_t>(height))) {
            throw Cancelled();
        }
        if (seconds < 0.1) {
            rows = std::min(most, 2 * rows);
        } else if (seconds > 0.4) {
            rows = std::max<std::size_t>(1, rows / 2);
        }
    }
}

}  // namespace mdapy::visualization::gpu

#endif  // MDAPY_VULKAN
