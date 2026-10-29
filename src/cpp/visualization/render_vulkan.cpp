// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The rays of a picture traced with Vulkan on Linux: the shader is
// render_vulkan.comp, compiled to SPIR-V twice, with its own hierarchy and
// with the ray queries of GPUs that have ray tracing units. The Vulkan
// loader is opened when first used, so mdapy needs no Vulkan to load; the
// build needs its headers.

#include "visualization/render_gpu.h"

#ifdef MDAPY_VULKAN

#define VK_NO_PROTOTYPES
#include <dlfcn.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// kSpirv and kSpirvRayQuery: the bytes of render_vulkan.spv and
// render_vulkan_rt.spv, written by CMake
#include "visualization/render_vulkan_spirv.h"

namespace mdapy::visualization::gpu {
namespace {

// Push constants: where each array of the scene starts, in words after the
// parameters, and the rows of one dispatch; the shader declares the same.
struct Tile {
    std::uint32_t arrays[13];
    std::uint32_t first_row, rows;
};

// The functions of Vulkan used here, found in the loader; those of ray
// queries only for a GPU that has them.
struct Functions {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
    PFN_vkEnumerateInstanceVersion EnumerateInstanceVersion = nullptr;
    PFN_vkCreateInstance CreateInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2 = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
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
    PFN_vkGetBufferDeviceAddress GetBufferDeviceAddress = nullptr;
    PFN_vkCreateAccelerationStructureKHR CreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR DestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR GetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR GetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAccelerationStructuresKHR = nullptr;
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

// The device extensions of ray queries.
const char* const kRayExtensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

// A compute pipeline of the shader and its layouts.
struct Pipeline {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

// The GPU, its queue and the compiled shaders, made once and kept for the
// life of the process; one picture at a time on its queue.
struct Device {
    Functions vk;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    VkDeviceSize max_range = 0;
    std::uint32_t family = 0;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    Pipeline own;      // the shader's own hierarchy
    Pipeline queried;  // ray queries, on a GPU with ray tracing units
    std::string name, error;
    std::mutex mutex;
};

#define MDAPY_INSTANCE_FUNCTION(name) \
    find(vk.name, vk.GetInstanceProcAddr(d.instance, "vk" #name), "vk" #name)
#define MDAPY_DEVICE_FUNCTION(name) find(vk.name, vk.GetDeviceProcAddr(d.device, "vk" #name), "vk" #name)

// Whether the GPU traces rays in hardware for compute shaders: Vulkan 1.2,
// ray queries and acceleration structures, unless MDAPY_GPU_RAY_TRACING=0.
bool has_ray_queries(const Functions& vk, VkPhysicalDevice physical, std::uint32_t instance_version,
                     const VkPhysicalDeviceProperties& properties) {
    const char* setting = std::getenv("MDAPY_GPU_RAY_TRACING");
    if ((setting != nullptr && std::strcmp(setting, "0") == 0) || instance_version < VK_API_VERSION_1_2 ||
        properties.apiVersion < VK_API_VERSION_1_2 || vk.GetPhysicalDeviceFeatures2 == nullptr) {
        return false;
    }
    std::uint32_t count = 0;
    vk.EnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vk.EnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
    for (const char* wanted : kRayExtensions) {
        if (std::none_of(extensions.begin(), extensions.end(),
                         [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, wanted) == 0; })) {
            return false;
        }
    }
    VkPhysicalDeviceRayQueryFeaturesKHR queries{};
    queries.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR structures{};
    structures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    structures.pNext = &queries;
    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.pNext = &structures;
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &vulkan12;
    vk.GetPhysicalDeviceFeatures2(physical, &features);
    return vulkan12.bufferDeviceAddress && structures.accelerationStructure && queries.rayQuery;
}

// The pipeline of a shader: the scene, three outputs and the nodes of the
// hierarchy; with ray queries, the hierarchy of the GPU too.
Pipeline make_pipeline(Device& d, const unsigned char* spirv, std::size_t bytes, bool queries) {
    const Functions& vk = d.vk;
    Pipeline p;
    VkDescriptorSetLayoutBinding bindings[6]{};
    const std::uint32_t count = queries ? 6 : 5;
    for (std::uint32_t b = 0; b < count; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = b < 5 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = count;
    set_info.pBindings = bindings;
    check(vk.CreateDescriptorSetLayout(d.device, &set_info, nullptr, &p.set_layout), "vkCreateDescriptorSetLayout");
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(Tile);
    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &p.set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    check(vk.CreatePipelineLayout(d.device, &layout_info, nullptr, &p.layout), "vkCreatePipelineLayout");

    // the words of the SPIR-V, aligned as Vulkan wants them
    std::vector<std::uint32_t> code(bytes / 4);
    std::memcpy(code.data(), spirv, 4 * code.size());
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
    pipeline_info.layout = p.layout;
    const VkResult made = vk.CreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &p.pipeline);
    vk.DestroyShaderModule(d.device, module, nullptr);
    check(made, "vkCreateComputePipelines");
    return p;
}

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
    // Vulkan 1.2 for ray queries where the loader has it, 1.0 otherwise
    std::uint32_t instance_version = VK_API_VERSION_1_0;
    if (vk.GetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion") != nullptr) {
        MDAPY_INSTANCE_FUNCTION(EnumerateInstanceVersion);
        check(vk.EnumerateInstanceVersion(&instance_version), "vkEnumerateInstanceVersion");
    }
    instance_version = instance_version >= VK_API_VERSION_1_2 ? VK_API_VERSION_1_2 : VK_API_VERSION_1_0;
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "mdapy";
    application.apiVersion = instance_version;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    check(vk.CreateInstance(&instance_info, nullptr, &d.instance), "vkCreateInstance");
    MDAPY_INSTANCE_FUNCTION(EnumeratePhysicalDevices);
    MDAPY_INSTANCE_FUNCTION(GetPhysicalDeviceProperties);
    MDAPY_INSTANCE_FUNCTION(GetPhysicalDeviceQueueFamilyProperties);
    MDAPY_INSTANCE_FUNCTION(GetPhysicalDeviceMemoryProperties);
    MDAPY_INSTANCE_FUNCTION(EnumerateDeviceExtensionProperties);
    MDAPY_INSTANCE_FUNCTION(CreateDevice);
    MDAPY_INSTANCE_FUNCTION(GetDeviceProcAddr);
    if (instance_version >= VK_API_VERSION_1_2) {
        MDAPY_INSTANCE_FUNCTION(GetPhysicalDeviceFeatures2);
    }

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
    const bool queries = has_ray_queries(vk, physical, instance_version, properties);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = d.family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceRayQueryFeaturesKHR query_features{};
    query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    query_features.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR structure_features{};
    structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    structure_features.pNext = &query_features;
    structure_features.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.pNext = &structure_features;
    vulkan12.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (queries) {
        device_info.pNext = &vulkan12;
        device_info.enabledExtensionCount = sizeof(kRayExtensions) / sizeof(kRayExtensions[0]);
        device_info.ppEnabledExtensionNames = kRayExtensions;
    }
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
    d.own = make_pipeline(d, kSpirv, sizeof(kSpirv), false);
    d.name = std::string(properties.deviceName) + " (Vulkan)";
    if (queries) {
        MDAPY_DEVICE_FUNCTION(GetBufferDeviceAddress);
        MDAPY_DEVICE_FUNCTION(CreateAccelerationStructureKHR);
        MDAPY_DEVICE_FUNCTION(DestroyAccelerationStructureKHR);
        MDAPY_DEVICE_FUNCTION(GetAccelerationStructureBuildSizesKHR);
        MDAPY_DEVICE_FUNCTION(GetAccelerationStructureDeviceAddressKHR);
        MDAPY_DEVICE_FUNCTION(CmdBuildAccelerationStructuresKHR);
        d.queried = make_pipeline(d, kSpirvRayQuery, sizeof(kSpirvRayQuery), true);
        d.name = std::string(properties.deviceName) + " (Vulkan, ray queries)";
    }
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
            d.own = Pipeline{};
            d.queried = Pipeline{};
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
        for (const VkAccelerationStructureKHR s : structures_) {
            vk.DestroyAccelerationStructureKHR(d_.device, s, nullptr);
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
        VkDeviceAddress address = 0;
    };

    // A buffer of the bytes given, in memory with the properties needed,
    // and those wanted when there is such; mapped when visible to the host;
    // with its address on the GPU when its usage asks for one.
    Buffer buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags need,
                  VkMemoryPropertyFlags want) {
        const Functions& vk = d_.vk;
        const bool addressed = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
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
        VkMemoryAllocateFlagsInfo flags{};
        flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.pNext = addressed ? &flags : nullptr;
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
        if (addressed) {
            VkBufferDeviceAddressInfo where{};
            where.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            where.buffer = b.buffer;
            b.address = vk.GetBufferDeviceAddress(d_.device, &where);
        }
        return b;
    }

    // An acceleration structure of the kind and size given and its
    // buffer.
    VkAccelerationStructureKHR structure(VkAccelerationStructureTypeKHR type, VkDeviceSize bytes) {
        const Buffer storage = buffer(bytes,
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        VkAccelerationStructureCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        info.buffer = storage.buffer;
        info.size = bytes;
        info.type = type;
        VkAccelerationStructureKHR made = VK_NULL_HANDLE;
        check(d_.vk.CreateAccelerationStructureKHR(d_.device, &info, nullptr, &made),
              "vkCreateAccelerationStructureKHR");
        structures_.push_back(made);
        return made;
    }

    // The descriptors of the five buffers, and of the hierarchy of the GPU
    // when there is one.
    VkDescriptorSet descriptors(const Pipeline& pipeline, const Buffer* buffers, VkAccelerationStructureKHR boxes) {
        const Functions& vk = d_.vk;
        const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5},
                                               {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}};
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = boxes != VK_NULL_HANDLE ? 2 : 1;
        pool_info.pPoolSizes = sizes;
        check(vk.CreateDescriptorPool(d_.device, &pool_info, nullptr, &descriptors_), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_info.descriptorPool = descriptors_;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &pipeline.set_layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        check(vk.AllocateDescriptorSets(d_.device, &set_info, &set), "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo infos[5]{};
        VkWriteDescriptorSet writes[6]{};
        for (std::uint32_t b = 0; b < 6; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            if (b < 5) {
                infos[b].buffer = buffers[b].buffer;
                infos[b].range = VK_WHOLE_SIZE;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &infos[b];
            }
        }
        VkWriteDescriptorSetAccelerationStructureKHR structure_info{};
        structure_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        structure_info.accelerationStructureCount = 1;
        structure_info.pAccelerationStructures = &boxes;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[5].pNext = &structure_info;
        vk.UpdateDescriptorSets(d_.device, boxes != VK_NULL_HANDLE ? 6 : 5, writes, 0, nullptr);
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
    std::vector<VkAccelerationStructureKHR> structures_;
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

// The bounding boxes of the primitives, kind after kind as their geometries
// list them: six numbers each, a little wider than the primitive, so that
// the tests of boxes of the GPU miss nothing the shader would hit.
void write_boxes(const Geometry& g, float* boxes) {
    const auto pad = [](float* box) {
        float largest = 0;
        for (int e = 0; e < 6; ++e) {
            largest = std::max(largest, std::fabs(box[e]));
        }
        const float margin = 1e-6f * (1.0f + largest);
        for (int e = 0; e < 3; ++e) {
            box[e] -= margin;
            box[e + 3] += margin;
        }
    };
    for (std::size_t i = 0; i < g.spheres; ++i, boxes += 6) {
        const float r = g.sphere_radius[i];
        for (int e = 0; e < 3; ++e) {
            boxes[e] = g.sphere_center[3 * i + e] - r;
            boxes[e + 3] = g.sphere_center[3 * i + e] + r;
        }
        pad(boxes);
    }
    for (std::size_t i = 0; i < g.cylinders; ++i, boxes += 6) {
        const float r = g.cylinder_radius[i];
        const float* ends = g.cylinder_end + 6 * i;
        for (int e = 0; e < 3; ++e) {
            boxes[e] = std::min(ends[e], ends[3 + e]) - r;
            boxes[e + 3] = std::max(ends[e], ends[3 + e]) + r;
        }
        pad(boxes);
    }
    for (std::size_t i = 0; i < g.triangles; ++i, boxes += 6) {
        const float* v = g.triangle_vertex + 9 * i;
        for (int e = 0; e < 3; ++e) {
            boxes[e] = std::min({v[e], v[3 + e], v[6 + e]});
            boxes[e + 3] = std::max({v[e], v[3 + e], v[6 + e]});
        }
        pad(boxes);
    }
}

// Records the builds of the hierarchy of the GPU over the boxes, whose
// buffer is filled when the commands run: a bottom level of the three
// geometries, a top level of one instance of it.
VkAccelerationStructureKHR build_hierarchy(Device& d, Picture& picture, VkCommandBuffer commands,
                                           const Geometry& g, const Picture::Buffer& boxes) {
    const Functions& vk = d.vk;
    const std::uint32_t counts[3] = {static_cast<std::uint32_t>(g.spheres), static_cast<std::uint32_t>(g.cylinders),
                                     static_cast<std::uint32_t>(g.triangles)};
    VkAccelerationStructureGeometryKHR geometries[3]{};
    VkAccelerationStructureBuildRangeInfoKHR ranges[3]{};
    std::uint32_t first = 0;
    for (int k = 0; k < 3; ++k) {
        geometries[k].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometries[k].geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        geometries[k].geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        geometries[k].geometry.aabbs.data.deviceAddress = boxes.address;
        geometries[k].geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
        ranges[k].primitiveCount = counts[k];
        ranges[k].primitiveOffset = static_cast<std::uint32_t>(sizeof(VkAabbPositionsKHR)) * first;
        first += counts[k];
    }
    // scratch at the alignment of every GPU (256 bytes at most)
    const auto scratch_address = [&](VkDeviceSize bytes) {
        const Picture::Buffer scratch =
            picture.buffer(bytes + 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        return (scratch.address + 255) & ~VkDeviceAddress{255};
    };
    const auto build = [&](VkAccelerationStructureBuildGeometryInfoKHR& info, const std::uint32_t* primitives,
                           const VkAccelerationStructureBuildRangeInfoKHR* range) {
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vk.GetAccelerationStructureBuildSizesKHR(d.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info,
                                                 primitives, &sizes);
        info.dstAccelerationStructure = picture.structure(info.type, sizes.accelerationStructureSize);
        info.scratchData.deviceAddress = scratch_address(sizes.buildScratchSize);
        vk.CmdBuildAccelerationStructuresKHR(commands, 1, &info, &range);
        return info.dstAccelerationStructure;
    };
    VkAccelerationStructureBuildGeometryInfoKHR bottom{};
    bottom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    bottom.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bottom.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bottom.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bottom.geometryCount = 3;
    bottom.pGeometries = geometries;
    const VkAccelerationStructureKHR lower = build(bottom, counts, ranges);
    barrier(vk, commands, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    VkAccelerationStructureDeviceAddressInfoKHR where{};
    where.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    where.accelerationStructure = lower;
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
    instance.mask = 0xFF;
    instance.accelerationStructureReference = vk.GetAccelerationStructureDeviceAddressKHR(d.device, &where);
    const Picture::Buffer instances = picture.buffer(
        sizeof(instance),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    std::memcpy(instances.mapped, &instance, sizeof(instance));
    VkAccelerationStructureGeometryKHR top_geometry{};
    top_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    top_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    top_geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    top_geometry.geometry.instances.data.deviceAddress = instances.address;
    VkAccelerationStructureBuildGeometryInfoKHR top{};
    top.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    top.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    top.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    top.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    top.geometryCount = 1;
    top.pGeometries = &top_geometry;
    const std::uint32_t one = 1;
    VkAccelerationStructureBuildRangeInfoKHR top_range{};
    top_range.primitiveCount = 1;
    return build(top, &one, &top_range);
}

}  // namespace

bool available() { return shared().own.pipeline != VK_NULL_HANDLE; }

std::string device() { return available() ? shared().name : std::string(); }

void trace(const Geometry& g, const Params& params, float* rgba, float* depth, std::int64_t* ids,
           const Control& control) {
    Device& d = shared();
    if (d.own.pipeline == VK_NULL_HANDLE) {
        throw std::runtime_error("no GPU: " + d.error);
    }
    const std::lock_guard<std::mutex> lock(d.mutex);
    const Functions& vk = d.vk;
    const auto width = static_cast<std::size_t>(params.width);
    const auto height = static_cast<std::size_t>(params.height);
    // the ray tracing units of the GPU where there are any and something to hit
    const bool queries = d.queried.pipeline != VK_NULL_HANDLE && params.refs > 0;
    const Pipeline& pipeline = queries ? d.queried : d.own;

    // the scene in words, the parameters first; the nodes of the hierarchy
    // apart, which the shader reads in quads, or the boxes of the primitives
    // for the ray tracing units
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
    const std::size_t node_bytes = queries ? 32 : 32 * std::max<std::size_t>(g.node_count, 1);
    const std::size_t box_bytes = queries ? sizeof(VkAabbPositionsKHR) * (g.spheres + g.cylinders + g.triangles) : 0;
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
    const std::size_t apart = queries ? box_bytes : node_bytes;
    const Picture::Buffer staging = picture.buffer(scene_bytes + apart, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, 0);
    auto* bytes = static_cast<unsigned char*>(staging.mapped);
    std::memcpy(bytes, &params, sizeof(Params));
    for (std::size_t k = 0; k < sizeof(parts) / sizeof(Part); ++k) {
        if (parts[k].words > 0) {
            std::memcpy(bytes + sizeof(Params) + 4 * std::size_t{tile.arrays[k]}, parts[k].data, 4 * parts[k].words);
        }
    }
    if (queries) {
        write_boxes(g, reinterpret_cast<float*>(bytes + scene_bytes));
    } else if (g.node_count > 0) {
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
    const VkCommandBuffer commands = picture.command_buffer();
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    check(vk.BeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
    const VkBufferCopy copy{0, 0, scene_bytes};
    vk.CmdCopyBuffer(commands, staging.buffer, buffers[0].buffer, 1, &copy);
    VkAccelerationStructureKHR hierarchy = VK_NULL_HANDLE;
    if (queries) {
        const Picture::Buffer boxes = picture.buffer(
            box_bytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        const VkBufferCopy box_copy{scene_bytes, 0, box_bytes};
        vk.CmdCopyBuffer(commands, staging.buffer, boxes.buffer, 1, &box_copy);
        barrier(vk, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        hierarchy = build_hierarchy(d, picture, commands, g, boxes);
        barrier(vk, commands, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    } else {
        const VkBufferCopy node_copy{scene_bytes, 0, node_bytes};
        vk.CmdCopyBuffer(commands, staging.buffer, buffers[4].buffer, 1, &node_copy);
        barrier(vk, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    check(vk.EndCommandBuffer(commands), "vkEndCommandBuffer");
    picture.submit(commands);
    const VkDescriptorSet set = picture.descriptors(pipeline, buffers, hierarchy);

    for (std::size_t row = 0; row < height;) {
        if (control.cancel != nullptr && control.cancel->cancelled()) {
            throw Cancelled();
        }
        const std::size_t count = std::min(rows, height - row);
        tile.first_row = static_cast<std::uint32_t>(row);
        tile.rows = static_cast<std::uint32_t>(count);
        const auto started = std::chrono::steady_clock::now();
        check(vk.BeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
        vk.CmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vk.CmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
        vk.CmdPushConstants(commands, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Tile), &tile);
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
