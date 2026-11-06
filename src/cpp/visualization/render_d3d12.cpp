// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The rays of a picture traced with Direct3D 12 on Windows: the shader is
// render_d3d12.hlsl, compiled to DXIL twice, with its own hierarchy and with the
// ray queries of GPUs that have ray tracing units. The libraries of the system
// are opened when first used, so mdapy loads where they are missing.

#include "visualization/render_gpu.h"

#ifdef MDAPY_D3D12

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// kDxil and kDxilRayQuery: the bytes of render_d3d12.dxil and
// render_d3d12_rt.dxil, written by CMake
#include "visualization/render_d3d12_dxil.h"

namespace mdapy::visualization::gpu {
namespace {

using Microsoft::WRL::ComPtr;

// Root constants: where each array of the scene starts, in words after the
// parameters, the rows of one dispatch and the kinds of the geometries of the
// hierarchy of the GPU; the shader declares the same.
struct Tile {
    std::uint32_t arrays[13];
    std::uint32_t first_row, rows;
    std::uint32_t geometry_kinds;
};

static_assert(sizeof(Params) == 236, "the shader reads the scene after the parameters, at this offset");

void check(HRESULT result, const char* what) {
    if (FAILED(result)) {
        throw std::runtime_error(std::string(what) + " failed with HRESULT " +
                                 std::to_string(static_cast<unsigned long>(result)));
    }
}

// The GPU, its queue and the compiled shaders, made once and kept for the life
// of the process; one picture at a time on its queue.
struct Device {
    ComPtr<ID3D12Device5> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> own;      // the shader's own hierarchy
    ComPtr<ID3D12PipelineState> queried;  // ray queries, on a GPU with ray tracing units
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList4> commands;
    ComPtr<ID3D12Fence> fence;
    HANDLE done = nullptr;
    std::uint64_t value = 0;
    std::string name, error;
    std::mutex mutex;
};

// Whether the GPU traces rays in hardware for compute shaders: Shader Model 6.5
// and the second tier of ray tracing, unless MDAPY_GPU_RAY_TRACING=0.
bool has_ray_queries(ID3D12Device* device) {
    if (!ray_tracing_allowed()) {
        return false;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_5};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options))) &&
           options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1 &&
           SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) &&
           model.HighestShaderModel >= D3D_SHADER_MODEL_6_5;
}

ComPtr<ID3D12PipelineState> make_pipeline(Device& d, const unsigned char* dxil, std::size_t bytes) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC info{};
    info.pRootSignature = d.root.Get();
    info.CS = {dxil, bytes};
    ComPtr<ID3D12PipelineState> pipeline;
    check(d.device->CreateComputePipelineState(&info, IID_PPV_ARGS(&pipeline)), "CreateComputePipelineState");
    return pipeline;
}

template <typename F>
F function(HMODULE library, const char* name) {
    const FARPROC found = library != nullptr ? GetProcAddress(library, name) : nullptr;
    if (found == nullptr) {
        throw std::runtime_error(std::string("no ") + name + " in the system");
    }
    return reinterpret_cast<F>(reinterpret_cast<void*>(found));
}

void open(Device& d) {
    const HMODULE d3d12 = LoadLibraryW(L"d3d12.dll"), dxgi = LoadLibraryW(L"dxgi.dll");
    const auto create_device = function<PFN_D3D12_CREATE_DEVICE>(d3d12, "D3D12CreateDevice");
    const auto serialize =
        function<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(d3d12, "D3D12SerializeRootSignature");
    using CreateFactory = HRESULT(WINAPI*)(UINT, REFIID, void**);
    const auto create_factory = function<CreateFactory>(dxgi, "CreateDXGIFactory2");

    ComPtr<IDXGIFactory6> factory;
    check(create_factory(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    // the GPU of the highest performance that can; never the processor, where
    // mdapy's own threads trace faster. A virtual machine without a GPU offers the
    // Basic Render Driver without the flag of software: the adapters of Microsoft
    // itself all run on the processor.
    constexpr UINT kMicrosoft = 0x1414;
    // a driver on the processor only for tests on machines without a GPU
    const bool software = GetEnvironmentVariableA("MDAPY_SOFTWARE_GPU", nullptr, 0) > 0;
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 description{};
    for (UINT k = 0; factory->EnumAdapterByGpuPreference(k, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                         IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++k) {
        adapter->GetDesc1(&description);
        const bool on_the_processor = (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
                                      description.VendorId == kMicrosoft;
        if ((software || !on_the_processor) &&
            SUCCEEDED(create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d.device)))) {
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_2};
            if (SUCCEEDED(d.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) &&
                model.HighestShaderModel >= D3D_SHADER_MODEL_6_2) {
                break;
            }
            d.device.Reset();
        }
        adapter.Reset();
    }
    if (d.device == nullptr) {
        throw std::runtime_error("no GPU with Direct3D 12 and Shader Model 6.2");
    }

    // the buffers as root parameters, so that no heap of descriptors is needed
    D3D12_ROOT_PARAMETER parameters[7]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants = {0, 0, sizeof(Tile) / 4};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[1].Descriptor = {0, 0};
    for (UINT u = 0; u < 3; ++u) {
        parameters[2 + u].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[2 + u].Descriptor = {u, 0};
    }
    parameters[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[5].Descriptor = {1, 0};
    parameters[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[6].Descriptor = {2, 0};
    D3D12_ROOT_SIGNATURE_DESC signature{};
    signature.NumParameters = 7;
    signature.pParameters = parameters;
    ComPtr<ID3DBlob> blob, message;
    check(serialize(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &message), "D3D12SerializeRootSignature");
    check(d.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&d.root)),
          "CreateRootSignature");

    D3D12_COMMAND_QUEUE_DESC queue{};
    queue.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    check(d.device->CreateCommandQueue(&queue, IID_PPV_ARGS(&d.queue)), "CreateCommandQueue");
    check(d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&d.allocator)),
          "CreateCommandAllocator");
    check(d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, d.allocator.Get(), nullptr,
                                      IID_PPV_ARGS(&d.commands)),
          "CreateCommandList");
    check(d.commands->Close(), "closing a command list");
    check(d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d.fence)), "CreateFence");
    d.done = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    d.own = make_pipeline(d, kDxil, sizeof(kDxil));
    char name[256] = {0};
    WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
    d.name = std::string(name) + " (Direct3D 12)";
    if (has_ray_queries(d.device.Get())) {
        d.queried = make_pipeline(d, kDxilRayQuery, sizeof(kDxilRayQuery));
        d.name = std::string(name) + " (Direct3D 12, ray tracing)";
    }
}

Device& shared() {
    // never destroyed: the libraries of the system may go first when the process ends
    static Device& d = *new Device();
    static std::once_flag once;
    std::call_once(once, [] {
        try {
            open(d);
        } catch (const std::exception& e) {
            d.error = e.what();
            d.own.Reset();
            d.queried.Reset();
        }
    });
    return d;
}

// Runs the commands recorded and waits for them.
void submit(Device& d) {
    check(d.commands->Close(), "closing a command list");
    ID3D12CommandList* lists[] = {d.commands.Get()};
    d.queue->ExecuteCommandLists(1, lists);
    check(d.queue->Signal(d.fence.Get(), ++d.value), "the GPU (Signal)");
    if (d.fence->GetCompletedValue() < d.value) {
        check(d.fence->SetEventOnCompletion(d.value, d.done), "the GPU (SetEventOnCompletion)");
        WaitForSingleObject(d.done, INFINITE);
    }
    check(d.device->GetDeviceRemovedReason(), "the GPU");
}

void record(Device& d) {
    check(d.allocator->Reset(), "resetting commands");
    check(d.commands->Reset(d.allocator.Get(), nullptr), "resetting commands");
}

// A buffer of the bytes given, in the memory of the kind given, mapped unless in
// the memory of the device; not zeroed, as all are written before they are read.
ComPtr<ID3D12Resource> make_buffer(Device& d, std::size_t bytes, D3D12_HEAP_TYPE kind, D3D12_RESOURCE_STATES state,
                                   bool writable, void** mapped = nullptr) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = kind;
    D3D12_RESOURCE_DESC info{};
    info.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    info.Width = bytes;
    info.Height = 1;
    info.DepthOrArraySize = 1;
    info.MipLevels = 1;
    info.SampleDesc.Count = 1;
    info.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    info.Flags = writable ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> buffer;
    if (FAILED(d.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_CREATE_NOT_ZEROED, &info, state, nullptr,
                                                 IID_PPV_ARGS(&buffer)))) {
        throw std::runtime_error("no GPU memory for the picture");
    }
    if (mapped != nullptr) {
        check(buffer->Map(0, nullptr, mapped), "mapping a buffer");
    }
    return buffer;
}

void transition(ID3D12GraphicsCommandList* commands, ID3D12Resource* buffer, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Transition.pResource = buffer;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    commands->ResourceBarrier(1, &barrier);
}

void written(ID3D12GraphicsCommandList* commands, ID3D12Resource* buffer) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = buffer;
    commands->ResourceBarrier(1, &barrier);
}

// The hierarchy of the GPU over the boxes and what holds it until the picture
// is done: a bottom level of the kinds of primitives there are, a top level of
// one instance of it.
struct Hierarchy {
    std::vector<ComPtr<ID3D12Resource>> kept;
    D3D12_GPU_VIRTUAL_ADDRESS address = 0;
    std::uint32_t kinds = 0;
};

Hierarchy build_hierarchy(Device& d, const Geometry& g, ID3D12Resource* boxes) {
    Hierarchy h;
    const std::size_t counts[3] = {g.spheres, g.cylinders, g.triangles};
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometries;
    std::size_t first = 0;
    for (std::uint32_t k = 0; k < 3; ++k) {
        if (counts[k] > 0) {
            D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
            geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
            geometry.AABBs.AABBCount = counts[k];
            geometry.AABBs.AABBs.StartAddress = boxes->GetGPUVirtualAddress() + sizeof(D3D12_RAYTRACING_AABB) * first;
            geometry.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);
            h.kinds |= k << (2 * geometries.size());
            geometries.push_back(geometry);
        }
        first += counts[k];
    }
    const auto build = [&](D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& inputs) {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizes{};
        d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &sizes);
        const ComPtr<ID3D12Resource> result =
            make_buffer(d, sizes.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, true);
        const ComPtr<ID3D12Resource> scratch = make_buffer(d, sizes.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                                           D3D12_RESOURCE_STATE_COMMON, true);
        h.kept.push_back(result);
        h.kept.push_back(scratch);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC info{};
        info.Inputs = inputs;
        info.DestAccelerationStructureData = result->GetGPUVirtualAddress();
        info.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
        d.commands->BuildRaytracingAccelerationStructure(&info, 0, nullptr);
        written(d.commands.Get(), result.Get());
        return result->GetGPUVirtualAddress();
    };
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottom{};
    bottom.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottom.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    bottom.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bottom.NumDescs = static_cast<UINT>(geometries.size());
    bottom.pGeometryDescs = geometries.data();
    const D3D12_GPU_VIRTUAL_ADDRESS lower = build(bottom);

    void* mapped = nullptr;
    const ComPtr<ID3D12Resource> instances = make_buffer(d, sizeof(D3D12_RAYTRACING_INSTANCE_DESC), D3D12_HEAP_TYPE_UPLOAD,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, false, &mapped);
    h.kept.push_back(instances);
    D3D12_RAYTRACING_INSTANCE_DESC instance{};
    instance.Transform[0][0] = instance.Transform[1][1] = instance.Transform[2][2] = 1.0f;
    instance.InstanceMask = 0xFF;
    instance.AccelerationStructure = lower;
    std::memcpy(mapped, &instance, sizeof(instance));
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS top{};
    top.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    top.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    top.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    top.NumDescs = 1;
    top.InstanceDescs = instances->GetGPUVirtualAddress();
    h.address = build(top);
    return h;
}

}  // namespace

bool available() { return shared().own != nullptr; }

bool uses_hierarchy() { return shared().queried == nullptr; }

std::string device() { return available() ? shared().name : std::string(); }

void trace(const Geometry& g, const Params& params, float* rgba, float* depth, std::int64_t* ids,
           const Control& control) {
    Device& d = shared();
    if (d.own == nullptr) {
        throw std::runtime_error("no GPU: " + d.error);
    }
    const std::lock_guard<std::mutex> lock(d.mutex);
    const auto width = static_cast<std::size_t>(params.width);
    const auto height = static_cast<std::size_t>(params.height);
    // the ray tracing units of the GPU where there are any and something to hit
    const bool queries = d.queried != nullptr && params.refs > 0;

    // the scene in words, the parameters first; the nodes of the hierarchy
    // apart, or the boxes of the primitives for the ray tracing units
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
        {g.refs, queries ? 0 : params.refs},
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
    const std::size_t box_bytes =
        queries ? sizeof(D3D12_RAYTRACING_AABB) * (g.spheres + g.cylinders + g.triangles) : 0;
    // what the shaders address in 32 bits
    if (scene_bytes > 0xFFFFFFFFu || node_bytes > 0xFFFFFFFFu) {
        throw std::runtime_error("the scene is too large for the GPU's buffers");
    }

    // rows in tiles: first of some million paths, then twice as many rows
    // after a tile shorter than 0.1 s and half as many after one longer than
    // 0.4 s, so the GPU is busy yet each dispatch short, far from the seconds
    // after which Windows resets a GPU, with cancellation and progress between
    // them; at most 2^22 subpixels
    const std::size_t paths = width * static_cast<std::size_t>(std::max(1, params.samples)) *
                              static_cast<std::size_t>(1 + params.bounces);
    const std::size_t most = std::min(height, std::max<std::size_t>(1, (std::size_t{1} << 22) / width));
    std::size_t rows = std::min(most, std::max<std::size_t>(1, (std::size_t{1} << 21) / paths));

    const std::size_t apart = queries ? box_bytes : node_bytes;
    void* staged = nullptr;
    const ComPtr<ID3D12Resource> staging = make_buffer(d, scene_bytes + apart, D3D12_HEAP_TYPE_UPLOAD,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ, false, &staged);
    auto* bytes = static_cast<unsigned char*>(staged);
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
    const ComPtr<ID3D12Resource> scene =
        make_buffer(d, scene_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, false);
    const ComPtr<ID3D12Resource> nodes =
        make_buffer(d, apart, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, false);
    // written in the memory of the GPU, read back after each tile
    const std::size_t sizes[3] = {16 * subpixels, 4 * subpixels, 8 * subpixels};
    ComPtr<ID3D12Resource> outputs[3], read_back[3];
    void* results[3] = {nullptr, nullptr, nullptr};
    const int wanted = depth != nullptr ? 3 : 1;  // the depths and the ids are for outlines
    for (int k = 0; k < 3; ++k) {
        outputs[k] = make_buffer(d, sizes[k], D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true);
        if (k < wanted) {
            read_back[k] = make_buffer(d, sizes[k], D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false,
                                       &results[k]);
        }
    }

    record(d);
    d.commands->CopyBufferRegion(scene.Get(), 0, staging.Get(), 0, scene_bytes);
    d.commands->CopyBufferRegion(nodes.Get(), 0, staging.Get(), scene_bytes, apart);
    Hierarchy hierarchy;
    if (queries) {
        transition(d.commands.Get(), nodes.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        hierarchy = build_hierarchy(d, g, nodes.Get());
        tile.geometry_kinds = hierarchy.kinds;
    }
    submit(d);

    for (std::size_t row = 0; row < height;) {
        if (control.cancel != nullptr && control.cancel->cancelled()) {
            throw Cancelled();
        }
        const std::size_t count = std::min(rows, height - row);
        tile.first_row = static_cast<std::uint32_t>(row);
        tile.rows = static_cast<std::uint32_t>(count);
        const auto started = std::chrono::steady_clock::now();
        record(d);
        d.commands->SetComputeRootSignature(d.root.Get());
        d.commands->SetPipelineState(queries ? d.queried.Get() : d.own.Get());
        d.commands->SetComputeRoot32BitConstants(0, sizeof(Tile) / 4, &tile, 0);
        d.commands->SetComputeRootShaderResourceView(1, scene->GetGPUVirtualAddress());
        for (UINT k = 0; k < 3; ++k) {
            d.commands->SetComputeRootUnorderedAccessView(2 + k, outputs[k]->GetGPUVirtualAddress());
        }
        d.commands->SetComputeRootShaderResourceView(5, nodes->GetGPUVirtualAddress());
        if (queries) {
            d.commands->SetComputeRootShaderResourceView(6, hierarchy.address);
        }
        d.commands->Dispatch(static_cast<UINT>((width + 7) / 8), static_cast<UINT>((count + 7) / 8), 1);
        const std::size_t first = row * width, traced = count * width;
        const std::size_t each[3] = {16, 4, 8};
        for (int k = 0; k < wanted; ++k) {
            transition(d.commands.Get(), outputs[k].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            d.commands->CopyBufferRegion(read_back[k].Get(), 0, outputs[k].Get(), 0, each[k] * traced);
        }
        submit(d);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::memcpy(rgba + 4 * first, results[0], 16 * traced);
        if (depth != nullptr) {
            std::memcpy(depth + first, results[1], 4 * traced);
            const auto* pairs = static_cast<const std::int32_t*>(results[2]);
            for (std::size_t p = 0; p < traced; ++p) {
                const std::int32_t kind = pairs[2 * p];
                ids[first + p] = kind < 0 ? -1 : (static_cast<std::int64_t>(kind) << 32) + pairs[2 * p + 1];
            }
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

#endif  // MDAPY_D3D12
