// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The rays of a picture traced on a GPU, with Metal on macOS and Vulkan on
// Linux: the same samples as on the processor, in single precision as there,
// so pictures differ only by rounding.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/parallel.h"

namespace mdapy::visualization::gpu {

// Everything the trace of a subpixel needs besides the scene, as plain
// numbers; the shader declares the same layout.
struct Params {
    float center[3], forward[3], right[3], up[3];
    float half_width, half_height, aperture, focal_distance;
    float start, nearest, farthest, epsilon;
    float key[3], highlight[3], fill[3], key_tangent[3], key_bitangent[3];
    float ambient, diffuse, specular, shininess, metalness, fill_intensity, base_light;
    float ao_distance, ao_strength, shadow_strength, fog, light_size, cone;
    float background[3];
    std::int32_t perspective, ao_samples, shadows, flat, toon_levels, bounces, samples;
    std::int32_t width, height;  // subpixels
    std::uint32_t refs;          // references in the hierarchy
};

// The scene and its bounding volume hierarchy: nodes of 8 words (lo,
// first, hi, count) and the references of the leaves, kind << 30 | index.
struct Geometry {
    const float* sphere_center;
    const float* sphere_radius;
    const float* sphere_color;
    const std::int32_t* sphere_id;
    std::size_t spheres;
    const float* cylinder_end;
    const float* cylinder_radius;
    const float* cylinder_color;
    const std::int32_t* cylinder_id;
    std::size_t cylinders;
    const float* triangle_vertex;
    const float* triangle_color;
    const float* triangle_alpha;
    const std::int32_t* triangle_id;
    std::size_t triangles;
    const void* nodes;
    std::size_t node_count;
    const std::uint32_t* refs;
};

// Whether a GPU can trace pictures.
bool available();

// The GPU that traces pictures and how, or empty when there is none.
std::string device();

// The subpixels of a picture, width x height of params: premultiplied RGBA
// (4 each), the depth and the id of the first hit (-1 for none), as the
// processor computes them. Throws std::runtime_error when no GPU can.
void trace(const Geometry& geometry, const Params& params, float* rgba, float* depth,
           std::int64_t* ids, const Control& control);

}  // namespace mdapy::visualization::gpu
