// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// A ray tracer for pictures of atoms: spheres, cylinders and triangles in a
// bounding volume hierarchy, shaded with ambient occlusion, shadows and
// outlines.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/parallel.h"

namespace mdapy::visualization {

// The primitives of a picture. Colors are linear RGB from 0 to 1. Each
// primitive has an id; outlines separate primitives of different ids.
struct Scene {
    std::vector<float> sphere_center;  // 3 per sphere
    std::vector<float> sphere_radius;
    std::vector<float> sphere_color;  // 3 per sphere
    std::vector<std::int32_t> sphere_id;
    std::vector<float> cylinder_end;  // 6 per cylinder: both ends
    std::vector<float> cylinder_radius;
    std::vector<float> cylinder_color;
    std::vector<std::int32_t> cylinder_id;
    std::vector<float> triangle_vertex;  // 9 per triangle
    std::vector<float> triangle_color;
    std::vector<float> triangle_alpha;  // opacity from 0 to 1
    std::vector<std::int32_t> triangle_id;
};

// An orthographic camera looks along forward at a window of half_width by
// half_height around center; a perspective camera stands at center and
// half_width and half_height are the tangents of half its angles of view.
// A perspective camera with an aperture > 0 sees through a lens of that
// radius, sharp at focal_distance along forward only (depth of field).
struct Camera {
    std::array<float, 3> center{}, forward{0, 0, -1}, up{0, 1, 0};
    float half_width = 1, half_height = 1;
    bool perspective = false;
    float aperture = 0, focal_distance = 1;
};

struct Style {
    // lights: an ambient term, a key light and a fill light, their
    // directions toward the lights in the frame of the camera (right, up,
    // back toward the viewer)
    float ambient = 0.35f, diffuse = 0.75f, specular = 0.25f, shininess = 40.0f;
    std::array<float, 3> key{-0.45f, 0.6f, 0.66f}, fill{0.6f, -0.2f, 0.77f};
    float fill_intensity = 0.25f;
    // ambient occlusion: rays per sample, how far they look, how dark
    int ao_samples = 8;
    float ao_distance = 4.0f, ao_strength = 0.8f;
    bool shadows = true;
    float shadow_strength = 0.5f;
    // flat: base colors without lights, darkened by the occlusion only;
    // toon_levels > 0 quantizes the diffuse light
    bool flat = false;
    int toon_levels = 0;
    // outlines where the primitive changes and the depth jumps
    float outline_width = 0.0f;  // in pixels
    float outline_depth = 0.5f;  // smallest jump of depth, in length units
    std::array<float, 3> outline_color{0, 0, 0};
    float fog = 0.0f;  // fraction of the background mixed in at the far end
    // global illumination: with bounces > 0, the light of each sample is
    // traced along a path that bounces diffusely off up to bounces more
    // primitives, lit by a sky of brightness ambient and by the key light of
    // brightness diffuse. The fill light, the occlusion, flat and
    // toon_levels are not used then.
    int bounces = 0;
    // rays per subpixel, the first through its middle, the others anywhere
    // in it, for smooth soft shadows, depth of field and global illumination
    int samples = 1;
    // the angular radius of the key light (radians): a disk for soft shadows
    float light_size = 0.0f;
    std::array<float, 4> background{1, 1, 1, 1};
    int supersample = 2;  // samples per pixel along each axis
    bool gpu = false;     // trace the rays on a GPU (see gpu_available)
};

// Whether pictures can be traced on a GPU: Metal on macOS.
bool gpu_available();

// An image of width by height pixels, RGBA in 8 bits, sRGB, rows from the top.
std::vector<std::uint8_t> render(const Scene& scene, const Camera& camera, const Style& style,
                                 int width, int height, const Control& control = {});

// The nearest hit of each ray of a scene: its distance (infinity for none)
// and the kind (0 sphere, 1 cylinder, 2 triangle, -1 none) and index of the
// primitive; for the tests.
void trace(const Scene& scene, std::int64_t rays, const float* origins, const float* directions,
           float* distance, std::int32_t* kind, std::int64_t* index);

}  // namespace mdapy::visualization
