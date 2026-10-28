// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Bindings of the renderer.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "bindings/control.h"
#include "visualization/render.h"

namespace mdapy::bindings {
namespace {

template <int Columns>
using Rows = nb::ndarray<const float, nb::shape<-1, Columns>, nb::c_contig, nb::device::cpu>;
using Floats = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Ids = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Three = nb::ndarray<const float, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Four = nb::ndarray<const float, nb::shape<4>, nb::c_contig, nb::device::cpu>;

template <class T, class A>
std::vector<T> copy(const A& array) {
    return std::vector<T>(array.data(), array.data() + array.size());
}

visualization::Scene make_scene(Rows<3> sphere_center, Floats sphere_radius, Rows<3> sphere_color,
                                Ids sphere_id, Rows<6> cylinder_end, Floats cylinder_radius,
                                Rows<3> cylinder_color, Ids cylinder_id, Rows<9> triangle_vertex,
                                Rows<3> triangle_color, Floats triangle_alpha, Ids triangle_id) {
    const std::size_t spheres = sphere_center.shape(0);
    const std::size_t cylinders = cylinder_end.shape(0);
    const std::size_t triangles = triangle_vertex.shape(0);
    if (sphere_radius.shape(0) != spheres || sphere_color.shape(0) != spheres ||
        sphere_id.shape(0) != spheres || cylinder_radius.shape(0) != cylinders ||
        cylinder_color.shape(0) != cylinders || cylinder_id.shape(0) != cylinders ||
        triangle_color.shape(0) != triangles || triangle_alpha.shape(0) != triangles ||
        triangle_id.shape(0) != triangles) {
        throw std::invalid_argument("the arrays of the primitives differ in length");
    }
    visualization::Scene scene;
    scene.sphere_center = copy<float>(sphere_center);
    scene.sphere_radius = copy<float>(sphere_radius);
    scene.sphere_color = copy<float>(sphere_color);
    scene.sphere_id = copy<std::int32_t>(sphere_id);
    scene.cylinder_end = copy<float>(cylinder_end);
    scene.cylinder_radius = copy<float>(cylinder_radius);
    scene.cylinder_color = copy<float>(cylinder_color);
    scene.cylinder_id = copy<std::int32_t>(cylinder_id);
    scene.triangle_vertex = copy<float>(triangle_vertex);
    scene.triangle_color = copy<float>(triangle_color);
    scene.triangle_alpha = copy<float>(triangle_alpha);
    scene.triangle_id = copy<std::int32_t>(triangle_id);
    return scene;
}

std::array<float, 3> three(const Three& a) { return {a.data()[0], a.data()[1], a.data()[2]}; }

nb::object render(Rows<3> sphere_center, Floats sphere_radius, Rows<3> sphere_color, Ids sphere_id,
                  Rows<6> cylinder_end, Floats cylinder_radius, Rows<3> cylinder_color,
                  Ids cylinder_id, Rows<9> triangle_vertex, Rows<3> triangle_color,
                  Floats triangle_alpha, Ids triangle_id, Three center, Three forward, Three up,
                  float half_width, float half_height, bool perspective, float aperture,
                  float focal_distance, int width, int height, float ambient, float diffuse,
                  float specular, float shininess, float metalness, Three key, Three highlight, Three fill,
                  float fill_intensity, int ao_samples, float ao_distance, float ao_strength,
                  bool shadows, float shadow_strength, bool flat, int toon_levels,
                  float outline_width, float outline_depth, Three outline_color, float fog,
                  Four background, int supersample, int bounces, int samples, float light_size,
                  float base_light, bool srgb, bool gpu, nb::handle progress, const CancelToken* cancel) {
    const visualization::Scene scene =
        make_scene(sphere_center, sphere_radius, sphere_color, sphere_id, cylinder_end,
                   cylinder_radius, cylinder_color, cylinder_id, triangle_vertex, triangle_color,
                   triangle_alpha, triangle_id);
    visualization::Camera camera;
    camera.center = three(center);
    camera.forward = three(forward);
    camera.up = three(up);
    camera.half_width = half_width;
    camera.half_height = half_height;
    camera.perspective = perspective;
    camera.aperture = aperture;
    camera.focal_distance = focal_distance;
    visualization::Style style;
    style.ambient = ambient;
    style.diffuse = diffuse;
    style.specular = specular;
    style.shininess = shininess;
    style.metalness = metalness;
    style.key = three(key);
    style.highlight = three(highlight);
    style.fill = three(fill);
    style.fill_intensity = fill_intensity;
    style.ao_samples = ao_samples;
    style.ao_distance = ao_distance;
    style.ao_strength = ao_strength;
    style.shadows = shadows;
    style.shadow_strength = shadow_strength;
    style.flat = flat;
    style.toon_levels = toon_levels;
    style.outline_width = outline_width;
    style.outline_depth = outline_depth;
    style.outline_color = three(outline_color);
    style.fog = fog;
    style.background = {background.data()[0], background.data()[1], background.data()[2],
                        background.data()[3]};
    style.supersample = supersample;
    style.bounces = bounces;
    style.samples = samples;
    style.light_size = light_size;
    style.base_light = base_light;
    style.srgb = srgb;
    style.gpu = gpu;
    auto* image = new std::vector<std::uint8_t>();
    nb::capsule owner(image, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
    run_released(
        [&](const Control& control) {
            *image = visualization::render(scene, camera, style, width, height, control);
        },
        progress, cancel);
    const std::size_t shape[3] = {static_cast<std::size_t>(height), static_cast<std::size_t>(width), 4};
    return nb::ndarray<nb::numpy, std::uint8_t>(image->data(), 3, shape, owner).cast();
}

nb::tuple trace(Rows<3> sphere_center, Floats sphere_radius, Rows<3> sphere_color, Ids sphere_id,
                Rows<6> cylinder_end, Floats cylinder_radius, Rows<3> cylinder_color,
                Ids cylinder_id, Rows<9> triangle_vertex, Rows<3> triangle_color,
                Floats triangle_alpha, Ids triangle_id, Rows<3> origins, Rows<3> directions) {
    const visualization::Scene scene =
        make_scene(sphere_center, sphere_radius, sphere_color, sphere_id, cylinder_end,
                   cylinder_radius, cylinder_color, cylinder_id, triangle_vertex, triangle_color,
                   triangle_alpha, triangle_id);
    const auto rays = static_cast<std::int64_t>(origins.shape(0));
    if (directions.shape(0) != origins.shape(0)) {
        throw std::invalid_argument("origins and directions differ in length");
    }
    std::vector<float> distance(static_cast<std::size_t>(rays));
    std::vector<std::int32_t> kind(static_cast<std::size_t>(rays));
    std::vector<std::int64_t> index(static_cast<std::size_t>(rays));
    visualization::trace(scene, rays, origins.data(), directions.data(), distance.data(),
                         kind.data(), index.data());
    const auto wrap = [](auto&& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        auto* kept = new std::vector<T>(std::move(values));
        nb::capsule owner(kept, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
        const std::size_t shape[1] = {kept->size()};
        return nb::ndarray<nb::numpy, T>(kept->data(), 1, shape, owner).cast();
    };
    return nb::make_tuple(wrap(std::move(distance)), wrap(std::move(kind)), wrap(std::move(index)));
}

}  // namespace

void bind_visualization(nb::module_& m) {
    m.def("render", &render, nb::arg("sphere_center"), nb::arg("sphere_radius"),
          nb::arg("sphere_color"), nb::arg("sphere_id"), nb::arg("cylinder_end"),
          nb::arg("cylinder_radius"), nb::arg("cylinder_color"), nb::arg("cylinder_id"),
          nb::arg("triangle_vertex"), nb::arg("triangle_color"), nb::arg("triangle_alpha"),
          nb::arg("triangle_id"), nb::arg("center"), nb::arg("forward"), nb::arg("up"),
          nb::arg("half_width"), nb::arg("half_height"), nb::arg("perspective"),
          nb::arg("aperture"), nb::arg("focal_distance"), nb::arg("width"),
          nb::arg("height"), nb::arg("ambient"), nb::arg("diffuse"),
          nb::arg("specular"), nb::arg("shininess"), nb::arg("metalness"), nb::arg("key"),
          nb::arg("highlight"), nb::arg("fill"),
          nb::arg("fill_intensity"), nb::arg("ao_samples"), nb::arg("ao_distance"),
          nb::arg("ao_strength"), nb::arg("shadows"), nb::arg("shadow_strength"),
          nb::arg("flat"), nb::arg("toon_levels"), nb::arg("outline_width"),
          nb::arg("outline_depth"), nb::arg("outline_color"), nb::arg("fog"),
          nb::arg("background"), nb::arg("supersample"), nb::arg("bounces") = 0,
          nb::arg("samples") = 1, nb::arg("light_size") = 0.0f, nb::arg("base_light") = 0.0f,
          nb::arg("srgb") = false, nb::arg("gpu") = false,
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("gpu_available", &visualization::gpu_available,
          "Whether pictures can be traced on a GPU.");
    m.def("trace", &trace, nb::arg("sphere_center"), nb::arg("sphere_radius"),
          nb::arg("sphere_color"), nb::arg("sphere_id"), nb::arg("cylinder_end"),
          nb::arg("cylinder_radius"), nb::arg("cylinder_color"), nb::arg("cylinder_id"),
          nb::arg("triangle_vertex"), nb::arg("triangle_color"), nb::arg("triangle_alpha"),
          nb::arg("triangle_id"), nb::arg("origins"), nb::arg("directions"));
}

}  // namespace mdapy::bindings
