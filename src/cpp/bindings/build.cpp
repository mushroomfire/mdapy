// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Bindings of the structure builder.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bindings/control.h"
#include "build/polycrystal.h"
#include "build/sqs.h"
#include "core/buffer.h"
#include "core/index.h"

namespace mdapy::bindings {
namespace {

using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Shells = nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Types = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Values = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Points = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrices = nb::ndarray<const double, nb::shape<-1, 3, 3>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Vector3 = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Spans = nb::ndarray<const std::int64_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

void check_pairs(const Offsets& offsets, const Neighbors& neighbors, const Shells& shell,
                 const Types& type, const Values& weights, int shells) {
    const auto n = static_cast<std::int64_t>(type.shape(0));
    if (static_cast<std::int64_t>(offsets.shape(0)) != n + 1 ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0)) ||
        shell.shape(0) != neighbors.shape(0) ||
        static_cast<int>(weights.shape(0)) != shells) {
        throw std::invalid_argument("the offsets, pairs, shells and weights do not fit");
    }
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
}

double sqs_objective(Offsets offsets, Neighbors neighbors, Shells shell, int shells, Types type,
                     int elements, Values weights) {
    check_pairs(offsets, neighbors, shell, type, weights, shells);
    return build::sqs_objective(static_cast<std::int64_t>(type.shape(0)), offsets.data(),
                                neighbors.data(), shell.data(), shells, elements, type.data(),
                                weights.data());
}

nb::tuple sqs(Offsets offsets, Neighbors neighbors, Shells shell, int shells, Types type,
              int elements, Values weights, std::int64_t steps, double temperature,
              std::uint64_t seed, nb::handle progress, const CancelToken* cancel) {
    check_pairs(offsets, neighbors, shell, type, weights, shells);
    const auto n = static_cast<std::int64_t>(type.shape(0));
    auto* out = new std::vector<std::int32_t>(type.data(), type.data() + n);
    nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::int32_t>*>(p); });
    build::SqsResult result{};
    run_released(
        [&](const Control& control) {
            result = build::sqs(n, offsets.data(), neighbors.data(), shell.data(), shells, elements,
                                out->data(), weights.data(), steps, temperature, seed, control);
        },
        progress, cancel);
    const std::size_t shape[1] = {static_cast<std::size_t>(n)};
    return nb::make_tuple(nb::ndarray<nb::numpy, std::int32_t>(out->data(), 1, shape, owner),
                          result.objective, result.start, result.accepted);
}

// A NumPy array that owns the memory of a buffer, without copying it.
template <class T>
nb::object to_array(Buffer<T>&& values, std::initializer_list<std::size_t> shape) {
    auto* owned = new Buffer<T>(std::move(values));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<Buffer<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.begin(), owner).cast();
}

nb::tuple fill_grains(Points seeds, Matrices rotations, Values radii, Spans spans, Offsets offsets,
                      Points faces, Matrix cell, Vector3 origin, Points sites, Offsets centers_of,
                      Points centers, int axis, double tolerance, bool copies, double margin,
                      nb::handle progress, const CancelToken* cancel) {
    const auto grains = static_cast<std::int64_t>(seeds.shape(0));
    const auto count = static_cast<std::int64_t>(sites.shape(0));
    const auto middles = static_cast<std::int64_t>(centers.shape(0));
    if (static_cast<std::int64_t>(rotations.shape(0)) != grains ||
        static_cast<std::int64_t>(radii.shape(0)) != grains ||
        static_cast<std::int64_t>(spans.shape(0)) != grains ||
        static_cast<std::int64_t>(offsets.shape(0)) != grains + 1 ||
        offsets.data()[grains] != static_cast<std::int64_t>(faces.shape(0)) ||
        static_cast<std::int64_t>(centers_of.shape(0)) != count || axis < -1 || axis > 2) {
        throw std::invalid_argument("the grains, faces and sites do not fit");
    }
    for (std::int64_t s = 0; s < count; ++s) {
        if (centers_of.data()[s] < 0 || centers_of.data()[s] >= middles) {
            throw std::invalid_argument("a site has no center");
        }
    }
    build::GrainFill fill;
    fill.grains = grains;
    fill.seeds = seeds.data();
    fill.rotations = rotations.data();
    fill.radii = radii.data();
    fill.spans = spans.data();
    fill.offsets = offsets.data();
    fill.faces = faces.data();
    fill.cell = cell.data();
    fill.origin = origin.data();
    fill.sites = count;
    fill.site_positions = sites.data();
    fill.centers_of = centers_of.data();
    fill.centers = middles;
    fill.center_positions = centers.data();
    fill.axis = axis;
    fill.tolerance = tolerance;
    fill.copies = copies;
    fill.margin = margin;
    build::GrainAtoms atoms;
    run_released([&](const Control& control) { atoms = build::fill_grains(fill, control); }, progress,
                 cancel);
    const std::size_t n = atoms.grain.size();
    nb::object copy = nb::none(), edge = nb::none();
    if (copies) {
        copy = to_array(std::move(atoms.copy), {n});
    }
    if (margin >= 0.0) {
        edge = to_array(std::move(atoms.edge), {n});
    }
    return nb::make_tuple(to_array(std::move(atoms.positions), {n, 3}),
                          to_array(std::move(atoms.grain), {n}),
                          to_array(std::move(atoms.site), {n}), copy, edge);
}

}  // namespace

void bind_build(nb::module_& m) {
    m.def("fill_grains", &fill_grains, nb::arg("seeds"), nb::arg("rotations"), nb::arg("radii"),
          nb::arg("spans"), nb::arg("offsets"), nb::arg("faces"), nb::arg("cell"), nb::arg("origin"),
          nb::arg("sites"), nb::arg("centers_of"), nb::arg("centers"), nb::arg("axis"),
          nb::arg("tolerance"), nb::arg("copies"), nb::arg("margin"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("sqs_objective", &sqs_objective, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shell"), nb::arg("shells"), nb::arg("type"), nb::arg("elements"),
          nb::arg("weights"));
    m.def("sqs", &sqs, nb::arg("offsets"), nb::arg("neighbors"), nb::arg("shell"),
          nb::arg("shells"), nb::arg("type"), nb::arg("elements"), nb::arg("weights"),
          nb::arg("steps"), nb::arg("temperature"), nb::arg("seed"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
