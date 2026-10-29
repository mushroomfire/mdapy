// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Bindings of the neighbor search.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bindings/control.h"
#include "bindings/frames.h"
#include "core/buffer.h"
#include "core/index.h"
#include "core/parallel.h"
#include "neighbor/cell_list.h"
#include "neighbor/frames.h"
#include "neighbor/kd_tree.h"

namespace mdapy::bindings {
namespace {

using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Vector3 = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Flags = nb::ndarray<const bool, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Distances = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Shifts = nb::ndarray<const std::int8_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Types = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Limits = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using MutableNeighbors = nb::ndarray<index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using MutableShifts = nb::ndarray<std::int8_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using MutableDistances = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// A NumPy array that owns the memory of a vector, without copying it.
template <class T, class A>
nb::object to_array(std::vector<T, A>&& values, std::initializer_list<std::size_t> shape) {
    auto* owned = new std::vector<T, A>(std::move(values));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T, A>*>(p); });
    return nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.begin(), owner).cast();
}

nb::tuple to_arrays(neighbor::NeighborPairs&& pairs) {
    const std::size_t n = pairs.offsets.size(), count = pairs.neighbors.size();
    return nb::make_tuple(to_array(std::move(pairs.offsets), {n}),
                          to_array(std::move(pairs.neighbors), {count}),
                          to_array(std::move(pairs.shifts), {count, 3}),
                          to_array(std::move(pairs.distances), {count}));
}

void check_pairs(std::int64_t n, Offsets offsets, Neighbors neighbors, Shifts shifts) {
    if (offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the pairs do not fit the atoms");
    }
}

nb::tuple build(Positions positions, Matrix matrix, Vector3 origin, Flags pbc, double cutoff,
                nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    neighbor::NeighborPairs pairs;
    run_released(
        [&](const Control& control) {
            pairs = neighbor::build(positions.data(), n, matrix.data(), origin.data(), pbc.data(),
                                    cutoff, control);
        },
        progress, cancel);
    return to_arrays(std::move(pairs));
}

// For each of many frames, the closest pair over the sum of their radii,
// where below close, else infinity.
nb::object closest_pairs(frames::Starts starts, frames::Positions positions, frames::Matrices matrices,
                         frames::Origins origins, frames::Flags pbc, Distances radii, double close,
                         nb::handle progress, const CancelToken* cancel) {
    const neighbor::Frames all = frames::of(starts, positions, matrices, origins, pbc);
    if (radii.shape(0) != positions.shape(0)) {
        throw std::invalid_argument("radii must have one value per atom");
    }
    std::vector<double> found;
    run_released(
        [&](const Control& control) { found = neighbor::closest_pairs(all, radii.data(), close, control); },
        progress, cancel);
    const std::size_t count = found.size();
    return to_array(std::move(found), {count});
}

nb::tuple nearest(Positions positions, Matrix matrix, Vector3 origin, Flags pbc, int k,
                  nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    neighbor::NeighborPairs pairs;
    run_released(
        [&](const Control& control) {
            pairs = neighbor::nearest(positions.data(), n, matrix.data(), origin.data(),
                                      pbc.data(), k, control);
        },
        progress, cancel);
    return to_arrays(std::move(pairs));
}

nb::tuple order_by_distance(Positions positions, Offsets offsets, Neighbors neighbors,
                            Shifts shifts, Matrix matrix, std::int64_t count, nb::handle progress,
                            const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    check_pairs(n, offsets, neighbors, shifts);
    std::vector<std::int64_t> order, kept;
    run_released(
        [&](const Control& control) {
            neighbor::order_by_distance(positions.data(), n, offsets.data(), neighbors.data(),
                                        shifts.data(), matrix.data(), count, order, kept,
                                        control);
        },
        progress, cancel);
    const std::size_t size = order.size();
    return nb::make_tuple(to_array(std::move(order), {size}),
                          to_array(std::move(kept), {static_cast<std::size_t>(n) + 1}));
}

void sort_nearest(Positions positions, Offsets offsets, MutableNeighbors neighbors,
                  MutableShifts shifts, MutableDistances distances, Matrix matrix,
                  std::int64_t count, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) || distances.shape(0) != neighbors.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the pairs do not fit the atoms");
    }
    run_released(
        [&](const Control& control) {
            neighbor::sort_nearest(positions.data(), n, offsets.data(), neighbors.data(),
                                   shifts.data(), distances.data(), matrix.data(), count,
                                   control);
        },
        progress, cancel);
}

nb::object pair_vectors(Positions positions, Offsets offsets, Neighbors neighbors, Shifts shifts,
                        Matrix matrix) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    check_pairs(n, offsets, neighbors, shifts);
    Buffer<double> vectors(3 * neighbors.shape(0));
    {
        nb::gil_scoped_release release;
        neighbor::pair_vectors(positions.data(), n, offsets.data(), neighbors.data(),
                               shifts.data(), matrix.data(), vectors.data());
    }
    return to_array(std::move(vectors), {neighbors.shape(0), 3});
}

// The largest of the values, found in parallel, or 0 for none.
double largest(Distances values) {
    const auto n = static_cast<std::int64_t>(values.shape(0));
    const double* data = values.data();
    nb::gil_scoped_release release;
    return parallel_reduce(
        n, 0, 0.0,
        [data](std::int64_t begin, std::int64_t end) {
            return *std::max_element(data + begin, data + end);
        },
        [](double a, double b) { return std::max(a, b); });
}

nb::tuple within(Offsets offsets, Neighbors neighbors, Shifts shifts, Distances distances,
                 std::optional<Types> type, Limits limits, nb::handle progress,
                 const CancelToken* cancel) {
    if (offsets.shape(0) < 1 || distances.shape(0) != neighbors.shape(0)) {
        throw std::invalid_argument("the pairs do not fit the atoms");
    }
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    check_pairs(n, offsets, neighbors, shifts);
    const auto types = static_cast<int>(limits.shape(0));
    if (types < 1 || limits.shape(1) != limits.shape(0)) {
        throw std::invalid_argument("limits must be a square table");
    }
    if (type) {
        if (type->shape(0) != static_cast<std::size_t>(n)) {
            throw std::invalid_argument("type must have a value for each atom");
        }
        for (std::int64_t i = 0; i < n; ++i) {
            if (type->data()[i] < 0 || type->data()[i] >= types) {
                throw std::invalid_argument("type must be from 0 to the size of limits");
            }
        }
    }
    neighbor::NeighborPairs pairs;
    run_released(
        [&](const Control& control) {
            pairs = neighbor::within(n, offsets.data(), neighbors.data(), shifts.data(),
                                     distances.data(), type ? type->data() : nullptr, types,
                                     limits.data(), control);
        },
        progress, cancel);
    return to_arrays(std::move(pairs));
}

nb::tuple follow(Positions positions, Offsets offsets, Neighbors neighbors, Shifts shifts,
                 Matrix matrix, double cutoff, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    check_pairs(n, offsets, neighbors, shifts);
    neighbor::NeighborPairs pairs;
    run_released(
        [&](const Control& control) {
            pairs = neighbor::follow(positions.data(), n, offsets.data(), neighbors.data(),
                                     shifts.data(), matrix.data(), cutoff, control);
        },
        progress, cancel);
    return to_arrays(std::move(pairs));
}

void make_tree(neighbor::KdTree* tree, Positions positions, Matrix matrix, Vector3 origin,
               Flags pbc) {
    run_released(
        [&](const Control&) {
            new (tree) neighbor::KdTree(positions.data(), static_cast<std::int64_t>(positions.shape(0)),
                                        matrix.data(), origin.data(), pbc.data());
        },
        nb::none(), nullptr);
}

nb::tuple tree_nearest(const neighbor::KdTree& tree, Positions points, int k, nb::handle progress,
                       const CancelToken* cancel) {
    const auto m = static_cast<std::size_t>(points.shape(0));
    const std::size_t size = m * static_cast<std::size_t>(k > 0 ? k : 0);
    Buffer<index_t> j(size);
    Buffer<std::int8_t> shifts(3 * size);
    Buffer<double> distances(size);
    run_released(
        [&](const Control& control) {
            tree.nearest_points(points.data(), static_cast<std::int64_t>(m), k, j.data(),
                                shifts.data(), distances.data(), control);
        },
        progress, cancel);
    const auto columns = static_cast<std::size_t>(k);
    return nb::make_tuple(to_array(std::move(j), {m, columns}),
                          to_array(std::move(shifts), {m, columns, 3}),
                          to_array(std::move(distances), {m, columns}));
}

}  // namespace

void bind_neighbor(nb::module_& m) {
    m.def("build", &build, nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"),
          nb::arg("pbc"), nb::arg("cutoff"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("closest_pairs", &closest_pairs, nb::arg("starts"), nb::arg("positions"),
          nb::arg("matrices"), nb::arg("origins"), nb::arg("pbc"), nb::arg("radii"),
          nb::arg("close"), nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("nearest", &nearest, nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"),
          nb::arg("pbc"), nb::arg("k"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("largest", &largest, nb::arg("values"));
    m.def("pair_vectors", &pair_vectors, nb::arg("positions"), nb::arg("offsets"),
          nb::arg("neighbors"), nb::arg("shifts"), nb::arg("matrix"));
    m.def("follow", &follow, nb::arg("positions"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("matrix"), nb::arg("cutoff"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("within", &within, nb::arg("offsets"), nb::arg("neighbors"), nb::arg("shifts"),
          nb::arg("distances"), nb::arg("type").none(), nb::arg("limits"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("sort_nearest", &sort_nearest, nb::arg("positions"), nb::arg("offsets"),
          nb::arg("neighbors"), nb::arg("shifts"), nb::arg("distances"), nb::arg("matrix"),
          nb::arg("count"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("order_by_distance", &order_by_distance, nb::arg("positions"), nb::arg("offsets"),
          nb::arg("neighbors"), nb::arg("shifts"), nb::arg("matrix"), nb::arg("count") = -1, nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    nb::class_<neighbor::KdTree>(m, "KdTree",
                                 "k-d tree of atoms in a box, searched for the nearest atom "
                                 "images of any points.")
        .def("__init__", &make_tree, nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"),
             nb::arg("pbc"))
        .def("__len__", &neighbor::KdTree::size)
        .def("nearest", &tree_nearest, nb::arg("points"), nb::arg("k"),
             nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
