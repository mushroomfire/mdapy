// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Bindings of the neighbor search.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bindings/control.h"
#include "core/parallel.h"
#include "neighbor/cell_list.h"

namespace mdapy::bindings {
namespace {

using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Vector3 = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Flags = nb::ndarray<const bool, nb::shape<3>, nb::c_contig, nb::device::cpu>;

// A NumPy array that owns the memory of a vector, without copying it.
template <class T>
nb::object to_array(std::vector<T>&& values, std::size_t rows, std::size_t columns) {
    auto* owned = new std::vector<T>(std::move(values));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    const std::size_t shape[2] = {rows, columns};
    return nb::ndarray<nb::numpy, T>(owned->data(), columns == 0 ? 1 : 2, shape, owner).cast();
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
    const std::size_t count = pairs.neighbors.size();
    return nb::make_tuple(to_array(std::move(pairs.offsets), static_cast<std::size_t>(n) + 1, 0),
                          to_array(std::move(pairs.neighbors), count, 0),
                          to_array(std::move(pairs.shifts), count, 3),
                          to_array(std::move(pairs.distances), count, 0));
}

using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Distances = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::tuple order_by_distance(Offsets offsets, Distances distances, std::int64_t count,
                            nb::handle progress, const CancelToken* cancel) {
    if (offsets.shape(0) < 1 ||
        offsets.data()[offsets.shape(0) - 1] != static_cast<std::int64_t>(distances.shape(0))) {
        throw std::invalid_argument("the offsets do not fit the distances");
    }
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    std::vector<std::int64_t> order, kept;
    run_released(
        [&](const Control& control) {
            neighbor::order_by_distance(n, offsets.data(), distances.data(), count, order, kept,
                                        control);
        },
        progress, cancel);
    const std::size_t size = order.size();
    return nb::make_tuple(to_array(std::move(order), size, 0),
                          to_array(std::move(kept), static_cast<std::size_t>(n) + 1, 0));
}

nb::tuple nearest(Positions positions, Matrix matrix, Vector3 origin, Flags pbc, int k,
                  nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    neighbor::NeighborPairs pairs;
    run_released(
        [&](const Control& control) {
            pairs = neighbor::nearest(positions.data(), n, matrix.data(), origin.data(), pbc.data(),
                                      k, control);
        },
        progress, cancel);
    const std::size_t count = pairs.neighbors.size();
    return nb::make_tuple(to_array(std::move(pairs.offsets), static_cast<std::size_t>(n) + 1, 0),
                          to_array(std::move(pairs.neighbors), count, 0),
                          to_array(std::move(pairs.shifts), count, 3),
                          to_array(std::move(pairs.distances), count, 0));
}

using Neighbors = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Shifts = nb::ndarray<const std::int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

nb::object pair_vectors(Positions positions, Offsets offsets, Neighbors neighbors, Shifts shifts,
                        Matrix matrix) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the pairs do not fit the atoms");
    }
    std::vector<double> vectors(3 * neighbors.shape(0));
    {
        nb::gil_scoped_release release;
        neighbor::pair_vectors(positions.data(), n, offsets.data(), neighbors.data(),
                               shifts.data(), matrix.data(), vectors.data());
    }
    return to_array(std::move(vectors), neighbors.shape(0), 3);
}

nb::tuple follow(Positions positions, Offsets offsets, Neighbors neighbors, Shifts shifts,
                 Matrix matrix, double cutoff, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the pairs do not fit the atoms");
    }
    neighbor::NeighborPairs pairs;
    std::vector<double> vectors;
    run_released(
        [&](const Control& control) {
            pairs = neighbor::follow(positions.data(), n, offsets.data(), neighbors.data(),
                                     shifts.data(), matrix.data(), cutoff, vectors, control);
        },
        progress, cancel);
    const std::size_t count = pairs.neighbors.size();
    return nb::make_tuple(to_array(std::move(pairs.offsets), static_cast<std::size_t>(n) + 1, 0),
                          to_array(std::move(pairs.neighbors), count, 0),
                          to_array(std::move(pairs.shifts), count, 3),
                          to_array(std::move(pairs.distances), count, 0),
                          to_array(std::move(vectors), count, 3));
}

nb::tuple nearest_sites(Positions sites, Matrix matrix, Vector3 origin, Flags pbc,
                        Positions points, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(sites.shape(0));
    const auto m = static_cast<std::int64_t>(points.shape(0));
    std::vector<std::int64_t> site(static_cast<std::size_t>(m));
    std::vector<double> distance2(static_cast<std::size_t>(m));
    run_released(
        [&](const Control& control) {
            neighbor::nearest_sites(sites.data(), n, matrix.data(), origin.data(), pbc.data(),
                                    points.data(), m, site.data(), distance2.data(), control);
        },
        progress, cancel);
    return nb::make_tuple(to_array(std::move(site), static_cast<std::size_t>(m), 0),
                          to_array(std::move(distance2), static_cast<std::size_t>(m), 0));
}

}  // namespace

void bind_neighbor(nb::module_& m) {
    m.def("build", &build, nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"),
          nb::arg("pbc"), nb::arg("cutoff"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("nearest", &nearest, nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"),
          nb::arg("pbc"), nb::arg("k"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("pair_vectors", &pair_vectors, nb::arg("positions"), nb::arg("offsets"),
          nb::arg("neighbors"), nb::arg("shifts"), nb::arg("matrix"));
    m.def("follow", &follow, nb::arg("positions"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("matrix"), nb::arg("cutoff"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("nearest_sites", &nearest_sites, nb::arg("sites"), nb::arg("matrix"), nb::arg("origin"),
          nb::arg("pbc"), nb::arg("points"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("order_by_distance", &order_by_distance, nb::arg("offsets"), nb::arg("distances"),
          nb::arg("count") = -1, nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
