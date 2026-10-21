// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Bindings of the neighbor search.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
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

}  // namespace

void bind_neighbor(nb::module_& m) {
    m.def("build", &build, nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"),
          nb::arg("pbc"), nb::arg("cutoff"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
