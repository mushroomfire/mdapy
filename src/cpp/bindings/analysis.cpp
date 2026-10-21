// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Bindings of the analyses.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "analysis/geometry.h"
#include "analysis/steinhardt.h"
#include "bindings/control.h"
#include "core/parallel.h"

namespace mdapy::bindings {
namespace {

using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vectors = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

nb::object steinhardt(Offsets offsets, Neighbors neighbors, Vectors vectors,
                      const std::vector<int>& degrees, bool average, nb::handle progress,
                      const CancelToken* cancel) {
    if (offsets.shape(0) < 1 || neighbors.shape(0) != vectors.shape(0) ||
        offsets.data()[offsets.shape(0) - 1] != static_cast<std::int64_t>(vectors.shape(0))) {
        throw std::invalid_argument("the offsets do not fit the vectors");
    }
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    auto* q = new double[static_cast<std::size_t>(n) * degrees.size()];
    nb::capsule owner(q, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(n), degrees.size()};
    nb::object result = nb::ndarray<nb::numpy, double>(q, 2, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::steinhardt(n, offsets.data(), neighbors.data(), vectors.data(), degrees,
                                 average, q, control);
        },
        progress, cancel);
    return result;
}

nb::tuple clusters(Offsets offsets, Neighbors neighbors) {
    if (offsets.shape(0) < 1 ||
        offsets.data()[offsets.shape(0) - 1] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the offsets do not fit the neighbors");
    }
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    auto* cluster = new std::int64_t[static_cast<std::size_t>(n)];
    nb::capsule owner(cluster, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
    std::vector<std::int64_t> sizes(static_cast<std::size_t>(n));
    std::int64_t count;
    {
        nb::gil_scoped_release release;
        count = analysis::clusters(n, offsets.data(), neighbors.data(), cluster, sizes.data());
    }
    const std::size_t shape[1] = {static_cast<std::size_t>(n)};
    auto* found = new std::int64_t[static_cast<std::size_t>(count)];
    std::copy(sizes.begin(), sizes.begin() + count, found);
    nb::capsule found_owner(found, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
    const std::size_t found_shape[1] = {static_cast<std::size_t>(count)};
    return nb::make_tuple(nb::ndarray<nb::numpy, std::int64_t>(cluster, 1, shape, owner),
                          nb::ndarray<nb::numpy, std::int64_t>(found, 1, found_shape, found_owner));
}

nb::object bond_angles(Offsets offsets, Vectors vectors, int bins, nb::handle progress,
                       const CancelToken* cancel) {
    if (offsets.shape(0) < 1 ||
        offsets.data()[offsets.shape(0) - 1] != static_cast<std::int64_t>(vectors.shape(0))) {
        throw std::invalid_argument("the offsets do not fit the vectors");
    }
    if (bins < 1) {
        throw std::invalid_argument("bins must be at least 1");
    }
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    auto* counts = new std::int64_t[static_cast<std::size_t>(bins)];
    nb::capsule owner(counts, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
    const std::size_t shape[1] = {static_cast<std::size_t>(bins)};
    nb::object result = nb::ndarray<nb::numpy, std::int64_t>(counts, 1, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::bond_angles(n, offsets.data(), vectors.data(), bins, counts, control);
        },
        progress, cancel);
    return result;
}

}  // namespace

void bind_analysis(nb::module_& m) {
    m.def("steinhardt", &steinhardt, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("vectors"), nb::arg("degrees"), nb::arg("average"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("clusters", &clusters, nb::arg("offsets"), nb::arg("neighbors"));
    m.def("bond_angles", &bond_angles, nb::arg("offsets"), nb::arg("vectors"), nb::arg("bins"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
