// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Bindings of the structure identification.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>

#include "bindings/control.h"
#include "core/parallel.h"
#include "identify/structure.h"

namespace mdapy::bindings {
namespace {

using Neighbors3 = nb::ndarray<const double, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu>;
using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vectors = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::object new_array(std::size_t size, T*& data) {
    data = new T[size];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    const std::size_t shape[1] = {size};
    return nb::ndarray<nb::numpy, T>(data, 1, shape, owner).cast();
}

// vectors: (n, k, 3) from each atom to its k neighbors.
nb::object centro_symmetry(Neighbors3 vectors, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(vectors.shape(0));
    const auto k = static_cast<int>(vectors.shape(1));
    if (k < 2 || k % 2 != 0) {
        throw std::invalid_argument("the number of neighbors must be even and at least 2");
    }
    double* csp;
    nb::object result = new_array(static_cast<std::size_t>(n), csp);
    run_released(
        [&](const Control& control) {
            identify::centro_symmetry(n, k, vectors.data(), csp, control);
        },
        progress, cancel);
    return result;
}

nb::object common_neighbor_analysis(Offsets offsets, Vectors vectors, double cutoff,
                                    nb::handle progress, const CancelToken* cancel) {
    if (offsets.shape(0) < 1 ||
        offsets.data()[offsets.shape(0) - 1] != static_cast<std::int64_t>(vectors.shape(0))) {
        throw std::invalid_argument("the offsets do not fit the vectors");
    }
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    std::int8_t* types;
    nb::object result = new_array(static_cast<std::size_t>(n), types);
    run_released(
        [&](const Control& control) {
            identify::common_neighbor_analysis(n, offsets.data(), vectors.data(), cutoff, types,
                                               control);
        },
        progress, cancel);
    return result;
}

using Nearest = nb::ndarray<const index_t, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu>;
using Vectors4 = nb::ndarray<const double, nb::shape<-1, 4, 3>, nb::c_contig, nb::device::cpu>;

// neighbors: (n, 4) indices of the four nearest neighbors; vectors (n, 4, 3).
nb::object identify_diamond(Nearest neighbors, Vectors4 vectors, nb::handle progress,
                            const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(neighbors.shape(0));
    if (vectors.shape(0) != neighbors.shape(0)) {
        throw std::invalid_argument("the vectors do not fit the neighbors");
    }
    for (std::int64_t p = 0; p < 4 * n; ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    std::int8_t* types;
    nb::object result = new_array(static_cast<std::size_t>(n), types);
    run_released(
        [&](const Control& control) {
            identify::identify_diamond(n, neighbors.data(), vectors.data(), types, control);
        },
        progress, cancel);
    return result;
}

using Pairs = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::object chill_plus(Offsets offsets, Pairs neighbors, Vectors vectors, nb::handle progress,
                      const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    if (n < 0 || neighbors.shape(0) != vectors.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(vectors.shape(0))) {
        throw std::invalid_argument("the offsets do not fit the vectors");
    }
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    std::int8_t* types;
    nb::object result = new_array(static_cast<std::size_t>(n), types);
    run_released(
        [&](const Control& control) {
            identify::chill_plus(n, offsets.data(), neighbors.data(), vectors.data(), types, control);
        },
        progress, cancel);
    return result;
}

}  // namespace

void bind_identify(nb::module_& m) {
    m.def("centro_symmetry", &centro_symmetry, nb::arg("vectors"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("common_neighbor_analysis", &common_neighbor_analysis, nb::arg("offsets"),
          nb::arg("vectors"), nb::arg("cutoff"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("chill_plus", &chill_plus, nb::arg("offsets"), nb::arg("neighbors"), nb::arg("vectors"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("identify_diamond", &identify_diamond, nb::arg("neighbors"), nb::arg("vectors"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
