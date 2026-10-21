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
#include <utility>
#include <vector>

#include "analysis/distribution.h"
#include "analysis/geometry.h"
#include "analysis/steinhardt.h"
#include "analysis/strain.h"
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

using Values = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::object pair_entropy(Offsets offsets, Values distances, double cutoff, double sigma,
                        Values density, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    if (n < 0 || offsets.data()[n] != static_cast<std::int64_t>(distances.shape(0)) ||
        density.shape(0) != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("the offsets do not fit the distances");
    }
    if (!(cutoff > 0.0) || !(sigma > 0.0)) {
        throw std::invalid_argument("cutoff and sigma must be positive");
    }
    auto* entropy = new double[static_cast<std::size_t>(n)];
    nb::capsule owner(entropy, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const std::size_t shape[1] = {static_cast<std::size_t>(n)};
    nb::object result = nb::ndarray<nb::numpy, double>(entropy, 1, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::pair_entropy(n, offsets.data(), distances.data(), cutoff, sigma,
                                   density.data(), entropy, control);
        },
        progress, cancel);
    return result;
}

using Positions3 = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix3 = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Origin3 = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Periodic = nb::ndarray<const bool, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Types = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::object pair_histogram(Positions3 positions, Matrix3 matrix, Origin3 origin, Periodic pbc,
                          double r_max, int bins, Types type, int types, nb::handle progress,
                          const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (type.shape(0) != positions.shape(0) || bins < 1 || types < 1) {
        throw std::invalid_argument("the types do not fit the atoms");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (type.data()[i] < 0 || type.data()[i] >= types) {
            throw std::invalid_argument("a type is outside the range");
        }
    }
    std::vector<std::int64_t> counts;
    run_released(
        [&](const Control& control) {
            counts = analysis::pair_histogram(positions.data(), n, matrix.data(), origin.data(),
                                              pbc.data(), r_max, bins, type.data(), types,
                                              control);
        },
        progress, cancel);
    auto* kept = new std::vector<std::int64_t>(std::move(counts));
    nb::capsule owner(kept, [](void* p) noexcept {
        delete static_cast<std::vector<std::int64_t>*>(p);
    });
    const std::size_t shape[3] = {static_cast<std::size_t>(types),
                                  static_cast<std::size_t>(types), static_cast<std::size_t>(bins)};
    return nb::ndarray<nb::numpy, std::int64_t>(kept->data(), 3, shape, owner).cast();
}

nb::tuple structure_factor(Positions3 positions, Matrix3 matrix, Types type, int types,
                           double q_max, int bins, nb::handle progress,
                           const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (type.shape(0) != positions.shape(0) || bins < 1 || types < 1 || !(q_max > 0.0)) {
        throw std::invalid_argument("invalid arguments to the structure factor");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (type.data()[i] < 0 || type.data()[i] >= types) {
            throw std::invalid_argument("a type is outside the range");
        }
    }
    std::vector<double> sums;
    std::vector<std::int64_t> vectors;
    run_released(
        [&](const Control& control) {
            analysis::structure_factor(positions.data(), n, matrix.data(), type.data(), types,
                                       q_max, bins, sums, vectors, control);
        },
        progress, cancel);
    auto* kept = new std::vector<double>(std::move(sums));
    nb::capsule owner(kept, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t shape[3] = {static_cast<std::size_t>(types),
                                  static_cast<std::size_t>(types), static_cast<std::size_t>(bins)};
    auto* counted = new std::vector<std::int64_t>(std::move(vectors));
    nb::capsule counted_owner(counted, [](void* p) noexcept {
        delete static_cast<std::vector<std::int64_t>*>(p);
    });
    const std::size_t count_shape[1] = {static_cast<std::size_t>(bins)};
    return nb::make_tuple(
        nb::ndarray<nb::numpy, double>(kept->data(), 3, shape, owner),
        nb::ndarray<nb::numpy, std::int64_t>(counted->data(), 1, count_shape, counted_owner));
}

using Shifts3 = nb::ndarray<const std::int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

// A one-dimensional NumPy array that owns the memory of a vector.
template <class T>
nb::object as_array(std::vector<T>&& values) {
    auto* kept = new std::vector<T>(std::move(values));
    nb::capsule owner(kept, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    const std::size_t shape[1] = {kept->size()};
    return nb::ndarray<nb::numpy, T>(kept->data(), 1, shape, owner).cast();
}

nb::tuple atomic_strain(Offsets offsets, Neighbors neighbors, Shifts3 shifts, Positions3 reference,
                        Positions3 displacement, Matrix3 reference_matrix, Matrix3 matrix,
                        nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(reference.shape(0));
    if (offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        displacement.shape(0) != reference.shape(0) || shifts.shape(0) != neighbors.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the pairs do not fit the atoms");
    }
    std::vector<double> strain(static_cast<std::size_t>(n) * 9), nonaffine(static_cast<std::size_t>(n));
    run_released(
        [&](const Control& control) {
            analysis::atomic_strain(n, offsets.data(), neighbors.data(), shifts.data(),
                                    reference.data(), displacement.data(),
                                    reference_matrix.data(), matrix.data(), strain.data(),
                                    nonaffine.data(), control);
        },
        progress, cancel);
    auto* kept = new std::vector<double>(std::move(strain));
    nb::capsule owner(kept, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t shape[3] = {static_cast<std::size_t>(n), 3, 3};
    return nb::make_tuple(nb::ndarray<nb::numpy, double>(kept->data(), 3, shape, owner),
                          as_array(std::move(nonaffine)));
}

}  // namespace

void bind_analysis(nb::module_& m) {
    m.def("steinhardt", &steinhardt, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("vectors"), nb::arg("degrees"), nb::arg("average"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("clusters", &clusters, nb::arg("offsets"), nb::arg("neighbors"));
    m.def("atomic_strain", &atomic_strain, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("reference"), nb::arg("displacement"),
          nb::arg("reference_matrix"), nb::arg("matrix"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("structure_factor", &structure_factor, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("type"), nb::arg("types"), nb::arg("q_max"), nb::arg("bins"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("pair_histogram", &pair_histogram, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("origin"), nb::arg("pbc"), nb::arg("r_max"), nb::arg("bins"), nb::arg("type"),
          nb::arg("types"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("pair_entropy", &pair_entropy, nb::arg("offsets"), nb::arg("distances"),
          nb::arg("cutoff"), nb::arg("sigma"), nb::arg("density"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("bond_angles", &bond_angles, nb::arg("offsets"), nb::arg("vectors"), nb::arg("bins"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
