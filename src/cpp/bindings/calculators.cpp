// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Bindings of the compiled calculators.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>

#include "bindings/control.h"
#include "calculators/eam.h"
#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::bindings {
namespace {

using Table = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Species = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vectors = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::object new_array(std::size_t ndim, const std::size_t* shape, T*& data) {
    std::size_t size = 1;
    for (std::size_t k = 0; k < ndim; ++k) {
        size *= shape[k];
    }
    data = new T[size];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(data, ndim, shape, owner).cast();
}

// embedding: (elements, nrho); density: (elements, nr); pair: (elements *
// (elements + 1) / 2, nr), the lower triangle by rows.
calculators::EamPotential make_eam(Table embedding, Table density, Table pair, double drho,
                                   double dr, double cutoff) {
    calculators::EamPotential potential;
    const auto elements = static_cast<int>(embedding.shape(0));
    if (density.shape(0) != embedding.shape(0) ||
        pair.shape(0) != static_cast<std::size_t>(elements * (elements + 1) / 2) ||
        pair.shape(1) != density.shape(1)) {
        throw std::invalid_argument("inconsistent EAM tables");
    }
    if (!(drho > 0.0) || !(dr > 0.0) || !(cutoff > 0.0)) {
        throw std::invalid_argument("EAM table spacings and cutoff must be positive");
    }
    potential.elements = elements;
    potential.cutoff = cutoff;
    const auto nrho = static_cast<int>(embedding.shape(1));
    const auto nr = static_cast<int>(density.shape(1));
    potential.rho_max = (nrho - 1) * drho;
    for (int e = 0; e < elements; ++e) {
        potential.embedding.emplace_back(embedding.data() + e * nrho, nrho, drho);
        potential.density.emplace_back(density.data() + e * nr, nr, dr);
    }
    for (std::size_t k = 0; k < pair.shape(0); ++k) {
        potential.pair.emplace_back(pair.data() + k * nr, nr, dr);
    }
    return potential;
}

nb::tuple compute_eam(const calculators::EamPotential& potential, Species species,
                      Offsets offsets, Neighbors neighbors, Vectors vectors, nb::handle progress,
                      const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(species.shape(0));
    if (offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != vectors.shape(0)) {
        throw std::invalid_argument("the neighbor list does not fit the atoms");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (species.data()[i] < 0 || species.data()[i] >= potential.elements) {
            throw std::invalid_argument("a species index is outside the potential");
        }
    }
    double *energies, *forces, *virials;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = new_array(3, shape3, virials);
    run_released(
        [&](const Control& control) {
            calculators::compute_eam(potential, n, species.data(), offsets.data(),
                                     neighbors.data(), vectors.data(), energies, forces, virials,
                                     control);
        },
        progress, cancel);
    return nb::make_tuple(e, f, w);
}

}  // namespace

void bind_calculators(nb::module_& m) {
    nb::class_<calculators::EamPotential>(m, "EamPotential")
        .def_ro("elements", &calculators::EamPotential::elements)
        .def_ro("cutoff", &calculators::EamPotential::cutoff)
        .def_ro("rho_max", &calculators::EamPotential::rho_max);
    m.def("make_eam", &make_eam, nb::arg("embedding"), nb::arg("density"), nb::arg("pair"),
          nb::arg("drho"), nb::arg("dr"), nb::arg("cutoff"));
    m.def("compute_eam", &compute_eam, nb::arg("potential"), nb::arg("species"),
          nb::arg("offsets"), nb::arg("neighbors"), nb::arg("vectors"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
