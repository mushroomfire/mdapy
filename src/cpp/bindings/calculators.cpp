// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Bindings of the compiled calculators.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bindings/control.h"
#include "bindings/frames.h"
#include "calculators/eam.h"
#include "calculators/nep.h"
#include "neighbor/frames.h"
#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::bindings {
namespace {

using Table = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Species = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Shifts = nb::ndarray<const std::int8_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Flat = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

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
                      Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                      Shifts shifts, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(species.shape(0));
    if (positions.shape(0) != species.shape(0) ||
        offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the neighbor list does not fit the atoms");
    }
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
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
            calculators::compute_eam(potential, n, species.data(), positions.data(),
                                     matrix.data(), offsets.data(), neighbors.data(),
                                     shifts.data(), energies, forces, virials, control);
        },
        progress, cancel);
    return nb::make_tuple(e, f, w);
}


std::vector<double> copy(Flat values) {
    return {values.data(), values.data() + values.shape(0)};
}

// sizes: types, n_radial, n_angular, k_radial, k_angular, l_max, four_body,
// five_body, neurons; the arrays flattened in the layout of NepModel.
calculators::NepModel make_nep(const std::vector<int>& sizes, Flat rc_radial, Flat rc_angular,
                               Flat w0, Flat b0, Flat w1, double b1, Flat c_radial,
                               Flat c_angular, Flat q_scaler, Flat zbl) {
    if (sizes.size() != 9) {
        throw std::invalid_argument("make_nep needs 9 sizes");
    }
    calculators::NepModel model;
    model.types = sizes[0];
    model.n_radial = sizes[1];
    model.n_angular = sizes[2];
    model.k_radial = sizes[3];
    model.k_angular = sizes[4];
    model.l_max = sizes[5];
    model.four_body = sizes[6] != 0;
    model.five_body = sizes[7] != 0;
    model.neurons = sizes[8];
    model.rc_radial = copy(rc_radial);
    model.rc_angular = copy(rc_angular);
    model.w0 = copy(w0);
    model.b0 = copy(b0);
    model.w1 = copy(w1);
    model.b1 = b1;
    model.c_radial = copy(c_radial);
    model.c_angular = copy(c_angular);
    model.q_scaler = copy(q_scaler);
    model.zbl = copy(zbl);
    model.prepare();
    return model;
}

void check_nep_input(const calculators::NepModel& model, Species species, Positions positions,
                     Offsets offsets, Neighbors neighbors, Shifts shifts) {
    const auto n = static_cast<std::int64_t>(species.shape(0));
    if (positions.shape(0) != species.shape(0) ||
        offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the neighbor list does not fit the atoms");
    }
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (species.data()[i] < 0 || species.data()[i] >= model.types) {
            throw std::invalid_argument("a species index is outside the potential");
        }
    }
}

nb::tuple compute_nep(const calculators::NepModel& model, Species species, Positions positions,
                      Matrix matrix, Offsets offsets, Neighbors neighbors, Shifts shifts,
                      nb::handle progress, const CancelToken* cancel) {
    check_nep_input(model, species, positions, offsets, neighbors, shifts);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    double *energies, *forces, *virials;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = new_array(3, shape3, virials);
    run_released(
        [&](const Control& control) {
            calculators::compute_nep(model, n, species.data(), positions.data(), matrix.data(),
                                     offsets.data(), neighbors.data(), shifts.data(), energies,
                                     forces, virials, nullptr, control);
        },
        progress, cancel);
    return nb::make_tuple(e, f, w);
}

// The descriptors alone, without the network and the forces.
nb::object nep_descriptors(const calculators::NepModel& model, Species species,
                           Positions positions, Matrix matrix, Offsets offsets,
                           Neighbors neighbors, Shifts shifts, nb::handle progress,
                           const CancelToken* cancel) {
    check_nep_input(model, species, positions, offsets, neighbors, shifts);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    double* q;
    const std::size_t shape[2] = {static_cast<std::size_t>(n), static_cast<std::size_t>(model.dim)};
    nb::object d = new_array(2, shape, q);
    run_released(
        [&](const Control& control) {
            calculators::compute_nep(model, n, species.data(), positions.data(), matrix.data(),
                                     offsets.data(), neighbors.data(), shifts.data(), nullptr,
                                     nullptr, nullptr, q, control);
        },
        progress, cancel);
    return d;
}

// Many frames, each as compute_nep finds it alone: with forces, the energies
// (n), forces (n x 3) and the virial of each frame (frames x 3 x 3); without,
// the mean descriptors of each frame (frames x dim).
nb::object nep_frames(const calculators::NepModel& model, Species species, frames::Starts starts,
                      frames::Positions positions, frames::Matrices matrices, frames::Origins origins,
                      frames::Flags pbc, double cutoff, bool forces, nb::handle progress,
                      const CancelToken* cancel) {
    const neighbor::Frames all = frames::of(starts, positions, matrices, origins, pbc);
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (static_cast<std::int64_t>(species.shape(0)) != n) {
        throw std::invalid_argument("species must have one value per atom");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (species.data()[i] < 0 || species.data()[i] >= model.types) {
            throw std::invalid_argument("a species index is outside the potential");
        }
    }
    const auto atoms = static_cast<std::size_t>(n), count = static_cast<std::size_t>(all.count);
    if (forces) {
        double *e, *f, *w;
        const std::size_t shape1[1] = {atoms}, shape2[2] = {atoms, 3}, shape3[3] = {count, 3, 3};
        nb::object energies = new_array(1, shape1, e);
        nb::object force = new_array(2, shape2, f);
        nb::object virials = new_array(3, shape3, w);
        run_released(
            [&](const Control& control) {
                calculators::compute_nep_frames(model, all, species.data(), cutoff, e, f, w, nullptr, control);
            },
            progress, cancel);
        return nb::make_tuple(energies, force, virials);
    }
    double* q;
    const std::size_t shape[2] = {count, static_cast<std::size_t>(model.dim)};
    nb::object descriptors = new_array(2, shape, q);
    run_released(
        [&](const Control& control) {
            calculators::compute_nep_frames(model, all, species.data(), cutoff, nullptr, nullptr, nullptr, q,
                                            control);
        },
        progress, cancel);
    return descriptors;
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
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    nb::class_<calculators::NepModel>(m, "NepModel")
        .def_ro("types", &calculators::NepModel::types)
        .def_ro("dim", &calculators::NepModel::dim);
    m.def("make_nep", &make_nep, nb::arg("sizes"), nb::arg("rc_radial"), nb::arg("rc_angular"),
          nb::arg("w0"), nb::arg("b0"), nb::arg("w1"), nb::arg("b1"), nb::arg("c_radial"),
          nb::arg("c_angular"), nb::arg("q_scaler"), nb::arg("zbl"));
    m.def("nep_descriptors", &nep_descriptors, nb::arg("model"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("compute_nep", &compute_nep, nb::arg("model"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("nep_frames", &nep_frames, nb::arg("model"), nb::arg("species"), nb::arg("starts"),
          nb::arg("positions"), nb::arg("matrices"), nb::arg("origins"), nb::arg("pbc"),
          nb::arg("cutoff"), nb::arg("forces"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
