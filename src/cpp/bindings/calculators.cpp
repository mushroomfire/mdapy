// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Bindings of the compiled calculators.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bindings/control.h"
#include "bindings/frames.h"
#include "calculators/adp.h"
#include "calculators/eam.h"
#include "calculators/many_body.h"
#include "calculators/meam.h"
#include "calculators/nep.h"
#include "calculators/once.h"
#include "calculators/pair.h"
#include "calculators/stillinger_weber.h"
#include "calculators/tersoff.h"
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
// rho_max: where the embedding energy goes on linearly, by default the end
// of its table; LAMMPS puts it one step beyond for funcfl files
calculators::EamPotential make_eam(Table embedding, Table density, Table pair, double drho,
                                   double dr, double cutoff, std::optional<double> rho_max) {
    if (pair.shape(1) != density.shape(1)) {
        throw std::invalid_argument("inconsistent EAM tables");
    }
    const auto nrho = static_cast<int>(embedding.shape(1));
    return calculators::make_eam_potential(
        embedding.data(), static_cast<int>(embedding.shape(0)), nrho, density.data(),
        static_cast<int>(density.shape(0)), pair.data(), static_cast<int>(pair.shape(0)),
        static_cast<int>(density.shape(1)), drho, dr, cutoff,
        rho_max.value_or((nrho - 1) * drho));
}

// Whether every index lies in [0, n): the least and the largest of them,
// found by the threads without the GIL, as each step of a dynamics asks.
bool within(const index_t* indices, std::int64_t count, std::int64_t n) {
    nb::gil_scoped_release release;
    using Range = std::pair<std::int64_t, std::int64_t>;
    const Range none{std::numeric_limits<std::int64_t>::max(),
                     std::numeric_limits<std::int64_t>::min()};
    const Range range = parallel_reduce(
        count, 0, none,
        [&](std::int64_t begin, std::int64_t end) {
            Range r = none;
            for (std::int64_t p = begin; p < end; ++p) {
                const auto j = static_cast<std::int64_t>(indices[p]);
                r.first = std::min(r.first, j);
                r.second = std::max(r.second, j);
            }
            return r;
        },
        [](const Range& a, const Range& b) {
            return Range{std::min(a.first, b.first), std::max(a.second, b.second)};
        });
    return count == 0 || (range.first >= 0 && range.second < n);
}

using Reverse = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// The reverse of each pair given, after checking that one is given for each
// and that each lies among the pairs; null when none is given.
const std::int64_t* reverse_of(const std::optional<Reverse>& reverse, Neighbors neighbors) {
    if (!reverse.has_value()) {
        return nullptr;
    }
    const auto count = static_cast<std::int64_t>(neighbors.shape(0));
    if (static_cast<std::int64_t>(reverse->shape(0)) != count) {
        throw std::invalid_argument("a reverse for each pair");
    }
    const std::int64_t* r = reverse->data();
    {
        nb::gil_scoped_release release;
        const bool inside = parallel_reduce(
            count, 0, true,
            [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t q = begin; q < end; ++q) {
                    if (r[q] < 0 || r[q] >= count) {
                        return false;
                    }
                }
                return true;
            },
            [](bool a, bool b) { return a && b; });
        if (!inside) {
            throw std::invalid_argument("a reverse pair is outside the pairs");
        }
    }
    return r;
}

// For each pair (i, j, s) of a full neighbor list, the index of (j, i, -s).
nb::object reverse_pairs(Offsets offsets, Neighbors neighbors, Shifts shifts) {
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    if (n < 0 || neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0)) ||
        !within(neighbors.data(), static_cast<std::int64_t>(neighbors.shape(0)), n)) {
        throw std::invalid_argument("not a neighbor list");
    }
    std::int64_t* out;
    const std::size_t shape[1] = {neighbors.shape(0)};
    nb::object result = new_array(1, shape, out);
    run_released(
        [&](const Control& control) {
            const std::vector<std::int64_t> reverse = calculators::reverse_pairs(
                n, offsets.data(), neighbors.data(), shifts.data(), control);
            std::copy(reverse.begin(), reverse.end(), out);
        },
        nb::none(), nullptr);
    return result;
}

// The number of centres among the atoms, the first of them.
std::int64_t centres_of(std::int64_t centers, Species species) {
    const auto n = static_cast<std::int64_t>(species.shape(0));
    if (centers < 0 || centers > n) {
        throw std::invalid_argument("the centres are not among the atoms");
    }
    return centers;
}

// The atoms and their neighbor list fit together and the species lie below count.
void check_input(std::int64_t count, Species species, Positions positions, Offsets offsets,
                 Neighbors neighbors, Shifts shifts) {
    const auto n = static_cast<std::int64_t>(species.shape(0));
    if (positions.shape(0) != species.shape(0) ||
        offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the neighbor list does not fit the atoms");
    }
    if (!within(neighbors.data(), static_cast<std::int64_t>(neighbors.shape(0)), n)) {
        throw std::invalid_argument("a neighbor index is outside the atoms");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (species.data()[i] < 0 || species.data()[i] >= count) {
            throw std::invalid_argument("a species index is outside the potential");
        }
    }
}

using Values = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// The first pass of EAM for the first centers atoms: their embedding
// energies and the slopes of their embedding functions.
nb::tuple eam_densities(const calculators::EamPotential& potential, Species species,
                        Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                        Shifts shifts, std::int64_t centers) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(centers, species);
    double *energies, *slopes;
    const std::size_t shape[1] = {static_cast<std::size_t>(c)};
    nb::object e = new_array(1, shape, energies);
    nb::object s = new_array(1, shape, slopes);
    run_released(
        [&](const Control& control) {
            calculators::eam_densities(potential, c, species.data(), positions.data(),
                                       matrix.data(), offsets.data(), neighbors.data(),
                                       shifts.data(), energies, slopes, control);
        },
        nb::none(), nullptr);
    return nb::make_tuple(e, s);
}

// The second: given the slopes of every atom, the energies of the centres
// (their embedding energies given, the pair energies added), their forces
// and, with virial, their virials (None without).
nb::tuple eam_forces(const calculators::EamPotential& potential, Species species,
                     Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                     Shifts shifts, Values slopes, Values embedding, bool virial, bool heat) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(static_cast<std::int64_t>(embedding.shape(0)), species);
    if (slopes.shape(0) != species.shape(0)) {
        throw std::invalid_argument("a slope for each atom");
    }
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {static_cast<std::size_t>(c)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(c), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(c), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    std::copy(embedding.data(), embedding.data() + c, energies);
    run_released(
        [&](const Control& control) {
            calculators::eam_forces(potential, c, species.data(), positions.data(),
                                    matrix.data(), offsets.data(), neighbors.data(),
                                    shifts.data(), slopes.data(), energies, forces, virials,
                                    control, heat);
        },
        nb::none(), nullptr);
    return nb::make_tuple(e, f, w);
}

nb::tuple compute_eam(const calculators::EamPotential& potential, Species species,
                      Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                      Shifts shifts, nb::handle progress, const CancelToken* cancel,
                      bool heat, bool virial) {
    const auto n = static_cast<std::int64_t>(species.shape(0));
    if (positions.shape(0) != species.shape(0) ||
        offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the neighbor list does not fit the atoms");
    }
    if (!within(neighbors.data(), static_cast<std::int64_t>(neighbors.shape(0)), n)) {
        throw std::invalid_argument("a neighbor index is outside the atoms");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (species.data()[i] < 0 || species.data()[i] >= potential.elements) {
            throw std::invalid_argument("a species index is outside the potential");
        }
    }
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    run_released(
        [&](const Control& control) {
            calculators::compute_eam(potential, n, species.data(), positions.data(),
                                     matrix.data(), offsets.data(), neighbors.data(),
                                     shifts.data(), energies, forces, virials, control,
                                     heat);
        },
        progress, cancel);
    return nb::make_tuple(e, f, w);
}


// style: "lj/cut", "morse", "ufm" or "zbl"; cutsq and offset: (species,
// species); coefficients: (species, species, pair_width(style)), see
// PairPotential.
calculators::PairPotential make_pair(const std::string& style, Flat cutsq, Flat offset,
                                     Flat coefficients) {
    calculators::PairStyle kind;
    if (style == "lj/cut") {
        kind = calculators::PairStyle::LennardJones;
    } else if (style == "morse") {
        kind = calculators::PairStyle::Morse;
    } else if (style == "ufm") {
        kind = calculators::PairStyle::UhlenbeckFord;
    } else if (style == "zbl") {
        kind = calculators::PairStyle::ZBL;
    } else {
        throw std::invalid_argument("unknown pair style");
    }
    const auto t = static_cast<std::size_t>(std::llround(std::sqrt(double(cutsq.shape(0)))));
    const auto width = static_cast<std::size_t>(calculators::pair_width(kind));
    if (t == 0 || t * t != cutsq.shape(0) || offset.shape(0) != t * t ||
        coefficients.shape(0) != width * t * t) {
        throw std::invalid_argument("inconsistent pair tables");
    }
    return calculators::make_pair_potential(kind, static_cast<int>(t), cutsq.data(),
                                            offset.data(), coefficients.data());
}

using Ids = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// The pairs of a full neighbor list of the atoms that a process of MPI
// computes on one thread, each pair once among the processes: offsets, j and
// shifts of a list of their own.
nb::tuple once_pairs(Offsets offsets, Neighbors neighbors, Shifts shifts, std::int64_t centers,
                     Ids ids, Positions positions, Matrix matrix) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (offsets.shape(0) != static_cast<std::size_t>(n) + 1 ||
        neighbors.shape(0) != shifts.shape(0) ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0)) ||
        ids.shape(0) != positions.shape(0) || centers < 0 || centers > n ||
        !within(neighbors.data(), static_cast<std::int64_t>(neighbors.shape(0)), n)) {
        throw std::invalid_argument("the neighbor list does not fit the atoms");
    }
    calculators::OncePairs once;
    {
        nb::gil_scoped_release release;
        once = calculators::once_pairs(n, centers, offsets.data(), neighbors.data(),
                                       shifts.data(), ids.data(), positions.data(),
                                       matrix.data());
    }
    std::int64_t* o;
    index_t* j;
    std::int8_t* sh;
    const std::size_t shape1[1] = {once.offsets.size()};
    const std::size_t shape2[1] = {once.neighbors.size()};
    const std::size_t shape3[2] = {once.neighbors.size(), 3};
    nb::object a = new_array(1, shape1, o);
    nb::object b = new_array(1, shape2, j);
    nb::object c = new_array(2, shape3, sh);
    std::copy(once.offsets.begin(), once.offsets.end(), o);
    std::copy(once.neighbors.begin(), once.neighbors.end(), j);
    std::copy(once.shifts.begin(), once.shifts.end(), sh);
    return nb::make_tuple(a, b, c);
}

// On one thread, each pair once among the processes of MPI: the densities the
// pairs of the first centers atoms give every atom.
nb::object eam_densities_once(const calculators::EamPotential& potential, Species species,
                              Positions positions, Matrix matrix, Offsets offsets,
                              Neighbors neighbors, Shifts shifts, std::int64_t centers) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(centers, species);
    double* rho;
    const std::size_t shape[1] = {species.shape(0)};
    nb::object out = new_array(1, shape, rho);
    std::fill(rho, rho + species.shape(0), 0.0);
    {
        nb::gil_scoped_release release;
        calculators::eam_densities_once(potential, c, species.data(), positions.data(),
                                        matrix.data(), offsets.data(), neighbors.data(),
                                        shifts.data(), rho);
    }
    return out;
}

// The embedding energies and slopes of the first atoms, one for each density.
nb::tuple eam_embedding(const calculators::EamPotential& potential, Species species,
                        Values rho) {
    const auto c = static_cast<std::int64_t>(rho.shape(0));
    if (c > static_cast<std::int64_t>(species.shape(0))) {
        throw std::invalid_argument("more densities than atoms");
    }
    for (std::int64_t i = 0; i < c; ++i) {
        if (species.data()[i] < 0 || species.data()[i] >= potential.elements) {
            throw std::invalid_argument("a species index is outside the potential");
        }
    }
    double *energies, *slopes;
    const std::size_t shape[1] = {static_cast<std::size_t>(c)};
    nb::object e = new_array(1, shape, energies);
    nb::object s = new_array(1, shape, slopes);
    calculators::eam_embedding(potential, c, species.data(), rho.data(), energies, slopes);
    return nb::make_tuple(e, s);
}

// On one thread, each pair once among the processes of MPI: the pair
// energies, forces and virials (None without virial) the pairs of the first
// centers atoms give every atom, from the slopes of every atom.
nb::tuple eam_forces_once(const calculators::EamPotential& potential, Species species,
                          Positions positions, Matrix matrix, Offsets offsets,
                          Neighbors neighbors, Shifts shifts, std::int64_t centers,
                          Values slopes, bool virial, bool heat) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(centers, species);
    if (slopes.shape(0) != species.shape(0)) {
        throw std::invalid_argument("a slope for each atom");
    }
    const std::size_t n = species.shape(0);
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {n};
    const std::size_t shape2[2] = {n, 3};
    const std::size_t shape3[3] = {n, 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    std::fill(energies, energies + n, 0.0);
    std::fill(forces, forces + 3 * n, 0.0);
    if (virials != nullptr) {
        std::fill(virials, virials + 9 * n, 0.0);
    }
    {
        nb::gil_scoped_release release;
        calculators::eam_forces_once(potential, c, species.data(), positions.data(),
                                     matrix.data(), offsets.data(), neighbors.data(),
                                     shifts.data(), slopes.data(), energies, forces, virials,
                                     heat);
    }
    return nb::make_tuple(e, f, w);
}

// On one thread, each pair once among the processes of MPI: the energies,
// forces and virials (None without virial) the pairs of the first centers
// atoms give every atom.
nb::tuple compute_pair_once(const calculators::PairPotential& potential, Species species,
                            Positions positions, Matrix matrix, Offsets offsets,
                            Neighbors neighbors, Shifts shifts, std::int64_t centers,
                            bool virial) {
    check_input(potential.species, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(centers, species);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    {
        nb::gil_scoped_release release;
        calculators::compute_pair_once(potential, n, c, species.data(), positions.data(),
                                       matrix.data(), offsets.data(), neighbors.data(),
                                       shifts.data(), energies, forces, virials);
    }
    return nb::make_tuple(e, f, w);
}

nb::tuple compute_pair(const calculators::PairPotential& potential, Species species,
                       Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                       Shifts shifts, nb::handle progress, const CancelToken* cancel, bool virial) {
    check_input(potential.species, species, positions, offsets, neighbors, shifts);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    run_released(
        [&](const Control& control) {
            calculators::compute_pair(potential, n, species.data(), positions.data(),
                                      matrix.data(), offsets.data(), neighbors.data(),
                                      shifts.data(), energies, forces, virials, control);
        },
        progress, cancel);
    return nb::make_tuple(e, f, w);
}

// rows: elements^3 rows of 14 numbers, see TersoffPotential::parameters
calculators::TersoffPotential make_tersoff(int elements, Flat rows) {
    const auto count = static_cast<std::size_t>(elements) * static_cast<std::size_t>(elements) *
                       static_cast<std::size_t>(elements);
    if (elements < 1 || rows.shape(0) != 14 * count) {
        throw std::invalid_argument("a Tersoff potential needs 14 numbers for each triplet");
    }
    return calculators::make_tersoff(elements, rows.data());
}

nb::tuple compute_tersoff(const calculators::TersoffPotential& potential, Species species,
                          Positions positions, Matrix matrix, Offsets offsets,
                          Neighbors neighbors, Shifts shifts, nb::handle progress,
                          const CancelToken* cancel, bool heat, std::int64_t centers,
                          const std::optional<Reverse>& reverse) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    const std::int64_t c = centers < 0 ? n : centres_of(centers, species);
    double *energies, *forces, *virials;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = new_array(3, shape3, virials);
    const std::int64_t* given = reverse_of(reverse, neighbors);
    run_released(
        [&](const Control& control) {
            calculators::compute_tersoff(potential, n, species.data(), positions.data(),
                                         matrix.data(), offsets.data(), neighbors.data(),
                                         shifts.data(), energies, forces, virials, control,
                                         heat, c, given);
        },
        progress, cancel);
    return nb::make_tuple(e, f, w);
}

// rows: elements^3 rows of 11 numbers, see StillingerWeberPotential::parameters
// On one thread, each term once among the processes of MPI: the energies,
// forces and virials (None without virial) the terms of the first centers
// atoms give every atom, the two-body terms of the pairs of once_*, the
// others over the full list. kernel: compute_stillinger_weber_once or
// compute_tersoff_once.
template <class Potential, class Kernel>
nb::tuple three_body_once(const Potential& potential, Kernel kernel, Species species,
                          Positions positions, Matrix matrix, Offsets offsets,
                          Neighbors neighbors, Shifts shifts, Offsets once_offsets,
                          Neighbors once_neighbors, Shifts once_shifts, std::int64_t centers,
                          bool virial) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    check_input(potential.elements, species, positions, once_offsets, once_neighbors,
                once_shifts);
    const std::int64_t c = centres_of(centers, species);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    {
        nb::gil_scoped_release release;
        kernel(potential, n, species.data(), positions.data(), matrix.data(), offsets.data(),
               neighbors.data(), shifts.data(), once_offsets.data(), once_neighbors.data(),
               once_shifts.data(), c, energies, forces, virials);
    }
    return nb::make_tuple(e, f, w);
}

calculators::StillingerWeberPotential make_stillinger_weber(int elements, Flat rows) {
    const auto count = static_cast<std::size_t>(elements) * static_cast<std::size_t>(elements) *
                       static_cast<std::size_t>(elements);
    if (elements < 1 || rows.shape(0) != 11 * count) {
        throw std::invalid_argument("a Stillinger-Weber potential needs 11 numbers for each triplet");
    }
    return calculators::make_stillinger_weber(elements, rows.data());
}

nb::tuple compute_stillinger_weber(const calculators::StillingerWeberPotential& potential,
                                   Species species, Positions positions, Matrix matrix,
                                   Offsets offsets, Neighbors neighbors, Shifts shifts,
                                   nb::handle progress, const CancelToken* cancel, bool heat,
                                   std::int64_t centers,
                                   const std::optional<Reverse>& reverse) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    const std::int64_t c = centers < 0 ? n : centres_of(centers, species);
    double *energies, *forces, *virials;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = new_array(3, shape3, virials);
    const std::int64_t* given = reverse_of(reverse, neighbors);
    run_released(
        [&](const Control& control) {
            calculators::compute_stillinger_weber(potential, n, species.data(), positions.data(),
                                                  matrix.data(), offsets.data(), neighbors.data(),
                                                  shifts.data(), energies, forces, virials,
                                                  control, heat, c, given);
        },
        progress, cancel);
    return nb::make_tuple(e, f, w);
}

// embedding: (elements, nrho); density: (elements, nr); pair, dipole and
// quadrupole: (elements * (elements + 1) / 2, nr), the lower triangle by rows
calculators::AdpPotential make_adp(Table embedding, Table density, Table pair, Table dipole,
                                   Table quadrupole, double drho, double dr, double cutoff) {
    const auto elements = static_cast<int>(embedding.shape(0));
    const auto pairs = static_cast<std::size_t>(elements * (elements + 1) / 2);
    if (density.shape(0) != static_cast<std::size_t>(elements) || pair.shape(0) != pairs ||
        dipole.shape(0) != pairs || quadrupole.shape(0) != pairs ||
        pair.shape(1) != density.shape(1) || dipole.shape(1) != density.shape(1) ||
        quadrupole.shape(1) != density.shape(1)) {
        throw std::invalid_argument("inconsistent ADP tables");
    }
    return calculators::make_adp_potential(
        embedding.data(), elements, static_cast<int>(embedding.shape(1)), density.data(),
        pair.data(), dipole.data(), quadrupole.data(), static_cast<int>(density.shape(1)), drho,
        dr, cutoff);
}

// The first pass of ADP for the first centers atoms: their energies and
// states (centers, 10): F'(rho), the dipole and the quadrupole.
nb::tuple adp_densities(const calculators::AdpPotential& potential, Species species,
                        Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                        Shifts shifts, std::int64_t centers) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(centers, species);
    double *energies, *state;
    const std::size_t shape1[1] = {static_cast<std::size_t>(c)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(c), calculators::kState};
    nb::object e = new_array(1, shape1, energies);
    nb::object s = new_array(2, shape2, state);
    run_released(
        [&](const Control& control) {
            calculators::adp_densities(potential, c, species.data(), positions.data(),
                                       matrix.data(), offsets.data(), neighbors.data(),
                                       shifts.data(), energies, state, control);
        },
        nb::none(), nullptr);
    return nb::make_tuple(e, s);
}

using States = nb::ndarray<const double, nb::shape<-1, calculators::kState>, nb::c_contig,
                           nb::device::cpu>;

// The second: given the states of every atom, the energies of the centres
// (theirs given, the pair energies added), their forces and, with virial,
// their virials (None without).
nb::tuple adp_forces(const calculators::AdpPotential& potential, Species species,
                     Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                     Shifts shifts, States state, Values given, bool virial) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(static_cast<std::int64_t>(given.shape(0)), species);
    if (state.shape(0) != species.shape(0)) {
        throw std::invalid_argument("a state for each atom");
    }
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {static_cast<std::size_t>(c)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(c), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(c), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    std::copy(given.data(), given.data() + c, energies);
    run_released(
        [&](const Control& control) {
            calculators::adp_forces(potential, c, species.data(), positions.data(),
                                    matrix.data(), offsets.data(), neighbors.data(),
                                    shifts.data(), state.data(), energies, forces, virials,
                                    control);
        },
        nb::none(), nullptr);
    return nb::make_tuple(e, f, w);
}

// On one thread, each pair once among the processes of MPI: the sums
// (n, 10) the pairs of the first centers atoms give every atom, its density,
// dipole and quadrupole.
nb::object adp_densities_once(const calculators::AdpPotential& potential, Species species,
                              Positions positions, Matrix matrix, Offsets offsets,
                              Neighbors neighbors, Shifts shifts, std::int64_t centers) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(centers, species);
    double* sums;
    const std::size_t shape[2] = {species.shape(0), calculators::kState};
    nb::object out = new_array(2, shape, sums);
    std::fill(sums, sums + calculators::kState * species.shape(0), 0.0);
    {
        nb::gil_scoped_release release;
        calculators::adp_densities_once(potential, c, species.data(), positions.data(),
                                        matrix.data(), offsets.data(), neighbors.data(),
                                        shifts.data(), sums);
    }
    return out;
}

// The energies and states (c, 10) of the first atoms, one for each row of
// whole sums.
nb::tuple adp_embedding(const calculators::AdpPotential& potential, Species species,
                        States sums) {
    const auto c = static_cast<std::int64_t>(sums.shape(0));
    if (c > static_cast<std::int64_t>(species.shape(0))) {
        throw std::invalid_argument("more sums than atoms");
    }
    for (std::int64_t i = 0; i < c; ++i) {
        if (species.data()[i] < 0 || species.data()[i] >= potential.elements) {
            throw std::invalid_argument("a species index is outside the potential");
        }
    }
    double *energies, *state;
    const std::size_t shape1[1] = {static_cast<std::size_t>(c)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(c), calculators::kState};
    nb::object e = new_array(1, shape1, energies);
    nb::object s = new_array(2, shape2, state);
    calculators::adp_embedding(potential, c, species.data(), sums.data(), energies, state);
    return nb::make_tuple(e, s);
}

// On one thread, each pair once among the processes of MPI: the pair
// energies, forces and virials (None without virial) the pairs of the first
// centers atoms give every atom, from the states of every atom.
nb::tuple adp_forces_once(const calculators::AdpPotential& potential, Species species,
                          Positions positions, Matrix matrix, Offsets offsets,
                          Neighbors neighbors, Shifts shifts, std::int64_t centers,
                          States state, bool virial) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const std::int64_t c = centres_of(centers, species);
    if (state.shape(0) != species.shape(0)) {
        throw std::invalid_argument("a state for each atom");
    }
    const std::size_t n = species.shape(0);
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {n};
    const std::size_t shape2[2] = {n, 3};
    const std::size_t shape3[3] = {n, 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    std::fill(energies, energies + n, 0.0);
    std::fill(forces, forces + 3 * n, 0.0);
    if (virials != nullptr) {
        std::fill(virials, virials + 9 * n, 0.0);
    }
    {
        nb::gil_scoped_release release;
        calculators::adp_forces_once(potential, c, species.data(), positions.data(),
                                     matrix.data(), offsets.data(), neighbors.data(),
                                     shifts.data(), state.data(), energies, forces, virials);
    }
    return nb::make_tuple(e, f, w);
}

nb::tuple compute_adp(const calculators::AdpPotential& potential, Species species,
                      Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                      Shifts shifts, nb::handle progress, const CancelToken* cancel, bool virial) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    double *energies, *forces, *virials = nullptr;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = virial ? new_array(3, shape3, virials) : nb::none();
    run_released(
        [&](const Control& control) {
            calculators::compute_adp(potential, n, species.data(), positions.data(),
                                     matrix.data(), offsets.data(), neighbors.data(),
                                     shifts.data(), energies, forces, virials, control);
        },
        progress, cancel);
    return nb::make_tuple(e, f, w);
}

using Array = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

// element: (17, elements), see MeamPotential::Row; ebound: (elements,
// elements); cmin, cmax: (elements, elements, elements); tables: (pairs, 7,
// nr); options: ialloy, mixture_ref_t, emb_lin_neg, bkgd_dyn; numbers: rc,
// delr, gsmooth_factor, dr, cutoff
calculators::MeamPotential make_meam(Table element, Table ebound, Array cmin, Array cmax,
                                     Array tables, const std::vector<int>& options,
                                     const std::vector<double>& numbers) {
    const auto e = static_cast<std::size_t>(element.shape(1));
    if (element.shape(0) != 17 || ebound.shape(0) != e || ebound.shape(1) != e ||
        cmin.ndim() != 3 || cmax.ndim() != 3 || cmin.size() != e * e * e ||
        cmax.size() != e * e * e || tables.ndim() != 3 ||
        tables.shape(0) != e * (e + 1) / 2 || tables.shape(1) != 7 || tables.shape(2) < 3 ||
        options.size() != 4 || numbers.size() != 5) {
        throw std::invalid_argument("inconsistent MEAM tables");
    }
    return calculators::make_meam_potential(static_cast<int>(e), element.data(), ebound.data(),
                                            cmin.data(), cmax.data(), tables.data(),
                                            static_cast<int>(tables.shape(2)), options.data(),
                                            numbers.data());
}

nb::tuple compute_meam(const calculators::MeamPotential& potential, Species species,
                       Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
                       Shifts shifts, nb::handle progress, const CancelToken* cancel,
                       const std::optional<Reverse>& reverse) {
    check_input(potential.elements, species, positions, offsets, neighbors, shifts);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    double *energies, *forces, *virials;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = new_array(3, shape3, virials);
    const std::int64_t* given = reverse_of(reverse, neighbors);
    run_released(
        [&](const Control& control) {
            calculators::compute_meam(potential, n, species.data(), positions.data(),
                                      matrix.data(), offsets.data(), neighbors.data(),
                                      shifts.data(), energies, forces, virials, control, given);
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
    if (!within(neighbors.data(), static_cast<std::int64_t>(neighbors.shape(0)), n)) {
        throw std::invalid_argument("a neighbor index is outside the atoms");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (species.data()[i] < 0 || species.data()[i] >= model.types) {
            throw std::invalid_argument("a species index is outside the potential");
        }
    }
}

nb::tuple compute_nep(const calculators::NepModel& model, Species species, Positions positions,
                      Matrix matrix, Offsets offsets, Neighbors neighbors, Shifts shifts,
                      nb::handle progress, const CancelToken* cancel, std::int64_t centers,
                      const std::optional<Reverse>& reverse) {
    check_nep_input(model, species, positions, offsets, neighbors, shifts);
    const auto n = static_cast<std::int64_t>(species.shape(0));
    const std::int64_t c = centers < 0 ? n : centres_of(centers, species);
    double *energies, *forces, *virials;
    const std::size_t shape1[1] = {static_cast<std::size_t>(n)};
    const std::size_t shape2[2] = {static_cast<std::size_t>(n), 3};
    const std::size_t shape3[3] = {static_cast<std::size_t>(n), 3, 3};
    nb::object e = new_array(1, shape1, energies);
    nb::object f = new_array(2, shape2, forces);
    nb::object w = new_array(3, shape3, virials);
    const std::int64_t* given = reverse_of(reverse, neighbors);
    run_released(
        [&](const Control& control) {
            calculators::compute_nep(model, n, species.data(), positions.data(), matrix.data(),
                                     offsets.data(), neighbors.data(), shifts.data(), energies,
                                     forces, virials, nullptr, control, c, given);
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
    nb::class_<calculators::MeamPotential>(m, "MeamPotential")
        .def_ro("elements", &calculators::MeamPotential::elements)
        .def_ro("cutoff", &calculators::MeamPotential::cutoff);
    m.def("make_meam", &make_meam, nb::arg("element"), nb::arg("ebound"), nb::arg("cmin"),
          nb::arg("cmax"), nb::arg("tables"), nb::arg("options"), nb::arg("numbers"));
    m.def("compute_meam", &compute_meam, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr, nb::arg("reverse") = nb::none());
    nb::class_<calculators::AdpPotential>(m, "AdpPotential")
        .def_ro("elements", &calculators::AdpPotential::elements)
        .def_ro("cutoff", &calculators::AdpPotential::cutoff);
    m.def("make_adp", &make_adp, nb::arg("embedding"), nb::arg("density"), nb::arg("pair"),
          nb::arg("dipole"), nb::arg("quadrupole"), nb::arg("drho"), nb::arg("dr"),
          nb::arg("cutoff"));
    m.def("compute_adp", &compute_adp, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr,
          nb::arg("virial") = true);
    m.def("adp_densities_once", &adp_densities_once, nb::arg("potential"),
          nb::arg("species"), nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"),
          nb::arg("neighbors"), nb::arg("shifts"), nb::arg("centers"));
    m.def("adp_embedding", &adp_embedding, nb::arg("potential"), nb::arg("species"),
          nb::arg("sums"));
    m.def("adp_forces_once", &adp_forces_once, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("centers"), nb::arg("state"), nb::arg("virial"));
    m.def("adp_densities", &adp_densities, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("centers"));
    m.def("adp_forces", &adp_forces, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("state"), nb::arg("given"), nb::arg("virial"));
    nb::class_<calculators::StillingerWeberPotential>(m, "StillingerWeberPotential")
        .def_ro("elements", &calculators::StillingerWeberPotential::elements)
        .def_ro("cutoff", &calculators::StillingerWeberPotential::cutoff);
    m.def("make_stillinger_weber", &make_stillinger_weber, nb::arg("elements"), nb::arg("rows"));
    m.def("compute_stillinger_weber", &compute_stillinger_weber, nb::arg("potential"),
          nb::arg("species"), nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"),
          nb::arg("neighbors"), nb::arg("shifts"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr, nb::arg("heat") = false,
          nb::arg("centers") = -1,
          nb::arg("reverse") = nb::none());
    nb::class_<calculators::TersoffPotential>(m, "TersoffPotential")
        .def_ro("elements", &calculators::TersoffPotential::elements)
        .def_ro("cutoff", &calculators::TersoffPotential::cutoff);
    m.def("make_tersoff", &make_tersoff, nb::arg("elements"), nb::arg("rows"));
    m.def("reverse_pairs", &reverse_pairs, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"));
    m.def(
        "compute_stillinger_weber_once",
        [](const calculators::StillingerWeberPotential& potential, Species species,
           Positions positions, Matrix matrix, Offsets offsets, Neighbors neighbors,
           Shifts shifts, Offsets once_offsets, Neighbors once_neighbors, Shifts once_shifts,
           std::int64_t centers, bool virial) {
            return three_body_once(potential, &calculators::compute_stillinger_weber_once,
                                   species, positions, matrix, offsets, neighbors, shifts,
                                   once_offsets, once_neighbors, once_shifts, centers, virial);
        },
        nb::arg("potential"), nb::arg("species"), nb::arg("positions"), nb::arg("matrix"),
        nb::arg("offsets"), nb::arg("neighbors"), nb::arg("shifts"), nb::arg("once_offsets"),
        nb::arg("once_neighbors"), nb::arg("once_shifts"), nb::arg("centers"),
        nb::arg("virial"));
    m.def(
        "compute_tersoff_once",
        [](const calculators::TersoffPotential& potential, Species species, Positions positions,
           Matrix matrix, Offsets offsets, Neighbors neighbors, Shifts shifts,
           Offsets once_offsets, Neighbors once_neighbors, Shifts once_shifts,
           std::int64_t centers, bool virial) {
            return three_body_once(potential, &calculators::compute_tersoff_once, species,
                                   positions, matrix, offsets, neighbors, shifts, once_offsets,
                                   once_neighbors, once_shifts, centers, virial);
        },
        nb::arg("potential"), nb::arg("species"), nb::arg("positions"), nb::arg("matrix"),
        nb::arg("offsets"), nb::arg("neighbors"), nb::arg("shifts"), nb::arg("once_offsets"),
        nb::arg("once_neighbors"), nb::arg("once_shifts"), nb::arg("centers"),
        nb::arg("virial"));
    m.def("compute_tersoff", &compute_tersoff, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr, nb::arg("heat") = false,
          nb::arg("centers") = -1,
          nb::arg("reverse") = nb::none());
    nb::class_<calculators::PairPotential>(m, "PairPotential")
        .def_ro("species", &calculators::PairPotential::species)
        .def_ro("cutoff", &calculators::PairPotential::cutoff);
    m.def("make_pair", &make_pair, nb::arg("style"), nb::arg("cutsq"), nb::arg("offset"),
          nb::arg("coefficients"));
    m.def("compute_pair_once", &compute_pair_once, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("centers"), nb::arg("virial"));
    m.def("once_pairs", &once_pairs, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("centers"), nb::arg("ids"), nb::arg("positions"),
          nb::arg("matrix"));
    m.def("eam_densities_once", &eam_densities_once, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("centers"));
    m.def("eam_embedding", &eam_embedding, nb::arg("potential"), nb::arg("species"),
          nb::arg("rho"));
    m.def("eam_forces_once", &eam_forces_once, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("centers"), nb::arg("slopes"), nb::arg("virial"),
          nb::arg("heat") = false);
    m.def("compute_pair", &compute_pair, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr,
          nb::arg("virial") = true);
    nb::class_<calculators::EamPotential>(m, "EamPotential")
        .def_ro("elements", &calculators::EamPotential::elements)
        .def_ro("cutoff", &calculators::EamPotential::cutoff)
        .def_ro("rho_max", &calculators::EamPotential::rho_max);
    m.def("make_eam", &make_eam, nb::arg("embedding"), nb::arg("density"), nb::arg("pair"),
          nb::arg("drho"), nb::arg("dr"), nb::arg("cutoff"), nb::arg("rho_max") = nb::none());
    m.def("compute_eam", &compute_eam, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr,
          nb::arg("heat") = false,
          nb::arg("virial") = true);
    m.def("eam_densities", &eam_densities, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("centers"));
    m.def("eam_forces", &eam_forces, nb::arg("potential"), nb::arg("species"),
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("slopes"), nb::arg("embedding"), nb::arg("virial"),
          nb::arg("heat") = false);
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
          nb::arg("cancel").none() = nullptr, nb::arg("centers") = -1,
          nb::arg("reverse") = nb::none());
    m.def("nep_frames", &nep_frames, nb::arg("model"), nb::arg("species"), nb::arg("starts"),
          nb::arg("positions"), nb::arg("matrices"), nb::arg("origins"), nb::arg("pbc"),
          nb::arg("cutoff"), nb::arg("forces"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
