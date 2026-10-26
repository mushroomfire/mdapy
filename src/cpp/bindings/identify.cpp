// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Bindings of the structure identification.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>

#include "bindings/control.h"
#include "bindings/list_view.h"
#include "core/index.h"
#include "core/parallel.h"
#include "identify/planar_faults.h"
#include "identify/ptm.h"
#include "identify/structure.h"

namespace mdapy::bindings {
namespace {

using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vectors = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::object new_array(std::size_t size, T*& data) {
    data = new T[size];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    const std::size_t shape[1] = {size};
    return nb::ndarray<nb::numpy, T>(data, 1, shape, owner).cast();
}

template <class T>
nb::object new_array(std::initializer_list<std::size_t> shape, T*& data) {
    std::size_t size = 1;
    for (std::size_t extent : shape) {
        size *= extent;
    }
    data = new T[size];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.begin(), owner).cast();
}

// A kernel of a neighbor list, read where it is, into a new array of one
// value of T per atom.
template <class T, class Kernel>
nb::object list_kernel(lists::Positions positions, lists::Matrix matrix, lists::Offsets offsets,
                       lists::Neighbors neighbors, lists::Shifts shifts,
                       lists::Distances distances, std::int64_t sorted, std::int64_t count,
                       double cutoff, nb::handle progress, const CancelToken* cancel,
                       Kernel kernel) {
    const neighbor::ListView list = lists::view(positions, matrix, offsets, neighbors, shifts,
                                                distances, sorted, count, cutoff);
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    T* values;
    nb::object result = new_array(static_cast<std::size_t>(n), values);
    run_released([&](const Control& control) { kernel(list, n, values, control); }, progress,
                 cancel);
    return result;
}

using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Indices = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Shifts = nb::ndarray<const std::int8_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Types = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// The nearest neighbors of each atom, nearest first, in CSR form with their
// periodic images; types for the chemical ordering, or none. Returns a
// dictionary of the per-atom results.
nb::dict polyhedral_template_matching(Positions positions, Matrix matrix, Offsets offsets,
                                      Indices j, Shifts shifts, std::optional<Types> types,
                                      std::int32_t structures, bool deformation, bool neighbors,
                                      nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (offsets.shape(0) != static_cast<std::size_t>(n + 1) ||
        offsets.data()[n] != static_cast<std::int64_t>(j.shape(0)) ||
        shifts.shape(0) != j.shape(0)) {
        throw std::invalid_argument("the neighbor list does not fit the atoms");
    }
    if (types && types->shape(0) != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("there must be a type for each atom");
    }
    const auto size = static_cast<std::size_t>(n);
    identify::PtmInput input;
    input.n = n;
    input.positions = positions.data();
    input.matrix = matrix.data();
    input.offsets = offsets.data();
    input.j = j.data();
    input.shifts = shifts.data();
    input.types = types ? types->data() : nullptr;
    input.structures = structures;
    input.deformation = deformation;
    identify::PtmOutput output;
    nb::dict result;
    result["structure"] = new_array({size}, output.structure);
    result["rmsd"] = new_array({size}, output.rmsd);
    result["interatomic_distance"] = new_array({size}, output.interatomic_distance);
    result["orientation"] = new_array({size, 4}, output.orientation);
    result["correspondences"] = new_array({size}, output.correspondences);
    result["template_index"] = new_array({size}, output.template_index);
    if (deformation) {
        result["deformation_gradient"] = new_array({size, 3, 3}, output.deformation);
    }
    if (types) {
        result["ordering"] = new_array({size}, output.ordering);
    }
    if (neighbors) {
        result["neighbors"] = new_array({size, 16}, output.neighbors);
    }
    run_released(
        [&](const Control& control) {
            identify::polyhedral_template_matching(input, output, control);
        },
        progress, cancel);
    return result;
}

// The first k pairs of each atom, its nearest.
nb::object centro_symmetry(lists::Positions positions, lists::Matrix matrix,
                           lists::Offsets offsets, lists::Neighbors neighbors,
                           lists::Shifts shifts, lists::Distances distances,
                           std::int64_t sorted, int k, nb::handle progress,
                           const CancelToken* cancel) {
    if (k < 2 || k % 2 != 0) {
        throw std::invalid_argument("the number of neighbors must be even and at least 2");
    }
    return list_kernel<double>(
        positions, matrix, offsets, neighbors, shifts, distances, sorted, k, 0.0, progress, cancel,
        [k](const neighbor::ListView& list, std::int64_t n, double* csp, const Control& control) {
            identify::centro_symmetry(list, n, k, csp, control);
        });
}

nb::object common_neighbor_parameter(lists::Positions positions, lists::Matrix matrix,
                                     lists::Offsets offsets, lists::Neighbors neighbors,
                                     lists::Shifts shifts, lists::Distances distances,
                                     std::int64_t sorted, double cutoff, nb::handle progress,
                                     const CancelToken* cancel) {
    if (!(cutoff > 0.0)) {
        throw std::invalid_argument("cutoff must be positive");
    }
    return list_kernel<double>(
        positions, matrix, offsets, neighbors, shifts, distances, sorted, 0, 0.0, progress, cancel,
        [cutoff](const neighbor::ListView& list, std::int64_t n, double* cnp,
                 const Control& control) {
            identify::common_neighbor_parameter(list, n, cutoff, cnp, control);
        });
}

using Box3 = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Origin3 = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Periodic = nb::ndarray<const bool, nb::shape<3>, nb::c_contig, nb::device::cpu>;

nb::object common_neighbor_parameter_search(Vectors positions, Box3 matrix, Origin3 origin,
                                            Periodic pbc, double cutoff, nb::handle progress,
                                            const CancelToken* cancel) {
    if (!(cutoff > 0.0)) {
        throw std::invalid_argument("cutoff must be positive");
    }
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    double* cnp;
    nb::object result = new_array(static_cast<std::size_t>(n), cnp);
    run_released(
        [&](const Control& control) {
            identify::common_neighbor_parameter_search(positions.data(), n, matrix.data(),
                                                       origin.data(), pbc.data(), cutoff, cnp,
                                                       control);
        },
        progress, cancel);
    return result;
}

nb::object common_neighbor_analysis_search(Vectors positions, Box3 matrix, Origin3 origin,
                                           Periodic pbc, double cutoff, nb::handle progress,
                                           const CancelToken* cancel) {
    if (!(cutoff > 0.0)) {
        throw std::invalid_argument("cutoff must be positive");
    }
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    std::int8_t* types;
    nb::object result = new_array(static_cast<std::size_t>(n), types);
    run_released(
        [&](const Control& control) {
            identify::common_neighbor_analysis_search(positions.data(), n, matrix.data(),
                                                      origin.data(), pbc.data(), cutoff, types,
                                                      control);
        },
        progress, cancel);
    return result;
}

// A kernel of the k nearest neighbors of every atom, without a list, into
// a new array of one value of T per atom.
template <class T, class Kernel>
nb::object nearest_kernel(Vectors positions, Box3 matrix, Origin3 origin, Periodic pbc, int k,
                          nb::handle progress, const CancelToken* cancel, Kernel kernel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    T* values;
    nb::object result = new_array(static_cast<std::size_t>(n), values);
    run_released(
        [&](const Control& control) {
            kernel(positions.data(), n, matrix.data(), origin.data(), pbc.data(), k, values,
                   control);
        },
        progress, cancel);
    return result;
}

nb::object centro_symmetry_nearest(Vectors positions, Box3 matrix, Origin3 origin, Periodic pbc,
                                   int k, nb::handle progress, const CancelToken* cancel) {
    return nearest_kernel<double>(positions, matrix, origin, pbc, k, progress, cancel,
                                  identify::centro_symmetry_nearest);
}

nb::object common_neighbor_analysis_nearest(Vectors positions, Box3 matrix, Origin3 origin,
                                            Periodic pbc, int k, nb::handle progress,
                                            const CancelToken* cancel) {
    return nearest_kernel<std::int8_t>(positions, matrix, origin, pbc, k, progress, cancel,
                                       identify::common_neighbor_analysis_nearest);
}

nb::object ackland_jones_nearest(Vectors positions, Box3 matrix, Origin3 origin, Periodic pbc,
                                 int k, nb::handle progress, const CancelToken* cancel) {
    return nearest_kernel<std::int8_t>(positions, matrix, origin, pbc, k, progress, cancel,
                                       identify::ackland_jones_nearest);
}

// The first count pairs of each atom, its nearest, for adaptive cutoffs;
// those closer than cutoff otherwise.
nb::object common_neighbor_analysis(lists::Positions positions, lists::Matrix matrix,
                                    lists::Offsets offsets, lists::Neighbors neighbors,
                                    lists::Shifts shifts, lists::Distances distances,
                                    std::int64_t sorted, std::int64_t count, double cutoff,
                                    nb::handle progress, const CancelToken* cancel) {
    return list_kernel<std::int8_t>(positions, matrix, offsets, neighbors, shifts, distances,
                                    sorted, count, cutoff, progress, cancel,
                                    [](const neighbor::ListView& list, std::int64_t n,
                                       std::int8_t* types, const Control& control) {
                                        identify::common_neighbor_analysis(list, n, types,
                                                                           control);
                                    });
}

// The first count pairs of each atom, its nearest.
nb::object ackland_jones(lists::Positions positions, lists::Matrix matrix, lists::Offsets offsets,
                         lists::Neighbors neighbors, lists::Shifts shifts,
                         lists::Distances distances, std::int64_t sorted, std::int64_t count,
                         nb::handle progress, const CancelToken* cancel) {
    return list_kernel<std::int8_t>(positions, matrix, offsets, neighbors, shifts, distances,
                                    sorted, count, 0.0, progress, cancel,
                                    [](const neighbor::ListView& list, std::int64_t n,
                                       std::int8_t* types, const Control& control) {
                                        identify::ackland_jones(list, n, types, control);
                                    });
}

// The first four pairs of each atom, its nearest.
nb::object identify_diamond(lists::Positions positions, lists::Matrix matrix,
                            lists::Offsets offsets, lists::Neighbors neighbors,
                            lists::Shifts shifts, lists::Distances distances, std::int64_t sorted,
                            nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    return list_kernel<std::int8_t>(positions, matrix, offsets, neighbors, shifts, distances,
                                    sorted, 4, 0.0, progress, cancel,
                                    [](const neighbor::ListView& list, std::int64_t count,
                                       std::int8_t* types, const Control& control) {
                                        identify::identify_diamond(list, count, types, control);
                                    });
}

// The pairs of each molecule closer than cutoff.
nb::object chill_plus(lists::Positions positions, lists::Matrix matrix, lists::Offsets offsets,
                      lists::Neighbors neighbors, lists::Shifts shifts,
                      lists::Distances distances, std::int64_t sorted, double cutoff,
                      nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    return list_kernel<std::int8_t>(positions, matrix, offsets, neighbors, shifts, distances,
                                    sorted, 0, cutoff, progress, cancel,
                                    [](const neighbor::ListView& list, std::int64_t count,
                                       std::int8_t* types, const Control& control) {
                                        identify::chill_plus(list, count, types, control);
                                    });
}

using Structures = nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using TemplateNeighbors =
    nb::ndarray<const index_t, nb::shape<-1, 16>, nb::c_contig, nb::device::cpu>;

// The planar fault type of each atom from its PTM type and its neighbors in
// the order of its template.
nb::object planar_faults(Structures structure, TemplateNeighbors neighbors, bool extrinsic,
                         nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(structure.shape(0));
    if (neighbors.shape(0) != structure.shape(0)) {
        throw std::invalid_argument("there must be the neighbors of each atom");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int8_t type = structure.data()[i];
        if (type != identify::kPtmHcp && !(extrinsic && type == identify::kPtmFcc)) {
            continue;
        }
        for (int k = 0; k < 12; ++k) {
            const index_t b = neighbors.data()[16 * i + k];
            if (b < 0 || b >= n) {
                throw std::invalid_argument(
                    "a neighbor of an hcp or fcc atom is outside the atoms");
            }
        }
    }
    std::int8_t* faults = nullptr;
    nb::object result = new_array(static_cast<std::size_t>(n), faults);
    run_released(
        [&](const Control& control) {
            identify::planar_faults(structure.data(), neighbors.data(), n, extrinsic, faults,
                                    control);
        },
        progress, cancel);
    return result;
}

}  // namespace

void bind_identify(nb::module_& m) {
    m.def("polyhedral_template_matching", &polyhedral_template_matching,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("j"),
          nb::arg("shifts"), nb::arg("types").none(), nb::arg("structures"),
          nb::arg("deformation"), nb::arg("neighbors"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("centro_symmetry", &centro_symmetry,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("k"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("common_neighbor_parameter_search", &common_neighbor_parameter_search,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"), nb::arg("pbc"),
          nb::arg("cutoff"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("common_neighbor_parameter", &common_neighbor_parameter,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("cutoff"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("common_neighbor_analysis_search", &common_neighbor_analysis_search,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"), nb::arg("pbc"),
          nb::arg("cutoff"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("centro_symmetry_nearest", &centro_symmetry_nearest, nb::arg("positions"),
          nb::arg("matrix"), nb::arg("origin"), nb::arg("pbc"), nb::arg("k"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("common_neighbor_analysis_nearest", &common_neighbor_analysis_nearest,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"), nb::arg("pbc"),
          nb::arg("k"), nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("ackland_jones_nearest", &ackland_jones_nearest, nb::arg("positions"),
          nb::arg("matrix"), nb::arg("origin"), nb::arg("pbc"), nb::arg("k"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("common_neighbor_analysis", &common_neighbor_analysis,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("count"),
          nb::arg("cutoff"), nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("chill_plus", &chill_plus,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("cutoff"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("ackland_jones", &ackland_jones,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("count"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("planar_faults", &planar_faults, nb::arg("structure"), nb::arg("neighbors"),
          nb::arg("extrinsic"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("identify_diamond", &identify_diamond,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), 
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
