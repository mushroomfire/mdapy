// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Bindings of the analyses.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "analysis/dislocations.h"
#include "analysis/distribution.h"
#include "analysis/elastic_strain.h"
#include "analysis/tessellation.h"
#include "analysis/geometry.h"
#include "analysis/isosurface.h"
#include "analysis/rings.h"
#include "analysis/soap.h"
#include "analysis/steinhardt.h"
#include "analysis/strain.h"
#include "analysis/surface_mesh.h"
#include "analysis/voronoi.h"
#include "bindings/control.h"
#include "bindings/list_view.h"
#include "core/buffer.h"
#include "core/parallel.h"

namespace mdapy::bindings {
namespace {

using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Types = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vectors = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Shifts3 = nb::ndarray<const std::int8_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Values = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// A one-dimensional NumPy array that owns the memory of a vector.
template <class T, class A>
nb::object as_array(std::vector<T, A>&& values) {
    auto* kept = new std::vector<T, A>(std::move(values));
    nb::capsule owner(kept, [](void* p) noexcept { delete static_cast<std::vector<T, A>*>(p); });
    const std::size_t shape[1] = {kept->size()};
    return nb::ndarray<nb::numpy, T>(kept->data(), 1, shape, owner).cast();
}

// The first count pairs of each atom (all when 0) closer than cutoff (all
// when 0), weighted by weights.
nb::object steinhardt(lists::Positions positions, lists::Matrix matrix, lists::Offsets offsets,
                      lists::Neighbors neighbors, lists::Shifts shifts,
                      lists::Distances distances, std::int64_t sorted, std::int64_t count,
                      double cutoff, std::optional<Values> weights,
                      const std::vector<int>& degrees, bool average, nb::handle progress,
                      const CancelToken* cancel) {
    const neighbor::ListView list = lists::view(positions, matrix, offsets, neighbors, shifts,
                                                distances, sorted, count, cutoff);
    if (weights && weights->shape(0) != neighbors.shape(0)) {
        throw std::invalid_argument("the weights do not fit the pairs");
    }
    const double* weight = weights ? weights->data() : nullptr;
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    auto* q = new double[static_cast<std::size_t>(n) * degrees.size()];
    nb::capsule owner(q, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(n), degrees.size()};
    nb::object result = nb::ndarray<nb::numpy, double>(q, 2, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::steinhardt(list, n, weight, degrees, average, q, control);
        },
        progress, cancel);
    return result;
}

// The solid-like bonds of each atom among the pairs steinhardt takes.
nb::object solid_bonds(lists::Positions positions, lists::Matrix matrix, lists::Offsets offsets,
                       lists::Neighbors neighbors, lists::Shifts shifts,
                       lists::Distances distances, std::int64_t sorted, std::int64_t count,
                       double cutoff, std::optional<Values> weights, int degree, double threshold,
                       nb::handle progress, const CancelToken* cancel) {
    const neighbor::ListView list = lists::view(positions, matrix, offsets, neighbors, shifts,
                                                distances, sorted, count, cutoff);
    if (weights && weights->shape(0) != neighbors.shape(0)) {
        throw std::invalid_argument("the weights do not fit the pairs");
    }
    const double* weight = weights ? weights->data() : nullptr;
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    std::vector<std::int32_t> bonds(static_cast<std::size_t>(n));
    run_released(
        [&](const Control& control) {
            analysis::solid_bonds(list, n, weight, degree, threshold, bonds.data(), control);
        },
        progress, cancel);
    return as_array(std::move(bonds));
}

// The bonds are the pairs closer than cutoff, or than the limit of the types
// of their atoms when limits is given.
nb::tuple clusters(lists::Positions positions, lists::Matrix matrix, lists::Offsets offsets,
                      lists::Neighbors neighbors, lists::Shifts shifts,
                      lists::Distances distances, std::int64_t sorted, double cutoff,
                   std::optional<Types> type, std::optional<lists::Limits> limits) {
    neighbor::ListView list =
        lists::view(positions, matrix, offsets, neighbors, shifts, distances, sorted, 0, cutoff);
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (limits) {
        if (!type) {
            throw std::invalid_argument("limits need a type for each atom");
        }
        list = lists::with_limits(list, n, *type, *limits);
    }
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    std::vector<std::int64_t> cluster(static_cast<std::size_t>(n)), sizes(static_cast<std::size_t>(n));
    std::vector<std::int32_t> images(3 * static_cast<std::size_t>(n));
    std::vector<std::uint8_t> spanning(static_cast<std::size_t>(n));
    std::int64_t count;
    {
        nb::gil_scoped_release release;
        count = analysis::clusters(list, n, cluster.data(), sizes.data(), images.data(),
                                   spanning.data());
    }
    sizes.resize(static_cast<std::size_t>(count));
    spanning.resize(static_cast<std::size_t>(count));
    return nb::make_tuple(as_array(std::move(cluster)), as_array(std::move(sizes)),
                          as_array(std::move(images)), as_array(std::move(spanning)));
}

// The bonds are the pairs closer than cutoff, or than the limit of the types
// of their atoms when limits is given.
nb::object bond_angles(lists::Positions positions, lists::Matrix matrix, lists::Offsets offsets,
                      lists::Neighbors neighbors, lists::Shifts shifts,
                      lists::Distances distances, std::int64_t sorted, double cutoff,
                       Types type, int types, int bins, std::optional<lists::Limits> limits,
                       nb::handle progress, const CancelToken* cancel) {
    neighbor::ListView list =
        lists::view(positions, matrix, offsets, neighbors, shifts, distances, sorted, 0, cutoff);
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (type.shape(0) != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("type must have one value per atom");
    }
    if (bins < 1 || types < 1) {
        throw std::invalid_argument("bins and types must be at least 1");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (type.data()[i] < 0 || type.data()[i] >= types) {
            throw std::invalid_argument("types must be from 0 to types - 1");
        }
    }
    if (limits) {
        list = lists::with_limits(list, n, type, *limits);
    }
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    const std::size_t pairs = static_cast<std::size_t>(types) * (types + 1) / 2;
    const std::size_t size = static_cast<std::size_t>(types) * pairs * bins;
    auto* counts = new std::int64_t[size];
    nb::capsule owner(counts, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
    const std::size_t shape[3] = {static_cast<std::size_t>(types), pairs, static_cast<std::size_t>(bins)};
    nb::object result = nb::ndarray<nb::numpy, std::int64_t>(counts, 3, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::bond_angles(list, n, type.data(), types, bins, counts, control);
        },
        progress, cancel);
    return result;
}

nb::object pair_entropy(lists::Positions positions, lists::Matrix matrix, lists::Offsets offsets,
                      lists::Neighbors neighbors, lists::Shifts shifts,
                      lists::Distances distances, std::int64_t sorted, double cutoff,
                        double sigma, double density, nb::handle progress,
                        const CancelToken* cancel) {
    const neighbor::ListView list =
        lists::view(positions, matrix, offsets, neighbors, shifts, distances, sorted, 0, 0.0);
    if (!(cutoff > 0.0) || !(sigma > 0.0) || !(density >= 0.0)) {
        throw std::invalid_argument("cutoff and sigma must be positive, density not negative");
    }
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    auto* entropy = new double[static_cast<std::size_t>(n)];
    nb::capsule owner(entropy, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const std::size_t shape[1] = {static_cast<std::size_t>(n)};
    nb::object result = nb::ndarray<nb::numpy, double>(entropy, 1, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::pair_entropy(list, n, cutoff, sigma, density, entropy, control);
        },
        progress, cancel);
    return result;
}

using Positions3 = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix3 = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Origin3 = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Periodic = nb::ndarray<const bool, nb::shape<3>, nb::c_contig, nb::device::cpu>;

nb::tuple voronoi(Positions3 positions, Matrix3 matrix, Origin3 origin, Periodic pbc,
                  double face_threshold, double relative_face_threshold, double edge_threshold,
                  bool faces, bool polyhedra, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    analysis::VoronoiCells cells;
    run_released(
        [&](const Control& control) {
            cells = analysis::voronoi(positions.data(), n, matrix.data(), origin.data(), pbc.data(),
                                      {face_threshold, relative_face_threshold, edge_threshold},
                                      faces, polyhedra, control);
        },
        progress, cancel);
    nb::object shapes = nb::none();
    if (polyhedra) {
        shapes = nb::make_tuple(
            as_array(std::move(cells.vertices)), as_array(std::move(cells.vertex_offsets)),
            as_array(std::move(cells.face_offsets)), as_array(std::move(cells.corner_offsets)),
            as_array(std::move(cells.corners)), as_array(std::move(cells.face_neighbors)),
            as_array(std::move(cells.face_areas)));
    }
    nb::object found = nb::none();
    if (faces) {
        found = nb::make_tuple(as_array(std::move(cells.offsets)),
                               as_array(std::move(cells.neighbors)),
                               as_array(std::move(cells.shifts)), as_array(std::move(cells.areas)),
                               as_array(std::move(cells.edges)));
    }
    return nb::make_tuple(as_array(std::move(cells.volumes)), as_array(std::move(cells.radii)),
                          as_array(std::move(cells.coordination)), found, shapes);
}

nb::tuple surface_mesh(Positions3 positions, Positions3 jittered, Matrix3 matrix,
                       Origin3 origin, Periodic pbc, double radius, int smoothing, bool regions,
                       nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (static_cast<std::int64_t>(jittered.shape(0)) != n) {
        throw std::invalid_argument("positions and jittered must have the same atoms");
    }
    analysis::SurfaceMesh mesh;
    analysis::Regions found;
    double solid_volume = 0.0;
    run_released(
        [&](const Control& control) {
            const analysis::Tetrahedra tetrahedra = analysis::delaunay(
                analysis::voronoi_vertices(jittered.data(), n, matrix.data(), origin.data(),
                                           pbc.data(), stage(control, 0, 4)),
                stage(control, 1, 4));
            if (tetrahedra.repeated) {
                throw std::invalid_argument(
                    "an atom meets its own periodic image: replicate the system along its "
                    "short periodic directions");
            }
            mesh = analysis::alpha_shape(tetrahedra, n, positions.data(), jittered.data(),
                                         matrix.data(), radius, stage(control, 2, 4));
            for (std::size_t t = 0; t < mesh.solid.size(); ++t) {
                solid_volume += mesh.solid[t] != 0 ? mesh.volumes[t] : 0.0;
            }
            if (regions) {
                found = analysis::surface_regions(tetrahedra, mesh);
            }
            analysis::smooth_mesh(mesh, matrix.data(), smoothing, stage(control, 3, 4));
        },
        progress, cancel);
    nb::object table = nb::none();
    if (regions) {
        table = nb::make_tuple(as_array(std::move(found.faces)), as_array(std::move(found.solid)),
                               as_array(std::move(found.exterior)),
                               as_array(std::move(found.periodic)),
                               as_array(std::move(found.volumes)));
    }
    return nb::make_tuple(as_array(std::move(mesh.atoms)), as_array(std::move(mesh.vertices)),
                          as_array(std::move(mesh.faces)), as_array(std::move(mesh.shifts)),
                          solid_volume, table);
}

nb::tuple dislocation_analysis(Positions3 positions, Positions3 jittered, Matrix3 matrix,
                               Matrix3 inverse, Origin3 origin, Periodic pbc, int crystal,
                               bool defects, int max_circuit, int stretch, double separation,
                               int smoothing, int mesh_smoothing, bool interface,
                               nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (static_cast<std::int64_t>(jittered.shape(0)) != n) {
        throw std::invalid_argument("positions and jittered must have the same atoms");
    }
    analysis::DislocationOptions options;
    options.crystal = crystal;
    options.defects = defects;
    options.max_circuit = max_circuit;
    options.stretch = stretch;
    options.separation = separation;
    options.smoothing = smoothing;
    options.mesh_smoothing = mesh_smoothing;
    options.interface = interface;
    analysis::Dislocations found;
    run_released(
        [&](const Control& control) {
            found = analysis::dislocation_analysis(positions.data(), jittered.data(), n,
                                                   matrix.data(), inverse.data(), origin.data(),
                                                   pbc.data(), options, control);
        },
        progress, cancel);
    std::vector<double> points, burgers;
    std::vector<std::int64_t> offsets = {0}, cluster, ends;
    std::vector<std::uint8_t> closed;
    for (const auto& s : found.network.segments) {
        points.insert(points.end(), s.points.begin(), s.points.end());
        offsets.push_back(static_cast<std::int64_t>(points.size() / 3));
        burgers.insert(burgers.end(), s.burgers.begin(), s.burgers.end());
        cluster.push_back(s.cluster);
        ends.insert(ends.end(), s.junctions.begin(), s.junctions.end());
        closed.push_back(s.closed ? 1 : 0);
    }
    return nb::make_tuple(
        as_array(std::move(found.structures.structure)), as_array(std::move(found.clusters.cluster)),
        as_array(std::move(found.clusters.structure)), as_array(std::move(found.clusters.size)),
        as_array(std::move(found.clusters.orientation)), as_array(std::move(points)),
        as_array(std::move(offsets)), as_array(std::move(burgers)), as_array(std::move(cluster)),
        as_array(std::move(ends)), as_array(std::move(closed)), found.network.junctions,
        as_array(std::move(found.defects.atoms)), as_array(std::move(found.defects.vertices)),
        as_array(std::move(found.defects.faces)), as_array(std::move(found.defects.shifts)),
        as_array(std::move(found.interface.atoms)), as_array(std::move(found.interface.vertices)),
        as_array(std::move(found.interface.faces)),
        as_array(std::move(found.interface.shifts)));
}

nb::object soap(Types species, int species_count, Vectors positions, Matrix3 matrix,
                Offsets offsets, Neighbors neighbors, Shifts3 shifts, int n_max, int l_max,
                double eta, Values alphas, Values betas, bool crossover, nb::handle progress,
                const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (static_cast<std::int64_t>(species.shape(0)) != n ||
        static_cast<std::int64_t>(offsets.shape(0)) != n + 1 ||
        neighbors.shape(0) != shifts.shape(0)) {
        throw std::invalid_argument("species, positions and neighbors do not match");
    }
    analysis::SoapBasis basis;
    basis.n_max = n_max;
    basis.l_max = l_max;
    basis.eta = eta;
    basis.alphas.assign(alphas.data(), alphas.data() + alphas.shape(0));
    basis.betas.assign(betas.data(), betas.data() + betas.shape(0));
    basis.crossover = crossover;
    const std::int64_t dimension = analysis::soap_dimension(basis, species_count);
    std::vector<double> found;
    run_released(
        [&](const Control& control) {
            found = analysis::soap(basis, species_count, species.data(), n, positions.data(),
                                   matrix.data(), offsets.data(), neighbors.data(),
                                   shifts.data(), control);
        },
        progress, cancel);
    auto* values = new std::vector<double>(std::move(found));
    nb::capsule owner(values, [](void* v) noexcept { delete static_cast<std::vector<double>*>(v); });
    const std::size_t shape[2] = {static_cast<std::size_t>(n), static_cast<std::size_t>(dimension)};
    return nb::ndarray<nb::numpy, double>(values->data(), 2, shape, owner).cast();
}

nb::tuple find_rings(Offsets offsets, Offsets other,
                     nb::ndarray<const std::int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> shifts,
                     int min_size, int max_size, nb::handle progress, const CancelToken* cancel) {
    analysis::RingBonds bonds;
    bonds.n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    if (bonds.n < 0 || static_cast<std::size_t>(offsets.data()[bonds.n]) != other.shape(0) ||
        other.shape(0) != shifts.shape(0)) {
        throw std::invalid_argument("offsets, other and shifts do not match");
    }
    bonds.offsets = offsets.data();
    bonds.other = other.data();
    bonds.shifts = shifts.data();
    analysis::Rings found;
    run_released(
        [&](const Control& control) {
            found = analysis::find_rings(bonds, min_size, max_size, control);
        },
        progress, cancel);
    return nb::make_tuple(as_array(std::move(found.atoms)), as_array(std::move(found.shifts)),
                          as_array(std::move(found.starts)));
}

nb::tuple isosurface(nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu> values,
                     Periodic pbc, double level, bool left_handed) {
    const std::array<std::int64_t, 3> n = {static_cast<std::int64_t>(values.shape(0)),
                                           static_cast<std::int64_t>(values.shape(1)),
                                           static_cast<std::int64_t>(values.shape(2))};
    analysis::Isosurface found;
    {
        nb::gil_scoped_release release;
        found = analysis::isosurface(values.data(), n, pbc.data(), level, left_handed);
    }
    return nb::make_tuple(as_array(std::move(found.vertices)), as_array(std::move(found.faces)),
                          as_array(std::move(found.shifts)));
}

nb::tuple elastic_strain(Positions3 positions, Matrix3 matrix, Matrix3 inverse, Origin3 origin,
                         Periodic pbc, int crystal, double a, double axial, nb::handle progress,
                         const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    analysis::ElasticStrainOptions options;
    options.crystal = crystal;
    options.a = a;
    options.axial = axial;
    analysis::ElasticStrain found;
    run_released(
        [&](const Control& control) {
            found = analysis::elastic_strain(positions.data(), n, matrix.data(), inverse.data(),
                                             origin.data(), pbc.data(), options, control);
        },
        progress, cancel);
    return nb::make_tuple(as_array(std::move(found.structure)), as_array(std::move(found.strain)));
}

nb::object steinhardt_nearest(Positions3 positions, Matrix3 matrix, Origin3 origin,
                              Periodic pbc, int k, const std::vector<int>& degrees,
                              nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    auto* q = new double[static_cast<std::size_t>(n) * degrees.size()];
    nb::capsule owner(q, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(n), degrees.size()};
    nb::object result = nb::ndarray<nb::numpy, double>(q, 2, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::steinhardt_nearest(positions.data(), n, matrix.data(), origin.data(),
                                         pbc.data(), k, degrees, q, control);
        },
        progress, cancel);
    return result;
}

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

using Limits = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Checks that a list in CSR form fits its distances and the types of its atoms.
std::int64_t check_list(Offsets offsets, Neighbors neighbors, Values distances, Types type,
                        int types) {
    if (offsets.shape(0) < 1 || distances.shape(0) != neighbors.shape(0) ||
        offsets.data()[offsets.shape(0) - 1] != static_cast<std::int64_t>(neighbors.shape(0)) ||
        type.shape(0) != offsets.shape(0) - 1 || types < 1) {
        throw std::invalid_argument("the pairs do not fit the atoms or their types");
    }
    const auto n = static_cast<std::int64_t>(type.shape(0));
    for (std::int64_t i = 0; i < n; ++i) {
        if (type.data()[i] < 0 || type.data()[i] >= types) {
            throw std::invalid_argument("a type is outside the range");
        }
    }
    return n;
}

nb::object pair_histogram_list(Offsets offsets, Neighbors neighbors, Values distances,
                               double r_max, int bins, Types type, int types,
                               nb::handle progress, const CancelToken* cancel) {
    const std::int64_t n = check_list(offsets, neighbors, distances, type, types);
    if (bins < 1 || !(r_max > 0.0)) {
        throw std::invalid_argument("bins and r_max must be positive");
    }
    std::vector<std::int64_t> counts;
    run_released(
        [&](const Control& control) {
            counts = analysis::pair_histogram_list(n, offsets.data(), neighbors.data(),
                                                   distances.data(), r_max, bins, type.data(),
                                                   types, control);
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

nb::object count_neighbors_list(Offsets offsets, Neighbors neighbors, Values distances,
                                Types type, int types, Limits limits, nb::handle progress,
                                const CancelToken* cancel) {
    const std::int64_t n = check_list(offsets, neighbors, distances, type, types);
    if (limits.shape(0) != static_cast<std::size_t>(types) ||
        limits.shape(1) != static_cast<std::size_t>(types)) {
        throw std::invalid_argument("the limits do not fit the types");
    }
    auto* counts = new std::int64_t[static_cast<std::size_t>(n) * types];
    nb::capsule owner(counts, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(n), static_cast<std::size_t>(types)};
    nb::object result = nb::ndarray<nb::numpy, std::int64_t>(counts, 2, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::count_neighbors_list(n, offsets.data(), neighbors.data(), distances.data(),
                                           type.data(), types, limits.data(), counts, control);
        },
        progress, cancel);
    return result;
}

nb::object count_neighbors(Positions3 positions, Matrix3 matrix, Origin3 origin, Periodic pbc,
                           Types type, int types, Limits limits, bool per_pair,
                           nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (type.shape(0) != positions.shape(0) || types < 1 ||
        limits.shape(0) != static_cast<std::size_t>(types) ||
        limits.shape(1) != static_cast<std::size_t>(types)) {
        throw std::invalid_argument("the types do not fit the atoms or the limits");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (type.data()[i] < 0 || type.data()[i] >= types) {
            throw std::invalid_argument("a type is outside the range");
        }
    }
    auto* counts = new std::int64_t[static_cast<std::size_t>(n) * types];
    nb::capsule owner(counts, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(n), static_cast<std::size_t>(types)};
    nb::object result = nb::ndarray<nb::numpy, std::int64_t>(counts, 2, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::count_neighbors(positions.data(), n, matrix.data(), origin.data(),
                                      pbc.data(), type.data(), types, limits.data(), per_pair,
                                      counts, control);
        },
        progress, cancel);
    return result;
}

nb::tuple structure_factor(Positions3 positions, Matrix3 matrix, Types type, int types,
                           double q_max, int bins, std::int64_t block, nb::handle progress,
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
                                       q_max, bins, sums, vectors, block, control);
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



// Arrays for the results of the atomic strain of n atoms, the strain tensor
// and the non-affine displacement only when asked for, and the tuple
// (shear, volumetric, strain (n, 3, 3) or None, nonaffine or None) that owns
// them.
struct StrainArrays {
    Buffer<double> strain, shear, volumetric, nonaffine;
    StrainArrays(std::int64_t n, bool tensor, bool d2min)
        : strain(tensor ? 9 * static_cast<std::size_t>(n) : 0), shear(static_cast<std::size_t>(n)),
          volumetric(static_cast<std::size_t>(n)),
          nonaffine(d2min ? static_cast<std::size_t>(n) : 0), tensor_(tensor), d2min_(d2min) {}
    analysis::StrainOutput output() {
        return {tensor_ ? strain.data() : nullptr, shear.data(), volumetric.data(),
                d2min_ ? nonaffine.data() : nullptr};
    }
    nb::tuple to_tuple() {
        const std::size_t n = shear.size();
        nb::object tensor = nb::none(), d2min = nb::none();
        if (tensor_) {
            auto* kept = new Buffer<double>(std::move(strain));
            nb::capsule owner(kept,
                              [](void* p) noexcept { delete static_cast<Buffer<double>*>(p); });
            const std::size_t shape[3] = {n, 3, 3};
            tensor = nb::cast(nb::ndarray<nb::numpy, double>(kept->data(), 3, shape, owner));
        }
        if (d2min_) {
            d2min = as_array(std::move(nonaffine));
        }
        return nb::make_tuple(as_array(std::move(shear)), as_array(std::move(volumetric)), tensor,
                              d2min);
    }

private:
    bool tensor_, d2min_;
};

analysis::StrainInput strain_input(Positions3 reference, Matrix3 reference_inverse,
                                   Origin3 reference_origin, Positions3 positions,
                                   Matrix3 inverse, Origin3 origin, Periodic pbc,
                                   Matrix3 reference_matrix, Matrix3 matrix) {
    if (positions.shape(0) != reference.shape(0)) {
        throw std::invalid_argument("the reference and the system have different atoms");
    }
    return {static_cast<std::int64_t>(reference.shape(0)),
            reference.data(),
            reference_inverse.data(),
            reference_origin.data(),
            positions.data(),
            inverse.data(),
            origin.data(),
            pbc.data(),
            reference_matrix.data(),
            matrix.data()};
}

nb::tuple atomic_strain_search(double cutoff, Positions3 reference, Matrix3 reference_inverse,
                               Origin3 reference_origin, Positions3 positions, Matrix3 inverse,
                               Origin3 origin, Periodic pbc, Matrix3 reference_matrix,
                               Matrix3 matrix, bool tensor, bool d2min, nb::handle progress,
                               const CancelToken* cancel) {
    const analysis::StrainInput input =
        strain_input(reference, reference_inverse, reference_origin, positions, inverse, origin,
                     pbc, reference_matrix, matrix);
    StrainArrays arrays(input.n, tensor, d2min);
    run_released(
        [&](const Control& control) {
            analysis::atomic_strain_search(cutoff, input, arrays.output(), control);
        },
        progress, cancel);
    return arrays.to_tuple();
}

nb::tuple atomic_strain_list(Offsets offsets, Neighbors neighbors, Shifts3 shifts,
                             Values distances, double cutoff, Positions3 reference,
                             Matrix3 reference_inverse, Origin3 reference_origin,
                             Positions3 positions, Matrix3 inverse, Origin3 origin, Periodic pbc,
                             Matrix3 reference_matrix, Matrix3 matrix, bool tensor, bool d2min,
                             nb::handle progress, const CancelToken* cancel) {
    const analysis::StrainInput input =
        strain_input(reference, reference_inverse, reference_origin, positions, inverse, origin,
                     pbc, reference_matrix, matrix);
    if (offsets.shape(0) != static_cast<std::size_t>(input.n) + 1 ||
        shifts.shape(0) != neighbors.shape(0) || distances.shape(0) != neighbors.shape(0) ||
        offsets.data()[input.n] != static_cast<std::int64_t>(neighbors.shape(0))) {
        throw std::invalid_argument("the pairs do not fit the atoms");
    }
    StrainArrays arrays(input.n, tensor, d2min);
    run_released(
        [&](const Control& control) {
            analysis::atomic_strain_list(offsets.data(), neighbors.data(), shifts.data(),
                                         distances.data(), cutoff, input, arrays.output(),
                                         control);
        },
        progress, cancel);
    return arrays.to_tuple();
}

}  // namespace

using Trajectory = nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

// positions: (n, frames, 3), the trajectory of each atom in a row.
nb::tuple lindemann(Trajectory positions, nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    const auto frames = static_cast<std::int64_t>(positions.shape(1));
    if (positions.shape(2) != 3 || n < 2 || frames < 1) {
        throw std::invalid_argument("positions must be (atoms >= 2, frames >= 1, 3)");
    }
    std::vector<double> atomic(static_cast<std::size_t>(n)), running(static_cast<std::size_t>(frames));
    run_released(
        [&](const Control& control) {
            analysis::lindemann(n, frames, positions.data(), atomic.data(), running.data(), control);
        },
        progress, cancel);
    return nb::make_tuple(as_array(std::move(atomic)), as_array(std::move(running)));
}

// The neighbors are the pairs closer than cutoff.
nb::object atomic_temperature(lists::Positions positions, lists::Matrix matrix,
                              lists::Offsets offsets, lists::Neighbors neighbors,
                              lists::Shifts shifts, lists::Distances distances,
                              std::int64_t sorted, double cutoff, Values mass,
                              Positions3 velocities, double energy_unit, double boltzmann,
                              nb::handle progress, const CancelToken* cancel) {
    const neighbor::ListView list =
        lists::view(positions, matrix, offsets, neighbors, shifts, distances, sorted, 0, cutoff);
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (mass.shape(0) != static_cast<std::size_t>(n) ||
        velocities.shape(0) != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("the masses and velocities do not fit the atoms");
    }
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
    std::vector<double> temperature(static_cast<std::size_t>(n));
    run_released(
        [&](const Control& control) {
            analysis::atomic_temperature(list, n, mass.data(), velocities.data(), energy_unit,
                                         boltzmann, temperature.data(), control);
        },
        progress, cancel);
    return as_array(std::move(temperature));
}

nb::object pair_entropy_search(Positions3 positions, Matrix3 matrix, Origin3 origin,
                               Periodic pbc, double cutoff, double sigma, double density,
                               nb::handle progress, const CancelToken* cancel) {
    if (!(cutoff > 0.0) || !(sigma > 0.0) || !(density >= 0.0)) {
        throw std::invalid_argument("cutoff and sigma must be positive, density not negative");
    }
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    std::vector<double> entropy(static_cast<std::size_t>(n));
    run_released(
        [&](const Control& control) {
            analysis::pair_entropy_search(positions.data(), n, matrix.data(), origin.data(),
                                          pbc.data(), cutoff, sigma, density, entropy.data(),
                                          control);
        },
        progress, cancel);
    return as_array(std::move(entropy));
}

nb::tuple clusters_search(Positions3 positions, Matrix3 matrix, Origin3 origin, Periodic pbc,
                          Types type, int types, Limits limits, bool per_pair,
                          nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (type.shape(0) != positions.shape(0) || types < 1 ||
        limits.shape(0) != static_cast<std::size_t>(types) ||
        limits.shape(1) != static_cast<std::size_t>(types)) {
        throw std::invalid_argument("the types do not fit the atoms or the limits");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (type.data()[i] < 0 || type.data()[i] >= types) {
            throw std::invalid_argument("a type is outside the range");
        }
    }
    std::vector<std::int64_t> cluster(static_cast<std::size_t>(n)), sizes(static_cast<std::size_t>(n));
    std::vector<std::int32_t> images(3 * static_cast<std::size_t>(n));
    std::vector<std::uint8_t> spanning(static_cast<std::size_t>(n));
    std::int64_t count = 0;
    run_released(
        [&](const Control& control) {
            count = analysis::clusters_search(positions.data(), n, matrix.data(), origin.data(),
                                              pbc.data(), type.data(), types, limits.data(),
                                              per_pair, cluster.data(), sizes.data(),
                                              images.data(), spanning.data(), control);
        },
        progress, cancel);
    sizes.resize(static_cast<std::size_t>(count));
    spanning.resize(static_cast<std::size_t>(count));
    return nb::make_tuple(as_array(std::move(cluster)), as_array(std::move(sizes)),
                          as_array(std::move(images)), as_array(std::move(spanning)));
}

nb::object bond_angles_search(Positions3 positions, Matrix3 matrix, Origin3 origin, Periodic pbc,
                              Types type, int types, Limits limits, bool per_pair, int bins,
                              nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (type.shape(0) != positions.shape(0) || types < 1 || bins < 1 ||
        limits.shape(0) != static_cast<std::size_t>(types) ||
        limits.shape(1) != static_cast<std::size_t>(types)) {
        throw std::invalid_argument("the types do not fit the atoms or the limits");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (type.data()[i] < 0 || type.data()[i] >= types) {
            throw std::invalid_argument("a type is outside the range");
        }
    }
    const std::size_t pairs = static_cast<std::size_t>(types) * (types + 1) / 2;
    auto* counts = new std::int64_t[static_cast<std::size_t>(types) * pairs * bins];
    nb::capsule owner(counts, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
    const std::size_t shape[3] = {static_cast<std::size_t>(types), pairs, static_cast<std::size_t>(bins)};
    nb::object result = nb::ndarray<nb::numpy, std::int64_t>(counts, 3, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::bond_angles_search(positions.data(), n, matrix.data(), origin.data(),
                                         pbc.data(), type.data(), types, limits.data(), per_pair,
                                         bins, counts, control);
        },
        progress, cancel);
    return result;
}

nb::object atomic_temperature_search(Positions3 positions, Matrix3 matrix, Origin3 origin,
                                     Periodic pbc, double cutoff, Values mass,
                                     Positions3 velocities, double energy_unit, double boltzmann,
                                     nb::handle progress, const CancelToken* cancel) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    if (mass.shape(0) != static_cast<std::size_t>(n) ||
        velocities.shape(0) != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("the masses and velocities do not fit the atoms");
    }
    if (!(cutoff > 0.0)) {
        throw std::invalid_argument("cutoff must be positive");
    }
    std::vector<double> temperature(static_cast<std::size_t>(n));
    run_released(
        [&](const Control& control) {
            analysis::atomic_temperature_search(positions.data(), n, matrix.data(), origin.data(),
                                                pbc.data(), cutoff, mass.data(),
                                                velocities.data(), energy_unit, boltzmann,
                                                temperature.data(), control);
        },
        progress, cancel);
    return as_array(std::move(temperature));
}

void bind_analysis(nb::module_& m) {
    m.def("steinhardt_nearest", &steinhardt_nearest, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("origin"), nb::arg("pbc"), nb::arg("k"), nb::arg("degrees"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("steinhardt", &steinhardt,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("count"),
          nb::arg("cutoff"), nb::arg("weights").none(), nb::arg("degrees"), nb::arg("average"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("solid_bonds", &solid_bonds,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("count"),
          nb::arg("cutoff"), nb::arg("weights").none(), nb::arg("degree"), nb::arg("threshold"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("clusters", &clusters,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("cutoff"),
          nb::arg("type").none() = nb::none(), nb::arg("limits").none() = nb::none());
    m.def("dislocation_analysis", &dislocation_analysis, nb::arg("positions"),
          nb::arg("jittered"), nb::arg("matrix"), nb::arg("inverse"), nb::arg("origin"),
          nb::arg("pbc"), nb::arg("crystal"), nb::arg("defects"), nb::arg("max_circuit"),
          nb::arg("stretch"), nb::arg("separation"), nb::arg("smoothing"),
          nb::arg("mesh_smoothing"), nb::arg("interface"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("soap", &soap, nb::arg("species"), nb::arg("species_count"), nb::arg("positions"),
          nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"), nb::arg("shifts"),
          nb::arg("n_max"), nb::arg("l_max"), nb::arg("eta"), nb::arg("alphas"), nb::arg("betas"),
          nb::arg("crossover"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("find_rings", &find_rings, nb::arg("offsets"), nb::arg("other"), nb::arg("shifts"),
          nb::arg("min_size"), nb::arg("max_size"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("isosurface", &isosurface, nb::arg("values"), nb::arg("pbc"), nb::arg("level"),
          nb::arg("left_handed"));
    m.def("elastic_strain", &elastic_strain, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("inverse"), nb::arg("origin"), nb::arg("pbc"), nb::arg("crystal"), nb::arg("a"),
          nb::arg("axial"), nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("surface_mesh", &surface_mesh, nb::arg("positions"), nb::arg("jittered"),
          nb::arg("matrix"), nb::arg("origin"), nb::arg("pbc"), nb::arg("radius"),
          nb::arg("smoothing"), nb::arg("regions"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("voronoi", &voronoi, nb::arg("positions"), nb::arg("matrix"), nb::arg("origin"),
          nb::arg("pbc"), nb::arg("face_threshold"), nb::arg("relative_face_threshold"),
          nb::arg("edge_threshold"), nb::arg("faces"), nb::arg("polyhedra"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("atomic_strain_search", &atomic_strain_search, nb::arg("cutoff"), nb::arg("reference"),
          nb::arg("reference_inverse"), nb::arg("reference_origin"), nb::arg("positions"),
          nb::arg("inverse"), nb::arg("origin"), nb::arg("pbc"), nb::arg("reference_matrix"),
          nb::arg("matrix"), nb::arg("tensor"), nb::arg("d2min"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("atomic_strain_list", &atomic_strain_list, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("cutoff"), nb::arg("reference"),
          nb::arg("reference_inverse"), nb::arg("reference_origin"), nb::arg("positions"),
          nb::arg("inverse"), nb::arg("origin"), nb::arg("pbc"), nb::arg("reference_matrix"),
          nb::arg("matrix"), nb::arg("tensor"), nb::arg("d2min"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("structure_factor", &structure_factor, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("type"), nb::arg("types"), nb::arg("q_max"), nb::arg("bins"),
          nb::arg("block") = 0, nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("pair_histogram", &pair_histogram, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("origin"), nb::arg("pbc"), nb::arg("r_max"), nb::arg("bins"), nb::arg("type"),
          nb::arg("types"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("pair_entropy", &pair_entropy,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("cutoff"),
          nb::arg("sigma"), nb::arg("density"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("lindemann", &lindemann, nb::arg("positions"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("atomic_temperature", &atomic_temperature,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("cutoff"),
          nb::arg("mass"), nb::arg("velocities"), nb::arg("energy_unit"), nb::arg("boltzmann"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("pair_entropy_search", &pair_entropy_search, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("origin"), nb::arg("pbc"), nb::arg("cutoff"), nb::arg("sigma"),
          nb::arg("density"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("pair_histogram_list", &pair_histogram_list, nb::arg("offsets"),
          nb::arg("neighbors"), nb::arg("distances"), nb::arg("r_max"), nb::arg("bins"),
          nb::arg("type"), nb::arg("types"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("count_neighbors_list", &count_neighbors_list, nb::arg("offsets"),
          nb::arg("neighbors"), nb::arg("distances"), nb::arg("type"), nb::arg("types"),
          nb::arg("limits"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("count_neighbors", &count_neighbors, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("origin"), nb::arg("pbc"), nb::arg("type"), nb::arg("types"),
          nb::arg("limits"), nb::arg("per_pair"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("clusters_search", &clusters_search, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("origin"), nb::arg("pbc"), nb::arg("type"), nb::arg("types"),
          nb::arg("limits"), nb::arg("per_pair"), nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("bond_angles_search", &bond_angles_search, nb::arg("positions"), nb::arg("matrix"),
          nb::arg("origin"), nb::arg("pbc"), nb::arg("type"), nb::arg("types"),
          nb::arg("limits"), nb::arg("per_pair"), nb::arg("bins"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("atomic_temperature_search", &atomic_temperature_search, nb::arg("positions"),
          nb::arg("matrix"), nb::arg("origin"), nb::arg("pbc"), nb::arg("cutoff"),
          nb::arg("mass"), nb::arg("velocities"), nb::arg("energy_unit"), nb::arg("boltzmann"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("bond_angles", &bond_angles,
          nb::arg("positions"), nb::arg("matrix"), nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shifts"), nb::arg("distances"), nb::arg("sorted"), nb::arg("cutoff"),
          nb::arg("type"), nb::arg("types"), nb::arg("bins"), nb::arg("limits").none(),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
