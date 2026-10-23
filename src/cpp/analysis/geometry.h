// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Geometry of bonds: clusters of bonded atoms and the distribution of bond
// angles.

#pragma once

#include <cstdint>

#include "core/index.h"
#include "core/parallel.h"
#include "neighbor/list_view.h"

namespace mdapy::analysis {

// The cluster of each of n atoms, given its bonds, its pairs in list with
// the shift of each (the bond vector x_j - x_i + shift . H): clusters are
// numbered from 0 by decreasing size, ties by their first atom. Returns the
// number of clusters; sizes and spanning (at least n long) get the size of
// each and whether it joins another image of itself across the periodic
// boundaries, and images (n x 3) the periodic image of each atom that places
// it next to the atoms it is bonded to, the first atom of each cluster at
// image 0. The images of a spanning cluster depend on the order of the bonds.
std::int64_t clusters(const neighbor::ListView& list, std::int64_t n, std::int64_t* cluster,
                      std::int64_t* sizes, std::int32_t* images, std::uint8_t* spanning);

// The same with the bonds found on a cell grid (positions: n x 3; matrix:
// rows a, b, c; origin: 3; pbc: 3): two atoms are bonded when closer than the
// limit of their pair of types, compared as count_neighbors does.
std::int64_t clusters_search(const double* positions, std::int64_t n, const double* matrix,
                             const double* origin, const bool* pbc, const std::int32_t* type,
                             int types, const double* limits, bool per_pair,
                             std::int64_t* cluster, std::int64_t* sizes, std::int32_t* images,
                             std::uint8_t* spanning, const Control& control = {});

// Counts of the angles between the bonds of each atom, its pairs in list,
// in bins of equal width from 0 to 180 degrees, by
// the type of the atom and the types a <= b of the two neighbors (type: n
// values below types): counts is types x pairs x bins, the pair (a, b)
// numbered in the order (0, 0), (0, 1), ..., (1, 1), (1, 2), ...
void bond_angles(const neighbor::ListView& list, std::int64_t n, const std::int32_t* type,
                 int types, int bins, std::int64_t* counts, const Control& control = {});

// The same with the bonds found on a cell grid (positions: n x 3; matrix:
// rows a, b, c; origin: 3; pbc: 3), bonded as count_neighbors counts them.
void bond_angles_search(const double* positions, std::int64_t n, const double* matrix,
                        const double* origin, const bool* pbc, const std::int32_t* type,
                        int types, const double* limits, bool per_pair, int bins,
                        std::int64_t* counts, const Control& control = {});

// The Lindemann index of n atoms over frames of their unwrapped positions
// (n x frames x 3, each atom's trajectory in a row): for each pair, the
// standard deviation of its distance over its mean, by Welford's running
// sums. atomic (n) gets each atom's mean over its pairs and running (frames)
// the mean over all pairs of the frames up to each one.
void lindemann(std::int64_t n, std::int64_t frames, const double* positions, double* atomic,
               double* running, const Control& control = {});

// The temperature of each of n atoms with its neighbors, its pairs in list
// (images of the atom itself left out), from the kinetic energy relative to
// their center of mass, 3 (members - 1) degrees of freedom: twice that
// energy times energy_unit over (3 (members - 1) boltzmann); NaN without
// neighbors.
void atomic_temperature(const neighbor::ListView& list, std::int64_t n, const double* mass,
                        const double* velocities, double energy_unit, double boltzmann,
                        double* temperature, const Control& control = {});

// The same with the neighbors closer than cutoff found on a cell grid
// (positions: n x 3; matrix: rows a, b, c; origin: 3; pbc: 3).
void atomic_temperature_search(const double* positions, std::int64_t n, const double* matrix,
                               const double* origin, const bool* pbc, double cutoff,
                               const double* mass, const double* velocities, double energy_unit,
                               double boltzmann, double* temperature, const Control& control = {});

// The pair entropy fingerprint of n atoms (Piaggi and Parrinello) from the
// distances of their pairs in list closer than cutoff: g(r) on the points
// k sigma up to the cutoff, each neighbor smeared by a Gaussian of width
// sigma, integrated by the trapezoid rule. The number density is density
// for every atom or, when density is zero, each atom's neighbors over the
// volume of the cutoff sphere.
void pair_entropy(const neighbor::ListView& list, std::int64_t n, double cutoff, double sigma,
                  double density, double* entropy, const Control& control = {});

// The same with the neighbors found on a cell grid, without a neighbor list
// (positions: n x 3; matrix: rows a, b, c; origin: 3; pbc: 3). The number
// density is density for every atom or, when density is zero, each atom's
// neighbors over the volume of the cutoff sphere.
void pair_entropy_search(const double* positions, std::int64_t n, const double* matrix,
                         const double* origin, const bool* pbc, double cutoff, double sigma,
                         double density, double* entropy, const Control& control = {});

}  // namespace mdapy::analysis
