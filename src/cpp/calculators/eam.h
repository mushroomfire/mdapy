// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The embedded atom method with tabulated functions, interpolated as LAMMPS
// pair_style eam does: a cubic spline through the tabulated values whose
// slopes come from fourth-order finite differences.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"
#include "neighbor/frame.h"

namespace mdapy::calculators {

// A function tabulated at x = 0, delta, 2 delta, ... and its spline.
class Spline {
public:
    // values: n >= 5 samples
    Spline(const double* values, int n, double delta);
    // The value and the derivative at x. Beyond the table the spline of the
    // last interval continues up to its end, as in LAMMPS. Defined here, so
    // that the kernels of every file inline them.
    void evaluate(double x, double& value, double& derivative) const {
        const auto& c = row(x, x);
        value = ((c[3] * x + c[4]) * x + c[5]) * x + c[6];
        derivative = (c[0] * x + c[1]) * x + c[2];
    }
    double value(double x) const {
        const auto& c = row(x, x);
        return ((c[3] * x + c[4]) * x + c[5]) * x + c[6];
    }
    // The table for the GPU: n + 1 rows of 7 coefficients, row 0 unused.
    int size() const { return n_; }
    double inverse_delta() const { return inverse_delta_; }
    const std::array<double, 7>* coefficients() const { return coefficients_.data(); }

private:
    // The coefficients of the interval of x, and in p the place in it.
    const std::array<double, 7>& row(double x, double& p) const {
        p = x * inverse_delta_ + 1.0;
        int m = static_cast<int>(p);
        m = std::max(1, std::min(m, n_ - 1));
        p -= m;
        p = std::min(p, 1.0);
        return coefficients_[static_cast<std::size_t>(m)];
    }

    int n_;
    double inverse_delta_;
    std::vector<std::array<double, 7>> coefficients_;  // index 1 to n
};

struct EamPotential {
    int elements = 0;
    double cutoff = 0.0;
    double rho_max = 0.0;  // the embedding energy is extrapolated linearly beyond
    std::vector<Spline> embedding;  // F(rho), one per element
    // rho(r) an atom gives its neighbours: one per element (eam/alloy), or
    // one per element and element of the neighbour, source * elements +
    // receiver (eam/fs)
    std::vector<Spline> density;
    bool density_per_pair = false;
    std::vector<Spline> pair;  // r * phi(r), one per pair of elements a >= b

    // The density an atom of element source gives one of element receiver.
    const Spline& density_of(int source, int receiver) const {
        return density[static_cast<std::size_t>(density_per_pair ? source * elements + receiver
                                                                 : source)];
    }

    const Spline& pair_of(int a, int b) const {
        if (a < b) {
            const int t = a;
            a = b;
            b = t;
        }
        return pair[static_cast<std::size_t>(a * (a + 1) / 2 + b)];
    }
};

// Energies (n), forces (n x 3) and virials (n x 9) of n atoms at positions
// (n x 3) in a box with lattice vectors as rows of matrix (3 x 3), with a
// full neighbor list in CSR form whose pair vectors are those of
// neighbor::pair_vector. The virial of each pair is shared equally by its
// two atoms.
// The potential of tables: embedding (elements, nrho); density (densities,
// nr), one per element or per element and element of the neighbour; pair
// (elements * (elements + 1) / 2, nr), the lower triangle by rows. rho_max:
// where the embedding energy goes on linearly. Throws std::invalid_argument
// for tables that do not fit.
EamPotential make_eam_potential(const double* embedding, int elements, int nrho,
                                const double* density, int densities, const double* pair,
                                int pairs, int nr, double drho, double dr, double cutoff,
                                double rho_max);

void compute_eam(const EamPotential& potential, std::int64_t n, const std::int32_t* species,
                 const double* positions, const double* matrix, const std::int64_t* offsets,
                 const index_t* neighbors, const std::int8_t* shifts, double* energies,
                 double* forces, double* virials, const Control& control = {}, bool heat = false);

// The two passes of compute_eam for the first centers atoms of n, as the
// processes of MPI split them: the electron density of each centre, its
// embedding energy (energies) and the slope F'(rho) of its embedding
// function; then, given the slopes of every atom of the neighbor list, the
// forces on the centres, their pair energies added to energies, and their
// virials unless virials is null.
void eam_densities(const EamPotential& potential, std::int64_t centers,
                   const std::int32_t* species, const double* positions, const double* matrix,
                   const std::int64_t* offsets, const index_t* neighbors,
                   const std::int8_t* shifts, double* energies, double* slopes,
                   const Control& control = {});
void eam_forces(const EamPotential& potential, std::int64_t centers, const std::int32_t* species,
                const double* positions, const double* matrix, const std::int64_t* offsets,
                const index_t* neighbors, const std::int8_t* shifts, const double* slopes,
                double* energies, double* forces, double* virials, const Control& control = {},
                bool heat = false);

// The same on one thread, each pair once among the processes of MPI, the
// pairs given by once_pairs (calculators/once.h), as LAMMPS's newton on: the
// densities that the pairs give each atom of the list, ghosts among them,
// added to rho (n); the
// embedding energies and slopes of the centres from their whole densities;
// the forces, pair energies and virials (unless null) that the pairs give
// every atom, added to forces (n x 3), energies (n) and virials (n x 9).
void eam_densities_once(const EamPotential& potential, std::int64_t centers,
                        const std::int32_t* species, const double* positions,
                        const double* matrix, const std::int64_t* offsets,
                        const index_t* neighbors, const std::int8_t* shifts, double* rho);
void eam_embedding(const EamPotential& potential, std::int64_t centers,
                   const std::int32_t* species, const double* rho, double* energies,
                   double* slopes);
void eam_forces_once(const EamPotential& potential, std::int64_t centers,
                     const std::int32_t* species, const double* positions, const double* matrix,
                     const std::int64_t* offsets, const index_t* neighbors,
                     const std::int8_t* shifts, const double* slopes, double* energies,
                     double* forces, double* virials, bool heat);

}  // namespace mdapy::calculators
