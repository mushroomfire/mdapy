// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Many-body potentials whose energy is a sum of terms computed at centre
// atoms, each a function of the vectors from its centre to its neighbors.
// A first pass computes the terms of each centre: the derivative of their
// energy with respect to the vector of each of its pairs, and the parts of
// their energy and virial that go to the centre and to each neighbor, as
// LAMMPS tallies them. A second pass gathers, for each atom, the forces and
// the parts of the energies and virials of the terms it takes part in, from
// its own pairs and their reverses. No atom writes to another's results, so
// the results are the same for any number of threads.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::calculators {

// For each pair p = (i, j, s) of a full neighbor list, the pair (j, i, -s).
// Throws std::invalid_argument if the list is not symmetric.
std::vector<std::int64_t> reverse_pairs(std::int64_t n, const std::int64_t* offsets,
                                        const index_t* neighbors, const std::int8_t* shifts,
                                        const Control& control);

// What the terms of each centre atom give: per pair p of the centre,
// derivative (3) of their energy with respect to r_p = x_j - x_i, and the
// energy (1) and the virial (9) that go to the neighbor j; per atom, the
// energy and the virial that stay with the centre. The virial of a term
// is the sum over its neighbors q of r_q (x) f_q, f_q the force of the term
// on q, as LAMMPS's v_tally3.
struct Shares {
    std::vector<double> derivative, energy, virial;      // per pair
    std::vector<double> own_energy, own_virial;          // per atom

    Shares(std::int64_t atoms, std::int64_t pairs)
        : derivative(3 * static_cast<std::size_t>(pairs), 0.0),
          energy(static_cast<std::size_t>(pairs), 0.0),
          virial(9 * static_cast<std::size_t>(pairs), 0.0),
          own_energy(static_cast<std::size_t>(atoms), 0.0),
          own_virial(9 * static_cast<std::size_t>(atoms), 0.0) {}

    // A term on the pair p alone, whose energy changes by the force f on the
    // neighbor (the opposite on the centre i); its virial r_p (x) f goes in
    // the part to_neighbor to the neighbor and the rest to the centre.
    void pair_force(std::int64_t i, std::int64_t p, const double* r, const double* f,
                    double to_neighbor) {
        double* t = &derivative[3 * static_cast<std::size_t>(p)];
        double* w = &virial[9 * static_cast<std::size_t>(p)];
        double* own = &own_virial[9 * static_cast<std::size_t>(i)];
        for (int a = 0; a < 3; ++a) {
            t[a] -= f[a];
            for (int b = 0; b < 3; ++b) {
                const double v = r[a] * f[b];
                w[3 * a + b] += to_neighbor * v;
                own[3 * a + b] += (1.0 - to_neighbor) * v;
            }
        }
    }
};

// Energies (n), forces (n x 3) and virials (n x 9) gathered from the shares.
void gather(const Shares& shares, std::int64_t n, const std::int64_t* offsets,
            const index_t* neighbors, const std::int8_t* shifts, double* energies,
            double* forces, double* virials, const Control& control);

}  // namespace mdapy::calculators
