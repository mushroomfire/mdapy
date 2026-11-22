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
    // per atom; own_force: forces an atom found on itself, as from pair
    // terms it computes whole, as its neighbor does
    std::vector<double> own_energy, own_virial, own_force;

    Shares(std::int64_t atoms, std::int64_t pairs)
        : derivative(3 * static_cast<std::size_t>(pairs), 0.0),
          energy(static_cast<std::size_t>(pairs), 0.0),
          virial(9 * static_cast<std::size_t>(pairs), 0.0),
          own_energy(static_cast<std::size_t>(atoms), 0.0),
          own_virial(9 * static_cast<std::size_t>(atoms), 0.0),
          own_force(3 * static_cast<std::size_t>(atoms), 0.0) {}

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

// Where the terms of a centre go, as the kernels of the potentials hand
// them: own_energy(i, e) to the centre, pair_energy(p, j, e) to the neighbor
// j of its pair p, pair_force(i, p, j, r, f, to_neighbor) as
// Shares::pair_force, derivative(i, p, j, f) a force f on j and -f on the
// centre, virial3(i, pj, j, pk, k, w) the virial w (9) to each of the
// three atoms of a term.
//
// ToShares: into the shares, which gather gives the atoms after, for any
// number of threads.
struct ToShares {
    Shares& shares;
    void own_energy(std::int64_t i, double e) { shares.own_energy[static_cast<std::size_t>(i)] += e; }
    void pair_energy(std::int64_t p, std::int64_t, double e) {
        shares.energy[static_cast<std::size_t>(p)] += e;
    }
    void pair_force(std::int64_t i, std::int64_t p, std::int64_t, const double* r, const double* f,
                    double to_neighbor) {
        shares.pair_force(i, p, r, f, to_neighbor);
    }
    void derivative(std::int64_t, std::int64_t p, std::int64_t, const double* f) {
        double* t = &shares.derivative[3 * static_cast<std::size_t>(p)];
        for (int a = 0; a < 3; ++a) t[a] -= f[a];
    }
    void virial3(std::int64_t i, std::int64_t pj, std::int64_t, std::int64_t pk, std::int64_t,
                 const double* w) {
        double* own = &shares.own_virial[9 * static_cast<std::size_t>(i)];
        double* wj = &shares.virial[9 * static_cast<std::size_t>(pj)];
        double* wk = &shares.virial[9 * static_cast<std::size_t>(pk)];
        for (int c = 0; c < 9; ++c) {
            own[c] += w[c];
            wj[c] += w[c];
            wk[c] += w[c];
        }
    }
};

// ToAtoms: straight into the energies (n), forces (n x 3) and virials
// (n x 9, or null for none) of the atoms, on one thread, as LAMMPS with
// newton on: each term once, its parts on the ghosts sent back after.
struct ToAtoms {
    double* energies;
    double* forces;
    double* virials;
    void own_energy(std::int64_t i, double e) { energies[i] += e; }
    void pair_energy(std::int64_t, std::int64_t j, double e) { energies[j] += e; }
    void pair_force(std::int64_t i, std::int64_t, std::int64_t j, const double* r, const double* f,
                    double to_neighbor) {
        for (int a = 0; a < 3; ++a) {
            forces[3 * j + a] += f[a];
            forces[3 * i + a] -= f[a];
        }
        if (virials == nullptr) return;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const double v = r[a] * f[b];
                virials[9 * j + 3 * a + b] += to_neighbor * v;
                virials[9 * i + 3 * a + b] += (1.0 - to_neighbor) * v;
            }
        }
    }
    void derivative(std::int64_t i, std::int64_t, std::int64_t j, const double* f) {
        for (int a = 0; a < 3; ++a) {
            forces[3 * j + a] += f[a];
            forces[3 * i + a] -= f[a];
        }
    }
    void virial3(std::int64_t i, std::int64_t, std::int64_t j, std::int64_t, std::int64_t k,
                 const double* w) {
        if (virials == nullptr) return;
        for (int c = 0; c < 9; ++c) {
            virials[9 * i + c] += w[c];
            virials[9 * j + c] += w[c];
            virials[9 * k + c] += w[c];
        }
    }
};

// Energies (n), forces (n x 3) and virials (n x 9) gathered from the shares.
// heat: the virials of the heat flux instead, those of Fan et al., PRB 92,
// 094301 (2015): for each atom i, the sum over its pairs p = (i, j) of
// r_p (x) the derivative of the terms of j by its reverse, which needs the
// positions and the box matrix; the shares of the virials are not read, and
// the kernels leave the energy of each term whole with its centre.
// reverse: the reverse of each pair, as reverse_pairs gives it, or null to
// find them.
void gather(const Shares& shares, std::int64_t n, const std::int64_t* offsets,
            const index_t* neighbors, const std::int8_t* shifts, double* energies,
            double* forces, double* virials, const Control& control, bool heat = false,
            const double* positions = nullptr, const double* matrix = nullptr,
            const std::int64_t* reverse = nullptr);

}  // namespace mdapy::calculators
