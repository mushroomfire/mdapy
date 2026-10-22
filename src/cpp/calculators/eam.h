// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// The embedded atom method with tabulated functions, interpolated as LAMMPS
// pair_style eam does: a cubic spline through the tabulated values whose
// slopes come from fourth-order finite differences.

#pragma once

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
    // last interval continues up to its end, as in LAMMPS.
    void evaluate(double x, double& value, double& derivative) const;

private:
    int n_;
    double inverse_delta_;
    std::vector<std::array<double, 7>> coefficients_;  // index 1 to n
};

struct EamPotential {
    int elements = 0;
    double cutoff = 0.0;
    double rho_max = 0.0;  // the embedding energy is extrapolated linearly beyond
    std::vector<Spline> embedding;  // F(rho), one per element
    std::vector<Spline> density;    // rho(r), one per element
    std::vector<Spline> pair;       // r * phi(r), one per pair of elements a >= b

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
void compute_eam(const EamPotential& potential, std::int64_t n, const std::int32_t* species,
                 const double* positions, const double* matrix, const std::int64_t* offsets,
                 const index_t* neighbors, const std::int8_t* shifts, double* energies,
                 double* forces, double* virials, const Control& control = {});

}  // namespace mdapy::calculators
