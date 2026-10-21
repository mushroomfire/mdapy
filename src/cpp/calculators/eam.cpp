// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "calculators/eam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mdapy::calculators {

Spline::Spline(const double* values, int n, double delta)
    : n_(n), inverse_delta_(1.0 / delta), coefficients_(static_cast<std::size_t>(n) + 1) {
    if (n < 5) {
        throw std::invalid_argument("a tabulated function needs at least 5 values");
    }
    auto& s = coefficients_;
    for (int m = 1; m <= n; ++m) {
        s[m][6] = values[m - 1];
    }
    s[1][5] = s[2][6] - s[1][6];
    s[2][5] = 0.5 * (s[3][6] - s[1][6]);
    s[n - 1][5] = 0.5 * (s[n][6] - s[n - 2][6]);
    s[n][5] = s[n][6] - s[n - 1][6];
    for (int m = 3; m <= n - 2; ++m) {
        s[m][5] = ((s[m - 2][6] - s[m + 2][6]) + 8.0 * (s[m + 1][6] - s[m - 1][6])) / 12.0;
    }
    for (int m = 1; m <= n - 1; ++m) {
        s[m][4] = 3.0 * (s[m + 1][6] - s[m][6]) - 2.0 * s[m][5] - s[m + 1][5];
        s[m][3] = s[m][5] + s[m + 1][5] - 2.0 * (s[m + 1][6] - s[m][6]);
    }
    s[n][4] = 0.0;
    s[n][3] = 0.0;
    for (int m = 1; m <= n; ++m) {
        s[m][2] = s[m][5] / delta;
        s[m][1] = 2.0 * s[m][4] / delta;
        s[m][0] = 3.0 * s[m][3] / delta;
    }
}

void Spline::evaluate(double x, double& value, double& derivative) const {
    double p = x * inverse_delta_ + 1.0;
    int m = static_cast<int>(p);
    m = std::max(1, std::min(m, n_ - 1));
    p -= m;
    p = std::min(p, 1.0);
    const auto& c = coefficients_[static_cast<std::size_t>(m)];
    value = ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
    derivative = (c[0] * p + c[1]) * p + c[2];
}

void compute_eam(const EamPotential& potential, std::int64_t n, const std::int32_t* species,
                 const std::int64_t* offsets, const std::int32_t* neighbors,
                 const double* vectors, double* energies, double* forces, double* virials,
                 const Control& control) {
    const double cutoff2 = potential.cutoff * potential.cutoff;
    std::vector<double> slope(static_cast<std::size_t>(n), 0.0);  // F'(rho)

    // Electron density at each atom, from the density function of each neighbor.
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                double sum = 0.0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const double* r = vectors + 3 * p;
                    const double r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                    if (r2 < cutoff2) {
                        double value, derivative;
                        potential.density[species[neighbors[p]]].evaluate(std::sqrt(r2), value,
                                                                          derivative);
                        sum += value;
                    }
                }
                double embedding;
                potential.embedding[species[i]].evaluate(sum, embedding, slope[i]);
                if (sum > potential.rho_max) {
                    embedding += slope[i] * (sum - potential.rho_max);
                }
                energies[i] = embedding;
            }
        },
        control);

    // Forces, pair energies and virials; each atom gathers from its own pairs.
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const int si = species[i];
                double f[3] = {0.0, 0.0, 0.0};
                double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                double pair_energy = 0.0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const double* v = vectors + 3 * p;
                    // del = x_i - x_j, as in LAMMPS
                    const double del[3] = {-v[0], -v[1], -v[2]};
                    const double r2 = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
                    if (!(r2 < cutoff2)) {
                        continue;
                    }
                    const std::int32_t j = neighbors[p];
                    const int sj = species[j];
                    const double r = std::sqrt(r2);
                    double rho_i, rho_ip, rho_j, rho_jp, z2, z2p;
                    potential.density[si].evaluate(r, rho_i, rho_ip);
                    potential.density[sj].evaluate(r, rho_j, rho_jp);
                    potential.pair_of(si, sj).evaluate(r, z2, z2p);
                    const double recip = 1.0 / r;
                    const double phi = z2 * recip;
                    const double phip = z2p * recip - phi * recip;
                    const double psip = slope[i] * rho_jp + slope[j] * rho_ip + phip;
                    const double fpair = -psip * recip;
                    for (int a = 0; a < 3; ++a) {
                        f[a] += del[a] * fpair;
                        for (int b = 0; b < 3; ++b) {
                            w[3 * a + b] += 0.5 * del[a] * del[b] * fpair;
                        }
                    }
                    pair_energy += 0.5 * phi;
                }
                energies[i] += pair_energy;
                for (int a = 0; a < 3; ++a) {
                    forces[3 * i + a] = f[a];
                }
                if (virials != nullptr) {
                    for (int k = 0; k < 9; ++k) {
                        virials[9 * i + k] = w[k];
                    }
                }
            }
        },
        control);
}

}  // namespace mdapy::calculators
