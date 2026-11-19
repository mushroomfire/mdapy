// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/scaled_basis.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace mdapy::analysis {

namespace {

constexpr double kPi = 3.14159265358979323846;

// f_k(s), k < count, into f
void chebyshev(double s, double cutoff, int count, double* f) {
    const double fc = 0.5 * (1.0 + std::cos(kPi * s / cutoff));
    const double x = 2.0 * (s / cutoff - 1.0) * (s / cutoff - 1.0) - 1.0;
    // T_k(x) by T_{k+1} = 2 x T_k - T_{k-1}
    double previous = 1.0, current = x;
    for (int k = 0; k < count; ++k) {
        double t = 1.0;
        if (k == 1) {
            t = x;
        } else if (k >= 2) {
            t = 2.0 * x * current - previous;
            previous = current;
            current = t;
        }
        f[k] = 0.5 * (t + 1.0) * fc;
    }
}

}  // namespace

std::int64_t scaled_basis_dimension(const ScaledBasis& basis, int species) {
    const std::int64_t chemical = species > 1 ? static_cast<std::int64_t>(basis.chemical) * species : 0;
    return basis.radial + static_cast<std::int64_t>(basis.angular) * basis.l_max + chemical;
}

std::vector<double> scaled_basis(const ScaledBasis& basis, int species_count,
                                 const std::int32_t* species, const double* radii,
                                 std::int64_t n, const double* positions, const double* matrix,
                                 const std::int64_t* offsets, const index_t* others,
                                 const std::int8_t* shifts, const Control& control) {
    const std::int64_t dimension = scaled_basis_dimension(basis, species_count);
    std::vector<double> out(static_cast<std::size_t>(n * dimension), 0.0);
    const int count = std::max({basis.radial, basis.angular, species_count > 1 ? basis.chemical : 0});
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> f(static_cast<std::size_t>(count));
            std::vector<double> units, values;  // of the angular neighbors
            std::vector<double> p(static_cast<std::size_t>(basis.l_max + 1));
            for (std::int64_t i = begin; i < end; ++i) {
                double* row = &out[static_cast<std::size_t>(i * dimension)];
                double* chemical = row + basis.radial + basis.angular * basis.l_max;
                units.clear();
                values.clear();
                for (std::int64_t q = offsets[i]; q < offsets[i + 1]; ++q) {
                    const auto j = static_cast<std::int64_t>(others[q]);
                    double d[3];
                    for (int e = 0; e < 3; ++e) {
                        d[e] = positions[3 * j + e] - positions[3 * i + e];
                        for (int g = 0; g < 3; ++g) {
                            d[e] += shifts[3 * q + g] * matrix[3 * g + e];
                        }
                    }
                    const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    const double s = r / (radii[species[i]] + radii[species[j]]);
                    if (!(s < basis.cutoff) || r == 0.0) {
                        continue;
                    }
                    chebyshev(s, basis.cutoff, count, f.data());
                    for (int k = 0; k < basis.radial; ++k) {
                        row[k] += f[static_cast<std::size_t>(k)];
                    }
                    if (species_count > 1) {
                        for (int k = 0; k < basis.chemical; ++k) {
                            chemical[species[j] * basis.chemical + k] += f[static_cast<std::size_t>(k)];
                        }
                    }
                    if (s < basis.angular_cutoff) {
                        chebyshev(s, basis.angular_cutoff, basis.angular, f.data());
                        units.insert(units.end(), {d[0] / r, d[1] / r, d[2] / r});
                        values.insert(values.end(), f.begin(), f.begin() + basis.angular);
                    }
                }
                // the pairs of angular neighbors, each pair j < k twice and
                // each neighbor with itself once
                const auto m = static_cast<std::int64_t>(units.size() / 3);
                double* angular = row + basis.radial;
                for (std::int64_t a = 0; a < m; ++a) {
                    const double* ua = &units[static_cast<std::size_t>(3 * a)];
                    const double* fa = &values[static_cast<std::size_t>(a * basis.angular)];
                    for (std::int64_t b = a; b < m; ++b) {
                        const double* ub = &units[static_cast<std::size_t>(3 * b)];
                        const double* fb = &values[static_cast<std::size_t>(b * basis.angular)];
                        const double c = std::clamp(ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2], -1.0, 1.0);
                        p[0] = 1.0;
                        if (basis.l_max >= 1) {
                            p[1] = c;
                        }
                        for (int l = 1; l < basis.l_max; ++l) {
                            p[static_cast<std::size_t>(l + 1)] =
                                ((2.0 * l + 1.0) * c * p[static_cast<std::size_t>(l)] -
                                 l * p[static_cast<std::size_t>(l - 1)]) /
                                (l + 1.0);
                        }
                        const double weight = a == b ? 1.0 : 2.0;
                        for (int k = 0; k < basis.angular; ++k) {
                            const double product = weight * fa[k] * fb[k];
                            for (int l = 1; l <= basis.l_max; ++l) {
                                angular[k * basis.l_max + l - 1] += product * p[static_cast<std::size_t>(l)];
                            }
                        }
                    }
                }
            }
        },
        control);
    return out;
}

}  // namespace mdapy::analysis
