// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/elastic_strain.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "analysis/dxa_clusters.h"
#include "analysis/dxa_math.h"
#include "analysis/dxa_structure.h"

namespace mdapy::analysis {

ElasticStrain elastic_strain(const double* positions, std::int64_t n, const double* matrix,
                             const double* inverse, const double* origin, const bool* pbc,
                             const ElasticStrainOptions& options, const Control& control) {
    using dxa::Mat;
    using dxa::Vec;
    dxa::Structures found = dxa::find_structures(positions, n, matrix, origin, pbc,
                                                 options.crystal, true, stage(control, 0, 2));
    ElasticStrain out;
    out.structure = std::move(found.structure);
    out.strain.assign(9 * static_cast<std::size_t>(n), 0.0);
    const dxa::Lattice* lattice = dxa::lattice(options.crystal);
    // the ideal vectors in Å in the frame of the template, and the inverse
    // of the sum of their outer products, the same for every atom
    std::array<Vec, 16> ideal{};
    Mat sum{};
    for (int k = 0; k < lattice->neighbors; ++k) {
        const Vec& t = lattice->vectors[static_cast<std::size_t>(k)];
        const Vec v = {t[0] * options.a, t[1] * options.a, t[2] * options.a * options.axial};
        ideal[static_cast<std::size_t>(k)] = v;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                sum[3 * i + j] += v[i] * v[j];
            }
        }
    }
    const Mat spread = *dxa::inverse(sum);
    const Mat to_fractional = dxa::transpose(Mat{inverse[0], inverse[1], inverse[2], inverse[3],
                                                 inverse[4], inverse[5], inverse[6],
                                                 inverse[7], inverse[8]});
    const Mat to_cartesian = dxa::transpose(Mat{matrix[0], matrix[1], matrix[2], matrix[3],
                                                matrix[4], matrix[5], matrix[6], matrix[7],
                                                matrix[8]});
    // a neighbor that is the atom itself or another twice, through periodic
    // images, whose vectors the nearest images would not give
    std::atomic<bool> met{false};
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t a = begin; a < end; ++a) {
                if (out.structure[static_cast<std::size_t>(a)] != options.crystal) {
                    continue;
                }
                const index_t* around = &found.neighbors[16 * static_cast<std::size_t>(a)];
                bool twice = false;
                for (int k = 0; k < lattice->neighbors && !twice; ++k) {
                    twice = around[k] == static_cast<index_t>(a);
                    for (int j = 0; j < k && !twice; ++j) {
                        twice = around[j] == around[k];
                    }
                }
                if (twice) {
                    met.store(true, std::memory_order_relaxed);
                    continue;
                }
                const Vec at = {positions[3 * a], positions[3 * a + 1], positions[3 * a + 2]};
                Mat w{};
                for (int k = 0; k < lattice->neighbors; ++k) {
                    const auto b = static_cast<std::int64_t>(around[k]);
                    Vec f = dxa::turn(to_fractional,
                                      dxa::minus({positions[3 * b], positions[3 * b + 1],
                                                  positions[3 * b + 2]},
                                                 at));
                    for (std::size_t e = 0; e < 3; ++e) {
                        if (pbc[e]) {
                            f[e] -= std::round(f[e]);
                        }
                    }
                    const Vec d = dxa::turn(to_cartesian, f);
                    const Vec& v = ideal[static_cast<std::size_t>(k)];
                    for (std::size_t i = 0; i < 3; ++i) {
                        for (std::size_t j = 0; j < 3; ++j) {
                            w[3 * i + j] += d[i] * v[j];
                        }
                    }
                }
                double* strain = &out.strain[9 * static_cast<std::size_t>(a)];
                const auto back = dxa::inverse(dxa::multiply(w, spread));
                if (!back) {
                    std::fill(strain, strain + 9, std::numeric_limits<double>::quiet_NaN());
                    continue;
                }
                const Mat& g = *back;
                for (std::size_t i = 0; i < 3; ++i) {
                    for (std::size_t j = 0; j < 3; ++j) {
                        const double gg = g[i] * g[j] + g[3 + i] * g[3 + j] + g[6 + i] * g[6 + j];
                        strain[3 * i + j] = ((i == j ? 1.0 : 0.0) - gg) / 2;
                    }
                }
            }
        },
        stage(control, 1, 2));
    if (met.load()) {
        throw std::invalid_argument("an atom meets its own periodic image");
    }
    return out;
}

}  // namespace mdapy::analysis
