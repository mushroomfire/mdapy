// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/soap.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "neighbor/frames.h"

namespace mdapy::analysis {

namespace {

constexpr double kPi = 3.14159265358979323846;

// The real spherical harmonics of degree up to l_max of a unit vector, l^2
// + l + m for each l and m: the cosines of m phi at m >= 0, the sines at
// m < 0. Their signs do not matter, as the power spectrum multiplies two of
// one l and m.
void harmonics(int l_max, double x, double y, double z, std::vector<double>& out,
               std::vector<double>& legendre) {
    const int size = l_max + 1;
    out.assign(static_cast<std::size_t>(size * size), 0.0);
    // the associated Legendre functions of cos theta = z, normalized, over
    // sin^m theta, which (x + iy)^m carries
    legendre.assign(static_cast<std::size_t>(size * size), 0.0);
    auto p = [&](int l, int m) -> double& {
        return legendre[static_cast<std::size_t>(l * size + m)];
    };
    p(0, 0) = 1.0 / std::sqrt(4 * kPi);
    for (int m = 1; m <= l_max; ++m) {
        p(m, m) = -std::sqrt((2.0 * m + 1.0) / (2.0 * m)) * p(m - 1, m - 1);
    }
    for (int m = 0; m < l_max; ++m) {
        p(m + 1, m) = std::sqrt(2.0 * m + 3.0) * z * p(m, m);
    }
    for (int m = 0; m <= l_max; ++m) {
        for (int l = m + 2; l <= l_max; ++l) {
            const double ll = static_cast<double>(l * l), mm = static_cast<double>(m * m);
            const double a = std::sqrt((4.0 * ll - 1.0) / (ll - mm));
            const double b = std::sqrt(((l - 1.0) * (l - 1.0) - mm) / (4.0 * (l - 1.0) * (l - 1.0) - 1.0));
            p(l, m) = a * (z * p(l - 1, m) - b * p(l - 2, m));
        }
    }
    double re = 1.0, im = 0.0;  // (x + iy)^m
    for (int m = 0; m <= l_max; ++m) {
        for (int l = m; l <= l_max; ++l) {
            const auto at = static_cast<std::size_t>(l * l + l);
            if (m == 0) {
                out[at] = p(l, 0);
            } else {
                out[at + static_cast<std::size_t>(m)] = std::sqrt(2.0) * p(l, m) * re;
                out[at - static_cast<std::size_t>(m)] = std::sqrt(2.0) * p(l, m) * im;
            }
        }
        const double next = re * x - im * y;
        im = re * y + im * x;
        re = next;
    }
}

}  // namespace

std::int64_t soap_dimension(const SoapBasis& basis, int species) {
    const std::int64_t n = basis.n_max, l = basis.l_max + 1, s = species;
    if (basis.crossover) {
        return (s * n) * (s * n + 1) / 2 * l;
    }
    return s * l * n * (n + 1) / 2;
}

std::vector<double> soap(const SoapBasis& basis, int species_count, const std::int32_t* species,
                         std::int64_t n, const double* positions, const double* matrix,
                         const std::int64_t* offsets, const index_t* others,
                         const std::int8_t* shifts, const Control& control) {
    const int n_max = basis.n_max, l_max = basis.l_max;
    const int lm = (l_max + 1) * (l_max + 1);
    const std::int64_t dimension = soap_dimension(basis, species_count);
    std::vector<double> out(static_cast<std::size_t>(n * dimension), 0.0);
    // the factor of each Gaussian radial function, pi^(3/2) eta^l /
    // (alpha + eta)^(l + 3/2), and its exponent alpha eta / (alpha + eta)
    std::vector<double> scale(static_cast<std::size_t>((l_max + 1) * n_max));
    std::vector<double> decay(scale.size());
    for (int l = 0; l <= l_max; ++l) {
        for (int k = 0; k < n_max; ++k) {
            const double alpha = basis.alphas[static_cast<std::size_t>(l * n_max + k)];
            const auto at = static_cast<std::size_t>(l * n_max + k);
            scale[at] = std::pow(kPi, 1.5) * std::pow(basis.eta, l) /
                        std::pow(alpha + basis.eta, l + 1.5);
            decay[at] = alpha * basis.eta / (alpha + basis.eta);
        }
    }
    // the power spectrum carries pi sqrt(8 / (2l + 1)), as in DScribe
    std::vector<double> prefactor(static_cast<std::size_t>(l_max + 1));
    for (int l = 0; l <= l_max; ++l) {
        prefactor[static_cast<std::size_t>(l)] = kPi * std::sqrt(8.0 / (2.0 * l + 1.0));
    }
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> c(static_cast<std::size_t>(species_count * n_max * lm));
            std::vector<double> y, legendre, radial(static_cast<std::size_t>(n_max));
            std::vector<double> gaussian(static_cast<std::size_t>(n_max));
            for (std::int64_t i = begin; i < end; ++i) {
                std::fill(c.begin(), c.end(), 0.0);
                // the atom itself, at the center: only l = 0
                auto add = [&](int s, double r2, double ux, double uy, double uz, double r) {
                    harmonics(l_max, ux, uy, uz, y, legendre);
                    double rl = 1.0;
                    for (int l = 0; l <= l_max; ++l) {
                        if (r == 0.0 && l > 0) {
                            break;
                        }
                        for (int k = 0; k < n_max; ++k) {
                            const auto at = static_cast<std::size_t>(l * n_max + k);
                            gaussian[static_cast<std::size_t>(k)] = scale[at] * std::exp(-decay[at] * r2);
                        }
                        for (int nn = 0; nn < n_max; ++nn) {
                            const double* beta =
                                &basis.betas[static_cast<std::size_t>((l * n_max + nn) * n_max)];
                            double sum = 0.0;
                            for (int k = 0; k < n_max; ++k) {
                                sum += beta[k] * gaussian[static_cast<std::size_t>(k)];
                            }
                            radial[static_cast<std::size_t>(nn)] = sum * rl;
                        }
                        for (int nn = 0; nn < n_max; ++nn) {
                            double* row = &c[static_cast<std::size_t>((s * n_max + nn) * lm)];
                            for (int m = l * l; m < (l + 1) * (l + 1); ++m) {
                                row[m] += radial[static_cast<std::size_t>(nn)] * y[static_cast<std::size_t>(m)];
                            }
                        }
                        rl *= r;
                    }
                };
                add(species[i], 0.0, 0.0, 0.0, 1.0, 0.0);
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const auto j = static_cast<std::int64_t>(others[p]);
                    double d[3];
                    for (int e = 0; e < 3; ++e) {
                        d[e] = positions[3 * j + e] - positions[3 * i + e];
                        for (int f = 0; f < 3; ++f) {
                            d[e] += shifts[3 * p + f] * matrix[3 * f + e];
                        }
                    }
                    const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    const double r = std::sqrt(r2);
                    if (r == 0.0) {
                        continue;
                    }
                    add(species[j], r2, d[0] / r, d[1] / r, d[2] / r, r);
                }
                // the power spectrum: species pairs, then l, then radial pairs
                double* row = &out[static_cast<std::size_t>(i * dimension)];
                std::int64_t at = 0;
                for (int s1 = 0; s1 < species_count; ++s1) {
                    const int last = basis.crossover ? species_count : s1 + 1;
                    for (int s2 = s1; s2 < last; ++s2) {
                        for (int l = 0; l <= l_max; ++l) {
                            for (int n1 = 0; n1 < n_max; ++n1) {
                                for (int n2 = s1 == s2 ? n1 : 0; n2 < n_max; ++n2) {
                                    const double* a = &c[static_cast<std::size_t>((s1 * n_max + n1) * lm)];
                                    const double* b = &c[static_cast<std::size_t>((s2 * n_max + n2) * lm)];
                                    double sum = 0.0;
                                    for (int m = l * l; m < (l + 1) * (l + 1); ++m) {
                                        sum += a[m] * b[m];
                                    }
                                    row[at++] = prefactor[static_cast<std::size_t>(l)] * sum;
                                }
                            }
                        }
                    }
                }
            }
        },
        control);
    return out;
}

std::vector<double> soap_frames(const SoapBasis& basis, int species_count, const std::int32_t* species,
                                const neighbor::Frames& frames, double cutoff,
                                const Control& control) {
    const auto dim = static_cast<std::size_t>(soap_dimension(basis, species_count));
    std::vector<double> found(dim * static_cast<std::size_t>(frames.count), 0.0);
    neighbor::for_each_frame(
        frames,
        [&](std::int64_t k) {
            const std::int64_t n = frames.atoms(k), s = frames.starts[k];
            double* mean = found.data() + dim * static_cast<std::size_t>(k);
            if (n == 0) {
                std::fill(mean, mean + dim, std::nan(""));
                return;
            }
            // the pairs closer than the cutoff, as the descriptors of one system keep them
            const neighbor::NeighborPairs pairs = frames.pairs(k, cutoff);
            std::vector<std::int64_t> offsets(static_cast<std::size_t>(n) + 1, 0);
            std::vector<index_t> others;
            std::vector<std::int8_t> shifts;
            for (std::int64_t i = 0; i < n; ++i) {
                for (std::int64_t p = pairs.offsets[i]; p < pairs.offsets[i + 1]; ++p) {
                    if (!(pairs.distances[p] < cutoff)) {
                        continue;
                    }
                    others.push_back(pairs.neighbors[p]);
                    shifts.insert(shifts.end(), &pairs.shifts[3 * p], &pairs.shifts[3 * p] + 3);
                }
                offsets[static_cast<std::size_t>(i) + 1] = static_cast<std::int64_t>(others.size());
            }
            const std::vector<double> each =
                soap(basis, species_count, species + s, n, frames.positions + 3 * s, frames.matrices + 9 * k,
                     offsets.data(), others.data(), shifts.data());
            for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
                for (std::size_t c = 0; c < dim; ++c) {
                    mean[c] += each[dim * i + c];
                }
            }
            for (std::size_t c = 0; c < dim; ++c) {
                mean[c] /= static_cast<double>(n);
            }
        },
        control);
    return found;
}

}  // namespace mdapy::analysis
