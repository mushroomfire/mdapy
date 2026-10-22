// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/steinhardt.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace mdapy::analysis {
namespace {

constexpr double kPi = 3.14159265358979323846;

// What real_harmonics needs for each l and m: (2m - 1)!!, and the
// normalization, computed once as each call used to compute it.
struct HarmonicTables {
    double factorial[kMaxDegree + 1];
    double norm[kMaxDegree + 1][kMaxDegree + 1];

    HarmonicTables() {
        for (int m = 0; m <= kMaxDegree; ++m) {
            double current = 1.0;
            for (int k = 1; k <= m; ++k) {
                current *= 2 * k - 1;
            }
            factorial[m] = current;
        }
        for (int l = 0; l <= kMaxDegree; ++l) {
            double ratio = 1.0;  // (l - m)! / (l + m)!
            norm[l][0] = std::sqrt((2 * l + 1) / (4.0 * kPi));
            for (int m = 1; m <= l; ++m) {
                ratio /= static_cast<double>((l + m) * (l - m + 1));
                norm[l][m] = std::sqrt(2.0 * (2 * l + 1) / (4.0 * kPi) * ratio);
            }
        }
    }
};

const HarmonicTables& tables() {
    static const HarmonicTables kTables;
    return kTables;
}

// What real_harmonics computes for every degree up to lmax at once: q[l][m]
// = d^m P_l / dz^m and (x + iy)^m, each by the same operations as there.
struct Harmonics {
    double q[kMaxDegree + 1][kMaxDegree + 1];
    double re[kMaxDegree + 1];
    double im[kMaxDegree + 1];

    Harmonics(int lmax, const double* u) {
        const HarmonicTables& t = tables();
        for (int m = 0; m <= lmax; ++m) {
            double previous = 0.0;
            double current = t.factorial[m];
            q[m][m] = current;
            for (int degree = m + 1; degree <= lmax; ++degree) {
                const double next =
                    ((2 * degree - 1) * u[2] * current - (degree + m - 1) * previous) /
                    (degree - m);
                previous = current;
                current = next;
                q[degree][m] = current;
            }
        }
        re[0] = 1.0;
        im[0] = 0.0;
        for (int m = 1; m <= lmax; ++m) {
            re[m] = re[m - 1] * u[0] - im[m - 1] * u[1];
            im[m] = re[m - 1] * u[1] + im[m - 1] * u[0];
        }
    }

    // The 2 l + 1 harmonics of degree l, as real_harmonics gives them.
    void of(int l, double* y) const {
        const HarmonicTables& t = tables();
        y[0] = t.norm[l][0] * q[l][0];
        for (int m = 1; m <= l; ++m) {
            const double norm = t.norm[l][m] * q[l][m];
            y[2 * m - 1] = norm * re[m];
            y[2 * m] = norm * im[m];
        }
    }
};

}  // namespace

void real_harmonics(int l, const double* u, double* y) {
    const HarmonicTables& t = tables();
    // q[m] = d^m P_l / dz^m, from the recurrences in l at fixed m
    double q[kMaxDegree + 1] = {};
    for (int m = 0; m <= l; ++m) {
        double previous = 0.0;
        double current = t.factorial[m];  // (2m - 1)!!
        for (int degree = m + 1; degree <= l; ++degree) {
            const double next =
                ((2 * degree - 1) * u[2] * current - (degree + m - 1) * previous) / (degree - m);
            previous = current;
            current = next;
        }
        q[m] = current;
    }
    double re = 1.0, im = 0.0;  // (x + iy)^m
    y[0] = t.norm[l][0] * q[0];
    for (int m = 1; m <= l; ++m) {
        const double r = re * u[0] - im * u[1];
        im = re * u[1] + im * u[0];
        re = r;
        const double norm = t.norm[l][m] * q[m];
        y[2 * m - 1] = norm * re;
        y[2 * m] = norm * im;
    }
}

namespace {

// The harmonics of the bonds of atom i, averaged with their weights, into
// sum (width values); y is room for the harmonics of one bond.
void bond_harmonics(std::int64_t i, const std::int64_t* offsets, const double* vectors,
                    const double* weights, const std::vector<int>& degrees, int lmax,
                    std::size_t width, double* y, double* sum) {
    for (std::size_t k = 0; k < width; ++k) {
        sum[k] = 0.0;
    }
    double total = 0.0;  // of the weights
    for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
        const double weight = weights != nullptr ? weights[p] : 1.0;
        total += weight;
        const double* v = vectors + 3 * p;
        const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double u[3] = {v[0] / r, v[1] / r, v[2] / r};
        const Harmonics harmonics(lmax, u);
        std::size_t at = 0;
        for (int l : degrees) {
            harmonics.of(l, y + at);
            at += 2 * static_cast<std::size_t>(l) + 1;
        }
        for (std::size_t k = 0; k < width; ++k) {
            sum[k] += weight * y[k];
        }
    }
    if (total > 0.0) {
        for (std::size_t k = 0; k < width; ++k) {
            sum[k] /= total;
        }
    }
}

// q_l of each degree from the harmonics of an atom.
void invariants(const double* harmonics, const std::vector<int>& degrees, double* q) {
    std::size_t at = 0;
    for (std::size_t d = 0; d < degrees.size(); ++d) {
        const int l = degrees[d];
        double sum = 0.0;
        for (int k = 0; k < 2 * l + 1; ++k) {
            sum += harmonics[at + k] * harmonics[at + k];
        }
        at += 2 * static_cast<std::size_t>(l) + 1;
        q[d] = std::sqrt(4.0 * kPi / (2 * l + 1) * sum);
    }
}

}  // namespace

void steinhardt(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
                const double* vectors, const double* weights, const std::vector<int>& degrees,
                bool average, double* q, const Control& control) {
    std::size_t width = 0;
    int lmax = 0;
    for (int l : degrees) {
        if (l < 0 || l > kMaxDegree) {
            throw std::invalid_argument("degrees must be from 0 to 20");
        }
        width += 2 * static_cast<std::size_t>(l) + 1;
        lmax = std::max(lmax, l);
    }
    const std::size_t count = degrees.size();
    if (!average) {
        // each atom alone: no harmonics kept between the atoms
        parallel_for(
            n, 0,
            [&](std::int64_t begin, std::int64_t end) {
                std::vector<double> y(width), sum(width);
                for (std::int64_t i = begin; i < end; ++i) {
                    bond_harmonics(i, offsets, vectors, weights, degrees, lmax, width, y.data(),
                                   sum.data());
                    invariants(sum.data(), degrees, q + static_cast<std::size_t>(i) * count);
                }
            },
            control);
        return;
    }
    // every atom writes its own harmonics, so they need no zeroing first
    std::unique_ptr<double[]> qlm(new double[static_cast<std::size_t>(n) * width]);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> y(width);
            for (std::int64_t i = begin; i < end; ++i) {
                bond_harmonics(i, offsets, vectors, weights, degrees, lmax, width, y.data(),
                               &qlm[static_cast<std::size_t>(i) * width]);
            }
        },
        control);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> mean(width);
            for (std::int64_t i = begin; i < end; ++i) {
                const double* own = &qlm[static_cast<std::size_t>(i) * width];
                for (std::size_t k = 0; k < width; ++k) {
                    mean[k] = own[k];
                }
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const double* other = &qlm[static_cast<std::size_t>(neighbors[p]) * width];
                    for (std::size_t k = 0; k < width; ++k) {
                        mean[k] += other[k];
                    }
                }
                const auto members = static_cast<double>(offsets[i + 1] - offsets[i] + 1);
                for (std::size_t k = 0; k < width; ++k) {
                    mean[k] /= members;
                }
                invariants(mean.data(), degrees, q + static_cast<std::size_t>(i) * count);
            }
        },
        control);
}

}  // namespace mdapy::analysis
