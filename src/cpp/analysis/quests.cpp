// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/quests.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <tuple>

namespace mdapy::analysis {

namespace {

// QUESTS adds these to the distances it divides by, so that atoms on top of
// each other give large finite values
constexpr double kNear = 1e-16;
constexpr double kCross = 1e-15;

}  // namespace

std::vector<double> quests(int k, double cutoff, std::int64_t n, const double* positions,
                           const double* matrix, const std::int64_t* offsets,
                           const index_t* others, const std::int8_t* shifts,
                           const Control& control) {
    const auto width = static_cast<std::size_t>(2 * k - 1);
    std::vector<double> out(static_cast<std::size_t>(n) * width, 0.0);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> d, r, w, row;
            std::vector<std::int64_t> order;
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t first = offsets[i];
                const std::int64_t count = offsets[i + 1] - first;
                d.resize(static_cast<std::size_t>(3 * count));
                r.resize(static_cast<std::size_t>(count));
                order.clear();
                for (std::int64_t q = 0; q < count; ++q) {
                    const std::int64_t p = first + q;
                    const auto j = static_cast<std::int64_t>(others[p]);
                    double* v = &d[static_cast<std::size_t>(3 * q)];
                    for (int e = 0; e < 3; ++e) {
                        v[e] = positions[3 * j + e] - positions[3 * i + e];
                        for (int f = 0; f < 3; ++f) {
                            v[e] += shifts[3 * p + f] * matrix[3 * f + e];
                        }
                    }
                    const double length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    r[static_cast<std::size_t>(q)] = length;
                    if (length < cutoff) {
                        order.push_back(q);
                    }
                }
                // nearest first, ties by atom and then image
                auto key = [&](std::int64_t q) {
                    const std::int64_t p = first + q;
                    return std::make_tuple(r[static_cast<std::size_t>(q)],
                                           static_cast<std::int64_t>(others[p]), shifts[3 * p],
                                           shifts[3 * p + 1], shifts[3 * p + 2]);
                };
                std::sort(order.begin(), order.end(),
                          [&](std::int64_t a, std::int64_t b) { return key(a) < key(b); });
                const auto m = static_cast<int>(std::min<std::size_t>(order.size(), static_cast<std::size_t>(k)));
                double* x = &out[static_cast<std::size_t>(i) * width];
                w.resize(static_cast<std::size_t>(m));
                for (int a = 0; a < m; ++a) {
                    const double ra = r[static_cast<std::size_t>(order[static_cast<std::size_t>(a)])];
                    const double z = ra / cutoff;
                    const double weight = (1.0 - z * z) * (1.0 - z * z);
                    w[static_cast<std::size_t>(a)] = weight;
                    x[a] = weight / (ra + kNear);
                }
                // the cross terms of each neighbor, largest first, summed
                // over the neighbors in their order
                for (int a = 0; a < m; ++a) {
                    row.clear();
                    const double* da = &d[static_cast<std::size_t>(3 * order[static_cast<std::size_t>(a)])];
                    for (int b = 0; b < m; ++b) {
                        if (b == a) {
                            continue;
                        }
                        const double* db = &d[static_cast<std::size_t>(3 * order[static_cast<std::size_t>(b)])];
                        const double dx = da[0] - db[0], dy = da[1] - db[1], dz = da[2] - db[2];
                        const double rab = std::sqrt(dx * dx + dy * dy + dz * dz);
                        row.push_back(std::sqrt(w[static_cast<std::size_t>(a)] * w[static_cast<std::size_t>(b)]) /
                                      (rab + kCross));
                    }
                    std::sort(row.begin(), row.end(), std::greater<>());
                    for (std::size_t s = 0; s < row.size(); ++s) {
                        x[static_cast<std::size_t>(k) + s] += row[s];
                    }
                }
                for (int s = 0; s < k - 1; ++s) {
                    x[k + s] /= k;
                }
            }
        },
        control);
    return out;
}

namespace {

// The squared distance of a and b, or a value of at least limit as soon as
// the partial sum reaches it: the sum only grows, so the rest cannot bring
// it back below.
double squared_distance(const double* a, const double* b, std::int64_t dimension, double limit) {
    double s = 0.0;
    std::int64_t e = 0;
    while (e < dimension) {
        const std::int64_t stop = std::min<std::int64_t>(dimension, e + 8);
        for (; e < stop; ++e) {
            const double t = a[e] - b[e];
            s += t * t;
        }
        if (s >= limit) {
            return s;
        }
    }
    return s;
}

}  // namespace

std::vector<double> log_gaussian_sums(std::int64_t m, const double* x, std::int64_t count,
                                      const double* y, std::int64_t dimension, double bandwidth,
                                      const Control& control) {
    std::vector<double> out(static_cast<std::size_t>(m), 0.0);
    const double scale = 2.0 * bandwidth * bandwidth;
    parallel_for(
        m, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const double* xi = x + i * dimension;
                // the nearest point first: every term is then at most one
                double best = std::numeric_limits<double>::infinity();
                for (std::int64_t j = 0; j < count; ++j) {
                    best = std::min(best, squared_distance(xi, y + j * dimension, dimension, best));
                }
                // terms whose exponent is below -750 are exactly zero in
                // double precision, so the points past it are left out
                // without changing the sum
                const double limit = best + 750.0 * scale;
                double sum = 0.0;
                for (std::int64_t j = 0; j < count; ++j) {
                    const double s = squared_distance(xi, y + j * dimension, dimension, limit);
                    if (s < limit) {
                        sum += std::exp(-(s - best) / scale);
                    }
                }
                out[static_cast<std::size_t>(i)] = -best / scale + std::log(sum);
            }
        },
        control);
    return out;
}

}  // namespace mdapy::analysis
