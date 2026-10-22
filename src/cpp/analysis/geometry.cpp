// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace mdapy::analysis {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::int64_t root(std::vector<std::int64_t>& parent, std::int64_t a) {
    while (parent[a] != a) {
        parent[a] = parent[parent[a]];  // path halving
        a = parent[a];
    }
    return a;
}

}  // namespace

std::int64_t clusters(std::int64_t n, const std::int64_t* offsets, const std::int32_t* neighbors,
                      std::int64_t* cluster, std::int64_t* sizes) {
    std::vector<std::int64_t> parent(static_cast<std::size_t>(n));
    std::iota(parent.begin(), parent.end(), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
            std::int64_t a = root(parent, i), b = root(parent, neighbors[p]);
            if (a != b) {
                // the smaller index becomes the root: the root is the first atom
                if (b < a) {
                    std::swap(a, b);
                }
                parent[b] = a;
            }
        }
    }
    std::vector<std::int64_t> count(static_cast<std::size_t>(n), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        cluster[i] = root(parent, i);
        ++count[cluster[i]];
    }
    std::vector<std::int64_t> roots;
    for (std::int64_t i = 0; i < n; ++i) {
        if (count[i] > 0) {
            roots.push_back(i);  // in order of first atom
        }
    }
    std::stable_sort(roots.begin(), roots.end(),
                     [&](std::int64_t a, std::int64_t b) { return count[a] > count[b]; });
    std::vector<std::int64_t> label(static_cast<std::size_t>(n), 0);
    for (std::size_t k = 0; k < roots.size(); ++k) {
        label[roots[k]] = static_cast<std::int64_t>(k);
        sizes[k] = count[roots[k]];
    }
    for (std::int64_t i = 0; i < n; ++i) {
        cluster[i] = label[cluster[i]];
    }
    return static_cast<std::int64_t>(roots.size());
}

void bond_angles(std::int64_t n, const std::int64_t* offsets, const std::int32_t* neighbors,
                 const double* vectors, const std::int32_t* type, int types, int bins,
                 std::int64_t* counts, const Control& control) {
    using Histogram = std::vector<std::int64_t>;
    const std::int64_t pairs = static_cast<std::int64_t>(types) * (types + 1) / 2;
    const auto size = static_cast<std::size_t>(types * pairs * bins);
    // the pair (a, b), a <= b, comes after the a pairs that start below a
    std::vector<std::int64_t> first(static_cast<std::size_t>(types));
    for (int a = 0; a < types; ++a) {
        first[a] = static_cast<std::int64_t>(a) * types - static_cast<std::int64_t>(a) * (a - 1) / 2 - a;
    }
    // at most 64 histograms in flight, whatever the number of types
    const std::int64_t grain = std::max(default_grain(n), (n + 63) / 64);
    const Histogram total = parallel_reduce(
        n, grain, Histogram(size, 0),
        [&](std::int64_t begin, std::int64_t end) {
            Histogram local(size, 0);
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t center = type[i] * pairs;
                for (std::int64_t a = offsets[i]; a < offsets[i + 1]; ++a) {
                    const double* u = vectors + 3 * a;
                    const double ru = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                    const std::int32_t ta = type[neighbors[a]];
                    for (std::int64_t b = a + 1; b < offsets[i + 1]; ++b) {
                        const double* v = vectors + 3 * b;
                        const double rv = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                        double c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (ru * rv);
                        c = std::max(-1.0, std::min(1.0, c));
                        const double angle = std::acos(c) * 180.0 / kPi;
                        const int bin = std::min(static_cast<int>(angle / 180.0 * bins), bins - 1);
                        const std::int32_t tb = type[neighbors[b]];
                        const std::int32_t low = std::min(ta, tb), high = std::max(ta, tb);
                        ++local[static_cast<std::size_t>((center + first[low] + high) * bins + bin)];
                    }
                }
            }
            return local;
        },
        [](Histogram a, const Histogram& b) {
            for (std::size_t k = 0; k < a.size(); ++k) {
                a[k] += b[k];
            }
            return a;
        },
        control);
    std::copy(total.begin(), total.end(), counts);
}

void atomic_temperature(std::int64_t n, const std::int64_t* offsets,
                        const std::int32_t* neighbors, const double* mass,
                        const double* velocities, double energy_unit, double boltzmann,
                        double* temperature, const Control& control) {
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                // the sums over the neighbors first, then the atom's own terms
                std::int64_t members = 1;
                double masses = 0.0, momentum[3] = {0.0, 0.0, 0.0}, squares = 0.0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const std::int64_t j = neighbors[p];
                    if (j == i) {
                        continue;  // an image of the atom itself
                    }
                    const double* v = velocities + 3 * j;
                    ++members;
                    masses += mass[j];
                    for (int e = 0; e < 3; ++e) {
                        momentum[e] += mass[j] * v[e];
                    }
                    squares += mass[j] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                }
                if (members == 1) {
                    temperature[i] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                const double* v = velocities + 3 * i;
                const double total = mass[i] + masses;
                double p2 = 0.0;
                for (int e = 0; e < 3; ++e) {
                    const double pe = mass[i] * v[e] + momentum[e];
                    p2 += pe * pe;
                }
                const double twice =
                    (mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + squares) - p2 / total;
                temperature[i] =
                    twice * energy_unit / (3.0 * static_cast<double>(members - 1) * boltzmann);
            }
        },
        control);
}

void pair_entropy(std::int64_t n, const std::int64_t* offsets, const double* distances,
                  double cutoff, double sigma, const double* density, double* entropy,
                  const Control& control) {
    const int bins = static_cast<int>(cutoff / sigma) + 1;
    std::vector<double> shell(static_cast<std::size_t>(bins), 0.0);  // 1 / r_k^2
    for (int k = 1; k < bins; ++k) {  // r = 0 has no shell
        shell[k] = 1.0 / ((k * sigma) * (k * sigma));
    }
    const double decay = std::exp(-1.0);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> g(static_cast<std::size_t>(bins));
            for (std::int64_t i = begin; i < end; ++i) {
                std::fill(g.begin(), g.end(), 0.0);
                // 4 pi rho r^2 g(r) = sum of normalized Gaussians
                const double scale =
                    1.0 / (4.0 * kPi * density[i] * std::sqrt(2.0 * kPi) * sigma);
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const double r = distances[p];
                    if (!(r < cutoff)) {
                        continue;
                    }
                    // the Gaussian at r_k = k sigma, exp(-(u - k)^2 / 2) with
                    // u = r / sigma, grows from k to k + 1 by exp(u - k - 1/2),
                    // a factor that shrinks by 1/e at each step: two
                    // exponentials per pair instead of one per bin
                    const double u = r / sigma;
                    double gauss = std::exp(-0.5 * (u - 1.0) * (u - 1.0));
                    double step = std::exp(u - 1.5);
                    for (int k = 1; k < bins; ++k) {
                        g[k] += scale * shell[k] * gauss;
                        gauss *= step;
                        step *= decay;
                    }
                }
                // -2 pi rho integral of (g ln g - g + 1) r^2 by the trapezoid rule
                double sum = 0.0;
                for (int k = 0; k < bins; ++k) {
                    const double rk = k * sigma;
                    const double term =
                        g[k] < 1e-10 ? rk * rk : (g[k] * std::log(g[k]) - g[k] + 1.0) * rk * rk;
                    sum += (k == 0 || k == bins - 1) ? 0.5 * term : term;
                }
                entropy[i] = -2.0 * kPi * density[i] * sum * sigma;
            }
        },
        control);
}

}  // namespace mdapy::analysis
