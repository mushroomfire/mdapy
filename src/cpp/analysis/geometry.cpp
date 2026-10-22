// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "neighbor/cell_list.h"

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

namespace {

// The fingerprint of one atom: its g(r) on the points r_k = k sigma, each
// neighbor a Gaussian, and the integral.
class EntropyShape {
public:
    EntropyShape(double cutoff, double sigma)
        : cutoff_(cutoff), sigma_(sigma), bins_(static_cast<int>(cutoff / sigma) + 1),
          shell_(static_cast<std::size_t>(bins_), 0.0), decay_(std::exp(-1.0)) {
        for (int k = 1; k < bins_; ++k) {  // r = 0 has no shell
            shell_[k] = 1.0 / ((k * sigma) * (k * sigma));
        }
    }

    int bins() const { return bins_; }

    // The factor of each Gaussian at the number density.
    double scale(double density) const {
        // 4 pi rho r^2 g(r) = sum of normalized Gaussians
        return 1.0 / (4.0 * kPi * density * std::sqrt(2.0 * kPi) * sigma_);
    }

    // Adds a neighbor at distance r to g; farther than the cutoff, none.
    void add(double r, double scale, std::vector<double>& g) const {
        if (!(r < cutoff_)) {
            return;
        }
        // the Gaussian at r_k = k sigma, exp(-(u - k)^2 / 2) with u = r /
        // sigma, grows from k to k + 1 by exp(u - k - 1/2), a factor that
        // shrinks by 1/e at each step: two exponentials per pair instead of
        // one per bin
        const double u = r / sigma_;
        double gauss = std::exp(-0.5 * (u - 1.0) * (u - 1.0));
        double step = std::exp(u - 1.5);
        for (int k = 1; k < bins_; ++k) {
            g[k] += scale * shell_[k] * gauss;
            gauss *= step;
            step *= decay_;
        }
    }

    // -2 pi rho integral of (g ln g - g + 1) r^2 by the trapezoid rule.
    double entropy(const std::vector<double>& g, double density) const {
        double sum = 0.0;
        for (int k = 0; k < bins_; ++k) {
            const double rk = k * sigma_;
            const double term =
                g[k] < 1e-10 ? rk * rk : (g[k] * std::log(g[k]) - g[k] + 1.0) * rk * rk;
            sum += (k == 0 || k == bins_ - 1) ? 0.5 * term : term;
        }
        return -2.0 * kPi * density * sum * sigma_;
    }

private:
    double cutoff_, sigma_;
    int bins_;
    std::vector<double> shell_;  // 1 / r_k^2
    double decay_;
};

}  // namespace

void pair_entropy(std::int64_t n, const std::int64_t* offsets, const double* distances,
                  double cutoff, double sigma, const double* density, double* entropy,
                  const Control& control) {
    const EntropyShape shape(cutoff, sigma);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> g(static_cast<std::size_t>(shape.bins()));
            for (std::int64_t i = begin; i < end; ++i) {
                std::fill(g.begin(), g.end(), 0.0);
                const double scale = shape.scale(density[i]);
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    shape.add(distances[p], scale, g);
                }
                entropy[i] = shape.entropy(g, density[i]);
            }
        },
        control);
}

void pair_entropy_search(const double* positions, std::int64_t n, const double* matrix,
                         const double* origin, const bool* pbc, double cutoff, double sigma,
                         double density, double* entropy, const Control& control) {
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, cutoff);
    const EntropyShape shape(cutoff, sigma);
    const double sphere = 4.0 / 3.0 * kPi * cutoff * cutoff * cutoff;
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> g(static_cast<std::size_t>(shape.bins())), found;
            for (std::int64_t t = begin; t < end; ++t) {
                const std::int64_t i = grid.atom(t);  // in cell order
                found.clear();
                grid.visit(i, [&](std::int32_t, const std::int32_t*, double r2) {
                    found.push_back(std::sqrt(r2));
                });
                const double rho =
                    density > 0.0 ? density : static_cast<double>(found.size()) / sphere;
                std::fill(g.begin(), g.end(), 0.0);
                const double scale = shape.scale(rho);
                for (const double r : found) {
                    shape.add(r, scale, g);
                }
                entropy[i] = shape.entropy(g, rho);
            }
        },
        control);
}

}  // namespace mdapy::analysis
