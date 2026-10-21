// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "neighbor/cell_list.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

namespace mdapy::neighbor {
namespace {

using Vector = std::array<double, 3>;

// Bins are searched with a cutoff larger by this factor, so that rounding in
// the fractional coordinates cannot hide a pair just inside the cutoff.
constexpr double kMargin = 1.0 + 1e-10;

std::int64_t floor_div(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}

// The grid of cells along one lattice direction.
struct Axis {
    bool periodic = true;
    double low = 0.0;          // fractional coordinate where cell 0 starts
    double width = 1.0;        // fractional width of a cell
    std::int64_t cells = 1;
    std::int64_t reach = 1;    // cells to search on each side
};

struct Pair {
    std::int32_t j;
    std::array<std::int32_t, 3> shift;
    double distance;
};

}  // namespace

NeighborPairs build(const double* positions, std::int64_t n, const double* matrix,
                    const double* origin, const bool* pbc, double cutoff,
                    const Control& control) {
    if (!(cutoff > 0.0) || !std::isfinite(cutoff)) {
        throw std::invalid_argument("cutoff must be a positive number");
    }
    if (n > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("too many atoms for 32-bit neighbor indices");
    }
    const double* h = matrix;
    // Inverse of H and the thickness of the cell along each lattice direction.
    const Vector a{h[0], h[1], h[2]}, b{h[3], h[4], h[5]}, c{h[6], h[7], h[8]};
    auto cross = [](const Vector& u, const Vector& v) {
        return Vector{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0]};
    };
    auto norm = [](const Vector& u) { return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); };
    const Vector bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
    const double det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
    if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
        throw std::invalid_argument("the box is singular");
    }
    // Columns of H^-1 are (b x c, c x a, a x b) / det.
    const Vector inverse_columns[3] = {bc, ca, ab};
    const double thickness[3] = {std::abs(det) / norm(bc), std::abs(det) / norm(ca),
                                 std::abs(det) / norm(ab)};

    // Fractional coordinates, wrapped into [0, 1) along periodic directions.
    std::vector<double> fractional(static_cast<std::size_t>(n) * 3);
    std::vector<std::int32_t> wraps(static_cast<std::size_t>(n) * 3, 0);
    for (std::int64_t i = 0; i < n; ++i) {
        const double x[3] = {positions[3 * i] - origin[0], positions[3 * i + 1] - origin[1],
                             positions[3 * i + 2] - origin[2]};
        for (int d = 0; d < 3; ++d) {
            const Vector& column = inverse_columns[d];
            double f = (x[0] * column[0] + x[1] * column[1] + x[2] * column[2]) / det;
            if (pbc[d]) {
                const double k = std::floor(f);
                f -= k;
                wraps[3 * i + d] = static_cast<std::int32_t>(k);
            }
            fractional[3 * i + d] = f;
        }
    }

    // Cells at least as wide as the cutoff where possible; along periodic
    // directions narrower boxes are covered by searching more cells.
    const double search = cutoff * kMargin;
    Axis axes[3];
    for (int d = 0; d < 3; ++d) {
        Axis& axis = axes[d];
        axis.periodic = pbc[d];
        double span = 1.0;
        if (!axis.periodic) {
            double low = 0.0, high = 0.0;
            if (n > 0) {
                low = high = fractional[d];
                for (std::int64_t i = 1; i < n; ++i) {
                    low = std::min(low, fractional[3 * i + d]);
                    high = std::max(high, fractional[3 * i + d]);
                }
            }
            axis.low = low;
            span = std::max(high - low, 1e-12);
        }
        const double length = span * thickness[d];  // Å across the atoms
        axis.cells = std::max<std::int64_t>(1, static_cast<std::int64_t>(length / search));
        axis.width = span / static_cast<double>(axis.cells);
        axis.reach = static_cast<std::int64_t>(std::ceil(search / (axis.width * thickness[d])));
        if (!axis.periodic) {
            axis.reach = std::min(axis.reach, axis.cells - 1);
        }
    }
    // Keep the number of cells proportional to the number of atoms.
    while (axes[0].cells * axes[1].cells * axes[2].cells > 8 * n + 64) {
        Axis& widest = *std::max_element(std::begin(axes), std::end(axes),
                                         [](const Axis& u, const Axis& v) { return u.cells < v.cells; });
        const double span = widest.width * static_cast<double>(widest.cells);
        widest.cells = std::max<std::int64_t>(1, widest.cells / 2);
        widest.width = span / static_cast<double>(widest.cells);
        const int d = static_cast<int>(&widest - axes);
        widest.reach = static_cast<std::int64_t>(std::ceil(search / (widest.width * thickness[d])));
        if (!widest.periodic) {
            widest.reach = std::min(widest.reach, widest.cells - 1);
        }
    }

    // Counting sort of the atoms into cells, in increasing index order.
    std::vector<std::int64_t> cell_of(static_cast<std::size_t>(n));
    std::vector<std::int64_t> cell_index(static_cast<std::size_t>(n) * 3);
    const std::int64_t total_cells = axes[0].cells * axes[1].cells * axes[2].cells;
    std::vector<std::int64_t> cell_start(static_cast<std::size_t>(total_cells) + 1, 0);
    for (std::int64_t i = 0; i < n; ++i) {
        std::int64_t index = 0;
        for (int d = 0; d < 3; ++d) {
            const Axis& axis = axes[d];
            auto k = static_cast<std::int64_t>((fractional[3 * i + d] - axis.low) / axis.width);
            k = std::clamp<std::int64_t>(k, 0, axis.cells - 1);
            cell_index[3 * i + d] = k;
            index = index * axis.cells + k;
        }
        cell_of[i] = index;
        ++cell_start[index + 1];
    }
    for (std::int64_t k = 0; k < total_cells; ++k) {
        cell_start[k + 1] += cell_start[k];
    }
    std::vector<std::int32_t> cell_atoms(static_cast<std::size_t>(n));
    {
        std::vector<std::int64_t> fill(cell_start.begin(), cell_start.end() - 1);
        for (std::int64_t i = 0; i < n; ++i) {
            cell_atoms[fill[cell_of[i]]++] = static_cast<std::int32_t>(i);
        }
    }

    const double cutoff2 = cutoff * cutoff;
    const std::int64_t grain = default_grain(n);
    const std::int64_t nchunks = n > 0 ? (n + grain - 1) / grain : 0;
    std::vector<std::vector<Pair>> chunk_pairs(static_cast<std::size_t>(nchunks));
    std::vector<std::int64_t> counts(static_cast<std::size_t>(n), 0);
    parallel_for(
        n, grain,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<Pair>& out = chunk_pairs[static_cast<std::size_t>(begin / grain)];
            std::vector<Pair> found;
            for (std::int64_t i = begin; i < end; ++i) {
                found.clear();
                const double* xi = positions + 3 * i;
                for (std::int64_t dx = -axes[0].reach; dx <= axes[0].reach; ++dx) {
                    for (std::int64_t dy = -axes[1].reach; dy <= axes[1].reach; ++dy) {
                        for (std::int64_t dz = -axes[2].reach; dz <= axes[2].reach; ++dz) {
                            const std::int64_t step[3] = {dx, dy, dz};
                            std::int64_t target = 0;
                            std::int32_t image[3] = {0, 0, 0};
                            bool inside = true;
                            for (int d = 0; d < 3 && inside; ++d) {
                                const Axis& axis = axes[d];
                                std::int64_t k = cell_index[3 * i + d] + step[d];
                                if (axis.periodic) {
                                    const std::int64_t wrap = floor_div(k, axis.cells);
                                    k -= wrap * axis.cells;
                                    image[d] = static_cast<std::int32_t>(wrap);
                                } else if (k < 0 || k >= axis.cells) {
                                    inside = false;
                                }
                                target = target * axis.cells + k;
                            }
                            if (!inside) {
                                continue;
                            }
                            for (std::int64_t s = cell_start[target]; s < cell_start[target + 1];
                                 ++s) {
                                const std::int32_t j = cell_atoms[s];
                                // shift in the original (unwrapped) coordinates
                                std::array<std::int32_t, 3> shift{
                                    image[0] + wraps[3 * i] - wraps[3 * j],
                                    image[1] + wraps[3 * i + 1] - wraps[3 * j + 1],
                                    image[2] + wraps[3 * i + 2] - wraps[3 * j + 2]};
                                if (j == i && shift[0] == 0 && shift[1] == 0 && shift[2] == 0) {
                                    continue;
                                }
                                const double* xj = positions + 3 * j;
                                double r2 = 0.0;
                                for (int e = 0; e < 3; ++e) {
                                    const double r = xj[e] - xi[e] + shift[0] * h[e] +
                                                     shift[1] * h[3 + e] + shift[2] * h[6 + e];
                                    r2 += r * r;
                                }
                                if (r2 < cutoff2) {
                                    found.push_back({j, shift, std::sqrt(r2)});
                                }
                            }
                        }
                    }
                }
                std::sort(found.begin(), found.end(), [](const Pair& p, const Pair& q) {
                    return std::tie(p.j, p.shift) < std::tie(q.j, q.shift);
                });
                counts[i] = static_cast<std::int64_t>(found.size());
                out.insert(out.end(), found.begin(), found.end());
            }
        },
        control);

    NeighborPairs result;
    result.offsets.resize(static_cast<std::size_t>(n) + 1, 0);
    for (std::int64_t i = 0; i < n; ++i) {
        result.offsets[i + 1] = result.offsets[i] + counts[i];
    }
    const auto total = static_cast<std::size_t>(result.offsets[n]);
    result.neighbors.reserve(total);
    result.shifts.reserve(3 * total);
    result.distances.reserve(total);
    for (const std::vector<Pair>& pairs : chunk_pairs) {
        for (const Pair& p : pairs) {
            result.neighbors.push_back(p.j);
            result.shifts.insert(result.shifts.end(), p.shift.begin(), p.shift.end());
            result.distances.push_back(p.distance);
        }
    }
    return result;
}

void order_by_distance(std::int64_t n, const std::int64_t* offsets, const double* distances,
                       std::int64_t count, std::vector<std::int64_t>& order,
                       std::vector<std::int64_t>& kept, const Control& control) {
    kept.assign(static_cast<std::size_t>(n) + 1, 0);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t m = offsets[i + 1] - offsets[i];
        kept[i + 1] = kept[i] + (count < 0 ? m : std::min(m, count));
    }
    order.resize(static_cast<std::size_t>(kept[n]));
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<std::int64_t> pairs;
            for (std::int64_t i = begin; i < end; ++i) {
                pairs.resize(static_cast<std::size_t>(offsets[i + 1] - offsets[i]));
                std::iota(pairs.begin(), pairs.end(), offsets[i]);
                const auto closer = [&](std::int64_t a, std::int64_t b) {
                    return distances[a] < distances[b] || (distances[a] == distances[b] && a < b);
                };
                const auto take = static_cast<std::ptrdiff_t>(kept[i + 1] - kept[i]);
                std::partial_sort(pairs.begin(), pairs.begin() + take, pairs.end(), closer);
                std::copy(pairs.begin(), pairs.begin() + take, order.begin() + kept[i]);
            }
        },
        control);
}

}  // namespace mdapy::neighbor
