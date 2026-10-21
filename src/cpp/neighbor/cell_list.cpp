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

struct Pair {
    std::int32_t j;
    std::array<std::int32_t, 3> shift;
    double distance;
};

}  // namespace

CellGrid::CellGrid(const double* positions, std::int64_t n, const double* matrix,
                   const double* origin, const bool* pbc, double cutoff) {
    if (!(cutoff > 0.0) || !std::isfinite(cutoff)) {
        throw std::invalid_argument("cutoff must be a positive number");
    }
    if (n > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("too many atoms for 32-bit neighbor indices");
    }
    std::copy(matrix, matrix + 9, h_);
    cutoff2_ = cutoff * cutoff;
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
    for (int d = 0; d < 3; ++d) {
        for (int e = 0; e < 3; ++e) {
            inverse_[3 * d + e] = inverse_columns[d][e] / det;
        }
        origin_[d] = origin[d];
    }
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
    for (int d = 0; d < 3; ++d) {
        Axis& axis = axes_[d];
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
    while (axes_[0].cells * axes_[1].cells * axes_[2].cells > 8 * n + 64) {
        Axis& widest = *std::max_element(std::begin(axes_), std::end(axes_),
                                         [](const Axis& u, const Axis& v) { return u.cells < v.cells; });
        const double span = widest.width * static_cast<double>(widest.cells);
        widest.cells = std::max<std::int64_t>(1, widest.cells / 2);
        widest.width = span / static_cast<double>(widest.cells);
        const int d = static_cast<int>(&widest - axes_);
        widest.reach = static_cast<std::int64_t>(std::ceil(search / (widest.width * thickness[d])));
        if (!widest.periodic) {
            widest.reach = std::min(widest.reach, widest.cells - 1);
        }
    }

    // Counting sort of the atoms into cells, in increasing index order.
    std::vector<std::int64_t> cell_of(static_cast<std::size_t>(n));
    std::vector<std::int64_t> cells(static_cast<std::size_t>(n) * 3);
    const std::int64_t total_cells = axes_[0].cells * axes_[1].cells * axes_[2].cells;
    cell_start_.assign(static_cast<std::size_t>(total_cells) + 1, 0);
    for (std::int64_t i = 0; i < n; ++i) {
        std::int64_t index = 0;
        for (int d = 0; d < 3; ++d) {
            const Axis& axis = axes_[d];
            auto k = static_cast<std::int64_t>((fractional[3 * i + d] - axis.low) / axis.width);
            k = std::clamp<std::int64_t>(k, 0, axis.cells - 1);
            cells[3 * i + d] = k;
            index = index * axis.cells + k;
        }
        cell_of[i] = index;
        ++cell_start_[index + 1];
    }
    for (std::int64_t k = 0; k < total_cells; ++k) {
        cell_start_[k + 1] += cell_start_[k];
    }
    slot_atoms_.resize(static_cast<std::size_t>(n));
    slot_positions_.resize(static_cast<std::size_t>(n) * 3);
    slot_wraps_.resize(static_cast<std::size_t>(n) * 3);
    slot_cells_.resize(static_cast<std::size_t>(n) * 3);
    slot_of_.resize(static_cast<std::size_t>(n));
    std::vector<std::int64_t> fill(cell_start_.begin(), cell_start_.end() - 1);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t t = fill[cell_of[i]]++;
        slot_atoms_[t] = static_cast<std::int32_t>(i);
        slot_of_[i] = static_cast<std::int32_t>(t);
        for (int d = 0; d < 3; ++d) {
            slot_positions_[3 * t + d] = positions[3 * i + d];
            slot_wraps_[3 * t + d] = wraps[3 * i + d];
            slot_cells_[3 * t + d] = cells[3 * i + d];
        }
    }
}

NeighborPairs build(const double* positions, std::int64_t n, const double* matrix,
                    const double* origin, const bool* pbc, double cutoff,
                    const Control& control) {
    const CellGrid grid(positions, n, matrix, origin, pbc, cutoff);
    // the atoms are searched in cell order; each chunk of slots keeps its
    // pairs, and each atom where its pairs are
    const std::int64_t grain = default_grain(n);
    const std::int64_t nchunks = n > 0 ? (n + grain - 1) / grain : 0;
    std::vector<std::vector<Pair>> chunk_pairs(static_cast<std::size_t>(nchunks));
    std::vector<std::int64_t> chunk(static_cast<std::size_t>(n), 0), start(static_cast<std::size_t>(n), 0);
    NeighborPairs result;
    result.offsets.resize(static_cast<std::size_t>(n) + 1, 0);
    parallel_for(
        n, grain,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<Pair>& out = chunk_pairs[static_cast<std::size_t>(begin / grain)];
            for (std::int64_t t = begin; t < end; ++t) {
                const std::int64_t i = grid.atom(t);
                const std::size_t first = out.size();
                grid.visit(i, [&](std::int32_t j, const std::int32_t* shift, double r2) {
                    out.push_back({j, {shift[0], shift[1], shift[2]}, std::sqrt(r2)});
                });
                std::sort(out.begin() + static_cast<std::ptrdiff_t>(first), out.end(),
                          [](const Pair& p, const Pair& q) {
                              return std::tie(p.j, p.shift) < std::tie(q.j, q.shift);
                          });
                chunk[i] = begin / grain;
                start[i] = static_cast<std::int64_t>(first);
                result.offsets[i + 1] = static_cast<std::int64_t>(out.size() - first);
            }
        },
        control);
    for (std::int64_t i = 0; i < n; ++i) {
        result.offsets[i + 1] += result.offsets[i];
    }
    const auto total = static_cast<std::size_t>(result.offsets[n]);
    result.neighbors.resize(total);
    result.shifts.resize(3 * total);
    result.distances.resize(total);
    parallel_for(
        n, grain,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const Pair* pairs = chunk_pairs[static_cast<std::size_t>(chunk[i])].data() + start[i];
                const std::int64_t first = result.offsets[i];
                for (std::int64_t p = 0; p < result.offsets[i + 1] - first; ++p) {
                    const auto at = static_cast<std::size_t>(first + p);
                    result.neighbors[at] = pairs[p].j;
                    for (int e = 0; e < 3; ++e) {
                        result.shifts[3 * at + e] = pairs[p].shift[e];
                    }
                    result.distances[at] = pairs[p].distance;
                }
            }
        },
        control);
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

NeighborPairs nearest(const double* positions, std::int64_t n, const double* matrix,
                      const double* origin, const bool* pbc, int k, const Control& control) {
    if (k < 1) {
        throw std::invalid_argument("the number of neighbors must be at least 1");
    }
    NeighborPairs result;
    result.offsets.resize(static_cast<std::size_t>(n) + 1);
    for (std::int64_t i = 0; i <= n; ++i) {
        result.offsets[i] = i * k;
    }
    result.neighbors.assign(static_cast<std::size_t>(n) * k, 0);
    result.shifts.assign(static_cast<std::size_t>(n) * k * 3, 0);
    result.distances.assign(static_cast<std::size_t>(n) * k, 0.0);
    if (n == 0) {
        return result;
    }
    // a sphere holding k atoms at the mean density, then larger for the atoms
    // that have fewer neighbors in it
    const double* h = matrix;
    const double volume = std::abs(h[0] * (h[4] * h[8] - h[5] * h[7]) -
                                   h[1] * (h[3] * h[8] - h[5] * h[6]) +
                                   h[2] * (h[3] * h[7] - h[4] * h[6]));
    double cutoff = 1.2 * std::cbrt(3.0 * k * volume / (4.0 * 3.14159265358979323846 * n));
    std::vector<std::int64_t> todo(static_cast<std::size_t>(n));
    bool all = true;
    while (!todo.empty()) {
        const CellGrid grid(positions, n, matrix, origin, pbc, cutoff);
        if (all) {
            // every atom, in cell order
            for (std::int64_t t = 0; t < n; ++t) {
                todo[static_cast<std::size_t>(t)] = grid.atom(t);
            }
            all = false;
        }
        std::vector<std::uint8_t> short_of(todo.size(), 0);
        parallel_for(
            static_cast<std::int64_t>(todo.size()), 0,
            [&](std::int64_t begin, std::int64_t end) {
                std::vector<Pair> found;
                for (std::int64_t t = begin; t < end; ++t) {
                    const std::int64_t i = todo[static_cast<std::size_t>(t)];
                    found.clear();
                    grid.visit(i, [&](std::int32_t j, const std::int32_t* shift, double r2) {
                        found.push_back({j, {shift[0], shift[1], shift[2]}, r2});
                    });
                    if (found.size() < static_cast<std::size_t>(k)) {
                        short_of[static_cast<std::size_t>(t)] = 1;
                        continue;
                    }
                    std::partial_sort(found.begin(), found.begin() + k, found.end(),
                                      [](const Pair& p, const Pair& q) {
                                          return std::tie(p.distance, p.j, p.shift) <
                                                 std::tie(q.distance, q.j, q.shift);
                                      });
                    for (int a = 0; a < k; ++a) {
                        const std::size_t at = static_cast<std::size_t>(i) * k + a;
                        result.neighbors[at] = found[a].j;
                        for (int e = 0; e < 3; ++e) {
                            result.shifts[3 * at + e] = found[a].shift[e];
                        }
                        result.distances[at] = std::sqrt(found[a].distance);
                    }
                }
            },
            control);
        std::vector<std::int64_t> left;
        for (std::size_t t = 0; t < todo.size(); ++t) {
            if (short_of[t]) {
                left.push_back(todo[t]);
            }
        }
        todo.swap(left);
        cutoff *= 1.5;
    }
    return result;
}

void pair_vectors(const double* positions, std::int64_t n, const std::int64_t* offsets,
                  const std::int32_t* neighbors, const std::int32_t* shifts, const double* matrix,
                  double* vectors, const Control& control) {
    const double* h = matrix;
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const double* xi = positions + 3 * i;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const double* xj = positions + 3 * static_cast<std::int64_t>(neighbors[p]);
                    const std::int32_t* s = shifts + 3 * p;
                    for (int e = 0; e < 3; ++e) {
                        vectors[3 * p + e] =
                            xj[e] - xi[e] + s[0] * h[e] + s[1] * h[3 + e] + s[2] * h[6 + e];
                    }
                }
            }
        },
        control);
}

void nearest_sites(const double* sites, std::int64_t n, const double* matrix,
                   const double* origin, const bool* pbc, const double* points, std::int64_t m,
                   std::int64_t* site, double* distance2, const Control& control) {
    if (n == 0 && m > 0) {
        throw std::invalid_argument("there are no sites");
    }
    const double* h = matrix;
    const double volume = std::abs(h[0] * (h[4] * h[8] - h[5] * h[7]) -
                                   h[1] * (h[3] * h[8] - h[5] * h[6]) +
                                   h[2] * (h[3] * h[7] - h[4] * h[6]));
    // a sphere holding a few sites at the mean density, larger where needed
    double cutoff = 1.5 * std::cbrt(volume / static_cast<double>(std::max<std::int64_t>(n, 1)));
    std::vector<std::int64_t> todo(static_cast<std::size_t>(m));
    std::iota(todo.begin(), todo.end(), 0);
    while (!todo.empty()) {
        const CellGrid grid(sites, n, matrix, origin, pbc, cutoff);
        std::vector<std::uint8_t> missed(todo.size(), 0);
        parallel_for(
            static_cast<std::int64_t>(todo.size()), 0,
            [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t t = begin; t < end; ++t) {
                    const std::int64_t p = todo[static_cast<std::size_t>(t)];
                    std::int64_t best = -1;
                    double best2 = 0.0;
                    grid.visit_point(points + 3 * p, [&](std::int32_t j, const std::int32_t*, double r2) {
                        if (best < 0 || r2 < best2 || (r2 == best2 && j < best)) {
                            best = j;
                            best2 = r2;
                        }
                    });
                    if (best < 0) {
                        missed[static_cast<std::size_t>(t)] = 1;
                    } else {
                        site[p] = best;
                        distance2[p] = best2;
                    }
                }
            },
            control);
        std::vector<std::int64_t> left;
        for (std::size_t t = 0; t < todo.size(); ++t) {
            if (missed[t]) {
                left.push_back(todo[t]);
            }
        }
        todo.swap(left);
        cutoff *= 2.0;
    }
}

}  // namespace mdapy::neighbor
