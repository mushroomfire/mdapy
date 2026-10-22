// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "neighbor/cell_list.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mdapy::neighbor {
namespace {

// The search reaches this factor beyond the cutoff, so that rounding in the
// fractional coordinates cannot hide a pair just inside the cutoff.
constexpr double kMargin = 1.0 + 1e-10;

// Cells across the cutoff: smaller cells hold fewer atoms beyond the cutoff.
constexpr double kCellsPerCutoff = 2.0;

// Most cells along one direction, so that cell indices stay far from overflow.
constexpr std::int64_t kMostCells = std::int64_t{1} << 30;

struct Extent {
    double low[3];
    double high[3];
    double size;  // largest component of a position or its image in the box
};

}  // namespace

CellGrid::CellGrid(const double* positions, std::int64_t n, const double* matrix,
                   const double* origin, const bool* pbc, double cutoff)
    : frame_(matrix, origin, pbc) {
    if (!(cutoff > 0.0) || !std::isfinite(cutoff)) {
        throw std::invalid_argument("cutoff must be a positive number");
    }
    check_atom_count(n);
    cutoff2_ = cutoff * cutoff;
    const double* h = frame_.h;

    // The periodic images that bring each atom into the box, and the extent
    // of the atoms in fractional coordinates.
    std::vector<std::int32_t> wraps(static_cast<std::size_t>(n) * 3);
    constexpr double kInf = std::numeric_limits<double>::infinity();
    const Extent none{{kInf, kInf, kInf}, {-kInf, -kInf, -kInf}, 0.0};
    const Extent extent = parallel_reduce(
        n, 0, none,
        [&](std::int64_t begin, std::int64_t end) {
            Extent part = none;
            for (std::int64_t i = begin; i < end; ++i) {
                const double* x = positions + 3 * i;
                double f[3], y[3];
                frame_.wrap(x, f, &wraps[3 * i]);
                frame_.image(x, &wraps[3 * i], y);
                for (int d = 0; d < 3; ++d) {
                    part.low[d] = std::min(part.low[d], f[d]);
                    part.high[d] = std::max(part.high[d], f[d]);
                    part.size = std::max(part.size, std::max(std::abs(x[d]), std::abs(y[d])));
                }
            }
            return part;
        },
        [](Extent p, const Extent& q) {
            for (int d = 0; d < 3; ++d) {
                p.low[d] = std::min(p.low[d], q.low[d]);
                p.high[d] = std::max(p.high[d], q.high[d]);
            }
            p.size = std::max(p.size, q.size);
            return p;
        });

    // Cells at least as wide as the cutoff where possible; along periodic
    // directions narrower boxes are covered by searching more cells.
    const double search = cutoff * kMargin;
    const auto reach = [&](const Axis& axis, int d) {
        auto k = static_cast<std::int64_t>(std::ceil(search / (axis.width * frame_.thickness[d])));
        return axis.periodic ? k : std::min(k, axis.cells - 1);
    };
    for (int d = 0; d < 3; ++d) {
        Axis& axis = axes_[d];
        axis.periodic = pbc[d];
        double span = 1.0;
        if (!axis.periodic) {
            axis.low = n > 0 ? extent.low[d] : 0.0;
            span = n > 0 ? std::max(extent.high[d] - extent.low[d], 1e-12) : 1e-12;
        }
        const double cells = std::floor(span * frame_.thickness[d] * kCellsPerCutoff / search);
        axis.cells = std::clamp<std::int64_t>(static_cast<std::int64_t>(std::min(cells, 1e18)), 1,
                                              kMostCells);
        axis.width = span / static_cast<double>(axis.cells);
        axis.reach = reach(axis, d);
    }
    // Keep the number of cells proportional to the number of atoms.
    while (axes_[0].cells * axes_[1].cells * axes_[2].cells > 8 * n + 64) {
        Axis& widest = *std::max_element(std::begin(axes_), std::end(axes_),
                                         [](const Axis& u, const Axis& v) { return u.cells < v.cells; });
        const double span = widest.width * static_cast<double>(widest.cells);
        widest.cells = std::max<std::int64_t>(1, widest.cells / 2);
        widest.width = span / static_cast<double>(widest.cells);
        widest.reach = reach(widest, static_cast<int>(&widest - axes_));
    }

    // Counting sort of the atoms into cells, in increasing index order.
    std::vector<std::int64_t> cell_of(static_cast<std::size_t>(n));
    parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            double f[3];
            std::int32_t w[3];
            frame_.wrap(positions + 3 * i, f, w);
            std::int64_t index = 0;
            for (int d = 0; d < 3; ++d) {
                const Axis& axis = axes_[d];
                const double k = std::floor((f[d] - axis.low) / axis.width);
                const double last = static_cast<double>(axis.cells - 1);
                index = index * axis.cells + static_cast<std::int64_t>(std::clamp(k, 0.0, last));
            }
            cell_of[i] = index;
        }
    });
    const std::int64_t total_cells = axes_[0].cells * axes_[1].cells * axes_[2].cells;
    cell_start_.assign(static_cast<std::size_t>(total_cells) + 1, 0);
    for (std::int64_t i = 0; i < n; ++i) {
        ++cell_start_[cell_of[i] + 1];
    }
    for (std::int64_t k = 0; k < total_cells; ++k) {
        cell_start_[k + 1] += cell_start_[k];
    }
    slot_cells_.resize(static_cast<std::size_t>(n));
    {
        std::vector<std::int64_t> fill(cell_start_.begin(), cell_start_.end() - 1);
        for (std::int64_t i = 0; i < n; ++i) {
            const std::int64_t t = fill[cell_of[i]]++;
            slot_cells_[t] = cell_of[i];
            cell_of[i] = t;  // now the slot of atom i
        }
    }
    slot_atoms_.resize(static_cast<std::size_t>(n));
    slot_positions_.resize(static_cast<std::size_t>(n) * 3);
    slot_wrapped_.resize(static_cast<std::size_t>(n) * 3);
    slot_wraps_.resize(static_cast<std::size_t>(n) * 3);
    parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            const std::int64_t t = cell_of[i];
            slot_atoms_[t] = static_cast<index_t>(i);
            frame_.image(positions + 3 * i, &wraps[3 * i], &slot_wrapped_[3 * t]);
            for (int e = 0; e < 3; ++e) {
                slot_positions_[3 * t + e] = positions[3 * i + e];
                slot_wraps_[3 * t + e] = wraps[3 * i + e];
            }
        }
    });

    // Every term of a distance stays below this size, the images searched
    // being at most a few box lengths away.
    double size = extent.size;
    for (int d = 0; d < 3; ++d) {
        const double* v = h + 3 * d;
        size += static_cast<double>(axes_[d].reach + 2) *
                std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    const double screen = cutoff + 1e-12 * size, inner = cutoff - 1e-12 * size;
    screen2_ = screen * screen;
    inner2_ = inner > 0.0 ? inner * inner : 0.0;
}

NeighborPairs build(const double* positions, std::int64_t n, const double* matrix,
                    const double* origin, const bool* pbc, double cutoff,
                    const Control& control) {
    const CellGrid grid(positions, n, matrix, origin, pbc, cutoff);
    NeighborPairs result;
    result.offsets.resize(static_cast<std::size_t>(n) + 1);
    result.offsets[0] = 0;
    // Count the pairs of each atom, then search again and write them where
    // they belong.
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t t = begin; t < end; ++t) {
                result.offsets[static_cast<std::int64_t>(grid.atom(t)) + 1] = grid.count(t);
            }
        },
        stage(control, 0, 2));
    for (std::int64_t i = 0; i < n; ++i) {
        result.offsets[i + 1] += result.offsets[i];
    }
    const auto total = static_cast<std::size_t>(result.offsets[n]);
    result.neighbors.resize(total);
    result.shifts.resize(3 * total);
    result.distances.resize(total);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t t = begin; t < end; ++t) {
                auto at = static_cast<std::size_t>(result.offsets[grid.atom(t)]);
                grid.visit(t, [&](index_t j, const std::int32_t* shift, double r2) {
                    result.neighbors[at] = j;
                    for (int e = 0; e < 3; ++e) {
                        result.shifts[3 * at + e] = static_cast<std::int8_t>(shift[e]);
                    }
                    result.distances[at] = std::sqrt(r2);
                    ++at;
                });
            }
        },
        stage(control, 1, 2));
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

void pair_vectors(const double* positions, std::int64_t n, const std::int64_t* offsets,
                  const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
                  double* vectors, const Control& control) {
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    pair_vector(positions + 3 * i, positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                                shifts + 3 * p, matrix, vectors + 3 * p);
                }
            }
        },
        control);
}

NeighborPairs follow(const double* positions, std::int64_t n, const std::int64_t* offsets,
                     const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
                     double cutoff, Buffer<double>& vectors, const Control& control) {
    const double cutoff2 = cutoff * cutoff;
    // the vector of pair p, as the search and pair_vectors compute it
    const auto vector = [&](std::int64_t i, std::int64_t p, double* v) {
        return pair_vector(positions + 3 * i, positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                           shifts + 3 * p, matrix, v);
    };
    NeighborPairs result;
    result.offsets.resize(static_cast<std::size_t>(n) + 1);
    result.offsets[0] = 0;
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                std::int64_t kept = 0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    double v[3];
                    kept += static_cast<std::int64_t>(vector(i, p, v) < cutoff2);
                }
                result.offsets[i + 1] = kept;
            }
        },
        stage(control, 0, 2));
    for (std::int64_t i = 0; i < n; ++i) {
        result.offsets[i + 1] += result.offsets[i];
    }
    const auto total = static_cast<std::size_t>(result.offsets[n]);
    result.neighbors.resize(total);
    result.shifts.resize(3 * total);
    result.distances.resize(total);
    vectors.resize(3 * total);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                auto at = static_cast<std::size_t>(result.offsets[i]);
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    double v[3];
                    const double r2 = vector(i, p, v);
                    if (r2 < cutoff2) {
                        result.neighbors[at] = neighbors[p];
                        for (int e = 0; e < 3; ++e) {
                            result.shifts[3 * at + e] = shifts[3 * p + e];
                            vectors[3 * at + e] = v[e];
                        }
                        result.distances[at] = std::sqrt(r2);
                        ++at;
                    }
                }
            }
        },
        stage(control, 1, 2));
    return result;
}

}  // namespace mdapy::neighbor
