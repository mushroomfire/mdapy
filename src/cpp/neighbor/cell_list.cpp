// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "neighbor/cell_list.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
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
            // the first direction varies fastest, as in the usual order of atoms
            std::int64_t index = 0;
            for (int d = 2; d >= 0; --d) {
                const Axis& axis = axes_[d];
                const double k = std::floor((f[d] - axis.low) / axis.width);
                const double last = static_cast<double>(axis.cells - 1);
                index = index * axis.cells + static_cast<std::int64_t>(std::clamp(k, 0.0, last));
            }
            cell_of[i] = index;
        }
    });
    // Counting sort of the atoms into cells, in parallel; within a cell the
    // atoms keep increasing index order, whatever the number of threads.
    const std::int64_t total_cells = axes_[0].cells * axes_[1].cells * axes_[2].cells;
    std::unique_ptr<std::atomic<std::int64_t>[]> counts(
        new std::atomic<std::int64_t>[static_cast<std::size_t>(total_cells)]);
    const auto clear = [&]() {
        parallel_for(total_cells, 0, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t k = begin; k < end; ++k) {
                counts[k].store(0, std::memory_order_relaxed);
            }
        });
    };
    clear();
    parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            counts[cell_of[i]].fetch_add(1, std::memory_order_relaxed);
        }
    });
    // the first slot of each cell: the sum of each block of cells, then the
    // cells of each block from where the block starts
    cell_start_.resize(static_cast<std::size_t>(total_cells) + 1);
    const std::int64_t grain = default_grain(total_cells);
    std::vector<std::int64_t> block_start((total_cells + grain - 1) / grain + 1, 0);
    parallel_for(total_cells, grain, [&](std::int64_t begin, std::int64_t end) {
        std::int64_t sum = 0;
        for (std::int64_t k = begin; k < end; ++k) {
            sum += counts[k].load(std::memory_order_relaxed);
        }
        block_start[begin / grain + 1] = sum;
    });
    for (std::size_t b = 1; b < block_start.size(); ++b) {
        block_start[b] += block_start[b - 1];
    }
    parallel_for(total_cells, grain, [&](std::int64_t begin, std::int64_t end) {
        std::int64_t sum = block_start[begin / grain];
        for (std::int64_t k = begin; k < end; ++k) {
            cell_start_[k] = sum;
            sum += counts[k].load(std::memory_order_relaxed);
        }
    });
    cell_start_[total_cells] = n;
    // each atom into a free slot of its cell, then the slots of each cell in
    // increasing atom order
    clear();
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            const std::int64_t c = cell_of[i];
            order[cell_start_[c] + counts[c].fetch_add(1, std::memory_order_relaxed)] = i;
        }
    });
    parallel_for(total_cells, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t k = begin; k < end; ++k) {
            std::sort(order.begin() + cell_start_[k], order.begin() + cell_start_[k + 1]);
        }
    });
    slot_cells_.resize(static_cast<std::size_t>(n) * 3);
    slots_.resize(static_cast<std::size_t>(n));
    parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t t = begin; t < end; ++t) {
            const std::int64_t i = order[t];
            for (std::int64_t c = cell_of[i], d = 0; d < 3; ++d) {
                slot_cells_[3 * t + d] = static_cast<std::int32_t>(c % axes_[d].cells);
                c /= axes_[d].cells;
            }
            Slot& slot = slots_[t];
            slot.atom = static_cast<index_t>(i);
            frame_.image(positions + 3 * i, &wraps[3 * i], slot.wrapped);
            for (int e = 0; e < 3; ++e) {
                slot.position[e] = positions[3 * i + e];
                slot.wraps[e] = wraps[3 * i + e];
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

namespace {

// Puts the first take of items in order by closer, the others after them in
// any order: selection, then a sort of the selected, faster than a heap.
template <class Closer>
void first_in_order(std::vector<std::int64_t>& items, std::ptrdiff_t take, Closer&& closer) {
    if (take < static_cast<std::ptrdiff_t>(items.size())) {
        std::nth_element(items.begin(), items.begin() + take, items.end(), closer);
    }
    std::sort(items.begin(), items.begin() + take, closer);
}

// The pairs of atom i (positions in the list, from offsets[i]) with its
// first take pairs in the order of a search of nearest neighbors: by their
// squared distance from pair_vector, ties by atom and then image.
void nearest_first(const double* positions, std::int64_t i, const std::int64_t* offsets,
                   const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
                   std::ptrdiff_t take, std::vector<std::int64_t>& pairs,
                   std::vector<double>& squares) {
    pairs.resize(static_cast<std::size_t>(offsets[i + 1] - offsets[i]));
    std::iota(pairs.begin(), pairs.end(), offsets[i]);
    squares.resize(pairs.size());
    for (std::size_t q = 0; q < pairs.size(); ++q) {
        const std::int64_t p = pairs[q];
        double r[3];
        squares[q] = pair_vector(positions + 3 * i,
                                 positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                                 shifts + 3 * p, matrix, r);
    }
    const std::int64_t first = offsets[i];
    const auto closer = [&](std::int64_t a, std::int64_t b) {
        const double ra = squares[static_cast<std::size_t>(a - first)];
        const double rb = squares[static_cast<std::size_t>(b - first)];
        if (ra != rb) {
            return ra < rb;
        }
        if (neighbors[a] != neighbors[b]) {
            return neighbors[a] < neighbors[b];
        }
        for (int e = 0; e < 3; ++e) {
            if (shifts[3 * a + e] != shifts[3 * b + e]) {
                return shifts[3 * a + e] < shifts[3 * b + e];
            }
        }
        return false;
    };
    first_in_order(pairs, take, closer);
}

// Places values (width per pair) of the pairs of one atom, from first, in
// the order of pairs, through scratch.
template <class T>
void permute(T* values, int width, std::int64_t first, const std::vector<std::int64_t>& pairs,
             std::vector<T>& scratch) {
    scratch.resize(pairs.size() * static_cast<std::size_t>(width));
    for (std::size_t q = 0; q < pairs.size(); ++q) {
        std::copy_n(values + width * pairs[q], width, scratch.data() + width * q);
    }
    std::copy(scratch.begin(), scratch.end(), values + width * first);
}

}  // namespace

void order_by_distance(const double* positions, std::int64_t n, const std::int64_t* offsets,
                       const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
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
            std::vector<double> squares;  // of the pairs of atom i, from offsets[i]
            for (std::int64_t i = begin; i < end; ++i) {
                const auto take = static_cast<std::ptrdiff_t>(kept[i + 1] - kept[i]);
                nearest_first(positions, i, offsets, neighbors, shifts, matrix, take, pairs,
                              squares);
                std::copy(pairs.begin(), pairs.begin() + take, order.begin() + kept[i]);
            }
        },
        control);
}

void sort_nearest(const double* positions, std::int64_t n, const std::int64_t* offsets,
                  index_t* neighbors, std::int8_t* shifts, double* distances,
                  const double* matrix, std::int64_t count, const Control& control) {
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<std::int64_t> pairs;
            std::vector<double> scratch;
            std::vector<index_t> scratch_neighbors;
            std::vector<std::int8_t> scratch_shifts;
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t m = offsets[i + 1] - offsets[i];
                const auto take = static_cast<std::ptrdiff_t>(count < 0 ? m : std::min(m, count));
                // the distances are the roots of the squared distances, so they
                // order the pairs as these do; equal ones are told apart by the
                // squares themselves, then by atom and image
                const auto square = [&](std::int64_t p) {
                    double r[3];
                    return pair_vector(positions + 3 * i,
                                       positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                                       shifts + 3 * p, matrix, r);
                };
                const auto closer = [&](std::int64_t a, std::int64_t b) {
                    if (distances[a] != distances[b]) {
                        return distances[a] < distances[b];
                    }
                    const double ra = square(a), rb = square(b);
                    if (ra != rb) {
                        return ra < rb;
                    }
                    if (neighbors[a] != neighbors[b]) {
                        return neighbors[a] < neighbors[b];
                    }
                    for (int e = 0; e < 3; ++e) {
                        if (shifts[3 * a + e] != shifts[3 * b + e]) {
                            return shifts[3 * a + e] < shifts[3 * b + e];
                        }
                    }
                    return false;
                };
                pairs.resize(static_cast<std::size_t>(m));
                std::iota(pairs.begin(), pairs.end(), offsets[i]);
                first_in_order(pairs, take, closer);
                permute(neighbors, 1, offsets[i], pairs, scratch_neighbors);
                permute(shifts, 3, offsets[i], pairs, scratch_shifts);
                permute(distances, 1, offsets[i], pairs, scratch);
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
                     double cutoff, const Control& control) {
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

NeighborPairs within(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
                     const std::int8_t* shifts, const double* distances,
                     const std::int32_t* type, int types, const double* limits,
                     const Control& control) {
    const auto limit = [&](std::int64_t i, std::int64_t p) {
        if (type == nullptr) {
            return limits[0];
        }
        return limits[static_cast<std::size_t>(type[i]) * types +
                      static_cast<std::size_t>(type[neighbors[p]])];
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
                    kept += static_cast<std::int64_t>(distances[p] < limit(i, p));
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
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                auto at = static_cast<std::size_t>(result.offsets[i]);
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    if (distances[p] < limit(i, p)) {
                        result.neighbors[at] = neighbors[p];
                        for (int e = 0; e < 3; ++e) {
                            result.shifts[3 * at + e] = shifts[3 * p + e];
                        }
                        result.distances[at] = distances[p];
                        ++at;
                    }
                }
            }
        },
        stage(control, 1, 2));
    return result;
}

}  // namespace mdapy::neighbor
