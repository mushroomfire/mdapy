// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "core/buffer.h"
#include "neighbor/cell_list.h"
#include "neighbor/frame.h"

namespace mdapy::analysis {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Atoms joined into clusters, each with the periodic image that places it
// next to the atoms it is bonded to: atom i at x_i + image_i . H, the first
// atom of its cluster at its own position. A union-find whose links carry
// the image of an atom relative to its parent. A cluster spans the box when
// a bond joins an atom to another image of its own cluster; its images then
// depend on the order of the bonds.
class Joined {
public:
    explicit Joined(std::int64_t n)
        : parent_(static_cast<std::size_t>(n)), image_(3 * static_cast<std::size_t>(n), 0),
          spans_(static_cast<std::size_t>(n), 0) {
        std::iota(parent_.begin(), parent_.end(), std::int64_t{0});
    }

    // Whether the cluster of root r spans the box.
    bool spans(std::int64_t r) const { return spans_[r] != 0; }

    // The root of the cluster of a, and the image of a relative to it.
    std::int64_t find(std::int64_t a, std::int64_t* image) {
        std::int64_t r = a;
        image[0] = image[1] = image[2] = 0;
        while (parent_[r] != r) {
            for (int e = 0; e < 3; ++e) {
                image[e] += image_[3 * r + e];
            }
            r = parent_[r];
        }
        // point the path at the root, each atom with its image relative to it
        std::int64_t rest[3] = {image[0], image[1], image[2]};
        while (parent_[a] != a) {
            const std::int64_t next = parent_[a];
            for (int e = 0; e < 3; ++e) {
                const std::int64_t own = image_[3 * a + e];
                image_[3 * a + e] = rest[e];
                rest[e] -= own;
            }
            parent_[a] = r;
            a = next;
        }
        return r;
    }

    // Joins the clusters of atoms a and b, bonded by the vector
    // x_b - x_a + shift . H; the smaller root becomes the root, so that the
    // root is the first atom of its cluster.
    template <class Shift>
    void join(std::int64_t a, std::int64_t b, const Shift* shift) {
        std::int64_t ia[3], ib[3];
        const std::int64_t ra = find(a, ia), rb = find(b, ib);
        if (ra == rb) {
            // a loop: around the box when b is at another image than the bond's
            for (int e = 0; e < 3; ++e) {
                if (ia[e] + static_cast<std::int64_t>(shift[e]) != ib[e]) {
                    spans_[ra] = 1;
                }
            }
            return;
        }
        // the image of the root of b relative to the root of a
        std::int64_t d[3];
        for (int e = 0; e < 3; ++e) {
            d[e] = ia[e] + static_cast<std::int64_t>(shift[e]) - ib[e];
        }
        const bool under_a = ra < rb;
        const std::int64_t child = under_a ? rb : ra;
        const std::int64_t root = under_a ? ra : rb;
        parent_[child] = root;
        spans_[root] = static_cast<std::uint8_t>(spans_[root] | spans_[child]);
        for (int e = 0; e < 3; ++e) {
            image_[3 * child + e] = under_a ? d[e] : -d[e];
        }
    }

private:
    std::vector<std::int64_t> parent_;
    std::vector<std::int64_t> image_;  // 3 per atom, relative to the parent
    std::vector<std::uint8_t> spans_;  // per root
};

// Numbers the clusters of the joined atoms by decreasing size, ties by
// their first atom, and gives each atom its image and each cluster whether
// it spans the box.
std::int64_t number(Joined& joined, std::int64_t n, std::int64_t* cluster, std::int64_t* sizes,
                    std::int32_t* images, std::uint8_t* spanning) {
    std::vector<std::int64_t> count(static_cast<std::size_t>(n), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        std::int64_t image[3];
        cluster[i] = joined.find(i, image);
        for (int e = 0; e < 3; ++e) {
            images[3 * i + e] = static_cast<std::int32_t>(image[e]);
        }
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
        spanning[k] = static_cast<std::uint8_t>(joined.spans(roots[k]));
    }
    for (std::int64_t i = 0; i < n; ++i) {
        cluster[i] = label[cluster[i]];
    }
    return static_cast<std::int64_t>(roots.size());
}

}  // namespace

std::int64_t clusters(const neighbor::ListView& list, std::int64_t n, std::int64_t* cluster,
                      std::int64_t* sizes, std::int32_t* images, std::uint8_t* spanning) {
    Joined joined(n);
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t p = list.offsets[i]; p < list.offsets[i + 1]; ++p) {
            if (list.close(i, p)) {
                joined.join(i, list.neighbors[p], list.shifts + 3 * p);
            }
        }
    }
    return number(joined, n, cluster, sizes, images, spanning);
}

std::int64_t clusters_search(const double* positions, std::int64_t n, const double* matrix,
                             const double* origin, const bool* pbc, const std::int32_t* type,
                             int types, const double* limits, bool per_pair,
                             std::int64_t* cluster, std::int64_t* sizes, std::int32_t* images,
                             std::uint8_t* spanning, const Control& control) {
    const auto pairs = static_cast<std::size_t>(types) * types;
    const double reach = *std::max_element(limits, limits + pairs);
    if (n == 0 || !(reach > 0.0)) {
        Joined joined(n);
        return number(joined, n, cluster, sizes, images, spanning);
    }
    // Each bond once, as (i, j) with i < j, joined in a fixed order. The
    // atoms are joined by their slots on the grid, where bonded atoms lie
    // close together in memory; the clusters do not depend on the order.
    struct Bond {
        std::int64_t a, b;  // slots
        std::int32_t shift[3];
    };
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, reach);
    const double reach2 = reach * reach;
    const std::int64_t grain = default_grain(n);
    std::vector<std::vector<Bond>> bonds(static_cast<std::size_t>((n + grain - 1) / grain));
    parallel_for(
        n, grain,
        [&](std::int64_t begin, std::int64_t end) {
            auto& out = bonds[static_cast<std::size_t>(begin / grain)];
            for (std::int64_t t = begin; t < end; ++t) {
                const index_t i = grid.atom(t);
                const double* row = limits + static_cast<std::size_t>(type[i]) * types;
                grid.visit_slots_measured(t, [&](std::int64_t s, const std::int32_t* d, double r2) {
                    // j < i: the bond is the other atom's; j == i: a bond
                    // to an image of i, which makes its cluster span the box
                    const index_t j = grid.atom(s);
                    if (j >= i && (per_pair ? std::sqrt(r2) < row[type[j]] : r2 < reach2)) {
                        out.push_back({t, s, {d[0], d[1], d[2]}});
                    }
                });
            }
        },
        control);
    Joined joined(n);
    for (const auto& chunk : bonds) {
        for (const auto& bond : chunk) {
            joined.join(bond.a, bond.b, bond.shift);
        }
    }
    // the root slot of each slot and its image relative to the root, then
    // for each root the first atom of its cluster and the slot of that atom
    const auto size = static_cast<std::size_t>(n);
    Buffer<std::int64_t> root(size), image(3 * size);
    std::vector<std::int64_t> first(size, n), first_slot(size, 0), count(size, 0);
    for (std::int64_t t = 0; t < n; ++t) {
        const std::int64_t r = joined.find(t, &image[3 * t]);
        root[t] = r;
        ++count[r];
        const std::int64_t i = grid.atom(t);
        if (i < first[r]) {
            first[r] = i;
            first_slot[r] = t;
        }
    }
    // the clusters by decreasing size, ties by their first atom
    std::vector<std::int64_t> roots;
    for (std::int64_t t = 0; t < n; ++t) {
        if (root[t] == t) {
            roots.push_back(t);
        }
    }
    std::sort(roots.begin(), roots.end(), [&](std::int64_t a, std::int64_t b) {
        return count[a] != count[b] ? count[a] > count[b] : first[a] < first[b];
    });
    std::vector<std::int64_t> label(size);
    for (std::size_t k = 0; k < roots.size(); ++k) {
        label[roots[k]] = static_cast<std::int64_t>(k);
        sizes[k] = count[roots[k]];
        spanning[k] = static_cast<std::uint8_t>(joined.spans(roots[k]));
    }
    // each atom with its image relative to the first atom of its cluster
    parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t t = begin; t < end; ++t) {
            const std::int64_t r = root[t], i = grid.atom(t);
            const std::int64_t* own = &image[3 * t];
            const std::int64_t* base = &image[3 * first_slot[r]];
            cluster[i] = label[r];
            for (int e = 0; e < 3; ++e) {
                images[3 * i + e] = static_cast<std::int32_t>(own[e] - base[e]);
            }
        }
    });
    return static_cast<std::int64_t>(roots.size());
}

namespace {

// Histograms of bond angles by the type of the atom and the types a <= b of
// its two neighbors, the pair (a, b) numbered after the pairs that start
// below a.
class AngleBins {
public:
    AngleBins(int types, int bins)
        : types_(types), bins_(bins), pairs_(static_cast<std::int64_t>(types) * (types + 1) / 2),
          first_(static_cast<std::size_t>(types)) {
        for (int a = 0; a < types; ++a) {
            first_[a] = static_cast<std::int64_t>(a) * types -
                        static_cast<std::int64_t>(a) * (a - 1) / 2 - a;
        }
    }

    std::size_t size() const { return static_cast<std::size_t>(types_ * pairs_ * bins_); }

    // Counts the angles between every two of the count bond vectors (count x
    // 3) of an atom of type center; type_of(a) is the type of neighbor a.
    template <class TypeOf>
    void add(const double* vectors, std::int64_t count, std::int32_t center, TypeOf type_of,
             std::vector<std::int64_t>& histogram) const {
        const std::int64_t base = center * pairs_;
        // the length of each bond once, as each angle would compute it
        thread_local std::vector<double> lengths;
        lengths.resize(static_cast<std::size_t>(count));
        for (std::int64_t a = 0; a < count; ++a) {
            const double* u = vectors + 3 * a;
            lengths[a] = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        }
        for (std::int64_t a = 0; a < count; ++a) {
            const double* u = vectors + 3 * a;
            const double ru = lengths[a];
            const std::int32_t ta = type_of(a);
            for (std::int64_t b = a + 1; b < count; ++b) {
                const double* v = vectors + 3 * b;
                const double rv = lengths[b];
                double c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (ru * rv);
                c = std::max(-1.0, std::min(1.0, c));
                const double angle = std::acos(c) * 180.0 / kPi;
                const int bin = std::min(static_cast<int>(angle / 180.0 * bins_), bins_ - 1);
                const std::int32_t tb = type_of(b);
                const std::int32_t low = std::min(ta, tb), high = std::max(ta, tb);
                ++histogram[static_cast<std::size_t>((base + first_[low] + high) * bins_ + bin)];
            }
        }
    }

private:
    int types_, bins_;
    std::int64_t pairs_;
    std::vector<std::int64_t> first_;
};

std::vector<std::int64_t> sum_histograms(std::vector<std::int64_t> a,
                                         const std::vector<std::int64_t>& b) {
    for (std::size_t k = 0; k < a.size(); ++k) {
        a[k] += b[k];
    }
    return a;
}

}  // namespace

void bond_angles(const neighbor::ListView& list, std::int64_t n, const std::int32_t* type,
                 int types, int bins, std::int64_t* counts, const Control& control) {
    using Histogram = std::vector<std::int64_t>;
    const AngleBins shape(types, bins);
    // at most 64 histograms in flight, whatever the number of types
    const std::int64_t grain = std::max(default_grain(n), (n + 63) / 64);
    const Histogram total = parallel_reduce(
        n, grain, Histogram(shape.size(), 0),
        [&](std::int64_t begin, std::int64_t end) {
            Histogram local(shape.size(), 0);
            std::vector<double> found;  // the bond vectors of an atom
            std::vector<index_t> others;
            for (std::int64_t i = begin; i < end; ++i) {
                found.clear();
                others.clear();
                list.for_each(i, [&](std::int64_t, index_t j, const double* r) {
                    found.insert(found.end(), r, r + 3);
                    others.push_back(j);
                });
                const index_t* own = others.data();
                shape.add(found.data(), static_cast<std::int64_t>(others.size()), type[i],
                          [own, type](std::int64_t a) { return type[own[a]]; }, local);
            }
            return local;
        },
        sum_histograms, control);
    std::copy(total.begin(), total.end(), counts);
}

void bond_angles_search(const double* positions, std::int64_t n, const double* matrix,
                        const double* origin, const bool* pbc, const std::int32_t* type,
                        int types, const double* limits, bool per_pair, int bins,
                        std::int64_t* counts, const Control& control) {
    using Histogram = std::vector<std::int64_t>;
    const AngleBins shape(types, bins);
    const auto pairs = static_cast<std::size_t>(types) * types;
    const double reach = *std::max_element(limits, limits + pairs);
    if (n == 0 || !(reach > 0.0)) {
        std::fill(counts, counts + shape.size(), 0);
        return;
    }
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, reach);
    const double reach2 = reach * reach;
    const std::int64_t grain = std::max(default_grain(n), (n + 63) / 64);
    const Histogram total = parallel_reduce(
        n, grain, Histogram(shape.size(), 0),
        [&](std::int64_t begin, std::int64_t end) {
            Histogram local(shape.size(), 0);
            std::vector<double> found;  // the bond vectors of an atom, as pair_vectors gives them
            std::vector<std::int32_t> kinds;
            for (std::int64_t t = begin; t < end; ++t) {
                const std::int64_t i = grid.atom(t);
                const double* xi = positions + 3 * i;
                const double* row = limits + static_cast<std::size_t>(type[i]) * types;
                found.clear();
                kinds.clear();
                grid.visit(t, [&](index_t j, const std::int32_t* s, double r2) {
                    if (per_pair ? std::sqrt(r2) < row[type[j]] : r2 < reach2) {
                        double r[3];
                        neighbor::pair_vector(xi, positions + 3 * static_cast<std::int64_t>(j), s,
                                              matrix, r);
                        found.insert(found.end(), r, r + 3);
                        kinds.push_back(type[j]);
                    }
                });
                shape.add(found.data(), static_cast<std::int64_t>(kinds.size()), type[i],
                          [&kinds](std::int64_t a) { return kinds[static_cast<std::size_t>(a)]; },
                          local);
            }
            return local;
        },
        sum_histograms, control);
    std::copy(total.begin(), total.end(), counts);
}

void lindemann(std::int64_t n, std::int64_t frames, const double* positions, double* atomic,
               double* running, const Control& control) {
    using Sums = std::vector<double>;
    // every ordered pair, so that each atom's mean is its own; each pair
    // is counted twice in running, which divides by n (n - 1)
    const Sums total = parallel_reduce(
        n, 0, Sums(static_cast<std::size_t>(frames), 0.0),
        [&](std::int64_t begin, std::int64_t end) {
            Sums local(static_cast<std::size_t>(frames), 0.0);
            for (std::int64_t i = begin; i < end; ++i) {
                const double* xi = positions + 3 * frames * i;
                double sum = 0.0;
                for (std::int64_t j = 0; j < n; ++j) {
                    if (j == i) {
                        continue;
                    }
                    const double* xj = positions + 3 * frames * j;
                    double mean = 0.0, deviations = 0.0, ratio = 0.0;
                    for (std::int64_t t = 0; t < frames; ++t) {
                        const double dx = xi[3 * t] - xj[3 * t];
                        const double dy = xi[3 * t + 1] - xj[3 * t + 1];
                        const double dz = xi[3 * t + 2] - xj[3 * t + 2];
                        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                        const double delta = d - mean;
                        mean += delta / static_cast<double>(t + 1);
                        deviations += delta * (d - mean);
                        ratio = std::sqrt(deviations / static_cast<double>(t + 1)) / mean;
                        local[static_cast<std::size_t>(t)] += ratio;
                    }
                    sum += ratio;
                }
                atomic[i] = sum / static_cast<double>(n - 1);
            }
            return local;
        },
        [](Sums a, const Sums& b) {
            for (std::size_t k = 0; k < a.size(); ++k) {
                a[k] += b[k];
            }
            return a;
        },
        control);
    const double pairs = static_cast<double>(n) * static_cast<double>(n - 1);
    for (std::int64_t t = 0; t < frames; ++t) {
        running[t] = total[static_cast<std::size_t>(t)] / pairs;
    }
}

namespace {

// The atoms around one atom, its neighbors summed first, for the temperature
// relative to their center of mass.
struct Group {
    std::int64_t members = 1;
    double masses = 0.0, momentum[3] = {0.0, 0.0, 0.0}, squares = 0.0;

    void add(double mass, const double* v) {
        ++members;
        masses += mass;
        for (int e = 0; e < 3; ++e) {
            momentum[e] += mass * v[e];
        }
        squares += mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    // With the atom itself of the mass and velocity given; NaN when alone.
    double temperature(double mass, const double* v, double energy_unit, double boltzmann) const {
        if (members == 1) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double total = mass + masses;
        double p2 = 0.0;
        for (int e = 0; e < 3; ++e) {
            const double pe = mass * v[e] + momentum[e];
            p2 += pe * pe;
        }
        const double twice = (mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + squares) - p2 / total;
        return twice * energy_unit / (3.0 * static_cast<double>(members - 1) * boltzmann);
    }
};

}  // namespace

void atomic_temperature(const neighbor::ListView& list, std::int64_t n, const double* mass,
                        const double* velocities, double energy_unit, double boltzmann,
                        double* temperature, const Control& control) {
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                Group group;
                list.for_each(i, [&](std::int64_t, index_t neighbor, const double*) {
                    const auto j = static_cast<std::int64_t>(neighbor);
                    if (j != i) {  // not an image of the atom itself
                        group.add(mass[j], velocities + 3 * j);
                    }
                });
                temperature[i] =
                    group.temperature(mass[i], velocities + 3 * i, energy_unit, boltzmann);
            }
        },
        control);
}

void atomic_temperature_search(const double* positions, std::int64_t n, const double* matrix,
                               const double* origin, const bool* pbc, double cutoff,
                               const double* mass, const double* velocities, double energy_unit,
                               double boltzmann, double* temperature, const Control& control) {
    if (n == 0) {
        return;
    }
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, cutoff);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t t = begin; t < end; ++t) {
                const std::int64_t i = grid.atom(t);
                Group group;
                grid.visit(t, [&](index_t j, const std::int32_t*, double) {
                    if (j != i) {  // not an image of the atom itself
                        group.add(mass[j], velocities + 3 * static_cast<std::int64_t>(j));
                    }
                });
                temperature[i] =
                    group.temperature(mass[i], velocities + 3 * i, energy_unit, boltzmann);
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

void pair_entropy(const neighbor::ListView& list, std::int64_t n, double cutoff, double sigma,
                  double density, double* entropy, const Control& control) {
    neighbor::ListView within = list;
    within.cutoff = cutoff;
    const EntropyShape shape(cutoff, sigma);
    const double sphere = 4.0 / 3.0 * kPi * cutoff * cutoff * cutoff;
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> g(static_cast<std::size_t>(shape.bins())), found;
            for (std::int64_t i = begin; i < end; ++i) {
                found.clear();
                within.for_each(i, [&](std::int64_t p, index_t, const double*) {
                    found.push_back(within.distances[p]);
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
                grid.visit(t, [&](index_t, const std::int32_t*, double r2) {
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
