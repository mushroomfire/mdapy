// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/steinhardt.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#include "neighbor/frame.h"
#include "neighbor/kd_tree.h"

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

// The harmonics of the bonds that each_bond(add) passes to add(vector,
// weight), averaged with their weights, into sum (width values); y is room
// for the harmonics of one bond.
template <class Bonds>
void bond_harmonics(Bonds&& each_bond, const std::vector<int>& degrees, int lmax,
                    std::size_t width, double* y, double* sum) {
    for (std::size_t k = 0; k < width; ++k) {
        sum[k] = 0.0;
    }
    double total = 0.0;  // of the weights
    each_bond([&](const double* v, double weight) {
        total += weight;
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
    });
    if (total > 0.0) {
        for (std::size_t k = 0; k < width; ++k) {
            sum[k] /= total;
        }
    }
}

// The bonds of atom i in list, each weighted by weights at its pair, or by
// 1 when weights is null.
auto bonds_of(const neighbor::ListView& list, std::int64_t i, const double* weights) {
    return [&list, i, weights](auto&& add) {
        list.for_each(i, [&](std::int64_t p, index_t, const double* v) {
            add(v, weights != nullptr ? weights[p] : 1.0);
        });
    };
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

namespace {

// The width of the harmonics of the degrees, and the largest degree.
std::pair<std::size_t, int> harmonics_of(const std::vector<int>& degrees) {
    std::size_t width = 0;
    int lmax = 0;
    for (int l : degrees) {
        if (l < 0 || l > kMaxDegree) {
            throw std::invalid_argument("degrees must be from 0 to 20");
        }
        width += 2 * static_cast<std::size_t>(l) + 1;
        lmax = std::max(lmax, l);
    }
    return {width, lmax};
}

// Room per thread for the bonds of one atom, kept between atoms; declared
// here, as a thread_local inside the lambda that uses it crashes MSVC.
struct NearestRoom {
    std::vector<double> vectors, y, sum;
};

NearestRoom& nearest_room() {
    thread_local NearestRoom room;
    return room;
}

}  // namespace

void steinhardt_nearest(const double* positions, std::int64_t n, const double* matrix,
                        const double* origin, const bool* pbc, int k,
                        const std::vector<int>& degrees, double* q, const Control& control) {
    const std::pair<std::size_t, int> shape = harmonics_of(degrees);
    const std::size_t width = shape.first;
    const int lmax = shape.second;
    if (n == 0) {
        return;
    }
    const neighbor::KdTree tree(positions, n, matrix, origin, pbc);
    tree.for_each_nearest(
        k,
        [&](std::int64_t i, const neighbor::Neighbor* found) {
            NearestRoom& room = nearest_room();
            std::vector<double>& vectors = room.vectors;
            std::vector<double>& y = room.y;
            std::vector<double>& sum = room.sum;
            vectors.resize(3 * static_cast<std::size_t>(k));
            y.resize(width);
            sum.resize(width);
            const double* xi = positions + 3 * i;
            for (int a = 0; a < k; ++a) {
                neighbor::pair_vector(xi, positions + 3 * static_cast<std::int64_t>(found[a].j),
                                      found[a].shift, matrix, &vectors[3 * a]);
            }
            bond_harmonics(
                [&](auto&& add) {
                    for (int a = 0; a < k; ++a) {
                        add(&vectors[3 * static_cast<std::size_t>(a)], 1.0);
                    }
                },
                degrees, lmax, width, y.data(), sum.data());
            invariants(sum.data(), degrees, q + static_cast<std::size_t>(i) * degrees.size());
        },
        control);
}

void steinhardt(const neighbor::ListView& list, std::int64_t n, const double* weights,
                const std::vector<int>& degrees, bool average, double* q,
                const Control& control) {
    const std::pair<std::size_t, int> shape = harmonics_of(degrees);
    const std::size_t width = shape.first;
    const int lmax = shape.second;
    const std::size_t count = degrees.size();
    if (!average) {
        // each atom alone: no harmonics kept between the atoms
        parallel_for(
            n, 0,
            [&](std::int64_t begin, std::int64_t end) {
                std::vector<double> y(width), sum(width);
                for (std::int64_t i = begin; i < end; ++i) {
                    bond_harmonics(bonds_of(list, i, weights), degrees, lmax, width, y.data(),
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
                bond_harmonics(bonds_of(list, i, weights), degrees, lmax, width, y.data(),
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
                const std::int64_t bonds = list.for_each(i, [&](std::int64_t, index_t j,
                                                                const double*) {
                    const double* other = &qlm[static_cast<std::size_t>(j) * width];
                    for (std::size_t k = 0; k < width; ++k) {
                        mean[k] += other[k];
                    }
                });
                const auto members = static_cast<double>(bonds + 1);
                for (std::size_t k = 0; k < width; ++k) {
                    mean[k] /= members;
                }
                invariants(mean.data(), degrees, q + static_cast<std::size_t>(i) * count);
            }
        },
        control);
}

void solid_bonds(const neighbor::ListView& list, std::int64_t n, const double* weights, int l,
                 double threshold, std::int32_t* bonds, const Control& control) {
    const std::vector<int> degrees = {l};
    const std::pair<std::size_t, int> shape = harmonics_of(degrees);
    const std::size_t width = shape.first;
    // the harmonics of every atom, of length 1, or 0 without pairs
    std::unique_ptr<double[]> unit(new double[static_cast<std::size_t>(n) * width]);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> y(width);
            for (std::int64_t i = begin; i < end; ++i) {
                double* own = &unit[static_cast<std::size_t>(i) * width];
                bond_harmonics(bonds_of(list, i, weights), degrees, l, width, y.data(), own);
                double norm = 0.0;
                for (std::size_t k = 0; k < width; ++k) {
                    norm += own[k] * own[k];
                }
                norm = std::sqrt(norm);
                for (std::size_t k = 0; k < width; ++k) {
                    own[k] = norm > 0.0 ? own[k] / norm : 0.0;
                }
            }
        },
        control);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const double* own = &unit[static_cast<std::size_t>(i) * width];
                std::int32_t solid = 0;
                list.for_each(i, [&](std::int64_t, index_t j, const double*) {
                    const double* other = &unit[static_cast<std::size_t>(j) * width];
                    double product = 0.0;
                    for (std::size_t k = 0; k < width; ++k) {
                        product += own[k] * other[k];
                    }
                    solid += product > threshold ? 1 : 0;
                });
                bonds[i] = solid;
            }
        },
        control);
}

}  // namespace mdapy::analysis
