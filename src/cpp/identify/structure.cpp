// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "identify/structure.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "analysis/steinhardt.h"
#include "identify/cna.h"
#include "neighbor/cell_list.h"
#include "neighbor/frame.h"
#include "neighbor/kd_tree.h"

namespace mdapy::identify {
namespace {

constexpr int kMaxNeighbors = 14;

// Squared length below which a first and a second neighbor vector lead back
// to the atom itself, in Å².
constexpr double kBackBond = 1e-4;

// Bonds among the first m neighbor vectors no longer than the cutoff.
void bond_masks(const double* v, int m, double cutoff2, std::uint32_t* bonds) {
    for (int a = 0; a < kMaxNeighbors; ++a) {
        bonds[a] = 0;
    }
    for (int a = 0; a < m; ++a) {
        for (int b = a + 1; b < m; ++b) {
            const double dx = v[3 * a] - v[3 * b];
            const double dy = v[3 * a + 1] - v[3 * b + 1];
            const double dz = v[3 * a + 2] - v[3 * b + 2];
            const auto bond = static_cast<std::uint32_t>(dx * dx + dy * dy + dz * dz <= cutoff2);
            bonds[a] |= bond << b;
            bonds[b] |= bond << a;
        }
    }
}

// fcc, hcp or ico from 12 neighbors, else other.
Structure twelve(const double* v, double cutoff2) {
    std::uint32_t bonds[kMaxNeighbors];
    bond_masks(v, 12, cutoff2, bonds);
    int n421 = 0, n422 = 0, n555 = 0;
    for (int j = 0; j < 12; ++j) {
        const int common = count(bonds[j]);
        if (common != 4 && common != 5) {
            return kOther;
        }
        const Signature s = signature(bonds, j);
        if (s.common == 4 && s.bonds == 2 && s.chain == 1) {
            ++n421;
        } else if (s.common == 4 && s.bonds == 2 && s.chain == 2) {
            ++n422;
        } else if (s.common == 5 && s.bonds == 5 && s.chain == 5) {
            ++n555;
        } else {
            return kOther;
        }
    }
    if (n421 == 12) {
        return kFcc;
    }
    if (n421 == 6 && n422 == 6) {
        return kHcp;
    }
    return n555 == 12 ? kIco : kOther;
}

// bcc from 14 neighbors, else other.
Structure fourteen(const double* v, double cutoff2) {
    std::uint32_t bonds[kMaxNeighbors];
    bond_masks(v, 14, cutoff2, bonds);
    int n444 = 0, n666 = 0;
    for (int j = 0; j < 14; ++j) {
        const int common = count(bonds[j]);
        if (common != 4 && common != 6) {
            return kOther;
        }
        const Signature s = signature(bonds, j);
        if (s.common == 4 && s.bonds == 4 && s.chain == 4) {
            ++n444;
        } else if (s.common == 6 && s.bonds == 6 && s.chain == 6) {
            ++n666;
        } else {
            return kOther;
        }
    }
    return n444 == 6 && n666 == 8 ? kBcc : kOther;
}

double length(const double* v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// Ackland-Jones type of one atom from the vectors to its m nearest neighbors,
// nearest first, of which the first 14 count.
Structure ackland_jones_of(const double* v, std::int64_t m) {
    if (m < 6) {
        return kOther;
    }
    const int k = static_cast<int>(std::min<std::int64_t>(m, kMaxNeighbors));
    double r2[kMaxNeighbors];
    for (int a = 0; a < k; ++a) {
        r2[a] = v[3 * a] * v[3 * a] + v[3 * a + 1] * v[3 * a + 1] + v[3 * a + 2] * v[3 * a + 2];
    }
    // the mean squared distance of the six nearest
    double r0 = 0.0;
    for (int a = 0; a < 6; ++a) {
        r0 += r2[a];
    }
    r0 /= 6.0;
    int n0 = 0, n1 = 0;
    for (int a = 0; a < k; ++a) {
        n1 += r2[a] < 1.55 * r0;
        n0 += r2[a] < 1.45 * r0;
    }
    // the cosines of the angles between the n0 nearest, binned
    int chi[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int a = 0; a < n0; ++a) {
        for (int b = a + 1; b < n0; ++b) {
            const double c = (v[3 * a] * v[3 * b] + v[3 * a + 1] * v[3 * b + 1] +
                              v[3 * a + 2] * v[3 * b + 2]) /
                             (std::sqrt(r2[a]) * std::sqrt(r2[b]));
            const int bin = c < -0.945  ? 0
                            : c < -0.915 ? 1
                            : c < -0.755 ? 2
                            : c < -0.195 ? 3
                            : c < 0.195  ? 4
                            : c < 0.245  ? 5
                            : c < 0.795  ? 6
                                         : 7;
            ++chi[bin];
        }
    }
    // deviations from each structure; without the bcc denominator, not bcc
    const double delta_cp = std::abs(1.0 - chi[6] / 24.0);
    const int denominator = chi[5] + chi[6] - chi[4];
    double delta_bcc = denominator != 0 ? 0.35 * chi[4] / denominator : delta_cp + 1.0;
    double delta_fcc = 0.61 * (std::abs(chi[0] + chi[1] - 6) + chi[2]) / 6.0;
    double delta_hcp =
        (std::abs(chi[0] - 3.0) + std::abs(chi[0] + chi[1] + chi[2] + chi[3] - 9)) / 12.0;
    if (chi[0] == 7) {
        delta_bcc = 0.0;
    } else if (chi[0] == 6) {
        delta_fcc = 0.0;
    } else if (chi[0] <= 3) {
        delta_hcp = 0.0;
    }
    if (chi[7] > 0) {
        return kOther;
    }
    if (chi[4] < 3) {
        return n1 > 13 || n1 < 11 ? kOther : kIco;
    }
    if (delta_bcc <= delta_cp) {
        return n1 < 11 ? kOther : kBcc;
    }
    if (n1 > 12 || n1 < 11) {
        return kOther;
    }
    return delta_fcc < delta_hcp ? kFcc : kHcp;
}

// The centro-symmetry parameter of one atom from the vectors to its k
// nearest neighbors; values is room for k (k - 1) / 2 numbers.
double centro_symmetry_of(const double* v, int k, double* values) {
    std::size_t at = 0;
    for (int a = 0; a < k; ++a) {
        for (int b = a + 1; b < k; ++b) {
            const double x = v[3 * a] + v[3 * b];
            const double y = v[3 * a + 1] + v[3 * b + 1];
            const double z = v[3 * a + 2] + v[3 * b + 2];
            values[at++] = x * x + y * y + z * z;
        }
    }
    const auto half = static_cast<std::ptrdiff_t>(k / 2);
    std::partial_sort(values, values + half, values + at);
    double sum = 0.0;
    for (std::ptrdiff_t a = 0; a < half; ++a) {
        sum += values[a];
    }
    return sum;
}

// The common neighbor analysis type of one atom with adaptive cutoffs, from
// the vectors to its m nearest neighbors, nearest first.
Structure adaptive_common_neighbors(const double* v, std::int64_t m) {
    // (1 + sqrt 2) / 2 lies halfway between the first and second shells of fcc
    const double scale = 0.5 * (1.0 + std::sqrt(2.0));
    Structure type = kOther;
    if (m >= 12) {
        double sum = 0.0;
        for (int a = 0; a < 12; ++a) {
            sum += length(v + 3 * a);
        }
        const double local = scale * sum / 12.0;
        type = twelve(v, local * local);
    }
    if (type == kOther && m >= 14) {
        double first = 0.0, second = 0.0;
        for (int a = 0; a < 8; ++a) {
            first += length(v + 3 * a);
        }
        for (int a = 8; a < 14; ++a) {
            second += length(v + 3 * a);
        }
        const double local = scale * (2.0 / std::sqrt(3.0) * first + second) / 14.0;
        type = fourteen(v, local * local);
    }
    return type;
}

// Calls compute(i, vectors) for each atom i with the vectors to its k
// nearest neighbors, nearest first, found by a k-d tree, as pair_vectors
// computes them for a list of nearest neighbors.
template <class Compute>
void with_nearest(const double* positions, std::int64_t n, const double* matrix,
                  const double* origin, const bool* pbc, int k, Compute&& compute,
                  const Control& control) {
    if (n == 0) {
        return;
    }
    const neighbor::KdTree tree(positions, n, matrix, origin, pbc);
    tree.for_each_nearest(
        k,
        [&](std::int64_t i, const neighbor::Neighbor* found) {
            double v[3 * kMaxNeighbors];
            const double* xi = positions + 3 * i;
            for (int a = 0; a < k; ++a) {
                neighbor::pair_vector(xi, positions + 3 * static_cast<std::int64_t>(found[a].j),
                                      found[a].shift, matrix, v + 3 * a);
            }
            compute(i, static_cast<const double*>(v));
        },
        control);
}

}  // namespace

void centro_symmetry(const neighbor::ListView& list, std::int64_t n, int k, double* csp,
                     const Control& control) {
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> values(static_cast<std::size_t>(k) * (k - 1) / 2);
            std::vector<double> v(3 * static_cast<std::size_t>(k));
            for (std::int64_t i = begin; i < end; ++i) {
                csp[i] = list.gather(i, v.data(), k) < k
                             ? std::numeric_limits<double>::quiet_NaN()
                             : centro_symmetry_of(v.data(), k, values.data());
            }
        },
        control);
}

void centro_symmetry_nearest(const double* positions, std::int64_t n, const double* matrix,
                             const double* origin, const bool* pbc, int k, double* csp,
                             const Control& control) {
    if (k < 2 || k > kMaxNeighbors) {
        throw std::invalid_argument("k must be from 2 to 14");
    }
    with_nearest(
        positions, n, matrix, origin, pbc, k,
        [&](std::int64_t i, const double* v) {
            double values[kMaxNeighbors * (kMaxNeighbors - 1) / 2];
            csp[i] = centro_symmetry_of(v, k, values);
        },
        control);
}

void common_neighbor_analysis_nearest(const double* positions, std::int64_t n,
                                      const double* matrix, const double* origin,
                                      const bool* pbc, int k, std::int8_t* types,
                                      const Control& control) {
    if (k < 1 || k > kMaxNeighbors) {
        throw std::invalid_argument("k must be from 1 to 14");
    }
    with_nearest(
        positions, n, matrix, origin, pbc, k,
        [&](std::int64_t i, const double* v) { types[i] = adaptive_common_neighbors(v, k); },
        control);
}

void ackland_jones_nearest(const double* positions, std::int64_t n, const double* matrix,
                           const double* origin, const bool* pbc, int k, std::int8_t* types,
                           const Control& control) {
    if (k < 1 || k > kMaxNeighbors) {
        throw std::invalid_argument("k must be from 1 to 14");
    }
    with_nearest(
        positions, n, matrix, origin, pbc, k,
        [&](std::int64_t i, const double* v) { types[i] = ackland_jones_of(v, k); },
        control);
}

namespace {

// The common neighbor parameter of one atom from the vectors (count x 3) to
// its neighbors: with v the vectors from the atom, r_ik + r_jk = v_j - 2 v_k.
double common_neighbor_sum(const double* vectors, std::int64_t count, double cutoff2) {
    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (std::int64_t j = 0; j < count; ++j) {
        const double* vj = vectors + 3 * j;
        double r[3] = {0.0, 0.0, 0.0};
        for (std::int64_t k = 0; k < count; ++k) {
            const double* vk = vectors + 3 * k;
            const double d[3] = {vk[0] - vj[0], vk[1] - vj[1], vk[2] - vj[2]};
            if (k != j && d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < cutoff2) {
                for (int e = 0; e < 3; ++e) {
                    r[e] += vj[e] - 2.0 * vk[e];
                }
            }
        }
        sum += r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    }
    return sum / static_cast<double>(count);
}

}  // namespace

void common_neighbor_parameter(const neighbor::ListView& list, std::int64_t n, double cutoff,
                               double* cnp, const Control& control) {
    neighbor::ListView within = list;
    within.cutoff = cutoff;
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> found;  // the vectors to the neighbors
            for (std::int64_t i = begin; i < end; ++i) {
                found.clear();
                within.for_each(i, [&](std::int64_t, index_t, const double* r) {
                    found.insert(found.end(), r, r + 3);
                });
                cnp[i] = common_neighbor_sum(found.data(),
                                             static_cast<std::int64_t>(found.size() / 3),
                                             cutoff * cutoff);
            }
        },
        control);
}

void common_neighbor_parameter_search(const double* positions, std::int64_t n,
                                      const double* matrix, const double* origin,
                                      const bool* pbc, double cutoff, double* cnp,
                                      const Control& control) {
    if (n == 0) {
        return;
    }
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, cutoff);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> found;  // the vectors to the neighbors, as pair_vectors gives them
            for (std::int64_t t = begin; t < end; ++t) {
                const std::int64_t i = grid.atom(t);
                const double* xi = positions + 3 * i;
                found.clear();
                grid.visit(t, [&](index_t j, const std::int32_t* s, double) {
                    double r[3];
                    neighbor::pair_vector(xi, positions + 3 * static_cast<std::int64_t>(j), s,
                                          matrix, r);
                    found.insert(found.end(), r, r + 3);
                });
                cnp[i] = common_neighbor_sum(found.data(),
                                             static_cast<std::int64_t>(found.size() / 3),
                                             cutoff * cutoff);
            }
        },
        control);
}

void common_neighbor_analysis(const neighbor::ListView& list, std::int64_t n,
                              std::int8_t* types, const Control& control) {
    const double cutoff = list.cutoff;
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            double v[3 * kMaxNeighbors];
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t m = list.gather(i, v, kMaxNeighbors);
                Structure type = kOther;
                if (cutoff > 0.0) {
                    if (m == 12) {
                        type = twelve(v, cutoff * cutoff);
                    } else if (m == 14) {
                        type = fourteen(v, cutoff * cutoff);
                    }
                } else {
                    type = adaptive_common_neighbors(v, std::min<std::int64_t>(m, kMaxNeighbors));
                }
                types[i] = type;
            }
        },
        control);
}

void common_neighbor_analysis_search(const double* positions, std::int64_t n,
                                     const double* matrix, const double* origin, const bool* pbc,
                                     double cutoff, std::int8_t* types, const Control& control) {
    if (n == 0) {
        return;
    }
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, cutoff);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            double found[3 * kMaxNeighbors];  // as pair_vectors gives them
            for (std::int64_t t = begin; t < end; ++t) {
                const std::int64_t i = grid.atom(t);
                const double* xi = positions + 3 * i;
                int m = 0;
                grid.visit(t, [&](index_t j, const std::int32_t* s, double) {
                    if (m < kMaxNeighbors) {
                        neighbor::pair_vector(xi, positions + 3 * static_cast<std::int64_t>(j), s,
                                              matrix, found + 3 * m);
                    }
                    ++m;
                });
                Structure type = kOther;
                if (m == 12) {
                    type = twelve(found, cutoff * cutoff);
                } else if (m == 14) {
                    type = fourteen(found, cutoff * cutoff);
                }
                types[i] = type;
            }
        },
        control);
}

void ackland_jones(const neighbor::ListView& list, std::int64_t n, std::int8_t* types,
                   const Control& control) {
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            double v[3 * kMaxNeighbors];
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t m = list.gather(i, v, kMaxNeighbors);
                types[i] = ackland_jones_of(v, std::min<std::int64_t>(m, kMaxNeighbors));
            }
        },
        control);
}

void identify_diamond(const neighbor::ListView& list, std::int64_t n, std::int8_t* types,
                      const Control& control) {
    const double scale = 0.5 * (1.0 + std::sqrt(2.0));
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            double second[12 * 3];
            double first[4 * 3], further[4 * 3];
            index_t neighbors[4];
            for (std::int64_t i = begin; i < end; ++i) {
                if (list.gather(i, first, 4, neighbors) < 4) {
                    types[i] = kNotDiamond;
                    continue;
                }
                int m = 0;
                bool back_to_i = true;  // i is among the four nearest of each neighbor
                bool enough = true;  // every neighbor has four neighbors of its own
                for (int a = 0; a < 4; ++a) {
                    const double* v = first + 3 * a;
                    enough = list.gather(static_cast<std::int64_t>(neighbors[a]), further, 4) >= 4;
                    if (!enough) {
                        break;
                    }
                    // the neighbors of j other than i: drop the bond back to i
                    int back = 0;
                    double shortest = 0.0;
                    for (int b = 0; b < 4; ++b) {
                        const double* w = further + 3 * b;
                        const double x = v[0] + w[0], y = v[1] + w[1], z = v[2] + w[2];
                        const double d = x * x + y * y + z * z;
                        if (b == 0 || d < shortest) {
                            shortest = d;
                            back = b;
                        }
                    }
                    back_to_i = back_to_i && shortest < kBackBond;
                    for (int b = 0; b < 4; ++b) {
                        if (b != back) {
                            const double* w = further + 3 * b;
                            for (int e = 0; e < 3; ++e) {
                                second[3 * m + e] = v[e] + w[e];
                            }
                            ++m;
                        }
                    }
                }
                if (!enough) {
                    types[i] = kNotDiamond;
                    continue;
                }
                double sum = 0.0;
                for (int a = 0; a < 12; ++a) {
                    sum += length(second + 3 * a);
                }
                const double local = scale * sum / 12.0;
                const Structure type = back_to_i ? twelve(second, local * local) : kOther;
                types[i] = type == kFcc   ? kCubicDiamond
                           : type == kHcp ? kHexagonalDiamond
                                          : kNotDiamond;
            }
        },
        control);
    // the neighbors of diamond atoms, then theirs, in the order of the atoms:
    // the first atom to reach one decides its type
    const auto mark = [&](std::int8_t cubic, std::int8_t hexagonal, std::int8_t cubic_mark,
                          std::int8_t hexagonal_mark) {
        double unused[4 * 3];
        index_t neighbors[4];
        for (std::int64_t i = 0; i < n; ++i) {
            if (types[i] != cubic && types[i] != hexagonal) {
                continue;
            }
            const std::int64_t m = std::min<std::int64_t>(list.gather(i, unused, 4, neighbors), 4);
            for (std::int64_t a = 0; a < m; ++a) {
                std::int8_t& t = types[neighbors[a]];
                if (t == kNotDiamond) {
                    t = types[i] == cubic ? cubic_mark : hexagonal_mark;
                }
            }
        }
    };
    mark(kCubicDiamond, kHexagonalDiamond, kCubicFirst, kHexagonalFirst);
    mark(kCubicFirst, kHexagonalFirst, kCubicSecond, kHexagonalSecond);
}

void chill_plus(const neighbor::ListView& list, std::int64_t n, std::int8_t* types,
                const Control& control) {
    // the sums of the real l = 3 harmonics over each molecule's bonds
    std::vector<double> q(static_cast<std::size_t>(n) * 7, 0.0);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            double y[7];
            for (std::int64_t i = begin; i < end; ++i) {
                double* qi = &q[static_cast<std::size_t>(i) * 7];
                list.for_each(i, [&](std::int64_t, index_t, const double* v) {
                    const double r = length(v);
                    const double u[3] = {v[0] / r, v[1] / r, v[2] / r};
                    analysis::real_harmonics(3, u, y);
                    for (int m = 0; m < 7; ++m) {
                        qi[m] += y[m];
                    }
                });
            }
        },
        control);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const double* qi = &q[static_cast<std::size_t>(i) * 7];
                double own = 0.0;
                for (int m = 0; m < 7; ++m) {
                    own += qi[m] * qi[m];
                }
                int eclipsed = 0, staggered = 0;
                const std::int64_t bonds = list.for_each(i, [&](std::int64_t, index_t j,
                                                                const double*) {
                    const double* qj = &q[static_cast<std::size_t>(j) * 7];
                    double product = 0.0, other = 0.0;
                    for (int m = 0; m < 7; ++m) {
                        product += qi[m] * qj[m];
                        other += qj[m] * qj[m];
                    }
                    const double scale = std::sqrt(own) * std::sqrt(other);
                    const double c = scale > 0.0 ? product / scale : 0.0;
                    eclipsed += c > -0.35 && c < 0.25;
                    staggered += c < -0.8;
                });
                Water type = kLiquid;
                if (bonds == 4) {
                    if (eclipsed == 4) {
                        type = kHydrate;
                    } else if (eclipsed == 3) {
                        type = kInterfacialHydrate;
                    } else if (staggered == 4) {
                        type = kCubicIce;
                    } else if (staggered == 3 && eclipsed == 1) {
                        type = kHexagonalIce;
                    } else if (staggered == 3 || staggered == 2) {
                        type = kInterfacialIce;
                    }
                }
                types[i] = type;
            }
        },
        control);
}

}  // namespace mdapy::identify
