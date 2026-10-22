// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "identify/structure.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <vector>

#include "analysis/steinhardt.h"

namespace mdapy::identify {
namespace {

constexpr int kMaxNeighbors = 14;

// Squared length below which a first and a second neighbor vector lead back
// to the atom itself, in Å².
constexpr double kBackBond = 1e-4;

int count(std::uint32_t bits) {
    return static_cast<int>(std::bitset<32>(bits).count());
}

// The CNA signature of the bond to neighbor j among m neighbors whose
// bonds are given as bitmasks: the number of common neighbors, the number
// of bonds among them and the number of bonds in the largest connected
// group of those bonds.
struct Signature {
    int common, bonds, chain;
};

// The loops below run over all neighbors and mask out those not in a set,
// rather than branch on membership, which the processor cannot predict.
std::uint32_t member(std::uint32_t set, int a) {
    return 0u - ((set >> a) & 1u);  // all bits when a is in the set
}

Signature signature(const std::uint32_t* bonds, int j) {
    const std::uint32_t common = bonds[j];
    int total = 0;
    for (int a = 0; a < kMaxNeighbors; ++a) {
        total += count(bonds[a] & common & member(common, a));
    }
    // connected groups of the common neighbors joined by bonds
    int chain = 0;
    std::uint32_t left = common;
    while (left != 0) {
        std::uint32_t group = left & (~left + 1u);  // lowest remaining neighbor
        std::uint32_t frontier = group;
        while (frontier != 0) {
            std::uint32_t next = 0;
            for (int a = 0; a < kMaxNeighbors; ++a) {
                next |= bonds[a] & common & member(frontier, a);
            }
            frontier = next & ~group;
            group |= next;
        }
        int inside = 0;
        for (int a = 0; a < kMaxNeighbors; ++a) {
            inside += count(bonds[a] & group & member(group, a));
        }
        chain = std::max(chain, inside / 2);
        left &= ~group;
    }
    return {count(common), total / 2, chain};
}

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

}  // namespace

void centro_symmetry(std::int64_t n, int k, const double* vectors, double* csp,
                     const Control& control) {
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> values(static_cast<std::size_t>(k) * (k - 1) / 2);
            for (std::int64_t i = begin; i < end; ++i) {
                const double* v = vectors + 3 * k * i;
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
                std::partial_sort(values.begin(), values.begin() + half, values.end());
                double sum = 0.0;
                for (std::ptrdiff_t a = 0; a < half; ++a) {
                    sum += values[a];
                }
                csp[i] = sum;
            }
        },
        control);
}

void common_neighbor_analysis(std::int64_t n, const std::int64_t* offsets, const double* vectors,
                              double cutoff, std::int8_t* types, const Control& control) {
    // (1 + sqrt 2) / 2 lies halfway between the first and second shells of fcc
    const double scale = 0.5 * (1.0 + std::sqrt(2.0));
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t m = offsets[i + 1] - offsets[i];
                const double* v = vectors + 3 * offsets[i];
                Structure type = kOther;
                if (cutoff > 0.0) {
                    if (m == 12) {
                        type = twelve(v, cutoff * cutoff);
                    } else if (m == 14) {
                        type = fourteen(v, cutoff * cutoff);
                    }
                } else {
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
                        const double local =
                            scale * (2.0 / std::sqrt(3.0) * first + second) / 14.0;
                        type = fourteen(v, local * local);
                    }
                }
                types[i] = type;
            }
        },
        control);
}

void identify_diamond(std::int64_t n, const std::int32_t* neighbors, const double* vectors,
                      std::int8_t* types, const Control& control) {
    const double scale = 0.5 * (1.0 + std::sqrt(2.0));
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            double second[12 * 3];
            for (std::int64_t i = begin; i < end; ++i) {
                int m = 0;
                bool back_to_i = true;  // i is among the four nearest of each neighbor
                for (int a = 0; a < 4; ++a) {
                    const double* v = vectors + 3 * (4 * i + a);
                    const std::int64_t j = neighbors[4 * i + a];
                    // the neighbors of j other than i: drop the bond back to i
                    int back = 0;
                    double shortest = 0.0;
                    for (int b = 0; b < 4; ++b) {
                        const double* w = vectors + 3 * (4 * j + b);
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
                            const double* w = vectors + 3 * (4 * j + b);
                            for (int e = 0; e < 3; ++e) {
                                second[3 * m + e] = v[e] + w[e];
                            }
                            ++m;
                        }
                    }
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
}

void chill_plus(std::int64_t n, const std::int64_t* offsets, const std::int32_t* neighbors,
                const double* vectors, std::int8_t* types, const Control& control) {
    // the sums of the real l = 3 harmonics over each molecule's bonds
    std::vector<double> q(static_cast<std::size_t>(n) * 7, 0.0);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            double y[7];
            for (std::int64_t i = begin; i < end; ++i) {
                double* qi = &q[static_cast<std::size_t>(i) * 7];
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const double* v = vectors + 3 * p;
                    const double r = length(v);
                    const double u[3] = {v[0] / r, v[1] / r, v[2] / r};
                    analysis::real_harmonics(3, u, y);
                    for (int m = 0; m < 7; ++m) {
                        qi[m] += y[m];
                    }
                }
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
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const double* qj = &q[static_cast<std::size_t>(neighbors[p]) * 7];
                    double product = 0.0, other = 0.0;
                    for (int m = 0; m < 7; ++m) {
                        product += qi[m] * qj[m];
                        other += qj[m] * qj[m];
                    }
                    const double scale = std::sqrt(own) * std::sqrt(other);
                    const double c = scale > 0.0 ? product / scale : 0.0;
                    eclipsed += c > -0.35 && c < 0.25;
                    staggered += c < -0.8;
                }
                Water type = kLiquid;
                if (offsets[i + 1] - offsets[i] == 4) {
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
