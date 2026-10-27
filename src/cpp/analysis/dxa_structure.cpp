// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/dxa_structure.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "analysis/dxa_clusters.h"
#include "analysis/dxa_math.h"
#include "identify/cna.h"
#include "identify/ptm.h"
#include "neighbor/frame.h"
#include "neighbor/kd_tree.h"

namespace mdapy::analysis::dxa {
namespace {

using identify::Signature;
using identify::signature;

// halfway between the first two shells of fcc, in units of the first
constexpr double kHalfway = 1.2071068;

// Squared length below which a first and a second neighbor vector lead back
// to the atom itself, in Å².
constexpr double kBack = 1e-8;

// The CNA signature of a bond as one number.
int code(const Signature& s) { return 100 * s.common + 10 * s.bonds + s.chain; }

bool diamond(int type) {
    return type == identify::kPtmCubicDiamond || type == identify::kPtmHexagonalDiamond;
}

// The first template of a lattice with the bonds among its neighbors and
// the signature of each.
struct Pattern {
    int structure = 0;
    int neighbors = 0;
    std::array<Vec, 16> vectors{};
    std::array<std::uint32_t, 16> bonds{};
    std::array<int, 16> codes{};
};

Pattern make_pattern(int type) {
    const Lattice* l = lattice(type);
    Pattern p;
    p.structure = type;
    p.neighbors = l->neighbors;
    p.vectors = l->templates[0];
    const int m = p.neighbors;
    auto length = [&](const Vec& v) { return std::sqrt(dot(v, v)); };
    auto at = [&](int a) { return p.vectors[static_cast<std::size_t>(a)]; };
    // the first shell, and the second of diamond
    double first = length(at(0)), second = 0.0;
    for (int a = 1; a < m; ++a) {
        first = std::min(first, length(at(a)));
    }
    for (int a = 0; a < m; ++a) {
        if (length(at(a)) > first + kTolerance && (second == 0.0 || length(at(a)) < second)) {
            second = length(at(a));
        }
    }
    auto in_first = [&](int a) { return length(at(a)) < first + kTolerance; };
    for (int a = 0; a < m; ++a) {
        for (int b = a + 1; b < m; ++b) {
            const double d = length(minus(at(a), at(b)));
            bool bond = false;
            if (diamond(type)) {
                // a first neighbor to the second ones beyond it, the
                // second ones among themselves
                if (in_first(a) != in_first(b)) {
                    bond = std::abs(d - first) < kTolerance;
                } else if (!in_first(a)) {
                    bond = d <= kHalfway * second;
                }
            } else {
                bond = d <= kHalfway * (type == identify::kPtmBcc ? 2 / std::sqrt(3.0) : 1.0) * first;
            }
            if (bond) {
                p.bonds[static_cast<std::size_t>(a)] |= 1u << b;
                p.bonds[static_cast<std::size_t>(b)] |= 1u << a;
            }
        }
    }
    for (int a = 0; a < m; ++a) {
        p.codes[static_cast<std::size_t>(a)] = code(signature(p.bonds.data(), a));
    }
    return p;
}

const Pattern& pattern(int type) {
    static const std::array<Pattern, 5> patterns = {
        make_pattern(identify::kPtmFcc), make_pattern(identify::kPtmHcp),
        make_pattern(identify::kPtmBcc), make_pattern(identify::kPtmCubicDiamond),
        make_pattern(identify::kPtmHexagonalDiamond)};
    for (const Pattern& p : patterns) {
        if (p.structure == type) {
            return p;
        }
    }
    throw std::invalid_argument("no lattice for this structure");
}

// The neighbors in the order of the template: each takes the place of a
// point of the template with the same signature and the same bonds to the
// places before it, and the order turns the template by a rotation, not a
// reflection. Whether one was found.
bool order(const Pattern& p, const double* v, const std::uint32_t* bonds, const int* codes,
           int* places) {
    const int m = p.neighbors;
    std::array<int, 16> tried{};  // the next neighbor to try at each place
    std::uint32_t used = 0;
    int k = 0;
    tried[0] = 0;
    while (k >= 0) {
        if (k == m) {
            // the matrix from the template to the neighbors, by its sign
            Mat f{};
            for (int s = 0; s < m; ++s) {
                const double* a = v + 3 * places[s];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        f[static_cast<std::size_t>(3 * i + j)] +=
                            a[i] * p.vectors[static_cast<std::size_t>(s)][static_cast<std::size_t>(j)];
                    }
                }
            }
            if (determinant(f) > 0) {
                return true;
            }
            --k;
            used &= ~(1u << places[k]);
            continue;
        }
        int found = -1;
        for (int a = tried[static_cast<std::size_t>(k)]; a < m && found < 0; ++a) {
            if ((used >> a) & 1u || codes[a] != p.codes[static_cast<std::size_t>(k)]) {
                continue;
            }
            bool fits = true;
            for (int s = 0; s < k && fits; ++s) {
                fits = ((bonds[a] >> places[s]) & 1u) == ((p.bonds[static_cast<std::size_t>(k)] >> s) & 1u);
            }
            if (fits) {
                found = a;
            }
        }
        if (found < 0) {
            --k;
            if (k >= 0) {
                used &= ~(1u << places[k]);
            }
            continue;
        }
        places[k] = found;
        used |= 1u << found;
        tried[static_cast<std::size_t>(k)] = found + 1;
        ++k;
        if (k < m) {
            tried[static_cast<std::size_t>(k)] = 0;
        }
    }
    return false;
}

void bond_masks(const double* v, int m, double cutoff2, std::uint32_t* bonds) {
    for (int a = 0; a < m; ++a) {
        bonds[a] = 0;
    }
    for (int a = 0; a < m; ++a) {
        for (int b = a + 1; b < m; ++b) {
            const double dx = v[3 * a] - v[3 * b];
            const double dy = v[3 * a + 1] - v[3 * b + 1];
            const double dz = v[3 * a + 2] - v[3 * b + 2];
            if (dx * dx + dy * dy + dz * dz <= cutoff2) {
                bonds[a] |= 1u << b;
                bonds[b] |= 1u << a;
            }
        }
    }
}

double length(const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

// The type of an atom from the vectors to its neighbors and their bonds,
// and the signature of each bond; 0 for other.
int classify(int crystal, bool defects, const std::uint32_t* bonds, int m, int* codes) {
    using namespace identify;
    if (diamond(crystal)) {
        for (int a = 0; a < 4; ++a) {
            codes[a] = code(signature(bonds, a));
            if (count(bonds[a]) != 3) {
                return 0;
            }
        }
        int n543 = 0, n544 = 0;
        for (int a = 4; a < m; ++a) {
            const Signature s = signature(bonds, a);
            codes[a] = code(s);
            if (s.common != 5 || s.bonds != 4) {
                return 0;
            }
            if (s.chain == 3) {
                ++n543;
            } else if (s.chain == 4) {
                ++n544;
            } else {
                return 0;
            }
        }
        if (n543 == 12 && (defects || crystal == kPtmCubicDiamond)) {
            return kPtmCubicDiamond;
        }
        return n543 == 6 && n544 == 6 && (defects || crystal == kPtmHexagonalDiamond)
                   ? kPtmHexagonalDiamond
                   : 0;
    }
    if (crystal == kPtmBcc) {
        int n444 = 0, n666 = 0;
        for (int a = 0; a < m; ++a) {
            const Signature s = signature(bonds, a);
            codes[a] = code(s);
            if (s.common == 4 && s.bonds == 4 && s.chain == 4) {
                ++n444;
            } else if (s.common == 6 && s.bonds == 6 && s.chain == 6) {
                ++n666;
            } else {
                return 0;
            }
        }
        return n444 == 6 && n666 == 8 ? kPtmBcc : 0;
    }
    int n421 = 0, n422 = 0;
    for (int a = 0; a < m; ++a) {
        const Signature s = signature(bonds, a);
        codes[a] = code(s);
        if (s.common != 4 || s.bonds != 2) {
            return 0;
        }
        if (s.chain == 1) {
            ++n421;
        } else if (s.chain == 2) {
            ++n422;
        } else {
            return 0;
        }
    }
    if (n421 == 12 && (defects || crystal == kPtmFcc)) {
        return kPtmFcc;
    }
    return n421 == 6 && n422 == 6 && (defects || crystal == kPtmHcp) ? kPtmHcp : 0;
}

}  // namespace

Structures find_structures(const double* positions, std::int64_t n, const double* matrix,
                           const double* origin, const bool* pbc, int crystal, bool defects,
                           const Control& control) {
    if (lattice(crystal) == nullptr) {
        throw std::invalid_argument("no lattice for this structure");
    }
    Structures out;
    out.structure.assign(static_cast<std::size_t>(n), 0);
    out.neighbors.assign(static_cast<std::size_t>(16 * n), -1);
    if (n == 0) {
        return out;
    }
    const neighbor::KdTree tree(positions, n, matrix, origin, pbc);
    // classifies atom i from its m neighbors (vectors v, atoms j) and their
    // bonds, and orders them
    auto finish = [&](std::int64_t i, const double* v, const index_t* j, const std::uint32_t* bonds,
                      int m) {
        int codes[16];
        const int type = classify(crystal, defects, bonds, m, codes);
        if (type == 0) {
            return;
        }
        int places[16];
        if (!order(pattern(type), v, bonds, codes, places)) {
            return;
        }
        out.structure[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(type);
        for (int k = 0; k < m; ++k) {
            out.neighbors[static_cast<std::size_t>(16 * i + k)] = j[places[k]];
        }
    };
    if (!diamond(crystal)) {
        const int m = crystal == identify::kPtmBcc ? 14 : 12;
        tree.for_each_nearest(
            m + 1,
            [&](std::int64_t i, const neighbor::Neighbor* found) {
                double v[3 * 15];
                index_t j[15];
                const double* xi = positions + 3 * i;
                for (int a = 0; a <= m; ++a) {
                    neighbor::pair_vector(xi, positions + 3 * static_cast<std::int64_t>(found[a].j),
                                          found[a].shift, matrix, v + 3 * a);
                    j[a] = found[a].j;
                }
                double scale = 0.0;
                const int first = crystal == identify::kPtmBcc ? 8 : 12;
                for (int a = 0; a < first; ++a) {
                    scale += length(v + 3 * a);
                }
                scale /= first;
                const double cutoff =
                    kHalfway * scale * (crystal == identify::kPtmBcc ? 2 / std::sqrt(3.0) : 1.0);
                if (length(v + 3 * m) <= cutoff) {
                    return;
                }
                std::uint32_t bonds[16];
                bond_masks(v, m, cutoff * cutoff, bonds);
                finish(i, v, j, bonds, m);
            },
            control);
        return out;
    }
    // diamond: the four nearest of every atom first
    std::vector<double> near(static_cast<std::size_t>(12 * n));
    std::vector<index_t> near_atoms(static_cast<std::size_t>(4 * n));
    tree.for_each_nearest(
        4,
        [&](std::int64_t i, const neighbor::Neighbor* found) {
            const double* xi = positions + 3 * i;
            for (int a = 0; a < 4; ++a) {
                neighbor::pair_vector(xi, positions + 3 * static_cast<std::int64_t>(found[a].j),
                                      found[a].shift, matrix, near.data() + 12 * i + 3 * a);
                near_atoms[static_cast<std::size_t>(4 * i + a)] = found[a].j;
            }
        },
        control);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                double v[3 * 16];
                index_t j[16];
                std::uint32_t bonds[16] = {};
                int m = 4;
                bool fits = true;
                for (int a = 0; a < 4 && fits; ++a) {
                    const double* va = near.data() + 12 * i + 3 * a;
                    const index_t ja = near_atoms[static_cast<std::size_t>(4 * i + a)];
                    for (int e = 0; e < 3; ++e) {
                        v[3 * a + e] = va[e];
                    }
                    j[a] = ja;
                    for (int b = 0; b < 4; ++b) {
                        const double* w = near.data() + 12 * static_cast<std::int64_t>(ja) + 3 * b;
                        const double x[3] = {va[0] + w[0], va[1] + w[1], va[2] + w[2]};
                        if (near_atoms[static_cast<std::size_t>(4 * static_cast<std::int64_t>(ja) + b)] == i &&
                            x[0] * x[0] + x[1] * x[1] + x[2] * x[2] < kBack) {
                            continue;
                        }
                        if (m == 16) {
                            fits = false;
                            break;
                        }
                        for (int e = 0; e < 3; ++e) {
                            v[3 * m + e] = x[e];
                        }
                        j[m] = near_atoms[static_cast<std::size_t>(4 * static_cast<std::int64_t>(ja) + b)];
                        bonds[a] |= 1u << m;
                        bonds[m] |= 1u << a;
                        ++m;
                    }
                    fits = fits && m == 3 * a + 7;
                }
                if (!fits) {
                    continue;
                }
                double scale = 0.0;
                for (int a = 4; a < 16; ++a) {
                    scale += length(v + 3 * a);
                }
                const double cutoff = kHalfway * scale / 12;
                for (int a = 4; a < 16; ++a) {
                    for (int b = a + 1; b < 16; ++b) {
                        const double dx = v[3 * a] - v[3 * b];
                        const double dy = v[3 * a + 1] - v[3 * b + 1];
                        const double dz = v[3 * a + 2] - v[3 * b + 2];
                        if (dx * dx + dy * dy + dz * dz <= cutoff * cutoff) {
                            bonds[a] |= 1u << b;
                            bonds[b] |= 1u << a;
                        }
                    }
                }
                finish(i, v, j, bonds, 16);
            }
        },
        control);
    return out;
}

}  // namespace mdapy::analysis::dxa
