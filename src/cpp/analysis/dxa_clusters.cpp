// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/dxa_clusters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <map>
#include <optional>
#include <utility>

#include "analysis/dxa_math.h"
#include "identify/ptm.h"
#include "identify/ptm_data.h"

namespace mdapy::analysis::dxa {
namespace {

Lattice make_lattice(int type) {
    namespace ptm = identify::ptm;
    const ptm::Reference& ref = ptm::reference(type);
    Lattice lattice;
    lattice.structure = type;
    lattice.neighbors = ref.neighbors;
    // the length of the first neighbor in units of the cubic lattice
    // constant; hcp as fcc and hexagonal as cubic diamond, so that the
    // frames of their stackings turn into each other
    double first = 0.0;
    bool cubic = true;
    switch (type) {
        case ptm::kFcc: first = std::sqrt(0.5); break;
        case ptm::kBcc: first = std::sqrt(3.0) / 2; break;
        case ptm::kCubicDiamond: first = std::sqrt(3.0) / 4; break;
        case ptm::kHcp: first = std::sqrt(0.5); cubic = false; break;
        default: first = std::sqrt(3.0) / 4; cubic = false; break;  // hexagonal diamond
    }
    const double scale = first / std::sqrt(dot({ref.templates[0][1][0], ref.templates[0][1][1],
                                                ref.templates[0][1][2]},
                                               {ref.templates[0][1][0], ref.templates[0][1][1],
                                                ref.templates[0][1][2]}));
    for (int k = 0; k < ref.neighbors; ++k) {
        for (int e = 0; e < 3; ++e) {
            const double v = ref.templates[0][static_cast<std::size_t>(k + 1)][e] * scale;
            lattice.vectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(e)] =
                cubic ? std::round(4 * v) / 4 : v;
        }
    }
    auto& t = lattice.vectors;
    const int n = lattice.neighbors;
    // the rotations of the lattice: those that take the neighbors that are
    // translations of the lattice, whose opposites are neighbors too, onto
    // themselves; for hcp and diamond they take the neighbors of an atom
    // onto those of an atom of another site
    auto in = [&](const std::vector<Vec>& set, const Vec& v) {
        return std::any_of(set.begin(), set.end(), [&](const Vec& w) { return close(v, w); });
    };
    std::vector<Vec> translations;
    auto find_translations = [&] {
        translations.clear();
        for (int k = 0; k < n; ++k) {
            if (in({t.begin(), t.begin() + n}, Vec{-t[k][0], -t[k][1], -t[k][2]})) {
                translations.push_back(t[k]);
            }
        }
    };
    find_translations();
    if (!cubic) {
        // the hexagonal lattices turned to c along z and a neighbor in the
        // basal plane along x, a neighbor below the plane at x = 0 toward
        // -y, and the neighbor along the c axis, of hexagonal diamond, up
        Vec normal = cross(translations[0], translations[1]);
        normal = scaled(normal, 1.0 / std::sqrt(dot(normal, normal)));
        bool turned = false;
        for (std::size_t k = 0; k < translations.size() && !turned; ++k) {
            for (const double sign : {1.0, -1.0}) {
                const Vec x = scaled(translations[k], 1.0 / std::sqrt(dot(translations[k], translations[k])));
                const Vec z = scaled(normal, sign);
                const Vec y = cross(z, x);
                std::array<Vec, 16> moved{};
                bool below = false, upright = true;
                for (int j = 0; j < n; ++j) {
                    const Vec& v = t[static_cast<std::size_t>(j)];
                    const Vec w = {dot(x, v), dot(y, v), dot(z, v)};
                    moved[static_cast<std::size_t>(j)] = w;
                    below = below || (w[2] < -kTolerance && std::abs(w[0]) < kTolerance &&
                                      w[1] < -kTolerance);
                    upright = upright && !(std::abs(w[0]) < kTolerance &&
                                           std::abs(w[1]) < kTolerance && w[2] < 0);
                }
                if (below && upright) {
                    t = moved;
                    turned = true;
                    break;
                }
            }
        }
        find_translations();
    }
    const Vec& ta = translations[0];
    std::size_t second = 1;
    while (std::abs(std::abs(dot(ta, translations[second])) -
                    std::sqrt(dot(ta, ta) * dot(translations[second], translations[second]))) <
           kTolerance) {
        ++second;
    }
    const Vec& tb = translations[second];
    const Mat from = *inverse(columns(ta, tb, cross(ta, tb)));
    lattice.group.push_back(kIdentity);
    for (const Vec& ti : translations) {
        for (const Vec& tj : translations) {
            if (std::abs(dot(ti, ti) - dot(ta, ta)) > kTolerance ||
                std::abs(dot(tj, tj) - dot(tb, tb)) > kTolerance ||
                std::abs(dot(ti, tj) - dot(ta, tb)) > kTolerance) {
                continue;
            }
            const Mat r = multiply(columns(ti, tj, cross(ti, tj)), from);
            if (determinant(r) < 0) {
                continue;
            }
            bool onto = true;
            for (std::size_t k = 0; k < translations.size() && onto; ++k) {
                onto = in(translations, turn(r, translations[k]));
            }
            if (onto &&
                std::none_of(lattice.group.begin(), lattice.group.end(),
                             [&](const Mat& g) { return close(g, r); })) {
                lattice.group.push_back(r);
            }
        }
    }
    const std::size_t order = lattice.group.size();
    lattice.product.assign(order, std::vector<std::int16_t>(order, 0));
    for (std::size_t g1 = 0; g1 < order; ++g1) {
        for (std::size_t g2 = 0; g2 < order; ++g2) {
            const Mat m = multiply(lattice.group[g1], lattice.group[g2]);
            for (std::size_t g = 0; g < order; ++g) {
                if (close(m, lattice.group[g])) {
                    lattice.product[g1][g2] = static_cast<std::int16_t>(g);
                }
            }
        }
    }
    // the common neighbors: the vectors between them are neighbor vectors
    // of an atom of some site
    auto neighbor_vector = [&](const Vec& v) {
        return std::any_of(lattice.group.begin(), lattice.group.end(), [&](const Mat& g) {
            return in({t.begin(), t.begin() + n}, turn(transpose(g), v));
        });
    };
    for (int k = 0; k < n; ++k) {
        double best = 0.0;
        for (int j1 = 0; j1 < n; ++j1) {
            const Vec d1 = minus(t[j1], t[k]);
            if (j1 == k || !neighbor_vector(d1)) {
                continue;
            }
            for (int j2 = j1 + 1; j2 < n; ++j2) {
                const Vec d2 = minus(t[j2], t[k]);
                if (j2 == k || !neighbor_vector(d2)) {
                    continue;
                }
                const Vec back = {-t[k][0], -t[k][1], -t[k][2]};
                const double volume = std::abs(dot(back, cross(d1, d2)));
                if (volume > best + kTolerance) {
                    best = volume;
                    lattice.common[static_cast<std::size_t>(k)] = {j1, j2};
                }
            }
        }
    }
    return lattice;
}

}  // namespace

const Lattice* lattice(int structure) {
    namespace ptm = identify::ptm;
    static const std::array<Lattice, 5> lattices = {
        make_lattice(ptm::kFcc), make_lattice(ptm::kHcp), make_lattice(ptm::kBcc),
        make_lattice(ptm::kCubicDiamond), make_lattice(ptm::kHexagonalDiamond)};
    for (const Lattice& l : lattices) {
        if (l.structure == structure) {
            return &l;
        }
    }
    return nullptr;
}

Clusters build_clusters(const ClusterInput& in, const Control& control) {
    const std::int64_t n = in.n;
    Clusters out;
    out.cluster.assign(static_cast<std::size_t>(n), -1);
    out.rotation.assign(static_cast<std::size_t>(n), 0);
    auto lattice_of = [&](std::int64_t i) { return lattice(in.structure[i]); };
    auto neighbor = [&](std::int64_t i, int k) -> std::int64_t { return in.neighbors[16 * i + k]; };
    auto place = [&](std::int64_t i, std::int64_t j) {
        const Lattice* l = lattice_of(i);
        for (int k = 0; k < l->neighbors; ++k) {
            if (neighbor(i, k) == j) {
                return k;
            }
        }
        return -1;
    };
    // the ideal vector to neighbor k of atom i in the frame of its cluster
    auto ideal = [&](std::int64_t i, int k) {
        const Lattice* l = lattice_of(i);
        return turn(l->group[static_cast<std::size_t>(out.rotation[static_cast<std::size_t>(i)])],
                    l->vectors[static_cast<std::size_t>(k)]);
    };
    // the matrix that takes the template of neighbor k of atom a into the
    // frame of a's cluster, from the vectors from it to a and to two
    // neighbors both share
    auto relate = [&](std::int64_t a, int k) -> std::optional<Mat> {
        const Lattice* la = lattice_of(a);
        const std::int64_t b = neighbor(a, k);
        const Lattice* lb = lattice_of(b);
        const int k1 = la->common[static_cast<std::size_t>(k)][0];
        const int k2 = la->common[static_cast<std::size_t>(k)][1];
        const std::int64_t c1 = neighbor(a, k1), c2 = neighbor(a, k2);
        if (lb == nullptr || c1 < 0 || c2 < 0) {
            return std::nullopt;
        }
        const int j0 = place(b, a), j1 = place(b, c1), j2 = place(b, c2);
        if (j0 < 0 || j1 < 0 || j2 < 0) {
            return std::nullopt;
        }
        const Vec ab = ideal(a, k);
        const Mat u = columns({-ab[0], -ab[1], -ab[2]}, minus(ideal(a, k1), ab),
                              minus(ideal(a, k2), ab));
        const auto& tb = lb->vectors;
        const auto w = inverse(columns(tb[static_cast<std::size_t>(j0)],
                                       tb[static_cast<std::size_t>(j1)],
                                       tb[static_cast<std::size_t>(j2)]));
        if (!w) {
            return std::nullopt;
        }
        return multiply(u, *w);
    };
    // the clusters, by flood fill
    for (std::int64_t seed = 0; seed < n; ++seed) {
        if (out.cluster[static_cast<std::size_t>(seed)] >= 0 || lattice_of(seed) == nullptr) {
            continue;
        }
        const auto c = static_cast<std::int64_t>(out.structure.size());
        out.structure.push_back(in.structure[seed]);
        out.size.push_back(1);
        out.cluster[static_cast<std::size_t>(seed)] = c;
        std::deque<std::int64_t> todo = {seed};
        while (!todo.empty()) {
            const std::int64_t a = todo.front();
            todo.pop_front();
            const Lattice* la = lattice_of(a);
            for (int k = 0; k < la->neighbors; ++k) {
                const std::int64_t b = neighbor(a, k);
                if (b < 0 || out.cluster[static_cast<std::size_t>(b)] >= 0 ||
                    in.structure[b] != in.structure[a]) {
                    continue;
                }
                const auto r = relate(a, k);
                if (!r) {
                    continue;
                }
                const auto& group = la->group;
                for (std::size_t g = 0; g < group.size(); ++g) {
                    if (close(*r, group[g])) {
                        out.cluster[static_cast<std::size_t>(b)] = c;
                        out.rotation[static_cast<std::size_t>(b)] = static_cast<std::int16_t>(g);
                        ++out.size[static_cast<std::size_t>(c)];
                        todo.push_back(b);
                        break;
                    }
                }
            }
        }
    }
    const auto count = out.structure.size();
    // the orientations by least squares over the bonds, and the transitions
    std::vector<std::array<double, 18>> sums(count, std::array<double, 18>{});
    std::map<std::pair<std::int64_t, std::int64_t>, std::size_t> found;
    for (std::int64_t a = 0; a < n; ++a) {
        const std::int64_t c1 = out.cluster[static_cast<std::size_t>(a)];
        if (c1 < 0) {
            continue;
        }
        const Lattice* la = lattice_of(a);
        for (int k = 0; k < la->neighbors; ++k) {
            const std::int64_t b = neighbor(a, k);
            if (b < 0) {
                continue;
            }
            const Vec v = ideal(a, k);
            Vec d = minus({in.positions[3 * b], in.positions[3 * b + 1], in.positions[3 * b + 2]},
                          {in.positions[3 * a], in.positions[3 * a + 1], in.positions[3 * a + 2]});
            Vec f = turn(transpose(Mat{in.inverse[0], in.inverse[1], in.inverse[2], in.inverse[3],
                                        in.inverse[4], in.inverse[5], in.inverse[6], in.inverse[7],
                                        in.inverse[8]}),
                          d);
            for (int e = 0; e < 3; ++e) {
                if (in.pbc[e]) {
                    f[static_cast<std::size_t>(e)] -= std::round(f[static_cast<std::size_t>(e)]);
                }
            }
            d = turn(transpose(Mat{in.matrix[0], in.matrix[1], in.matrix[2], in.matrix[3],
                                    in.matrix[4], in.matrix[5], in.matrix[6], in.matrix[7],
                                    in.matrix[8]}),
                      f);
            auto& s = sums[static_cast<std::size_t>(c1)];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    s[static_cast<std::size_t>(3 * i + j)] += v[i] * v[j];
                    s[static_cast<std::size_t>(9 + 3 * i + j)] += d[i] * v[j];
                }
            }
            const std::int64_t c2 = out.cluster[static_cast<std::size_t>(b)];
            if (c2 < 0 || c2 == c1) {
                continue;
            }
            const auto key = std::minmax(c1, c2);
            const auto known = found.find(key);
            if (known != found.end()) {
                ++out.transitions[known->second].area;
                continue;
            }
            const auto r = relate(a, k);
            if (!r) {
                continue;
            }
            // from the frame of c2 into that of c1, then the other way round
            const Lattice* lb = lattice_of(b);
            const Mat to_c1 = multiply(
                *r, transpose(lb->group[static_cast<std::size_t>(
                        out.rotation[static_cast<std::size_t>(b)])]));
            if (!close(multiply(to_c1, transpose(to_c1)), kIdentity)) {
                continue;
            }
            found[key] = out.transitions.size();
            out.transitions.push_back({c1, c2, transpose(to_c1), 1});
        }
    }
    out.orientation.resize(9 * count);
    for (std::size_t c = 0; c < count; ++c) {
        Mat v, w;
        std::copy(sums[c].begin(), sums[c].begin() + 9, v.begin());
        std::copy(sums[c].begin() + 9, sums[c].end(), w.begin());
        const auto vi = inverse(v);
        const Mat o = vi ? multiply(w, *vi) : Mat{};
        std::copy(o.begin(), o.end(), out.orientation.begin() + static_cast<std::ptrdiff_t>(9 * c));
    }
    // the frame of each cluster of the crystal turned by the rotation of
    // the lattice that brings its orientation closest to the axes of the
    // box, so that lattice vectors read as usual in a crystal along them
    const Lattice* crystal = lattice(in.crystal);
    std::vector<std::int16_t> turned(count, 0);
    for (std::size_t c = 0; c < count && crystal != nullptr; ++c) {
        if (out.structure[c] != in.crystal) {
            continue;
        }
        Mat o;
        std::copy(out.orientation.begin() + static_cast<std::ptrdiff_t>(9 * c),
                  out.orientation.begin() + static_cast<std::ptrdiff_t>(9 * c + 9), o.begin());
        const double scale = std::cbrt(std::abs(determinant(o)));
        if (scale == 0.0) {
            continue;
        }
        double smallest = 0.0;
        for (std::size_t g = 0; g < crystal->group.size(); ++g) {
            const Mat m = multiply(o, transpose(crystal->group[g]));
            double deviation = 0.0;
            for (std::size_t e = 0; e < 9; ++e) {
                deviation += std::abs(m[e] / scale - kIdentity[e]);
            }
            if (g == 0 || deviation < smallest - kTolerance) {
                smallest = deviation;
                turned[c] = static_cast<std::int16_t>(g);
            }
        }
        const Mat m = multiply(o, transpose(crystal->group[static_cast<std::size_t>(turned[c])]));
        std::copy(m.begin(), m.end(), out.orientation.begin() + static_cast<std::ptrdiff_t>(9 * c));
    }
    for (std::int64_t a = 0; a < n; ++a) {
        const std::int64_t c = out.cluster[static_cast<std::size_t>(a)];
        if (c >= 0 && turned[static_cast<std::size_t>(c)] != 0) {
            auto& r = out.rotation[static_cast<std::size_t>(a)];
            r = crystal->product[static_cast<std::size_t>(turned[static_cast<std::size_t>(c)])]
                                [static_cast<std::size_t>(r)];
        }
    }
    auto frame = [&](std::int64_t c) {
        const std::int16_t g = turned[static_cast<std::size_t>(c)];
        return g == 0 ? kIdentity : crystal->group[static_cast<std::size_t>(g)];
    };
    for (Transition& t : out.transitions) {
        t.matrix = multiply(frame(t.to), multiply(t.matrix, transpose(frame(t.from))));
    }
    (void)control;
    return out;
}

}  // namespace mdapy::analysis::dxa
