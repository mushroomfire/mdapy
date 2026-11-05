// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "build/polycrystal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mdapy::build {
namespace {

double dot(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The rows of copies of the unit in one grain: the copies with the same first
// two indices. The faces of the cell are planes, so that the centers of a row
// inside them are one range of the third index, found without placing them. A
// center within a thin band of a face, where rounding or the tolerance could
// decide, is tested alone.
class Rows {
public:
    explicit Rows(const GrainFill& fill)
        : fill_(fill),
          first_out_(static_cast<std::size_t>(fill.centers)),
          last_out_(static_cast<std::size_t>(fill.centers)),
          first_in_(static_cast<std::size_t>(fill.centers)),
          last_in_(static_cast<std::size_t>(fill.centers)),
          first_deep_(static_cast<std::size_t>(fill.centers)),
          last_deep_(static_cast<std::size_t>(fill.centers)),
          taken_(static_cast<std::size_t>(fill.centers)),
          edges_(static_cast<std::size_t>(fill.centers)),
          site_shift_(static_cast<std::size_t>(3 * fill.sites)),
          center_shift_(static_cast<std::size_t>(3 * fill.centers)) {}

    void enter(std::int64_t g) {
        g_ = g;
        seed_ = fill_.seeds + 3 * g;
        const double* rotation = fill_.rotations + 9 * g;
        for (int v = 0; v < 3; ++v) {
            const double* a = fill_.cell + 3 * v;
            for (int e = 0; e < 3; ++e) {
                lattice_[v][e] = dot(rotation + 3 * e, a);
            }
        }
        double origin[3];
        for (int e = 0; e < 3; ++e) {
            origin[e] = dot(rotation + 3 * e, fill_.origin);
        }
        const auto shifts = [&](std::int64_t count, const double* fractions, std::vector<double>& out) {
            for (std::int64_t s = 0; s < count; ++s) {
                for (int e = 0; e < 3; ++e) {
                    out[static_cast<std::size_t>(3 * s + e)] =
                        fractions[3 * s] * lattice_[0][e] + fractions[3 * s + 1] * lattice_[1][e] +
                        fractions[3 * s + 2] * lattice_[2][e] - origin[e];
                }
            }
        };
        shifts(fill_.sites, fill_.site_positions, site_shift_);
        shifts(fill_.centers, fill_.center_positions, center_shift_);

        faces_ = fill_.faces + 3 * fill_.offsets[g];
        planes_ = fill_.offsets[g + 1] - fill_.offsets[g];
        const auto planes = static_cast<std::size_t>(planes_);
        half_.resize(planes);
        band_.resize(planes);
        margin_.resize(planes);
        rate0_.resize(planes);
        rate1_.resize(planes);
        rate2_.resize(planes);
        over_.resize(planes);
        room_.resize(planes);
        at_center_.resize(planes * static_cast<std::size_t>(fill_.centers));
        for (std::int64_t p = 0; p < planes_; ++p) {
            const auto q = static_cast<std::size_t>(p);
            double normal[3] = {faces_[3 * p], faces_[3 * p + 1], faces_[3 * p + 2]};
            const double length2 = dot(normal, normal);
            half_[q] = 0.5 * length2;
            // |x| - |x - n| <= tolerance gives x . n <= n . n / 2 + tolerance (|x| + |n| / 2)
            band_[q] = fill_.tolerance * (fill_.radii[g] + 0.5 * std::sqrt(length2));
            if (fill_.axis >= 0) {
                normal[fill_.axis] = 0.0;  // across the axis only; the length stays that of the face
            }
            margin_[q] = fill_.margin * std::sqrt(dot(normal, normal));
            rate0_[q] = dot(lattice_[0], normal);
            rate1_[q] = dot(lattice_[1], normal);
            rate2_[q] = dot(lattice_[2], normal);
            over_[q] = rate2_[q] != 0.0 ? 1.0 / rate2_[q] : 0.0;
            for (std::int64_t a = 0; a < fill_.centers; ++a) {
                at_center_[static_cast<std::size_t>(a) * planes + q] =
                    dot(center_shift_.data() + 3 * a, normal);
            }
        }
    }

    // The atoms and the copies with atoms of a row, in order: visit(k) for each
    // copy k, with taken(center) telling the centers it takes.
    template <class Visit>
    void walk(std::int64_t c0, std::int64_t c1, Visit&& visit) {
        clip(c0, c1);
        for (std::int64_t k = first_; k <= last_; ++k) {
            bool any = false;
            for (std::int64_t a = 0; a < fill_.centers; ++a) {
                const auto b = static_cast<std::size_t>(a);
                bool in = k >= first_in_[b] && k <= last_in_[b];
                if (!in && k >= first_out_[b] && k <= last_out_[b]) {
                    in = closest(c0, c1, k, a);
                }
                taken_[b] = in;
                any = any || in;
                edges_[b] = k < first_deep_[b] || k > last_deep_[b];
            }
            if (any) {
                visit(k);
            }
        }
    }

    bool taken(std::int64_t site) const {
        return taken_[static_cast<std::size_t>(fill_.centers_of[site])] != 0;
    }

    // Writes the place of a site in copy (c0, c1, k).
    void place(std::int64_t c0, std::int64_t c1, std::int64_t k, std::int64_t site, double* out) const {
        const double* shift = site_shift_.data() + 3 * site;
        for (int e = 0; e < 3; ++e) {
            out[e] = seed_[e] + static_cast<double>(c0) * lattice_[0][e] +
                     static_cast<double>(c1) * lattice_[1][e] +
                     static_cast<double>(k) * lattice_[2][e] + shift[e];
        }
    }

    std::int64_t grain() const { return g_; }

    // Whether the center of a site just taken is within the margin of a face.
    bool edge(std::int64_t site) const {
        return edges_[static_cast<std::size_t>(fill_.centers_of[site])] != 0;
    }

private:
    void clip(std::int64_t c0, std::int64_t c1) {
        const std::int64_t span = fill_.spans[3 * g_ + 2];
        const auto planes = static_cast<std::size_t>(planes_);
        const auto x0 = static_cast<double>(c0), x1 = static_cast<double>(c1);
        for (std::size_t p = 0; p < planes; ++p) {
            room_[p] = half_[p] - x0 * rate0_[p] - x1 * rate1_[p];
        }
        first_ = span + 1;
        last_ = -span - 1;
        for (std::size_t a = 0; a < static_cast<std::size_t>(fill_.centers); ++a) {
            const double* at = at_center_.data() + a * planes;
            auto low_out = static_cast<double>(-span), high_out = static_cast<double>(span);
            double low_in = low_out, high_in = high_out;
            double low_deep = low_out, high_deep = high_out;
            bool empty = false;
            for (std::size_t p = 0; p < planes; ++p) {
                const double room = room_[p] - at[p];
                if (rate2_[p] > 0.0) {
                    const double place = room * over_[p], wide = band_[p] * over_[p];
                    high_out = std::min(high_out, place + wide);
                    high_in = std::min(high_in, place - wide);
                    high_deep = std::min(high_deep, place - margin_[p] * over_[p]);
                } else if (rate2_[p] < 0.0) {
                    const double place = room * over_[p], wide = band_[p] * over_[p];
                    low_out = std::max(low_out, place + wide);
                    low_in = std::max(low_in, place - wide);
                    low_deep = std::max(low_deep, place - margin_[p] * over_[p]);
                } else if (room + band_[p] < 0.0) {
                    empty = true;
                } else {
                    if (room - band_[p] < 0.0) {
                        low_in = 1.0;
                        high_in = 0.0;
                    }
                    if (room - margin_[p] < 0.0) {
                        low_deep = 1.0;
                        high_deep = 0.0;
                    }
                }
                if (empty || low_out > high_out) {
                    empty = true;
                    break;
                }
            }
            if (empty) {
                first_out_[a] = first_in_[a] = first_deep_[a] = 1;
                last_out_[a] = last_in_[a] = last_deep_[a] = 0;
                continue;
            }
            first_out_[a] = static_cast<std::int64_t>(std::ceil(low_out));
            last_out_[a] = static_cast<std::int64_t>(std::floor(high_out));
            first_in_[a] = static_cast<std::int64_t>(std::ceil(low_in));
            last_in_[a] = static_cast<std::int64_t>(std::floor(high_in));
            first_deep_[a] = static_cast<std::int64_t>(std::ceil(low_deep));
            last_deep_[a] = static_cast<std::int64_t>(std::floor(high_deep));
            if (first_out_[a] <= last_out_[a]) {
                first_ = std::min(first_, first_out_[a]);
                last_ = std::max(last_, last_out_[a]);
            }
        }
    }

    // Whether a center is no farther from the seed than from the seeds across
    // the faces, up to the tolerance.
    bool closest(std::int64_t c0, std::int64_t c1, std::int64_t k, std::int64_t a) const {
        double x[3];
        for (int e = 0; e < 3; ++e) {
            x[e] = static_cast<double>(c0) * lattice_[0][e] + static_cast<double>(c1) * lattice_[1][e] +
                   static_cast<double>(k) * lattice_[2][e] + center_shift_[static_cast<std::size_t>(3 * a + e)];
        }
        if (fill_.axis >= 0) {
            x[fill_.axis] = 0.0;
        }
        const double own = std::sqrt(dot(x, x));
        for (std::int64_t p = 0; p < planes_; ++p) {
            const double d[3] = {x[0] - faces_[3 * p], x[1] - faces_[3 * p + 1], x[2] - faces_[3 * p + 2]};
            if (own > std::sqrt(dot(d, d)) + fill_.tolerance) {
                return false;
            }
        }
        return true;
    }

    const GrainFill& fill_;
    std::int64_t g_ = 0;
    const double* seed_ = nullptr;
    const double* faces_ = nullptr;
    std::int64_t planes_ = 0;
    std::int64_t first_ = 0, last_ = -1;
    double lattice_[3][3] = {};
    std::vector<std::int64_t> first_out_, last_out_, first_in_, last_in_, first_deep_, last_deep_;
    std::vector<char> taken_, edges_;
    std::vector<double> site_shift_, center_shift_;
    std::vector<double> half_, band_, margin_, rate0_, rate1_, rate2_, over_, room_, at_center_;
};

}  // namespace

GrainAtoms fill_grains(const GrainFill& fill, const Control& control) {
    const auto grains = static_cast<std::size_t>(fill.grains);
    std::vector<std::int64_t> first_row(grains + 1, 0);
    for (std::size_t g = 0; g < grains; ++g) {
        const std::int64_t* span = fill.spans + 3 * g;
        first_row[g + 1] = first_row[g] + (2 * span[0] + 1) * (2 * span[1] + 1);
    }
    const std::int64_t rows = first_row[grains];

    // the rows of a chunk, with their indices; a grain has at least one row
    const auto each_row = [&](Rows& grain_rows, std::int64_t begin, std::int64_t end, auto&& body) {
        auto g = static_cast<std::size_t>(
            std::upper_bound(first_row.begin(), first_row.end(), begin) - first_row.begin() - 1);
        for (std::int64_t row = begin; row < end; ++g) {
            grain_rows.enter(static_cast<std::int64_t>(g));
            const std::int64_t* span = fill.spans + 3 * g;
            const std::int64_t width = 2 * span[1] + 1;
            const std::int64_t stop = std::min(end, first_row[g + 1]);
            for (; row < stop; ++row) {
                const std::int64_t local = row - first_row[g];
                body(row, local / width - span[0], local % width - span[1]);
            }
        }
    };

    // how many atoms and copies come before each row
    Buffer<std::int64_t> atoms_before(static_cast<std::size_t>(rows + 1));
    Buffer<std::int64_t> copies_before(static_cast<std::size_t>(rows + 1));
    atoms_before[0] = copies_before[0] = 0;
    parallel_for(
        rows, 0,
        [&](std::int64_t begin, std::int64_t end) {
            Rows grain_rows(fill);
            each_row(grain_rows, begin, end, [&](std::int64_t row, std::int64_t c0, std::int64_t c1) {
                std::int64_t atoms = 0, copies = 0;
                grain_rows.walk(c0, c1, [&](std::int64_t) {
                    for (std::int64_t s = 0; s < fill.sites; ++s) {
                        atoms += grain_rows.taken(s) ? 1 : 0;
                    }
                    ++copies;
                });
                atoms_before[static_cast<std::size_t>(row + 1)] = atoms;
                copies_before[static_cast<std::size_t>(row + 1)] = copies;
            });
        },
        stage(control, 0, 2));
    prefix_sum(atoms_before.data(), rows + 1);
    prefix_sum(copies_before.data(), rows + 1);

    const auto total = static_cast<std::size_t>(atoms_before[static_cast<std::size_t>(rows)]);
    GrainAtoms result;
    result.positions.resize(3 * total);
    result.grain.resize(total);
    result.site.resize(total);
    if (fill.copies) {
        result.copy.resize(total);
    }
    if (fill.margin >= 0.0) {
        result.edge.resize(total);
    }
    parallel_for(
        rows, 0,
        [&](std::int64_t begin, std::int64_t end) {
            Rows grain_rows(fill);
            each_row(grain_rows, begin, end, [&](std::int64_t row, std::int64_t c0, std::int64_t c1) {
                auto at = static_cast<std::size_t>(atoms_before[static_cast<std::size_t>(row)]);
                std::int64_t copy = copies_before[static_cast<std::size_t>(row)];
                grain_rows.walk(c0, c1, [&](std::int64_t k) {
                    for (std::int64_t s = 0; s < fill.sites; ++s) {
                        if (!grain_rows.taken(s)) {
                            continue;
                        }
                        grain_rows.place(c0, c1, k, s, result.positions.data() + 3 * at);
                        result.grain[at] = grain_rows.grain();
                        result.site[at] = s;
                        if (fill.copies) {
                            result.copy[at] = copy;
                        }
                        if (fill.margin >= 0.0) {
                            result.edge[at] = grain_rows.edge(s) ? 1 : 0;
                        }
                        ++at;
                    }
                    ++copy;
                });
            });
        },
        stage(control, 1, 2));
    return result;
}

}  // namespace mdapy::build
