// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "build/crystal.h"

#include <algorithm>
#include <cstddef>

namespace mdapy::build {

CrystalAtoms fill_crystal(const CrystalFill& fill, const Control& control) {
    const std::int64_t n1 = fill.counts[0], n2 = fill.counts[1], n3 = fill.counts[2];
    const std::int64_t* plane_rows = fill.plane_rows;
    const std::int64_t* row_atoms = fill.row_atoms;
    const std::int64_t rows = plane_rows[fill.planes];
    const std::int64_t layer = row_atoms[rows] * n1 * n2;  // the atoms of one cell's height

    CrystalAtoms atoms;
    atoms.positions.resize(3 * static_cast<std::size_t>(layer * n3));
    atoms.kind.resize(static_cast<std::size_t>(layer * n3));
    double* positions = atoms.positions.data();
    std::uint32_t* kind = atoms.kind.data();

    // An item is one row in one cell. A small crystal stays on the calling thread.
    const std::int64_t items = rows * n1 * n2 * n3;
    const std::int64_t grain = std::max<std::int64_t>(default_grain(items), 4096);
    parallel_for(
        items, grain,
        [&](std::int64_t begin, std::int64_t end) {
            // where this chunk starts; each next item costs no division
            const std::int64_t line = begin / n1;
            std::int64_t k1 = begin % n1;
            std::int64_t k3 = line / (rows * n2);
            std::int64_t rest = line % (rows * n2);
            std::int64_t p = 0;
            while (rest >= n2 * (plane_rows[p + 1] - plane_rows[p])) {
                rest -= n2 * (plane_rows[p + 1] - plane_rows[p]);
                ++p;
            }
            std::int64_t k2 = rest / (plane_rows[p + 1] - plane_rows[p]);
            std::int64_t r = plane_rows[p] + rest % (plane_rows[p + 1] - plane_rows[p]);
            for (std::int64_t item = begin; item < end; ++item) {
                const std::int64_t before = row_atoms[plane_rows[p]];
                const std::int64_t plane = row_atoms[plane_rows[p + 1]] - before;
                std::int64_t i = k3 * layer + before * n1 * n2 + k2 * plane * n1 +
                                 (row_atoms[r] - before) * n1 +
                                 k1 * (row_atoms[r + 1] - row_atoms[r]);
                for (std::int64_t a = row_atoms[r]; a < row_atoms[r + 1]; ++a, ++i) {
                    const double x = static_cast<double>(k1) + fill.fractional[3 * a];
                    const double y = static_cast<double>(k2) + fill.fractional[3 * a + 1];
                    const double z = static_cast<double>(k3) + fill.fractional[3 * a + 2];
                    for (int e = 0; e < 3; ++e) {
                        positions[3 * i + e] = x * fill.vectors[e] + y * fill.vectors[3 + e] +
                                               z * fill.vectors[6 + e];
                    }
                    kind[i] = static_cast<std::uint32_t>(fill.kinds[a]);
                }
                if (++k1 == n1) {
                    k1 = 0;
                    if (++r == plane_rows[p + 1]) {
                        if (++k2 == n2) {
                            k2 = 0;
                            if (++p == fill.planes) {
                                p = 0;
                                ++k3;
                            }
                        }
                        r = plane_rows[p];
                    }
                }
            }
        },
        control);
    return atoms;
}

}  // namespace mdapy::build
