// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The cell and its reciprocal, the FFT grids, the spheres of plane waves and the
// k-points. Hartree atomic units: lengths in bohr, energies in hartree. A plane
// wave of a k-point is |k + G> with G = m1 b1 + m2 b2 + m3 b3, its kinetic
// energy |k + G|^2 / 2; a sphere holds those of kinetic energy up to a cutoff.
// k-points are given in fractions of the reciprocal vectors.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mdapy::dft {

using Vec3 = std::array<double, 3>;
using Int3 = std::array<int, 3>;

struct Lattice {
    std::array<Vec3, 3> a{};  // the cell vectors, rows, bohr
    std::array<Vec3, 3> b{};  // the reciprocal vectors, a_i . b_j = 2 pi delta_ij, 1/bohr
    double volume = 0.0;      // bohr^3

    // From the three cell vectors, rows of nine numbers; throws for a cell
    // without volume.
    static Lattice from_rows(const double* rows);

    // k + G in Cartesian coordinates, for k in fractions of b and G of the
    // Miller indices m.
    Vec3 cartesian(const Vec3& k, const Int3& m) const;
};

// The smallest size at least n whose only prime factors are 2, 3, 5 and 7.
int good_size(int n);

// The FFT grid that holds every G of kinetic energy up to cutoff, |G|^2/2 <=
// cutoff, without aliasing: along each vector 2 max|m_i| + 1 points or more, of
// good sizes. For the density of norm-conserving pseudopotentials the cutoff is
// four times that of the wavefunctions.
Int3 fft_grid(const Lattice& lattice, double cutoff);

// The plane waves of a k-point up to a kinetic energy, in the order of their
// kinetic energy and then of their Miller indices, with their place in an FFT
// box of the given size: index ((m1 mod n1) n2 + (m2 mod n2)) n3 + (m3 mod n3).
// With half, at Gamma only, the half of the sphere that real wavefunctions keep,
// G with m3 > 0, or m3 = 0 and m2 > 0, or m3 = m2 = 0 and m1 >= 0, placed in the
// box of a real transform, of n3 / 2 + 1 along the third vector; partner is
// then the place of -G for the G of m3 = 0, whose coefficient the real
// transform needs too, and -1 for the others. Throws when the box is too small.
struct Sphere {
    std::vector<Int3> miller;
    std::vector<double> kinetic;
    std::vector<std::int64_t> box;
    std::vector<std::int64_t> partner;
    std::size_t size() const noexcept { return miller.size(); }
};

Sphere make_sphere(const Lattice& lattice, const Vec3& k, double cutoff, const Int3& grid,
                   bool half);

// The plane waves of the sphere chosen in the cell basis, if given, with their
// kinetic energies in lattice, in that order: a fixed set of Miller indices
// as the cell strains, for differences of the energy at a fixed basis.
Sphere make_sphere(const Lattice& lattice, const Lattice* basis, const Vec3& k, double cutoff,
                   const Int3& grid, bool half);

// The k-points of a grid of n points along each reciprocal vector, shifted by
// offset of the spacing, (j + offset) / n, folded into [-1/2, 1/2): offset 0 is
// centred on Gamma. With time reversal each k and -k (modulo a reciprocal
// vector) are one point of twice the weight. The weights add up to one.
struct KPoints {
    std::vector<Vec3> k;
    std::vector<double> weight;
};

KPoints monkhorst_pack(const Int3& n, const Vec3& offset, bool time_reversal);

// The points of a Gamma-centred grid of spacing at most spacing, in 1/bohr
// with the factor 2 pi, as VASP's KSPACING: max(1, ceil(|b_i| / spacing)).
Int3 grid_of_spacing(const Lattice& lattice, double spacing);

}  // namespace mdapy::dft
