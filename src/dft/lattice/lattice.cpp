// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/lattice/lattice.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

namespace mdapy::dft {

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

double norm(const Vec3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

// m modulo n, in [0, n)
std::int64_t wrap(int m, int n) {
    const int r = m % n;
    return r < 0 ? r + n : r;
}

}  // namespace

Lattice Lattice::from_rows(const double* rows) {
    Lattice lattice;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) lattice.a[i][j] = rows[3 * i + j];
    const auto& a = lattice.a;
    // the cofactors: b_i = 2 pi (a_j x a_k) / (a_i . (a_j x a_k))
    const Vec3 c0{a[1][1] * a[2][2] - a[1][2] * a[2][1], a[1][2] * a[2][0] - a[1][0] * a[2][2],
                  a[1][0] * a[2][1] - a[1][1] * a[2][0]};
    const Vec3 c1{a[2][1] * a[0][2] - a[2][2] * a[0][1], a[2][2] * a[0][0] - a[2][0] * a[0][2],
                  a[2][0] * a[0][1] - a[2][1] * a[0][0]};
    const Vec3 c2{a[0][1] * a[1][2] - a[0][2] * a[1][1], a[0][2] * a[1][0] - a[0][0] * a[1][2],
                  a[0][0] * a[1][1] - a[0][1] * a[1][0]};
    const double det = a[0][0] * c0[0] + a[0][1] * c0[1] + a[0][2] * c0[2];
    if (!(std::fabs(det) > 0.0) || !std::isfinite(det))
        throw std::invalid_argument("the cell has no volume");
    for (int j = 0; j < 3; ++j) {
        lattice.b[0][j] = kTwoPi * c0[j] / det;
        lattice.b[1][j] = kTwoPi * c1[j] / det;
        lattice.b[2][j] = kTwoPi * c2[j] / det;
    }
    lattice.volume = std::fabs(det);
    return lattice;
}

Vec3 Lattice::cartesian(const Vec3& k, const Int3& m) const {
    Vec3 out{};
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) out[j] += (k[i] + m[i]) * b[i][j];
    return out;
}

int good_size(int n) {
    if (n < 1) throw std::invalid_argument("an FFT size must be positive");
    for (int m = n;; ++m) {
        int r = m;
        for (const int p : {2, 3, 5})
            while (r % p == 0) r /= p;
        if (r == 1) return m;
    }
}

Int3 fft_grid(const Lattice& lattice, double cutoff) {
    if (!(cutoff > 0.0)) throw std::invalid_argument("the cutoff must be positive");
    const double radius = std::sqrt(2.0 * cutoff);
    // m_i = a_i . G / 2 pi is at most |a_i| |G| / 2 pi; the largest |m_i| of
    // the G of the sphere, which can be smaller in a skewed cell, as Quantum
    // ESPRESSO takes it
    Int3 bound{}, most{};
    for (int i = 0; i < 3; ++i)
        bound[i] = static_cast<int>(std::floor(radius * norm(lattice.a[i]) / kTwoPi));
    for (int m0 = -bound[0]; m0 <= bound[0]; ++m0)
        for (int m1 = -bound[1]; m1 <= bound[1]; ++m1)
            for (int m2 = -bound[2]; m2 <= bound[2]; ++m2) {
                const Vec3 g = lattice.cartesian({0.0, 0.0, 0.0}, {m0, m1, m2});
                if (0.5 * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) > cutoff) continue;
                most = {std::max(most[0], std::abs(m0)), std::max(most[1], std::abs(m1)),
                        std::max(most[2], std::abs(m2))};
            }
    Int3 grid{};
    for (int i = 0; i < 3; ++i) grid[i] = good_size(2 * most[i] + 1);
    return grid;
}

Sphere make_sphere(const Lattice& lattice, const Vec3& k, double cutoff, const Int3& grid,
                   bool half) {
    if (!(cutoff > 0.0)) throw std::invalid_argument("the cutoff must be positive");
    if (half && (k[0] != 0.0 || k[1] != 0.0 || k[2] != 0.0))
        throw std::invalid_argument("real wavefunctions are those of Gamma only");
    const double radius = std::sqrt(2.0 * cutoff);
    const double reach = radius + norm(lattice.cartesian(k, {0, 0, 0}));
    Int3 bound{};
    for (int i = 0; i < 3; ++i)
        bound[i] = static_cast<int>(std::ceil(reach * norm(lattice.a[i]) / kTwoPi)) + 1;
    struct Found {
        double kinetic;
        Int3 m;
    };
    std::vector<Found> found;
    for (int m0 = -bound[0]; m0 <= bound[0]; ++m0)
        for (int m1 = -bound[1]; m1 <= bound[1]; ++m1)
            for (int m2 = -bound[2]; m2 <= bound[2]; ++m2) {
                if (half && !(m2 > 0 || (m2 == 0 && (m1 > 0 || (m1 == 0 && m0 >= 0))))) continue;
                const Vec3 q = lattice.cartesian(k, {m0, m1, m2});
                const double kinetic = 0.5 * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
                if (kinetic <= cutoff) found.push_back({kinetic, {m0, m1, m2}});
            }
    std::sort(found.begin(), found.end(), [](const Found& x, const Found& y) {
        return std::tie(x.kinetic, x.m) < std::tie(y.kinetic, y.m);
    });
    Sphere sphere;
    sphere.miller.reserve(found.size());
    sphere.kinetic.reserve(found.size());
    sphere.box.reserve(found.size());
    sphere.partner.reserve(found.size());
    const std::int64_t last = half ? grid[2] / 2 + 1 : grid[2];
    for (const Found& f : found) {
        for (int i = 0; i < 3; ++i)
            if (2 * std::abs(f.m[i]) >= grid[i])
                throw std::invalid_argument("the FFT grid is too small for the sphere: " +
                                            std::to_string(grid[i]) + " points along vector " +
                                            std::to_string(i + 1));
        sphere.miller.push_back(f.m);
        sphere.kinetic.push_back(f.kinetic);
        sphere.box.push_back((wrap(f.m[0], grid[0]) * grid[1] + wrap(f.m[1], grid[1])) * last +
                             wrap(f.m[2], grid[2]));
        if (half && f.m[2] == 0 && !(f.m[0] == 0 && f.m[1] == 0))
            sphere.partner.push_back(
                (wrap(-f.m[0], grid[0]) * grid[1] + wrap(-f.m[1], grid[1])) * last);
        else
            sphere.partner.push_back(-1);
    }
    return sphere;
}

Sphere make_sphere(const Lattice& lattice, const Lattice* basis, const Vec3& k, double cutoff,
                   const Int3& grid, bool half) {
    if (basis == nullptr) return make_sphere(lattice, k, cutoff, grid, half);
    Sphere sphere = make_sphere(*basis, k, cutoff, grid, half);
    for (std::size_t g = 0; g < sphere.size(); ++g) {
        const Vec3 q = lattice.cartesian(k, sphere.miller[g]);
        sphere.kinetic[g] = 0.5 * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    }
    return sphere;
}

KPoints monkhorst_pack(const Int3& n, const Vec3& offset, bool time_reversal) {
    for (int i = 0; i < 3; ++i)
        if (n[i] < 1) throw std::invalid_argument("a k-point grid needs one point or more");
    auto fold = [](double x) { return x - std::floor(x + 0.5); };
    std::vector<Vec3> all;
    for (int j0 = 0; j0 < n[0]; ++j0)
        for (int j1 = 0; j1 < n[1]; ++j1)
            for (int j2 = 0; j2 < n[2]; ++j2)
                all.push_back({fold((j0 + offset[0]) / n[0]), fold((j1 + offset[1]) / n[1]),
                               fold((j2 + offset[2]) / n[2])});
    // whether x and y are the same point modulo a reciprocal vector
    auto same = [&](const Vec3& x, const Vec3& y) {
        for (int i = 0; i < 3; ++i) {
            const double d = x[i] - y[i];
            if (std::fabs(d - std::round(d)) > 1e-9) return false;
        }
        return true;
    };
    KPoints points;
    std::vector<bool> taken(all.size(), false);
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (taken[i]) continue;
        taken[i] = true;
        double weight = 1.0;
        if (time_reversal) {
            const Vec3 minus{-all[i][0], -all[i][1], -all[i][2]};
            if (!same(all[i], minus))
                for (std::size_t j = i + 1; j < all.size(); ++j)
                    if (!taken[j] && same(all[j], minus)) {
                        taken[j] = true;
                        weight = 2.0;
                        break;
                    }
        }
        points.k.push_back(all[i]);
        points.weight.push_back(weight / static_cast<double>(all.size()));
    }
    return points;
}

Int3 grid_of_spacing(const Lattice& lattice, double spacing) {
    if (!(spacing > 0.0)) throw std::invalid_argument("the k spacing must be positive");
    Int3 n{};
    for (int i = 0; i < 3; ++i)
        n[i] = std::max(1, static_cast<int>(std::ceil(norm(lattice.b[i]) / spacing - 1e-12)));
    return n;
}

}  // namespace mdapy::dft
