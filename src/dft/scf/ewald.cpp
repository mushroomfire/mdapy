// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/ewald.h"

#include <cmath>
#include <complex>
#include <stdexcept>

#include "dft/lattice/lattice.h"

namespace mdapy::dft {

EwaldResult ewald(const double* cell, const std::vector<double>& positions, const std::vector<double>& charges) {
    const std::size_t n = charges.size();
    if (positions.size() != 3 * n) throw std::invalid_argument("positions (atoms, 3) and a charge for each");
    const Lattice lattice = Lattice::from_rows(cell);
    const double pi = 3.14159265358979323846, volume = lattice.volume;
    // the split, scaled with the cube root of the atoms; the sums to 1e-16
    const double eta = pi / std::cbrt(volume * volume) * std::cbrt(static_cast<double>(std::max<std::size_t>(n, 1)));
    const double root = std::sqrt(eta), rmax = 8.5 / root, gmax = 2.0 * root * 6.1;
    EwaldResult out;
    out.forces.assign(3 * n, 0.0);
    std::array<double, 9>& d = out.strain;
    auto norm = [](const Vec3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    // reciprocal space
    int span[3];
    for (int i = 0; i < 3; ++i)
        span[i] = static_cast<int>(std::ceil(gmax * norm(lattice.a[static_cast<std::size_t>(i)]) / (2.0 * pi)));
    double reciprocal = 0.0;
    std::vector<std::complex<double>> phase(n);
    for (int m0 = -span[0]; m0 <= span[0]; ++m0)
        for (int m1 = -span[1]; m1 <= span[1]; ++m1)
            for (int m2 = -span[2]; m2 <= span[2]; ++m2) {
                const Vec3 g = lattice.cartesian({0.0, 0.0, 0.0}, {m0, m1, m2});
                const double g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                if (g2 <= 1e-12 || g2 > gmax * gmax) continue;
                const double w = std::exp(-g2 / (4.0 * eta)) / g2;
                std::complex<double> s(0.0, 0.0);
                for (std::size_t i = 0; i < n; ++i) {
                    const double x = g[0] * positions[3 * i] + g[1] * positions[3 * i + 1] + g[2] * positions[3 * i + 2];
                    phase[i] = std::complex<double>(std::cos(x), std::sin(x));
                    s += charges[i] * phase[i];
                }
                const double s2 = std::norm(s);
                const double e = 2.0 * pi / volume * w * s2;
                reciprocal += e;
                // F_i = (4 pi / Omega) z_i w G Im[exp(iG.x_i) conj(S)]
                for (std::size_t i = 0; i < n; ++i) {
                    const double f = 4.0 * pi / volume * charges[i] * w * (phase[i] * std::conj(s)).imag();
                    for (int a = 0; a < 3; ++a) out.forces[3 * i + static_cast<std::size_t>(a)] += f * g[static_cast<std::size_t>(a)];
                }
                // -delta E_G + (4 pi / Omega) |S|^2 w (1/(4 eta) + 1/G^2) G_a G_b
                const double scale = 4.0 * pi / volume * s2 * w * (1.0 / (4.0 * eta) + 1.0 / g2);
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        d[static_cast<std::size_t>(3 * a + b)] += scale * g[static_cast<std::size_t>(a)] * g[static_cast<std::size_t>(b)];
            }
    out.energy += reciprocal;
    for (int a = 0; a < 3; ++a) d[static_cast<std::size_t>(4 * a)] -= reciprocal;
    // real space
    for (int i = 0; i < 3; ++i)
        span[i] = static_cast<int>(std::ceil(rmax * norm(lattice.b[static_cast<std::size_t>(i)]) / (2.0 * pi))) + 1;
    const double two = 2.0 * root / std::sqrt(pi);
    for (int c0 = -span[0]; c0 <= span[0]; ++c0)
        for (int c1 = -span[1]; c1 <= span[1]; ++c1)
            for (int c2 = -span[2]; c2 <= span[2]; ++c2) {
                Vec3 image{};
                for (int a = 0; a < 3; ++a)
                    image[static_cast<std::size_t>(a)] = c0 * lattice.a[0][static_cast<std::size_t>(a)] +
                                                         c1 * lattice.a[1][static_cast<std::size_t>(a)] +
                                                         c2 * lattice.a[2][static_cast<std::size_t>(a)];
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        double v[3];
                        for (int a = 0; a < 3; ++a)
                            v[a] = positions[3 * i + static_cast<std::size_t>(a)] - positions[3 * j + static_cast<std::size_t>(a)] +
                                   image[static_cast<std::size_t>(a)];
                        const double r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                        if (r2 >= rmax * rmax || r2 <= 1e-20) continue;
                        const double r = std::sqrt(r2), erfc = std::erfc(root * r), zz = charges[i] * charges[j];
                        out.energy += 0.5 * zz * erfc / r;
                        const double radial = zz * (erfc / r2 + two * std::exp(-eta * r2) / r) / r;
                        for (int a = 0; a < 3; ++a) {
                            out.forces[3 * i + static_cast<std::size_t>(a)] += radial * v[a];
                            for (int b = 0; b < 3; ++b) d[static_cast<std::size_t>(3 * a + b)] -= 0.5 * radial * v[a] * v[b];
                        }
                    }
            }
    double z2 = 0.0, z = 0.0;
    for (const double q : charges) {
        z2 += q * q;
        z += q;
    }
    out.energy -= std::sqrt(eta / pi) * z2;
    const double background = pi * z * z / (2.0 * volume * eta);
    out.energy -= background;
    for (int a = 0; a < 3; ++a) d[static_cast<std::size_t>(4 * a)] += background;
    return out;
}

}  // namespace mdapy::dft
