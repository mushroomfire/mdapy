// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/core/check.h"
#include "dft/xc/xc.h"

namespace mdapy::dft {

namespace {

__global__ void xc_kernel(Functional functional, std::size_t count, const double* rho,
                          const double* sigma, double threshold, bool polarized, bool signed_gradient,
                          double* f, double* v, double* vs) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    const double up0 = rho[2 * i], down0 = rho[2 * i + 1];
    double out = 0.0, vu = 0.0, vd = 0.0, suu = 0.0, sud = 0.0, sdd = 0.0;
    const double n = up0 + down0, an = fabs(n), sign = n < 0.0 ? -1.0 : 1.0;
    if (an > threshold) {
        // LDA at |n| and zeta within [-1, 1]
        const double edge = 1.0 - 1e-12;
        double zeta = (up0 - down0) / an;
        zeta = zeta > edge ? edge : (zeta < -edge ? -edge : zeta);
        using D2 = Dual<2>;
        const D2 lda = xc::lda_energy(functional, D2::input(0.5 * an * (1.0 + zeta), 0),
                                      D2::input(0.5 * an * (1.0 - zeta), 1));
        out = sign * lda.v;
        vu = lda.d[0];
        vd = lda.d[1];
        if (functional == Functional::Pbe || functional == Functional::PbePw) {
            const double s_uu = sigma[3 * i], s_ud = sigma[3 * i + 1], s_dd = sigma[3 * i + 2];
            const double grad2 = s_uu + 2.0 * s_ud + s_dd;
            using D3 = Dual<3>;
            if (!polarized) {
                if (an > 1e-6 && grad2 > 1e-10) {
                    // of n and |grad n|^2, the exchange of each half by spin scaling
                    const D3 nn = D3::input(an, 0), g2 = D3::input(grad2, 1);
                    const D3 e = 2.0 * xc::exchange_gradient(0.5 * nn, 0.25 * g2) +
                                 xc::correlation_gradient(functional, nn, D3(0.0), g2);
                    out += (signed_gradient ? sign : 1.0) * e.v;
                    vu += e.d[0];
                    vd += e.d[0];
                    suu += e.d[1];
                    sud += 2.0 * e.d[1];
                    sdd += e.d[1];
                }
            } else {
                // exchange where n > 1e-10, of each spin above 1e-10
                if (n > 1e-10 && up0 > 1e-10 && s_uu > 1e-20) {
                    const D3 e = xc::exchange_gradient(D3::input(up0, 0), D3::input(s_uu, 1));
                    out += e.v;
                    vu += e.d[0];
                    suu += e.d[1];
                }
                if (n > 1e-10 && down0 > 1e-10 && s_dd > 1e-20) {
                    const D3 e = xc::exchange_gradient(D3::input(down0, 0), D3::input(s_dd, 1));
                    out += e.v;
                    vd += e.d[0];
                    sdd += e.d[1];
                }
                // correlation where n (not |n|) > 1e-6
                if (n > 1e-6 && grad2 > 1e-12) {
                    const double zc = 1.0 - 1e-6;
                    using D5 = Dual<5>;
                    const D5 u = D5::input(up0, 0), d = D5::input(down0, 1);
                    const D5 nn = u + d;
                    D5 z = (u - d) / nn;
                    if (z.v > zc) z = D5(zc);
                    if (z.v < -zc) z = D5(-zc);
                    const D5 g2 = D5::input(s_uu, 2) + 2.0 * D5::input(s_ud, 3) + D5::input(s_dd, 4);
                    const D5 e = xc::correlation_gradient(functional, nn, z, g2);
                    out += e.v;
                    vu += e.d[0];
                    vd += e.d[1];
                    suu += e.d[2];
                    sud += e.d[3];
                    sdd += e.d[4];
                }
            }
        }
    }
    f[i] = out;
    v[2 * i] = vu;
    v[2 * i + 1] = vd;
    vs[3 * i] = suu;
    vs[3 * i + 1] = sud;
    vs[3 * i + 2] = sdd;
}

}  // namespace

void evaluate_xc(Functional functional, std::size_t count, const double* rho, const double* sigma,
                 double threshold, double* f, double* v, double* vs, bool polarized, bool signed_gradient) {
    if (count == 0) return;
    const unsigned threads = 128;
    const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
    xc_kernel<<<blocks, threads>>>(functional, count, rho, sigma, threshold, polarized, signed_gradient, f, v,
                                   vs);
    check_cuda(cudaGetLastError(), "the exchange-correlation kernel");
}

}  // namespace mdapy::dft
