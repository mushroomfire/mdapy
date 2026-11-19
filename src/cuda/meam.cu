// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The modified embedded atom method on the device, as calculators/meam.cpp
// computes it, in four passes: the screening of each pair, by the end that
// owns it, for both ends; the partial densities of each atom and its
// embedding; the forces of each pair from both ends and of the screening
// from the owner; and the gather of calculators/many_body.h. The terms of
// pairs are in the precision of the computation; what each atom holds, its
// densities and their derivatives, in double precision, as their
// combinations subtract.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "calculators/meam.h"
#include "cuda/atoms.cuh"
#include "cuda/many_body.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

using M = calculators::MeamPotential;

// What each atom holds after the densities.
struct MeamAtom {
    double rho0, rho1, rho2, rho3, dgamma1, dgamma2, dgamma3, frhop, arho2b;
    double arho1[3], arho2[6], arho3[10], arho3b[3], t_ave[3], tsq_ave[3];
};

template <class Real>
struct MeamTables {
    const double* element;  // 17 rows of elements
    const double* ebound;   // elements^2
    const double* cmin;     // elements^3
    const double* cmax;
    const Real* table;      // pairs a <= b, 7 rows of nr
    int elements, ialloy, mix_ref_t, emb_lin_neg, bkgd_dyn, nr;
    double rc, delr, gsmooth_factor, dr;
};

__device__ inline double row_of(const double* element, int elements, int row, int a) {
    return element[row * elements + a];
}

template <class Real>
__device__ inline bool is_zero(Real x) {
    return fabs(x) < Real(1e-20);
}

template <class Real>
__device__ inline bool is_one(Real x) {
    return fabs(x - Real(1)) < Real(1e-20);
}

template <class Real>
__device__ inline Real cutoff_and_slope(Real x, Real& slope) {
    if (x >= Real(1)) {
        slope = Real(0);
        return Real(1);
    }
    if (x <= Real(0)) {
        slope = Real(0);
        return Real(0);
    }
    const Real a = Real(1) - x, a3 = a * a * a, a4 = a * a3, b = Real(1) - a4;
    slope = Real(8) * b * a3;
    return b * b;
}

template <class Real>
__device__ inline Real dc_by_rij(Real rij2, Real rik2, Real rjk2) {
    const Real rij4 = rij2 * rij2, a = rik2 - rjk2, b = rik2 + rjk2, asq = a * a;
    Real denom = rij4 - asq;
    denom = denom * denom;
    return Real(-4) * (Real(-2) * rij2 * asq + rij4 * b + asq * b) / denom;
}

template <class Real>
__device__ inline void dc_by_rik_rjk(Real rij2, Real rik2, Real rjk2, Real& d1, Real& d2) {
    const Real rij4 = rij2 * rij2, rik4 = rik2 * rik2, rjk4 = rjk2 * rjk2, a = rik2 - rjk2;
    Real denom = rij4 - a * a;
    denom = denom * denom;
    d1 = Real(4) * rij2 * (rij4 + rik4 + Real(2) * rik2 * rjk2 - Real(3) * rjk4 - Real(2) * rij2 * a) / denom;
    d2 = Real(4) * rij2 * (rij4 - Real(3) * rik4 + Real(2) * rik2 * rjk2 + rjk4 + Real(2) * rij2 * a) / denom;
}

__device__ inline double g_of_gamma(double gsmooth, double gamma, int ibar) {
    switch (ibar) {
        case 0:
        case 4: {
            const double switchpoint = -gsmooth / (gsmooth + 1);
            if (gamma < switchpoint) {
                if (gamma == 0.0) return 0.0;
                return sqrt(1 / (gsmooth + 1) * pow(switchpoint / gamma, gsmooth));
            }
            return sqrt(1.0 + gamma);
        }
        case 1:
            return exp(gamma / 2.0);
        case 3:
            return 2.0 / (1.0 + exp(-gamma));
        default:
            return 1.0 + gamma >= 0 ? sqrt(1.0 + gamma) : -sqrt(-1.0 - gamma);
    }
}

__device__ inline double g_and_slope(double gsmooth, double gamma, int ibar, double& dg) {
    double g;
    switch (ibar) {
        case 0:
        case 4: {
            const double switchpoint = -gsmooth / (gsmooth + 1);
            if (gamma < switchpoint) {
                g = sqrt(1 / (gsmooth + 1) * pow(switchpoint / gamma, gsmooth));
                dg = -gsmooth * g / (2.0 * gamma);
                return g;
            }
            g = sqrt(1.0 + gamma);
            dg = 1.0 / (2.0 * g);
            return g;
        }
        case 1:
            g = exp(gamma / 2.0);
            dg = g / 2.0;
            return g;
        case 3:
            g = 2.0 / (1.0 + exp(-gamma));
            dg = g * (2.0 - g) / 2;
            return g;
        default:
            if (1.0 + gamma >= 0) {
                g = sqrt(1.0 + gamma);
                dg = 1.0 / (2.0 * g);
            } else {
                g = -sqrt(-1.0 - gamma);
                dg = -1.0 / (2.0 * g);
            }
            return g;
    }
}

__constant__ int kV2[6] = {1, 2, 2, 1, 2, 1};
__constant__ int kV3[10] = {1, 3, 3, 3, 6, 3, 1, 3, 3, 1};
__constant__ int kVind2[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
__constant__ int kVind3[3][3][3] = {{{0, 1, 2}, {1, 3, 4}, {2, 4, 5}},
                                    {{1, 3, 4}, {3, 6, 7}, {4, 7, 8}},
                                    {{2, 4, 5}, {4, 7, 8}, {5, 8, 9}}};

template <class Real>
__device__ inline Real pair_vector(std::int64_t n, std::int64_t i, int k, const Real4<Real>* x,
                                   const Real4<Real>& xi, const std::int32_t* indices,
                                   const Real* steps, Real* r, int& species, std::int64_t& j) {
    const auto packed = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
    j = packed & kIndexMask;
    const Real4<Real> xj = x[j];
    r[0] = xj.x - xi.x;
    r[1] = xj.y - xi.y;
    r[2] = xj.z - xi.z;
    add_step(steps, packed >> kIndexBits, r[0], r[1], r[2]);
    species = static_cast<int>(xj.w);
    return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
}

// The screening of each pair within rc and its derivative, computed by the
// end whose place in the lists comes first and written for both.
template <class Real>
__global__ void meam_screen(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                            const std::int32_t* indices, const std::int32_t* counts,
                            const std::int32_t* reverse, MeamTables<Real> m, Real* scr,
                            Real* fcp, Real* dscr) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int ei = static_cast<int>(xi.w);
    const int e = m.elements;
    const Real rc = static_cast<Real>(m.rc), rc2 = rc * rc;
    const Real drinv = static_cast<Real>(1.0 / m.delr);
    const int count = counts[i];
    for (int a = 0; a < count; ++a) {
        const std::int64_t p = static_cast<std::int64_t>(a) * n + i;
        Real rij_v[3];
        int ej;
        std::int64_t j;
        const Real rij2 = pair_vector(n, i, a, x, xi, indices, steps, rij_v, ej, j);
        const std::int64_t q = static_cast<std::int64_t>(reverse[p]) * n + j;
        if (!(p < q)) continue;  // the other end computes it
        Real sij = 0, fcij = 0, dsij = 0;
        if (!(rij2 > rc2)) {
            const Real rbound = static_cast<Real>(m.ebound[ei * e + ej]) * rij2;
            const Real rij = sqrt(rij2);
            const Real rnorm = (rc - rij) * drinv;
            sij = Real(1);
            for (int b = 0; b < count; ++b) {
                if (b == a) continue;
                Real rik_v[3];
                int ek;
                std::int64_t k;
                const Real rik2 = pair_vector(n, i, b, x, xi, indices, steps, rik_v, ek, k);
                const Real rjk_v[3] = {rik_v[0] - rij_v[0], rik_v[1] - rij_v[1], rik_v[2] - rij_v[2]};
                const Real rjk2 = rjk_v[0] * rjk_v[0] + rjk_v[1] * rjk_v[1] + rjk_v[2] * rjk_v[2];
                if (rjk2 > rbound || rik2 > rbound) continue;
                const Real xik = rik2 / rij2, xjk = rjk2 / rij2;
                const Real aa = Real(1) - (xik - xjk) * (xik - xjk);
                if (aa <= Real(0)) continue;
                const Real cikj = (Real(2) * (xik + xjk) + aa - Real(2)) / aa;
                const int t = (ei * e + ej) * e + ek;
                const Real cmax = static_cast<Real>(m.cmax[t]), cmin = static_cast<Real>(m.cmin[t]);
                if (cikj >= cmax) continue;
                if (cikj <= cmin) {
                    sij = Real(0);
                    break;
                }
                Real slope;
                sij *= cutoff_and_slope((cikj - cmin) / (cmax - cmin), slope);
            }
            Real dfc;
            fcij = cutoff_and_slope(rnorm, dfc);
            const Real dfcij = dfc * drinv;
            const Real sfcij = sij * fcij;
            if (!is_zero(sfcij) && !is_one(sfcij)) {
                for (int b = 0; b < count; ++b) {
                    if (b == a) continue;
                    Real rik_v[3];
                    int ek;
                    std::int64_t k;
                    const Real rik2 = pair_vector(n, i, b, x, xi, indices, steps, rik_v, ek, k);
                    const Real rjk_v[3] = {rik_v[0] - rij_v[0], rik_v[1] - rij_v[1], rik_v[2] - rij_v[2]};
                    const Real rjk2 = rjk_v[0] * rjk_v[0] + rjk_v[1] * rjk_v[1] + rjk_v[2] * rjk_v[2];
                    if (rjk2 > rbound || rik2 > rbound) continue;
                    const Real xik = rik2 / rij2, xjk = rjk2 / rij2;
                    const Real aa = Real(1) - (xik - xjk) * (xik - xjk);
                    if (aa <= Real(0)) continue;
                    const Real cikj = (Real(2) * (xik + xjk) + aa - Real(2)) / aa;
                    const int t = (ei * e + ej) * e + ek;
                    const Real cmax = static_cast<Real>(m.cmax[t]), cmin = static_cast<Real>(m.cmin[t]);
                    if (cikj >= cmax) continue;
                    const Real delc = cmax - cmin;
                    Real dfikj;
                    const Real sikj = cutoff_and_slope((cikj - cmin) / delc, dfikj);
                    dsij += dfikj / (delc * sikj) * dc_by_rij(rij2, rik2, rjk2);
                }
                dsij = dsij * sfcij - sij * dfcij / rij;
            }
        }
        scr[p] = scr[q] = sij;
        fcp[p] = fcp[q] = fcij;
        dscr[p] = dscr[q] = dsij;
    }
}

template <class Real, bool Energy>
__global__ void meam_density(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                             const std::int32_t* indices, const std::int32_t* counts,
                             MeamTables<Real> m, const Real* scr, const Real* fcp,
                             MeamAtom* atoms, double* embed) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int ei = static_cast<int>(xi.w);
    const int e = m.elements;
    const Real rc2 = static_cast<Real>(m.rc * m.rc);
    MeamAtom at{};
    const int count = counts[i];
    for (int a = 0; a < count; ++a) {
        const std::int64_t p = static_cast<std::int64_t>(a) * n + i;
        if (is_zero(scr[p])) continue;
        Real d[3];
        int ej;
        std::int64_t j;
        const Real rij2 = pair_vector(n, i, a, x, xi, indices, steps, d, ej, j);
        if (!(rij2 < rc2)) continue;
        const Real sij = scr[p] * fcp[p];
        const Real rij = sqrt(rij2);
        const Real aj = rij / static_cast<Real>(row_of(m.element, e, M::RE, ej)) - Real(1);
        const Real ro0j = static_cast<Real>(row_of(m.element, e, M::RHO0, ej));
        const Real rhoa0j = ro0j * exp(-static_cast<Real>(row_of(m.element, e, M::BETA0, ej)) * aj) * sij;
        Real rhoa1j = ro0j * exp(-static_cast<Real>(row_of(m.element, e, M::BETA1, ej)) * aj) * sij;
        Real rhoa2j = ro0j * exp(-static_cast<Real>(row_of(m.element, e, M::BETA2, ej)) * aj) * sij;
        Real rhoa3j = ro0j * exp(-static_cast<Real>(row_of(m.element, e, M::BETA3, ej)) * aj) * sij;
        const Real t1j = static_cast<Real>(row_of(m.element, e, M::T1, ej));
        const Real t2j = static_cast<Real>(row_of(m.element, e, M::T2, ej));
        const Real t3j = static_cast<Real>(row_of(m.element, e, M::T3, ej));
        if (m.ialloy == 1) {
            rhoa1j *= t1j;
            rhoa2j *= t2j;
            rhoa3j *= t3j;
        }
        at.rho0 += rhoa0j;
        if (m.ialloy != 2) {
            at.t_ave[0] += t1j * rhoa0j;
            at.t_ave[1] += t2j * rhoa0j;
            at.t_ave[2] += t3j * rhoa0j;
        }
        if (m.ialloy == 1) {
            at.tsq_ave[0] += t1j * t1j * rhoa0j;
            at.tsq_ave[1] += t2j * t2j * rhoa0j;
            at.tsq_ave[2] += t3j * t3j * rhoa0j;
        }
        at.arho2b += rhoa2j;
        const Real a1 = rhoa1j / rij, a2 = rhoa2j / rij2, a3 = rhoa3j / (rij2 * rij);
        int nv2 = 0, nv3 = 0;
        for (int u = 0; u < 3; ++u) {
            at.arho1[u] += a1 * d[u];
            at.arho3b[u] += rhoa3j * d[u] / rij;
            for (int v = u; v < 3; ++v) {
                at.arho2[nv2] += a2 * d[u] * d[v];
                ++nv2;
                for (int w = v; w < 3; ++w) {
                    at.arho3[nv3] += a3 * d[u] * d[v] * d[w];
                    ++nv3;
                }
            }
        }
    }
    // the densities combined, in double precision
    double rho1 = 0.0, rho2 = -1.0 / 3.0 * at.arho2b * at.arho2b, rho3 = 0.0;
    for (int u = 0; u < 3; ++u) {
        rho1 += at.arho1[u] * at.arho1[u];
        rho3 -= 3.0 / 5.0 * at.arho3b[u] * at.arho3b[u];
    }
    for (int k = 0; k < 6; ++k) rho2 += kV2[k] * at.arho2[k] * at.arho2[k];
    for (int k = 0; k < 10; ++k) rho3 += kV3[k] * at.arho3[k] * at.arho3[k];
    const double rho0 = at.rho0;
    if (rho0 > 0.0) {
        for (int k = 0; k < 3; ++k) {
            if (m.ialloy == 1) {
                at.t_ave[k] = is_zero(at.tsq_ave[k]) ? 0.0 : at.t_ave[k] / at.tsq_ave[k];
            } else if (m.ialloy == 2) {
                at.t_ave[k] = row_of(m.element, e, M::T1 + k, ei);
            } else {
                at.t_ave[k] = at.t_ave[k] / rho0;
            }
        }
    }
    double gamma = at.t_ave[0] * rho1 + at.t_ave[1] * rho2 + at.t_ave[2] * rho3;
    if (rho0 > 0.0) gamma = gamma / (rho0 * rho0);
    const double z = row_of(m.element, e, M::Z, ei);
    const int ibar = static_cast<int>(row_of(m.element, e, M::IBAR, ei));
    const double shp[3] = {row_of(m.element, e, M::S1, ei), row_of(m.element, e, M::S2, ei),
                           row_of(m.element, e, M::S3, ei)};
    const double g = g_of_gamma(m.gsmooth_factor, gamma, ibar);
    const double rho = rho0 * g;
    double rho_bkgd, gbar = 1.0, dgbar = 0.0;
    if (m.mix_ref_t == 1) {
        if (ibar > 0) {
            const double gam = (at.t_ave[0] * shp[0] + at.t_ave[1] * shp[1] + at.t_ave[2] * shp[2]) / (z * z);
            gbar = g_and_slope(m.gsmooth_factor, gam, ibar, dgbar);
        }
        rho_bkgd = row_of(m.element, e, M::RHO0, ei) * z * gbar;
    } else if (m.bkgd_dyn == 1) {
        rho_bkgd = row_of(m.element, e, M::RHO0, ei) * z;
    } else {
        rho_bkgd = row_of(m.element, e, M::RHO_REF, ei);
    }
    const double rhob = rho / rho_bkgd, denom = 1.0 / rho_bkgd;
    double dg;
    const double g2 = g_and_slope(m.gsmooth_factor, gamma, ibar, dg);
    at.rho1 = rho1;
    at.rho2 = rho2;
    at.rho3 = rho3;
    at.dgamma1 = (g2 - 2 * dg * gamma) * denom;
    at.dgamma2 = is_zero(rho0) ? 0.0 : (dg / rho0) * denom;
    at.dgamma3 = m.mix_ref_t == 1 ? rho0 * g2 * dgbar / (gbar * z * z) * denom : 0.0;
    // the embedding energy and its derivative
    const double aec = row_of(m.element, e, M::A, ei) * row_of(m.element, e, M::EC, ei);
    double energy;
    if (rhob > 0.0) {
        const double lrb = log(rhob);
        at.frhop = aec * (1.0 + lrb);
        energy = aec * rhob * lrb;
    } else if (m.emb_lin_neg == 0) {
        at.frhop = 0.0;
        energy = 0.0;
    } else {
        at.frhop = -aec;
        energy = -aec * rhob;
    }
    atoms[i] = at;
    if (Energy) embed[i] = energy;
}

template <class Real, bool Energy, bool Virial, bool Atomic>
__global__ void meam_force(std::int64_t n, std::int64_t plane, const Real4<Real>* x,
                           Images<Real> images, const std::int32_t* indices,
                           const std::int32_t* counts, const std::int32_t* reverse,
                           MeamTables<Real> m, const Real* scr, const Real* fcp, const Real* dscr,
                           const MeamAtom* atoms, const double* embed, Real4<Real>* derivative,
                           double* share_virial, double* own_force, double* energies,
                           double* virials) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // the derivatives of the screening of the pairs of this centre, which it
    // holds alone, from zero
    for (int a = 0; a < counts[i]; ++a) {
        derivative[static_cast<std::int64_t>(a) * n + i] = four<Real>(0.0, 0.0, 0.0);
    }
    const Real4<Real> xi = x[i];
    const int ei = static_cast<int>(xi.w);
    const int e = m.elements;
    const Real rc2 = static_cast<Real>(m.rc * m.rc);
    const MeamAtom& ai = atoms[i];
    double fsum[3] = {0.0, 0.0, 0.0};
    double own_virial[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double own_energy = Energy ? embed[i] : 0.0;
    const auto R = [&](int row, int a) { return static_cast<Real>(row_of(m.element, e, row, a)); };
    const int count = counts[i];
    for (int a = 0; a < count; ++a) {
        const std::int64_t p = static_cast<std::int64_t>(a) * n + i;
        if (is_zero(scr[p])) continue;
        Real delij[3];
        int ej;
        std::int64_t j;
        const Real rij2 = pair_vector(n, i, a, x, xi, indices, steps, delij, ej, j);
        if (!(rij2 < rc2)) continue;
        const MeamAtom& aj = atoms[j];
        Real sij = scr[p] * fcp[p];
        const Real rij = sqrt(rij2), recip = Real(1) / rij;
        // the pair function from its table
        const int lo = ei < ej ? ei : ej, hi = ei < ej ? ej : ei;
        const Real* tab = m.table + static_cast<std::int64_t>(lo * e - lo * (lo - 1) / 2 + (hi - lo)) * 7 * m.nr;
        Real ppos = rij / static_cast<Real>(m.dr);
        int kk = static_cast<int>(ppos);
        kk = kk < m.nr - 2 ? kk : m.nr - 2;
        ppos = ppos - kk < Real(1) ? ppos - kk : Real(1);
        const Real phi = ((tab[3 * m.nr + kk] * ppos + tab[2 * m.nr + kk]) * ppos + tab[m.nr + kk]) * ppos + tab[kk];
        const Real phip = (tab[6 * m.nr + kk] * ppos + tab[5 * m.nr + kk]) * ppos + tab[4 * m.nr + kk];
        if (Energy) own_energy += 0.5 * static_cast<double>(phi * sij);

        const Real invrei = Real(1) / R(M::RE, ei);
        const Real aii = rij * invrei - Real(1), ro0i = R(M::RHO0, ei);
        Real rhoa0i = ro0i * exp(-R(M::BETA0, ei) * aii);
        Real drhoa0i = -R(M::BETA0, ei) * invrei * rhoa0i;
        Real rhoa1i = ro0i * exp(-R(M::BETA1, ei) * aii);
        Real drhoa1i = -R(M::BETA1, ei) * invrei * rhoa1i;
        Real rhoa2i = ro0i * exp(-R(M::BETA2, ei) * aii);
        Real drhoa2i = -R(M::BETA2, ei) * invrei * rhoa2i;
        Real rhoa3i = ro0i * exp(-R(M::BETA3, ei) * aii);
        Real drhoa3i = -R(M::BETA3, ei) * invrei * rhoa3i;
        Real rhoa0j, drhoa0j, rhoa1j, drhoa1j, rhoa2j, drhoa2j, rhoa3j, drhoa3j;
        if (ei != ej) {
            const Real invrej = Real(1) / R(M::RE, ej);
            const Real ajj = rij * invrej - Real(1), ro0j = R(M::RHO0, ej);
            rhoa0j = ro0j * exp(-R(M::BETA0, ej) * ajj);
            drhoa0j = -R(M::BETA0, ej) * invrej * rhoa0j;
            rhoa1j = ro0j * exp(-R(M::BETA1, ej) * ajj);
            drhoa1j = -R(M::BETA1, ej) * invrej * rhoa1j;
            rhoa2j = ro0j * exp(-R(M::BETA2, ej) * ajj);
            drhoa2j = -R(M::BETA2, ej) * invrej * rhoa2j;
            rhoa3j = ro0j * exp(-R(M::BETA3, ej) * ajj);
            drhoa3j = -R(M::BETA3, ej) * invrej * rhoa3j;
        } else {
            rhoa0j = rhoa0i;
            drhoa0j = drhoa0i;
            rhoa1j = rhoa1i;
            drhoa1j = drhoa1i;
            rhoa2j = rhoa2i;
            drhoa2j = drhoa2i;
            rhoa3j = rhoa3i;
            drhoa3j = drhoa3i;
        }
        const Real t1mi = R(M::T1, ei), t2mi = R(M::T2, ei), t3mi = R(M::T3, ei);
        const Real t1mj = R(M::T1, ej), t2mj = R(M::T2, ej), t3mj = R(M::T3, ej);
        if (m.ialloy == 1) {
            rhoa1j *= t1mj;
            rhoa2j *= t2mj;
            rhoa3j *= t3mj;
            rhoa1i *= t1mi;
            rhoa2i *= t2mi;
            rhoa3i *= t3mi;
            drhoa1j *= t1mj;
            drhoa2j *= t2mj;
            drhoa3j *= t3mj;
            drhoa1i *= t1mi;
            drhoa2i *= t2mi;
            drhoa3i *= t3mi;
        }
        Real arho1i[3], arho1j[3], arho2i[6], arho2j[6], arho3i[10], arho3j[10], arho3bi[3], arho3bj[3];
        for (int u = 0; u < 3; ++u) {
            arho1i[u] = static_cast<Real>(ai.arho1[u]);
            arho1j[u] = static_cast<Real>(aj.arho1[u]);
            arho3bi[u] = static_cast<Real>(ai.arho3b[u]);
            arho3bj[u] = static_cast<Real>(aj.arho3b[u]);
        }
        for (int u = 0; u < 6; ++u) {
            arho2i[u] = static_cast<Real>(ai.arho2[u]);
            arho2j[u] = static_cast<Real>(aj.arho2[u]);
        }
        for (int u = 0; u < 10; ++u) {
            arho3i[u] = static_cast<Real>(ai.arho3[u]);
            arho3j[u] = static_cast<Real>(aj.arho3[u]);
        }
        int nv2 = 0, nv3 = 0;
        Real arg1i1 = 0, arg1j1 = 0, arg1i2 = 0, arg1j2 = 0, arg1i3 = 0, arg1j3 = 0, arg3i3 = 0, arg3j3 = 0;
        for (int u = 0; u < 3; ++u) {
            for (int v = u; v < 3; ++v) {
                for (int w = v; w < 3; ++w) {
                    const Real arg = delij[u] * delij[v] * delij[w] * Real(kV3[nv3]);
                    arg1i3 += arho3i[nv3] * arg;
                    arg1j3 -= arho3j[nv3] * arg;
                    ++nv3;
                }
                const Real arg = delij[u] * delij[v] * Real(kV2[nv2]);
                arg1i2 += arho2i[nv2] * arg;
                arg1j2 += arho2j[nv2] * arg;
                ++nv2;
            }
            arg1i1 += arho1i[u] * delij[u];
            arg1j1 -= arho1j[u] * delij[u];
            arg3i3 += arho3bi[u] * delij[u];
            arg3j3 -= arho3bj[u] * delij[u];
        }
        const Real drho0dr1 = drhoa0j * sij, drho0dr2 = drhoa0i * sij;
        Real a1 = Real(2) * sij / rij;
        const Real drho1dr1 = a1 * (drhoa1j - rhoa1j / rij) * arg1i1;
        const Real drho1dr2 = a1 * (drhoa1i - rhoa1i / rij) * arg1j1;
        Real drho1drm1[3], drho1drm2[3], drho2drm1[3], drho2drm2[3], drho3drm1[3], drho3drm2[3];
        for (int u = 0; u < 3; ++u) {
            drho1drm1[u] = a1 * rhoa1j * arho1i[u];
            drho1drm2[u] = -a1 * rhoa1i * arho1j[u];
        }
        const Real arho2bi = static_cast<Real>(ai.arho2b), arho2bj = static_cast<Real>(aj.arho2b);
        Real a2 = Real(2) * sij / rij2;
        const Real drho2dr1 = a2 * (drhoa2j - Real(2) * rhoa2j / rij) * arg1i2 -
                              Real(2.0 / 3.0) * arho2bi * drhoa2j * sij;
        const Real drho2dr2 = a2 * (drhoa2i - Real(2) * rhoa2i / rij) * arg1j2 -
                              Real(2.0 / 3.0) * arho2bj * drhoa2i * sij;
        a2 = Real(4) * sij / rij2;
        for (int u = 0; u < 3; ++u) {
            drho2drm1[u] = 0;
            drho2drm2[u] = 0;
            for (int v = 0; v < 3; ++v) {
                drho2drm1[u] += arho2i[kVind2[u][v]] * delij[v];
                drho2drm2[u] -= arho2j[kVind2[u][v]] * delij[v];
            }
            drho2drm1[u] = a2 * rhoa2j * drho2drm1[u];
            drho2drm2[u] = -a2 * rhoa2i * drho2drm2[u];
        }
        const Real rij3 = rij * rij2;
        Real a3 = Real(2) * sij / rij3;
        Real a3a = Real(6.0 / 5.0) * sij / rij;
        const Real drho3dr1 = a3 * (drhoa3j - Real(3) * rhoa3j / rij) * arg1i3 -
                              a3a * (drhoa3j - rhoa3j / rij) * arg3i3;
        const Real drho3dr2 = a3 * (drhoa3i - Real(3) * rhoa3i / rij) * arg1j3 -
                              a3a * (drhoa3i - rhoa3i / rij) * arg3j3;
        a3 = Real(6) * sij / rij3;
        a3a = Real(6) * sij / (Real(5) * rij);
        for (int u = 0; u < 3; ++u) {
            drho3drm1[u] = 0;
            drho3drm2[u] = 0;
            int k2 = 0;
            for (int v = 0; v < 3; ++v) {
                for (int w = v; w < 3; ++w) {
                    const Real arg = delij[v] * delij[w] * Real(kV2[k2]);
                    drho3drm1[u] += arho3i[kVind3[u][v][w]] * arg;
                    drho3drm2[u] += arho3j[kVind3[u][v][w]] * arg;
                    ++k2;
                }
            }
            drho3drm1[u] = (a3 * drho3drm1[u] - a3a * arho3bi[u]) * rhoa3j;
            drho3drm2[u] = (-a3 * drho3drm2[u] + a3a * arho3bj[u]) * rhoa3i;
        }
        const Real t1i = static_cast<Real>(ai.t_ave[0]), t2i = static_cast<Real>(ai.t_ave[1]),
                   t3i = static_cast<Real>(ai.t_ave[2]);
        const Real t1j = static_cast<Real>(aj.t_ave[0]), t2j = static_cast<Real>(aj.t_ave[1]),
                   t3j = static_cast<Real>(aj.t_ave[2]);
        const auto div0 = [](Real num, double den) {
            return is_zero(den) ? Real(0) : num / static_cast<Real>(den);
        };
        Real dt1dr1 = 0, dt1dr2 = 0, dt2dr1 = 0, dt2dr2 = 0, dt3dr1 = 0, dt3dr2 = 0;
        if (m.ialloy == 1) {
            dt1dr1 = div0(drhoa0j * sij, ai.tsq_ave[0]) * (t1mj - t1i * t1mj * t1mj);
            dt1dr2 = div0(drhoa0i * sij, aj.tsq_ave[0]) * (t1mi - t1j * t1mi * t1mi);
            dt2dr1 = div0(drhoa0j * sij, ai.tsq_ave[1]) * (t2mj - t2i * t2mj * t2mj);
            dt2dr2 = div0(drhoa0i * sij, aj.tsq_ave[1]) * (t2mi - t2j * t2mi * t2mi);
            dt3dr1 = div0(drhoa0j * sij, ai.tsq_ave[2]) * (t3mj - t3i * t3mj * t3mj);
            dt3dr2 = div0(drhoa0i * sij, aj.tsq_ave[2]) * (t3mi - t3j * t3mi * t3mi);
        } else if (m.ialloy != 2) {
            const Real ai_ = div0(drhoa0j * sij, ai.rho0);
            const Real aj_ = div0(drhoa0i * sij, aj.rho0);
            dt1dr1 = ai_ * (t1mj - t1i);
            dt1dr2 = aj_ * (t1mi - t1j);
            dt2dr1 = ai_ * (t2mj - t2i);
            dt2dr2 = aj_ * (t2mi - t2j);
            dt3dr1 = ai_ * (t3mj - t3i);
            dt3dr2 = aj_ * (t3mi - t3j);
        }
        const Real shpi[3] = {R(M::S1, ei), R(M::S2, ei), R(M::S3, ei)};
        const Real shpj[3] = {R(M::S1, ej), R(M::S2, ej), R(M::S3, ej)};
        const Real dg1i = static_cast<Real>(ai.dgamma1), dg2i = static_cast<Real>(ai.dgamma2),
                   dg3i = static_cast<Real>(ai.dgamma3);
        const Real dg1j = static_cast<Real>(aj.dgamma1), dg2j = static_cast<Real>(aj.dgamma2),
                   dg3j = static_cast<Real>(aj.dgamma3);
        const Real rho1i = static_cast<Real>(ai.rho1), rho2i = static_cast<Real>(ai.rho2),
                   rho3i = static_cast<Real>(ai.rho3);
        const Real rho1j = static_cast<Real>(aj.rho1), rho2j = static_cast<Real>(aj.rho2),
                   rho3j = static_cast<Real>(aj.rho3);
        const Real drhodr1 = dg1i * drho0dr1 +
                             dg2i * (dt1dr1 * rho1i + t1i * drho1dr1 + dt2dr1 * rho2i +
                                     t2i * drho2dr1 + dt3dr1 * rho3i + t3i * drho3dr1) -
                             dg3i * (shpi[0] * dt1dr1 + shpi[1] * dt2dr1 + shpi[2] * dt3dr1);
        const Real drhodr2 = dg1j * drho0dr2 +
                             dg2j * (dt1dr2 * rho1j + t1j * drho1dr2 + dt2dr2 * rho2j +
                                     t2j * drho2dr2 + dt3dr2 * rho3j + t3j * drho3dr2) -
                             dg3j * (shpj[0] * dt1dr2 + shpj[1] * dt2dr2 + shpj[2] * dt3dr2);
        Real drhodrm1[3], drhodrm2[3];
        for (int u = 0; u < 3; ++u) {
            drhodrm1[u] = dg2i * (t1i * drho1drm1[u] + t2i * drho2drm1[u] + t3i * drho3drm1[u]);
            drhodrm2[u] = dg2j * (t1j * drho1drm2[u] + t2j * drho2drm2[u] + t3j * drho3drm2[u]);
        }
        Real drhods1 = 0, drhods2 = 0;
        const Real dscr_p = dscr[p];
        if (!is_zero(dscr_p)) {
            const Real drho0ds1 = rhoa0j, drho0ds2 = rhoa0i;
            const Real b1 = Real(2) / rij, b2 = Real(2) / rij2, b3 = Real(2) / rij3,
                       b3a = Real(6) / (Real(5) * rij);
            const Real drho1ds1 = b1 * rhoa1j * arg1i1;
            const Real drho1ds2 = b1 * rhoa1i * arg1j1;
            const Real drho2ds1 = b2 * rhoa2j * arg1i2 - Real(2.0 / 3.0) * arho2bi * rhoa2j;
            const Real drho2ds2 = b2 * rhoa2i * arg1j2 - Real(2.0 / 3.0) * arho2bj * rhoa2i;
            const Real drho3ds1 = b3 * rhoa3j * arg1i3 - b3a * rhoa3j * arg3i3;
            const Real drho3ds2 = b3 * rhoa3i * arg1j3 - b3a * rhoa3i * arg3j3;
            Real dt1ds1 = 0, dt1ds2 = 0, dt2ds1 = 0, dt2ds2 = 0, dt3ds1 = 0, dt3ds2 = 0;
            if (m.ialloy == 1) {
                dt1ds1 = div0(rhoa0j, ai.tsq_ave[0]) * (t1mj - t1i * t1mj * t1mj);
                dt1ds2 = div0(rhoa0i, aj.tsq_ave[0]) * (t1mi - t1j * t1mi * t1mi);
                dt2ds1 = div0(rhoa0j, ai.tsq_ave[1]) * (t2mj - t2i * t2mj * t2mj);
                dt2ds2 = div0(rhoa0i, aj.tsq_ave[1]) * (t2mi - t2j * t2mi * t2mi);
                dt3ds1 = div0(rhoa0j, ai.tsq_ave[2]) * (t3mj - t3i * t3mj * t3mj);
                dt3ds2 = div0(rhoa0i, aj.tsq_ave[2]) * (t3mi - t3j * t3mi * t3mi);
            } else if (m.ialloy != 2) {
                const Real ai_ = div0(rhoa0j, ai.rho0);
                const Real aj_ = div0(rhoa0i, aj.rho0);
                dt1ds1 = ai_ * (t1mj - t1i);
                dt1ds2 = aj_ * (t1mi - t1j);
                dt2ds1 = ai_ * (t2mj - t2i);
                dt2ds2 = aj_ * (t2mi - t2j);
                dt3ds1 = ai_ * (t3mj - t3i);
                dt3ds2 = aj_ * (t3mi - t3j);
            }
            drhods1 = dg1i * drho0ds1 +
                      dg2i * (dt1ds1 * rho1i + t1i * drho1ds1 + dt2ds1 * rho2i + t2i * drho2ds1 +
                              dt3ds1 * rho3i + t3i * drho3ds1) -
                      dg3i * (shpi[0] * dt1ds1 + shpi[1] * dt2ds1 + shpi[2] * dt3ds1);
            drhods2 = dg1j * drho0ds2 +
                      dg2j * (dt1ds2 * rho1j + t1j * drho1ds2 + dt2ds2 * rho2j + t2j * drho2ds2 +
                              dt3ds2 * rho3j + t3j * drho3ds2) -
                      dg3j * (shpj[0] * dt1ds2 + shpj[1] * dt2ds2 + shpj[2] * dt3ds2);
        }
        const Real frhopi = static_cast<Real>(ai.frhop), frhopj = static_cast<Real>(aj.frhop);
        const Real dudrij = phip * sij + frhopi * drhodr1 + frhopj * drhodr2;
        const Real dudsij = is_zero(dscr_p) ? Real(0) : phi + frhopi * drhods1 + frhopj * drhods2;
        const Real force = dudrij * recip + dudsij * dscr_p;
        Real fi[3];
        for (int u = 0; u < 3; ++u) {
            fi[u] = delij[u] * force + (frhopi * drhodrm1[u] + frhopj * drhodrm2[u]);
            fsum[u] += static_cast<double>(fi[u]);
        }
        if (Virial) {
            for (int u = 0; u < 3; ++u) {
                for (int v = 0; v < 3; ++v) {
                    own_virial[3 * u + v] += static_cast<double>(Real(-0.25) * (delij[u] * fi[v] + delij[v] * fi[u]));
                }
            }
        }

        // the screening by each k, once for each pair: a third to each atom
        const std::int64_t q = static_cast<std::int64_t>(reverse[p]) * n + j;
        if (!(p < q) || is_zero(sij) || is_one(sij)) continue;
        const Real ebound = static_cast<Real>(m.ebound[ei * e + ej]);
        for (int b = 0; b < count; ++b) {
            if (b == a) continue;
            const std::int64_t pb = static_cast<std::int64_t>(b) * n + i;
            Real dik[3];
            int ek;
            std::int64_t k;
            const Real rik2 = pair_vector(n, i, b, x, xi, indices, steps, dik, ek, k);
            const int t = (ei * e + ej) * e + ek;
            const Real cmax = static_cast<Real>(m.cmax[t]), cmin = static_cast<Real>(m.cmin[t]);
            sij = scr[p] * fcp[p];
            Real dsij1 = 0, dsij2 = 0;
            Real djk[3] = {dik[0] - delij[0], dik[1] - delij[1], dik[2] - delij[2]};
            if (!is_zero(sij) && !is_one(sij)) {
                const Real rbound = rij2 * ebound;
                const Real delc = cmax - cmin;
                const Real rjk2 = djk[0] * djk[0] + djk[1] * djk[1] + djk[2] * djk[2];
                if (rjk2 <= rbound && rik2 <= rbound) {
                    const Real xik = rik2 / rij2, xjk = rjk2 / rij2;
                    const Real aa = Real(1) - (xik - xjk) * (xik - xjk);
                    if (!is_zero(aa)) {
                        const Real cikj = (Real(2) * (xik + xjk) + aa - Real(2)) / aa;
                        if (cikj >= cmin && cikj <= cmax) {
                            Real dfc;
                            const Real sikj = cutoff_and_slope((cikj - cmin) / delc, dfc);
                            Real d1, d2;
                            dc_by_rik_rjk(rij2, rik2, rjk2, d1, d2);
                            const Real coef = sij / delc * dfc / sikj;
                            dsij1 = coef * d1;
                            dsij2 = coef * d2;
                        }
                    }
                }
            }
            if (is_zero(dsij1) && is_zero(dsij2)) continue;
            const Real force1 = dudsij * dsij1, force2 = dudsij * dsij2;
            Real f_i[3], f_j[3], f_k[3];
            for (int u = 0; u < 3; ++u) {
                f_i[u] = force1 * dik[u];
                f_j[u] = force2 * djk[u];
                f_k[u] = -(f_i[u] + f_j[u]);
            }
            add_to<Real>(derivative, p, -f_j[0], -f_j[1], -f_j[2]);
            add_to<Real>(derivative, pb, -f_k[0], -f_k[1], -f_k[2]);
            if (Virial) {
                for (int u = 0; u < 3; ++u) {
                    for (int v = 0; v < 3; ++v) {
                        const double w = static_cast<double>(
                            Real(-1.0 / 6.0) * (dik[u] * f_i[v] + djk[u] * f_j[v] + dik[v] * f_i[u] + djk[v] * f_j[u]));
                        if (Atomic) {
                            own_virial[3 * u + v] += w;
                            share_virial[(3 * u + v) * plane + p] += w;
                            share_virial[(3 * u + v) * plane + pb] += w;
                        } else {
                            own_virial[3 * u + v] += 3.0 * w;
                        }
                    }
                }
            }
        }
    }
    for (int u = 0; u < 3; ++u) own_force[3 * i + u] = fsum[u];
    if (Energy) energies[i] = own_energy;
    if (Virial) {
        for (int c = 0; c < 9; ++c) virials[static_cast<std::int64_t>(c) * n + i] = own_virial[c];
    }
}

template <class Real>
class Meam final : public Potential {
public:
    explicit Meam(const calculators::MeamPotential& p) : p_(p) {
        element_.upload(p.element.data(), p.element.size());
        ebound_.upload(p.ebound.data(), p.ebound.size());
        cmin_.upload(p.cmin.data(), p.cmin.size());
        cmax_.upload(p.cmax.data(), p.cmax.size());
        std::vector<Real> table(p.table.begin(), p.table.end());
        table_.upload(table.data(), table.size());
    }

    double cutoff() const override { return p_.cutoff; }
    int species() const override { return p_.elements; }

    void compute(Atoms& atoms, Neighbors& neighbors, const Box& box, bool energy, bool virial,
                 bool atomic) override {
        Atoms::Impl& a = atoms.impl();
        Neighbors::Impl& l = neighbors.impl();
        if (l.n != a.n || l.builds == 0 || l.cutoff + 1e-12 < p_.cutoff) {
            throw std::logic_error("the neighbor list does not fit the atoms or the cutoff");
        }
        const std::int64_t n = a.n;
        atom_data_.resize(static_cast<std::size_t>(n));
        embed_.resize(static_cast<std::size_t>(n));
        own_force_.resize(static_cast<std::size_t>(3 * n));
        if (virial) a.virials.resize(static_cast<std::size_t>(9 * n));
        const Real4<Real>* x = l.local<Real>(a, box);
        const Images<Real> b = l.steps<Real>(box);
        MeamTables<Real> t{element_.data(), ebound_.data(), cmin_.data(), cmax_.data(),
                           table_.data(), p_.elements, p_.ialloy, p_.mix_ref_t, p_.emb_lin_neg,
                           p_.bkgd_dyn, p_.nr, p_.rc, p_.delr, p_.gsmooth_factor, p_.dr};
        const unsigned g = blocks(n);
        // the pairs within the radius of the neighbors needed: those within
        // rc, and every atom that can screen one
        pairs_.build(l, x, b, static_cast<Real>(p_.cutoff * p_.cutoff));
        const std::int32_t* idx = pairs_.indices();
        const std::int32_t* cnt = pairs_.counts();
        const std::int32_t* reverse = pairs_.reverse();
        const std::int64_t plane = pairs_.plane(n);
        at_least(scr_, static_cast<std::size_t>(plane));
        at_least(fcp_, static_cast<std::size_t>(plane));
        at_least(dscr_, static_cast<std::size_t>(plane));
        at_least(derivative_, static_cast<std::size_t>(plane));
        if (atomic && virial) {
            at_least(share_virial_, static_cast<std::size_t>(9 * plane));
            share_virial_.zero();
        }
        meam_screen<Real><<<g, kBlock>>>(n, x, b, idx, cnt, reverse, t, scr_.data(), fcp_.data(),
                                         dscr_.data());
        MDAPY_LAUNCHED();
        if (energy) {
            meam_density<Real, true><<<g, kBlock>>>(n, x, b, idx, cnt, t, scr_.data(), fcp_.data(),
                                                    atom_data_.data(), embed_.data());
        } else {
            meam_density<Real, false><<<g, kBlock>>>(n, x, b, idx, cnt, t, scr_.data(), fcp_.data(),
                                                     atom_data_.data(), embed_.data());
        }
        MDAPY_LAUNCHED();
        const auto force = [&](auto kernel) {
            kernel<<<g, kBlock>>>(n, plane, x, b, idx, cnt, reverse, t, scr_.data(), fcp_.data(),
                                  dscr_.data(), atom_data_.data(), embed_.data(), derivative_.data(),
                                  share_virial_.data(), own_force_.data(), a.energies.data(),
                                  a.virials.data());
        };
        if (atomic) {
            if (energy && virial) force(meam_force<Real, true, true, true>);
            else if (energy) force(meam_force<Real, true, false, true>);
            else if (virial) force(meam_force<Real, false, true, true>);
            else force(meam_force<Real, false, false, true>);
        } else {
            if (energy && virial) force(meam_force<Real, true, true, false>);
            else if (energy) force(meam_force<Real, true, false, false>);
            else if (virial) force(meam_force<Real, false, true, false>);
            else force(meam_force<Real, false, false, false>);
        }
        MDAPY_LAUNCHED();
        gather<Real>(n, plane, pairs_, derivative_.data(), nullptr, share_virial_.data(),
                     own_force_.data(), true, false, virial, atomic, a.forces.data(),
                     a.energies.data(), a.virials.data());
        a.has_energies = energy;
        a.has_virials = virial;
    }

private:
    calculators::MeamPotential p_;
    Buffer<double> element_, ebound_, cmin_, cmax_;
    Buffer<Real> table_;
    Buffer<Real> scr_, fcp_, dscr_;
    Buffer<Real4<Real>> derivative_;
    Buffer<double> share_virial_, embed_, own_force_;
    Buffer<MeamAtom> atom_data_;
    ShortPairs pairs_;
};

}  // namespace

std::unique_ptr<Potential> make_meam(const calculators::MeamPotential& potential,
                                     Precision precision) {
    if (precision == Precision::Mixed) {
        return std::make_unique<Meam<float>>(potential);
    }
    return std::make_unique<Meam<double>>(potential);
}

}  // namespace mdapy::cuda
