// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/meam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "calculators/many_body.h"
#include "neighbor/frame.h"

namespace mdapy::calculators {

namespace {

using M = MeamPotential;

bool is_zero(double x) { return std::fabs(x) < 1e-20; }
bool is_one(double x) { return std::fabs(x - 1.0) < 1e-20; }
double divide_or_zero(double n, double d) { return is_zero(d) ? 0.0 : n / d; }

// (1 - (1 - x)^4)^2 and its derivative, 0 below 0 and 1 above 1
double cutoff_and_slope(double x, double& slope) {
    if (x >= 1.0) {
        slope = 0.0;
        return 1.0;
    }
    if (x <= 0.0) {
        slope = 0.0;
        return 0.0;
    }
    const double a = 1.0 - x, a3 = a * a * a, a4 = a * a3, b = 1.0 - a4;
    slope = 8 * b * a3;
    return b * b;
}

double cutoff(double x) {
    double slope;
    return cutoff_and_slope(x, slope);
}

// The derivative of C_ikj by r_ij^2, and by r_ik^2 and r_jk^2
double dc_by_rij(double rij2, double rik2, double rjk2) {
    const double rij4 = rij2 * rij2, a = rik2 - rjk2, b = rik2 + rjk2, asq = a * a;
    double denom = rij4 - asq;
    denom = denom * denom;
    return -4 * (-2 * rij2 * asq + rij4 * b + asq * b) / denom;
}

void dc_by_rik_rjk(double rij2, double rik2, double rjk2, double& d1, double& d2) {
    const double rij4 = rij2 * rij2, rik4 = rik2 * rik2, rjk4 = rjk2 * rjk2, a = rik2 - rjk2;
    double denom = rij4 - a * a;
    denom = denom * denom;
    d1 = 4 * rij2 * (rij4 + rik4 + 2 * rik2 * rjk2 - 3 * rjk4 - 2 * rij2 * a) / denom;
    d2 = 4 * rij2 * (rij4 - 3 * rik4 + 2 * rik2 * rjk2 + rjk4 + 2 * rij2 * a) / denom;
}

double g_of_gamma(const M& m, double gamma, int ibar) {
    switch (ibar) {
        case 0:
        case 4: {
            const double switchpoint = -m.gsmooth_factor / (m.gsmooth_factor + 1);
            if (gamma < switchpoint) {
                if (gamma == 0.0) {
                    return 0.0;
                }
                return std::sqrt(1 / (m.gsmooth_factor + 1) *
                                 std::pow(switchpoint / gamma, m.gsmooth_factor));
            }
            return std::sqrt(1.0 + gamma);
        }
        case 1:
            return std::exp(gamma / 2.0);
        case 3:
            return 2.0 / (1.0 + std::exp(-gamma));
        default:  // -5
            return 1.0 + gamma >= 0 ? std::sqrt(1.0 + gamma) : -std::sqrt(-1.0 - gamma);
    }
}

double g_and_slope(const M& m, double gamma, int ibar, double& dg) {
    double g;
    switch (ibar) {
        case 0:
        case 4: {
            const double switchpoint = -m.gsmooth_factor / (m.gsmooth_factor + 1);
            if (gamma < switchpoint) {
                g = std::sqrt(1 / (m.gsmooth_factor + 1) *
                              std::pow(switchpoint / gamma, m.gsmooth_factor));
                dg = -m.gsmooth_factor * g / (2.0 * gamma);
                return g;
            }
            g = std::sqrt(1.0 + gamma);
            dg = 1.0 / (2.0 * g);
            return g;
        }
        case 1:
            g = std::exp(gamma / 2.0);
            dg = g / 2.0;
            return g;
        case 3:
            g = 2.0 / (1.0 + std::exp(-gamma));
            dg = g * (2.0 - g) / 2;
            return g;
        default:  // -5
            if (1.0 + gamma >= 0) {
                g = std::sqrt(1.0 + gamma);
                dg = 1.0 / (2.0 * g);
            } else {
                g = -std::sqrt(-1.0 - gamma);
                dg = -1.0 / (2.0 * g);
            }
            return g;
    }
}

double embedding(const M& m, int a, double rhobar, double& df) {
    const double aec = m.of(M::A, a) * m.of(M::EC, a);
    if (rhobar > 0.0) {
        const double lrb = std::log(rhobar);
        df = aec * (1.0 + lrb);
        return aec * rhobar * lrb;
    }
    if (m.emb_lin_neg == 0) {
        df = 0.0;
        return 0.0;
    }
    df = -aec;
    return -aec * rhobar;
}

constexpr int kV2[6] = {1, 2, 2, 1, 2, 1};
constexpr int kV3[10] = {1, 3, 3, 3, 6, 3, 1, 3, 3, 1};
// the index of a symmetric 2- and 3-index component in 6 and 10
constexpr int kVind2[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
constexpr int kVind3[3][3][3] = {{{0, 1, 2}, {1, 3, 4}, {2, 4, 5}},
                                 {{1, 3, 4}, {3, 6, 7}, {4, 7, 8}},
                                 {{2, 4, 5}, {4, 7, 8}, {5, 8, 9}}};

// What each atom holds after the densities
struct Atoms {
    std::vector<double> rho0, rho1, rho2, rho3, gamma, dgamma1, dgamma2, dgamma3, frhop;
    std::vector<double> arho1, arho2, arho3, arho2b, arho3b, t_ave, tsq_ave;

    explicit Atoms(std::size_t n)
        : rho0(n), rho1(n), rho2(n), rho3(n), gamma(n), dgamma1(n), dgamma2(n), dgamma3(n),
          frhop(n), arho1(3 * n), arho2(6 * n), arho3(10 * n), arho2b(n), arho3b(3 * n),
          t_ave(3 * n), tsq_ave(3 * n) {}
};

}  // namespace

MeamPotential make_meam_potential(int elements, const double* element, const double* ebound,
                                  const double* cmin, const double* cmax, const double* tables,
                                  int nr, const int* options, const double* numbers) {
    MeamPotential m;
    const auto e = static_cast<std::size_t>(elements);
    m.elements = elements;
    m.ialloy = options[0];
    m.mix_ref_t = options[1];
    m.emb_lin_neg = options[2];
    m.bkgd_dyn = options[3];
    m.rc = numbers[0];
    m.delr = numbers[1];
    m.gsmooth_factor = numbers[2];
    m.dr = numbers[3];
    m.cutoff = numbers[4];
    m.nr = nr;
    if (!(m.rc > 0.0) || !(m.delr > 0.0) || !(m.dr > 0.0) || !(m.cutoff >= m.rc)) {
        throw std::invalid_argument("MEAM cutoffs and spacings must be positive");
    }
    m.element.assign(element, element + 17 * e);
    m.ebound.assign(ebound, ebound + e * e);
    m.cmin.assign(cmin, cmin + e * e * e);
    m.cmax.assign(cmax, cmax + e * e * e);
    m.table.assign(tables, tables + e * (e + 1) / 2 * 7 * static_cast<std::size_t>(nr));
    return m;
}

void compute_meam(const MeamPotential& m, std::int64_t n, const std::int32_t* species,
                  const double* positions, const double* matrix, const std::int64_t* offsets,
                  const index_t* neighbors, const std::int8_t* shifts, double* energies,
                  double* forces, double* virials, const Control& control) {
    const std::int64_t pairs = offsets[n];
    const auto np = static_cast<std::size_t>(pairs);
    const double rc2 = m.rc * m.rc, drinv = 1.0 / m.delr;
    const std::vector<std::int64_t> reverse =
        reverse_pairs(n, offsets, neighbors, shifts, {control.cancel, {}});
    const auto vector_of = [&](std::int64_t i, std::int64_t p, double* r) {
        return neighbor::pair_vector(positions + 3 * i,
                                     positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                                     shifts + 3 * p, matrix, r);
    };

    // 1. The screening of each pair within rc, from one of its two ends
    std::vector<double> scr(np, 0.0), fcp(np, 0.0), dscr(np, 0.0);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const int ei = species[i];
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    if (reverse[static_cast<std::size_t>(p)] < p) {
                        continue;
                    }
                    double rij_v[3];
                    const double rij2 = vector_of(i, p, rij_v);
                    if (rij2 > rc2) {
                        continue;
                    }
                    const int ej = species[neighbors[p]];
                    const double rbound = m.ebound[static_cast<std::size_t>(ei * m.elements + ej)] * rij2;
                    const double rij = std::sqrt(rij2);
                    const double rnorm = (m.rc - rij) * drinv;
                    double sij = 1.0;
                    for (std::int64_t q = offsets[i]; q < offsets[i + 1]; ++q) {
                        if (q == p) {
                            continue;
                        }
                        double rik_v[3];
                        const double rik2 = vector_of(i, q, rik_v);
                        const double rjk_v[3] = {rik_v[0] - rij_v[0], rik_v[1] - rij_v[1],
                                                 rik_v[2] - rij_v[2]};
                        const double rjk2 =
                            rjk_v[0] * rjk_v[0] + rjk_v[1] * rjk_v[1] + rjk_v[2] * rjk_v[2];
                        if (rjk2 > rbound || rik2 > rbound) {
                            continue;
                        }
                        const double xik = rik2 / rij2, xjk = rjk2 / rij2;
                        const double a = 1 - (xik - xjk) * (xik - xjk);
                        if (a <= 0.0) {
                            continue;
                        }
                        double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
                        const std::size_t t = m.triple(ei, ej, species[neighbors[q]]);
                        const double cmax = m.cmax[t], cmin = m.cmin[t];
                        if (cikj >= cmax) {
                            continue;
                        }
                        if (cikj <= cmin) {
                            sij = 0.0;
                            break;
                        }
                        sij *= cutoff((cikj - cmin) / (cmax - cmin));
                    }
                    double dfc;
                    const double fcij = cutoff_and_slope(rnorm, dfc);
                    const double dfcij = dfc * drinv;
                    double dsij = 0.0;
                    const double sfcij = sij * fcij;
                    if (!is_zero(sfcij) && !is_one(sfcij)) {
                        for (std::int64_t q = offsets[i]; q < offsets[i + 1]; ++q) {
                            if (q == p) {
                                continue;
                            }
                            double rik_v[3];
                            const double rik2 = vector_of(i, q, rik_v);
                            const double rjk_v[3] = {rik_v[0] - rij_v[0], rik_v[1] - rij_v[1],
                                                     rik_v[2] - rij_v[2]};
                            const double rjk2 =
                                rjk_v[0] * rjk_v[0] + rjk_v[1] * rjk_v[1] + rjk_v[2] * rjk_v[2];
                            if (rjk2 > rbound || rik2 > rbound) {
                                continue;
                            }
                            const double xik = rik2 / rij2, xjk = rjk2 / rij2;
                            const double a = 1 - (xik - xjk) * (xik - xjk);
                            if (a <= 0.0) {
                                continue;
                            }
                            double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
                            const std::size_t t = m.triple(ei, ej, species[neighbors[q]]);
                            const double cmax = m.cmax[t], cmin = m.cmin[t];
                            if (cikj >= cmax) {
                                continue;
                            }
                            const double delc = cmax - cmin;
                            double dfikj;
                            const double sikj = cutoff_and_slope((cikj - cmin) / delc, dfikj);
                            dsij += dfikj / (delc * sikj) * dc_by_rij(rij2, rik2, rjk2);
                        }
                        dsij = dsij * sfcij - sij * dfcij / rij;
                    }
                    const auto k = static_cast<std::size_t>(p);
                    scr[k] = sij;
                    fcp[k] = fcij;
                    dscr[k] = dsij;
                }
            }
        },
        stage(control, 0, 3));
    for (std::size_t p = 0; p < np; ++p) {  // the other end of each pair
        const auto r = static_cast<std::size_t>(reverse[p]);
        if (r < p) {
            scr[p] = scr[r];
            fcp[p] = fcp[r];
            dscr[p] = dscr[r];
        }
    }

    // 2. The partial densities of each atom, and its embedding energy
    const auto atoms = static_cast<std::size_t>(n);
    Atoms at(atoms);
    Shares shares(n, pairs);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                const int ei = species[i];
                double* arho1 = &at.arho1[3 * ii];
                double* arho2 = &at.arho2[6 * ii];
                double* arho3 = &at.arho3[10 * ii];
                double* arho3b = &at.arho3b[3 * ii];
                double* t_ave = &at.t_ave[3 * ii];
                double* tsq_ave = &at.tsq_ave[3 * ii];
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const auto pp = static_cast<std::size_t>(p);
                    if (is_zero(scr[pp])) {
                        continue;
                    }
                    double d[3];
                    const double rij2 = vector_of(i, p, d);
                    if (!(rij2 < rc2)) {
                        continue;
                    }
                    const int ej = species[neighbors[p]];
                    const double sij = scr[pp] * fcp[pp];
                    const double rij = std::sqrt(rij2);
                    const double aj = rij / m.of(M::RE, ej) - 1.0;
                    const double ro0j = m.of(M::RHO0, ej);
                    const double rhoa0j = ro0j * std::exp(-m.of(M::BETA0, ej) * aj) * sij;
                    double rhoa1j = ro0j * std::exp(-m.of(M::BETA1, ej) * aj) * sij;
                    double rhoa2j = ro0j * std::exp(-m.of(M::BETA2, ej) * aj) * sij;
                    double rhoa3j = ro0j * std::exp(-m.of(M::BETA3, ej) * aj) * sij;
                    const double t1j = m.of(M::T1, ej), t2j = m.of(M::T2, ej),
                                 t3j = m.of(M::T3, ej);
                    if (m.ialloy == 1) {
                        rhoa1j *= t1j;
                        rhoa2j *= t2j;
                        rhoa3j *= t3j;
                    }
                    at.rho0[ii] += rhoa0j;
                    if (m.ialloy != 2) {
                        t_ave[0] += t1j * rhoa0j;
                        t_ave[1] += t2j * rhoa0j;
                        t_ave[2] += t3j * rhoa0j;
                    }
                    if (m.ialloy == 1) {
                        tsq_ave[0] += t1j * t1j * rhoa0j;
                        tsq_ave[1] += t2j * t2j * rhoa0j;
                        tsq_ave[2] += t3j * t3j * rhoa0j;
                    }
                    at.arho2b[ii] += rhoa2j;
                    const double a1 = rhoa1j / rij, a2 = rhoa2j / rij2, a3 = rhoa3j / (rij2 * rij);
                    int nv2 = 0, nv3 = 0;
                    for (int x = 0; x < 3; ++x) {
                        arho1[x] += a1 * d[x];
                        arho3b[x] += rhoa3j * d[x] / rij;
                        for (int y = x; y < 3; ++y) {
                            arho2[nv2] += a2 * d[x] * d[y];
                            ++nv2;
                            for (int z = y; z < 3; ++z) {
                                arho3[nv3] += a3 * d[x] * d[y] * d[z];
                                ++nv3;
                            }
                        }
                    }
                }
                // the densities combined
                double rho1 = 0.0, rho2 = -1.0 / 3.0 * at.arho2b[ii] * at.arho2b[ii], rho3 = 0.0;
                for (int x = 0; x < 3; ++x) {
                    rho1 += arho1[x] * arho1[x];
                    rho3 -= 3.0 / 5.0 * arho3b[x] * arho3b[x];
                }
                for (int k = 0; k < 6; ++k) {
                    rho2 += kV2[k] * arho2[k] * arho2[k];
                }
                for (int k = 0; k < 10; ++k) {
                    rho3 += kV3[k] * arho3[k] * arho3[k];
                }
                const double rho0 = at.rho0[ii];
                if (rho0 > 0.0) {
                    for (int k = 0; k < 3; ++k) {
                        if (m.ialloy == 1) {
                            t_ave[k] = divide_or_zero(t_ave[k], tsq_ave[k]);
                        } else if (m.ialloy == 2) {
                            t_ave[k] = m.of(static_cast<M::Row>(M::T1 + k), ei);
                        } else {
                            t_ave[k] = t_ave[k] / rho0;
                        }
                    }
                }
                double gamma = t_ave[0] * rho1 + t_ave[1] * rho2 + t_ave[2] * rho3;
                if (rho0 > 0.0) {
                    gamma = gamma / (rho0 * rho0);
                }
                const double z = m.of(M::Z, ei);
                const int ibar = static_cast<int>(m.of(M::IBAR, ei));
                const double shp[3] = {m.of(M::S1, ei), m.of(M::S2, ei), m.of(M::S3, ei)};
                const double g = g_of_gamma(m, gamma, ibar);
                const double rho = rho0 * g;
                double rho_bkgd, gbar = 1.0, dgbar = 0.0;
                if (m.mix_ref_t == 1) {
                    if (ibar > 0) {
                        const double gam =
                            (t_ave[0] * shp[0] + t_ave[1] * shp[1] + t_ave[2] * shp[2]) / (z * z);
                        gbar = g_and_slope(m, gam, ibar, dgbar);
                    }
                    rho_bkgd = m.of(M::RHO0, ei) * z * gbar;
                } else if (m.bkgd_dyn == 1) {
                    rho_bkgd = m.of(M::RHO0, ei) * z;
                } else {
                    rho_bkgd = m.of(M::RHO_REF, ei);
                }
                const double rhob = rho / rho_bkgd, denom = 1.0 / rho_bkgd;
                double dg;
                const double g2 = g_and_slope(m, gamma, ibar, dg);
                at.gamma[ii] = gamma;
                at.rho1[ii] = rho1;
                at.rho2[ii] = rho2;
                at.rho3[ii] = rho3;
                at.dgamma1[ii] = (g2 - 2 * dg * gamma) * denom;
                at.dgamma2[ii] = is_zero(rho0) ? 0.0 : (dg / rho0) * denom;
                at.dgamma3[ii] = m.mix_ref_t == 1 ? rho0 * g2 * dgbar / (gbar * z * z) * denom : 0.0;
                shares.own_energy[ii] += embedding(m, ei, rhob, at.frhop[ii]);
            }
        },
        stage(control, 1, 3));

    // 3. The forces of each pair, from both ends, and of the screening, from one
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                const int ei = species[i];
                double* own_force = &shares.own_force[3 * ii];
                double* own_virial = &shares.own_virial[9 * ii];
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const auto pp = static_cast<std::size_t>(p);
                    if (is_zero(scr[pp])) {
                        continue;
                    }
                    double delij[3];
                    const double rij2 = vector_of(i, p, delij);
                    if (!(rij2 < rc2)) {
                        continue;
                    }
                    const auto j = static_cast<std::int64_t>(neighbors[p]);
                    const auto jj = static_cast<std::size_t>(j);
                    const int ej = species[j];
                    double sij = scr[pp] * fcp[pp];
                    const double rij = std::sqrt(rij2), recip = 1.0 / rij;
                    // the pair function from its table
                    const double* tab = m.pair_table(ei, ej);
                    const auto row = [&](int r) { return tab + static_cast<std::size_t>(r) * static_cast<std::size_t>(m.nr); };
                    double ppos = rij / m.dr;
                    int kk = std::min(static_cast<int>(ppos), m.nr - 2);
                    ppos = std::min(ppos - kk, 1.0);
                    const double phi =
                        ((row(3)[kk] * ppos + row(2)[kk]) * ppos + row(1)[kk]) * ppos + row(0)[kk];
                    const double phip = (row(6)[kk] * ppos + row(5)[kk]) * ppos + row(4)[kk];
                    shares.own_energy[ii] += 0.5 * phi * sij;

                    const double invrei = 1.0 / m.of(M::RE, ei);
                    const double ai = rij * invrei - 1.0, ro0i = m.of(M::RHO0, ei);
                    double rhoa0i = ro0i * std::exp(-m.of(M::BETA0, ei) * ai);
                    double drhoa0i = -m.of(M::BETA0, ei) * invrei * rhoa0i;
                    double rhoa1i = ro0i * std::exp(-m.of(M::BETA1, ei) * ai);
                    double drhoa1i = -m.of(M::BETA1, ei) * invrei * rhoa1i;
                    double rhoa2i = ro0i * std::exp(-m.of(M::BETA2, ei) * ai);
                    double drhoa2i = -m.of(M::BETA2, ei) * invrei * rhoa2i;
                    double rhoa3i = ro0i * std::exp(-m.of(M::BETA3, ei) * ai);
                    double drhoa3i = -m.of(M::BETA3, ei) * invrei * rhoa3i;
                    double rhoa0j, drhoa0j, rhoa1j, drhoa1j, rhoa2j, drhoa2j, rhoa3j, drhoa3j;
                    if (ei != ej) {
                        const double invrej = 1.0 / m.of(M::RE, ej);
                        const double aj = rij * invrej - 1.0, ro0j = m.of(M::RHO0, ej);
                        rhoa0j = ro0j * std::exp(-m.of(M::BETA0, ej) * aj);
                        drhoa0j = -m.of(M::BETA0, ej) * invrej * rhoa0j;
                        rhoa1j = ro0j * std::exp(-m.of(M::BETA1, ej) * aj);
                        drhoa1j = -m.of(M::BETA1, ej) * invrej * rhoa1j;
                        rhoa2j = ro0j * std::exp(-m.of(M::BETA2, ej) * aj);
                        drhoa2j = -m.of(M::BETA2, ej) * invrej * rhoa2j;
                        rhoa3j = ro0j * std::exp(-m.of(M::BETA3, ej) * aj);
                        drhoa3j = -m.of(M::BETA3, ej) * invrej * rhoa3j;
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
                    const double t1mi = m.of(M::T1, ei), t2mi = m.of(M::T2, ei),
                                 t3mi = m.of(M::T3, ei);
                    const double t1mj = m.of(M::T1, ej), t2mj = m.of(M::T2, ej),
                                 t3mj = m.of(M::T3, ej);
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
                    const double* arho1i = &at.arho1[3 * ii];
                    const double* arho1j = &at.arho1[3 * jj];
                    const double* arho2i = &at.arho2[6 * ii];
                    const double* arho2j = &at.arho2[6 * jj];
                    const double* arho3i = &at.arho3[10 * ii];
                    const double* arho3j = &at.arho3[10 * jj];
                    const double* arho3bi = &at.arho3b[3 * ii];
                    const double* arho3bj = &at.arho3b[3 * jj];
                    int nv2 = 0, nv3 = 0;
                    double arg1i1 = 0, arg1j1 = 0, arg1i2 = 0, arg1j2 = 0, arg1i3 = 0, arg1j3 = 0,
                           arg3i3 = 0, arg3j3 = 0;
                    for (int x = 0; x < 3; ++x) {
                        for (int y = x; y < 3; ++y) {
                            for (int z = y; z < 3; ++z) {
                                const double arg = delij[x] * delij[y] * delij[z] * kV3[nv3];
                                arg1i3 += arho3i[nv3] * arg;
                                arg1j3 -= arho3j[nv3] * arg;
                                ++nv3;
                            }
                            const double arg = delij[x] * delij[y] * kV2[nv2];
                            arg1i2 += arho2i[nv2] * arg;
                            arg1j2 += arho2j[nv2] * arg;
                            ++nv2;
                        }
                        arg1i1 += arho1i[x] * delij[x];
                        arg1j1 -= arho1j[x] * delij[x];
                        arg3i3 += arho3bi[x] * delij[x];
                        arg3j3 -= arho3bj[x] * delij[x];
                    }
                    const double drho0dr1 = drhoa0j * sij, drho0dr2 = drhoa0i * sij;
                    double a1 = 2 * sij / rij;
                    const double drho1dr1 = a1 * (drhoa1j - rhoa1j / rij) * arg1i1;
                    const double drho1dr2 = a1 * (drhoa1i - rhoa1i / rij) * arg1j1;
                    a1 = 2.0 * sij / rij;
                    double drho1drm1[3], drho1drm2[3], drho2drm1[3], drho2drm2[3], drho3drm1[3],
                        drho3drm2[3];
                    for (int x = 0; x < 3; ++x) {
                        drho1drm1[x] = a1 * rhoa1j * arho1i[x];
                        drho1drm2[x] = -a1 * rhoa1i * arho1j[x];
                    }
                    double a2 = 2 * sij / rij2;
                    const double drho2dr1 = a2 * (drhoa2j - 2 * rhoa2j / rij) * arg1i2 -
                                            2.0 / 3.0 * at.arho2b[ii] * drhoa2j * sij;
                    const double drho2dr2 = a2 * (drhoa2i - 2 * rhoa2i / rij) * arg1j2 -
                                            2.0 / 3.0 * at.arho2b[jj] * drhoa2i * sij;
                    a2 = 4 * sij / rij2;
                    for (int x = 0; x < 3; ++x) {
                        drho2drm1[x] = 0.0;
                        drho2drm2[x] = 0.0;
                        for (int y = 0; y < 3; ++y) {
                            drho2drm1[x] += arho2i[kVind2[x][y]] * delij[y];
                            drho2drm2[x] -= arho2j[kVind2[x][y]] * delij[y];
                        }
                        drho2drm1[x] = a2 * rhoa2j * drho2drm1[x];
                        drho2drm2[x] = -a2 * rhoa2i * drho2drm2[x];
                    }
                    const double rij3 = rij * rij2;
                    double a3 = 2 * sij / rij3;
                    double a3a = 6.0 / 5.0 * sij / rij;
                    const double drho3dr1 = a3 * (drhoa3j - 3 * rhoa3j / rij) * arg1i3 -
                                            a3a * (drhoa3j - rhoa3j / rij) * arg3i3;
                    const double drho3dr2 = a3 * (drhoa3i - 3 * rhoa3i / rij) * arg1j3 -
                                            a3a * (drhoa3i - rhoa3i / rij) * arg3j3;
                    a3 = 6 * sij / rij3;
                    a3a = 6 * sij / (5 * rij);
                    for (int x = 0; x < 3; ++x) {
                        drho3drm1[x] = 0.0;
                        drho3drm2[x] = 0.0;
                        nv2 = 0;
                        for (int y = 0; y < 3; ++y) {
                            for (int z = y; z < 3; ++z) {
                                const double arg = delij[y] * delij[z] * kV2[nv2];
                                drho3drm1[x] += arho3i[kVind3[x][y][z]] * arg;
                                drho3drm2[x] += arho3j[kVind3[x][y][z]] * arg;
                                ++nv2;
                            }
                        }
                        drho3drm1[x] = (a3 * drho3drm1[x] - a3a * arho3bi[x]) * rhoa3j;
                        drho3drm2[x] = (-a3 * drho3drm2[x] + a3a * arho3bj[x]) * rhoa3i;
                    }
                    const double* tavi = &at.t_ave[3 * ii];
                    const double* tavj = &at.t_ave[3 * jj];
                    const double t1i = tavi[0], t2i = tavi[1], t3i = tavi[2];
                    const double t1j = tavj[0], t2j = tavj[1], t3j = tavj[2];
                    const double* tsqi = &at.tsq_ave[3 * ii];
                    const double* tsqj = &at.tsq_ave[3 * jj];
                    double dt1dr1 = 0, dt1dr2 = 0, dt2dr1 = 0, dt2dr2 = 0, dt3dr1 = 0, dt3dr2 = 0;
                    if (m.ialloy == 1) {
                        const double a1i = divide_or_zero(drhoa0j * sij, tsqi[0]);
                        const double a1j = divide_or_zero(drhoa0i * sij, tsqj[0]);
                        const double a2i = divide_or_zero(drhoa0j * sij, tsqi[1]);
                        const double a2j = divide_or_zero(drhoa0i * sij, tsqj[1]);
                        const double a3i = divide_or_zero(drhoa0j * sij, tsqi[2]);
                        const double a3j = divide_or_zero(drhoa0i * sij, tsqj[2]);
                        dt1dr1 = a1i * (t1mj - t1i * t1mj * t1mj);
                        dt1dr2 = a1j * (t1mi - t1j * t1mi * t1mi);
                        dt2dr1 = a2i * (t2mj - t2i * t2mj * t2mj);
                        dt2dr2 = a2j * (t2mi - t2j * t2mi * t2mi);
                        dt3dr1 = a3i * (t3mj - t3i * t3mj * t3mj);
                        dt3dr2 = a3j * (t3mi - t3j * t3mi * t3mi);
                    } else if (m.ialloy != 2) {
                        const double ai_ = is_zero(at.rho0[ii]) ? 0.0 : drhoa0j * sij / at.rho0[ii];
                        const double aj_ = is_zero(at.rho0[jj]) ? 0.0 : drhoa0i * sij / at.rho0[jj];
                        dt1dr1 = ai_ * (t1mj - t1i);
                        dt1dr2 = aj_ * (t1mi - t1j);
                        dt2dr1 = ai_ * (t2mj - t2i);
                        dt2dr2 = aj_ * (t2mi - t2j);
                        dt3dr1 = ai_ * (t3mj - t3i);
                        dt3dr2 = aj_ * (t3mi - t3j);
                    }
                    const double shpi[3] = {m.of(M::S1, ei), m.of(M::S2, ei), m.of(M::S3, ei)};
                    const double shpj[3] = {m.of(M::S1, ej), m.of(M::S2, ej), m.of(M::S3, ej)};
                    const double drhodr1 =
                        at.dgamma1[ii] * drho0dr1 +
                        at.dgamma2[ii] * (dt1dr1 * at.rho1[ii] + t1i * drho1dr1 +
                                          dt2dr1 * at.rho2[ii] + t2i * drho2dr1 +
                                          dt3dr1 * at.rho3[ii] + t3i * drho3dr1) -
                        at.dgamma3[ii] * (shpi[0] * dt1dr1 + shpi[1] * dt2dr1 + shpi[2] * dt3dr1);
                    const double drhodr2 =
                        at.dgamma1[jj] * drho0dr2 +
                        at.dgamma2[jj] * (dt1dr2 * at.rho1[jj] + t1j * drho1dr2 +
                                          dt2dr2 * at.rho2[jj] + t2j * drho2dr2 +
                                          dt3dr2 * at.rho3[jj] + t3j * drho3dr2) -
                        at.dgamma3[jj] * (shpj[0] * dt1dr2 + shpj[1] * dt2dr2 + shpj[2] * dt3dr2);
                    double drhodrm1[3], drhodrm2[3];
                    for (int x = 0; x < 3; ++x) {
                        drhodrm1[x] = at.dgamma2[ii] * (t1i * drho1drm1[x] + t2i * drho2drm1[x] +
                                                        t3i * drho3drm1[x]);
                        drhodrm2[x] = at.dgamma2[jj] * (t1j * drho1drm2[x] + t2j * drho2drm2[x] +
                                                        t3j * drho3drm2[x]);
                    }
                    double drhods1 = 0.0, drhods2 = 0.0;
                    if (!is_zero(dscr[pp])) {
                        const double drho0ds1 = rhoa0j, drho0ds2 = rhoa0i;
                        const double b1 = 2.0 / rij, b2 = 2.0 / rij2, b3 = 2.0 / rij3,
                                     b3a = 6.0 / (5.0 * rij);
                        const double drho1ds1 = b1 * rhoa1j * arg1i1;
                        const double drho1ds2 = b1 * rhoa1i * arg1j1;
                        const double drho2ds1 =
                            b2 * rhoa2j * arg1i2 - 2.0 / 3.0 * at.arho2b[ii] * rhoa2j;
                        const double drho2ds2 =
                            b2 * rhoa2i * arg1j2 - 2.0 / 3.0 * at.arho2b[jj] * rhoa2i;
                        const double drho3ds1 = b3 * rhoa3j * arg1i3 - b3a * rhoa3j * arg3i3;
                        const double drho3ds2 = b3 * rhoa3i * arg1j3 - b3a * rhoa3i * arg3j3;
                        double dt1ds1 = 0, dt1ds2 = 0, dt2ds1 = 0, dt2ds2 = 0, dt3ds1 = 0,
                               dt3ds2 = 0;
                        if (m.ialloy == 1) {
                            const double a1i = divide_or_zero(rhoa0j, tsqi[0]);
                            const double a1j = divide_or_zero(rhoa0i, tsqj[0]);
                            const double a2i = divide_or_zero(rhoa0j, tsqi[1]);
                            const double a2j = divide_or_zero(rhoa0i, tsqj[1]);
                            const double a3i = divide_or_zero(rhoa0j, tsqi[2]);
                            const double a3j = divide_or_zero(rhoa0i, tsqj[2]);
                            dt1ds1 = a1i * (t1mj - t1i * t1mj * t1mj);
                            dt1ds2 = a1j * (t1mi - t1j * t1mi * t1mi);
                            dt2ds1 = a2i * (t2mj - t2i * t2mj * t2mj);
                            dt2ds2 = a2j * (t2mi - t2j * t2mi * t2mi);
                            dt3ds1 = a3i * (t3mj - t3i * t3mj * t3mj);
                            dt3ds2 = a3j * (t3mi - t3j * t3mi * t3mi);
                        } else if (m.ialloy != 2) {
                            const double ai_ = is_zero(at.rho0[ii]) ? 0.0 : rhoa0j / at.rho0[ii];
                            const double aj_ = is_zero(at.rho0[jj]) ? 0.0 : rhoa0i / at.rho0[jj];
                            dt1ds1 = ai_ * (t1mj - t1i);
                            dt1ds2 = aj_ * (t1mi - t1j);
                            dt2ds1 = ai_ * (t2mj - t2i);
                            dt2ds2 = aj_ * (t2mi - t2j);
                            dt3ds1 = ai_ * (t3mj - t3i);
                            dt3ds2 = aj_ * (t3mi - t3j);
                        }
                        drhods1 = at.dgamma1[ii] * drho0ds1 +
                                  at.dgamma2[ii] * (dt1ds1 * at.rho1[ii] + t1i * drho1ds1 +
                                                    dt2ds1 * at.rho2[ii] + t2i * drho2ds1 +
                                                    dt3ds1 * at.rho3[ii] + t3i * drho3ds1) -
                                  at.dgamma3[ii] *
                                      (shpi[0] * dt1ds1 + shpi[1] * dt2ds1 + shpi[2] * dt3ds1);
                        drhods2 = at.dgamma1[jj] * drho0ds2 +
                                  at.dgamma2[jj] * (dt1ds2 * at.rho1[jj] + t1j * drho1ds2 +
                                                    dt2ds2 * at.rho2[jj] + t2j * drho2ds2 +
                                                    dt3ds2 * at.rho3[jj] + t3j * drho3ds2) -
                                  at.dgamma3[jj] *
                                      (shpj[0] * dt1ds2 + shpj[1] * dt2ds2 + shpj[2] * dt3ds2);
                    }
                    const double dudrij =
                        phip * sij + at.frhop[ii] * drhodr1 + at.frhop[jj] * drhodr2;
                    const double dudsij = is_zero(dscr[pp])
                                              ? 0.0
                                              : phi + at.frhop[ii] * drhods1 + at.frhop[jj] * drhods2;
                    double dudrijm[3];
                    for (int x = 0; x < 3; ++x) {
                        dudrijm[x] = at.frhop[ii] * drhodrm1[x] + at.frhop[jj] * drhodrm2[x];
                    }
                    const double force = dudrij * recip + dudsij * dscr[pp];
                    double fi[3];
                    for (int x = 0; x < 3; ++x) {
                        fi[x] = delij[x] * force + dudrijm[x];
                        own_force[x] += fi[x];
                    }
                    // the virial of the pair, half to each atom, symmetric
                    for (int x = 0; x < 3; ++x) {
                        for (int y = 0; y < 3; ++y) {
                            own_virial[3 * x + y] += -0.25 * (delij[x] * fi[y] + delij[y] * fi[x]);
                        }
                    }

                    // the screening by each k, once for each pair: a third to each atom
                    if (reverse[pp] < p || is_zero(sij) || is_one(sij)) {
                        continue;
                    }
                    double* tj = &shares.derivative[3 * pp];
                    double* wj = &shares.virial[9 * pp];
                    const std::size_t eb = static_cast<std::size_t>(ei * m.elements + ej);
                    for (std::int64_t q = offsets[i]; q < offsets[i + 1]; ++q) {
                        if (q == p) {
                            continue;
                        }
                        const auto qq = static_cast<std::size_t>(q);
                        const std::size_t t = m.triple(ei, ej, species[neighbors[q]]);
                        const double cmax = m.cmax[t], cmin = m.cmin[t];
                        sij = scr[pp] * fcp[pp];
                        double dsij1 = 0.0, dsij2 = 0.0;
                        double dik[3] = {0, 0, 0}, djk[3] = {0, 0, 0};
                        if (!is_zero(sij) && !is_one(sij)) {
                            const double rbound = rij2 * m.ebound[eb];
                            const double delc = cmax - cmin;
                            const double rik2 = vector_of(i, q, dik);
                            for (int x = 0; x < 3; ++x) {
                                djk[x] = dik[x] - delij[x];
                            }
                            const double rjk2 = djk[0] * djk[0] + djk[1] * djk[1] + djk[2] * djk[2];
                            if (rjk2 <= rbound && rik2 <= rbound) {
                                const double xik = rik2 / rij2, xjk = rjk2 / rij2;
                                const double a = 1 - (xik - xjk) * (xik - xjk);
                                if (!is_zero(a)) {
                                    double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
                                    if (cikj >= cmin && cikj <= cmax) {
                                        double dfc;
                                        const double sikj =
                                            cutoff_and_slope((cikj - cmin) / delc, dfc);
                                        double d1, d2;
                                        dc_by_rik_rjk(rij2, rik2, rjk2, d1, d2);
                                        const double coef = sij / delc * dfc / sikj;
                                        dsij1 = coef * d1;
                                        dsij2 = coef * d2;
                                    }
                                }
                            }
                        }
                        if (is_zero(dsij1) && is_zero(dsij2)) {
                            continue;
                        }
                        const double force1 = dudsij * dsij1, force2 = dudsij * dsij2;
                        double f_i[3], f_j[3], f_k[3];
                        for (int x = 0; x < 3; ++x) {
                            f_i[x] = force1 * dik[x];
                            f_j[x] = force2 * djk[x];
                            f_k[x] = -(f_i[x] + f_j[x]);
                        }
                        double* tk = &shares.derivative[3 * qq];
                        double* wk = &shares.virial[9 * qq];
                        for (int x = 0; x < 3; ++x) {
                            tj[x] -= f_j[x];
                            tk[x] -= f_k[x];
                        }
                        for (int x = 0; x < 3; ++x) {
                            for (int y = 0; y < 3; ++y) {
                                const double w = -(1.0 / 6.0) * (dik[x] * f_i[y] + djk[x] * f_j[y] +
                                                                 dik[y] * f_i[x] + djk[y] * f_j[x]);
                                own_virial[3 * x + y] += w;
                                wj[3 * x + y] += w;
                                wk[3 * x + y] += w;
                            }
                        }
                    }
                }
            }
        },
        stage(control, 2, 3));
    gather(shares, n, offsets, neighbors, shifts, energies, forces, virials, {control.cancel, {}});
}

}  // namespace mdapy::calculators
