// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/d3.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "neighbor/frame.h"

namespace mdapy::calculators {

namespace {

// The steepness of the counting function of the coordination number and the
// width of the Gaussian weights of the references.
constexpr double kCountSteepness = 16.0;
constexpr double kWeightWidth = 4.0;
// The scale of R0AB in the three-body damping.
constexpr double kAtmScale = 4.0 / 3.0;
// Pairs closer than 1e-6 Å, coincident atoms, are passed over.
constexpr double kClosest2 = 1e-12;

// The count of a pair of the coordination number at distance r, and its
// derivative with r.
inline void count(double rc, double r2, double& value, double& slope) {
    const double r = std::sqrt(r2);
    value = 1.0 / (1.0 + std::exp(-kCountSteepness * (rc / r - 1.0)));
    slope = -value * (1.0 - value) * kCountSteepness * rc / r2;
}

inline double power(double x, int k) {
    double out = 1.0;
    for (int i = 0; i < k; ++i) out *= x;
    return out;
}

// The damping of a pair at r2 = r^2: e, with the energy of the pair -C6 e, and
// g = (de/dr) / r.
inline void pair_term(const D3Potential& p, std::size_t k, double r2, double& e, double& g) {
    const double ratio = p.ratio[k];
    if (p.damping == D3Damping::Rational) {
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double t6 = 1.0 / (r6 + p.radius6[k]);
        const double t8 = 1.0 / (r4 * r4 + p.radius8[k]);
        e = p.s6 * t6 + p.s8 * ratio * t8;
        g = p.s6 * (-6.0 * r4 * t6 * t6) + p.s8 * ratio * (-8.0 * r6 * t8 * t8);
        return;
    }
    // (R / r)^alpha and (R / r)^(alpha + 2) from (R / r)^2
    const double r6 = r2 * r2 * r2;
    const double r8 = r6 * r2;
    const double q6 = p.radius6[k] * p.radius6[k] / r2;
    const double q8 = p.radius8[k] * p.radius8[k] / r2;
    const double t6 = power(q6, p.half_alpha);
    const double t8 = power(q8, p.half_alpha + 1);
    const double f6 = 1.0 / (1.0 + 6.0 * t6);
    const double f8 = 1.0 / (1.0 + 6.0 * t8);
    const double d6 = (-6.0 * f6 + 6.0 * p.alpha * t6 * f6 * f6) / r2;
    const double d8 = (-8.0 * f8 + 6.0 * (p.alpha + 2.0) * t8 * f8 * f8) / r2;
    e = p.s6 * f6 / r6 + p.s8 * ratio * f8 / r8;
    g = p.s6 * d6 / r6 + p.s8 * ratio * d8 / r8;
}

// A neighbor of an atom within the cutoff of the triangles.
struct Corner {
    double v[3];  // x_j - x_i of its image
    double r2;
    double c6, dc6;  // of the pair, and its derivative with the CN of the atom
    double radius;   // 4/3 R0AB
    index_t j;
};

// The triangles of atom i with two of its neighbors: a third of the energy of
// each, its force on i, a third of its virial, and its derivative with the
// coordination number of i through the two pairs of i. w and u: the weights
// and the contracted C6 of the atoms (see compute_d3).
void triangles(const D3Potential& p, const std::int32_t* species, const double* w,
               const double* u, std::int64_t width, const std::vector<Corner>& corners,
               double& energy, double* f, double* vir, double& dedcn, bool virial) {
    const double cut2 = p.three_body_cutoff * p.three_body_cutoff;
    const double third = 1.0 / 3.0;
    const double exponent = (p.alpha + 2.0) / 3.0;
    const std::size_t m = corners.size();
    for (std::size_t a = 0; a < m; ++a) {
        const Corner& cj = corners[a];
        const int sj = species[cj.j];
        const double* wj = w + kD3References * static_cast<std::int64_t>(cj.j);
        const std::int64_t slot = p.slots[static_cast<std::size_t>(sj)];
        const int nj = p.references[static_cast<std::size_t>(sj)];
        for (std::size_t b = a + 1; b < m; ++b) {
            const Corner& ck = corners[b];
            const double vjk[3] = {ck.v[0] - cj.v[0], ck.v[1] - cj.v[1], ck.v[2] - cj.v[2]};
            const double r2jk = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
            if (!(r2jk < cut2) || r2jk < kClosest2) continue;
            const int sk = species[ck.j];
            const double* uk = u + width * static_cast<std::int64_t>(ck.j) + slot;
            double c6jk = 0.0;
            for (int r = 0; r < nj; ++r) c6jk += wj[r] * uk[r];
            const double r2ij = cj.r2, r2ik = ck.r2;
            const double c9 = -p.s9 * std::sqrt(std::fabs(cj.c6 * ck.c6 * c6jk) / p.hartree);
            const double r0 = cj.radius * ck.radius *
                              p.atm_radius[static_cast<std::size_t>(sj) * p.species + sk];
            const double rr2 = r2ij * r2ik * r2jk;
            const double r1 = std::sqrt(rr2);
            const double r3 = rr2 * r1;
            const double r5 = r3 * rr2;
            const double t = std::pow(r0 / r1, exponent);
            const double fdmp = 1.0 / (1.0 + 6.0 * t);
            const double ang = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) *
                                   (-r2ij + r2jk + r2ik) / r5 +
                               1.0 / r3;
            const double dfdmp = -2.0 * (p.alpha + 2.0) * t * fdmp * fdmp;
            auto dang = [&](double x, double y, double z) {
                // the derivative of ang along the edge of x, the others y and z
                return -0.375 *
                       (x * x * x + x * x * (y + z) + x * (3.0 * y * y + 2.0 * y * z + 3.0 * z * z) -
                        5.0 * (y - z) * (y - z) * (y + z)) /
                       r5;
            };
            const double gij = c9 * (-dang(r2ij, r2jk, r2ik) * fdmp + ang * dfdmp) / r2ij;
            const double gik = c9 * (-dang(r2ik, r2jk, r2ij) * fdmp + ang * dfdmp) / r2ik;
            const double e = -c9 * ang * fdmp;
            energy += third * e;
            dedcn += 0.5 * e * (cj.dc6 / cj.c6 + ck.dc6 / ck.c6);
            for (int c = 0; c < 3; ++c) {
                f[c] += gij * cj.v[c] + gik * ck.v[c];
            }
            if (virial) {
                const double gjk = c9 * (-dang(r2jk, r2ik, r2ij) * fdmp + ang * dfdmp) / r2jk;
                for (int c = 0; c < 3; ++c) {
                    for (int d = 0; d < 3; ++d) {
                        vir[3 * c + d] -= third * (gij * cj.v[c] * cj.v[d] +
                                                   gik * ck.v[c] * ck.v[d] +
                                                   gjk * vjk[c] * vjk[d]);
                    }
                }
            }
        }
    }
}

}  // namespace

D3Potential make_d3_potential(D3Damping damping, int species, const int* references,
                              const double* reference_cn, const double* rcov,
                              const double* r4r2, const double* vdw, const double* c6,
                              std::int64_t c6_size, double s6, double s8, double s9, double a1,
                              double a2, double rs6, double rs8, double alpha, double hartree,
                              double cutoff, double cn_cutoff, double three_body_cutoff) {
    if (species < 1) {
        throw std::invalid_argument("a D3 potential needs one species or more");
    }
    for (const double c : {cutoff, cn_cutoff, three_body_cutoff}) {
        if (!(c > 0.0) || !std::isfinite(c)) {
            throw std::invalid_argument("the cutoffs of D3 must be positive");
        }
    }
    if (!(alpha > 0.0 && alpha <= 40.0) || alpha != 2.0 * std::floor(alpha / 2.0)) {
        throw std::invalid_argument("the exponent of zero damping must be an even number");
    }
    D3Potential p;
    p.damping = damping;
    p.species = species;
    p.s6 = s6;
    p.s8 = s8;
    p.s9 = s9;
    p.alpha = alpha;
    p.half_alpha = static_cast<int>(alpha / 2.0);
    p.hartree = hartree;
    p.cutoff = cutoff;
    p.cn_cutoff = cn_cutoff;
    p.three_body_cutoff = three_body_cutoff;
    p.reach = std::max(cutoff, cn_cutoff);
    if (s9 != 0.0) p.reach = std::max(p.reach, three_body_cutoff);
    const auto s = static_cast<std::size_t>(species);
    p.references.assign(references, references + s);
    p.reference_cn.assign(reference_cn, reference_cn + s * kD3References);
    p.rcov.assign(rcov, rcov + s);
    p.ratio.resize(s * s);
    p.radius6.resize(s * s);
    p.radius8.resize(s * s);
    p.atm_radius.resize(s * s);
    p.blocks.resize(s * s);
    p.slots.assign(s + 1, 0);
    std::int64_t offset = 0;
    for (std::size_t i = 0; i < s; ++i) {
        if (p.references[i] < 1 || p.references[i] > kD3References) {
            throw std::invalid_argument("each species of D3 has 1 to 7 references");
        }
        p.slots[i + 1] = p.slots[i] + p.references[i];
        for (std::size_t j = 0; j < s; ++j) {
            const std::size_t k = i * s + j;
            const double ratio = 3.0 * r4r2[i] * r4r2[j];
            p.ratio[k] = ratio;
            if (damping == D3Damping::Rational) {
                const double r0 = a1 * std::sqrt(ratio) + a2;
                const double r02 = r0 * r0;
                p.radius6[k] = r02 * r02 * r02;
                p.radius8[k] = r02 * r02 * r02 * r02;
            } else {
                p.radius6[k] = rs6 * vdw[k];
                p.radius8[k] = rs8 * vdw[k];
            }
            p.atm_radius[k] = kAtmScale * vdw[k];
            p.blocks[k] = offset;
            offset += static_cast<std::int64_t>(p.references[i]) * p.references[j];
        }
    }
    if (offset != c6_size) {
        throw std::invalid_argument("the C6 table of D3 does not fit its references");
    }
    p.c6.assign(c6, c6 + c6_size);
    return p;
}

void compute_d3(const D3Potential& p, std::int64_t n, const std::int32_t* species,
                const double* positions, const double* matrix, const std::int64_t* offsets,
                const index_t* neighbors, const std::int8_t* shifts, double* energies,
                double* forces, double* virials, const Control& control) {
    const auto s = static_cast<std::size_t>(p.species);
    const bool virial = virials != nullptr;
    const bool atm = p.s9 != 0.0;
    const double cut2 = p.cutoff * p.cutoff;
    const double cn2 = p.cn_cutoff * p.cn_cutoff;
    const double atm2 = p.three_body_cutoff * p.three_body_cutoff;
    // the contracted C6 of each atom k, of each species t: u_k[slot t + a] =
    // sum_b C6(t, s_k)[a][b] w_k[b], so that the C6 of a pair (i, k) is
    // w_i . u_k[slot s_i] and its derivative with CN_i dw_i . u_k[slot s_i]
    const std::int64_t width = p.slots[s];
    std::vector<double> cn(static_cast<std::size_t>(n), 0.0);
    std::vector<double> w(static_cast<std::size_t>(n) * kD3References, 0.0);
    std::vector<double> dw(static_cast<std::size_t>(n) * kD3References, 0.0);
    std::vector<double> u(static_cast<std::size_t>(n * width), 0.0);
    std::vector<double> dedcn(static_cast<std::size_t>(n), 0.0);
    auto vector_of = [&](std::int64_t i, std::int64_t q, double v[3]) {
        neighbor::pair_vector(positions + 3 * i,
                              positions + 3 * static_cast<std::int64_t>(neighbors[q]),
                              shifts + 3 * q, matrix, v);
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    };
    // the coordination numbers, the weights of the references, their
    // derivatives with the CN, and the contracted C6
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const std::size_t si = static_cast<std::size_t>(species[i]);
                double sum = 0.0;
                for (std::int64_t q = offsets[i]; q < offsets[i + 1]; ++q) {
                    double v[3];
                    const double r2 = vector_of(i, q, v);
                    if (!(r2 < cn2) || r2 < kClosest2) continue;
                    const double rc =
                        p.rcov[si] + p.rcov[static_cast<std::size_t>(species[neighbors[q]])];
                    double value, slope;
                    count(rc, r2, value, slope);
                    sum += value;
                }
                cn[static_cast<std::size_t>(i)] = sum;
                const int m = p.references[si];
                const double* ref = p.reference_cn.data() + si * kD3References;
                double d[kD3References], g[kD3References];
                double least = std::numeric_limits<double>::infinity();
                for (int a = 0; a < m; ++a) {
                    d[a] = ref[a] - sum;
                    least = std::min(least, d[a] * d[a]);
                }
                double norm = 0.0;
                for (int a = 0; a < m; ++a) {
                    g[a] = std::exp(-kWeightWidth * (d[a] * d[a] - least));
                    norm += g[a];
                }
                double* wi = w.data() + kD3References * i;
                double* dwi = dw.data() + kD3References * i;
                double mean = 0.0;
                for (int a = 0; a < m; ++a) {
                    wi[a] = g[a] / norm;
                    mean += wi[a] * 2.0 * kWeightWidth * d[a];
                }
                for (int a = 0; a < m; ++a) {
                    dwi[a] = wi[a] * (2.0 * kWeightWidth * d[a] - mean);
                }
                double* ui = u.data() + width * i;
                for (std::size_t t = 0; t < s; ++t) {
                    const double* block = p.c6.data() + p.blocks[t * s + si];
                    const int nt = p.references[t];
                    for (int a = 0; a < nt; ++a) {
                        double sum_b = 0.0;
                        for (int b = 0; b < m; ++b) sum_b += block[a * m + b] * wi[b];
                        ui[p.slots[t] + a] = sum_b;
                    }
                }
            }
        },
        stage(control, 0, 3));
    // the pairs and the triangles
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<Corner> corners;
            for (std::int64_t i = begin; i < end; ++i) {
                const int si = species[i];
                const int ni = p.references[static_cast<std::size_t>(si)];
                const std::int64_t slot = p.slots[static_cast<std::size_t>(si)];
                const double* wi = w.data() + kD3References * i;
                const double* dwi = dw.data() + kD3References * i;
                double f[3] = {0.0, 0.0, 0.0};
                double vir[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                double energy = 0.0, de = 0.0;
                corners.clear();
                for (std::int64_t q = offsets[i]; q < offsets[i + 1]; ++q) {
                    double v[3];
                    const double r2 = vector_of(i, q, v);
                    if (r2 < kClosest2) continue;
                    const bool pair = r2 < cut2;
                    const bool corner = atm && r2 < atm2;
                    if (!pair && !corner) continue;
                    const index_t j = neighbors[q];
                    const std::size_t sj = static_cast<std::size_t>(species[j]);
                    const double* uj = u.data() + width * static_cast<std::int64_t>(j) + slot;
                    double c6 = 0.0, dc6 = 0.0;
                    for (int a = 0; a < ni; ++a) {
                        c6 += wi[a] * uj[a];
                        dc6 += dwi[a] * uj[a];
                    }
                    const std::size_t k = static_cast<std::size_t>(si) * s + sj;
                    if (corner) {
                        corners.push_back({{v[0], v[1], v[2]}, r2, c6, dc6, p.atm_radius[k], j});
                    }
                    if (!pair) continue;
                    double e, g;
                    pair_term(p, k, r2, e, g);
                    energy -= 0.5 * c6 * e;
                    de -= dc6 * e;
                    // the force on i along x_i - x_j = -v
                    const double fpair = -c6 * g;
                    for (int a = 0; a < 3; ++a) {
                        f[a] += v[a] * fpair;
                    }
                    if (virial) {
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                vir[3 * a + b] += 0.5 * v[a] * v[b] * c6 * g;
                            }
                        }
                    }
                }
                if (atm) {
                    triangles(p, species, w.data(), u.data(), width, corners, energy, f, vir,
                              de, virial);
                }
                energies[i] = energy;
                dedcn[static_cast<std::size_t>(i)] = de;
                for (int a = 0; a < 3; ++a) {
                    forces[3 * i + a] = f[a];
                }
                if (virial) {
                    for (int a = 0; a < 9; ++a) {
                        virials[9 * i + a] = vir[a];
                    }
                }
            }
        },
        stage(control, 1, 3));
    // the coordination numbers through the C6
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const std::size_t si = static_cast<std::size_t>(species[i]);
                const double dei = dedcn[static_cast<std::size_t>(i)];
                double f[3] = {0.0, 0.0, 0.0};
                double vir[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                for (std::int64_t q = offsets[i]; q < offsets[i + 1]; ++q) {
                    double v[3];
                    const double r2 = vector_of(i, q, v);
                    if (!(r2 < cn2) || r2 < kClosest2) continue;
                    const index_t j = neighbors[q];
                    const double rc = p.rcov[si] + p.rcov[static_cast<std::size_t>(species[j])];
                    double value, slope;
                    count(rc, r2, value, slope);
                    // the force on i along x_i - x_j = -v
                    const double fpair =
                        (dei + dedcn[static_cast<std::size_t>(j)]) * slope / std::sqrt(r2);
                    for (int a = 0; a < 3; ++a) {
                        f[a] += v[a] * fpair;
                    }
                    if (virial) {
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                vir[3 * a + b] -= 0.5 * v[a] * v[b] * fpair;
                            }
                        }
                    }
                }
                for (int a = 0; a < 3; ++a) {
                    forces[3 * i + a] += f[a];
                }
                if (virial) {
                    for (int a = 0; a < 9; ++a) {
                        virials[9 * i + a] += vir[a];
                    }
                }
            }
        },
        stage(control, 2, 3));
}

}  // namespace mdapy::calculators
