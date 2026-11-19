// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The Tersoff potential on the device, as calculators/tersoff.cpp computes
// it, in the passes of calculators/many_body.h. A first pass finds the bond
// order of each bond i-j, from the zeta of the third atoms k; the second
// computes, for each pair of a centre, the derivative of the terms of that
// centre with respect to it: its repulsion, its bond, the third atoms of its
// bond, and the bonds it is the third atom of, so that each pair is written
// once; the third gathers the forces.

#include <stdexcept>
#include <vector>

#include "calculators/tersoff.h"
#include "cuda/atoms.cuh"
#include "cuda/many_body.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

template <class Real>
struct TersoffEntry {
    Real m, gamma, lam3, c2, d2, c2d2, h, n, beta, lam2, b, r, big_d, lam1, a, cutsq, c1, c2_, c3,
        c4;
};

template <class Real>
struct TersoffTables {
    const TersoffEntry<Real>* entries;  // (i * e + j) * e + k
    const unsigned char* same;          // i * e + j: the repulsions i j j and j i i agree
    int elements;
    Real cutmax2;
};

template <class Real>
__device__ inline Real cutoff_function(Real r, const TersoffEntry<Real>& p) {
    if (r < p.r - p.big_d) return Real(1);
    if (r > p.r + p.big_d) return Real(0);
    return Real(0.5) * (Real(1) - sin(Real(1.57079632679489661923) * (r - p.r) / p.big_d));
}

template <class Real>
__device__ inline Real cutoff_slope(Real r, const TersoffEntry<Real>& p) {
    if (r < p.r - p.big_d || r > p.r + p.big_d) return Real(0);
    return -(Real(0.78539816339744830962) / p.big_d) *
           cos(Real(1.57079632679489661923) * (r - p.r) / p.big_d);
}

template <class Real>
__device__ inline Real attraction(Real r, const TersoffEntry<Real>& p) {
    if (r > p.r + p.big_d) return Real(0);
    return -p.b * exp(-p.lam2 * r) * cutoff_function(r, p);
}

template <class Real>
__device__ inline Real attraction_slope(Real r, const TersoffEntry<Real>& p) {
    if (r > p.r + p.big_d) return Real(0);
    return p.b * exp(-p.lam2 * r) * (p.lam2 * cutoff_function(r, p) - cutoff_slope(r, p));
}

template <class Real>
__device__ inline Real bond_order(Real zeta, const TersoffEntry<Real>& p) {
    const Real x = p.beta * zeta;
    if (x > p.c1) return Real(1) / sqrt(x);
    if (x > p.c2_) return (Real(1) - pow(x, -p.n) / (Real(2) * p.n)) / sqrt(x);
    if (x < p.c4) return Real(1);
    if (x < p.c3) return Real(1) - pow(x, p.n) / (Real(2) * p.n);
    return pow(Real(1) + pow(x, p.n), Real(-1) / (Real(2) * p.n));
}

template <class Real>
__device__ inline Real bond_order_slope(Real zeta, const TersoffEntry<Real>& p) {
    const Real x = p.beta * zeta;
    if (x > p.c1) return p.beta * Real(-0.5) * pow(x, Real(-1.5));
    if (x > p.c2_) {
        return p.beta * (Real(-0.5) * pow(x, Real(-1.5)) *
                         (Real(1) - (Real(1) + Real(1) / (Real(2) * p.n)) * pow(x, -p.n)));
    }
    if (x < p.c4) return Real(0);
    if (x < p.c3) return Real(-0.5) * p.beta * pow(x, p.n - Real(1));
    const Real xn = pow(x, p.n);
    return Real(-0.5) * pow(Real(1) + xn, Real(-1) - (Real(1) / (Real(2) * p.n))) * xn / zeta;
}

// g(theta) = gamma (1 + c^2/d^2 - c^2 / (d^2 + s^2)), s = h - cos theta, as
// gamma (1 + c^2/d^2 s^2 / (d^2 + s^2)): the same, without the difference
// of two numbers of some 1e7 for the c and d of silicon, which single
// precision would lose
template <class Real>
__device__ inline Real angular(Real cosine, const TersoffEntry<Real>& p) {
    const Real s = p.h - cosine;
    return p.gamma * (Real(1) + p.c2d2 * (s * s) / (p.d2 + s * s));
}

template <class Real>
__device__ inline Real angular_slope(Real cosine, const TersoffEntry<Real>& p) {
    const Real s = p.h - cosine;
    const Real inverse = Real(1) / (p.d2 + s * s);
    return p.gamma * (Real(-2) * p.c2 * s) * inverse * inverse;
}

template <class Real>
__device__ inline Real exponential(Real rij, Real rik, const TersoffEntry<Real>& p) {
    const Real s = p.lam3 * (rij - rik);
    const Real x = p.m == Real(3) ? s * s * s : s;
    if (x > Real(69.0776)) return Real(1.0e30);
    if (x < Real(-69.0776)) return Real(0);
    return exp(x);
}

template <class Real>
__device__ inline Real exponential_slope(Real rij, Real rik, Real value,
                                         const TersoffEntry<Real>& p) {
    if (p.m == Real(3)) {
        const Real d = rij - rik;
        return Real(3) * p.lam3 * p.lam3 * p.lam3 * d * d * value;
    }
    return p.lam3 * value;
}

// The forces on j and k of the term of k in zeta_ij, times prefactor.
template <class Real>
__device__ inline void zeta_forces(Real prefactor, const Real* u, Real rij, const Real* v,
                                   Real rik, const TersoffEntry<Real>& p, Real* fj, Real* fk) {
    const Real fc = cutoff_function(rik, p), dfc = cutoff_slope(rik, p);
    const Real ex = exponential(rij, rik, p), dex = exponential_slope(rij, rik, ex, p);
    const Real cosine = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const Real g = angular(cosine, p), dg = angular_slope(cosine, p);
    for (int a = 0; a < 3; ++a) {
        const Real dcos_j = (v[a] - cosine * u[a]) / rij;
        const Real dcos_k = (u[a] - cosine * v[a]) / rik;
        fj[a] = prefactor * (fc * dg * ex * dcos_j + fc * g * dex * u[a]);
        fk[a] = prefactor * (dfc * g * ex * v[a] + fc * dg * ex * dcos_k - fc * g * dex * v[a]);
    }
}

// x_j - x_i of the pair k of atom i, its length squared and the species of j.
template <class Real>
__device__ inline Real vector_of(std::int64_t n, std::int64_t i, int k, const Real4<Real>* x,
                                 const Real4<Real>& xi, const std::int32_t* indices,
                                 const Real* steps, Real* r, int& species) {
    const auto packed = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
    const Real4<Real> xj = x[packed & kIndexMask];
    r[0] = xj.x - xi.x;
    r[1] = xj.y - xi.y;
    r[2] = xj.z - xi.z;
    add_step(steps, packed >> kIndexBits, r[0], r[1], r[2]);
    species = static_cast<int>(xj.w);
    return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
}

// The bond order b_ij of each bond and -dE/dzeta_ij, in planes of pairs.
template <class Real>
__global__ void tersoff_zeta(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                             const std::int32_t* indices, const std::int32_t* counts,
                             TersoffTables<Real> t, Real* order, Real* prefactor) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int si = static_cast<int>(xi.w);
    const int e = t.elements;
    const int count = counts[i];
    for (int a = 0; a < count; ++a) {
        Real rj[3];
        int sj;
        const Real rsq1 = vector_of(n, i, a, x, xi, indices, steps, rj, sj);
        const TersoffEntry<Real> pij = t.entries[(si * e + sj) * e + sj];
        Real bij = 0, pre = 0;
        if (rsq1 < t.cutmax2 && rsq1 < pij.cutsq) {
            const Real rij = sqrt(rsq1), inverse = Real(1) / rij;
            const Real u[3] = {rj[0] * inverse, rj[1] * inverse, rj[2] * inverse};
            Real zeta = 0;
            for (int b = 0; b < count; ++b) {
                if (b == a) continue;
                Real rk[3];
                int sk;
                const Real rsq2 = vector_of(n, i, b, x, xi, indices, steps, rk, sk);
                const TersoffEntry<Real> pijk = t.entries[(si * e + sj) * e + sk];
                if (!(rsq2 < t.cutmax2) || !(rsq2 < pijk.cutsq)) continue;
                const Real rik = sqrt(rsq2), kinverse = Real(1) / rik;
                const Real v[3] = {rk[0] * kinverse, rk[1] * kinverse, rk[2] * kinverse};
                const Real cosine = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
                zeta += cutoff_function(rik, pijk) * angular(cosine, pijk) *
                        exponential(rij, rik, pijk);
            }
            bij = bond_order(zeta, pij);
            pre = Real(-0.5) * attraction(rij, pij) * bond_order_slope(zeta, pij);
        }
        const std::int64_t p = static_cast<std::int64_t>(a) * n + i;
        order[p] = bij;
        prefactor[p] = pre;
    }
}

template <class Real, bool Energy, bool Virial, bool Atomic>
__global__ void tersoff_terms(std::int64_t n, std::int64_t plane, const Real4<Real>* x,
                              Images<Real> images, const std::int32_t* indices,
                              const std::int32_t* counts, TersoffTables<Real> t,
                              const Real* order, const Real* prefactor, Real* derivative,
                              double* share_energy, double* share_virial, double* energies,
                              double* virials) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real third = Real(1) / Real(3);
    const Real4<Real> xi = x[i];
    const int si = static_cast<int>(xi.w);
    const int e = t.elements;
    double own_energy = 0.0;
    double own_virial[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int count = counts[i];
    for (int a = 0; a < count; ++a) {
        const std::int64_t pa = static_cast<std::int64_t>(a) * n + i;
        Real rp[3];
        int sp;
        const Real rsq = vector_of(n, i, a, x, xi, indices, steps, rp, sp);
        double deriv[3] = {0.0, 0.0, 0.0};
        double share_e = 0.0;
        double share_w[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (rsq < t.cutmax2) {
            const Real r = sqrt(rsq);
            const Real inverse = Real(1) / r;
            const Real u[3] = {rp[0] * inverse, rp[1] * inverse, rp[2] * inverse};
            // the repulsion, half from each atom, all with the centre
            const TersoffEntry<Real> pij = t.entries[(si * e + sp) * e + sp];
            const TersoffEntry<Real> pji = t.entries[(sp * e + si) * e + si];
            const bool same = t.same[si * e + sp] != 0;
            const Real weight = same ? Real(1) : Real(0.5);
            Real energy = 0, fpair = 0;
            for (int s = 0; s < (same ? 1 : 2); ++s) {
                const TersoffEntry<Real>& q = s == 0 ? pij : pji;
                if (rsq < q.cutsq) {
                    const Real fc = cutoff_function(r, q), dfc = cutoff_slope(r, q);
                    const Real ex = exp(-q.lam1 * r);
                    fpair += weight * (-q.a * ex * (dfc - fc * q.lam1) / r);
                    energy += weight * (fc * q.a * ex);
                }
            }
            if (!(energy == Real(0) && fpair == Real(0))) {
                if (Energy) own_energy += 0.5 * static_cast<double>(energy);
                const Real f[3] = {Real(0.5) * fpair * rp[0], Real(0.5) * fpair * rp[1],
                                   Real(0.5) * fpair * rp[2]};
                for (int c = 0; c < 3; ++c) {
                    deriv[c] -= static_cast<double>(f[c]);
                    if (Virial) {
                        for (int d = 0; d < 3; ++d) {
                            own_virial[3 * c + d] += static_cast<double>(rp[c] * f[d]);
                        }
                    }
                }
            }
            // its bond at fixed zeta, half to each atom, and the third atoms of it
            if (rsq < pij.cutsq) {
                const Real bij = order[pa], pre = prefactor[pa];
                const Real fa = attraction(r, pij), dfa = attraction_slope(r, pij);
                const Real bond_energy = Real(0.5) * bij * fa;
                const Real bond_fpair = Real(0.5) * bij * dfa * inverse;
                if (Energy) {
                    own_energy += 0.5 * static_cast<double>(bond_energy);
                    share_e += 0.5 * static_cast<double>(bond_energy);
                }
                const Real f[3] = {-bond_fpair * rp[0], -bond_fpair * rp[1], -bond_fpair * rp[2]};
                for (int c = 0; c < 3; ++c) {
                    deriv[c] -= static_cast<double>(f[c]);
                    if (Virial) {
                        for (int d = 0; d < 3; ++d) {
                            const double v = static_cast<double>(rp[c] * f[d]);
                            share_w[3 * c + d] += 0.5 * v;
                            own_virial[3 * c + d] += 0.5 * v;
                        }
                    }
                }
                for (int b = 0; b < count; ++b) {
                    if (b == a) continue;
                    Real rk[3];
                    int sk;
                    const Real rsq2 = vector_of(n, i, b, x, xi, indices, steps, rk, sk);
                    const TersoffEntry<Real> pijk = t.entries[(si * e + sp) * e + sk];
                    if (!(rsq2 < t.cutmax2) || !(rsq2 < pijk.cutsq)) continue;
                    const Real rik = sqrt(rsq2), kinverse = Real(1) / rik;
                    const Real v[3] = {rk[0] * kinverse, rk[1] * kinverse, rk[2] * kinverse};
                    Real fj[3], fk[3];
                    zeta_forces(pre, u, r, v, rik, pijk, fj, fk);
                    for (int c = 0; c < 3; ++c) deriv[c] -= static_cast<double>(fj[c]);
                    if (Virial) {
                        for (int c = 0; c < 3; ++c) {
                            for (int d = 0; d < 3; ++d) {
                                const double w = static_cast<double>(third * (rp[c] * fj[d] + rk[c] * fk[d]));
                                own_virial[3 * c + d] += w;
                                share_w[3 * c + d] += w;
                            }
                        }
                    }
                }
            }
            // as the third atom of each other bond of the centre
            for (int b = 0; b < count; ++b) {
                if (b == a) continue;
                Real rj[3];
                int sj;
                const Real rsq1 = vector_of(n, i, b, x, xi, indices, steps, rj, sj);
                const TersoffEntry<Real> pbond = t.entries[(si * e + sj) * e + sj];
                if (!(rsq1 < t.cutmax2) || !(rsq1 < pbond.cutsq)) continue;
                const TersoffEntry<Real> pijk = t.entries[(si * e + sj) * e + sp];
                if (!(rsq < pijk.cutsq)) continue;
                const Real rij = sqrt(rsq1), jinverse = Real(1) / rij;
                const Real uj[3] = {rj[0] * jinverse, rj[1] * jinverse, rj[2] * jinverse};
                Real fj[3], fk[3];
                zeta_forces(prefactor[static_cast<std::int64_t>(b) * n + i], uj, rij, u, r, pijk,
                            fj, fk);
                for (int c = 0; c < 3; ++c) deriv[c] -= static_cast<double>(fk[c]);
                if (Virial) {
                    for (int c = 0; c < 3; ++c) {
                        for (int d = 0; d < 3; ++d) {
                            share_w[3 * c + d] +=
                                static_cast<double>(third * (rj[c] * fj[d] + rp[c] * fk[d]));
                        }
                    }
                }
            }
        }
        for (int c = 0; c < 3; ++c) derivative[c * plane + pa] = static_cast<Real>(deriv[c]);
        if (Atomic) {
            if (Energy) share_energy[pa] = share_e;
            if (Virial) {
                for (int c = 0; c < 9; ++c) share_virial[c * plane + pa] = share_w[c];
            }
        } else {
            if (Energy) own_energy += share_e;
            if (Virial) {
                for (int c = 0; c < 9; ++c) own_virial[c] += share_w[c];
            }
        }
    }
    if (Energy) energies[i] = own_energy;
    if (Virial) {
        for (int c = 0; c < 9; ++c) virials[static_cast<std::int64_t>(c) * n + i] = own_virial[c];
    }
}

template <class Real>
class Tersoff final : public Potential {
public:
    explicit Tersoff(const calculators::TersoffPotential& p)
        : cutoff_(p.cutoff), elements_(p.elements) {
        const int e = p.elements;
        std::vector<TersoffEntry<Real>> entries(static_cast<std::size_t>(e * e * e));
        for (int k = 0; k < e * e * e; ++k) {
            const calculators::TersoffParameters& q = p.parameters[static_cast<std::size_t>(k)];
            TersoffEntry<Real>& t = entries[static_cast<std::size_t>(k)];
            t.m = static_cast<Real>(q.m);
            t.gamma = static_cast<Real>(q.gamma);
            t.lam3 = static_cast<Real>(q.lam3);
            t.c2 = static_cast<Real>(q.c * q.c);
            t.d2 = static_cast<Real>(q.d * q.d);
            t.c2d2 = static_cast<Real>((q.c * q.c) / (q.d * q.d));
            t.h = static_cast<Real>(q.h);
            t.n = static_cast<Real>(q.n);
            t.beta = static_cast<Real>(q.beta);
            t.lam2 = static_cast<Real>(q.lam2);
            t.b = static_cast<Real>(q.b);
            t.r = static_cast<Real>(q.r);
            t.big_d = static_cast<Real>(q.big_d);
            t.lam1 = static_cast<Real>(q.lam1);
            t.a = static_cast<Real>(q.a);
            t.cutsq = static_cast<Real>(q.cutsq);
            t.c1 = static_cast<Real>(q.c1);
            t.c2_ = static_cast<Real>(q.c2);
            t.c3 = static_cast<Real>(q.c3);
            t.c4 = static_cast<Real>(q.c4);
        }
        std::vector<unsigned char> same(static_cast<std::size_t>(e * e));
        for (int a = 0; a < e; ++a) {
            for (int b = 0; b < e; ++b) {
                const auto& ij = p.of(a, b, b);
                const auto& ji = p.of(b, a, a);
                same[static_cast<std::size_t>(a * e + b)] = ij.a == ji.a && ij.lam1 == ji.lam1 &&
                                                            ij.r == ji.r && ij.big_d == ji.big_d;
            }
        }
        entries_.upload(entries.data(), entries.size());
        same_.upload(same.data(), same.size());
    }

    double cutoff() const override { return cutoff_; }
    int species() const override { return elements_; }

    void compute(Atoms& atoms, Neighbors& neighbors, const Box& box, bool energy, bool virial,
                 bool atomic) override {
        Atoms::Impl& a = atoms.impl();
        Neighbors::Impl& l = neighbors.impl();
        if (l.n != a.n || l.builds == 0 || l.cutoff + 1e-12 < cutoff_) {
            throw std::logic_error("the neighbor list does not fit the atoms or the cutoff");
        }
        const std::int64_t n = a.n;
        const std::int64_t plane = static_cast<std::int64_t>(l.most) * n;
        const std::int32_t* reverse = l.reversed();
        derivative_.resize(static_cast<std::size_t>(3 * plane));
        order_.resize(static_cast<std::size_t>(plane));
        prefactor_.resize(static_cast<std::size_t>(plane));
        if (atomic && energy) share_energy_.resize(static_cast<std::size_t>(plane));
        if (atomic && virial) share_virial_.resize(static_cast<std::size_t>(9 * plane));
        if (virial) a.virials.resize(static_cast<std::size_t>(9 * n));
        const Real4<Real>* x = l.local<Real>(a, box);
        const Images<Real> b = l.steps<Real>(box);
        const TersoffTables<Real> t{entries_.data(), same_.data(), elements_,
                                    static_cast<Real>(cutoff_ * cutoff_)};
        const unsigned g = blocks(n);
        tersoff_zeta<Real><<<g, kBlock>>>(n, x, b, l.indices.data(), l.counts.data(), t,
                                          order_.data(), prefactor_.data());
        MDAPY_LAUNCHED();
        const auto terms = [&](auto kernel) {
            kernel<<<g, kBlock>>>(n, plane, x, b, l.indices.data(), l.counts.data(), t,
                                  order_.data(), prefactor_.data(), derivative_.data(),
                                  share_energy_.data(), share_virial_.data(), a.energies.data(),
                                  a.virials.data());
        };
        if (atomic) {
            if (energy && virial) terms(tersoff_terms<Real, true, true, true>);
            else if (energy) terms(tersoff_terms<Real, true, false, true>);
            else if (virial) terms(tersoff_terms<Real, false, true, true>);
            else terms(tersoff_terms<Real, false, false, true>);
        } else {
            if (energy && virial) terms(tersoff_terms<Real, true, true, false>);
            else if (energy) terms(tersoff_terms<Real, true, false, false>);
            else if (virial) terms(tersoff_terms<Real, false, true, false>);
            else terms(tersoff_terms<Real, false, false, false>);
        }
        MDAPY_LAUNCHED();
        const auto gather = [&](auto kernel) {
            kernel<<<g, kBlock>>>(n, plane, l.indices.data(), l.counts.data(), reverse,
                                  derivative_.data(), share_energy_.data(), share_virial_.data(),
                                  nullptr, a.forces.data(), a.energies.data(), a.virials.data());
        };
        if (atomic && energy && virial) gather(gather_shares<Real, true, true, true>);
        else if (atomic && energy) gather(gather_shares<Real, true, false, true>);
        else if (atomic && virial) gather(gather_shares<Real, false, true, true>);
        else gather(gather_shares<Real, false, false, false>);
        MDAPY_LAUNCHED();
        a.has_energies = energy;
        a.has_virials = virial;
    }

private:
    double cutoff_;
    int elements_;
    Buffer<TersoffEntry<Real>> entries_;
    Buffer<unsigned char> same_;
    Buffer<Real> derivative_, order_, prefactor_;
    Buffer<double> share_energy_, share_virial_;
};

}  // namespace

std::unique_ptr<Potential> make_tersoff(const calculators::TersoffPotential& potential,
                                        Precision precision) {
    if (precision == Precision::Mixed) {
        return std::make_unique<Tersoff<float>>(potential);
    }
    return std::make_unique<Tersoff<double>>(potential);
}

}  // namespace mdapy::cuda
