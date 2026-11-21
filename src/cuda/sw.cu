// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The Stillinger-Weber potential on the device, as
// calculators/stillinger_weber.cpp computes it, in the two passes of
// calculators/many_body.h: each atom computes the terms it is the centre
// of, the derivative of their energy with respect to the vector of each of
// its pairs, then gathers its force from its pairs and their reverses. Each
// three-body term is computed by both of its pairs, so that the derivative
// of each pair is written once. The energy and the virial of each term go to
// its atoms as the processor shares them when the atoms' own are asked for;
// otherwise each centre keeps those of its terms, which leaves the totals.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "calculators/stillinger_weber.h"
#include "cuda/atoms.cuh"
#include "cuda/many_body.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

// whole: the exponents p and q when they are whole numbers from 0 to 16,
// for r^-p and r^-q by multiplication, else -1 and pow.
template <class Real>
struct SwEntry {
    Real cut, cutsq, sigma, p, q, c1, c2, c3, c4, c5, c6, sigma_gamma, lambda_epsilon,
        lambda_epsilon2, costheta, pad;
    int whole_p, whole_q, pad2[2];
};

template <class Real>
struct SwTables {
    const SwEntry<Real>* entries;  // (i * e + j) * e + k
    const unsigned char* same2;    // i * e + j: the entries i j j and j i i agree
    const unsigned char* same3;    // (i * e + j) * e + k: the entries i j k and i k j agree
    int elements;
};

// Whether a pair at r^2 is within the cutoff of an entry: by r^2, and by r,
// whose square root may round to the cutoff itself in single precision for
// an r^2 just below its square, where 1 / (r - cut) would be infinite.
template <class Real>
__device__ inline bool inside(const SwEntry<Real>& e, Real rsq) {
    return rsq < e.cutsq && sqrt(rsq) < e.cut;
}

// r^-k, k a whole number from 0 to 16, by squaring: a few roundings, where
// pow costs some hundred operations in double precision.
template <class Real>
__device__ inline Real inverse_power(Real r, int k) {
    Real base = Real(1) / r, out = Real(1);
    while (k > 0) {
        if (k & 1) out *= base;
        base *= base;
        k >>= 1;
    }
    return out;
}

// The pair term of an entry at r^2: its energy and fpair.
template <class Real>
__device__ inline void sw_two(const SwEntry<Real>& e, Real rsq, Real& energy, Real& fpair) {
    const Real r = sqrt(rsq), rinvsq = Real(1) / rsq;
    const Real rp = e.whole_p >= 0 ? inverse_power(r, e.whole_p) : pow(r, -e.p);
    const Real rq = e.whole_q >= 0 ? inverse_power(r, e.whole_q) : pow(r, -e.q);
    const Real rainv = Real(1) / (r - e.cut);
    const Real rainvsq = rainv * rainv * r;
    const Real expsrainv = exp(e.sigma * rainv);
    fpair = (e.c1 * rp - e.c2 * rq + (e.c3 * rp - e.c4 * rq) * rainvsq) * expsrainv * rinvsq;
    energy = (e.c5 * rp - e.c6 * rq) * expsrainv;
}

// The derivative of each pair, its 3 components in a Real4 at k n + i for
// pair k of atom i. Atomic: the energy (1 plane of plane values) and the
// virial (9 planes) of each pair's share, by pair; else the
// shares stay with the centre.
// Only the first centers atoms, those of the process under MPI, are centres:
// the terms of the others are left to their own processes.
template <class Real, bool Energy, bool Virial, bool Atomic>
__global__ void sw_terms(std::int64_t n, std::int64_t centers, std::int64_t plane,
                         const Real4<Real>* x,
                         Images<Real> images, const std::int32_t* indices,
                         const std::int32_t* counts, SwTables<Real> t, Real4<Real>* derivative,
                         double* own_force, double* share_energy, double* share_virial,
                         double* energies, double* virials) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i >= centers) {
        ghost_terms<Real, Energy, Virial, Atomic>(n, plane, i, counts[i], derivative, own_force,
                                                  share_energy, share_virial, energies, virials);
        return;
    }
    const Real third = Real(1) / Real(3);
    const Real4<Real> xi = x[i];
    const int si = static_cast<int>(xi.w);
    const int e = t.elements;
    double own_energy = 0.0;
    double own_virial[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double own[3] = {0.0, 0.0, 0.0};  // the sum of the derivatives of its pairs
    const int count = counts[i];
    for (int a = 0; a < count; ++a) {
        const std::int64_t pa = static_cast<std::int64_t>(a) * n + i;
        const auto packed = static_cast<std::uint32_t>(indices[pa]);
        const Real4<Real> xj = x[packed & kIndexMask];
        Real rj[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};  // x_j - x_i
        add_step(steps, packed >> kIndexBits, rj[0], rj[1], rj[2]);
        const Real rsq1 = rj[0] * rj[0] + rj[1] * rj[1] + rj[2] * rj[2];
        const int sj = static_cast<int>(xj.w);
        const SwEntry<Real> pij = t.entries[(si * e + sj) * e + sj];
        const SwEntry<Real> pji = t.entries[(sj * e + si) * e + si];
        double deriv[3] = {0.0, 0.0, 0.0};
        double share_e = 0.0;
        double share_w[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        // the two-body term by the entries i j j and j i i: their mean, or
        // the one when they agree; half from each atom, all with the centre
        {
            const bool same = t.same2[si * e + sj] != 0;
            const Real weight = same ? Real(1) : Real(0.5);
            Real energy = 0, fpair = 0;
            bool any = false;
            if (inside(pij, rsq1)) {
                any = true;
                Real ee, ff;
                sw_two(pij, rsq1, ee, ff);
                energy += weight * ee;
                fpair += weight * ff;
            }
            if (!same && inside(pji, rsq1)) {
                any = true;
                Real ee, ff;
                sw_two(pji, rsq1, ee, ff);
                energy += weight * ee;
                fpair += weight * ff;
            }
            if (any) {
                if (Energy) own_energy += 0.5 * static_cast<double>(energy);
                const Real f[3] = {Real(0.5) * fpair * rj[0], Real(0.5) * fpair * rj[1],
                                   Real(0.5) * fpair * rj[2]};
                for (int c = 0; c < 3; ++c) {
                    deriv[c] -= static_cast<double>(f[c]);
                    if (Virial) {
                        for (int d = 0; d < 3; ++d) {
                            own_virial[3 * c + d] += static_cast<double>(rj[c] * f[d]);
                        }
                    }
                }
            }
        }
        // the three-body terms of this pair with every other, a third of each
        // to each of the three atoms; each term is met from both its pairs
        if (inside(pij, rsq1)) {
            const Real r1 = sqrt(rsq1), rinvsq1 = Real(1) / rsq1;
            const Real rainv1 = Real(1) / (r1 - pij.cut);
            const Real gsrainv1 = pij.sigma_gamma * rainv1;
            const Real gsrainvsq1 = gsrainv1 * rainv1 / r1;
            const Real expgsrainv1 = exp(gsrainv1);
            for (int b = 0; b < count; ++b) {
                if (b == a) continue;
                const auto pk = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(b) * n + i]);
                const Real4<Real> xk = x[pk & kIndexMask];
                Real rk[3] = {xk.x - xi.x, xk.y - xi.y, xk.z - xi.z};
                add_step(steps, pk >> kIndexBits, rk[0], rk[1], rk[2]);
                const Real rsq2 = rk[0] * rk[0] + rk[1] * rk[1] + rk[2] * rk[2];
                const int sk = static_cast<int>(xk.w);
                const SwEntry<Real> pik = t.entries[(si * e + sk) * e + sk];
                if (!inside(pik, rsq2)) continue;
                const Real r2 = sqrt(rsq2), rinvsq2 = Real(1) / rsq2;
                const Real rainv2 = Real(1) / (r2 - pik.cut);
                const Real gsrainv2 = pik.sigma_gamma * rainv2;
                const Real gsrainvsq2 = gsrainv2 * rainv2 / r2;
                const Real expgsrainv2 = exp(gsrainv2);
                const Real rinv12 = Real(1) / (r1 * r2);
                const Real cs = (rj[0] * rk[0] + rj[1] * rk[1] + rj[2] * rk[2]) * rinv12;
                const Real facexp = expgsrainv1 * expgsrainv2;
                // the entries i j k and i k j: their mean, or the one
                const SwEntry<Real> pjk = t.entries[(si * e + sj) * e + sk];
                const SwEntry<Real> pkj = t.entries[(si * e + sk) * e + sj];
                const bool same = t.same3[(si * e + sj) * e + sk] != 0;
                Real facrad, facang;
                {
                    const Real delcs = cs - pjk.costheta;
                    facrad = pjk.lambda_epsilon * facexp * delcs * delcs;
                    facang = pjk.lambda_epsilon2 * facexp * delcs;
                }
                if (!same) {
                    const Real delcs = cs - pkj.costheta;
                    facrad = Real(0.5) * facrad + Real(0.5) * (pkj.lambda_epsilon * facexp * delcs * delcs);
                    facang = Real(0.5) * facang + Real(0.5) * (pkj.lambda_epsilon2 * facexp * delcs);
                }
                const Real frad1 = facrad * gsrainvsq1, frad2 = facrad * gsrainvsq2;
                const Real facang12 = rinv12 * facang;
                const Real csfacang = cs * facang;
                const Real csfac1 = rinvsq1 * csfacang, csfac2 = rinvsq2 * csfacang;
                Real fj[3], fk[3];
                for (int c = 0; c < 3; ++c) {
                    fj[c] = rj[c] * (frad1 + csfac1) - rk[c] * facang12;
                    fk[c] = rk[c] * (frad2 + csfac2) - rj[c] * facang12;
                }
                for (int c = 0; c < 3; ++c) deriv[c] -= static_cast<double>(fj[c]);
                if (Energy) {
                    const double part = static_cast<double>(third * facrad);
                    own_energy += 0.5 * part;  // met twice
                    share_e += part;
                }
                if (Virial) {
                    for (int c = 0; c < 3; ++c) {
                        for (int d = 0; d < 3; ++d) {
                            const double w = static_cast<double>(third * (rj[c] * fj[d] + rk[c] * fk[d]));
                            own_virial[3 * c + d] += 0.5 * w;
                            share_w[3 * c + d] += w;
                        }
                    }
                }
            }
        }
        derivative[pa] = four<Real>(deriv[0], deriv[1], deriv[2]);
        for (int c = 0; c < 3; ++c) own[c] += deriv[c];
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
    for (int c = 0; c < 3; ++c) own_force[3 * i + c] = own[c];
    if (Energy) energies[i] = own_energy;
    if (Virial) {
        for (int c = 0; c < 9; ++c) virials[static_cast<std::int64_t>(c) * n + i] = own_virial[c];
    }
}

template <class Real>
class Sw final : public Potential {
public:
    explicit Sw(const calculators::StillingerWeberPotential& p)
        : cutoff_(p.cutoff), elements_(p.elements) {
        const int e = p.elements;
        std::vector<SwEntry<Real>> entries(static_cast<std::size_t>(e * e * e));
        for (int k = 0; k < e * e * e; ++k) {
            const calculators::StillingerWeberParameters& q = p.parameters[static_cast<std::size_t>(k)];
            SwEntry<Real>& t = entries[static_cast<std::size_t>(k)];
            t.cut = static_cast<Real>(q.cut);
            t.cutsq = static_cast<Real>(q.cutsq);
            t.sigma = static_cast<Real>(q.sigma);
            t.p = static_cast<Real>(q.p);
            t.q = static_cast<Real>(q.q);
            t.c1 = static_cast<Real>(q.c1);
            t.c2 = static_cast<Real>(q.c2);
            t.c3 = static_cast<Real>(q.c3);
            t.c4 = static_cast<Real>(q.c4);
            t.c5 = static_cast<Real>(q.c5);
            t.c6 = static_cast<Real>(q.c6);
            t.sigma_gamma = static_cast<Real>(q.sigma_gamma);
            t.lambda_epsilon = static_cast<Real>(q.lambda_epsilon);
            t.lambda_epsilon2 = static_cast<Real>(q.lambda_epsilon2);
            t.costheta = static_cast<Real>(q.costheta);
            t.pad = Real(0);
            const auto whole = [](double v) {
                return v >= 0.0 && v <= 16.0 && v == std::floor(v) ? static_cast<int>(v) : -1;
            };
            t.whole_p = whole(q.p);
            t.whole_q = whole(q.q);
            t.pad2[0] = t.pad2[1] = 0;
        }
        // where the entries of a pair or of an angle agree, as the processor
        // compares them
        std::vector<unsigned char> same2(static_cast<std::size_t>(e * e)), same3(entries.size());
        for (int a = 0; a < e; ++a) {
            for (int b = 0; b < e; ++b) {
                const auto& ij = p.of(a, b, b);
                const auto& ji = p.of(b, a, a);
                same2[static_cast<std::size_t>(a * e + b)] =
                    ij.cutsq == ji.cutsq && ij.cut == ji.cut && ij.sigma == ji.sigma &&
                    ij.p == ji.p && ij.q == ji.q && ij.c1 == ji.c1 && ij.c2 == ji.c2 &&
                    ij.c3 == ji.c3 && ij.c4 == ji.c4 && ij.c5 == ji.c5 && ij.c6 == ji.c6;
                for (int c = 0; c < e; ++c) {
                    const auto& jk = p.of(a, b, c);
                    const auto& kj = p.of(a, c, b);
                    same3[static_cast<std::size_t>((a * e + b) * e + c)] =
                        jk.lambda_epsilon == kj.lambda_epsilon &&
                        jk.lambda_epsilon2 == kj.lambda_epsilon2 && jk.costheta == kj.costheta;
                }
            }
        }
        entries_.upload(entries.data(), entries.size());
        same2_.upload(same2.data(), same2.size());
        same3_.upload(same3.data(), same3.size());
    }

    double cutoff() const override { return cutoff_; }
    int species() const override { return elements_; }
    // Under MPI the atoms of the process alone are the centres, the forces
    // on the ghosts sent back; compute and compute_heat take every atom, as
    // the parts of an overlay, whose ghosts reach twice the cutoff.
    bool thin() const override { return true; }
    bool reverse() const override { return true; }
    void compute_stage(int stage, Atoms& atoms, Neighbors& neighbors, const Box& box,
                       bool energy, bool virial, bool heat) override {
        if (stage != 0) return;
        if (heat) {
            evaluate(atoms, neighbors, box, true, false, false, true, true);
        } else {
            evaluate(atoms, neighbors, box, energy, virial, false, false, true);
        }
    }

    void compute(Atoms& atoms, Neighbors& neighbors, const Box& box, bool energy, bool virial,
                 bool atomic) override {
        evaluate(atoms, neighbors, box, energy, virial, atomic, false, false);
    }

    // without atomic each term stays whole with its centre, and the
    // derivatives give the virials of the heat flux
    void compute_heat(Atoms& atoms, Neighbors& neighbors, const Box& box) override {
        evaluate(atoms, neighbors, box, true, false, false, true, false);
    }

private:
    // centred: the first owned atoms alone are centres
    void evaluate(Atoms& atoms, Neighbors& neighbors, const Box& box, bool energy, bool virial,
                  bool atomic, bool heat, bool centred) {
        Atoms::Impl& a = atoms.impl();
        Neighbors::Impl& l = neighbors.impl();
        if (l.n != a.n || l.builds == 0 || l.cutoff + 1e-12 < cutoff_) {
            throw std::logic_error("the neighbor list does not fit the atoms or the cutoff");
        }
        const std::int64_t n = a.n;
        if (virial || heat) a.virials.resize(static_cast<std::size_t>(9 * n));
        const Real4<Real>* x = l.local<Real>(a, box);
        const Images<Real> b = l.steps<Real>(box);
        // the pairs within the largest cutoff of the entries, every pair an
        // entry counts
        pairs_.build(l, x, b, static_cast<Real>(cutoff_ * cutoff_));
        const std::int64_t plane = pairs_.plane(n);
        at_least(derivative_, static_cast<std::size_t>(plane));
        if (atomic && energy) at_least(share_energy_, static_cast<std::size_t>(plane));
        if (atomic && virial) at_least(share_virial_, static_cast<std::size_t>(9 * plane));
        const SwTables<Real> t{entries_.data(), same2_.data(), same3_.data(), elements_};
        own_force_.resize(static_cast<std::size_t>(3 * n));
        Args args{n, centred ? a.active() : n, plane, x, b, pairs_.indices(), pairs_.counts(), t,
                  derivative_.data(),
                  own_force_.data(), share_energy_.data(), share_virial_.data(),
                  a.energies.data(), a.virials.data()};
        if (atomic) {
            run<true>(args, energy, virial);
        } else {
            run<false>(args, energy, virial);
        }
        gather<Real>(n, plane, pairs_, derivative_.data(), share_energy_.data(),
                     share_virial_.data(), own_force_.data(), false, energy, virial, atomic,
                     a.forces.data(), a.energies.data(), a.virials.data());
        if (heat) heat_virials<Real>(n, x, b, pairs_, derivative_.data(), a.virials.data());
        a.has_energies = energy;
        a.has_virials = virial || heat;
    }

    struct Args {
        std::int64_t n, centers, plane;
        const Real4<Real>* x;
        Images<Real> images;
        const std::int32_t* indices;
        const std::int32_t* counts;
        SwTables<Real> t;
        Real4<Real>* derivative;
        double* own_force;
        double* share_energy;
        double* share_virial;
        double* energies;
        double* virials;
    };

    template <bool Energy, bool Virial, bool Atomic>
    static void terms(const Args& r) {
        sw_terms<Real, Energy, Virial, Atomic><<<blocks(r.n), kBlock>>>(
            r.n, r.centers, r.plane, r.x, r.images, r.indices, r.counts, r.t, r.derivative,
            r.own_force,
            r.share_energy, r.share_virial, r.energies, r.virials);
        MDAPY_LAUNCHED();
    }

    template <bool Atomic>
    static void run(const Args& r, bool energy, bool virial) {
        if (energy && virial) terms<true, true, Atomic>(r);
        else if (energy) terms<true, false, Atomic>(r);
        else if (virial) terms<false, true, Atomic>(r);
        else terms<false, false, Atomic>(r);
    }

    double cutoff_;
    int elements_;
    Buffer<SwEntry<Real>> entries_;
    Buffer<unsigned char> same2_, same3_;
    Buffer<Real4<Real>> derivative_;
    Buffer<double> share_energy_, share_virial_, own_force_;
    ShortPairs pairs_;
};

}  // namespace

std::unique_ptr<Potential> make_stillinger_weber(
    const calculators::StillingerWeberPotential& potential, Precision precision) {
    if (precision == Precision::Mixed) {
        return std::make_unique<Sw<float>>(potential);
    }
    return std::make_unique<Sw<double>>(potential);
}

}  // namespace mdapy::cuda
