// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The angular-dependent potential on the device, as calculators/adp.cpp
// computes it: the density, the dipole and the quadrupole at each atom and its
// energy, then the forces, from a full neighbor list, each atom gathering from
// its own pairs.

#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/neighbors.cuh"
#include "cuda/splines.cuh"

namespace mdapy::cuda {

namespace {

template <class Real>
struct AdpTables {
    Splines<Real> radial;  // densities by element, then r phi, u and w by pair
    Splines<double> embedding;
    int elements;
    int pairs;  // elements (elements + 1) / 2
    Real cutsq;
};

__device__ inline int pair_index(int a, int b) {
    return a >= b ? a * (a + 1) / 2 + b : b * (b + 1) / 2 + a;
}

// F'(rho), the dipole and the quadrupole of an atom: (F', mu_x, mu_y, mu_z),
// (lambda_xx, _yy, _zz, _yz), (lambda_xz, _xy, 0, 0).
template <class Real>
struct Moments {
    Real4<Real> a, b, c;
};

template <class Real, bool Energy>
__global__ void adp_density(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                            const std::int32_t* indices, const std::int32_t* counts,
                            AdpTables<Real> t, Moments<Real>* moments, double* energies) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int si = static_cast<int>(xi.w);
    double rho = 0.0;
    double mu[3] = {0.0, 0.0, 0.0};
    double lambda[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // xx yy zz yz xz xy
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const auto packed = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
        const Real4<Real> xj = x[packed & kIndexMask];
        Real d[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
        add_step(steps, packed >> kIndexBits, d[0], d[1], d[2]);
        const Real r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (!(r2 < t.cutsq)) continue;
        const Real del[3] = {-d[0], -d[1], -d[2]};  // x_i - x_j
        const int sj = static_cast<int>(xj.w);
        const Real r = sqrt(r2);
        rho += static_cast<double>(spline_value(t.radial, sj, r));
        const int p = pair_index(si, sj);
        const Real u = spline_value(t.radial, t.elements + t.pairs + p, r);
        const Real w = spline_value(t.radial, t.elements + 2 * t.pairs + p, r);
        for (int a = 0; a < 3; ++a) mu[a] += static_cast<double>(u * del[a]);
        lambda[0] += static_cast<double>(w * del[0] * del[0]);
        lambda[1] += static_cast<double>(w * del[1] * del[1]);
        lambda[2] += static_cast<double>(w * del[2] * del[2]);
        lambda[3] += static_cast<double>(w * del[1] * del[2]);
        lambda[4] += static_cast<double>(w * del[0] * del[2]);
        lambda[5] += static_cast<double>(w * del[0] * del[1]);
    }
    double embedding, slope;
    spline(t.embedding, si, rho, embedding, slope);
    Moments<Real> m;
    m.a.x = static_cast<Real>(slope);
    m.a.y = static_cast<Real>(mu[0]);
    m.a.z = static_cast<Real>(mu[1]);
    m.a.w = static_cast<Real>(mu[2]);
    m.b.x = static_cast<Real>(lambda[0]);
    m.b.y = static_cast<Real>(lambda[1]);
    m.b.z = static_cast<Real>(lambda[2]);
    m.b.w = static_cast<Real>(lambda[3]);
    m.c.x = static_cast<Real>(lambda[4]);
    m.c.y = static_cast<Real>(lambda[5]);
    m.c.z = Real(0);
    m.c.w = Real(0);
    moments[i] = m;
    if (Energy) {
        const double nu = lambda[0] + lambda[1] + lambda[2];
        energies[i] = embedding + 0.5 * (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]) +
                      0.5 * (lambda[0] * lambda[0] + lambda[1] * lambda[1] + lambda[2] * lambda[2]) +
                      1.0 * (lambda[3] * lambda[3] + lambda[4] * lambda[4] + lambda[5] * lambda[5]) -
                      1.0 / 6.0 * nu * nu;
    }
}

template <class Real, bool Energy, bool Virial>
__global__ void adp_force(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                          const std::int32_t* indices, const std::int32_t* counts,
                          AdpTables<Real> t, const Moments<Real>* moments, double* forces,
                          double* energies, double* virials) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int si = static_cast<int>(xi.w);
    const Moments<Real> mi = moments[i];
    double f[3] = {0.0, 0.0, 0.0};
    double w[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double energy = 0.0;
    const Real third = Real(1) / Real(3);
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const auto packed = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
        const int j = static_cast<int>(packed & kIndexMask);
        const Real4<Real> xj = x[j];
        Real d[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
        add_step(steps, packed >> kIndexBits, d[0], d[1], d[2]);
        const Real r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (!(r2 < t.cutsq)) continue;
        const Real delx = -d[0], dely = -d[1], delz = -d[2];  // x_i - x_j
        const int sj = static_cast<int>(xj.w);
        const Real r = sqrt(r2);
        Real rho_i, rhoip, rho_j, rhojp, z2, z2p, u2, u2p, w2, w2p;
        spline(t.radial, si, r, rho_i, rhoip);
        spline(t.radial, sj, r, rho_j, rhojp);
        const int p = pair_index(si, sj);
        spline(t.radial, t.elements + p, r, z2, z2p);
        spline(t.radial, t.elements + t.pairs + p, r, u2, u2p);
        spline(t.radial, t.elements + 2 * t.pairs + p, r, w2, w2p);
        const Real recip = Real(1) / r;
        const Real phi = z2 * recip;
        const Real phip = z2p * recip - phi * recip;
        const Moments<Real> mj = moments[j];
        const Real psip = mi.a.x * rhojp + mj.a.x * rhoip + phip;
        const Real fpair = -psip * recip;
        const Real delmux = mi.a.y - mj.a.y, delmuy = mi.a.z - mj.a.z, delmuz = mi.a.w - mj.a.w;
        const Real trdelmu = delmux * delx + delmuy * dely + delmuz * delz;
        const Real sxx = mi.b.x + mj.b.x, syy = mi.b.y + mj.b.y, szz = mi.b.z + mj.b.z;
        const Real syz = mi.b.w + mj.b.w, sxz = mi.c.x + mj.c.x, sxy = mi.c.y + mj.c.y;
        const Real tradellam = sxx * delx * delx + syy * dely * dely + szz * delz * delz +
                               Real(2) * sxy * delx * dely + Real(2) * sxz * delx * delz +
                               Real(2) * syz * dely * delz;
        const Real nu = sxx + syy + szz;
        const Real wr = w2p * r + Real(2) * w2;
        const Real adpx = delmux * u2 + trdelmu * u2p * delx * recip +
                          Real(2) * w2 * (sxx * delx + sxy * dely + sxz * delz) +
                          w2p * delx * recip * tradellam - third * nu * wr * delx;
        const Real adpy = delmuy * u2 + trdelmu * u2p * dely * recip +
                          Real(2) * w2 * (sxy * delx + syy * dely + syz * delz) +
                          w2p * dely * recip * tradellam - third * nu * wr * dely;
        const Real adpz = delmuz * u2 + trdelmu * u2p * delz * recip +
                          Real(2) * w2 * (sxz * delx + syz * dely + szz * delz) +
                          w2p * delz * recip * tradellam - third * nu * wr * delz;
        const Real force[3] = {delx * fpair - adpx, dely * fpair - adpy, delz * fpair - adpz};
        const Real del[3] = {delx, dely, delz};
        for (int a = 0; a < 3; ++a) f[a] += static_cast<double>(force[a]);
        if (Virial) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    w[3 * a + b] += static_cast<double>(Real(0.5) * del[a] * force[b]);
                }
            }
        }
        if (Energy) energy += static_cast<double>(Real(0.5) * phi);
    }
    for (int a = 0; a < 3; ++a) forces[3 * i + a] = f[a];
    if (Energy) energies[i] += energy;
    if (Virial) {
        for (int c = 0; c < 9; ++c) virials[static_cast<std::int64_t>(c) * n + i] = w[c];
    }
}

template <class Real>
class Adp final : public Potential {
public:
    explicit Adp(const calculators::AdpPotential& p)
        : cutoff_(p.cutoff), elements_(p.elements),
          pairs_(p.elements * (p.elements + 1) / 2) {
        Rows<Real> radial;
        for (const auto& s : p.density) radial.add(s);
        for (const auto& s : p.pair) radial.add(s);
        for (const auto& s : p.dipole) radial.add(s);
        for (const auto& s : p.quadrupole) radial.add(s);
        Rows<double> embedding;
        for (const auto& s : p.embedding) embedding.add(s);
        radial_.upload(radial);
        embedding_.upload(embedding);
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
        virial = virial || atomic;
        const std::int64_t n = a.n;
        moments_.resize(static_cast<std::size_t>(n));
        if (virial) a.virials.resize(static_cast<std::size_t>(9 * n));
        const Real4<Real>* x = l.local<Real>(a, box);
        const Images<Real> b = l.steps<Real>(box);
        AdpTables<Real> t{radial_.view(), embedding_.view(), elements_, pairs_,
                          static_cast<Real>(cutoff_ * cutoff_)};
        const unsigned g = blocks(n);
        const auto* idx = l.indices.data();
        const auto* cnt = l.counts.data();
        if (energy) {
            adp_density<Real, true><<<g, kBlock>>>(n, x, b, idx, cnt, t, moments_.data(),
                                                   a.energies.data());
        } else {
            adp_density<Real, false><<<g, kBlock>>>(n, x, b, idx, cnt, t, moments_.data(), nullptr);
        }
        MDAPY_LAUNCHED();
        double* e = energy ? a.energies.data() : nullptr;
        double* w = virial ? a.virials.data() : nullptr;
        if (energy && virial) {
            adp_force<Real, true, true><<<g, kBlock>>>(n, x, b, idx, cnt, t, moments_.data(),
                                                       a.forces.data(), e, w);
        } else if (energy) {
            adp_force<Real, true, false><<<g, kBlock>>>(n, x, b, idx, cnt, t, moments_.data(),
                                                        a.forces.data(), e, w);
        } else if (virial) {
            adp_force<Real, false, true><<<g, kBlock>>>(n, x, b, idx, cnt, t, moments_.data(),
                                                        a.forces.data(), e, w);
        } else {
            adp_force<Real, false, false><<<g, kBlock>>>(n, x, b, idx, cnt, t, moments_.data(),
                                                         a.forces.data(), e, w);
        }
        MDAPY_LAUNCHED();
        a.has_energies = energy;
        a.has_virials = virial;
    }

private:
    double cutoff_;
    int elements_;
    int pairs_;
    DeviceRows<Real> radial_;
    DeviceRows<double> embedding_;
    Buffer<Moments<Real>> moments_;
};

}  // namespace

std::unique_ptr<Potential> make_adp(const calculators::AdpPotential& potential,
                                    Precision precision) {
    if (precision == Precision::Mixed) {
        return std::make_unique<Adp<float>>(potential);
    }
    return std::make_unique<Adp<double>>(potential);
}

}  // namespace mdapy::cuda
