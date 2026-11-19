// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The embedded atom method on the device, as calculators/eam.cpp computes it:
// the density at each atom and its embedding energy, then the forces, from a
// full neighbor list, each atom gathering from its own pairs. The splines of
// the pairs are in the precision of the computation, those of the embedding
// in double precision.

#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/neighbors.cuh"
#include "cuda/splines.cuh"

namespace mdapy::cuda {

namespace {

template <class Real>
struct Tables {
    Splines<Real> pairs;      // densities, then pair terms r phi(r)
    Splines<double> embedding;
    const int* density;       // the spline of the density source gives receiver, source * e + receiver
    const int* pair;          // the spline of r phi of a pair, a * e + b
    int elements;
    Real cutsq;
    double rho_max;
};

template <class Real, bool Energy>
__global__ void density(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                        const std::int32_t* indices, const std::int32_t* counts,
                        Tables<Real> t, Real* fp, double* energies) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int si = static_cast<int>(xi.w);
    double rho = 0.0;
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const auto packed = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
        const Real4<Real> xj = x[packed & kIndexMask];
        Real dx = xj.x - xi.x, dy = xj.y - xi.y, dz = xj.z - xi.z;
        add_step(steps, packed >> kIndexBits, dx, dy, dz);
        const Real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < t.cutsq) {
            const int sj = static_cast<int>(xj.w);
            rho += static_cast<double>(spline_value(t.pairs, t.density[sj * t.elements + si], sqrt(r2)));
        }
    }
    double embedding, slope;
    spline(t.embedding, si, rho, embedding, slope);
    if (rho > t.rho_max) {
        embedding += slope * (rho - t.rho_max);
    }
    fp[i] = static_cast<Real>(slope);
    if (Energy) energies[i] = embedding;
}

template <class Real, bool Energy, bool Virial>
__global__ void force(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                      const std::int32_t* indices, const std::int32_t* counts, Tables<Real> t,
                      const Real* fp, double* forces, double* energies, double* virials) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int si = static_cast<int>(xi.w);
    const Real fpi = fp[i];
    double f[3] = {0.0, 0.0, 0.0};
    double w[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double energy = 0.0;
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const auto packed = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
        const int j = static_cast<int>(packed & kIndexMask);
        const Real4<Real> xj = x[j];
        // del = x_i - x_j, as in LAMMPS
        Real d[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
        add_step(steps, packed >> kIndexBits, d[0], d[1], d[2]);
        const Real del[3] = {-d[0], -d[1], -d[2]};
        const Real r2 = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
        if (!(r2 < t.cutsq)) continue;
        const int sj = static_cast<int>(xj.w);
        const Real r = sqrt(r2);
        // rho_ip: of the density i gives j; rho_jp: j gives i, the same
        // spline for atoms of one element in eam/alloy
        Real rho_i, rho_ip, rho_j, rho_jp, z2, z2p;
        const int gives = t.density[si * t.elements + sj];
        const int takes = t.density[sj * t.elements + si];
        spline(t.pairs, gives, r, rho_i, rho_ip);
        if (takes == gives) {
            rho_j = rho_i;
            rho_jp = rho_ip;
        } else {
            spline(t.pairs, takes, r, rho_j, rho_jp);
        }
        spline(t.pairs, t.pair[si * t.elements + sj], r, z2, z2p);
        const Real recip = Real(1) / r;
        const Real phi = z2 * recip;
        const Real phip = z2p * recip - phi * recip;
        const Real psip = fpi * rho_jp + fp[j] * rho_ip + phip;
        const Real fpair = -psip * recip;
        for (int a = 0; a < 3; ++a) {
            f[a] += static_cast<double>(del[a] * fpair);
        }
        if (Virial) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    w[3 * a + b] += static_cast<double>(Real(0.5) * del[a] * del[b] * fpair);
                }
            }
        }
        if (Energy) energy += static_cast<double>(Real(0.5) * phi);
    }
    for (int a = 0; a < 3; ++a) {
        forces[3 * i + a] = f[a];
    }
    if (Energy) energies[i] += energy;
    if (Virial) {
        for (int c = 0; c < 9; ++c) {
            virials[static_cast<std::int64_t>(c) * n + i] = w[c];
        }
    }
}

template <class Real>
class Eam final : public Potential {
public:
    explicit Eam(const calculators::EamPotential& p)
        : cutoff_(p.cutoff), elements_(p.elements), rho_max_(p.rho_max) {
        const int e = p.elements;
        Rows<Real> pairs;
        for (const auto& s : p.density) pairs.add(s);
        for (const auto& s : p.pair) pairs.add(s);
        Rows<double> embedding;
        for (const auto& s : p.embedding) embedding.add(s);
        std::vector<int> density(static_cast<std::size_t>(e * e)), pair(density.size());
        for (int a = 0; a < e; ++a) {
            for (int b = 0; b < e; ++b) {
                density[a * e + b] = p.density_per_pair ? a * e + b : a;
                const int hi = a > b ? a : b, lo = a > b ? b : a;
                pair[a * e + b] = static_cast<int>(p.density.size()) + hi * (hi + 1) / 2 + lo;
            }
        }
        pairs_.upload(pairs);
        embedding_.upload(embedding);
        density_.upload(density.data(), density.size());
        pair_.upload(pair.data(), pair.size());
    }

    double cutoff() const override { return cutoff_; }
    int species() const override { return elements_; }

    void compute(Atoms& atoms, Neighbors& neighbors, const Box& box, bool energy,
                 bool virial, bool atomic) override {
        Atoms::Impl& a = atoms.impl();
        Neighbors::Impl& l = neighbors.impl();
        if (l.n != a.n || l.builds == 0 || l.cutoff + 1e-12 < cutoff_) {
            throw std::logic_error("the neighbor list does not fit the atoms or the cutoff");
        }
        virial = virial || atomic;
        const std::int64_t n = a.n;
        fp_.resize(static_cast<std::size_t>(n));
        if (virial) a.virials.resize(static_cast<std::size_t>(9 * n));
        const Real4<Real>* x = l.local<Real>(a, box);
        const Images<Real> b = l.steps<Real>(box);
        Tables<Real> t{pairs_.view(), embedding_.view(), density_.data(), pair_.data(),
                       elements_, static_cast<Real>(cutoff_ * cutoff_), rho_max_};
        const unsigned g = blocks(n);
        if (energy) {
            density<Real, true><<<g, kBlock>>>(n, x, b, l.indices.data(), l.counts.data(), t,
                                               fp_.data(), a.energies.data());
        } else {
            density<Real, false><<<g, kBlock>>>(n, x, b, l.indices.data(), l.counts.data(), t,
                                                fp_.data(), nullptr);
        }
        MDAPY_LAUNCHED();
        double* e = energy ? a.energies.data() : nullptr;
        double* w = virial ? a.virials.data() : nullptr;
        const auto* idx = l.indices.data();
        const auto* cnt = l.counts.data();
        if (energy && virial) {
            force<Real, true, true><<<g, kBlock>>>(n, x, b, idx, cnt, t, fp_.data(),
                                                   a.forces.data(), e, w);
        } else if (energy) {
            force<Real, true, false><<<g, kBlock>>>(n, x, b, idx, cnt, t, fp_.data(),
                                                    a.forces.data(), e, w);
        } else if (virial) {
            force<Real, false, true><<<g, kBlock>>>(n, x, b, idx, cnt, t, fp_.data(),
                                                    a.forces.data(), e, w);
        } else {
            force<Real, false, false><<<g, kBlock>>>(n, x, b, idx, cnt, t, fp_.data(),
                                                     a.forces.data(), e, w);
        }
        MDAPY_LAUNCHED();
        a.has_energies = energy;
        a.has_virials = virial;
    }

private:
    double cutoff_;
    int elements_;
    double rho_max_;
    DeviceRows<Real> pairs_;
    DeviceRows<double> embedding_;
    Buffer<int> density_, pair_;
    Buffer<Real> fp_;
};

}  // namespace

std::unique_ptr<Potential> make_eam(const calculators::EamPotential& potential,
                                    Precision precision) {
    if (precision == Precision::Mixed) {
        return std::make_unique<Eam<float>>(potential);
    }
    return std::make_unique<Eam<double>>(potential);
}

}  // namespace mdapy::cuda
