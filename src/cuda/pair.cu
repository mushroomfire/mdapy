// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Pair potentials on the device, Lennard-Jones and Morse, as
// calculators/pair.cpp computes them, from a full neighbor list.

#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

using calculators::PairStyle;

template <class Real>
struct PairTables {
    const Real4<Real>* coefficients;  // species x species
    const Real* cutsq;
    const Real* offset;
    int species;
};

template <class Real, PairStyle Style, bool Energy, bool Virial>
__global__ void pair_force(std::int64_t n, const Real4<Real>* x, DeviceBox<Real> box,
                           const std::int32_t* indices, const std::int32_t* counts,
                           PairTables<Real> t, double* forces, double* energies,
                           double* virials) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int si = static_cast<int>(xi.w);
    double f[3] = {0.0, 0.0, 0.0};
    double w[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double energy = 0.0;
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const int j = indices[static_cast<std::int64_t>(k) * n + i];
        const Real4<Real> xj = x[j];
        Real del[3] = {xi.x - xj.x, xi.y - xj.y, xi.z - xj.z};  // x_i - x_j
        minimum_image(box, del[0], del[1], del[2]);
        const Real r2 = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
        const int pair = si * t.species + static_cast<int>(xj.w);
        if (!(r2 < t.cutsq[pair])) continue;
        const Real4<Real> c = t.coefficients[pair];
        Real fpair, e;
        if (Style == PairStyle::LennardJones) {
            const Real r2inv = Real(1) / r2;
            const Real r6inv = r2inv * r2inv * r2inv;
            fpair = r6inv * (c.x * r6inv - c.y) * r2inv;
            e = r6inv * (c.z * r6inv - c.w) - t.offset[pair];
        } else {
            const Real r = sqrt(r2);
            const Real dexp = exp(-c.y * (r - c.z));
            fpair = c.w * (dexp * dexp - dexp) / r;
            e = c.x * (dexp * dexp - Real(2) * dexp) - t.offset[pair];
        }
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
        if (Energy) energy += static_cast<double>(Real(0.5) * e);
    }
    for (int a = 0; a < 3; ++a) {
        forces[3 * i + a] = f[a];
    }
    if (Energy) energies[i] = energy;
    if (Virial) {
        for (int c = 0; c < 9; ++c) {
            virials[static_cast<std::int64_t>(c) * n + i] = w[c];
        }
    }
}

template <class Real>
class Pair final : public Potential {
public:
    explicit Pair(const calculators::PairPotential& p)
        : style_(p.style), species_(p.species), cutoff_(p.cutoff) {
        const auto t = static_cast<std::size_t>(p.species);
        std::vector<Real4<Real>> coefficients(t * t);
        std::vector<Real> cutsq(t * t), offset(t * t);
        for (std::size_t k = 0; k < t * t; ++k) {
            coefficients[k].x = static_cast<Real>(p.coefficients[4 * k]);
            coefficients[k].y = static_cast<Real>(p.coefficients[4 * k + 1]);
            coefficients[k].z = static_cast<Real>(p.coefficients[4 * k + 2]);
            coefficients[k].w = static_cast<Real>(p.coefficients[4 * k + 3]);
            cutsq[k] = static_cast<Real>(p.cutsq[k]);
            offset[k] = static_cast<Real>(p.offset[k]);
        }
        coefficients_.upload(coefficients.data(), coefficients.size());
        cutsq_.upload(cutsq.data(), cutsq.size());
        offset_.upload(offset.data(), offset.size());
    }

    double cutoff() const override { return cutoff_; }
    int species() const override { return species_; }

    void compute(Atoms& atoms, const Neighbors& neighbors, const Box& box, bool energy,
                 bool virial, bool atomic_virials) override {
        Atoms::Impl& a = atoms.impl();
        const Neighbors::Impl& l = neighbors.impl();
        if (l.n != a.n || l.builds == 0 || l.cutoff + 1e-12 < cutoff_) {
            throw std::logic_error("the neighbor list does not fit the atoms or the cutoff");
        }
        virial = virial || atomic_virials;
        const std::int64_t n = a.n;
        if (virial) a.virials.resize(static_cast<std::size_t>(9 * n));
        if (style_ == PairStyle::LennardJones) {
            launch<PairStyle::LennardJones>(a, l, box, energy, virial);
        } else {
            launch<PairStyle::Morse>(a, l, box, energy, virial);
        }
        a.has_energies = energy;
        a.has_virials = virial;
    }

private:
    template <PairStyle Style>
    void launch(Atoms::Impl& a, const Neighbors::Impl& l, const Box& box, bool energy,
                bool virial) {
        const std::int64_t n = a.n;
        const Real4<Real>* x = a.positions<Real>();
        const DeviceBox<Real> b = device_box<Real>(box);
        PairTables<Real> t{coefficients_.data(), cutsq_.data(), offset_.data(), species_};
        const unsigned g = blocks(n);
        double* e = energy ? a.energies.data() : nullptr;
        double* w = virial ? a.virials.data() : nullptr;
        const auto* idx = l.indices.data();
        const auto* cnt = l.counts.data();
        if (energy && virial) {
            pair_force<Real, Style, true, true>
                <<<g, kBlock>>>(n, x, b, idx, cnt, t, a.forces.data(), e, w);
        } else if (energy) {
            pair_force<Real, Style, true, false>
                <<<g, kBlock>>>(n, x, b, idx, cnt, t, a.forces.data(), e, w);
        } else if (virial) {
            pair_force<Real, Style, false, true>
                <<<g, kBlock>>>(n, x, b, idx, cnt, t, a.forces.data(), e, w);
        } else {
            pair_force<Real, Style, false, false>
                <<<g, kBlock>>>(n, x, b, idx, cnt, t, a.forces.data(), e, w);
        }
        MDAPY_LAUNCHED();
    }

    PairStyle style_;
    int species_;
    double cutoff_;
    Buffer<Real4<Real>> coefficients_;
    Buffer<Real> cutsq_, offset_;
};

}  // namespace

std::unique_ptr<Potential> make_pair(const calculators::PairPotential& potential,
                                     Precision precision) {
    if (precision == Precision::Mixed) {
        return std::make_unique<Pair<float>>(potential);
    }
    return std::make_unique<Pair<double>>(potential);
}

}  // namespace mdapy::cuda
