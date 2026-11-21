// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Potentials added together on the device, as calculators/_overlay.py adds
// them on the processor: each computes in turn, and what it gives is added
// to what the ones before it gave.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

__global__ void add_kernel(std::int64_t n, const double* in, double* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] += in[i];
}

void add(Buffer<double>& kept, double* out, std::int64_t n) {
    add_kernel<<<blocks(n), kBlock>>>(n, kept.data(), out);
    MDAPY_LAUNCHED();
}

void keep(Buffer<double>& kept, const double* in, std::int64_t n) {
    kept.resize(static_cast<std::size_t>(n));
    if (n > 0) {
        MDAPY_CUDA(cudaMemcpy(kept.data(), in, static_cast<std::size_t>(n) * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
}

class Overlay final : public Potential {
public:
    explicit Overlay(std::vector<Potential*> parts) : parts_(std::move(parts)) {
        if (parts_.size() < 2) {
            throw std::invalid_argument("an Overlay adds two potentials or more");
        }
        for (const Potential* p : parts_) {
            if (p == nullptr) throw std::invalid_argument("no potential");
            cutoff_ = std::max(cutoff_, p->cutoff());
            species_ = std::max(species_, p->species());
        }
    }

    double cutoff() const override { return cutoff_; }
    int species() const override { return species_; }

    void compute(Atoms& atoms, Neighbors& neighbors, const Box& box, bool energy, bool virial,
                 bool atomic) override {
        run(atoms, [&](Potential& p) { p.compute(atoms, neighbors, box, energy, virial, atomic); });
    }

    void compute_heat(Atoms& atoms, Neighbors& neighbors, const Box& box) override {
        run(atoms, [&](Potential& p) { p.compute_heat(atoms, neighbors, box); });
    }

private:
    // The first computes; each next one's forces, energies and virials have
    // those of the ones before added, where both computed them.
    template <class Compute>
    void run(Atoms& atoms, Compute compute) {
        Atoms::Impl& a = atoms.impl();
        const std::int64_t n = a.n;
        compute(*parts_[0]);
        for (std::size_t k = 1; k < parts_.size(); ++k) {
            const bool energies = a.has_energies, virials = a.has_virials;
            keep(forces_, a.forces.data(), 3 * n);
            if (energies) keep(energies_, a.energies.data(), n);
            if (virials) keep(virials_, a.virials.data(), 9 * n);
            compute(*parts_[k]);
            add(forces_, a.forces.data(), 3 * n);
            a.has_energies = energies && a.has_energies;
            a.has_virials = virials && a.has_virials;
            if (a.has_energies) add(energies_, a.energies.data(), n);
            if (a.has_virials) add(virials_, a.virials.data(), 9 * n);
        }
    }

    std::vector<Potential*> parts_;
    double cutoff_ = 0.0;
    int species_ = 0;
    Buffer<double> forces_, energies_, virials_;
};

}  // namespace

std::unique_ptr<Potential> make_overlay(std::vector<Potential*> parts) {
    return std::make_unique<Overlay>(std::move(parts));
}

}  // namespace mdapy::cuda
