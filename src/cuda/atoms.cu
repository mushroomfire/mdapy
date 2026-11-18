// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"

namespace mdapy::cuda {

namespace {

__global__ void set_w(std::int64_t n, const std::int32_t* species, double4* x) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i].w = static_cast<double>(species[i]);
    }
}

__global__ void to_single(std::int64_t n, const double4* x, float4* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const double4 a = x[i];
        out[i] = make_float4(static_cast<float>(a.x), static_cast<float>(a.y),
                             static_cast<float>(a.z), static_cast<float>(a.w));
    }
}

}  // namespace

template <>
const double4* Atoms::Impl::positions<double>() {
    return x.data();
}

template <>
const float4* Atoms::Impl::positions<float>() {
    if (single_version != version) {
        x_single.resize(static_cast<std::size_t>(n));
        to_single<<<blocks(n), kBlock>>>(n, x.data(), x_single.data());
        MDAPY_LAUNCHED();
        single_version = version;
    }
    return x_single.data();
}

Atoms::Atoms(std::int64_t n) : impl_(std::make_unique<Impl>()) {
    if (n < 1) {
        throw std::invalid_argument("the GPU needs atoms");
    }
    impl_->n = n;
    impl_->x.resize(static_cast<std::size_t>(n));
    impl_->x.zero();
    impl_->species.resize(static_cast<std::size_t>(n));
    impl_->species.zero();
    impl_->forces.resize(static_cast<std::size_t>(3 * n));
    impl_->forces.zero();
    impl_->energies.resize(static_cast<std::size_t>(n));
    impl_->version = 1;
}

Atoms::~Atoms() = default;

std::int64_t Atoms::size() const { return impl_->n; }

void Atoms::set_species(const std::int32_t* species) {
    Impl& a = *impl_;
    a.species.upload(species, static_cast<std::size_t>(a.n));
    set_w<<<blocks(a.n), kBlock>>>(a.n, a.species.data(), a.x.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

void Atoms::set_positions(const double* positions) {
    Impl& a = *impl_;
    std::vector<double4> packed(static_cast<std::size_t>(a.n));
    for (std::int64_t i = 0; i < a.n; ++i) {
        packed[i] = make_double4(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2],
                                 0.0);
    }
    a.x.upload(packed.data(), packed.size());
    set_w<<<blocks(a.n), kBlock>>>(a.n, a.species.data(), a.x.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

void Atoms::get_positions(double* positions) const {
    const Impl& a = *impl_;
    std::vector<double4> packed(static_cast<std::size_t>(a.n));
    a.x.download(packed.data(), packed.size());
    for (std::int64_t i = 0; i < a.n; ++i) {
        positions[3 * i] = packed[i].x;
        positions[3 * i + 1] = packed[i].y;
        positions[3 * i + 2] = packed[i].z;
    }
}

void Atoms::get_forces(double* forces) const {
    impl_->forces.download(forces, static_cast<std::size_t>(3 * impl_->n));
}

void Atoms::get_energies(double* energies) const {
    if (!impl_->has_energies) {
        throw std::logic_error("no energies were computed");
    }
    impl_->energies.download(energies, static_cast<std::size_t>(impl_->n));
}

void Atoms::get_virials(double* virials) const {
    const Impl& a = *impl_;
    if (!a.has_virials) {
        throw std::logic_error("no virials of the atoms were computed");
    }
    std::vector<double> columns(static_cast<std::size_t>(9 * a.n));
    a.virials.download(columns.data(), columns.size());
    for (std::int64_t i = 0; i < a.n; ++i) {
        for (int k = 0; k < 9; ++k) {
            virials[9 * i + k] = columns[static_cast<std::size_t>(k * a.n + i)];
        }
    }
}

double Atoms::total_energy() const {
    if (!impl_->has_energies) {
        throw std::logic_error("no energies were computed");
    }
    double out = 0.0;
    sum_columns(impl_->energies.data(), 1, impl_->n, &out);
    return out;
}

void Atoms::total_virial(double* virial) const {
    const Impl& a = *impl_;
    if (!a.has_virials) {
        throw std::logic_error("no virial was computed");
    }
    sum_columns(a.virials.data(), 9, a.n, virial);
}

}  // namespace mdapy::cuda
