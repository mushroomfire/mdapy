// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"

namespace mdapy::cuda {

namespace {

__global__ void set_w(std::int64_t n, const std::int32_t* species, double4a* x) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i].w = static_cast<double>(species[i]);
    }
}

__global__ void to_single(std::int64_t n, const double4a* x, float4* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const double4a a = x[i];
        out[i] = make_float4(static_cast<float>(a.x), static_cast<float>(a.y),
                             static_cast<float>(a.z), static_cast<float>(a.w));
    }
}

template <class T>
__global__ void gather(std::int64_t n, const std::int32_t* order, const T* in, T* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[order[i]];
}

// Three doubles of an atom, moved as one.
struct Triple {
    double v[3];
};

__global__ void identity(std::int64_t n, std::int32_t* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<std::int32_t>(i);
}

template <class T>
void gathered(std::int64_t n, const std::int32_t* order, Buffer<T>& buffer, Buffer<T>& spare) {
    if (buffer.size() != static_cast<std::size_t>(n)) return;  // not set
    spare.resize(static_cast<std::size_t>(n));
    gather<<<blocks(n), kBlock>>>(n, order, buffer.data(), spare.data());
    MDAPY_LAUNCHED();
    buffer.swap(spare);
}

}  // namespace

const std::vector<std::int32_t>& Atoms::Impl::order() const {
    if (stale) {
        host_original.resize(static_cast<std::size_t>(n));
        original.download(host_original.data(), host_original.size());
        stale = false;
    }
    return host_original;
}

template <class T>
void Atoms::Impl::to_places(const T* in, T* out, int width) const {
    const auto& o = order();
    for (std::int64_t p = 0; p < n; ++p) {
        const std::int64_t from = o.empty() ? p : o[p];
        for (int k = 0; k < width; ++k) out[p * width + k] = in[from * width + k];
    }
}

template <class T>
void Atoms::Impl::to_caller(const T* in, T* out, int width) const {
    const auto& o = order();
    for (std::int64_t p = 0; p < n; ++p) {
        const std::int64_t to = o.empty() ? p : o[p];
        for (int k = 0; k < width; ++k) out[to * width + k] = in[p * width + k];
    }
}

template void Atoms::Impl::to_places<double>(const double*, double*, int) const;
template void Atoms::Impl::to_places<std::int32_t>(const std::int32_t*, std::int32_t*, int) const;
template void Atoms::Impl::to_caller<double>(const double*, double*, int) const;
template void Atoms::Impl::to_caller<std::int32_t>(const std::int32_t*, std::int32_t*, int) const;

void Atoms::Impl::permute(const std::int32_t* order) {
    if (original.size() != static_cast<std::size_t>(n)) {
        original.resize(static_cast<std::size_t>(n));
        identity<<<blocks(n), kBlock>>>(n, original.data());
        MDAPY_LAUNCHED();
    }
    Buffer<double4a> x2;
    gathered(n, order, x, x2);
    Buffer<int3> m2;
    gathered(n, order, images, m2);
    Buffer<std::int32_t> s2;
    gathered(n, order, species, s2);
    Buffer<std::int32_t> o2;
    gathered(n, order, original, o2);
    Buffer<double> w2;
    gathered(n, order, masses, w2);
    Buffer<std::int32_t> h2;
    gathered(n, order, movable, h2);
    Buffer<std::int32_t> g2;
    gathered(n, order, groups, g2);
    Buffer<double> k2;
    gathered(n, order, spring_constants, k2);
    Buffer<std::int64_t> i2;
    gathered(n, order, ids, i2);
    // three doubles of each atom
    const auto triples = [&](Buffer<double>& buffer) {
        if (buffer.size() != static_cast<std::size_t>(3 * n)) return;
        Buffer<Triple> moved(static_cast<std::size_t>(n));
        gather<<<blocks(n), kBlock>>>(n, order, reinterpret_cast<const Triple*>(buffer.data()),
                                      moved.data());
        MDAPY_LAUNCHED();
        MDAPY_CUDA(cudaMemcpy(buffer.data(), moved.data(), static_cast<std::size_t>(n) * sizeof(Triple),
                              cudaMemcpyDeviceToDevice));
    };
    triples(velocities);
    triples(prescribed);
    triples(added);
    triples(sites);
    ++version;
    stale = true;
}

template <>
const double4a* Atoms::Impl::positions<double>() {
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
    impl_->images.resize(static_cast<std::size_t>(n));
    impl_->images.zero();
    impl_->version = 1;
}

Atoms::~Atoms() = default;

std::int64_t Atoms::size() const { return impl_->n; }

void Atoms::set_species(const std::int32_t* species) {
    Impl& a = *impl_;
    std::vector<std::int32_t> placed(static_cast<std::size_t>(a.n));
    a.to_places(species, placed.data(), 1);
    a.species.upload(placed.data(), placed.size());
    set_w<<<blocks(a.n), kBlock>>>(a.n, a.species.data(), a.x.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

void Atoms::set_positions(const double* positions) {
    Impl& a = *impl_;
    std::vector<double> placed(static_cast<std::size_t>(3 * a.n));
    a.to_places(positions, placed.data(), 3);
    std::vector<double4a> packed(static_cast<std::size_t>(a.n));
    for (std::int64_t i = 0; i < a.n; ++i) {
        packed[i] = double4a{placed[3 * i], placed[3 * i + 1], placed[3 * i + 2], 0.0};
    }
    a.x.upload(packed.data(), packed.size());
    set_w<<<blocks(a.n), kBlock>>>(a.n, a.species.data(), a.x.data());
    MDAPY_LAUNCHED();
    a.images.zero();
    ++a.version;
}

void Atoms::get_images(std::int32_t* images) const {
    const Impl& a = *impl_;
    std::vector<std::int32_t> placed(static_cast<std::size_t>(3 * a.n));
    a.images.download(reinterpret_cast<int3*>(placed.data()), static_cast<std::size_t>(a.n));
    a.to_caller(placed.data(), images, 3);
}

void Atoms::get_positions(double* positions) const {
    const Impl& a = *impl_;
    std::vector<double4a> packed(static_cast<std::size_t>(a.n));
    a.x.download(packed.data(), packed.size());
    std::vector<double> placed(static_cast<std::size_t>(3 * a.n));
    for (std::int64_t i = 0; i < a.n; ++i) {
        placed[3 * i] = packed[i].x;
        placed[3 * i + 1] = packed[i].y;
        placed[3 * i + 2] = packed[i].z;
    }
    a.to_caller(placed.data(), positions, 3);
}

void Atoms::set_forces(const double* forces) {
    Impl& a = *impl_;
    std::vector<double> placed(static_cast<std::size_t>(3 * a.n));
    a.to_places(forces, placed.data(), 3);
    a.forces.resize(placed.size());
    a.forces.upload(placed.data(), placed.size());
}

void Atoms::get_forces(double* forces) const {
    const Impl& a = *impl_;
    std::vector<double> placed(static_cast<std::size_t>(3 * a.n));
    a.forces.download(placed.data(), placed.size());
    a.to_caller(placed.data(), forces, 3);
}

void Atoms::get_energies(double* energies) const {
    if (!impl_->has_energies) {
        throw std::logic_error("no energies were computed");
    }
    std::vector<double> placed(static_cast<std::size_t>(impl_->n));
    impl_->energies.download(placed.data(), placed.size());
    impl_->to_caller(placed.data(), energies, 1);
}

void Atoms::get_virials(double* virials) const {
    const Impl& a = *impl_;
    if (!a.has_virials) {
        throw std::logic_error("no virials of the atoms were computed");
    }
    std::vector<double> columns(static_cast<std::size_t>(9 * a.n));
    a.virials.download(columns.data(), columns.size());
    std::vector<double> placed(columns.size());
    for (std::int64_t i = 0; i < a.n; ++i) {
        for (int k = 0; k < 9; ++k) {
            placed[9 * i + k] = columns[static_cast<std::size_t>(k * a.n + i)];
        }
    }
    a.to_caller(placed.data(), virials, 9);
}

double Atoms::total_energy() const {
    if (!impl_->has_energies) {
        throw std::logic_error("no energies were computed");
    }
    double out = 0.0;
    sum_columns(impl_->energies.data(), 1, impl_->active(), impl_->n, &out);
    return out;
}

void Atoms::total_virial(double* virial) const {
    const Impl& a = *impl_;
    if (!a.has_virials) {
        throw std::logic_error("no virial was computed");
    }
    sum_columns(a.virials.data(), 9, a.active(), a.n, virial);
}

void Atoms::set_owned(std::int64_t owned) {
    Impl& a = *impl_;
    if (owned < 0 || owned > a.n) {
        throw std::invalid_argument("owned must be between 0 and the number of atoms");
    }
    a.owned = owned;
}

std::int64_t Atoms::owned() const { return impl_->active(); }

void Atoms::set_ids(const std::int64_t* ids) {
    Impl& a = *impl_;
    if (ids == nullptr) {
        a.ids.resize(0);
        return;
    }
    const auto& o = a.order();
    std::vector<std::int64_t> placed(static_cast<std::size_t>(a.n));
    for (std::int64_t p = 0; p < a.n; ++p) placed[p] = ids[o.empty() ? p : o[p]];
    a.ids.upload(placed.data(), placed.size());
}

}  // namespace mdapy::cuda
