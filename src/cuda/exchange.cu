// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// What the processes of MPI exchange each step, gathered on the device into
// messages and scattered from them: the positions of the atoms that are the
// ghosts of other processes, shifted across periodic boundaries, values of
// the atoms a potential needs on its ghosts, and the forces on the ghosts,
// added to their atoms. The messages are those of the borders of the bricks
// (domain.cu), fixed between two builds.

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda/exchange.cuh"

namespace mdapy::cuda {

namespace {

__global__ void pack_positions_kernel(std::int64_t count, const std::int32_t* places,
                                      const double4a* x, double3 shift, double* out) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const double4a p = x[places[k]];
    out[3 * k] = p.x + shift.x;
    out[3 * k + 1] = p.y + shift.y;
    out[3 * k + 2] = p.z + shift.z;
}

__global__ void unpack_positions_kernel(std::int64_t count, const std::int32_t* places,
                                        const double* in, double4a* x) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    double4a& p = x[places[k]];
    p.x = in[3 * k];
    p.y = in[3 * k + 1];
    p.z = in[3 * k + 2];
}

__global__ void pack_triples_kernel(std::int64_t count, const std::int32_t* places,
                                    const double* values, double* out) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::int64_t at = places[k];
    for (int c = 0; c < 3; ++c) out[3 * k + c] = values[3 * at + c];
}

// the rows of one message are distinct
__global__ void add_triples_kernel(std::int64_t count, const std::int32_t* places,
                                   const double* in, double* values) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::int64_t at = places[k];
    for (int c = 0; c < 3; ++c) values[3 * at + c] += in[3 * k + c];
}

__global__ void pack_bytes_kernel(std::int64_t total, std::int64_t bytes,
                                  const std::int32_t* places, const unsigned char* values,
                                  unsigned char* out) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= total) return;
    const std::int64_t row = k / bytes, b = k % bytes;
    out[k] = values[static_cast<std::int64_t>(places[row]) * bytes + b];
}

__global__ void unpack_bytes_kernel(std::int64_t total, std::int64_t bytes,
                                    const std::int32_t* places, const unsigned char* in,
                                    unsigned char* values) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= total) return;
    const std::int64_t row = k / bytes, b = k % bytes;
    values[static_cast<std::int64_t>(places[row]) * bytes + b] = in[k];
}

// the 9 columns, of pitch n, of the rows at places
__global__ void pack_columns_kernel(std::int64_t count, std::int64_t n, const std::int32_t* places,
                                    const double* values, double* out) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::int64_t at = places[k];
    for (int c = 0; c < 9; ++c) out[9 * k + c] = values[c * n + at];
}

__global__ void add_columns_kernel(std::int64_t count, std::int64_t n, const std::int32_t* places,
                                   const double* in, double* values) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::int64_t at = places[k];
    for (int c = 0; c < 9; ++c) values[c * n + at] += in[9 * k + c];
}

void check_virials(const Atoms::Impl& a) {
    if (a.virials.size() != static_cast<std::size_t>(9 * a.n)) {
        throw std::logic_error("no virials of the atoms were computed");
    }
}

}  // namespace

Exchange::Exchange() : impl_(std::make_unique<Impl>()) {}
Exchange::~Exchange() = default;

void Exchange::set_device(bool device) { impl_->device = device; }
bool Exchange::device() const { return impl_->device; }

void Exchange::clear() { impl_->used = 0; }
int Exchange::messages() const { return impl_->used; }

std::int64_t Exchange::sent(int k) const {
    return static_cast<std::int64_t>(impl_->at(k).sent.size());
}

std::int64_t Exchange::received(int k) const {
    return static_cast<std::int64_t>(impl_->at(k).received.size());
}

Exchange::View Exchange::pack_positions(const Atoms& atoms, int k, const double* shift) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    const auto count = static_cast<std::int64_t>(s.sent.size());
    const auto bytes = static_cast<std::size_t>(count) * 3 * sizeof(double);
    s.out.resize(bytes);
    if (count > 0) {
        pack_positions_kernel<<<blocks(count), kBlock>>>(
            count, s.sent.data(), atoms.impl().x.data(), make_double3(shift[0], shift[1], shift[2]),
            reinterpret_cast<double*>(s.out.data()));
        MDAPY_LAUNCHED();
    }
    return e.staged(s, bytes);
}

Exchange::View Exchange::receive_positions(int k) {
    Impl::Slot& s = impl_->at(k);
    return impl_->receive(s, s.received.size() * 3 * sizeof(double));
}

void Exchange::unpack_positions(Atoms& atoms, int k) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    const auto count = static_cast<std::int64_t>(s.received.size());
    const unsigned char* in =
        e.arrived(s, static_cast<std::size_t>(count) * 3 * sizeof(double));
    if (count == 0) return;
    Atoms::Impl& a = atoms.impl();
    unpack_positions_kernel<<<blocks(count), kBlock>>>(
        count, s.received.data(), reinterpret_cast<const double*>(in), a.x.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

Exchange::View Exchange::pack_values(const void* values, std::size_t bytes, int k) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    const auto total = static_cast<std::int64_t>(s.sent.size() * bytes);
    s.out.resize(static_cast<std::size_t>(total));
    if (total > 0) {
        pack_bytes_kernel<<<blocks(total), kBlock>>>(
            total, static_cast<std::int64_t>(bytes), s.sent.data(),
            static_cast<const unsigned char*>(values), s.out.data());
        MDAPY_LAUNCHED();
    }
    return e.staged(s, static_cast<std::size_t>(total));
}

Exchange::View Exchange::receive_values(std::size_t bytes, int k) {
    Impl::Slot& s = impl_->at(k);
    return impl_->receive(s, s.received.size() * bytes);
}

void Exchange::unpack_values(void* values, std::size_t bytes, int k) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    const auto total = static_cast<std::int64_t>(s.received.size() * bytes);
    const unsigned char* in = e.arrived(s, static_cast<std::size_t>(total));
    if (total == 0) return;
    unpack_bytes_kernel<<<blocks(total), kBlock>>>(total, static_cast<std::int64_t>(bytes),
                                                   s.received.data(), in,
                                                   static_cast<unsigned char*>(values));
    MDAPY_LAUNCHED();
}

Exchange::View Exchange::pack_forces(const Atoms& atoms, int k) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    const auto count = static_cast<std::int64_t>(s.received.size());
    const auto bytes = static_cast<std::size_t>(count) * 3 * sizeof(double);
    s.out.resize(bytes);
    if (count > 0) {
        pack_triples_kernel<<<blocks(count), kBlock>>>(count, s.received.data(),
                                                       atoms.impl().forces.data(),
                                                       reinterpret_cast<double*>(s.out.data()));
        MDAPY_LAUNCHED();
    }
    return e.staged(s, bytes);
}

Exchange::View Exchange::receive_forces(int k) {
    Impl::Slot& s = impl_->at(k);
    return impl_->receive(s, s.sent.size() * 3 * sizeof(double));
}

void Exchange::add_forces(Atoms& atoms, int k) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    const auto count = static_cast<std::int64_t>(s.sent.size());
    const unsigned char* in =
        e.arrived(s, static_cast<std::size_t>(count) * 3 * sizeof(double));
    if (count == 0) return;
    add_triples_kernel<<<blocks(count), kBlock>>>(count, s.sent.data(),
                                                  reinterpret_cast<const double*>(in),
                                                  atoms.impl().forces.data());
    MDAPY_LAUNCHED();
}

Exchange::View Exchange::pack_virials(const Atoms& atoms, int k) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    const Atoms::Impl& a = atoms.impl();
    check_virials(a);
    const auto count = static_cast<std::int64_t>(s.received.size());
    const auto bytes = static_cast<std::size_t>(count) * 9 * sizeof(double);
    s.out.resize(bytes);
    if (count > 0) {
        pack_columns_kernel<<<blocks(count), kBlock>>>(count, a.n, s.received.data(),
                                                       a.virials.data(),
                                                       reinterpret_cast<double*>(s.out.data()));
        MDAPY_LAUNCHED();
    }
    return e.staged(s, bytes);
}

Exchange::View Exchange::receive_virials(int k) {
    Impl::Slot& s = impl_->at(k);
    return impl_->receive(s, s.sent.size() * 9 * sizeof(double));
}

void Exchange::add_virials(Atoms& atoms, int k) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    Atoms::Impl& a = atoms.impl();
    check_virials(a);
    const auto count = static_cast<std::int64_t>(s.sent.size());
    const unsigned char* in =
        e.arrived(s, static_cast<std::size_t>(count) * 9 * sizeof(double));
    if (count == 0) return;
    add_columns_kernel<<<blocks(count), kBlock>>>(count, a.n, s.sent.data(),
                                                  reinterpret_cast<const double*>(in),
                                                  a.virials.data());
    MDAPY_LAUNCHED();
}

}  // namespace mdapy::cuda
