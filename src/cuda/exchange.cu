// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// What the processes of MPI exchange each step, gathered on the device into
// messages and scattered from them: the positions of the atoms that are the
// ghosts of other processes, shifted across periodic boundaries, and the
// forces on the ghosts, added to their atoms. The messages are those of the
// borders of the bricks (simulation/_domain.py), fixed between two builds.

#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/cuda.h"

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

// in the order of the message, so that an atom sent twice in one message
// would add in a fixed order; the rows of one message are distinct
__global__ void add_triples_kernel(std::int64_t count, const std::int32_t* places,
                                   const double* in, double* values) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::int64_t at = places[k];
    for (int c = 0; c < 3; ++c) values[3 * at + c] += in[3 * k + c];
}

}  // namespace

struct Exchange::Impl {
    // the places of the rows each message sends, and of those it receives
    std::vector<std::unique_ptr<Buffer<std::int32_t>>> sent, received;
    Buffer<double> scratch;
};

Exchange::Exchange() : impl_(std::make_unique<Impl>()) {}
Exchange::~Exchange() = default;

void Exchange::set(const Atoms& atoms, int k, const std::int32_t* rows, std::int64_t count,
                   std::int64_t start, std::int64_t received) {
    Impl& e = *impl_;
    const Atoms::Impl& a = atoms.impl();
    if (k < 0 || count < 0 || received < 0 || start < 0 || start + received > a.n) {
        throw std::invalid_argument("a message outside the atoms");
    }
    while (static_cast<std::size_t>(k) >= e.sent.size()) {
        e.sent.push_back(std::make_unique<Buffer<std::int32_t>>());
        e.received.push_back(std::make_unique<Buffer<std::int32_t>>());
    }
    // the place of each of the caller's rows
    const auto& order = a.order();
    std::vector<std::int32_t> place(static_cast<std::size_t>(a.n));
    for (std::int64_t p = 0; p < a.n; ++p) {
        place[order.empty() ? p : order[p]] = static_cast<std::int32_t>(p);
    }
    std::vector<std::int32_t> out(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        if (rows[i] < 0 || rows[i] >= a.n) {
            throw std::invalid_argument("a row of a message outside the atoms");
        }
        out[i] = place[rows[i]];
    }
    e.sent[k]->upload(out.data(), out.size());
    std::vector<std::int32_t> in(static_cast<std::size_t>(received));
    for (std::int64_t i = 0; i < received; ++i) in[i] = place[start + i];
    e.received[k]->upload(in.data(), in.size());
}

void Exchange::clear() {
    impl_->sent.clear();
    impl_->received.clear();
}

namespace {

const Buffer<std::int32_t>& message(
    const std::vector<std::unique_ptr<Buffer<std::int32_t>>>& all, int k) {
    if (k < 0 || static_cast<std::size_t>(k) >= all.size()) {
        throw std::invalid_argument("no such message");
    }
    return *all[static_cast<std::size_t>(k)];
}

}  // namespace

std::int64_t Exchange::sent(int k) const {
    return static_cast<std::int64_t>(message(impl_->sent, k).size());
}

std::int64_t Exchange::received(int k) const {
    return static_cast<std::int64_t>(message(impl_->received, k).size());
}

void Exchange::pack_positions(const Atoms& atoms, int k, const double* shift,
                              double* out) const {
    Impl& e = *impl_;
    const Buffer<std::int32_t>& places = message(e.sent, k);
    const auto count = static_cast<std::int64_t>(places.size());
    if (count == 0) return;
    e.scratch.resize(static_cast<std::size_t>(3 * count));
    pack_positions_kernel<<<blocks(count), kBlock>>>(
        count, places.data(), atoms.impl().x.data(), make_double3(shift[0], shift[1], shift[2]),
        e.scratch.data());
    MDAPY_LAUNCHED();
    e.scratch.download(out, static_cast<std::size_t>(3 * count));
}

void Exchange::unpack_positions(Atoms& atoms, int k, const double* in) const {
    Impl& e = *impl_;
    const Buffer<std::int32_t>& places = message(e.received, k);
    const auto count = static_cast<std::int64_t>(places.size());
    if (count == 0) return;
    e.scratch.resize(static_cast<std::size_t>(3 * count));
    e.scratch.upload(in, static_cast<std::size_t>(3 * count));
    Atoms::Impl& a = atoms.impl();
    unpack_positions_kernel<<<blocks(count), kBlock>>>(count, places.data(), e.scratch.data(),
                                                       a.x.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

void Exchange::pack_forces(const Atoms& atoms, int k, double* out) const {
    Impl& e = *impl_;
    const Buffer<std::int32_t>& places = message(e.received, k);
    const auto count = static_cast<std::int64_t>(places.size());
    if (count == 0) return;
    e.scratch.resize(static_cast<std::size_t>(3 * count));
    pack_triples_kernel<<<blocks(count), kBlock>>>(count, places.data(),
                                                   atoms.impl().forces.data(), e.scratch.data());
    MDAPY_LAUNCHED();
    e.scratch.download(out, static_cast<std::size_t>(3 * count));
}

void Exchange::add_forces(Atoms& atoms, int k, const double* in) const {
    Impl& e = *impl_;
    const Buffer<std::int32_t>& places = message(e.sent, k);
    const auto count = static_cast<std::int64_t>(places.size());
    if (count == 0) return;
    e.scratch.resize(static_cast<std::size_t>(3 * count));
    e.scratch.upload(in, static_cast<std::size_t>(3 * count));
    add_triples_kernel<<<blocks(count), kBlock>>>(count, places.data(), e.scratch.data(),
                                                  atoms.impl().forces.data());
    MDAPY_LAUNCHED();
}

}  // namespace mdapy::cuda
