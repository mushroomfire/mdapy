// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The atoms of one process of MPI rebuilt on the device at a build of the
// bricks (simulation/_domain_device.py), as LAMMPS's KOKKOS exchanges and
// borders its atoms: the ghosts dropped, the atoms that leave the brick taken
// out with all they carry and those that come in added, then the ghosts
// added after them, each set of rows chosen on the device, so that only what
// goes to other processes leaves it. What each atom carries with it: its
// position and species, the boxes it crossed, its mass, velocity, force and
// index among all; what the controls of a run set on the atoms the caller
// gives again after.

#include <thrust/iterator/counting_iterator.h>

#include <cub/cub.cuh>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/cuda.h"
#include "cuda/exchange.cuh"

namespace mdapy::cuda {

namespace {

template <class T>
__global__ void gather_rows(std::int64_t count, int width, const std::int32_t* rows,
                            const T* in, T* out) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count * width) return;
    const std::int64_t r = k / width, c = k % width;
    out[k] = in[static_cast<std::int64_t>(rows[r]) * width + c];
}

__global__ void wrap_kernel(std::int64_t n, DeviceBox<double> box, double4a* x, int3* images) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double4a p = x[i];
    const double r[3] = {p.x - box.lo[0], p.y - box.lo[1], p.z - box.lo[2]};
    double whole[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
        if (!box.pbc[k]) continue;
        const double s =
            r[0] * box.inverse[k] + r[1] * box.inverse[3 + k] + r[2] * box.inverse[6 + k];
        // a position gone wrong stays, for the caller to find
        if (isfinite(s)) whole[k] = floor(s);
    }
    if (whole[0] == 0.0 && whole[1] == 0.0 && whole[2] == 0.0) return;
    p.x -= whole[0] * box.h[0] + whole[1] * box.h[3] + whole[2] * box.h[6];
    p.y -= whole[0] * box.h[1] + whole[1] * box.h[4] + whole[2] * box.h[7];
    p.z -= whole[0] * box.h[2] + whole[1] * box.h[5] + whole[2] * box.h[8];
    x[i] = p;
    int3 m = images[i];
    m.x += static_cast<int>(whole[0]);
    m.y += static_cast<int>(whole[1]);
    m.z += static_cast<int>(whole[2]);
    images[i] = m;
}

// What each atom carries, in a fixed order: f(buffer, width).
template <class A, class F>
void each_carried(A& a, F&& f) {
    f(a.x, 1);
    f(a.species, 1);
    f(a.images, 1);
    f(a.masses, 1);
    f(a.ids, 1);
    f(a.velocities, 3);
    f(a.forces, 3);
}

// After the atoms changed number: what the controls set dropped, the energies
// and virials of a computation gone, the order the caller's.
void renumbered(Atoms::Impl& a, std::int64_t n) {
    a.n = n;
    a.owned = 0;
    a.original.resize(0);
    a.host_original.clear();
    a.stale = false;
    a.movable.resize(0);
    a.prescribed.resize(0);
    a.added.resize(0);
    a.groups.resize(0);
    a.sites.resize(0);
    a.spring_constants.resize(0);
    a.energies.resize(static_cast<std::size_t>(n));
    a.energies.zero();
    a.virials.resize(0);
    a.has_energies = a.has_virials = false;
    ++a.version;
    a.single_version = 0;
}

void check_carried(const Atoms::Impl& a) {
    const auto n = static_cast<std::size_t>(a.n);
    if (a.velocities.size() != 3 * n || a.masses.size() != n || a.ids.size() != n ||
        a.forces.size() != 3 * n) {
        throw std::logic_error("the atoms carry no masses, velocities, forces or ids");
    }
}

// The brick each atom is in along a vector, counts of them, as a step from
// at, the nearer way round along a periodic one; 127 for a position gone
// wrong. worst: the largest step, in size.
__global__ void steps_kernel(std::int64_t count, const double4a* x, double3 lo, double3 column,
                             int counts, int at, int periodic, std::int8_t* out, int* worst) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const double4a p = x[k];
    const double s = (p.x - lo.x) * column.x + (p.y - lo.y) * column.y + (p.z - lo.z) * column.z;
    if (!isfinite(s)) {
        out[k] = 127;
        atomicMax(worst, 127);
        return;
    }
    const double b = floor(s * counts);
    int brick = b < 0.0 ? 0 : (b > counts - 1 ? counts - 1 : static_cast<int>(b));
    int step = brick - at;
    if (periodic) {
        const int half = counts / 2;
        step = ((step + half) % counts + counts) % counts - half;
    }
    step = step < -100 ? -100 : (step > 100 ? 100 : step);
    out[k] = static_cast<std::int8_t>(step);
    if (step != 0) atomicMax(worst, step < 0 ? -step : step);
}

struct StepIs {
    const std::int8_t* step;
    int value;
    __device__ bool operator()(std::int32_t i) const { return step[i] == value; }
};

// Whether the fractional coordinate along a vector of a row is in [low, high).
struct InSlab {
    const double4a* x;
    double3 lo, column;
    double low, high;
    __device__ bool operator()(std::int32_t i) const {
        const double4a p = x[i];
        const double s =
            (p.x - lo.x) * column.x + (p.y - lo.y) * column.y + (p.z - lo.z) * column.z;
        return s >= low && s < high;
    }
};

// The rows from begin, count of them, that keep, in their order, into out;
// their number into found on the device.
template <class Keep>
void select(Exchange::Impl& e, std::int64_t begin, std::int64_t count, Keep keep,
            Buffer<std::int32_t>& out, int* found) {
    out.resize(static_cast<std::size_t>(count));
    if (count == 0) return;
    const thrust::counting_iterator<std::int32_t> rows(static_cast<std::int32_t>(begin));
    std::size_t bytes = 0;
    MDAPY_CUDA(cub::DeviceSelect::If(nullptr, bytes, rows, out.data(), found,
                                     static_cast<int>(count), keep));
    if (e.temp.size() < bytes) e.temp.resize(bytes);
    MDAPY_CUDA(cub::DeviceSelect::If(e.temp.data(), bytes, rows, out.data(), found,
                                     static_cast<int>(count), keep));
}

double3 column_of(const double* h, int d) {
    Box box{};
    std::memcpy(box.h, h, sizeof(box.h));
    const DeviceBox<double> dbox = device_box<double>(box);
    return make_double3(dbox.inverse[d], dbox.inverse[3 + d], dbox.inverse[6 + d]);
}

// The ghosts of rows: their positions plus shift, the species as w, then
// their ids.
__global__ void ghosts_kernel(std::int64_t count, const std::int32_t* rows, const double4a* x,
                              const std::int64_t* ids, double3 shift, double4a* out,
                              std::int64_t* out_ids) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    double4a p = x[rows[k]];
    p.x += shift.x;
    p.y += shift.y;
    p.z += shift.z;
    out[k] = p;
    out_ids[k] = ids[rows[k]];
}

__global__ void species_kernel(std::int64_t count, const double4a* x, std::int32_t* species) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count) species[k] = static_cast<std::int32_t>(x[k].w);
}

__global__ void inverse_kernel(std::int64_t n, const std::int32_t* original, std::int32_t* place) {
    const std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < n) place[original == nullptr ? p : original[p]] = static_cast<std::int32_t>(p);
}

// out[k] = place[rows[k]], or place[start + k] without rows
__global__ void places_kernel(std::int64_t count, const std::int32_t* rows, std::int64_t start,
                              const std::int32_t* place, std::int32_t* out) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count) out[k] = place[rows == nullptr ? start + k : rows[k]];
}

// buffer grown by count rows of width, the new ones copied from the device,
// or zero
template <class T>
void grown(Buffer<T>& buffer, std::int64_t n, std::int64_t count, int width, const T* from) {
    const std::size_t old = static_cast<std::size_t>(n * width);
    const std::size_t added = static_cast<std::size_t>(count * width);
    buffer.grow(old + added);
    if (added == 0) return;
    if (from != nullptr) {
        MDAPY_CUDA(cudaMemcpy(buffer.data() + old, from, added * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    } else {
        MDAPY_CUDA(cudaMemset(buffer.data() + old, 0, added * sizeof(T)));
    }
}

}  // namespace

void Atoms::truncate(std::int64_t count) {
    Impl& a = *impl_;
    check_carried(a);
    if (count < 0 || count > a.n) {
        throw std::invalid_argument("more rows than the atoms");
    }
    // the first rows, where they are
    each_carried(a, [&](auto& buffer, int width) { buffer.resize(static_cast<std::size_t>(count * width)); });
    renumbered(a, count);
}

void Atoms::settle() {
    Impl& a = *impl_;
    a.original.resize(0);
    a.host_original.clear();
    a.stale = false;
}

std::size_t Atoms::row_bytes() {
    return sizeof(double4a) + sizeof(std::int32_t) + sizeof(int3) + sizeof(double) +
           sizeof(std::int64_t) + 6 * sizeof(double);
}

void Atoms::wrap(const double* h, const double* lo, const bool* pbc) {
    Impl& a = *impl_;
    Box box{};
    std::memcpy(box.h, h, sizeof(box.h));
    std::memcpy(box.lo, lo, sizeof(box.lo));
    for (int k = 0; k < 3; ++k) box.pbc[k] = pbc[k];
    const std::int64_t m = a.active();
    if (m == 0) return;
    wrap_kernel<<<blocks(m), kBlock>>>(m, device_box<double>(box), a.x.data(), a.images.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

void Atoms::get_ids(std::int64_t* out) const {
    const Impl& a = *impl_;
    if (a.ids.size() != static_cast<std::size_t>(a.n)) {
        throw std::logic_error("the atoms have no ids");
    }
    std::vector<std::int64_t> placed(static_cast<std::size_t>(a.n));
    a.ids.download(placed.data(), placed.size());
    const auto& o = a.order();
    for (std::int64_t p = 0; p < a.n; ++p) out[o.empty() ? p : o[p]] = placed[p];
}

std::array<std::int64_t, 3> Exchange::leaving(const Atoms& atoms, int d, const double* h,
                                              const double* lo, int counts, int at,
                                              bool periodic) {
    Impl& e = *impl_;
    const Atoms::Impl& a = atoms.impl();
    if (d < 0 || d > 2 || counts < 1 || at < 0 || at >= counts) {
        throw std::invalid_argument("a vector or a brick outside the box");
    }
    const std::int64_t n = a.n;
    e.found.resize(4);
    e.found.zero();
    if (n > 0) {
        e.step.resize(static_cast<std::size_t>(n));
        steps_kernel<<<blocks(n), kBlock>>>(n, a.x.data(), make_double3(lo[0], lo[1], lo[2]),
                                            column_of(h, d), counts, at, periodic ? 1 : 0,
                                            e.step.data(), e.found.data() + 3);
        MDAPY_LAUNCHED();
    }
    select(e, 0, n, StepIs{e.step.data(), -1}, e.sides[0].chosen, e.found.data());
    select(e, 0, n, StepIs{e.step.data(), 1}, e.sides[1].chosen, e.found.data() + 1);
    select(e, 0, n, StepIs{e.step.data(), 0}, e.stay, e.found.data() + 2);
    int found[4];
    e.found.download(found, 4);
    e.sides[0].chosen_count = found[0];
    e.sides[1].chosen_count = found[1];
    e.stayed = found[2];
    e.sides[0].count = e.sides[1].count = 0;
    return {found[0], found[1], found[3]};
}

Exchange::View Exchange::pack_leaving(const Atoms& atoms, int side) {
    Impl& e = *impl_;
    if (side != 0 && side != 1) throw std::invalid_argument("side must be 0 or 1");
    const Atoms::Impl& a = atoms.impl();
    check_carried(a);
    Impl::Slot& s = e.sides[side];
    const std::int64_t count = s.chosen_count;
    const std::size_t bytes = static_cast<std::size_t>(count) * Atoms::row_bytes();
    s.out.resize(bytes);
    std::size_t at = 0;
    each_carried(a, [&](const auto& buffer, int width) {
        using T = std::remove_const_t<std::remove_reference_t<decltype(*buffer.data())>>;
        if (count > 0) {
            gather_rows<T><<<blocks(count * width), kBlock>>>(
                count, width, s.chosen.data(), buffer.data(),
                reinterpret_cast<T*>(s.out.data() + at));
            MDAPY_LAUNCHED();
        }
        at += static_cast<std::size_t>(count * width) * sizeof(T);
    });
    return e.staged(s, bytes);
}

Exchange::View Exchange::receive_arriving(int side, std::int64_t count) {
    Impl& e = *impl_;
    if (side != 0 && side != 1) throw std::invalid_argument("side must be 0 or 1");
    if (count < 0) throw std::invalid_argument("a count of atoms not negative");
    e.sides[side].count = count;
    return e.receive(e.sides[side], static_cast<std::size_t>(count) * Atoms::row_bytes());
}

void Exchange::migrate(Atoms& atoms) {
    Impl& e = *impl_;
    Atoms::Impl& a = atoms.impl();
    check_carried(a);
    const std::int64_t kept = e.stayed, c0 = e.sides[0].count, c1 = e.sides[1].count;
    const std::int64_t n = kept + c0 + c1;
    const unsigned char* in[2] = {
        e.arrived(e.sides[0], static_cast<std::size_t>(c0) * Atoms::row_bytes()),
        e.arrived(e.sides[1], static_cast<std::size_t>(c1) * Atoms::row_bytes())};
    const std::int64_t came[2] = {c0, c1};
    std::size_t at[2] = {0, 0};
    each_carried(a, [&](auto& buffer, int width) {
        using T = std::remove_reference_t<decltype(*buffer.data())>;
        const auto w = static_cast<std::size_t>(width);
        // the rows that stay, then those that came, gathered in the room of
        // the atoms, then back in the memory the buffer has
        a.spare.resize(static_cast<std::size_t>(n) * w * sizeof(T));
        T* out = reinterpret_cast<T*>(a.spare.data());
        if (kept > 0) {
            gather_rows<T><<<blocks(kept * width), kBlock>>>(kept, width, e.stay.data(),
                                                             buffer.data(), out);
            MDAPY_LAUNCHED();
        }
        std::size_t row = static_cast<std::size_t>(kept);
        for (int side = 0; side < 2; ++side) {
            const std::size_t length = static_cast<std::size_t>(came[side]) * w * sizeof(T);
            if (length > 0) {
                MDAPY_CUDA(cudaMemcpy(out + row * w, in[side] + at[side], length,
                                      cudaMemcpyDeviceToDevice));
            }
            at[side] += length;
            row += static_cast<std::size_t>(came[side]);
        }
        buffer.resize(static_cast<std::size_t>(n) * w);
        if (n > 0) {
            MDAPY_CUDA(cudaMemcpy(buffer.data(), out, static_cast<std::size_t>(n) * w * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
        }
    });
    e.stayed = 0;
    e.sides[0].chosen_count = e.sides[1].chosen_count = 0;
    e.sides[0].count = e.sides[1].count = 0;
    renumbered(a, n);
}

std::size_t Exchange::ghost_bytes() { return sizeof(double4a) + sizeof(std::int64_t); }

std::int64_t Exchange::border(const Atoms& atoms, int d, const double* h, const double* lo,
                              double low, double high, std::int64_t begin, std::int64_t end,
                              bool none, const double* shift) {
    Impl& e = *impl_;
    const Atoms::Impl& a = atoms.impl();
    if (d < 0 || d > 2 || begin < 0 || end < begin || end > a.n) {
        throw std::invalid_argument("rows or a vector outside the atoms and the box");
    }
    if (a.ids.size() != static_cast<std::size_t>(a.n)) {
        throw std::logic_error("the atoms have no ids");
    }
    if (static_cast<std::size_t>(e.used) == e.slots.size()) {
        e.slots.push_back(std::make_unique<Impl::Slot>());
    }
    Impl::Slot& s = *e.slots[static_cast<std::size_t>(e.used++)];
    s.start = s.count = 0;
    std::int64_t count = 0;
    if (!none && end > begin) {
        e.found.resize(1);
        e.found.zero();
        select(e, begin, end - begin,
               InSlab{a.x.data(), make_double3(lo[0], lo[1], lo[2]), column_of(h, d), low, high},
               s.chosen, e.found.data());
        int found = 0;
        e.found.download(&found, 1);
        count = found;
    }
    s.chosen_count = count;
    const std::size_t bytes = static_cast<std::size_t>(count) * ghost_bytes();
    s.out.resize(bytes);
    if (count > 0) {
        ghosts_kernel<<<blocks(count), kBlock>>>(
            count, s.chosen.data(), a.x.data(), a.ids.data(),
            make_double3(shift[0], shift[1], shift[2]), reinterpret_cast<double4a*>(s.out.data()),
            reinterpret_cast<std::int64_t*>(s.out.data() + count * sizeof(double4a)));
        MDAPY_LAUNCHED();
    }
    s.out_view = e.staged(s, bytes);
    return count;
}

Exchange::View Exchange::border_out(int k) const { return impl_->at(k).out_view; }

Exchange::View Exchange::border_in(int k, std::int64_t count) {
    if (count < 0) throw std::invalid_argument("a count of ghosts not negative");
    Impl::Slot& s = impl_->at(k);
    s.count = count;
    return impl_->receive(s, static_cast<std::size_t>(count) * ghost_bytes());
}

void Exchange::append_border(Atoms& atoms, int k) {
    Impl& e = *impl_;
    Impl::Slot& s = e.at(k);
    Atoms::Impl& a = atoms.impl();
    check_carried(a);
    const std::int64_t n = a.n, count = s.count;
    const unsigned char* in = e.arrived(s, static_cast<std::size_t>(count) * ghost_bytes());
    s.start = n;
    grown(a.x, n, count, 1, count > 0 ? reinterpret_cast<const double4a*>(in) : nullptr);
    grown(a.ids, n, count, 1,
          count > 0 ? reinterpret_cast<const std::int64_t*>(in + count * sizeof(double4a))
                    : nullptr);
    a.species.grow(static_cast<std::size_t>(n + count));
    if (count > 0) {
        species_kernel<<<blocks(count), kBlock>>>(count, a.x.data() + n, a.species.data() + n);
        MDAPY_LAUNCHED();
    }
    grown(a.images, n, count, 1, static_cast<const int3*>(nullptr));
    grown(a.masses, n, count, 1, static_cast<const double*>(nullptr));
    grown(a.velocities, n, count, 3, static_cast<const double*>(nullptr));
    grown(a.forces, n, count, 3, static_cast<const double*>(nullptr));
    renumbered(a, n + count);
}

void Exchange::place(const Atoms& atoms) {
    Impl& e = *impl_;
    const Atoms::Impl& a = atoms.impl();
    const std::int64_t n = a.n;
    e.place.resize(static_cast<std::size_t>(n));
    const bool sorted = a.original.size() == static_cast<std::size_t>(n);
    if (n > 0) {
        inverse_kernel<<<blocks(n), kBlock>>>(n, sorted ? a.original.data() : nullptr,
                                              e.place.data());
        MDAPY_LAUNCHED();
    }
    for (int k = 0; k < e.used; ++k) {
        Impl::Slot& s = *e.slots[static_cast<std::size_t>(k)];
        if (s.start + s.count > n) throw std::logic_error("a message outside the atoms");
        s.sent.resize(static_cast<std::size_t>(s.chosen_count));
        s.received.resize(static_cast<std::size_t>(s.count));
        if (s.chosen_count > 0) {
            places_kernel<<<blocks(s.chosen_count), kBlock>>>(s.chosen_count, s.chosen.data(), 0,
                                                              e.place.data(), s.sent.data());
            MDAPY_LAUNCHED();
        }
        if (s.count > 0) {
            places_kernel<<<blocks(s.count), kBlock>>>(s.count, nullptr, s.start,
                                                       e.place.data(), s.received.data());
            MDAPY_LAUNCHED();
        }
    }
}

}  // namespace mdapy::cuda
