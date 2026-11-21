// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The atoms of one process of MPI rebuilt on the device at a build of the
// bricks (simulation/_domain_device.py), as LAMMPS's KOKKOS exchanges and
// borders its atoms: the ghosts dropped, the atoms that leave the brick taken
// out with all they carry and those that come in added, then the ghosts
// added after them. What each atom carries with it: its position and species,
// the boxes it crossed, its mass, velocity, force and index among all; what
// the controls of a run set on the atoms the caller gives again after.

#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/cuda.h"

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

__global__ void fraction_kernel(std::int64_t begin, std::int64_t count, const double4a* x,
                                double3 lo, double3 column, double* out) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const double4a p = x[begin + k];
    out[k] = (p.x - lo.x) * column.x + (p.y - lo.y) * column.y + (p.z - lo.z) * column.z;
}

// The places of rows given from the host, on the device.
void device_rows(const std::int32_t* rows, std::int64_t count, std::int64_t n,
                 Buffer<std::int32_t>& out) {
    for (std::int64_t k = 0; k < count; ++k) {
        if (rows[k] < 0 || rows[k] >= n) {
            throw std::invalid_argument("a row outside the atoms");
        }
    }
    out.upload(rows, static_cast<std::size_t>(count));
}

// What each atom carries, in a fixed order: f(buffer, width).
template <class F>
void each_carried(Atoms::Impl& a, F&& f) {
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

}  // namespace

void Atoms::settle() {
    Impl& a = *impl_;
    a.original.resize(0);
    a.host_original.clear();
    a.stale = false;
}

void Atoms::keep_rows(const std::int32_t* rows, std::int64_t count) {
    Impl& a = *impl_;
    check_carried(a);
    Buffer<std::int32_t> places;
    device_rows(rows, count, a.n, places);
    each_carried(a, [&](auto& buffer, int width) {
        using T = std::remove_reference_t<decltype(*buffer.data())>;
        Buffer<T> kept(static_cast<std::size_t>(count * width));
        if (count > 0) {
            gather_rows<<<blocks(count * width), kBlock>>>(count, width, places.data(),
                                                           buffer.data(), kept.data());
            MDAPY_LAUNCHED();
        }
        buffer.swap(kept);
    });
    renumbered(a, count);
}

std::vector<unsigned char> Atoms::pack_rows(const std::int32_t* rows,
                                            std::int64_t count) const {
    Impl& a = *impl_;
    check_carried(a);
    Buffer<std::int32_t> places;
    device_rows(rows, count, a.n, places);
    std::vector<unsigned char> out;
    each_carried(a, [&](auto& buffer, int width) {
        using T = std::remove_reference_t<decltype(*buffer.data())>;
        const std::size_t size = static_cast<std::size_t>(count * width);
        if (size == 0) return;
        Buffer<T> packed(size);
        gather_rows<<<blocks(count * width), kBlock>>>(count, width, places.data(),
                                                       buffer.data(), packed.data());
        MDAPY_LAUNCHED();
        const std::size_t at = out.size();
        out.resize(at + size * sizeof(T));
        packed.download(reinterpret_cast<T*>(out.data() + at), size);
    });
    return out;
}

std::size_t Atoms::row_bytes() {
    return sizeof(double4a) + sizeof(std::int32_t) + sizeof(int3) + sizeof(double) +
           sizeof(std::int64_t) + 6 * sizeof(double);
}

namespace {

// buffer grown by count rows of width, the new ones from host or zero
template <class T>
void grown(Buffer<T>& buffer, std::int64_t n, std::int64_t count, int width, const T* host) {
    const std::size_t old = static_cast<std::size_t>(n * width);
    const std::size_t added = static_cast<std::size_t>(count * width);
    Buffer<T> out(old + added);
    if (old > 0) {
        MDAPY_CUDA(cudaMemcpy(out.data(), buffer.data(), old * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    }
    if (added > 0) {
        if (host != nullptr) {
            MDAPY_CUDA(cudaMemcpy(out.data() + old, host, added * sizeof(T),
                                  cudaMemcpyHostToDevice));
        } else {
            MDAPY_CUDA(cudaMemset(out.data() + old, 0, added * sizeof(T)));
        }
    }
    buffer.swap(out);
}

}  // namespace

void Atoms::append_rows(const unsigned char* bytes, std::size_t size, std::int64_t count) {
    Impl& a = *impl_;
    check_carried(a);
    if (size != static_cast<std::size_t>(count) * row_bytes()) {
        throw std::invalid_argument("the bytes of atoms packed by pack_rows");
    }
    std::size_t at = 0;
    const std::int64_t n = a.n;
    each_carried(a, [&](auto& buffer, int width) {
        using T = std::remove_reference_t<decltype(*buffer.data())>;
        const std::size_t length = static_cast<std::size_t>(count * width);
        std::vector<T> rows(length);
        if (length > 0) std::memcpy(rows.data(), bytes + at, length * sizeof(T));
        at += length * sizeof(T);
        grown(buffer, n, count, width, length > 0 ? rows.data() : nullptr);
    });
    renumbered(a, n + count);
}

void Atoms::append_ghosts(const double* positions, const std::int32_t* species,
                          const std::int64_t* ids, std::int64_t count) {
    Impl& a = *impl_;
    check_carried(a);
    const std::int64_t n = a.n;
    std::vector<double4a> x(static_cast<std::size_t>(count));
    for (std::int64_t k = 0; k < count; ++k) {
        x[k] = double4a{positions[3 * k], positions[3 * k + 1], positions[3 * k + 2],
                        static_cast<double>(species[k])};
    }
    grown(a.x, n, count, 1, count > 0 ? x.data() : nullptr);
    grown(a.species, n, count, 1, count > 0 ? species : nullptr);
    grown(a.images, n, count, 1, static_cast<const int3*>(nullptr));
    grown(a.masses, n, count, 1, static_cast<const double*>(nullptr));
    grown(a.ids, n, count, 1, count > 0 ? ids : nullptr);
    grown(a.velocities, n, count, 3, static_cast<const double*>(nullptr));
    grown(a.forces, n, count, 3, static_cast<const double*>(nullptr));
    renumbered(a, n + count);
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

void Atoms::fractions(int d, const double* h, const double* lo, std::int64_t begin,
                      std::int64_t end, double* out) const {
    const Impl& a = *impl_;
    if (d < 0 || d > 2 || begin < 0 || end < begin || end > a.n) {
        throw std::invalid_argument("rows or a vector outside the atoms and the box");
    }
    const std::int64_t count = end - begin;
    if (count == 0) return;
    Box box{};
    std::memcpy(box.h, h, sizeof(box.h));
    std::memcpy(box.lo, lo, sizeof(box.lo));
    const DeviceBox<double> dbox = device_box<double>(box);
    const double3 column =
        make_double3(dbox.inverse[d], dbox.inverse[3 + d], dbox.inverse[6 + d]);
    Buffer<double> values(static_cast<std::size_t>(count));
    fraction_kernel<<<blocks(count), kBlock>>>(begin, count, a.x.data(),
                                               make_double3(lo[0], lo[1], lo[2]), column,
                                               values.data());
    MDAPY_LAUNCHED();
    values.download(out, static_cast<std::size_t>(count));
}

void Atoms::gather_positions(const std::int32_t* rows, std::int64_t count, const double* shift,
                             double* out) const {
    const Impl& a = *impl_;
    if (count == 0) return;
    Buffer<std::int32_t> places;
    device_rows(rows, count, a.n, places);
    Buffer<double4a> packed(static_cast<std::size_t>(count));
    gather_rows<<<blocks(count), kBlock>>>(count, 1, places.data(), a.x.data(), packed.data());
    MDAPY_LAUNCHED();
    std::vector<double4a> host(static_cast<std::size_t>(count));
    packed.download(host.data(), host.size());
    for (std::int64_t k = 0; k < count; ++k) {
        out[3 * k] = host[k].x + shift[0];
        out[3 * k + 1] = host[k].y + shift[1];
        out[3 * k + 2] = host[k].z + shift[2];
    }
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

}  // namespace mdapy::cuda
