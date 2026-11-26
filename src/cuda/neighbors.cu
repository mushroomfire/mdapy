// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The neighbor list on the device: the atoms binned into cells at least as
// wide as the cutoff and the skin, sorted by cell with a stable sort, so that
// the list comes out in the same order every time; each atom then looks at
// the atoms of its cell and the cells around it.

#include "cuda/primitives.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/atoms.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

constexpr double kPi = 3.14159265358979323846;

// The thickness of the box along each vector: its volume over the area of
// the face of the other two.
void thickness(const Box& box, double out[3]) {
    const double* h = box.h;
    auto cross = [&](int a, int b, double c[3]) {
        c[0] = h[3 * a + 1] * h[3 * b + 2] - h[3 * a + 2] * h[3 * b + 1];
        c[1] = h[3 * a + 2] * h[3 * b] - h[3 * a] * h[3 * b + 2];
        c[2] = h[3 * a] * h[3 * b + 1] - h[3 * a + 1] * h[3 * b];
    };
    double c[3];
    cross(1, 2, c);
    const double volume = std::fabs(h[0] * c[0] + h[1] * c[1] + h[2] * c[2]);
    for (int k = 0; k < 3; ++k) {
        cross((k + 1) % 3, (k + 2) % 3, c);
        out[k] = volume / std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    }
}

double volume_of(const Box& box) {
    const double* h = box.h;
    return std::fabs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                     h[2] * (h[3] * h[7] - h[4] * h[6]));
}

// m = inverse(a) b for 3 x 3 matrices of rows.
void inverse_times(const Box& a, const Box& b, double m[9]) {
    const DeviceBox<double> da = device_box<double>(a);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += da.inverse[3 * r + k] * b.h[3 * k + c];
            }
            m[3 * r + c] = s;
        }
    }
}

// Each atom into the box along its periodic directions, counting the boxes
// it was moved by.
__global__ void wrap(std::int64_t n, DeviceBox<double> box, double4a* x, int3* images) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double4a p = x[i];
    const double r[3] = {p.x - box.lo[0], p.y - box.lo[1], p.z - box.lo[2]};
    int shift[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k) {
        if (!box.pbc[k]) continue;
        const double s = r[0] * box.inverse[k] + r[1] * box.inverse[3 + k] + r[2] * box.inverse[6 + k];
        shift[k] = static_cast<int>(floor(s));
    }
    if (shift[0] == 0 && shift[1] == 0 && shift[2] == 0) return;
    p.x -= shift[0] * box.h[0] + shift[1] * box.h[3] + shift[2] * box.h[6];
    p.y -= shift[0] * box.h[1] + shift[1] * box.h[4] + shift[2] * box.h[7];
    p.z -= shift[0] * box.h[2] + shift[1] * box.h[5] + shift[2] * box.h[8];
    x[i] = p;
    int3 m = images[i];
    m.x += shift[0];
    m.y += shift[1];
    m.z += shift[2];
    images[i] = m;
}

// The ghosts, from place first_ghost on, are given keys of their own after
// those of the atoms of the process, so that the sort keeps them last.
__global__ void bin(std::int64_t n, const double4a* x, DeviceBox<double> box, int3 cells,
                    std::int64_t first_ghost, std::int32_t* keys, std::int32_t* order) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4a p = x[i];
    const double r[3] = {p.x - box.lo[0], p.y - box.lo[1], p.z - box.lo[2]};
    const int count[3] = {cells.x, cells.y, cells.z};
    int c[3];
    for (int k = 0; k < 3; ++k) {
        // the atoms are wrapped, and an atom the wrap left at 1 by rounding
        // stays in the last cell, where it is, not the first: the cells must
        // hold the atoms as they are for the shifts of the list. Atoms
        // outside a direction that is not periodic go to its first or last
        // cell, which the cells around them still reach
        const double s =
            r[0] * box.inverse[k] + r[1] * box.inverse[3 + k] + r[2] * box.inverse[6 + k];
        const double scaled = s * count[k];
        // (a position that is not finite lands in a cell too, never outside)
        c[k] = !(scaled >= 0.0) ? 0
                                : (scaled >= count[k] ? count[k] - 1 : static_cast<int>(scaled));
    }
    const int total = cells.x * cells.y * cells.z;
    keys[i] = (c[2] * cells.y + c[1]) * cells.x + c[0] + (i >= first_ghost ? total : 0);
    order[i] = static_cast<std::int32_t>(i);
}

__global__ void bounds(std::int64_t n, const std::int32_t* keys, std::int32_t* begin,
                       std::int32_t* end) {
    const std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const std::int32_t k = keys[p];
    if (p == 0 || keys[p - 1] != k) begin[k] = static_cast<std::int32_t>(p);
    if (p == n - 1 || keys[p + 1] != k) end[k] = static_cast<std::int32_t>(p + 1);
}

// The cells around a cell along a direction: offsets from low to high,
// wrapped when periodic, else those inside.
struct Around {
    int low[3], high[3], wrap[3];
};

// Shifts: at least 3 cells along every periodic direction, so that the
// cells around an atom are distinct and the box each lies through is that
// of the wrap of its index: x_j - x_i + s h, s of the cell, for every atom of
// it. Otherwise the shortest image of each pair from its fractional
// coordinates.
// With shifts the pairs are found in single precision, x and the box h in
// float, within the reach and a margin larger than their rounding: a list
// with a few pairs more, never fewer. Without, in double precision.
// Each atom writes its pairs in a row of its own, rows[i * most + m], so that
// its stores fill whole sectors; columns() makes the columns of the list.
template <bool Shifts>
__global__ void list(std::int64_t n, const double4a* x, const float4* xf, DeviceBox<double> box,
                     DeviceBox<float> boxf, int3 cells, Around around, const std::int32_t* keys,
                     const std::int32_t* begin, const std::int32_t* end, int ranges,
                     double reach2, float reach2f, int most, std::int32_t* rows,
                     std::int32_t* counts, int* found) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4a xi = x[i];
    const float4 fi = Shifts ? xf[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int total = cells.x * cells.y * cells.z;
    const int key = keys[i] % total;  // the atoms are in the order of their cells
    const int own[3] = {key % cells.x, (key / cells.x) % cells.y, key / (cells.x * cells.y)};
    const int count[3] = {cells.x, cells.y, cells.z};
    int m = 0;
    for (int dz = around.low[2]; dz <= around.high[2]; ++dz) {
        int cz = own[2] + dz, sz = 0;
        if (around.wrap[2]) {
            if (cz < 0) { cz += count[2]; sz = -1; }
            else if (cz >= count[2]) { cz -= count[2]; sz = 1; }
        } else if (cz < 0 || cz >= count[2]) continue;
        for (int dy = around.low[1]; dy <= around.high[1]; ++dy) {
            int cy = own[1] + dy, sy = 0;
            if (around.wrap[1]) {
                if (cy < 0) { cy += count[1]; sy = -1; }
                else if (cy >= count[1]) { cy -= count[1]; sy = 1; }
            } else if (cy < 0 || cy >= count[1]) continue;
            for (int dx = around.low[0]; dx <= around.high[0]; ++dx) {
                int cx = own[0] + dx, sx = 0;
                if (around.wrap[0]) {
                    if (cx < 0) { cx += count[0]; sx = -1; }
                    else if (cx >= count[0]) { cx -= count[0]; sx = 1; }
                } else if (cx < 0 || cx >= count[0]) continue;
                const int cell = (cz * cells.y + cy) * cells.x + cx;
                // the image of the cell: the atoms of the cell before the wrap
                const float image[3] = {sx * boxf.h[0] + sy * boxf.h[3] + sz * boxf.h[6],
                                        sx * boxf.h[1] + sy * boxf.h[4] + sz * boxf.h[7],
                                        sx * boxf.h[2] + sy * boxf.h[5] + sz * boxf.h[8]};
                // the code of the pairs: the cell around, with cells; the image
                // of the box of each pair, found below, without
                const int cell_code = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1);
                // the atoms of the process in the cell, then its ghosts there
                for (int range = 0; range < ranges; ++range)
                for (int j = begin[cell + range * total]; j < end[cell + range * total]; ++j) {
                    if (j == i) continue;
                    int code = cell_code;
                    bool near;
                    if (Shifts) {
                        const float4 fj = xf[j];
                        const float r0 = fj.x - fi.x + image[0], r1 = fj.y - fi.y + image[1],
                                    r2 = fj.z - fi.z + image[2];
                        near = r0 * r0 + r1 * r1 + r2 * r2 < reach2f;
                    } else {
                        const double4a xj = x[j];
                        double r[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
                        // the shortest image: atoms in the box along the periodic
                        // directions are at most one box apart
                        const double d[3] = {r[0], r[1], r[2]};
                        code = 13;
                        int stride = 1;
                        for (int k = 0; k < 3; ++k, stride *= 3) {
                            if (!box.pbc[k]) continue;
                            const double s = d[0] * box.inverse[k] + d[1] * box.inverse[3 + k] +
                                             d[2] * box.inverse[6 + k];
                            const int shift = -static_cast<int>(rint(s));
                            if (shift != 0) {
                                for (int c = 0; c < 3; ++c) r[c] += shift * box.h[3 * k + c];
                                code += shift * stride;
                            }
                        }
                        near = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] < reach2;
                    }
                    if (near) {
                        if (m < most) {
                            rows[i * most + m] = static_cast<std::int32_t>(
                                static_cast<std::uint32_t>(j) | (static_cast<std::uint32_t>(code) << kIndexBits));
                        }
                        ++m;
                    }
                }
            }
        }
    }
    counts[i] = m < most ? m : most;
    atomicMax(found, m);
}

// The columns of the list from its rows, out[k * n + i] = rows[i * most + k]
// for k < width, through tiles of 32 atoms by 32 pairs in shared memory, so
// that both the reads and the writes are coalesced.
__global__ void columns(std::int64_t n, int most, int width, const std::int32_t* rows,
                        std::int32_t* out) {
    __shared__ std::int32_t tile[32][33];
    const std::int64_t i0 = static_cast<std::int64_t>(blockIdx.x) * 32;
    const int k0 = static_cast<int>(blockIdx.y) * 32;
    for (int r = static_cast<int>(threadIdx.y); r < 32; r += static_cast<int>(blockDim.y)) {
        const std::int64_t i = i0 + r;
        const int k = k0 + static_cast<int>(threadIdx.x);
        if (i < n && k < width) tile[r][threadIdx.x] = rows[i * most + k];
    }
    __syncthreads();
    for (int r = static_cast<int>(threadIdx.y); r < 32; r += static_cast<int>(blockDim.y)) {
        const int k = k0 + r;
        const std::int64_t i = i0 + static_cast<std::int64_t>(threadIdx.x);
        if (i < n && k < width) out[static_cast<std::int64_t>(k) * n + i] = tile[threadIdx.x][r];
    }
}

struct Matrix {
    double m[9];
};

// The position of each atom from the corner of its cell, the cell of the
// last build in the box now, which moves with the box: l = x - lo - sum
// c_k / n_k h_k, in a precision, the species as w.
template <class Real>
__global__ void to_local(std::int64_t n, const double4a* x, const std::int32_t* keys, int3 cells,
                         int with_cells, Matrix h, double3 lo, Real4<Real>* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4a p = x[i];
    double f[3] = {0.0, 0.0, 0.0};
    if (with_cells) {
        const int key = keys[i] % (cells.x * cells.y * cells.z);
        f[0] = static_cast<double>(key % cells.x) / cells.x;
        f[1] = static_cast<double>((key / cells.x) % cells.y) / cells.y;
        f[2] = static_cast<double>(key / (cells.x * cells.y)) / cells.z;
    }
    const double corner[3] = {lo.x + f[0] * h.m[0] + f[1] * h.m[3] + f[2] * h.m[6],
                              lo.y + f[0] * h.m[1] + f[1] * h.m[4] + f[2] * h.m[7],
                              lo.z + f[0] * h.m[2] + f[1] * h.m[5] + f[2] * h.m[8]};
    Real4<Real> l;
    l.x = static_cast<Real>(p.x - corner[0]);
    l.y = static_cast<Real>(p.y - corner[1]);
    l.z = static_cast<Real>(p.z - corner[2]);
    l.w = static_cast<Real>(p.w);
    out[i] = l;
}

// The square of how far each atom moved since the list was built, without
// what moving with the box gives it: (x - lo) - (x0 - lo0) m, m = h0^-1 h.

__global__ void moved(std::int64_t n, const double4a* x, const double4a* x0, double3 lo,
                      double3 lo0, Matrix m, double* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4a a = x[i];
    const double4a b = x0[i];
    const double r0[3] = {b.x - lo0.x, b.y - lo0.y, b.z - lo0.z};
    double d[3] = {a.x - lo.x, a.y - lo.y, a.z - lo.z};
    for (int c = 0; c < 3; ++c) {
        d[c] -= r0[0] * m.m[c] + r0[1] * m.m[3 + c] + r0[2] * m.m[6 + c];
    }
    out[i] = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
}

// The place of (j, i) in the list of j for each pair (i, j): the code of the
// reverse pair is 26 minus that of the pair, the cell around and the image
// of the box both reversed; -1 where it is missing, which a list built from
// the same distances both ways never has.
__global__ void find_reverse(std::int64_t n, const std::int32_t* indices,
                             const std::int32_t* counts, std::int32_t* reverse, int* missing) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const auto packed = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
        const std::int64_t j = packed & kIndexMask;
        const std::uint32_t wanted =
            static_cast<std::uint32_t>(i) | ((26u - (packed >> kIndexBits)) << kIndexBits);
        int found = -1;
        const int others = counts[j];
        for (int m = 0; m < others; ++m) {
            if (static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(m) * n + j]) == wanted) {
                found = m;
                break;
            }
        }
        reverse[static_cast<std::int64_t>(k) * n + i] = found;
        if (found < 0) atomicExch(missing, 1);
    }
}

}  // namespace

const std::int32_t* Neighbors::Impl::reversed() {
    if (reverse_build != builds) {
        reverse.resize(static_cast<std::size_t>(most) * static_cast<std::size_t>(n));
        found.zero();
        find_reverse<<<blocks(n), kBlock>>>(n, indices.data(), counts.data(), reverse.data(),
                                            found.data());
        MDAPY_LAUNCHED();
        int missing = 0;
        found.download(&missing, 1);
        if (missing) {
            throw std::logic_error("the neighbor list is not symmetric");
        }
        reverse_build = builds;
    }
    return reverse.data();
}

template <class Real>
const Real4<Real>* Neighbors::Impl::local(const Atoms::Impl& atoms, const Box& box) {
    constexpr int k = sizeof(Real) == sizeof(float) ? 0 : 1;
    Buffer<Real4<Real>>* out;
    if constexpr (k == 0) {
        out = &local_single;
    } else {
        out = &local_double;
    }
    if (local_version[k] != atoms.version || std::memcmp(&local_box[k], &box, sizeof(Box)) != 0) {
        out->resize(static_cast<std::size_t>(n));
        Matrix h{};
        std::memcpy(h.m, box.h, sizeof(h.m));
        to_local<Real><<<blocks(n), kBlock>>>(n, atoms.x.data(), keys_sorted.data(), cells,
                                              with_cells ? 1 : 0, h,
                                              make_double3(box.lo[0], box.lo[1], box.lo[2]),
                                              out->data());
        MDAPY_LAUNCHED();
        local_version[k] = atoms.version;
        local_box[k] = box;
    }
    return out->data();
}

template <class Real>
Images<Real> Neighbors::Impl::steps(const Box& box) const {
    Images<Real> out{};
    const int count[3] = {cells.x, cells.y, cells.z};
    for (int r = 0; r < 3; ++r) {
        const int c = with_cells ? count[r] : 1;
        for (int k = 0; k < 3; ++k) out.h[3 * r + k] = static_cast<Real>(box.h[3 * r + k] / c);
    }
    return out;
}

template const float4* Neighbors::Impl::local<float>(const Atoms::Impl&, const Box&);
template const double4a* Neighbors::Impl::local<double>(const Atoms::Impl&, const Box&);
template Images<float> Neighbors::Impl::steps<float>(const Box&) const;
template Images<double> Neighbors::Impl::steps<double>(const Box&) const;

Neighbors::Neighbors(double cutoff, double skin) : impl_(std::make_unique<Impl>()) {
    if (!(cutoff > 0.0) || !(skin >= 0.0)) {
        throw std::invalid_argument("the cutoff must be positive and the skin not negative");
    }
    impl_->cutoff = cutoff;
    impl_->skin = skin;
    impl_->found.resize(1);
}

Neighbors::~Neighbors() = default;

double Neighbors::cutoff() const { return impl_->cutoff; }
double Neighbors::skin() const { return impl_->skin; }
std::int64_t Neighbors::builds() const { return impl_->builds; }
int Neighbors::most() const { return impl_->most; }

namespace {

// The thickness of a periodic box must hold the minimum image of each pair.
void check_thickness(const Box& box, double reach, double thick[3]) {
    thickness(box, thick);
    for (int k = 0; k < 3; ++k) {
        if (box.pbc[k] && !(thick[k] > 2.0 * reach)) {
            throw std::invalid_argument(
                "the box is " + std::to_string(thick[k]) + " A thick along its vector " +
                std::to_string(k + 1) + ": on the GPU it must be thicker than twice the cutoff "
                "and the skin, " + std::to_string(2.0 * reach) +
                " A, along each periodic direction; compute smaller boxes on the processor");
        }
    }
}

}  // namespace

bool Neighbors::update(Atoms& atoms, const Box& box) {
    if (!stale(atoms, box)) {
        return false;
    }
    build(atoms, box);
    return true;
}

bool Neighbors::stale(Atoms& atoms, const Box& box) {
    Impl& l = *impl_;
    Atoms::Impl& a = atoms.impl();
    const double reach = l.cutoff + l.skin;
    double thick[3];
    check_thickness(box, reach, thick);
    const std::int64_t n = a.n;
    bool build = l.builds == 0 || l.n != n;
    if (!build && l.atoms_version == a.version &&
        std::memcmp(&l.box_built, &box, sizeof(Box)) == 0) {
        return false;  // nothing moved
    }
    if (!build) {
        // the atoms moved, and the box changed shape: a pair can come closer
        // by twice the largest displacement and by its length times the strain
        double m[9];
        inverse_times(l.box_built, box, m);
        double strain = 0.0;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                const double e = m[3 * r + c] - (r == c ? 1.0 : 0.0);
                strain += e * e;
            }
        }
        strain = std::sqrt(strain);
        Matrix mm{};
        std::memcpy(mm.m, m, sizeof(m));
        l.displacement.resize(static_cast<std::size_t>(n));
        moved<<<blocks(n), kBlock>>>(
            n, a.x.data(), l.x_built.data(), make_double3(box.lo[0], box.lo[1], box.lo[2]),
            make_double3(l.box_built.lo[0], l.box_built.lo[1], l.box_built.lo[2]), mm,
            l.displacement.data());
        MDAPY_LAUNCHED();
        const double farthest = std::sqrt(largest(l.displacement.data(), n));
        if (!std::isfinite(farthest)) {
            throw NotFinite("the positions are not finite: atoms may have overlapped");
        }
        build = 2.0 * farthest + reach * strain > l.skin;
    }
    return build;
}

void Neighbors::build(Atoms& atoms, const Box& box) {
    Impl& l = *impl_;
    Atoms::Impl& a = atoms.impl();
    const double reach = l.cutoff + l.skin;
    double thick[3];
    check_thickness(box, reach, thick);
    const DeviceBox<double> dbox = device_box<double>(box);
    const std::int64_t n = a.n;
    if (n > static_cast<std::int64_t>(kIndexMask)) {
        throw std::invalid_argument("the GPU holds at most 134217727 atoms");
    }
    wrap<<<blocks(n), kBlock>>>(n, dbox, a.x.data(), a.images.data());
    MDAPY_LAUNCHED();
    ++a.version;
    // cells at least as wide as the reach, at most about 8 per atom
    int cells[3];
    for (int k = 0; k < 3; ++k) {
        cells[k] = std::max(1, static_cast<int>(std::floor(thick[k] / reach)));
    }
    while (static_cast<double>(cells[0]) * cells[1] * cells[2] > 8.0 * n + 64.0) {
        const int k = static_cast<int>(std::max_element(cells, cells + 3) - cells);
        cells[k] = std::max(1, cells[k] / 2);
    }
    const int total = cells[0] * cells[1] * cells[2];
    // the ghosts after the atoms of the process, when it has some; they are
    // in the caller's order at a build under MPI, the atoms just given
    const std::int64_t first_ghost =
        a.owned > 0 && a.owned < n && a.original.size() == 0 ? a.owned : n;
    if (a.owned > 0 && a.owned < n && a.original.size() != 0) {
        throw std::logic_error("under MPI the list is built once, for the atoms just given");
    }
    l.ghosted = first_ghost < n;
    l.total = total;
    const int keyed = l.ghosted ? 2 * total : total;
    Around around{};
    for (int k = 0; k < 3; ++k) {
        if (box.pbc[k]) {
            around.wrap[k] = 1;
            around.low[k] = cells[k] >= 3 ? -1 : 0;
            around.high[k] = cells[k] >= 2 ? 1 : 0;
        } else {
            around.wrap[k] = 0;
            around.low[k] = -1;
            around.high[k] = 1;
        }
    }
    const auto size = static_cast<std::size_t>(n);
    l.keys.resize(size);
    l.keys_sorted.resize(size);
    l.order.resize(size);
    l.order_sorted.resize(size);
    l.counts.resize(size);
    l.cell_begin.resize(static_cast<std::size_t>(keyed));
    l.cell_end.resize(static_cast<std::size_t>(keyed));
    const int3 grid = make_int3(cells[0], cells[1], cells[2]);
    bin<<<blocks(n), kBlock>>>(n, a.x.data(), dbox, grid, first_ghost, l.keys.data(),
                               l.order.data());
    MDAPY_LAUNCHED();
    int bits = 1;
    while ((1 << bits) < keyed) ++bits;
    std::size_t scratch = 0;
    MDAPY_CUDA(cub::DeviceRadixSort::SortPairs(nullptr, scratch, l.keys.data(),
                                               l.keys_sorted.data(), l.order.data(),
                                               l.order_sorted.data(), static_cast<int>(n), 0,
                                               bits));
    if (l.sort_scratch.size() < scratch) l.sort_scratch.resize(scratch);
    MDAPY_CUDA(cub::DeviceRadixSort::SortPairs(l.sort_scratch.data(), scratch, l.keys.data(),
                                               l.keys_sorted.data(), l.order.data(),
                                               l.order_sorted.data(), static_cast<int>(n), 0,
                                               bits));
    // the atoms in the order of their cells, so that neighbors are near in memory
    a.permute(l.order_sorted.data());
    l.cell_begin.zero();
    l.cell_end.zero();
    bounds<<<blocks(n), kBlock>>>(n, l.keys_sorted.data(), l.cell_begin.data(),
                                  l.cell_end.data());
    MDAPY_LAUNCHED();
    if (l.most == 0) {
        const double density = n / volume_of(box);
        l.most = static_cast<int>(std::ceil(1.3 * density * 4.0 / 3.0 * kPi * reach * reach *
                                            reach)) + 8;
    }
    for (;;) {
        l.indices.resize(static_cast<std::size_t>(l.most) * size);
        l.rows.resize(static_cast<std::size_t>(l.most) * size);
        l.found.zero();
        bool shifts = true;
        for (int k = 0; k < 3; ++k) shifts = shifts && (!box.pbc[k] || cells[k] >= 3);
        l.with_cells = shifts;
        l.cells = grid;
        // the rounding of positions in single precision, with room to spare
        double extent = std::fabs(box.lo[0]) + std::fabs(box.lo[1]) + std::fabs(box.lo[2]);
        for (int k = 0; k < 9; ++k) extent += std::fabs(box.h[k]);
        const double margin = 1e-3 + 1e-6 * extent;
        const auto reach2f = static_cast<float>((reach + margin) * (reach + margin));
        const float4* xf = shifts ? a.positions<float>() : nullptr;
        const DeviceBox<float> fbox = device_box<float>(box);
        if (shifts) {
            list<true><<<blocks(n), kBlock>>>(n, a.x.data(), xf, dbox, fbox, grid, around,
                                              l.keys_sorted.data(), l.cell_begin.data(),
                                              l.cell_end.data(), l.ghosted ? 2 : 1, reach * reach, reach2f, l.most,
                                              l.rows.data(), l.counts.data(), l.found.data());
        } else {
            list<false><<<blocks(n), kBlock>>>(n, a.x.data(), xf, dbox, fbox, grid, around,
                                               l.keys_sorted.data(), l.cell_begin.data(),
                                               l.cell_end.data(), l.ghosted ? 2 : 1, reach * reach, reach2f, l.most,
                                               l.rows.data(), l.counts.data(), l.found.data());
        }
        MDAPY_LAUNCHED();
        int found = 0;
        l.found.download(&found, 1);
        if (found <= l.most) {
            if (found > 0) {
                const dim3 tiles(static_cast<unsigned>((n + 31) / 32),
                                 static_cast<unsigned>((found + 31) / 32));
                columns<<<tiles, dim3(32, 8)>>>(n, l.most, found, l.rows.data(),
                                                l.indices.data());
                MDAPY_LAUNCHED();
            }
            break;
        }
        l.most = found + found / 8 + 4;  // more columns, and once more
    }
    l.x_built.resize(size);
    MDAPY_CUDA(cudaMemcpy(l.x_built.data(), a.x.data(), size * sizeof(double4a),
                          cudaMemcpyDeviceToDevice));
    l.box_built = box;
    l.atoms_version = a.version;
    l.n = n;
    ++l.builds;
}

void Neighbors::get(const Atoms& atoms, std::int32_t* counts, std::int32_t* indices,
                    std::int32_t* images) const {
    const Impl& l = *impl_;
    const auto n = static_cast<std::size_t>(l.n);
    const auto most = static_cast<std::size_t>(l.most);
    std::vector<std::int32_t> columns(most * n), placed(n), keys(n);
    l.counts.download(placed.data(), n);
    l.indices.download(columns.data(), columns.size());
    l.keys_sorted.download(keys.data(), n);
    const int count[3] = {l.cells.x, l.cells.y, l.cells.z};
    // the image of the box a pair is reached through, from the cell around
    auto image = [&](std::size_t p, std::uint32_t code) {
        if (!l.with_cells) return static_cast<std::int32_t>(code);
        const int key = keys[p] % l.total;
        const int own[3] = {key % count[0], (key / count[0]) % count[1], key / (count[0] * count[1])};
        const int d[3] = {static_cast<int>(code % 3u) - 1, static_cast<int>((code / 3u) % 3u) - 1,
                          static_cast<int>(code / 9u) - 1};
        int s[3];
        for (int k = 0; k < 3; ++k) {
            const int c = own[k] + d[k];
            s[k] = c < 0 ? -1 : (c >= count[k] ? 1 : 0);
        }
        return static_cast<std::int32_t>((s[0] + 1) + 3 * (s[1] + 1) + 9 * (s[2] + 1));
    };
    const auto& original = atoms.impl().order();  // the caller's atom at each place
    auto caller = [&](std::size_t p) { return original.empty() ? p : static_cast<std::size_t>(original[p]); };
    for (std::size_t p = 0; p < n; ++p) {
        const std::size_t i = caller(p);
        counts[i] = placed[p];
        for (std::size_t k = 0; k < most; ++k) {
            const bool there = static_cast<int>(k) < placed[p];
            const auto packed = static_cast<std::uint32_t>(columns[k * n + p]);
            indices[i * most + k] =
                there ? static_cast<std::int32_t>(caller(packed & kIndexMask)) : -1;
            images[i * most + k] = there ? image(p, packed >> kIndexBits) : -1;
        }
    }
}

}  // namespace mdapy::cuda
