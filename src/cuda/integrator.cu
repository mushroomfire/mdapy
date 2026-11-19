// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The steps of MDAPY's molecular dynamics on every atom, on the device: the
// same arithmetic as simulation/_integrator.py does with NumPy, which keeps
// the variables of the thermostats and the barostat and calls these.

#include <curand_kernel.h>

#include <stdexcept>
#include <vector>

#include "cuda/atoms.cuh"

namespace mdapy::cuda {

namespace {

// One thread for each component, so that the reads and writes are coalesced.
__global__ void kick_kernel(std::int64_t components, double c, const double* f, const double* m,
                            double* v) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= components) return;
    v[k] += c * f[k] / m[k / 3];
}

// kick then drift, as one pass: the same arithmetic.
__global__ void kick_drift_kernel(std::int64_t n, double c, double dt, const double* f,
                                  const double* m, double* v, double4a* x) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double mass = m[i];
    double u[3];
    for (int a = 0; a < 3; ++a) {
        u[a] = v[3 * i + a] + c * f[3 * i + a] / mass;
        v[3 * i + a] = u[a];
    }
    double4a p = x[i];
    p.x += dt * u[0];
    p.y += dt * u[1];
    p.z += dt * u[2];
    x[i] = p;
}

__global__ void drift_kernel(std::int64_t n, double dt, const double* v, double4a* x) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double4a p = x[i];
    p.x += dt * v[3 * i];
    p.y += dt * v[3 * i + 1];
    p.z += dt * v[3 * i + 2];
    x[i] = p;
}

__global__ void scale_kernel(std::int64_t n, double3 factor, double* v) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[3 * i] *= factor.x;
    v[3 * i + 1] *= factor.y;
    v[3 * i + 2] *= factor.z;
}

__global__ void shear_kernel(std::int64_t n, double half, double yz, double xz, double xy,
                             double* v) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double* u = v + 3 * i;
    u[0] -= half * (u[1] * xy + u[2] * xz);
    u[1] -= half * u[2] * yz;
}

struct Affine {
    double old_lo[3], inverse[9], lo[3], h[9];
};

__global__ void remap_kernel(std::int64_t n, Affine t, double4a* x) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double4a p = x[i];
    const double r[3] = {p.x - t.old_lo[0], p.y - t.old_lo[1], p.z - t.old_lo[2]};
    double s[3];
    for (int k = 0; k < 3; ++k) {
        s[k] = r[0] * t.inverse[k] + r[1] * t.inverse[3 + k] + r[2] * t.inverse[6 + k];
    }
    p.x = t.lo[0] + (s[0] * t.h[0] + s[1] * t.h[3] + s[2] * t.h[6]);
    p.y = t.lo[1] + (s[0] * t.h[1] + s[1] * t.h[4] + s[2] * t.h[7]);
    p.z = t.lo[2] + (s[0] * t.h[2] + s[1] * t.h[5] + s[2] * t.h[8]);
    x[i] = p;
}

// Gaussian numbers of a counter-based generator: atom i, call counter.
__global__ void langevin_kernel(std::int64_t n, double c1, double c2, double kt,
                                unsigned long long seed, unsigned long long counter,
                                const double* m, double* v) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(i), 4ULL * counter, &state);
    const double2 g01 = curand_normal2_double(&state);
    const double2 g23 = curand_normal2_double(&state);
    const double sigma = sqrt(kt / m[i]);
    v[3 * i] = c1 * v[3 * i] + c2 * sigma * g01.x;
    v[3 * i + 1] = c1 * v[3 * i + 1] + c2 * sigma * g01.y;
    v[3 * i + 2] = c1 * v[3 * i + 2] + c2 * sigma * g23.x;
}

struct Rotation {
    double r[9];
};

// A vector in the axes of the system, w = u R^T, its components along
// which the atom may not move set to zero, and back, w R.
__device__ inline void project(const Rotation& t, std::int32_t m, double* u) {
    double w[3];
    for (int b = 0; b < 3; ++b) {
        w[b] = ((m >> b) & 1) ? u[0] * t.r[3 * b] + u[1] * t.r[3 * b + 1] + u[2] * t.r[3 * b + 2]
                              : 0.0;
    }
    for (int c = 0; c < 3; ++c) {
        u[c] = w[0] * t.r[c] + w[1] * t.r[3 + c] + w[2] * t.r[6 + c];
    }
}

// The vector of each atom projected, and set, the velocities prescribed
// added (null for forces).
__global__ void restrict_kernel(std::int64_t n, Rotation t, const std::int32_t* movable,
                                const double* prescribed, double* v) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::int32_t m = movable[i];
    if (m == 7) return;
    double* u = v + 3 * i;
    project(t, m, u);
    if (prescribed != nullptr) {
        for (int c = 0; c < 3; ++c) u[c] += prescribed[3 * i + c];
    }
}

__global__ void add_kernel(std::int64_t components, const double* added, double* f) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < components) f[k] += added[k];
}

// The forces of the atoms of group bit, zero for the others, in 3 columns.
__global__ void group_force_kernel(std::int64_t n, int bit, const std::int32_t* groups,
                                   const double* f, double* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bool in = ((groups[i] >> bit) & 1) != 0;
    for (int c = 0; c < 3; ++c) out[c * n + i] = in ? f[3 * i + c] : 0.0;
}

// m |v|^2 of each atom over the components that move freely.
__global__ void free_kinetic_kernel(std::int64_t n, Rotation t, const std::int32_t* movable,
                                    const double* m, const double* v, double* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double u[3] = {v[3 * i], v[3 * i + 1], v[3 * i + 2]};
    if (movable != nullptr && movable[i] != 7) project(t, movable[i], u);
    out[i] = m[i] * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

// A Langevin bath on the atoms of group bit: the change of m |v|^2 / 2 of
// each atom in out, zero for the others.
__global__ void bath_kernel(std::int64_t n, int bit, const std::int32_t* groups, double c1,
                            double c2, double kt, unsigned long long seed,
                            unsigned long long counter, Rotation t, const std::int32_t* movable,
                            const double* m, double* v, double* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 0.0;
    if (((groups[i] >> bit) & 1) == 0) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(i), 4ULL * counter, &state);
    const double2 g01 = curand_normal2_double(&state);
    const double2 g23 = curand_normal2_double(&state);
    double g[3] = {g01.x, g01.y, g23.x};
    if (movable != nullptr && movable[i] != 7) project(t, movable[i], g);
    const double mass = m[i], sigma = sqrt(kt / mass);
    double* u = v + 3 * i;
    double before = 0.0, after = 0.0;
    for (int c = 0; c < 3; ++c) {
        before += u[c] * u[c];
        u[c] = c1 * u[c] + c2 * sigma * g[c];
        after += u[c] * u[c];
    }
    out[i] = 0.5 * mass * (after - before);
}

// m v_a v_b of each atom in 6 columns, xx yy zz yz xz xy.
__global__ void kinetic_kernel(std::int64_t n, const double* m, const double* v, double* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double* u = v + 3 * i;
    const double mass = m[i];
    out[i] = mass * u[0] * u[0];
    out[n + i] = mass * u[1] * u[1];
    out[2 * n + i] = mass * u[2] * u[2];
    out[3 * n + i] = mass * u[1] * u[2];
    out[4 * n + i] = mass * u[0] * u[2];
    out[5 * n + i] = mass * u[0] * u[1];
}

}  // namespace

void Atoms::set_masses(const double* masses) {
    std::vector<double> placed(static_cast<std::size_t>(impl_->n));
    impl_->to_places(masses, placed.data(), 1);
    impl_->masses.upload(placed.data(), placed.size());
}

void Atoms::set_velocities(const double* velocities) {
    std::vector<double> placed(static_cast<std::size_t>(3 * impl_->n));
    impl_->to_places(velocities, placed.data(), 3);
    impl_->velocities.upload(placed.data(), placed.size());
}

void Atoms::get_velocities(double* velocities) const {
    if (impl_->velocities.size() == 0) {
        throw std::logic_error("the atoms have no velocities");
    }
    std::vector<double> placed(static_cast<std::size_t>(3 * impl_->n));
    impl_->velocities.download(placed.data(), placed.size());
    impl_->to_caller(placed.data(), velocities, 3);
}

void Atoms::kick(double c) {
    Impl& a = *impl_;
    kick_kernel<<<blocks(3 * a.n), kBlock>>>(3 * a.n, c, a.forces.data(), a.masses.data(),
                                             a.velocities.data());
    MDAPY_LAUNCHED();
}

void Atoms::kick_drift(double c, double dt) {
    Impl& a = *impl_;
    kick_drift_kernel<<<blocks(a.n), kBlock>>>(a.n, c, dt, a.forces.data(), a.masses.data(),
                                               a.velocities.data(), a.x.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

void Atoms::drift(double dt) {
    Impl& a = *impl_;
    drift_kernel<<<blocks(a.n), kBlock>>>(a.n, dt, a.velocities.data(), a.x.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

void Atoms::scale(double fx, double fy, double fz) {
    Impl& a = *impl_;
    scale_kernel<<<blocks(a.n), kBlock>>>(a.n, make_double3(fx, fy, fz), a.velocities.data());
    MDAPY_LAUNCHED();
}

void Atoms::shear(double half, double yz, double xz, double xy) {
    Impl& a = *impl_;
    shear_kernel<<<blocks(a.n), kBlock>>>(a.n, half, yz, xz, xy, a.velocities.data());
    MDAPY_LAUNCHED();
}

void Atoms::remap(const double* old_lo, const double* old_inverse, const double* lo,
                  const double* h) {
    Impl& a = *impl_;
    Affine t{};
    for (int k = 0; k < 3; ++k) {
        t.old_lo[k] = old_lo[k];
        t.lo[k] = lo[k];
    }
    for (int k = 0; k < 9; ++k) {
        t.inverse[k] = old_inverse[k];
        t.h[k] = h[k];
    }
    remap_kernel<<<blocks(a.n), kBlock>>>(a.n, t, a.x.data());
    MDAPY_LAUNCHED();
    ++a.version;
}

void Atoms::langevin(double c1, double c2, double kt, std::uint64_t seed,
                     std::uint64_t counter) {
    Impl& a = *impl_;
    langevin_kernel<<<blocks(a.n), kBlock>>>(a.n, c1, c2, kt, seed, counter, a.masses.data(),
                                             a.velocities.data());
    MDAPY_LAUNCHED();
}

void Atoms::set_movable(const bool* movable, const double* rotation) {
    Impl& a = *impl_;
    std::vector<std::int32_t> bits(static_cast<std::size_t>(a.n));
    for (std::int64_t i = 0; i < a.n; ++i) {
        bits[i] = (movable[3 * i] ? 1 : 0) | (movable[3 * i + 1] ? 2 : 0) |
                  (movable[3 * i + 2] ? 4 : 0);
    }
    std::vector<std::int32_t> placed(bits.size());
    a.to_places(bits.data(), placed.data(), 1);
    a.movable.upload(placed.data(), placed.size());
    for (int k = 0; k < 9; ++k) a.rotation[k] = rotation[k];
}

namespace {

bool moves_freely(const Atoms::Impl& a) {
    return a.movable.size() != static_cast<std::size_t>(a.n);
}

Rotation rotation_of(const Atoms::Impl& a) {
    Rotation t{};
    for (int k = 0; k < 9; ++k) t.r[k] = a.rotation[k];
    return t;
}

void restricted(Atoms::Impl& a, double* vectors, const double* prescribed) {
    if (moves_freely(a)) return;  // every atom moves
    restrict_kernel<<<blocks(a.n), kBlock>>>(a.n, rotation_of(a), a.movable.data(), prescribed,
                                             vectors);
    MDAPY_LAUNCHED();
}

// Three doubles of each atom from the caller, at the places of the atoms.
void placed_triples(Atoms::Impl& a, const double* values, Buffer<double>& out) {
    if (values == nullptr) {
        out.resize(0);
        return;
    }
    std::vector<double> placed(static_cast<std::size_t>(3 * a.n));
    a.to_places(values, placed.data(), 3);
    out.upload(placed.data(), placed.size());
}

}  // namespace

void Atoms::clear_movable() { impl_->movable.resize(0); }

void Atoms::restrict_forces() { restricted(*impl_, impl_->forces.data(), nullptr); }

void Atoms::restrict_velocities() {
    Impl& a = *impl_;
    const bool set = a.prescribed.size() == static_cast<std::size_t>(3 * a.n);
    restricted(a, a.velocities.data(), set ? a.prescribed.data() : nullptr);
}

void Atoms::set_prescribed(const double* velocities) {
    placed_triples(*impl_, velocities, impl_->prescribed);
}

void Atoms::set_added(const double* forces) { placed_triples(*impl_, forces, impl_->added); }

void Atoms::add_forces() {
    Impl& a = *impl_;
    if (a.added.size() != static_cast<std::size_t>(3 * a.n)) return;
    add_kernel<<<blocks(3 * a.n), kBlock>>>(3 * a.n, a.added.data(), a.forces.data());
    MDAPY_LAUNCHED();
}

void Atoms::set_groups(const std::int32_t* bits) {
    Impl& a = *impl_;
    if (bits == nullptr) {
        a.groups.resize(0);
        return;
    }
    std::vector<std::int32_t> placed(static_cast<std::size_t>(a.n));
    a.to_places(bits, placed.data(), 1);
    a.groups.upload(placed.data(), placed.size());
}

void Atoms::group_forces(int count, double* sums) const {
    Impl& a = *impl_;
    if (a.groups.size() != static_cast<std::size_t>(a.n)) {
        throw std::logic_error("the atoms have no groups");
    }
    a.scratch.resize(static_cast<std::size_t>(6 * a.n));
    for (int g = 0; g < count; ++g) {
        group_force_kernel<<<blocks(a.n), kBlock>>>(a.n, g, a.groups.data(), a.forces.data(),
                                                    a.scratch.data());
        MDAPY_LAUNCHED();
        sum_columns(a.scratch.data(), 3, a.n, sums + 3 * g);
    }
}

double Atoms::kinetic_free() const {
    Impl& a = *impl_;
    a.scratch.resize(static_cast<std::size_t>(6 * a.n));
    free_kinetic_kernel<<<blocks(a.n), kBlock>>>(
        a.n, rotation_of(a), moves_freely(a) ? nullptr : a.movable.data(), a.masses.data(),
        a.velocities.data(), a.scratch.data());
    MDAPY_LAUNCHED();
    double out = 0.0;
    sum_columns(a.scratch.data(), 1, a.n, &out);
    return out;
}

double Atoms::bath(int bit, double c1, double c2, double kt, std::uint64_t seed,
                   std::uint64_t counter) {
    Impl& a = *impl_;
    if (a.groups.size() != static_cast<std::size_t>(a.n)) {
        throw std::logic_error("the atoms have no groups");
    }
    a.scratch.resize(static_cast<std::size_t>(6 * a.n));
    bath_kernel<<<blocks(a.n), kBlock>>>(a.n, bit, a.groups.data(), c1, c2, kt, seed, counter,
                                         rotation_of(a), moves_freely(a) ? nullptr : a.movable.data(),
                                         a.masses.data(), a.velocities.data(), a.scratch.data());
    MDAPY_LAUNCHED();
    double out = 0.0;
    sum_columns(a.scratch.data(), 1, a.n, &out);
    return out;
}

void Atoms::kinetic(double* tensor) const {
    Impl& a = *impl_;
    a.scratch.resize(static_cast<std::size_t>(6 * a.n));
    kinetic_kernel<<<blocks(a.n), kBlock>>>(a.n, a.masses.data(), a.velocities.data(),
                                            a.scratch.data());
    MDAPY_LAUNCHED();
    double s[6];
    sum_columns(a.scratch.data(), 6, a.n, s);
    const int at[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            tensor[3 * r + c] = s[at[r][c]];
        }
    }
}

}  // namespace mdapy::cuda
