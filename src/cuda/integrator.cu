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

__global__ void kick_kernel(std::int64_t n, double c, const double* f, const double* m,
                            double* v) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double mass = m[i];
    for (int a = 0; a < 3; ++a) {
        v[3 * i + a] += c * f[3 * i + a] / mass;
    }
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

// The vector of each atom in the axes of the system, w = v R^T, its
// components along which the atom may not move set to zero, and back, w R.
__global__ void restrict_kernel(std::int64_t n, Rotation t, const std::int32_t* movable,
                                double* v) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::int32_t m = movable[i];
    if (m == 7) return;
    double* u = v + 3 * i;
    double w[3];
    for (int b = 0; b < 3; ++b) {
        w[b] = ((m >> b) & 1) ? u[0] * t.r[3 * b] + u[1] * t.r[3 * b + 1] + u[2] * t.r[3 * b + 2]
                              : 0.0;
    }
    for (int c = 0; c < 3; ++c) {
        u[c] = w[0] * t.r[c] + w[1] * t.r[3 + c] + w[2] * t.r[6 + c];
    }
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
    kick_kernel<<<blocks(a.n), kBlock>>>(a.n, c, a.forces.data(), a.masses.data(),
                                         a.velocities.data());
    MDAPY_LAUNCHED();
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

void restricted(Atoms::Impl& a, double* vectors) {
    if (a.movable.size() != static_cast<std::size_t>(a.n)) return;  // every atom moves
    Rotation t{};
    for (int k = 0; k < 9; ++k) t.r[k] = a.rotation[k];
    restrict_kernel<<<blocks(a.n), kBlock>>>(a.n, t, a.movable.data(), vectors);
    MDAPY_LAUNCHED();
}

}  // namespace

void Atoms::restrict_forces() { restricted(*impl_, impl_->forces.data()); }

void Atoms::restrict_velocities() { restricted(*impl_, impl_->velocities.data()); }

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
