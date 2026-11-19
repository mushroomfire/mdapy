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

__global__ void reflect_kernel(std::int64_t n, double3 normal, double offset, double speed,
                               double4a* x, double* v) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double4a p = x[i];
    const double depth = p.x * normal.x + p.y * normal.y + p.z * normal.z - offset;
    if (!(depth < 0.0)) return;
    p.x -= 2.0 * depth * normal.x;
    p.y -= 2.0 * depth * normal.y;
    p.z -= 2.0 * depth * normal.z;
    x[i] = p;
    double* u = v + 3 * i;
    const double change = 2.0 * (speed - (u[0] * normal.x + u[1] * normal.y + u[2] * normal.z));
    u[0] += change * normal.x;
    u[1] += change * normal.y;
    u[2] += change * normal.z;
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

// The noise of the Langevin dynamics that keeps the momentum: dv = c2
// sigma g of each atom, the same numbers as langevin_kernel, and m dv in
// three columns for their sum.
__global__ void noise_kernel(std::int64_t n, double c2, double kt, unsigned long long seed,
                             unsigned long long counter, const double* m, double* dv,
                             double* momenta) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(i), 4ULL * counter, &state);
    const double2 g01 = curand_normal2_double(&state);
    const double2 g23 = curand_normal2_double(&state);
    const double sigma = c2 * sqrt(kt / m[i]);
    const double g[3] = {g01.x, g01.y, g23.x};
    for (int c = 0; c < 3; ++c) {
        dv[3 * i + c] = sigma * g[c];
        momenta[c * n + i] = m[i] * sigma * g[c];
    }
}

// v = c1 v + dv less an equal share of the momentum of the noise.
__global__ void keep_kernel(std::int64_t n, double c1, double3 share, const double* m,
                            const double* dv, double* v) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double inverse = 1.0 / m[i];
    v[3 * i] = c1 * v[3 * i] + dv[3 * i] - share.x * inverse;
    v[3 * i + 1] = c1 * v[3 * i + 1] + dv[3 * i + 1] - share.y * inverse;
    v[3 * i + 2] = c1 * v[3 * i + 2] + dv[3 * i + 2] - share.z * inverse;
}

// f = (1 - w) kept + w f, or (1 - w) f with no kept forces.
__global__ void mix_kernel(std::int64_t components, double w, const double* kept, double* f) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= components) return;
    f[k] = kept != nullptr ? (1.0 - w) * kept[k] + w * f[k] : (1.0 - w) * f[k];
}

struct Cell {
    double h[9];
};

// The forces and energies of springs to sites, the positions unwrapped by
// their images.
__global__ void springs_kernel(std::int64_t n, Cell box, const double4a* x, const int3* images,
                               const double* sites, const double* k, double* f, double* e) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4a p = x[i];
    const int3 m = images != nullptr ? images[i] : make_int3(0, 0, 0);
    const double u[3] = {p.x, p.y, p.z};
    double squared = 0.0;
    for (int c = 0; c < 3; ++c) {
        const double d = u[c] + m.x * box.h[c] + m.y * box.h[3 + c] + m.z * box.h[6 + c] -
                         sites[3 * i + c];
        f[3 * i + c] = -k[i] * d;
        squared += d * d;
    }
    e[i] = 0.5 * k[i] * squared;
}

// The push of the field through the virial of each atom, in three columns:
// push_b = sum_a field_a W_ab, W in 9 columns xx xy xz yx yy yz zx zy zz.
__global__ void push_kernel(std::int64_t n, double3 field, const double* w, double* push) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double e[3] = {field.x, field.y, field.z};
    for (int b = 0; b < 3; ++b) {
        double sum = 0.0;
        for (int a = 0; a < 3; ++a) sum += e[a] * w[(3 * a + b) * n + i];
        push[b * n + i] = sum;
    }
}

// f += push less its mean.
__global__ void pushed_kernel(std::int64_t n, double3 mean, const double* push, double* f) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f[3 * i] += push[i] - mean.x;
    f[3 * i + 1] += push[n + i] - mean.y;
    f[3 * i + 2] += push[2 * n + i] - mean.z;
}

// W_i v_i of each atom in three columns.
__global__ void flux_kernel(std::int64_t n, const double* w, const double* v, double* out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int a = 0; a < 3; ++a) {
        double sum = 0.0;
        for (int b = 0; b < 3; ++b) sum += w[(3 * a + b) * n + i] * v[3 * i + b];
        out[a * n + i] = sum;
    }
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

void Atoms::reflect(const double* normal, double offset, double speed) {
    Impl& a = *impl_;
    reflect_kernel<<<blocks(a.n), kBlock>>>(a.n, make_double3(normal[0], normal[1], normal[2]),
                                            offset, speed, a.x.data(), a.velocities.data());
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
                     std::uint64_t counter, bool keep) {
    Impl& a = *impl_;
    if (!keep) {
        langevin_kernel<<<blocks(a.n), kBlock>>>(a.n, c1, c2, kt, seed, counter,
                                                 a.masses.data(), a.velocities.data());
        MDAPY_LAUNCHED();
        return;
    }
    a.scratch.resize(static_cast<std::size_t>(6 * a.n));
    a.stash.resize(static_cast<std::size_t>(3 * a.n));  // the noise, until it is added
    noise_kernel<<<blocks(a.n), kBlock>>>(a.n, c2, kt, seed, counter, a.masses.data(),
                                          a.stash.data(), a.scratch.data());
    MDAPY_LAUNCHED();
    double total[3];
    sum_columns(a.scratch.data(), 3, a.n, total);
    const double inverse = 1.0 / static_cast<double>(a.n);
    keep_kernel<<<blocks(a.n), kBlock>>>(
        a.n, c1, make_double3(total[0] * inverse, total[1] * inverse, total[2] * inverse),
        a.masses.data(), a.stash.data(), a.velocities.data());
    MDAPY_LAUNCHED();
}

void Atoms::stash_forces() {
    Impl& a = *impl_;
    a.stash.resize(static_cast<std::size_t>(3 * a.n));
    MDAPY_CUDA(cudaMemcpy(a.stash.data(), a.forces.data(),
                          static_cast<std::size_t>(3 * a.n) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
}

void Atoms::mix_forces(double weight, bool other) {
    Impl& a = *impl_;
    mix_kernel<<<blocks(3 * a.n), kBlock>>>(3 * a.n, weight, other ? a.stash.data() : nullptr,
                                            a.forces.data());
    MDAPY_LAUNCHED();
}

void Atoms::set_springs(const double* sites, const double* constants) {
    Impl& a = *impl_;
    if (sites == nullptr) {
        a.sites.resize(0);
        a.spring_constants.resize(0);
        return;
    }
    std::vector<double> placed(static_cast<std::size_t>(3 * a.n));
    a.to_places(sites, placed.data(), 3);
    a.sites.upload(placed.data(), placed.size());
    std::vector<double> k(static_cast<std::size_t>(a.n));
    a.to_places(constants, k.data(), 1);
    a.spring_constants.upload(k.data(), k.size());
}

double Atoms::springs(const double* h) {
    Impl& a = *impl_;
    if (a.sites.size() != static_cast<std::size_t>(3 * a.n)) {
        throw std::logic_error("the atoms have no springs");
    }
    Cell box;
    for (int k = 0; k < 9; ++k) box.h[k] = h[k];
    a.forces.resize(static_cast<std::size_t>(3 * a.n));
    a.energies.resize(static_cast<std::size_t>(a.n));
    const bool wrapped = a.images.size() == static_cast<std::size_t>(a.n);
    springs_kernel<<<blocks(a.n), kBlock>>>(a.n, box, a.x.data(),
                                            wrapped ? a.images.data() : nullptr,
                                            a.sites.data(), a.spring_constants.data(),
                                            a.forces.data(), a.energies.data());
    MDAPY_LAUNCHED();
    a.has_energies = true;
    double out = 0.0;
    sum_columns(a.energies.data(), 1, a.n, &out);
    return out;
}

void Atoms::drive(const double* field) {
    Impl& a = *impl_;
    if (!a.has_virials || a.virials.size() != static_cast<std::size_t>(9 * a.n)) {
        throw std::logic_error("the field needs the virials of each atom");
    }
    a.scratch.resize(static_cast<std::size_t>(6 * a.n));
    push_kernel<<<blocks(a.n), kBlock>>>(a.n, make_double3(field[0], field[1], field[2]),
                                         a.virials.data(), a.scratch.data());
    MDAPY_LAUNCHED();
    double total[3];
    sum_columns(a.scratch.data(), 3, a.n, total);
    const double inverse = 1.0 / static_cast<double>(a.n);
    pushed_kernel<<<blocks(a.n), kBlock>>>(
        a.n, make_double3(total[0] * inverse, total[1] * inverse, total[2] * inverse),
        a.scratch.data(), a.forces.data());
    MDAPY_LAUNCHED();
}

void Atoms::heat_flux(double* flux) const {
    Impl& a = *impl_;
    if (!a.has_virials || a.virials.size() != static_cast<std::size_t>(9 * a.n)) {
        throw std::logic_error("the heat flux needs the virials of each atom");
    }
    a.scratch.resize(static_cast<std::size_t>(6 * a.n));
    flux_kernel<<<blocks(a.n), kBlock>>>(a.n, a.virials.data(), a.velocities.data(),
                                         a.scratch.data());
    MDAPY_LAUNCHED();
    sum_columns(a.scratch.data(), 3, a.n, flux);
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
