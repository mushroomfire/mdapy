// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Electronic stopping on the device: the arithmetic of _HostAtoms in
// simulation/_integrator.py, as LAMMPS's fix electron/stopping.

#include <stdexcept>

#include "cuda/atoms.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

struct StoppingTable {
    const double* energies;  // (count,)
    const double* values;    // (species, count)
    int count;
    double least;            // the least kinetic energy
    int needed;              // the least number of neighbors
    double cutsq;            // within which they count
    int bit;                 // the group of the atoms that may feel it, -1 for all
    double mvv2e;
};

__global__ void stopping_kernel(std::int64_t n, StoppingTable t, const double4a* x,
                                Images<double> images, const std::int32_t* indices,
                                const std::int32_t* counts, const double* masses,
                                const double* v, const std::int32_t* groups, double* f,
                                double* out) {
    __shared__ double steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 0.0;
    out[n + i] = 0.0;
    if (t.bit >= 0 && ((groups[i] >> t.bit) & 1) == 0) return;
    const double* u = v + 3 * i;
    const double v2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const double e = 0.5 * t.mvv2e * masses[i] * v2;
    if (!(e >= t.least && e >= t.energies[0])) return;
    const double4a xi = x[i];
    if (t.needed > 0) {
        int close = 0;
        const int count = counts[i];
        for (int k = 0; k < count; ++k) {
            const auto packed =
                static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
            const double4a xj = x[packed & kIndexMask];
            double d[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
            add_step(steps, packed >> kIndexBits, d[0], d[1], d[2]);
            if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < t.cutsq) ++close;
        }
        if (close < t.needed) return;
    }
    if (e > t.energies[t.count - 1]) {
        out[n + i] = 1.0;
    }
    // the first energy of the table not below e, as the bisection of LAMMPS
    int low = 0, high = t.count;
    while (low < high) {
        const int middle = (low + high) / 2;
        if (t.energies[middle] < e) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    const int up = min(max(low, 1), t.count - 1);
    const int down = up - 1;
    const int s = static_cast<int>(xi.w);
    const double lo = t.values[s * t.count + down];
    const double hi = t.values[s * t.count + up];
    const double drag =
        (hi - lo) / (t.energies[up] - t.energies[down]) * (e - t.energies[down]) + lo;
    const double speed = sqrt(v2);
    const double factor = drag / speed;
    for (int c = 0; c < 3; ++c) f[3 * i + c] -= factor * u[c];
    out[i] = drag * speed;
}

}  // namespace

struct Stopping::Impl {
    Buffer<double> energies;
    Buffer<double> values;
    int count = 0;
    int species = 0;
};

Stopping::Stopping(const double* energies, int count, const double* values, int species)
    : impl_(std::make_unique<Impl>()) {
    if (count < 2 || species < 1) {
        throw std::invalid_argument("a stopping table needs two energies or more");
    }
    impl_->energies.upload(energies, static_cast<std::size_t>(count));
    impl_->values.upload(values, static_cast<std::size_t>(count) * species);
    impl_->count = count;
    impl_->species = species;
}

Stopping::~Stopping() = default;

void Stopping::apply(Atoms& atoms, Neighbors& neighbors, const Box& box, double least,
                     int needed, double cutoff, int bit, double mvv2e, double* out) const {
    Atoms::Impl& a = atoms.impl();
    Neighbors::Impl& l = neighbors.impl();
    if (bit >= 0 && a.groups.size() != static_cast<std::size_t>(a.n)) {
        throw std::invalid_argument("the group of the stopping is not set");
    }
    const Impl& s = *impl_;
    const StoppingTable t{s.energies.data(), s.values.data(), s.count, least, needed,
                          cutoff * cutoff, bit, mvv2e};
    const double4a* x = l.local<double>(a, box);
    const Images<double> images = l.steps<double>(box);
    a.scratch.resize(static_cast<std::size_t>(2 * a.n));
    stopping_kernel<<<blocks(a.n), kBlock>>>(
        a.n, t, x, images, l.indices.data(), l.counts.data(), a.masses.data(),
        a.velocities.data(), bit >= 0 ? a.groups.data() : nullptr, a.forces.data(),
        a.scratch.data());
    MDAPY_LAUNCHED();
    sum_columns(a.scratch.data(), 2, a.n, out);
}

}  // namespace mdapy::cuda
