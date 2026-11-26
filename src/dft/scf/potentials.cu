// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/potentials.h"

#include "dft/core/sums.cuh"

#include <algorithm>

#include <cmath>
#include <stdexcept>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;
constexpr double kPi = 3.14159265358979323846264338327950288;

unsigned blocks(std::size_t work) {
    return static_cast<unsigned>((work + kThreads - 1) / kThreads);
}

void launched(const char* what) { check_cuda(cudaGetLastError(), what); }

__global__ void to_complex(std::size_t n, const double* in, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = make_cuDoubleComplex(in[i], 0.0);
}

__global__ void real_part(std::size_t n, const Complex* in, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = in[i].x;
}

// i G_j c(G) of count functions on the sphere
__global__ void times_ig(std::size_t n, int j, const double* g, const Complex* in, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const double gj = g[3 * i + j];
    out[i] = make_cuDoubleComplex(-gj * in[i].y, gj * in[i].x);
}

// the sum over j of i G_j h_j(G), the components of the three h_j one after the other
__global__ void divergence_kernel(std::size_t n, const double* g, const Complex* h, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    double re = 0.0, im = 0.0;
    for (int j = 0; j < 3; ++j) {
        const Complex c = h[static_cast<std::size_t>(j) * n + i];
        re -= g[3 * i + j] * c.y;
        im += g[3 * i + j] * c.x;
    }
    out[i] = make_cuDoubleComplex(re, im);
}

// V_H(G) = 4 pi rho(G) / G^2, zero at G = 0, and the sum of 4 pi |rho(G)|^2 / G^2
// of each block (energy[block], added by ordered_sums)
__global__ void hartree_kernel(std::size_t n, const double* g, const Complex* rho, Complex* v,
                               double* energy) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    double e = 0.0;
    if (i < n) {
        const double g2 = g[3 * i] * g[3 * i] + g[3 * i + 1] * g[3 * i + 1] + g[3 * i + 2] * g[3 * i + 2];
        if (g2 > 1e-12) {
            const double f = 4.0 * kPi / g2;
            v[i] = make_cuDoubleComplex(f * rho[i].x, f * rho[i].y);
            e = f * (rho[i].x * rho[i].x + rho[i].y * rho[i].y);
        } else {
            v[i] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    // the sum of the block
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = e;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) energy[blockIdx.x] = partial[0];
}

// the sum of a[i] b[i], or of a[i] when b is null, of each block (sum[block])
__global__ void dot_kernel(std::size_t n, const double* a, const double* b, double* sum) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    double value = 0.0;
    if (i < n) value = b == nullptr ? a[i] : a[i] * b[i];
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sum[blockIdx.x] = partial[0];
}

// the spin densities with half the core each, interleaved as the functionals take them
__global__ void spin_densities(std::size_t points, int spins, const double* rho, const double* core,
                               double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points) return;
    const double c = core == nullptr ? 0.0 : 0.5 * core[i];
    if (spins == 1) {
        out[2 * i] = out[2 * i + 1] = 0.5 * rho[i] + c;
    } else {
        out[2 * i] = rho[i] + c;
        out[2 * i + 1] = rho[points + i] + c;
    }
}

// out = a + b, count components
__global__ void add_components(std::size_t count, const Complex* a, const Complex* b, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < count) out[i] = make_cuDoubleComplex(a[i].x + b[i].x, a[i].y + b[i].y);
}

// the spin densities interleaved (2 per point) apart, (spin, point)
__global__ void apart_kernel(std::size_t points, const double* interleaved, double* apart) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points) return;
    apart[i] = interleaved[2 * i];
    apart[points + i] = interleaved[2 * i + 1];
}

// sigma of the gradients, grads (spin, component, point)
__global__ void sigma_kernel(std::size_t points, const double* grads, double* sigma) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points) return;
    double uu = 0.0, ud = 0.0, dd = 0.0;
    for (int j = 0; j < 3; ++j) {
        const double a = grads[static_cast<std::size_t>(j) * points + i];
        const double b = grads[static_cast<std::size_t>(3 + j) * points + i];
        uu += a * a;
        ud += a * b;
        dd += b * b;
    }
    sigma[3 * i] = uu;
    sigma[3 * i + 1] = ud;
    sigma[3 * i + 2] = dd;
}

// h_s = 2 vs_ss grad n_s + vs_ud grad n_other, (component, point) of spin s
__global__ void flux_kernel(std::size_t points, int spin, const double* grads, const double* vs,
                            double* h) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points) return;
    const double same = vs[3 * i + (spin == 0 ? 0 : 2)], cross = vs[3 * i + 1];
    const int other = 1 - spin;
    for (int j = 0; j < 3; ++j)
        h[static_cast<std::size_t>(j) * points + i] =
            2.0 * same * grads[static_cast<std::size_t>(3 * spin + j) * points + i] +
            cross * grads[static_cast<std::size_t>(3 * other + j) * points + i];
}

// hxc_s = V_H + vrho_s - div h_s (div null for LDA)
__global__ void combine_kernel(std::size_t points, int spin, const double* hartree, const double* v,
                               const double* divergence, double* hxc) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points) return;
    hxc[i] = hartree[i] + v[2 * i + spin] - (divergence == nullptr ? 0.0 : divergence[i]);
}

// sums (6 a block, sums[6 block + d]) = sum over the points of the block of
// sum_s h_s,a grad n_s,b, h_s the flux of GGA, for the pairs xx yy zz yz xz xy
__global__ void gga_stress_kernel(std::size_t points, const double* grads, const double* vs, double* sums) {
    double local[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int pairs[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < points;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        for (int spin = 0; spin < 2; ++spin) {
            const double same = vs[3 * i + (spin == 0 ? 0 : 2)], cross = vs[3 * i + 1];
            const int other = 1 - spin;
            for (int d = 0; d < 6; ++d) {
                const int a = pairs[d][0], b = pairs[d][1];
                const double h = 2.0 * same * grads[static_cast<std::size_t>(3 * spin + a) * points + i] +
                                 cross * grads[static_cast<std::size_t>(3 * other + a) * points + i];
                local[d] += h * grads[static_cast<std::size_t>(3 * spin + b) * points + i];
            }
        }
    }
    __shared__ double partial[6][kThreads];
    for (int d = 0; d < 6; ++d) partial[d][threadIdx.x] = local[d];
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k)
            for (int d = 0; d < 6; ++d) partial[d][threadIdx.x] += partial[d][threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int d = 0; d < 6; ++d) sums[6 * blockIdx.x + d] = partial[d][0];
}

// the sum of the partial sums of count blocks in sum, in their order
double total(DeviceArray<double>& sum, unsigned count) {
    double host = 0.0;
    ordered_sums<1>(static_cast<int>(count), sum.data(), &host);
    return host;
}

}  // namespace

Potentials::Potentials(const Lattice& lattice, const Int3& grid, double cutoff, Functional functional,
                       int spins, double threshold, const Lattice* basis, Kernels kernels)
    : lattice_(lattice),
      grid_(grid),
      functional_(functional),
      spins_(spins),
      threshold_(threshold),
      fft_(grid, kernels) {
    if (spins != 1 && spins != 2) throw std::invalid_argument("one or two spins");
    sphere_ = make_sphere(lattice, basis, Vec3{0.0, 0.0, 0.0}, cutoff, grid, false);
    box_.upload(sphere_.box.data(), sphere_.size());
    std::vector<double> g(3 * sphere_.size());
    for (std::size_t i = 0; i < sphere_.size(); ++i) {
        const Vec3 q = lattice.cartesian(Vec3{0.0, 0.0, 0.0}, sphere_.miller[i]);
        for (int j = 0; j < 3; ++j) g[3 * i + j] = q[static_cast<std::size_t>(j)];
    }
    g_.upload(g.data(), g.size());
}

void Potentials::set_core(const double* core) {
    if (core == nullptr) {
        core_.resize(0);
        return;
    }
    core_.upload(core, fft_.points());
}

void Potentials::to_sphere(const double* values, Complex* components, int count) {
    const std::size_t points = fft_.points(), n = sphere_.size();
    boxes_.reserve(points * static_cast<std::size_t>(count));
    to_complex<<<blocks(points * count), kThreads>>>(points * count, values, boxes_.data());
    launched("to_complex");
    fft_.to_reciprocal(boxes_.data(), count);
    gather(boxes_.data(), points, box_.data(), n, count, 1.0 / static_cast<double>(points), components);
}

void Potentials::from_sphere(const Complex* components, double* values, int count) {
    const std::size_t points = fft_.points(), n = sphere_.size();
    boxes_.reserve(points * static_cast<std::size_t>(count));
    scatter(components, n, box_.data(), nullptr, count, boxes_.data(), points);
    fft_.to_real(boxes_.data(), count);
    real_part<<<blocks(points * count), kThreads>>>(points * count, boxes_.data(), values);
    launched("real_part");
}

void Potentials::restrict(double* values, int count) {
    components_.resize(sphere_.size() * static_cast<std::size_t>(count));
    to_sphere(values, components_.data(), count);
    from_sphere(components_.data(), values, count);
}

PotentialEnergies Potentials::compute(const double* rho, double* hxc) {
    const std::size_t points = fft_.points(), n = sphere_.size();
    const double dv = lattice_.volume / static_cast<double>(points);
    PotentialEnergies out;
    // the partial sums of the blocks of a kernel, one more for their sum
    DeviceArray<double>& sum = sum_;
    sum.reserve(std::max(blocks(n), blocks(points)) + 1);
    // the total density on the sphere, and the Hartree potential
    work_.resize(points * 4);
    components_.reserve(2 * n);
    if (spins_ == 2) {
        // rho_up + rho_down
        spare_.reserve(2 * n);
        to_sphere(rho, spare_.data(), 2);
        add_components<<<blocks(n), kThreads>>>(n, spare_.data(), spare_.data() + n, components_.data());
        launched("the total density");
    } else {
        to_sphere(rho, components_.data(), 1);
    }
    DeviceArray<Complex>& vh = vh_;
    vh.reserve(n);
    hartree_kernel<<<blocks(n), kThreads>>>(n, g_.data(), components_.data(), vh.data(), sum.data());
    launched("the Hartree potential");
    out.hartree = 0.5 * lattice_.volume * total(sum, blocks(n));
    double* hartree = work_.data();
    from_sphere(vh.data(), hartree, 1);
    // the spin densities with the core, their gradients and sigma
    rho_.resize(2 * points);
    spin_densities<<<blocks(points), kThreads>>>(points, spins_, rho, core_.size() ? core_.data() : nullptr,
                                                 rho_.data());
    launched("the spin densities");
    sigma_.resize(3 * points);
    sigma_.zero();
    const bool gga = xc::gradient(functional_);
    if (gga) {
        // n_up and n_down apart, on the sphere, and the 3 components of each gradient
        apart_.reserve(2 * points);
        apart_kernel<<<blocks(points), kThreads>>>(points, rho_.data(), apart_.data());
        launched("the spin densities apart");
        spare_.reserve(3 * n);
        Complex* dens = spare_.data();
        Complex* comp = spare_.data() + 2 * n;
        to_sphere(apart_.data(), dens, 2);
        grads_.resize(6 * points);
        for (int s = 0; s < 2; ++s)
            for (int j = 0; j < 3; ++j) {
                times_ig<<<blocks(n), kThreads>>>(n, j, g_.data(), dens + static_cast<std::size_t>(s) * n, comp);
                launched("a gradient");
                from_sphere(comp, grads_.data() + static_cast<std::size_t>(3 * s + j) * points, 1);
            }
        sigma_kernel<<<blocks(points), kThreads>>>(points, grads_.data(), sigma_.data());
        launched("sigma");
    }
    f_.resize(points);
    v_.resize(2 * points);
    vs_.resize(3 * points);
    evaluate_xc(functional_, points, rho_.data(), sigma_.data(), threshold_, f_.data(), v_.data(), vs_.data(),
                spins_ == 2);
    dot_kernel<<<blocks(points), kThreads>>>(points, f_.data(), nullptr, sum.data());
    launched("the XC energy");
    out.xc = dv * total(sum, blocks(points));
    // the potential of each spin
    double* divergence = work_.data() + points;
    DeviceArray<double>& h = flux_;
    DeviceArray<Complex>& hcomp = flux_g_;
    DeviceArray<Complex>& div = div_;
    for (int s = 0; s < spins_; ++s) {
        const double* div_ptr = nullptr;
        if (gga) {
            h.reserve(3 * points);
            flux_kernel<<<blocks(points), kThreads>>>(points, s, grads_.data(), vs_.data(), h.data());
            launched("the flux of GGA");
            hcomp.reserve(3 * n);
            div.reserve(n);
            to_sphere(h.data(), hcomp.data(), 3);
            divergence_kernel<<<blocks(n), kThreads>>>(n, g_.data(), hcomp.data(), div.data());
            launched("the divergence");
            from_sphere(div.data(), divergence, 1);
            div_ptr = divergence;
        }
        combine_kernel<<<blocks(points), kThreads>>>(points, s, hartree, v_.data(), div_ptr,
                                                    hxc + static_cast<std::size_t>(s) * points);
        launched("the potential");
        dot_kernel<<<blocks(points), kThreads>>>(points, hxc + static_cast<std::size_t>(s) * points,
                                                 rho + static_cast<std::size_t>(s) * points, sum.data());
        launched("int V rho");
        out.hxc_density += dv * total(sum, blocks(points));
    }
    return out;
}

double Potentials::integral(const double* a, const double* b, int count) {
    const std::size_t points = fft_.points();
    const std::size_t n = points * static_cast<std::size_t>(count);
    DeviceArray<double>& sum = sum_;
    sum.reserve(blocks(n) + 1);
    dot_kernel<<<blocks(n), kThreads>>>(n, a, b, sum.data());  // b null: the integral of a
    launched("an integral");
    return lattice_.volume / static_cast<double>(points) * total(sum, blocks(n));
}

std::array<double, 6> Potentials::gga_strain() const {
    std::array<double, 6> out{};
    if (!xc::gradient(functional_)) return out;
    const std::size_t points = fft_.points();
    if (grads_.size() < 6 * points || vs_.size() < 3 * points) throw std::logic_error("no potential computed");
    const unsigned grid = std::min<unsigned>(blocks(points), 1024u);
    DeviceArray<double> sums(6 * (static_cast<std::size_t>(grid) + 1));
    gga_stress_kernel<<<grid, kThreads>>>(points, grads_.data(), vs_.data(), sums.data());
    launched("the stress of GGA");
    ordered_sums<6>(static_cast<int>(grid), sums.data(), out.data());
    const double dv = lattice_.volume / static_cast<double>(points);
    for (double& x : out) x *= -dv;
    return out;
}

}  // namespace mdapy::dft
