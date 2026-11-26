// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/mixer.h"

#include "dft/core/check.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace mdapy::dft {

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

// x of A x = b, A symmetric positive semidefinite m x m: by its eigenvectors
// (Jacobi), directions of eigenvalues below cut times the largest dropped
std::vector<double> solve_symmetric(std::vector<double> a, const std::vector<double>& b, int m, double cut) {
    std::vector<double> v(static_cast<std::size_t>(m * m), 0.0);
    for (int i = 0; i < m; ++i) v[static_cast<std::size_t>(i * m + i)] = 1.0;
    auto at = [&](std::vector<double>& x, int i, int j) -> double& { return x[static_cast<std::size_t>(i * m + j)]; };
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; ++j) off += at(a, i, j) * at(a, i, j);
        if (off < 1e-30) break;
        for (int p = 0; p < m; ++p)
            for (int q = p + 1; q < m; ++q) {
                if (std::abs(at(a, p, q)) < 1e-300) continue;
                const double theta = (at(a, q, q) - at(a, p, p)) / (2.0 * at(a, p, q));
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < m; ++k) {
                    const double akp = at(a, k, p), akq = at(a, k, q);
                    at(a, k, p) = c * akp - s * akq;
                    at(a, k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < m; ++k) {
                    const double apk = at(a, p, k), aqk = at(a, q, k);
                    at(a, p, k) = c * apk - s * aqk;
                    at(a, q, k) = s * apk + c * aqk;
                }
                for (int k = 0; k < m; ++k) {
                    const double vkp = at(v, k, p), vkq = at(v, k, q);
                    at(v, k, p) = c * vkp - s * vkq;
                    at(v, k, q) = s * vkp + c * vkq;
                }
            }
    }
    double largest = 0.0;
    for (int i = 0; i < m; ++i) largest = std::max(largest, at(a, i, i));
    std::vector<double> x(static_cast<std::size_t>(m), 0.0);
    for (int e = 0; e < m; ++e) {
        const double lambda = at(a, e, e);
        if (!(lambda > cut * largest)) continue;
        double proj = 0.0;
        for (int k = 0; k < m; ++k) proj += at(v, k, e) * b[static_cast<std::size_t>(k)];
        for (int k = 0; k < m; ++k) x[static_cast<std::size_t>(k)] += at(v, k, e) * proj / lambda;
    }
    return x;
}

constexpr int kThreads = 256;
constexpr int kGrid = 1024;  // the most blocks of a sum over the sphere
constexpr int kChunk = 16;   // vectors of a pass of the sums and of the combination

unsigned blocks(std::size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }
unsigned grid_of(std::size_t n) { return std::min<unsigned>(blocks(n), static_cast<unsigned>(kGrid)); }

struct Chunk {
    const cuDoubleComplex* p[kChunk];
};
struct Coefficients {
    double c[kChunk];
};

// The sum of the values of the threads of a block into part[0], in an order
// that is always the same.
__device__ __forceinline__ void block_sum(double* part) {
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
}

// r = out - in; with a step before, dr = r - last_r and drho = in - last_in;
// then last_in = in and last_r = r
__global__ void step_kernel(std::size_t size, const cuDoubleComplex* in, const cuDoubleComplex* out,
                            cuDoubleComplex* r, cuDoubleComplex* last_in, cuDoubleComplex* last_r,
                            cuDoubleComplex* dr, cuDoubleComplex* drho) {
    const std::size_t j = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (j >= size) return;
    const cuDoubleComplex x = in[j], y = out[j];
    const cuDoubleComplex res = make_cuDoubleComplex(y.x - x.x, y.y - x.y);
    r[j] = res;
    if (dr != nullptr) {
        const cuDoubleComplex lr = last_r[j], li = last_in[j];
        dr[j] = make_cuDoubleComplex(res.x - lr.x, res.y - lr.y);
        drho[j] = make_cuDoubleComplex(x.x - li.x, x.y - li.y);
    }
    last_in[j] = x;
    last_r[j] = res;
}

// partial[i blocks + block] = the sum over the elements of the block of
// metric (a_i . b) (the real part of conj(a_i) b), i < count
__global__ void dots_kernel(std::size_t size, Chunk a, int count, const cuDoubleComplex* __restrict__ b,
                            const double* __restrict__ metric, double* __restrict__ partial) {
    __shared__ double part[kThreads];
    double sum[kChunk];
#pragma unroll
    for (int i = 0; i < kChunk; ++i) sum[i] = 0.0;
    for (std::size_t j = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; j < size;
         j += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const cuDoubleComplex y = b[j];
        const double w = metric[j];
#pragma unroll
        for (int i = 0; i < kChunk; ++i)
            if (i < count) {
                const cuDoubleComplex x = a.p[i][j];
                sum[i] += w * (x.x * y.x + x.y * y.y);
            }
    }
#pragma unroll
    for (int i = 0; i < kChunk; ++i)
        if (i < count) {
            part[threadIdx.x] = sum[i];
            block_sum(part);
            if (threadIdx.x == 0) partial[static_cast<std::size_t>(i) * gridDim.x + blockIdx.x] = part[0];
            __syncthreads();
        }
}

// partial[block] = the sum over the elements of the block of |out - in|^2 / G^2
// on the total density (G = 0 left out) and |out - in|^2 on the magnetization
__global__ void residual_kernel(std::size_t n, std::size_t size, const cuDoubleComplex* __restrict__ in,
                                const cuDoubleComplex* __restrict__ out, const double* __restrict__ g2,
                                double* __restrict__ partial) {
    __shared__ double part[kThreads];
    double s = 0.0;
    for (std::size_t j = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; j < size;
         j += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        if (j == 0) continue;
        const double re = out[j].x - in[j].x, im = out[j].y - in[j].y;
        s += j < n ? (re * re + im * im) / g2[j] : re * re + im * im;
    }
    part[threadIdx.x] = s;
    block_sum(part);
    if (threadIdx.x == 0) partial[blockIdx.x] = part[0];
}

// out[i] = the sum of the blocks partial[i blocks ...] in their order, a block
// to each i
__global__ void column_sums_kernel(int count, const double* __restrict__ partial, double* __restrict__ out) {
    __shared__ double part[kThreads];
    const double* column = partial + static_cast<std::size_t>(blockIdx.x) * count;
    double s = 0.0;
    for (int b = static_cast<int>(threadIdx.x); b < count; b += kThreads) s += column[b];
    part[threadIdx.x] = s;
    block_sum(part);
    if (threadIdx.x == 0) out[blockIdx.x] = part[0];
}

// rho = base_rho - sum_i c_i drho_i and res = base_res - sum_i c_i dr_i over a
// chunk of the differences, in their order
__global__ void combine_kernel(std::size_t size, int count, Coefficients c, Chunk drho, Chunk dr,
                               const cuDoubleComplex* base_rho, const cuDoubleComplex* base_res,
                               cuDoubleComplex* rho_out, cuDoubleComplex* res_out) {
    const std::size_t j = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (j >= size) return;
    cuDoubleComplex rho = base_rho[j], res = base_res[j];
#pragma unroll
    for (int i = 0; i < kChunk; ++i)
        if (i < count) {
            const double ci = c.c[i];
            const cuDoubleComplex x = drho.p[i][j], y = dr.p[i][j];
            rho.x -= ci * x.x;
            rho.y -= ci * x.y;
            res.x -= ci * y.x;
            res.y -= ci * y.y;
        }
    rho_out[j] = rho;
    res_out[j] = res;
}

// next = rho + factor total on the total density (with a preconditioner
// given; G = 0 left out) and + scale P res from `from` on; next[0] = first[0]
__global__ void finish_kernel(std::size_t size, std::size_t n, std::size_t from, double scale,
                              const double* __restrict__ precondition, const cuDoubleComplex* rho,
                              const cuDoubleComplex* __restrict__ res, const cuDoubleComplex* __restrict__ total,
                              double factor, const cuDoubleComplex* __restrict__ first, cuDoubleComplex* next) {
    const std::size_t j = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (j >= size) return;
    cuDoubleComplex v = rho[j];
    if (total != nullptr && j < n) {
        v.x += factor * total[j].x;
        v.y += factor * total[j].y;
    }
    if (j >= from) {
        const double p = scale * precondition[j];
        v.x += p * res[j].x;
        v.y += p * res[j].y;
    }
    if (j == 0) v = first[0];
    next[j] = v;
}

void launched(const char* what) { check_cuda(cudaGetLastError(), what); }

}  // namespace

Mixer::Mixer(std::vector<double> g2, double volume, int channels, MixingOptions options)
    : g2_(std::move(g2)), volume_(volume), channels_(channels), options_(options), n_(g2_.size()) {
    if (channels < 1 || channels > 2) throw std::invalid_argument("one or two channels of density");
    if (n_ < 2 || g2_[0] != 0.0) throw std::invalid_argument("the sphere of the density starts at G = 0");
    if (!(options_.beta > 0.0) || options_.history < 1 || !(options_.metric_weight >= 1.0) ||
        options_.kerker < 0.0 || options_.minimum < 0.0 || !(options_.magnetic_beta > 0.0) ||
        options_.magnetic_kerker < 0.0 || !(options_.conditioning >= 1.0))
        throw std::invalid_argument("mixing: beta and magnetic_beta > 0, history >= 1, metric_weight >= 1, "
                                    "kerker >= 0, minimum >= 0, conditioning >= 1");
    double shortest = 0.0;
    for (std::size_t g = 1; g < n_; ++g)
        if (g2_[g] > 0.0 && (shortest == 0.0 || g2_[g] < shortest)) shortest = g2_[g];
    const double q1 = (options_.metric_weight - 1.0) * shortest;
    std::vector<double> metric(n_ * channels_, 1.0), precondition(n_ * channels_, options_.magnetic_beta);
    if (channels_ == 2 && options_.magnetic_kerker > 0.0)
        for (std::size_t g = 1; g < n_; ++g) {
            const double x = g2_[g], q = options_.magnetic_kerker;
            precondition[n_ + g] = options_.magnetic_beta * x / (x + q * q);
        }
    for (std::size_t g = 0; g < n_; ++g) {
        const double x = g2_[g];
        metric[g] = g == 0 ? 0.0 : (x + q1) / x;
        precondition[g] = g == 0 ? 0.0
                                 : (options_.kerker > 0.0
                                        ? std::max(options_.minimum,
                                                   options_.beta * x / (x + options_.kerker * options_.kerker))
                                        : options_.beta);
    }
    metric_.upload(metric.data(), metric.size());
    precondition_.upload(precondition.data(), precondition.size());
    g2_device_.upload(g2_.data(), g2_.size());
}

void Mixer::reset() {
    started_ = false;
    order_.clear();
    products_.clear();
    extra_inputs_.clear();
    extra_residuals_.clear();
}

void Mixer::set_preconditioner(std::function<void(std::vector<cuDoubleComplex>&)> apply) {
    preconditioner_ = std::move(apply);
}

std::vector<cuDoubleComplex> gmres(
    const std::function<std::vector<cuDoubleComplex>(const std::vector<cuDoubleComplex>&)>& apply,
    const std::vector<cuDoubleComplex>& b, int iterations, double tolerance) {
    using C = std::complex<double>;
    const std::size_t n = b.size();
    auto norm = [](const std::vector<C>& v) {
        double s = 0.0;
        for (const auto& x : v) s += std::norm(x);
        return std::sqrt(s);
    };
    std::vector<C> r(n);
    for (std::size_t i = 0; i < n; ++i) r[i] = C(b[i].x, b[i].y);
    const double beta = norm(r);
    std::vector<cuDoubleComplex> x(n, make_cuDoubleComplex(0.0, 0.0));
    if (!(beta > 0.0) || iterations < 1) return x;
    const int m = iterations;
    std::vector<std::vector<C>> v;
    v.reserve(static_cast<std::size_t>(m) + 1);
    for (auto& e : r) e /= beta;
    v.push_back(std::move(r));
    // the Hessenberg matrix by columns, its Givens rotations and the residual
    std::vector<std::vector<C>> h(static_cast<std::size_t>(m), std::vector<C>(static_cast<std::size_t>(m) + 1));
    std::vector<double> cs(static_cast<std::size_t>(m));
    std::vector<C> sn(static_cast<std::size_t>(m)), g(static_cast<std::size_t>(m) + 1);
    g[0] = beta;
    int k = 0;
    std::vector<cuDoubleComplex> in(n);
    for (int j = 0; j < m; ++j) {
        const auto& vj = v[static_cast<std::size_t>(j)];
        for (std::size_t i = 0; i < n; ++i) in[i] = make_cuDoubleComplex(vj[i].real(), vj[i].imag());
        const auto out = apply(in);
        std::vector<C> w(n);
        for (std::size_t i = 0; i < n; ++i) w[i] = C(out[i].x, out[i].y);
        auto& col = h[static_cast<std::size_t>(j)];
        for (int i = 0; i <= j; ++i) {  // modified Gram-Schmidt
            const auto& vi = v[static_cast<std::size_t>(i)];
            C d(0.0, 0.0);
            for (std::size_t q = 0; q < n; ++q) d += std::conj(vi[q]) * w[q];
            col[static_cast<std::size_t>(i)] = d;
            for (std::size_t q = 0; q < n; ++q) w[q] -= d * vi[q];
        }
        const double next = norm(w);
        col[static_cast<std::size_t>(j) + 1] = next;
        for (int i = 0; i < j; ++i) {  // the rotations so far
            const C a = col[static_cast<std::size_t>(i)], c2 = col[static_cast<std::size_t>(i) + 1];
            col[static_cast<std::size_t>(i)] = cs[static_cast<std::size_t>(i)] * a + sn[static_cast<std::size_t>(i)] * c2;
            col[static_cast<std::size_t>(i) + 1] =
                -std::conj(sn[static_cast<std::size_t>(i)]) * a + cs[static_cast<std::size_t>(i)] * c2;
        }
        const C a = col[static_cast<std::size_t>(j)], bb = col[static_cast<std::size_t>(j) + 1];
        const double rr = std::sqrt(std::norm(a) + std::norm(bb));
        if (std::abs(a) == 0.0) {
            cs[static_cast<std::size_t>(j)] = 0.0;
            sn[static_cast<std::size_t>(j)] = 1.0;
        } else {
            cs[static_cast<std::size_t>(j)] = std::abs(a) / rr;
            sn[static_cast<std::size_t>(j)] = a * std::conj(bb) / (rr * std::abs(a));
        }
        col[static_cast<std::size_t>(j)] = cs[static_cast<std::size_t>(j)] * a + sn[static_cast<std::size_t>(j)] * bb;
        col[static_cast<std::size_t>(j) + 1] = 0.0;
        g[static_cast<std::size_t>(j) + 1] = -std::conj(sn[static_cast<std::size_t>(j)]) * g[static_cast<std::size_t>(j)];
        g[static_cast<std::size_t>(j)] *= cs[static_cast<std::size_t>(j)];
        k = j + 1;
        if (std::abs(g[static_cast<std::size_t>(j) + 1]) < tolerance * beta || !(next > 0.0)) break;
        for (auto& e : w) e /= next;
        v.push_back(std::move(w));
    }
    // H y = g, upper triangular k x k, and x = V y
    std::vector<C> y(static_cast<std::size_t>(k));
    for (int i = k - 1; i >= 0; --i) {
        C s = g[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < k; ++j) s -= h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(j)];
        y[static_cast<std::size_t>(i)] = s / h[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < k; ++i)
        for (std::size_t q = 0; q < n; ++q) {
            const C t = y[static_cast<std::size_t>(i)] * v[static_cast<std::size_t>(i)][q];
            x[q].x += t.real();
            x[q].y += t.imag();
        }
    return x;
}


void Mixer::set_scale(double factor) {
    if (!(factor > 0.0)) throw std::invalid_argument("the scale of the mixing must be positive");
    scale_ = factor;
}

void Mixer::dots(const std::vector<const cuDoubleComplex*>& a, const cuDoubleComplex* b, double* out) {
    const std::size_t size = n_ * channels_;
    const unsigned grid = grid_of(size);
    for (std::size_t start = 0; start < a.size(); start += kChunk) {
        const int count = static_cast<int>(std::min<std::size_t>(kChunk, a.size() - start));
        Chunk chunk{};
        for (int i = 0; i < count; ++i) chunk.p[i] = a[start + static_cast<std::size_t>(i)];
        partial_.reserve(static_cast<std::size_t>(count) * grid);
        sums_.reserve(static_cast<std::size_t>(count));
        dots_kernel<<<grid, kThreads>>>(size, chunk, count, b, metric_.data(), partial_.data());
        launched("the products of the mixing");
        column_sums_kernel<<<static_cast<unsigned>(count), kThreads>>>(static_cast<int>(grid), partial_.data(),
                                                                     sums_.data());
        launched("the sums of the mixing");
        sums_.download(out + start, static_cast<std::size_t>(count));
    }
}

double Mixer::residual_energy(const cuDoubleComplex* in, const cuDoubleComplex* out, bool total_only) {
    const std::size_t size = total_only ? n_ : n_ * channels_;
    const unsigned grid = grid_of(size);
    partial_.reserve(grid);
    sums_.reserve(1);
    residual_kernel<<<grid, kThreads>>>(n_, size, in, out, g2_device_.data(), partial_.data());
    launched("the residual of the density");
    column_sums_kernel<<<1, kThreads>>>(static_cast<int>(grid), partial_.data(), sums_.data());
    launched("the sum of the residual");
    double s = 0.0;
    sums_.download(&s, 1);
    return 2.0 * kPi * volume_ * s;
}

void Mixer::mix(const cuDoubleComplex* in, const cuDoubleComplex* out, cuDoubleComplex* next,
                std::vector<double>* extra, const std::vector<double>* extra_out) {
    const std::size_t size = n_ * channels_;
    const bool along = extra != nullptr;
    if (along != (extra_out != nullptr) || (along && (extra->size() != extra_out->size() ||
                                                      extra->size() % static_cast<std::size_t>(channels_) != 0)))
        throw std::invalid_argument("the components that come along: their input and output, channels of one size");
    if (along != !extra_inputs_.empty() && started_)
        throw std::invalid_argument("components that come along in every step or none");
    last_input_.resize(size);
    last_residual_.resize(size);
    residual_.resize(size);
    combined_.resize(size);
    // the residual, and the difference from the last step into a slot: that of
    // the oldest difference when as many as the history are kept, which goes
    cuDoubleComplex *dr = nullptr, *drho = nullptr;
    int slot = -1;
    if (started_) {
        if (static_cast<int>(order_.size()) >= options_.history) {
            slot = order_.front();
            order_.pop_front();
            products_.pop_front();
            for (auto& p : products_) p.erase(p.begin());
        } else {
            slot = static_cast<int>(order_.size());
        }
        if (static_cast<std::size_t>(slot) >= residual_slots_.size()) {
            residual_slots_.resize(static_cast<std::size_t>(slot) + 1);
            input_slots_.resize(static_cast<std::size_t>(slot) + 1);
        }
        residual_slots_[static_cast<std::size_t>(slot)].resize(size);
        input_slots_[static_cast<std::size_t>(slot)].resize(size);
        dr = residual_slots_[static_cast<std::size_t>(slot)].data();
        drho = input_slots_[static_cast<std::size_t>(slot)].data();
    }
    step_kernel<<<blocks(size), kThreads>>>(size, in, out, residual_.data(), last_input_.data(),
                                            last_residual_.data(), dr, drho);
    launched("the residual of the mixing");
    started_ = true;
    if (slot >= 0) {  // the products of the new difference with those kept and itself
        std::vector<const cuDoubleComplex*> a;
        for (const int s : order_) a.push_back(residual_slots_[static_cast<std::size_t>(s)].data());
        a.push_back(dr);
        std::vector<double> row(a.size());
        dots(a, dr, row.data());
        order_.push_back(slot);
        products_.push_back(std::move(row));
    }
    if (along) {
        std::vector<double> er(extra->size());
        for (std::size_t i = 0; i < er.size(); ++i) er[i] = (*extra_out)[i] - (*extra)[i];
        extra_inputs_.push_back(*extra);
        extra_residuals_.push_back(std::move(er));
        while (static_cast<int>(extra_inputs_.size()) > options_.history + 1) {
            extra_inputs_.pop_front();
            extra_residuals_.pop_front();
        }
    }
    std::vector<double> x, xr;
    if (along) {
        x = extra_inputs_.back();
        xr = extra_residuals_.back();
    }
    const int m = static_cast<int>(order_.size());  // differences
    const cuDoubleComplex* rho = in;
    if (m > 0) {
        std::vector<const cuDoubleComplex*> dr_all, drho_all;
        for (const int s : order_) {
            dr_all.push_back(residual_slots_[static_cast<std::size_t>(s)].data());
            drho_all.push_back(input_slots_[static_cast<std::size_t>(s)].data());
        }
        std::vector<double> a(static_cast<std::size_t>(m * m)), b(static_cast<std::size_t>(m));
        dots(dr_all, residual_.data(), b.data());
        for (int i = 0; i < m; ++i)
            for (int j = 0; j <= i; ++j)
                a[static_cast<std::size_t>(i * m + j)] = a[static_cast<std::size_t>(j * m + i)] =
                    products_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        // A = dR^T dR: its eigenvalues are the squares of the singular values
        const std::vector<double> gamma =
            solve_symmetric(a, b, m, 1.0 / (options_.conditioning * options_.conditioning));
        if (along)
            for (int i = 0; i < m; ++i) {
                const double c = gamma[static_cast<std::size_t>(i)];
                const auto& x0 = extra_inputs_[static_cast<std::size_t>(i)];
                const auto& x1 = extra_inputs_[static_cast<std::size_t>(i + 1)];
                const auto& r0 = extra_residuals_[static_cast<std::size_t>(i)];
                const auto& r1 = extra_residuals_[static_cast<std::size_t>(i + 1)];
                for (std::size_t j = 0; j < x.size(); ++j) {
                    x[j] -= c * (x1[j] - x0[j]);
                    xr[j] -= c * (r1[j] - r0[j]);
                }
            }
        // the combination, chunk after chunk of the differences: rho into
        // combined_, the residual in place
        for (int start = 0; start < m; start += kChunk) {
            const int count = std::min(kChunk, m - start);
            Chunk pr{}, pd{};
            Coefficients c{};
            for (int i = 0; i < count; ++i) {
                pr.p[i] = drho_all[static_cast<std::size_t>(start + i)];
                pd.p[i] = dr_all[static_cast<std::size_t>(start + i)];
                c.c[i] = gamma[static_cast<std::size_t>(start + i)];
            }
            combine_kernel<<<blocks(size), kThreads>>>(size, count, c, pr, pd, start == 0 ? in : combined_.data(),
                                                     residual_.data(), combined_.data(), residual_.data());
            launched("the combination of the mixing");
        }
        rho = combined_.data();
    }
    std::size_t from = 0;
    const cuDoubleComplex* total = nullptr;
    if (preconditioner_) {  // the total density by the preconditioner given, on the host
        std::vector<cuDoubleComplex> host(n_);
        residual_.download(host.data(), n_);
        host[0] = make_cuDoubleComplex(0.0, 0.0);
        preconditioner_(host);
        total_.upload(host.data(), n_);
        total = total_.data();
        from = n_;
    }
    finish_kernel<<<blocks(size), kThreads>>>(size, n_, from, scale_, precondition_.data(), rho, residual_.data(),
                                              total, scale_ * options_.beta, last_input_.data(), next);
    launched("the next density of the mixing");
    if (along) {
        const std::size_t per = x.size() / static_cast<std::size_t>(channels_);
        for (std::size_t j = 0; j < x.size(); ++j)
            (*extra)[j] = x[j] + scale_ * (j < per ? options_.beta : options_.magnetic_beta) * xr[j];
    }
}

}  // namespace mdapy::dft
