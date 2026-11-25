// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/fft/fft.h"

#include "dft/core/profile.h"

#include <stdexcept>
#include <vector>

#include "dft/fft/box_fft.h"

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;

unsigned blocks(std::size_t work) {
    return static_cast<unsigned>((work + kThreads - 1) / kThreads);
}

__global__ void scatter_kernel(const Complex* c, std::size_t n, const std::int64_t* box,
                               const std::int64_t* partner, int count, Complex* boxes,
                               std::size_t points) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t b = i / n, g = i % n;
    const Complex value = c[i];
    boxes[b * points + box[g]] = value;
    if (partner != nullptr && partner[g] >= 0) boxes[b * points + partner[g]] = cuConj(value);
}

__global__ void gather_kernel(const Complex* boxes, std::size_t points, const std::int64_t* box,
                              std::size_t n, int count, double scale, Complex* c) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t b = i / n, g = i % n;
    const Complex value = boxes[b * points + box[g]];
    c[i] = make_cuDoubleComplex(scale * value.x, scale * value.y);
}

__global__ void gather_add_kernel(const Complex* boxes, std::size_t points,
                                  const std::int64_t* box, std::size_t n, int count, double scale,
                                  Complex* c) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t b = i / n, g = i % n;
    const Complex value = boxes[b * points + box[g]];
    c[i] = make_cuDoubleComplex(c[i].x + scale * value.x, c[i].y + scale * value.y);
}

__global__ void move_kernel(const Complex* from, std::size_t from_points,
                            const std::int64_t* from_index, Complex* to, std::size_t to_points,
                            const std::int64_t* to_index, std::size_t kept, int count) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= kept * count) return;
    const std::size_t b = i / kept, g = i % kept;
    to[b * to_points + to_index[g]] = from[b * from_points + from_index[g]];
}

__global__ void scatter_single_kernel(const cuComplex* c, std::size_t n, const std::int64_t* box,
                                      const std::int64_t* partner, int count, cuComplex* boxes, std::size_t points) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t b = i / n, g = i % n;
    const cuComplex value = c[i];
    boxes[b * points + box[g]] = value;
    if (partner != nullptr && partner[g] >= 0) boxes[b * points + partner[g]] = cuConjf(value);
}

__global__ void gather_add_single_kernel(const cuComplex* boxes, std::size_t points, const std::int64_t* box,
                                         std::size_t n, int count, float scale, cuComplex* c) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t b = i / n, g = i % n;
    const cuComplex value = boxes[b * points + box[g]];
    c[i] = make_cuComplex(c[i].x + scale * value.x, c[i].y + scale * value.y);
}

void launched(const char* what) { check_cuda(cudaGetLastError(), what); }

double* reals(Complex* values) { return reinterpret_cast<double*>(values); }
float* reals(cuComplex* values) { return reinterpret_cast<float*>(values); }

}  // namespace

Fft::Fft(const Int3& grid, Kernels kernels) : grid_(grid) {
    for (int i = 0; i < 3; ++i)
        if (grid[i] < 1) throw std::invalid_argument("an FFT grid needs points along every vector");
    if (kernels == Kernels::Mdapy && BoxFft::supported(grid)) own_ = std::make_unique<BoxFft>(grid);
}

Fft::~Fft() {
    for (auto& entry : plans_) cufftDestroy(entry.second);
}

std::size_t Fft::points() const noexcept {
    return static_cast<std::size_t>(grid_[0]) * grid_[1] * grid_[2];
}

std::size_t Fft::spectrum() const noexcept {
    return static_cast<std::size_t>(grid_[0]) * grid_[1] * (grid_[2] / 2 + 1);
}

cufftHandle Fft::plan(cufftType type, int count) {
    const auto key = std::make_pair(static_cast<int>(type), count);
    const auto found = plans_.find(key);
    if (found != plans_.end()) return found->second;
    int n[3] = {grid_[0], grid_[1], grid_[2]};
    const int complex_points = static_cast<int>(points());
    const int half = static_cast<int>(spectrum());
    int idist = complex_points, odist = complex_points;
    if (type == CUFFT_D2Z || type == CUFFT_R2C) odist = half;
    if (type == CUFFT_Z2D || type == CUFFT_C2R) idist = half;
    cufftHandle handle{};
    check_cufft(cufftPlanMany(&handle, 3, n, nullptr, 1, idist, nullptr, 1, odist, type, count),
                "cufftPlanMany");
    check_cufft(cufftSetStream(handle, cudaStreamPerThread), "cufftSetStream");
    plans_.emplace(key, handle);
    return handle;
}

void Fft::to_real(Complex* boxes, int count, int window) {
    MDAPY_SPAN("fft");
    if (own_) return own_->inverse(reals(boxes), count, window);
    check_cufft(cufftExecZ2Z(plan(CUFFT_Z2Z, count), boxes, boxes, CUFFT_INVERSE),
                "cufftExecZ2Z");
}

void Fft::to_reciprocal(Complex* boxes, int count, int window) {
    MDAPY_SPAN("fft");
    if (own_) return own_->forward(reals(boxes), count, window);
    check_cufft(cufftExecZ2Z(plan(CUFFT_Z2Z, count), boxes, boxes, CUFFT_FORWARD),
                "cufftExecZ2Z");
}

void Fft::to_real(cuComplex* boxes, int count, int window) {
    MDAPY_SPAN("fft");
    if (own_) return own_->inverse(reals(boxes), count, window);
    check_cufft(cufftExecC2C(plan(CUFFT_C2C, count), boxes, boxes, CUFFT_INVERSE), "cufftExecC2C");
}

void Fft::to_reciprocal(cuComplex* boxes, int count, int window) {
    MDAPY_SPAN("fft");
    if (own_) return own_->forward(reals(boxes), count, window);
    check_cufft(cufftExecC2C(plan(CUFFT_C2C, count), boxes, boxes, CUFFT_FORWARD), "cufftExecC2C");
}

void Fft::to_real(cuComplex* spectrum, float* values, int count, int window) {
    MDAPY_SPAN("fft");
    if (own_) return own_->inverse(reals(spectrum), values, count, window);
    check_cufft(cufftExecC2R(plan(CUFFT_C2R, count), spectrum, values), "cufftExecC2R");
}

void Fft::to_reciprocal(const float* values, cuComplex* spectrum, int count, int window) {
    MDAPY_SPAN("fft");
    if (own_) return own_->forward(values, reals(spectrum), count, window);
    check_cufft(cufftExecR2C(plan(CUFFT_R2C, count), const_cast<float*>(values), spectrum), "cufftExecR2C");
}

void Fft::to_real(Complex* spectrum, double* values, int count, int window) {
    MDAPY_SPAN("fft");
    if (own_) return own_->inverse(reals(spectrum), values, count, window);
    check_cufft(cufftExecZ2D(plan(CUFFT_Z2D, count), spectrum, values), "cufftExecZ2D");
}

void Fft::to_reciprocal(const double* values, Complex* spectrum, int count, int window) {
    MDAPY_SPAN("fft");
    if (own_) return own_->forward(values, reals(spectrum), count, window);
    // cuFFT takes a pointer that it does not write for the real-to-complex input
    check_cufft(cufftExecD2Z(plan(CUFFT_D2Z, count), const_cast<double*>(values), spectrum),
                "cufftExecD2Z");
}

void scatter(const Complex* coefficients, std::size_t n, const std::int64_t* box,
             const std::int64_t* partner, int count, Complex* boxes, std::size_t points) {
    MDAPY_SPAN("scatter");
    check_cuda(cudaMemset(boxes, 0, static_cast<std::size_t>(count) * points * sizeof(Complex)),
               "cudaMemset");
    if (n == 0 || count == 0) return;
    scatter_kernel<<<blocks(n * count), kThreads>>>(coefficients, n, box, partner, count, boxes,
                                                    points);
    launched("scatter");
}

namespace {

__device__ inline Complex zero_of(const Complex*) { return make_cuDoubleComplex(0.0, 0.0); }
__device__ inline cuComplex zero_of(const cuComplex*) { return make_cuComplex(0.0f, 0.0f); }

// The planes of the first index within window of 0 of count boxes to zeros,
// in one launch: (2 window + 1) plane values a box.
template <class T>
__global__ void zero_planes_kernel(std::size_t total, std::size_t plane, int n1, int window, T* boxes) {
    const std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (e >= total) return;
    const std::size_t w = 2 * static_cast<std::size_t>(window) + 1, per = w * plane;
    const std::size_t box = e / per, rest = e - box * per, i = rest / plane;
    const std::size_t x = i <= static_cast<std::size_t>(window) ? i : static_cast<std::size_t>(n1) - (w - i);
    boxes[(box * static_cast<std::size_t>(n1) + x) * plane + (rest - i * plane)] = zero_of(boxes);
}

// count boxes of the grid to zeros, or only their planes of the first index
// within window of 0
template <class T>
void zero_window(T* boxes, int count, const Int3& grid, int window) {
    const std::size_t plane = static_cast<std::size_t>(grid[1]) * grid[2];
    if (window < 0 || 2 * window + 1 >= grid[0]) {
        check_cuda(cudaMemset(boxes, 0, static_cast<std::size_t>(count) * grid[0] * plane * sizeof(T)), "cudaMemset");
        return;
    }
    const std::size_t total = static_cast<std::size_t>(count) * (2 * static_cast<std::size_t>(window) + 1) * plane;
    zero_planes_kernel<T><<<blocks(total), kThreads>>>(total, plane, grid[0], window, boxes);
    launched("the planes of the sphere to zeros");
}

}  // namespace

void scatter_window(const Complex* coefficients, std::size_t n, const std::int64_t* box, const std::int64_t* partner,
                    int count, Complex* boxes, const Int3& grid, int window) {
    MDAPY_SPAN("scatter");
    zero_window(boxes, count, grid, window);
    if (n == 0 || count == 0) return;
    const std::size_t points = static_cast<std::size_t>(grid[0]) * grid[1] * grid[2];
    scatter_kernel<<<blocks(n * count), kThreads>>>(coefficients, n, box, partner, count, boxes, points);
    launched("scatter");
}

void scatter_window(const cuComplex* coefficients, std::size_t n, const std::int64_t* box,
                    const std::int64_t* partner, int count, cuComplex* boxes, const Int3& grid, int window) {
    MDAPY_SPAN("scatter");
    zero_window(boxes, count, grid, window);
    if (n == 0 || count == 0) return;
    const std::size_t points = static_cast<std::size_t>(grid[0]) * grid[1] * grid[2];
    scatter_single_kernel<<<blocks(n * count), kThreads>>>(coefficients, n, box, partner, count, boxes, points);
    launched("scatter");
}

void gather(const Complex* boxes, std::size_t points, const std::int64_t* box, std::size_t n,
            int count, double scale, Complex* coefficients) {
    MDAPY_SPAN("gather");
    if (n == 0 || count == 0) return;
    gather_kernel<<<blocks(n * count), kThreads>>>(boxes, points, box, n, count, scale,
                                                   coefficients);
    launched("gather");
}

void gather_add(const Complex* boxes, std::size_t points, const std::int64_t* box, std::size_t n,
                int count, double scale, Complex* coefficients) {
    MDAPY_SPAN("gather");
    if (n == 0 || count == 0) return;
    gather_add_kernel<<<blocks(n * count), kThreads>>>(boxes, points, box, n, count, scale,
                                                       coefficients);
    launched("gather_add");
}

void scatter(const cuComplex* coefficients, std::size_t n, const std::int64_t* box, int count, cuComplex* boxes,
             std::size_t points) {
    MDAPY_SPAN("scatter");
    scatter(coefficients, n, box, nullptr, count, boxes, points);
}

void scatter(const cuComplex* coefficients, std::size_t n, const std::int64_t* box, const std::int64_t* partner,
             int count, cuComplex* boxes, std::size_t points) {
    MDAPY_SPAN("scatter");
    check_cuda(cudaMemset(boxes, 0, static_cast<std::size_t>(count) * points * sizeof(cuComplex)), "cudaMemset");
    if (n == 0 || count == 0) return;
    scatter_single_kernel<<<blocks(n * count), kThreads>>>(coefficients, n, box, partner, count, boxes, points);
    launched("scatter");
}

void gather_add(const cuComplex* boxes, std::size_t points, const std::int64_t* box, std::size_t n, int count,
                float scale, cuComplex* coefficients) {
    MDAPY_SPAN("gather");
    if (n == 0 || count == 0) return;
    gather_add_single_kernel<<<blocks(n * count), kThreads>>>(boxes, points, box, n, count, scale, coefficients);
    launched("gather_add");
}

Transfer::Transfer(const Int3& coarse, const Int3& fine) {
    for (int i = 0; i < 3; ++i)
        if (coarse[i] < 1 || fine[i] < coarse[i])
            throw std::invalid_argument("the fine grid must hold the coarse one");
    coarse_points_ = static_cast<std::size_t>(coarse[0]) * coarse[1] * coarse[2];
    fine_points_ = static_cast<std::size_t>(fine[0]) * fine[1] * fine[2];
    // the components |m_i| < n_i / 2 of the coarse grid: the Nyquist planes of an
    // even size are left out, as their frequency is ambiguous
    auto range = [](int n) {
        std::vector<int> ms;
        for (int m = -(n - 1) / 2; m <= (n - 1) / 2; ++m) ms.push_back(m);
        return ms;
    };
    auto wrap = [](int m, int n) { return static_cast<std::int64_t>(((m % n) + n) % n); };
    std::vector<std::int64_t> from, to;
    for (const int m0 : range(coarse[0]))
        for (const int m1 : range(coarse[1]))
            for (const int m2 : range(coarse[2])) {
                from.push_back((wrap(m0, coarse[0]) * coarse[1] + wrap(m1, coarse[1])) * coarse[2] +
                               wrap(m2, coarse[2]));
                to.push_back((wrap(m0, fine[0]) * fine[1] + wrap(m1, fine[1])) * fine[2] +
                             wrap(m2, fine[2]));
            }
    coarse_.upload(from.data(), from.size());
    fine_.upload(to.data(), to.size());
}

void Transfer::to_fine(const Complex* coarse, Complex* fine, int count) const {
    MDAPY_SPAN("transfer");
    check_cuda(
        cudaMemset(fine, 0, static_cast<std::size_t>(count) * fine_points_ * sizeof(Complex)),
        "cudaMemset");
    const std::size_t kept = coarse_.size();
    if (kept == 0 || count == 0) return;
    move_kernel<<<blocks(kept * count), kThreads>>>(coarse, coarse_points_, coarse_.data(), fine,
                                                    fine_points_, fine_.data(), kept, count);
    launched("to_fine");
}

void Transfer::to_coarse(const Complex* fine, Complex* coarse, int count) const {
    MDAPY_SPAN("transfer");
    check_cuda(
        cudaMemset(coarse, 0, static_cast<std::size_t>(count) * coarse_points_ * sizeof(Complex)),
        "cudaMemset");
    const std::size_t kept = coarse_.size();
    if (kept == 0 || count == 0) return;
    move_kernel<<<blocks(kept * count), kThreads>>>(fine, fine_points_, fine_.data(), coarse,
                                                    coarse_points_, coarse_.data(), kept, count);
    launched("to_coarse");
}

}  // namespace mdapy::dft
