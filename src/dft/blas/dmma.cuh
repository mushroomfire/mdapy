// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Products of double precision on the tensor cores of NVIDIA GPUs from compute
// capability 8.0 (mma.sync m8n8k4 of f64, true FP64 in another order of its
// sums), for blas.cu, which includes it inside its namespace after GemmArgs:
// real numbers, and complex ones as four real products of their parts.
//
// A block holds BM x BN values of the result, its warps 32 x 32 each as 4 x 4
// products of 8 x 8, from BK terms of the sum at a time in shared memory,
// copied there by cp.async in Stages stages, those outside the matrices zeros.
// Each matrix is kept in shared memory with the index it is read along in
// memory contiguous, rows padded to 4 modulo 16 values, so that the copies and
// the reads of the fragments of a half-warp meet no conflicts of banks.
// Complex values are kept as their real and imaginary parts apart.

#if !defined(MDAPY_GPU_HIP)

template <class S>
struct DmmaForm;
template <>
struct DmmaForm<double> {
    static constexpr int bm = 128, bn = 128, bk = 16, stages = 3, parts = 1;
};
template <>
struct DmmaForm<Cx<double>> {
    static constexpr int bm = 64, bn = 64, bk = 16, stages = 3, parts = 2;
};

// the doubles of the shared memory of a block: its stages of the parts of a and b
template <class S, bool TransA, bool TransB>
constexpr int dmma_shared() {
    using F = DmmaForm<S>;
    constexpr int a = TransA ? F::bm * (F::bk + 4) : F::bk * (F::bm + 4);
    constexpr int b = TransB ? F::bk * (F::bn + 4) : F::bn * (F::bk + 4);
    return F::stages * F::parts * (a + b);
}

__device__ __forceinline__ void copy_async(double* to, const double* from, bool in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(to));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(address), "l"(from), "r"(in ? 8 : 0));
#else
    *to = in ? *from : 0.0;
#endif
}

__device__ __forceinline__ void commit_async() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int Pending>
__device__ __forceinline__ void wait_async() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending));
#endif
}

// d += a b of an 8 x 4 by 4 x 8 product: a[g, t], b[t, g], d[g, 2 t + i] of
// the lane 4 g + t
__device__ __forceinline__ void mma_884(double& d0, double& d1, double a, double b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
                 : "+d"(d0), "+d"(d1)
                 : "d"(a), "d"(b));
#else
    (void)d0, (void)d1, (void)a, (void)b;
#endif
}

// d += a b of a 16 x 4 by 4 x 8 product (compute capability 9.0): a[g + 8 h, t]
// in a_h, b[t, g], d[g + 8 h, 2 t + i] in d_(2 h + i)
__device__ __forceinline__ void mma_1684(double* d, double a0, double a1, double b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                 : "+d"(d[0]), "+d"(d[1]), "+d"(d[2]), "+d"(d[3])
                 : "d"(a0), "d"(a1), "d"(b0));
#else
    (void)d, (void)a0, (void)a1, (void)b0;
#endif
}

// d += a b of a 16 x 8 by 8 x 8 product (compute capability 9.0): a[g + 8 h,
// t + 4 w] in a_(h + 2 w), b[t + 4 w, g] in b_w, d as mma_1684
__device__ __forceinline__ void mma_1688(double* d, const double* a, const double* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, "
        "%3};\n"
        : "+d"(d[0]), "+d"(d[1]), "+d"(d[2]), "+d"(d[3])
        : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]), "d"(b[0]), "d"(b[1]));
#else
    (void)d, (void)a, (void)b;
#endif
}

// the instruction of the products on compute capability 9.0: 884, 1684 or 1688
#ifndef MDAPY_DMMA_SHAPE
#define MDAPY_DMMA_SHAPE 1688
#endif

template <class S, bool TransA, bool TransB>
__global__ void __launch_bounds__((DmmaForm<S>::bm / 32) * (DmmaForm<S>::bn / 32) * 32)
    dmma_kernel(const GemmArgs g, const S alpha, const S beta, const S* __restrict__ a, const S* __restrict__ b, S* c,
                S* partial) {
    using F = DmmaForm<S>;
    constexpr int kBM = F::bm, kBN = F::bn, kBK = F::bk, kStages = F::stages, kParts = F::parts;
    constexpr int kWarpsM = kBM / 32, kThreads = kWarpsM * (kBN / 32) * 32;
    // the pitch of the rows of each matrix in shared memory and the doubles of a stage of one part
    constexpr int kPitchA = TransA ? kBK + 4 : kBM + 4, kPitchB = TransB ? kBN + 4 : kBK + 4;
    constexpr int kSizeA = TransA ? kBM * kPitchA : kBK * kPitchA, kSizeB = TransB ? kBK * kPitchB : kBN * kPitchB;
    extern __shared__ double shared[];
    double* as = shared;                                // [stage][part][...]
    double* bs = shared + kStages * kParts * kSizeA;  // [stage][part][...]

    const unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const unsigned gi = lane / 4, ti = lane % 4;
    const unsigned wm = (warp % kWarpsM) * 32, wn = (warp / kWarpsM) * 32;
    const unsigned i0 = blockIdx.x * kBM, j0 = blockIdx.y * kBN, z = blockIdx.z;
    const unsigned m = static_cast<unsigned>(g.m), n = static_cast<unsigned>(g.n);
    unsigned first = 0, last = static_cast<unsigned>(g.k);
    if (g.segments > 1) {
        first = z * static_cast<unsigned>(g.length);
        last = first + static_cast<unsigned>(g.length) < last ? first + static_cast<unsigned>(g.length) : last;
    } else {
        a += z * g.stride_a;
        b += z * g.stride_b;
        c += z * g.stride_c;
    }
    const double* ad = reinterpret_cast<const double*>(a);
    const double* bd = reinterpret_cast<const double*>(b);
    const unsigned chunks = last > first ? (last - first + kBK - 1) / kBK : 0;

    // the terms [t0, t0 + kBK) of the block's rows of op(a) and columns of op(b) into a stage
    auto load = [&](unsigned stage, unsigned t0) {
        double* to_a = as + stage * kParts * kSizeA;
        double* to_b = bs + stage * kParts * kSizeB;
#pragma unroll
        for (int q = 0; q < kBM * kBK / kThreads; ++q) {
            const unsigned e = tid + q * kThreads;
            const unsigned i = TransA ? e / kBK : e % kBM, t = TransA ? e % kBK : e / kBM;
            const bool in = i0 + i < m && t0 + t < last;
            const long long at = in ? (TransA ? (t0 + t) + (i0 + i) * g.lda : (i0 + i) + (t0 + t) * g.lda) : 0;
            const unsigned place = TransA ? i * kPitchA + t : t * kPitchA + i;
#pragma unroll
            for (int p = 0; p < kParts; ++p) copy_async(to_a + p * kSizeA + place, ad + at * kParts + p, in);
        }
#pragma unroll
        for (int q = 0; q < kBN * kBK / kThreads; ++q) {
            const unsigned e = tid + q * kThreads;
            const unsigned j = TransB ? e % kBN : e / kBK, t = TransB ? e / kBN : e % kBK;
            const bool in = j0 + j < n && t0 + t < last;
            const long long at = in ? (TransB ? (j0 + j) + (t0 + t) * g.ldb : (t0 + t) + (j0 + j) * g.ldb) : 0;
            const unsigned place = TransB ? t * kPitchB + j : j * kPitchB + t;
#pragma unroll
            for (int p = 0; p < kParts; ++p) copy_async(to_b + p * kSizeB + place, bd + at * kParts + p, in);
        }
    };

    // the accumulators: [p][q][part][2] of the 4 x 4 products of the warp
    double acc[4][4][kParts][2];
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int r = 0; r < kParts; ++r) acc[p][q][r][0] = acc[p][q][r][1] = 0.0;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (static_cast<unsigned>(s) < chunks) load(s, first + s * kBK);
        commit_async();
    }
    // the signs of the imaginary parts: of op(a) and op(b), conjugated or not
    const double sign_a = g.conj_a ? -1.0 : 1.0, sign_b = g.conj_b ? -1.0 : 1.0;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        wait_async<kStages - 2>();
        __syncthreads();
        // the stage computed last is free: the next chunk into it
        const unsigned next = chunk + kStages - 1;
        if (next < chunks) load(next % kStages, first + next * kBK);
        commit_async();
        const double* sa = as + (chunk % kStages) * kParts * kSizeA;
        const double* sb = bs + (chunk % kStages) * kParts * kSizeB;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && MDAPY_DMMA_SHAPE != 884
        // the pairs of rows p = 2 h, 2 h + 1 of the accumulators are the halves
        // of the 16 rows of the products of 16 x 8
        constexpr int kSteps = MDAPY_DMMA_SHAPE == 1688 ? 8 : 4, kHalves = kSteps / 4;
#pragma unroll
        for (int t = 0; t < kBK; t += kSteps) {
            // fa[h][part][4 w + 2 v + r]... in the order of the fragments: a[g + 8 v, t + 4 w]
            double fa[2][kParts][2 * kHalves], fb[4][kParts][kHalves];
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int w = 0; w < kHalves; ++w)
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const unsigned i = wm + h * 16 + v * 8 + gi, k = t + w * 4 + ti;
                        const unsigned place = TransA ? i * kPitchA + k : k * kPitchA + i;
#pragma unroll
                        for (int r = 0; r < kParts; ++r) fa[h][r][2 * w + v] = sa[r * kSizeA + place];
                    }
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int w = 0; w < kHalves; ++w) {
                    const unsigned j = wn + q * 8 + gi, k = t + w * 4 + ti;
                    const unsigned place = TransB ? k * kPitchB + j : j * kPitchB + k;
#pragma unroll
                    for (int r = 0; r < kParts; ++r) fb[q][r][w] = sb[r * kSizeB + place];
                }
            if constexpr (kParts == 2) {
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int e = 0; e < 2 * kHalves; ++e) fa[h][1][e] *= sign_a;
#pragma unroll
                for (int q = 0; q < 4; ++q)
#pragma unroll
                    for (int w = 0; w < kHalves; ++w) fb[q][1][w] *= sign_b;
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    // the accumulators of rows 2 h and 2 h + 1, in the order of the fragment of d
                    double d[kParts][4];
#pragma unroll
                    for (int r = 0; r < kParts; ++r) {
                        d[r][0] = acc[2 * h][q][r][0];
                        d[r][1] = acc[2 * h][q][r][1];
                        d[r][2] = acc[2 * h + 1][q][r][0];
                        d[r][3] = acc[2 * h + 1][q][r][1];
                    }
                    auto mma = [&](double* to, const double* x, const double* y, double sign) {
                        if constexpr (kSteps == 8) {
                            const double xs[4] = {sign * x[0], sign * x[1], sign * x[2], sign * x[3]};
                            mma_1688(to, xs, y);
                        } else {
                            mma_1684(to, sign * x[0], sign * x[1], y[0]);
                        }
                    };
                    mma(d[0], fa[h][0], fb[q][0], 1.0);
                    if constexpr (kParts == 2) {
                        mma(d[0], fa[h][1], fb[q][1], -1.0);
                        mma(d[1], fa[h][0], fb[q][1], 1.0);
                        mma(d[1], fa[h][1], fb[q][0], 1.0);
                    }
#pragma unroll
                    for (int r = 0; r < kParts; ++r) {
                        acc[2 * h][q][r][0] = d[r][0];
                        acc[2 * h][q][r][1] = d[r][1];
                        acc[2 * h + 1][q][r][0] = d[r][2];
                        acc[2 * h + 1][q][r][1] = d[r][3];
                    }
                }
        }
#else
#pragma unroll
        for (int t = 0; t < kBK; t += 4) {
            double fa[4][kParts], fb[4][kParts];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const unsigned i = wm + p * 8 + gi, k = t + ti;
                const unsigned place = TransA ? i * kPitchA + k : k * kPitchA + i;
#pragma unroll
                for (int r = 0; r < kParts; ++r) fa[p][r] = sa[r * kSizeA + place];
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned j = wn + q * 8 + gi, k = t + ti;
                const unsigned place = TransB ? k * kPitchB + j : j * kPitchB + k;
#pragma unroll
                for (int r = 0; r < kParts; ++r) fb[q][r] = sb[r * kSizeB + place];
            }
            if constexpr (kParts == 2) {
#pragma unroll
                for (int p = 0; p < 4; ++p) fa[p][1] *= sign_a;
#pragma unroll
                for (int q = 0; q < 4; ++q) fb[q][1] *= sign_b;
            }
#pragma unroll
            for (int p = 0; p < 4; ++p)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    mma_884(acc[p][q][0][0], acc[p][q][0][1], fa[p][0], fb[q][0]);
                    if constexpr (kParts == 2) {
                        // (ar + i ai)(br + i bi): re ar br - ai bi, im ar bi + ai br
                        mma_884(acc[p][q][0][0], acc[p][q][0][1], -fa[p][1], fb[q][1]);
                        mma_884(acc[p][q][1][0], acc[p][q][1][1], fa[p][0], fb[q][1]);
                        mma_884(acc[p][q][1][0], acc[p][q][1][1], fa[p][1], fb[q][0]);
                    }
                }
        }
#endif
    }
    wait_async<0>();

#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const unsigned i = i0 + wm + p * 8 + gi, j = j0 + wn + q * 8 + 2 * ti + e;
                if (i >= m || j >= n) continue;
                S value;
                if constexpr (kParts == 2)
                    value = {acc[p][q][0][e], acc[p][q][1][e]};
                else
                    value = acc[p][q][0][e];
                if (g.segments > 1) {
                    partial[(static_cast<long long>(z) * n + j) * m + i] = value;
                } else {
                    S* to = c + i + j * g.ldc;
                    const S scaled = product(alpha, value);
                    *to = is_zero(beta) ? scaled : plus(scaled, beta, *to);
                }
            }
}

template <class S, bool TransA, bool TransB>
void launch_dmma(const GemmArgs& g, dim3 grid, S alpha, S beta, const S* a, const S* b, S* c, S* partial) {
    using F = DmmaForm<S>;
    constexpr int threads = (F::bm / 32) * (F::bn / 32) * 32;
    constexpr std::size_t bytes = dmma_shared<S, TransA, TransB>() * sizeof(double);
    auto kernel = dmma_kernel<S, TransA, TransB>;
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)),
               "the shared memory of a product");
    kernel<<<grid, threads, bytes>>>(g, alpha, beta, a, b, c, partial);
    check_cuda(cudaGetLastError(), "a product");
}

#endif  // !MDAPY_GPU_HIP
