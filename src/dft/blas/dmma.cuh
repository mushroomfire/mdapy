// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Products of double precision on the tensor cores of NVIDIA GPUs from compute
// capability 8.0 (mma.sync of f64, true FP64 in another order of its sums),
// for blas.cu, which includes it inside its namespace after GemmArgs: real
// numbers, and complex ones as four real products of their parts.
//
// A block holds BM x BN values of the result, its warps WM x WN each as
// products of 16 x 8 (8 x 8 on capability 8.0), from BK terms of the sum at a
// time in shared memory, there in Stages stages, those outside the matrices
// zeros. Each matrix is kept in shared memory as it lies in memory, the index
// it is contiguous along first, complex values as they are, rows padded so
// that the copies and the reads of the fragments meet no more conflicts of
// banks than their sizes need.

// Copy of the kernel: how the matrices come to shared memory. kCopy8: 8 bytes
// a copy, for real matrices with a column that does not start on 16 bytes;
// kCopy16: 16 bytes a copy; kCopyMaps: whole tiles through the maps (compute
// capability 9.0, columns on 16 bytes, one product).
constexpr int kCopy8 = 0, kCopy16 = 1, kCopyMaps = 2;

#if !defined(MDAPY_GPU_HIP)

// per: the blocks a multiprocessor is to hold at once, which bounds the registers of a thread
template <int BM, int BN, int BK, int Stages, int WM, int WN, int Per>
struct DmmaTiles {
    static constexpr int bm = BM, bn = BN, bk = BK, stages = Stages, wm = WM, wn = WN, per = Per;
    static constexpr int threads = (BM / WM) * (BN / WN) * 32;
    static_assert(WM % 16 == 0 && WN % 8 == 0 && BK % 4 == 0, "the warps hold whole products of 16 x 8 x 4");
    static_assert(BM * BK % threads == 0 && BN * BK % threads == 0, "the copies share the threads evenly");
};
// measured on a GH200 (tests/cpp/dft_blas_check.cu)
template <class S>
struct DmmaForm;
template <>
struct DmmaForm<double> : DmmaTiles<128, 64, 32, 2, 32, 32, 2> {
    static constexpr int parts = 1;
};
template <>
struct DmmaForm<Cx<double>> : DmmaTiles<64, 64, 16, 2, 32, 16, 2> {
    static constexpr int parts = 2;
};

// The rows of a matrix in shared memory, in values of S along the index it is
// contiguous along in memory, padded so that the fragments of a quarter of a
// warp (16-byte complex values) or a half (8-byte real ones) meet no conflicts
// of banks: 4 modulo 16 real values, 2 modulo 8 complex ones.
template <class S>
__host__ __device__ constexpr int dmma_pad() {
    return DmmaForm<S>::parts == 1 ? 4 : 2;
}
template <class S, bool Trans>
__host__ __device__ constexpr int dmma_pitch(int rows) {
    return (Trans ? DmmaForm<S>::bk : rows) + dmma_pad<S>();
}
template <class S, bool Trans>
__host__ __device__ constexpr int dmma_size(int rows) {
    return (Trans ? rows : DmmaForm<S>::bk) * dmma_pitch<S, Trans>(rows);
}
// the doubles of the shared memory of a block: a barrier of the copies of each
// stage (in 128 bytes, on which the tiles of the maps start), then its stages
// of a and b
template <class S>
__host__ __device__ constexpr int dmma_bars() {
    return 16 * ((DmmaForm<S>::stages + 15) / 16);
}
template <class S, bool TransA, bool TransB>
__host__ __device__ constexpr int dmma_shared() {
    using F = DmmaForm<S>;
    return dmma_bars<S>() + F::stages * F::parts * (dmma_size<S, TransA>(F::bm) + dmma_size<S, !TransB>(F::bn));
}

// Copies Bytes (8 or 16) from memory into shared memory, all of them or the
// first in with zeros after them.
template <int Bytes>
__device__ __forceinline__ void copy_async(double* to, const double* from) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(to));
    if constexpr (Bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(from));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(address), "l"(from));
#else
    for (unsigned e = 0; e < Bytes / 8; ++e) to[e] = from[e];
#endif
}
template <int Bytes>
__device__ __forceinline__ void copy_async(double* to, const double* from, unsigned in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(to));
    if constexpr (Bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(address), "l"(from), "r"(in));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(address), "l"(from), "r"(in));
#else
    for (unsigned e = 0; e < Bytes / 8; ++e) to[e] = 8 * e < in ? from[e] : 0.0;
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

// From compute capability 9.0 one instruction copies a whole tile of a matrix
// (cp.async.bulk.tensor), described to the GPU by a map of the matrix that
// the driver encodes; values outside the matrix arrive as zeros. A barrier in
// shared memory (mbarrier) of each stage counts the bytes that have arrived.
struct alignas(64) DmmaMap {
    unsigned long long opaque[16];  // CUtensorMap of the driver
};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void bar_init(double* bar) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(bar))));
}
__device__ __forceinline__ void bar_fence() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::); }
// the barrier completes once bytes more have arrived
__device__ __forceinline__ void bar_expect(double* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<unsigned>(__cvta_generic_to_shared(bar))),
                 "r"(bytes)
                 : "memory");
}
// waits until the barrier has completed for the time of this parity (0, 1, 0, ...)
__device__ __forceinline__ void bar_wait(double* bar, unsigned parity) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    unsigned done = 0;
    while (done == 0)
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done)
                     : "r"(address), "r"(parity)
                     : "memory");
}
// the box of the map that starts at (along, across) into shared memory
__device__ __forceinline__ void copy_box(double* to, const DmmaMap* map, int along, int across, double* bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            static_cast<unsigned>(__cvta_generic_to_shared(to))),
        "l"(map), "r"(along), "r"(across), "r"(static_cast<unsigned>(__cvta_generic_to_shared(bar)))
        : "memory");
}
#else
__device__ __forceinline__ void bar_init(double*) {}
__device__ __forceinline__ void bar_fence() {}
__device__ __forceinline__ void bar_expect(double*, unsigned) {}
__device__ __forceinline__ void bar_wait(double*, unsigned) {}
__device__ __forceinline__ void copy_box(double*, const DmmaMap*, int, int, double*) {}
#endif

// The product of an instruction: kDmmaRows x 4 by 4 x 8, the rows 16 on
// compute capability 9.0 (m16n8k4) and 8 on 8.0 (m8n8k4). The lane 4 g + t
// holds a[g + 8 v, t] in a[v], b[t, g] in b and d[g + 8 v, 2 t + e] in
// d[2 v + e]; d += a b.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
constexpr int kDmmaRows = 16;
__device__ __forceinline__ void mma(double* d, const double* a, double b) {
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                 : "+d"(d[0]), "+d"(d[1]), "+d"(d[2]), "+d"(d[3])
                 : "d"(a[0]), "d"(a[1]), "d"(b));
}
#else
constexpr int kDmmaRows = 8;
__device__ __forceinline__ void mma(double* d, const double* a, double b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
                 : "+d"(d[0]), "+d"(d[1])
                 : "d"(a[0]), "d"(b));
#else
    (void)d, (void)a, (void)b;
#endif
}
#endif

// x with its sign changed where flip is the sign bit, as it is where it is zero
__device__ __forceinline__ double flipped(double x, int flip) {
    return __hiloint2double(__double2hiint(x) ^ flip, __double2loint(x));
}

template <class S, bool TransA, bool TransB, int Copy>
__global__ void __launch_bounds__(DmmaForm<S>::threads, DmmaForm<S>::per)
    dmma_kernel(const GemmArgs g, const S alpha, const S beta, const S* __restrict__ a, const S* __restrict__ b, S* c,
                S* partial, const __grid_constant__ DmmaMap map_a, const __grid_constant__ DmmaMap map_b) {
    using F = DmmaForm<S>;
    constexpr int kBM = F::bm, kBN = F::bn, kBK = F::bk, kStages = F::stages, kParts = F::parts;
    constexpr int kWarpsM = kBM / F::wm, kThreads = F::threads;
    // the products of a warp along each index, and the halves of 8 rows of one
    constexpr int kAM = F::wm / kDmmaRows, kAN = F::wn / 8, kHalves = kDmmaRows / 8;
    // the pitch of the rows of each matrix in shared memory and the values of a stage, of S
    constexpr int kPitchA = dmma_pitch<S, TransA>(kBM), kPitchB = dmma_pitch<S, !TransB>(kBN);
    constexpr int kSizeA = dmma_size<S, TransA>(kBM), kSizeB = dmma_size<S, !TransB>(kBN);
    extern __shared__ __align__(128) double shared[];
    double* as = shared + dmma_bars<S>();        // [stage][value][part]
    double* bs = as + kStages * kParts * kSizeA;  // [stage][value][part]

    const unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const unsigned gi = lane / 4, ti = lane % 4;
    const unsigned wm = (warp % kWarpsM) * F::wm, wn = (warp / kWarpsM) * F::wn;
    const unsigned i0 = (g.swap ? blockIdx.y : blockIdx.x) * kBM, j0 = (g.swap ? blockIdx.x : blockIdx.y) * kBN;
    const unsigned z = blockIdx.z;
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
    const unsigned chunks = last > first ? (last - first + kBK - 1) / kBK : 0;

    // Each matrix is copied along the index it is contiguous along in memory,
    // kUnit values a copy (16 bytes, but for real values with kCopy8), as it
    // lies in shared memory: copy q of a thread takes the values along,
    // along + kUnit of the row across + q kStep of the tile.
    constexpr int kUnit = kParts == 1 && Copy != kCopy8 ? 2 : 1, kBytes = 8 * kParts * kUnit;
    constexpr int kLineA = (TransA ? kBK : kBM) / kUnit, kLineB = (TransB ? kBN : kBK) / kUnit;
    constexpr int kCopiesA = kBM * kBK / kUnit / kThreads, kCopiesB = kBN * kBK / kUnit / kThreads;
    constexpr int kStepA = kThreads / kLineA, kStepB = kThreads / kLineB;
    static_assert(kStepA * kLineA == kThreads && kStepB * kLineB == kThreads, "the threads copy whole rows");
    const unsigned along_a = (tid % kLineA) * kUnit, across_a = tid / kLineA;
    const unsigned along_b = (tid % kLineB) * kUnit, across_b = tid / kLineB;
    // the first copies of the thread: in memory at the term 0, and in a stage
    const double* from_a = reinterpret_cast<const double*>(
        a + (TransA ? along_a + (i0 + across_a) * g.lda : i0 + along_a + across_a * g.lda));
    const double* from_b = reinterpret_cast<const double*>(
        b + (TransB ? j0 + along_b + across_b * g.ldb : along_b + (j0 + across_b) * g.ldb));
    const long long step_a = kParts * kStepA * g.lda, step_b = kParts * kStepB * g.ldb;
    const unsigned to_a = (across_a * kPitchA + along_a) * kParts, to_b = (across_b * kPitchB + along_b) * kParts;
    const bool inside = i0 + kBM <= m && j0 + kBN <= n;
    const bool active = i0 + wm < m && j0 + wn < n;  // the warp holds values of the result
    if constexpr (Copy == kCopyMaps) {
        static_assert(kParts * kSizeA % 16 == 0 && kParts * kSizeB % 16 == 0, "the tiles of the maps start on 128 bytes");
        if (tid == 0) {
#pragma unroll
            for (int s = 0; s < kStages; ++s) bar_init(shared + s);
            bar_fence();
        }
        __syncthreads();
        // the first thread copies for all: a warp outside the result has nothing to do
        if (!active) return;
    }

    // the terms [t0, t0 + kBK) of the block's rows of op(a) and columns of op(b) into a stage
    auto load = [&](unsigned stage, unsigned t0) {
        if constexpr (Copy == kCopyMaps) {
            // the tiles with their padding, of doubles along the first index of the maps
            if (tid == 0) {
                const int term = t0, i = i0, j = j0;
                bar_expect(shared + stage, 8 * kParts * (kSizeA + kSizeB));
                copy_box(as + stage * kParts * kSizeA, &map_a, kParts * (TransA ? term : i), TransA ? i : term,
                         shared + stage);
                copy_box(bs + stage * kParts * kSizeB, &map_b, kParts * (TransB ? j : term), TransB ? term : j,
                         shared + stage);
            }
            return;
        }
        double* sa = as + stage * kParts * kSizeA + to_a;
        double* sb = bs + stage * kParts * kSizeB + to_b;
        const long long term = t0;
        const double* pa = from_a + kParts * (TransA ? term : term * g.lda);
        const double* pb = from_b + kParts * (TransB ? term * g.ldb : term);
        if (inside && t0 + kBK <= last) {
#pragma unroll
            for (int q = 0; q < kCopiesA; ++q) copy_async<kBytes>(sa + q * kStepA * kPitchA * kParts, pa + q * step_a);
#pragma unroll
            for (int q = 0; q < kCopiesB; ++q) copy_async<kBytes>(sb + q * kStepB * kPitchB * kParts, pb + q * step_b);
            return;
        }
        // at the edges: the bytes of a copy within the matrix, zeros after them
        auto bytes = [](unsigned along, unsigned across, unsigned end_along, unsigned end_across) {
            if (across >= end_across || along >= end_along) return 0u;
            return 8u * kParts * (along + kUnit <= end_along ? kUnit : end_along - along);
        };
        const unsigned terms = last - t0;
#pragma unroll
        for (int q = 0; q < kCopiesA; ++q) {
            const unsigned across = across_a + q * kStepA;
            const unsigned in = TransA ? bytes(along_a, across, terms, m - i0) : bytes(along_a, across, m - i0, terms);
            copy_async<kBytes>(sa + q * kStepA * kPitchA * kParts, in > 0 ? pa + q * step_a : from_a, in);
        }
#pragma unroll
        for (int q = 0; q < kCopiesB; ++q) {
            const unsigned across = across_b + q * kStepB;
            const unsigned in = TransB ? bytes(along_b, across, n - j0, terms) : bytes(along_b, across, terms, n - j0);
            copy_async<kBytes>(sb + q * kStepB * kPitchB * kParts, in > 0 ? pb + q * step_b : from_b, in);
        }
    };

    // The sums of a warp: [product along m][along n][part][2 half + e]. Of
    // complex numbers the real part and minus sign_b times the imaginary one,
    // with sign the sign of the imaginary part of op(a) or op(b): from
    // na = -sign_a sign_b ai and nb = -bi they are ar br + na bi and
    // ar nb + na br, four real products.
    double acc[kAM][kAN][kParts][2 * kHalves];
#pragma unroll
    for (int h = 0; h < kAM; ++h)
#pragma unroll
        for (int q = 0; q < kAN; ++q)
#pragma unroll
            for (int r = 0; r < kParts; ++r)
#pragma unroll
                for (int e = 0; e < 2 * kHalves; ++e) acc[h][q][r][e] = 0.0;
    const int flip_a = g.conj_a == g.conj_b ? static_cast<int>(0x80000000u) : 0;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (static_cast<unsigned>(s) < chunks) load(s, first + s * kBK);
        commit_async();
    }
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        if constexpr (Copy == kCopyMaps) {
            bar_wait(shared + chunk % kStages, (chunk / kStages) & 1);
        } else {
            wait_async<kStages - 2>();
        }
        __syncthreads();
        // the stage computed last is free: the next chunk into it
        const unsigned next = chunk + kStages - 1;
        if (next < chunks) load(next % kStages, first + next * kBK);
        commit_async();
        if (Copy != kCopyMaps && !active) continue;
        const double* sa = as + (chunk % kStages) * kParts * kSizeA;
        const double* sb = bs + (chunk % kStages) * kParts * kSizeB;
#pragma unroll
        for (int t = 0; t < kBK; t += 4) {
            // fa: ar, na; fb: br, bi, nb
            double fa[kAM][kParts][kHalves], fb[kAN][2 * kParts - 1];
#pragma unroll
            for (int h = 0; h < kAM; ++h)
#pragma unroll
                for (int v = 0; v < kHalves; ++v) {
                    const unsigned i = wm + h * kDmmaRows + v * 8 + gi, k = t + ti;
                    const unsigned place = TransA ? i * kPitchA + k : k * kPitchA + i;
                    if constexpr (kParts == 2) {
                        const double2 value = reinterpret_cast<const double2*>(sa)[place];
                        fa[h][0][v] = value.x;
                        fa[h][1][v] = flipped(value.y, flip_a);
                    } else {
                        fa[h][0][v] = sa[place];
                    }
                }
#pragma unroll
            for (int q = 0; q < kAN; ++q) {
                const unsigned j = wn + q * 8 + gi, k = t + ti;
                const unsigned place = TransB ? k * kPitchB + j : j * kPitchB + k;
                if constexpr (kParts == 2) {
                    const double2 value = reinterpret_cast<const double2*>(sb)[place];
                    fb[q][0] = value.x;
                    fb[q][1] = value.y;
                    fb[q][2] = -value.y;
                } else {
                    fb[q][0] = sb[place];
                }
            }
            // every sum once in each pass: a product does not wait for the one before it
#pragma unroll
            for (int pass = 0; pass < kParts * kParts; ++pass)
#pragma unroll
                for (int h = 0; h < kAM; ++h)
#pragma unroll
                    for (int q = 0; q < kAN; ++q) {
                        if (pass == 0) mma(acc[h][q][0], fa[h][0], fb[q][0]);
                        if constexpr (kParts == 2) {
                            if (pass == 1) mma(acc[h][q][1], fa[h][0], fb[q][2]);
                            if (pass == 2) mma(acc[h][q][0], fa[h][1], fb[q][1]);
                            if (pass == 3) mma(acc[h][q][1], fa[h][1], fb[q][0]);
                        }
                    }
        }
    }
    wait_async<0>();

#pragma unroll
    for (int h = 0; h < kAM; ++h)
#pragma unroll
        for (int q = 0; q < kAN; ++q)
#pragma unroll
            for (int e = 0; e < 2 * kHalves; ++e) {
                const unsigned i = i0 + wm + h * kDmmaRows + (e / 2) * 8 + gi, j = j0 + wn + q * 8 + 2 * ti + e % 2;
                if (i >= m || j >= n) continue;
                S value;
                if constexpr (kParts == 2)
                    value = {acc[h][q][0][e], g.conj_b ? acc[h][q][1][e] : -acc[h][q][1][e]};
                else
                    value = acc[h][q][0][e];
                if (g.segments > 1) {
                    partial[(static_cast<long long>(z) * n + j) * m + i] = value;
                } else {
                    S* to = c + i + j * g.ldc;
                    const S scaled = product(alpha, value);
                    *to = is_zero(beta) ? scaled : plus(scaled, beta, *to);
                }
            }
}

// The map of a matrix of doubles, along values in each of its across columns
// stride bytes apart, for boxes of box_along by box_across; false where the
// driver has none.
inline bool dmma_map(DmmaMap* map, const void* address, long long along, long long across, long long stride,
                     int box_along, int box_across) {
#if defined(__linux__)
    using Encode = int (*)(DmmaMap*, int, unsigned, void*, const unsigned long long*, const unsigned long long*,
                           const unsigned*, const unsigned*, int, int, int, int);
    static const Encode encode = [] {
        void* driver = dlopen("libcuda.so.1", RTLD_LAZY);
        return driver == nullptr ? nullptr : reinterpret_cast<Encode>(dlsym(driver, "cuTensorMapEncodeTiled"));
    }();
    if (encode == nullptr) return false;
    const unsigned long long dims[2] = {static_cast<unsigned long long>(along), static_cast<unsigned long long>(across)};
    const unsigned long long strides[1] = {static_cast<unsigned long long>(stride)};
    const unsigned box[2] = {static_cast<unsigned>(box_along), static_cast<unsigned>(box_across)}, steps[2] = {1, 1};
    constexpr int kFloat64 = 8;  // CU_TENSOR_MAP_DATA_TYPE_FLOAT64; no interleave, swizzle or promotion, zeros outside
    return encode(map, kFloat64, 2, const_cast<void*>(address), dims, strides, box, steps, 0, 0, 0, 0) == 0;
#else
    (void)map, (void)address, (void)along, (void)across, (void)stride, (void)box_along, (void)box_across;
    return false;
#endif
}

// false: the maps could not be made, and nothing ran
template <class S, bool TransA, bool TransB, int Copy>
bool launch_dmma(const GemmArgs& g, dim3 grid, S alpha, S beta, const S* a, const S* b, S* c, S* partial) {
    using F = DmmaForm<S>;
    constexpr int threads = F::threads, parts = F::parts;
    constexpr std::size_t bytes = dmma_shared<S, TransA, TransB>() * sizeof(double);
    DmmaMap map_a{}, map_b{};
    if constexpr (Copy == kCopyMaps) {
        constexpr int pitch_a = parts * dmma_pitch<S, TransA>(F::bm), pitch_b = parts * dmma_pitch<S, !TransB>(F::bn);
        static_assert(pitch_a <= 256 && pitch_b <= 256, "the boxes of the maps hold 256 values along an index");
        const long long m = g.m, n = g.n, k = g.k, wide = 8 * parts;
        if (!dmma_map(&map_a, a, parts * (TransA ? k : m), TransA ? m : k, wide * g.lda, pitch_a, TransA ? F::bm : F::bk) ||
            !dmma_map(&map_b, b, parts * (TransB ? n : k), TransB ? k : n, wide * g.ldb, pitch_b, TransB ? F::bk : F::bn))
            return false;
    }
    auto kernel = dmma_kernel<S, TransA, TransB, Copy>;
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)),
               "the shared memory of a product");
    kernel<<<grid, threads, bytes>>>(g, alpha, beta, a, b, c, partial, map_a, map_b);
    check_cuda(cudaGetLastError(), "a product");
    return true;
}

#endif  // !MDAPY_GPU_HIP
