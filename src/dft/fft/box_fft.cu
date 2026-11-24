// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The transform of one axis, in a block: the lines are loaded into shared
// memory in their natural order, transformed in place by decimation in
// frequency (Gentleman-Sande), a pass per radix, and stored through the order
// the passes leave them in. A pass of radix r on sub-sequences of length L = r m
// replaces x[j + p m], p < r, by
//     y_q = (sum_p x[j + p m] w_r^(p q)) w_L^(j q),   w_k = exp(s 2 pi i / k),
// at j + q m, after which sub-sequence q holds the components q + r f' of the
// sequence; the last pass (m = 1) has no roots to multiply by.
//
// Shared memory of a block: for the strided axes (the first and the second)
// the lines are neighbouring columns, point j of line c at j C + c, as they
// lie in the box, so that loads and stores are coalesced; for the contiguous
// axis (the third) line c starts at c n', n' = n | 1 odd so that the threads of
// a group, which work on the same point of different lines, do not share a
// bank.
//
// Nothing here depends on the size of a warp or of a wavefront: the threads of
// a block share the work by their index, and meet at barriers of the block.

#include "dft/fft/box_fft.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/portable.cuh"

namespace mdapy::dft {

namespace {

constexpr int kMaxPasses = 12;
constexpr std::size_t kMaxShared = 40 * 1024;  // bytes of shared memory of a block
constexpr int kAhead = 4;  // loads of a thread on their way at a time
constexpr int kStrided = 0, kContiguous = 1, kRealToComplex = 2, kComplexToReal = 3;

template <class R>
struct Cx {
    R x, y;
};

struct Pass {
    int radix;       // r
    int m;           // L / r
    int length;      // L
    int step;        // n / L: w_L^k is root k step of the n
    unsigned inv_m;  // ceil(2^32 / m), for m > 1
};

struct AxisArgs {
    int n;              // points of a line
    int half;           // n / 2 + 1
    int passes;
    int log_c;          // lines of a block, as a power of two
    int sj, sc;         // strides in shared memory of the points of a line and of the lines
    unsigned inv_n;     // ceil(2^32 / n)
    unsigned inv_half;  // ceil(2^32 / half)
    int columns;        // strided: lines of a plane
    int chunks;         // strided: blocks of a plane
    long long stride;   // strided: between the points of a line
    long long plane;    // strided: points of a plane, n columns
    long long lines;    // contiguous: all the lines
    Pass pass[kMaxPasses];
};

// cos and sin of 2 pi k / p for p = 3, 5, 7
__host__ __device__ constexpr double unit_cos(int p, int k) {
    return p == 3   ? (k == 0 ? 1.0 : -0.5)
           : p == 5 ? (k == 0 ? 1.0 : (k == 1 || k == 4) ? 0.3090169943749474241022934 : -0.8090169943749474241022934)
                    : (k == 0               ? 1.0
                       : (k == 1 || k == 6) ? 0.6234898018587335305250049
                       : (k == 2 || k == 5) ? -0.2225209339563144042889026
                                            : -0.9009688679024191262361023);
}

__host__ __device__ constexpr double unit_sin(int p, int k) {
    return p == 3   ? (k == 0 ? 0.0 : k == 1 ? 0.8660254037844386467637232 : -0.8660254037844386467637232)
           : p == 5 ? (k == 0   ? 0.0
                       : k == 1 ? 0.9510565162951535721164393
                       : k == 2 ? 0.5877852522924731291687059
                       : k == 3 ? -0.5877852522924731291687059
                                : -0.9510565162951535721164393)
                    : (k == 0   ? 0.0
                       : k == 1 ? 0.7818314824680298087084445
                       : k == 2 ? 0.9749279121818236070181317
                       : k == 3 ? 0.4338837391175581204757683
                       : k == 4 ? -0.4338837391175581204757683
                       : k == 5 ? -0.9749279121818236070181317
                                : -0.7818314824680298087084445);
}

template <class R>
__device__ __forceinline__ Cx<R> add(const Cx<R> a, const Cx<R> b) {
    return {a.x + b.x, a.y + b.y};
}

template <class R>
__device__ __forceinline__ Cx<R> sub(const Cx<R> a, const Cx<R> b) {
    return {a.x - b.x, a.y - b.y};
}

template <class R>
__device__ __forceinline__ Cx<R> mul(const Cx<R> a, const Cx<R> b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

// a times s i
template <class R, int Sign>
__device__ __forceinline__ Cx<R> times_i(const Cx<R> a) {
    return Sign > 0 ? Cx<R>{-a.y, a.x} : Cx<R>{a.y, -a.x};
}

// y_q = sum_p x_p w_4^(p q) of four values
template <class R, int Sign>
__device__ __forceinline__ void four(Cx<R>& x0, Cx<R>& x1, Cx<R>& x2, Cx<R>& x3) {
    const Cx<R> a = add(x0, x2), b = sub(x0, x2), c = add(x1, x3), d = times_i<R, Sign>(sub(x1, x3));
    x0 = add(a, c);
    x1 = add(b, d);
    x2 = sub(a, c);
    x3 = sub(b, d);
}

// y_q = sum_p x_p w_r^(p q), in place
template <class R, int Sign, int Radix>
__device__ __forceinline__ void butterfly(Cx<R>* x) {
    if (Radix == 2) {
        const Cx<R> a = x[0];
        x[0] = add(a, x[1]);
        x[1] = sub(a, x[1]);
    } else if (Radix == 4) {
        four<R, Sign>(x[0], x[1], x[2], x[3]);
    } else if (Radix == 8) {
        // the even components from x_p + x_(p + 4), the odd ones from
        // (x_p - x_(p + 4)) w_8^p
        const R h = static_cast<R>(0.7071067811865475244008444);
        Cx<R> a0 = add(x[0], x[4]), a1 = add(x[1], x[5]), a2 = add(x[2], x[6]), a3 = add(x[3], x[7]);
        Cx<R> b0 = sub(x[0], x[4]);
        const Cx<R> d1 = sub(x[1], x[5]), d3 = sub(x[3], x[7]);
        Cx<R> b2 = times_i<R, Sign>(sub(x[2], x[6]));
        const R s = static_cast<R>(Sign);
        Cx<R> b1 = {h * (d1.x - s * d1.y), h * (s * d1.x + d1.y)};
        Cx<R> b3 = {h * (-d3.x - s * d3.y), h * (s * d3.x - d3.y)};
        four<R, Sign>(a0, a1, a2, a3);
        four<R, Sign>(b0, b1, b2, b3);
        x[0] = a0;
        x[1] = b0;
        x[2] = a1;
        x[3] = b1;
        x[4] = a2;
        x[5] = b2;
        x[6] = a3;
        x[7] = b3;
    } else {
        // an odd prime: with a_p = x_p + x_(r - p) and b_p = x_p - x_(r - p),
        // y_q = x_0 + sum_p a_p cos(2 pi p q / r) + s i sum_p b_p sin(2 pi p q / r)
        // and y_(r - q) its partner with the other sign
        constexpr int half = (Radix - 1) / 2;
        Cx<R> a[half > 0 ? half : 1], b[half > 0 ? half : 1];
#pragma unroll
        for (int p = 1; p <= half; ++p) {
            a[p - 1] = add(x[p], x[Radix - p]);
            b[p - 1] = sub(x[p], x[Radix - p]);
        }
        const Cx<R> x0 = x[0];
        Cx<R> sum = x0;
#pragma unroll
        for (int p = 0; p < half; ++p) sum = add(sum, a[p]);
        x[0] = sum;
#pragma unroll
        for (int q = 1; q <= half; ++q) {
            Cx<R> even = x0, odd = {static_cast<R>(0), static_cast<R>(0)};
#pragma unroll
            for (int p = 1; p <= half; ++p) {
                const R c = static_cast<R>(unit_cos(Radix, (p * q) % Radix));
                const R s = static_cast<R>(unit_sin(Radix, (p * q) % Radix));
                even.x += c * a[p - 1].x;
                even.y += c * a[p - 1].y;
                odd.x += s * b[p - 1].x;
                odd.y += s * b[p - 1].y;
            }
            const Cx<R> turned = times_i<R, Sign>(odd);
            x[q] = add(even, turned);
            x[Radix - q] = sub(even, turned);
        }
    }
}

// One pass over the lines of a block.
template <class R, int Sign, int Radix>
__device__ __forceinline__ void run_pass(Cx<R>* shared, const AxisArgs& a, const Pass ps, const Cx<R>* __restrict__ roots,
                                         unsigned valid) {
    const unsigned mask = (1u << a.log_c) - 1u;
    const unsigned items = static_cast<unsigned>(a.n / Radix) << a.log_c;
    const int d = ps.m * a.sj;
    for (unsigned w = threadIdx.x; w < items; w += blockDim.x) {
        const unsigned c = w & mask;
        if (c >= valid) continue;
        const unsigned u = w >> a.log_c;
        unsigned k = u, j = 0;
        if (ps.m > 1) {
            k = __umulhi(u, ps.inv_m);
            j = u - k * static_cast<unsigned>(ps.m);
        }
        Cx<R>* at = shared + (k * static_cast<unsigned>(ps.length) + j) * static_cast<unsigned>(a.sj) +
                    c * static_cast<unsigned>(a.sc);
        Cx<R> x[Radix];
#pragma unroll
        for (int q = 0; q < Radix; ++q) x[q] = at[q * d];
        butterfly<R, Sign, Radix>(x);
        at[0] = x[0];
        if (ps.m > 1) {
            const unsigned t = j * static_cast<unsigned>(ps.step);
#pragma unroll
            for (int q = 1; q < Radix; ++q) {
                Cx<R> root = roots[q * t];
                if (Sign > 0) root.y = -root.y;
                at[q * d] = mul(x[q], root);
            }
        } else {
#pragma unroll
            for (int q = 1; q < Radix; ++q) at[q * d] = x[q];
        }
    }
}

// The transform of one axis of the lines of a block. roots: exp(-2 pi i k / n),
// k < n. source[f]: the place in a line of component f after the passes.
// Mode kStrided and kContiguous: complex lines in place (in is out).
// kRealToComplex: n reals a line to n / 2 + 1 components. kComplexToReal: back.
template <class R, int Sign, int Mode>
__global__ void axis_kernel(const AxisArgs a, const Cx<R>* __restrict__ roots, const int* __restrict__ source,
                            const R* in, R* out) {
    extern __shared__ double aligned_shared[];
    Cx<R>* shared = reinterpret_cast<Cx<R>*>(aligned_shared);
    const unsigned lines = 1u << a.log_c, mask = lines - 1u;
    const unsigned n = static_cast<unsigned>(a.n), half = static_cast<unsigned>(a.half);
    const unsigned sc = static_cast<unsigned>(a.sc);
    unsigned valid = 0;
    long long first = 0;  // of the block: its first point (strided) or line (contiguous)
    if (Mode == kStrided) {
        const unsigned plane = blockIdx.x / static_cast<unsigned>(a.chunks);
        const unsigned chunk = blockIdx.x - plane * static_cast<unsigned>(a.chunks);
        const unsigned left = static_cast<unsigned>(a.columns) - chunk * lines;
        valid = left < lines ? left : lines;
        first = plane * a.plane + chunk * lines;
        const Cx<R>* from = reinterpret_cast<const Cx<R>*>(in) + first;
        // kAhead loads of a thread are on their way at a time
        if ((threadIdx.x & mask) < valid)
            for (unsigned e0 = threadIdx.x; e0 < n * lines; e0 += kAhead * blockDim.x) {
                Cx<R> value[kAhead];
#pragma unroll
                for (int i = 0; i < kAhead; ++i) {
                    const unsigned e = e0 + i * blockDim.x;
                    if (e < n * lines) value[i] = from[(e >> a.log_c) * a.stride + (e & mask)];
                }
#pragma unroll
                for (int i = 0; i < kAhead; ++i) {
                    const unsigned e = e0 + i * blockDim.x;
                    if (e < n * lines) shared[e] = value[i];
                }
            }
    } else {
        first = static_cast<long long>(blockIdx.x) * lines;
        const long long left = a.lines - first;
        valid = left < static_cast<long long>(lines) ? static_cast<unsigned>(left) : lines;
        if (Mode == kContiguous) {
            const Cx<R>* from = reinterpret_cast<const Cx<R>*>(in) + first * n;
            for (unsigned e0 = threadIdx.x; e0 < valid * n; e0 += kAhead * blockDim.x) {
                Cx<R> value[kAhead];
#pragma unroll
                for (int i = 0; i < kAhead; ++i) {
                    const unsigned e = e0 + i * blockDim.x;
                    if (e < valid * n) value[i] = from[e];
                }
#pragma unroll
                for (int i = 0; i < kAhead; ++i) {
                    const unsigned e = e0 + i * blockDim.x;
                    if (e < valid * n) {
                        const unsigned c = __umulhi(e, a.inv_n), j = e - c * n;
                        shared[c * sc + j] = value[i];
                    }
                }
            }
        } else if (Mode == kRealToComplex) {
            const R* from = in + first * n;
            for (unsigned e0 = threadIdx.x; e0 < valid * n; e0 += 2 * kAhead * blockDim.x) {
                R value[2 * kAhead];
#pragma unroll
                for (int i = 0; i < 2 * kAhead; ++i) {
                    const unsigned e = e0 + i * blockDim.x;
                    if (e < valid * n) value[i] = from[e];
                }
#pragma unroll
                for (int i = 0; i < 2 * kAhead; ++i) {
                    const unsigned e = e0 + i * blockDim.x;
                    if (e < valid * n) {
                        const unsigned c = __umulhi(e, a.inv_n), j = e - c * n;
                        shared[c * sc + j] = {value[i], static_cast<R>(0)};
                    }
                }
            }
        } else {
            // the components n - f are the conjugates of the components f
            const Cx<R>* from = reinterpret_cast<const Cx<R>*>(in) + first * half;
            for (unsigned e0 = threadIdx.x; e0 < valid * half; e0 += kAhead * blockDim.x) {
                Cx<R> value[kAhead];
#pragma unroll
                for (int i = 0; i < kAhead; ++i) {
                    const unsigned e = e0 + i * blockDim.x;
                    if (e < valid * half) value[i] = from[e];
                }
#pragma unroll
                for (int i = 0; i < kAhead; ++i) {
                    const unsigned e = e0 + i * blockDim.x;
                    if (e < valid * half) {
                        const unsigned c = __umulhi(e, a.inv_half), f = e - c * half;
                        shared[c * sc + f] = value[i];
                        if (f > 0 && 2 * f < n) shared[c * sc + n - f] = {value[i].x, -value[i].y};
                    }
                }
            }
        }
    }
    for (int p = 0; p < a.passes; ++p) {
        __syncthreads();
        const Pass ps = a.pass[p];
        switch (ps.radix) {
            case 2: run_pass<R, Sign, 2>(shared, a, ps, roots, valid); break;
            case 3: run_pass<R, Sign, 3>(shared, a, ps, roots, valid); break;
            case 4: run_pass<R, Sign, 4>(shared, a, ps, roots, valid); break;
            case 5: run_pass<R, Sign, 5>(shared, a, ps, roots, valid); break;
            case 7: run_pass<R, Sign, 7>(shared, a, ps, roots, valid); break;
            default: run_pass<R, Sign, 8>(shared, a, ps, roots, valid); break;
        }
    }
    __syncthreads();
    if (Mode == kStrided) {
        Cx<R>* to = reinterpret_cast<Cx<R>*>(out) + first;
        for (unsigned e = threadIdx.x; e < n * lines; e += blockDim.x) {
            const unsigned c = e & mask, f = e >> a.log_c;
            if (c < valid) to[f * a.stride + c] = shared[(static_cast<unsigned>(source[f]) << a.log_c) + c];
        }
    } else if (Mode == kContiguous) {
        Cx<R>* to = reinterpret_cast<Cx<R>*>(out) + first * n;
        for (unsigned e = threadIdx.x; e < valid * n; e += blockDim.x) {
            const unsigned c = __umulhi(e, a.inv_n), f = e - c * n;
            to[e] = shared[c * sc + static_cast<unsigned>(source[f])];
        }
    } else if (Mode == kRealToComplex) {
        Cx<R>* to = reinterpret_cast<Cx<R>*>(out) + first * half;
        for (unsigned e = threadIdx.x; e < valid * half; e += blockDim.x) {
            const unsigned c = __umulhi(e, a.inv_half), f = e - c * half;
            to[e] = shared[c * sc + static_cast<unsigned>(source[f])];
        }
    } else {
        R* to = out + first * n;
        for (unsigned e = threadIdx.x; e < valid * n; e += blockDim.x) {
            const unsigned c = __umulhi(e, a.inv_n), j = e - c * n;
            to[e] = shared[c * sc + static_cast<unsigned>(source[j])].x;
        }
    }
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

// The radices of the passes of a line of n points, the largest last (the pass
// that multiplies by no root); empty if n has another prime than 2, 3, 5, 7.
std::vector<int> radices(int n) {
    int twos = 0, counts[3] = {0, 0, 0};
    const int primes[3] = {3, 5, 7};
    while (n % 2 == 0) n /= 2, ++twos;
    for (int i = 0; i < 3; ++i)
        while (n % primes[i] == 0) n /= primes[i], ++counts[i];
    if (n != 1) return {};
    int eights = 0, fours = 0, two = 0;
    while (twos >= 3 && twos != 4) twos -= 3, ++eights;
    fours = twos / 2;
    two = twos % 2;
    std::vector<int> out;
    out.insert(out.end(), two, 2);
    out.insert(out.end(), counts[0], 3);
    out.insert(out.end(), fours, 4);
    out.insert(out.end(), counts[1], 5);
    out.insert(out.end(), counts[2], 7);
    out.insert(out.end(), eights, 8);
    return out;
}

unsigned inverse_of(int n) {  // ceil(2^32 / n), n > 1
    return static_cast<unsigned>(((1ull << 32) + static_cast<unsigned long long>(n) - 1) / static_cast<unsigned long long>(n));
}

template <class T>
T* uploaded(const std::vector<T>& host) {
    void* memory = nullptr;
    check(cudaMalloc(&memory, host.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(memory, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return static_cast<T*>(memory);
}

}  // namespace

struct BoxFft::Tables {
    int n = 0;
    std::vector<Pass> passes;
    Cx<double>* roots = nullptr;
    Cx<float>* roots_single = nullptr;
    int* source = nullptr;

    explicit Tables(int size) : n(size) {
        const std::vector<int> r = radices(n);
        int length = n;
        for (const int radix : r) {
            Pass pass{};
            pass.radix = radix;
            pass.m = length / radix;
            pass.length = length;
            pass.step = n / length;
            pass.inv_m = pass.m > 1 ? inverse_of(pass.m) : 0u;
            passes.push_back(pass);
            length = pass.m;
        }
        std::vector<Cx<double>> w(static_cast<std::size_t>(n));
        std::vector<Cx<float>> ws(static_cast<std::size_t>(n));
        const long double two_pi = 6.283185307179586476925286766559005768L;
        for (int k = 0; k < n; ++k) {
            const long double angle = two_pi * static_cast<long double>(k) / static_cast<long double>(n);
            w[k] = {static_cast<double>(std::cos(angle)), static_cast<double>(-std::sin(angle))};
            ws[k] = {static_cast<float>(std::cos(angle)), static_cast<float>(-std::sin(angle))};
        }
        // place p = d_1 n / r_1 + d_2 n / (r_1 r_2) + ... holds component
        // f = d_1 + r_1 (d_2 + r_2 (...))
        std::vector<int> src(static_cast<std::size_t>(n));
        for (int p = 0; p < n; ++p) {
            int f = 0, weight = 1, rest = p, len = n;
            for (const int radix : r) {
                len /= radix;
                f += weight * (rest / len);
                rest %= len;
                weight *= radix;
            }
            src[f] = p;
        }
        roots = uploaded(w);
        roots_single = uploaded(ws);
        source = uploaded(src);
    }
    ~Tables() {
        cudaFree(roots);
        cudaFree(roots_single);
        cudaFree(source);
    }
    Tables(const Tables&) = delete;
    Tables& operator=(const Tables&) = delete;

    const Cx<double>* roots_of(double) const { return roots; }
    const Cx<float>* roots_of(float) const { return roots_single; }

    AxisArgs args() const {
        AxisArgs a{};
        a.n = n;
        a.half = n / 2 + 1;
        a.passes = static_cast<int>(passes.size());
        a.inv_n = n > 1 ? inverse_of(n) : 0u;
        a.inv_half = a.half > 1 ? inverse_of(a.half) : 0u;
        for (std::size_t i = 0; i < passes.size(); ++i) a.pass[i] = passes[i];
        return a;
    }
};

bool BoxFft::supported(const std::array<int, 3>& grid) noexcept {
    for (const int n : grid) {
        if (n < 2) return false;
        const std::vector<int> r = radices(n);
        if (r.empty() || r.size() > static_cast<std::size_t>(kMaxPasses)) return false;
        if (static_cast<std::size_t>(n | 1) * sizeof(Cx<double>) > kMaxShared) return false;
    }
    return true;
}

BoxFft::BoxFft(const std::array<int, 3>& grid) : grid_(grid) {
    if (!supported(grid)) throw std::invalid_argument("the sizes of a box must be products of 2, 3, 5 and 7");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j)
            if (grid[j] == grid[i]) tables_[i] = tables_[j];
        if (tables_[i] == nullptr) tables_[i] = new Tables(grid[i]);
    }
}

BoxFft::~BoxFft() {
    for (int i = 0; i < 3; ++i) {
        bool shared = false;
        for (int j = 0; j < i; ++j) shared = shared || tables_[j] == tables_[i];
        if (!shared) delete tables_[i];
    }
}

std::size_t BoxFft::points() const noexcept {
    return static_cast<std::size_t>(grid_[0]) * grid_[1] * grid_[2];
}

std::size_t BoxFft::spectrum() const noexcept {
    return static_cast<std::size_t>(grid_[0]) * grid_[1] * (grid_[2] / 2 + 1);
}

void BoxFft::tune(int precision, int kind, int log_lines, int threads) {
    tuned_[precision][kind][0] = log_lines;
    tuned_[precision][kind][1] = threads;
}

namespace {

// Lines of a block as a power of two and its threads, for lines of n points of
// bytes each in shared memory, at least floor lines when they fit.
void geometry(int n, std::size_t bytes, int kind, const int* tuned, int& log_c, int& threads) {
    const std::size_t line = static_cast<std::size_t>(kind == kStrided ? n : (n | 1)) * bytes;
    log_c = 0;
    if (tuned[0] > 0) {
        log_c = tuned[0];
    } else {
        // strided: 128 bytes of each row at least, the line of the cache
        const std::size_t target = 16 * 1024;
        while (log_c < 6 && (line << (log_c + 1)) <= target) ++log_c;
        if (kind == kStrided)
            while ((bytes << log_c) < 128 && (line << (log_c + 1)) <= kMaxShared) ++log_c;
    }
    while (log_c > 0 && (line << log_c) > kMaxShared) --log_c;
    if (tuned[1] > 0) {
        threads = tuned[1];
    } else {
        const long long work = (static_cast<long long>(n) << log_c) / 4;
        threads = static_cast<int>(64 * ((work + 63) / 64));
        threads = threads < 64 ? 64 : threads > 256 ? 256 : threads;
    }
}

template <class R, int Sign, int Mode>
void launch(const AxisArgs& a, unsigned blocks, int threads, const Cx<R>* roots, const int* source, const R* in,
            R* out) {
    const std::size_t line = static_cast<std::size_t>(Mode == kStrided ? a.n : (a.n | 1)) * sizeof(Cx<R>);
    axis_kernel<R, Sign, Mode><<<blocks, threads, line << a.log_c>>>(a, roots, source, in, out);
    check(cudaGetLastError(), "the transform of an axis");
}

}  // namespace

// The lines along an axis whose points are stride apart, columns of them side
// by side in each of the planes.
template <class R, int Sign>
void BoxFft::strided(R* boxes, int axis, long long planes, long long columns, long long stride) {
    const Tables& t = *tables_[axis];
    AxisArgs a = t.args();
    int threads = 0;
    geometry(t.n, sizeof(Cx<R>), kStrided, tuned_[sizeof(R) == 4][kStrided], a.log_c, threads);
    a.sj = 1 << a.log_c;
    a.sc = 1;
    a.columns = static_cast<int>(columns);
    a.chunks = static_cast<int>((columns + (1 << a.log_c) - 1) >> a.log_c);
    a.stride = stride;
    a.plane = stride * t.n;
    if (axes_ & 8) a.passes = 0;
    launch<R, Sign, kStrided>(a, static_cast<unsigned>(planes * a.chunks), threads, t.roots_of(R{}), t.source, boxes,
                              boxes);
}

template <class R>
void BoxFft::complex_boxes(R* boxes, int count, int sign) {
    if (count < 1) return;
    const long long n1 = grid_[0], n2 = grid_[1], n3 = grid_[2];
    const Tables& t = *tables_[2];
    AxisArgs a = t.args();
    int threads = 0;
    geometry(t.n, sizeof(Cx<R>), kContiguous, tuned_[sizeof(R) == 4][kContiguous], a.log_c, threads);
    a.sj = 1;
    a.sc = t.n | 1;
    a.lines = count * n1 * n2;
    if (axes_ & 8) a.passes = 0;
    const unsigned blocks = static_cast<unsigned>((a.lines + (1 << a.log_c) - 1) >> a.log_c);
    if (sign < 0) {
        if (axes_ & 4) launch<R, -1, kContiguous>(a, blocks, threads, t.roots_of(R{}), t.source, boxes, boxes);
        if (axes_ & 2) strided<R, -1>(boxes, 1, count * n1, n3, n3);
        if (axes_ & 1) strided<R, -1>(boxes, 0, count, n2 * n3, n2 * n3);
    } else {
        if (axes_ & 4) launch<R, 1, kContiguous>(a, blocks, threads, t.roots_of(R{}), t.source, boxes, boxes);
        if (axes_ & 2) strided<R, 1>(boxes, 1, count * n1, n3, n3);
        if (axes_ & 1) strided<R, 1>(boxes, 0, count, n2 * n3, n2 * n3);
    }
}

template <class R>
void BoxFft::real_forward(const R* values, R* spectrum, int count) {
    if (count < 1) return;
    const long long n1 = grid_[0], n2 = grid_[1], half = grid_[2] / 2 + 1;
    const Tables& t = *tables_[2];
    AxisArgs a = t.args();
    int threads = 0;
    geometry(t.n, sizeof(Cx<R>), kContiguous, tuned_[sizeof(R) == 4][kContiguous], a.log_c, threads);
    a.sj = 1;
    a.sc = t.n | 1;
    a.lines = count * n1 * n2;
    const unsigned blocks = static_cast<unsigned>((a.lines + (1 << a.log_c) - 1) >> a.log_c);
    launch<R, -1, kRealToComplex>(a, blocks, threads, t.roots_of(R{}), t.source, values, spectrum);
    strided<R, -1>(spectrum, 1, count * n1, half, half);
    strided<R, -1>(spectrum, 0, count, n2 * half, n2 * half);
}

template <class R>
void BoxFft::real_inverse(R* spectrum, R* values, int count) {
    if (count < 1) return;
    const long long n1 = grid_[0], n2 = grid_[1], half = grid_[2] / 2 + 1;
    strided<R, 1>(spectrum, 0, count, n2 * half, n2 * half);
    strided<R, 1>(spectrum, 1, count * n1, half, half);
    const Tables& t = *tables_[2];
    AxisArgs a = t.args();
    int threads = 0;
    geometry(t.n, sizeof(Cx<R>), kContiguous, tuned_[sizeof(R) == 4][kContiguous], a.log_c, threads);
    a.sj = 1;
    a.sc = t.n | 1;
    a.lines = count * n1 * n2;
    const unsigned blocks = static_cast<unsigned>((a.lines + (1 << a.log_c) - 1) >> a.log_c);
    launch<R, 1, kComplexToReal>(a, blocks, threads, t.roots_of(R{}), t.source, spectrum, values);
}

void BoxFft::forward(double* boxes, int count) { complex_boxes(boxes, count, -1); }
void BoxFft::inverse(double* boxes, int count) { complex_boxes(boxes, count, 1); }
void BoxFft::forward(float* boxes, int count) { complex_boxes(boxes, count, -1); }
void BoxFft::inverse(float* boxes, int count) { complex_boxes(boxes, count, 1); }
void BoxFft::forward(const double* values, double* spectrum, int count) { real_forward(values, spectrum, count); }
void BoxFft::inverse(double* spectrum, double* values, int count) { real_inverse(spectrum, values, count); }
void BoxFft::forward(const float* values, float* spectrum, int count) { real_forward(values, spectrum, count); }
void BoxFft::inverse(float* spectrum, float* values, int count) { real_inverse(spectrum, values, count); }

}  // namespace mdapy::dft
