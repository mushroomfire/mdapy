// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The linear algebra of the DFT with MDAPY's kernels against the library of
// the vendor (cuBLAS), on its own (no Python): results and times. Build and run
// with tests/cpp/dft_blas_check.sh.
//
//   ./dft_blas_check check             every routine, type and transposition on a set of shapes
//   ./dft_blas_check bench [repeats]   the products of the shapes of the DFT
//   ./dft_blas_check large [repeats]   larger products, of cells of hundreds of atoms
//   ./dft_blas_check time TYPE OPA OPB m n k [repeats]   one product; TYPE z c d s, OP N T C
//   ./dft_blas_check solve TYPE m n [repeats]             one triangular solve of m rows and n columns
//
// The environment variable SMALL_TILES has the portable products of double
// precision take one of their forms whatever the result and the GPU: the large
// blocks (0), the small ones (1), those of two (2) or three (3) to a
// multiprocessor. SEGMENTS has the timed products split their sums into at
// most so many segments (1: not at all).
//
// Errors are the largest |ours - vendor's| over the largest |vendor's| of a result.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "dft/blas/blas.h"

using mdapy::dft::Blas;
using mdapy::dft::Complex;
using mdapy::dft::Kernels;
using mdapy::dft::Op;

namespace {

void ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// the reals a scalar is made of
template <class T>
struct Parts {
    static constexpr int count = 1;
    using Real = T;
};
template <>
struct Parts<Complex> {
    static constexpr int count = 2;
    using Real = double;
};
template <>
struct Parts<cuComplex> {
    static constexpr int count = 2;
    using Real = float;
};

template <class T>
T scalar(double re, double im) {
    using R = typename Parts<T>::Real;
    T out{};
    R* p = reinterpret_cast<R*>(&out);
    p[0] = static_cast<R>(re);
    if (Parts<T>::count == 2) p[1] = static_cast<R>(im);
    return out;
}

template <class T>
struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t size) : n(size) { ok(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& host) { ok(cudaMemcpy(p, host.data(), n * sizeof(T), cudaMemcpyHostToDevice), "put"); }
    std::vector<T> get() const {
        std::vector<T> host(n);
        ok(cudaMemcpy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "get");
        return host;
    }
    void copy(const Device& other) { ok(cudaMemcpy(p, other.p, n * sizeof(T), cudaMemcpyDeviceToDevice), "copy"); }
};

template <class T>
std::vector<T> noise(std::size_t n, unsigned seed) {
    using R = typename Parts<T>::Real;
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<T> v(n);
    R* p = reinterpret_cast<R*>(v.data());
    for (std::size_t i = 0; i < n * Parts<T>::count; ++i) p[i] = static_cast<R>(u(gen));
    return v;
}

// the largest |a - b| over the largest |b|; infinite if a value is not finite
template <class T>
double error_of(const std::vector<T>& a, const std::vector<T>& b) {
    using R = typename Parts<T>::Real;
    const R* x = reinterpret_cast<const R*>(a.data());
    const R* y = reinterpret_cast<const R*>(b.data());
    double top = 0.0, most = 0.0;
    for (std::size_t i = 0; i < a.size() * Parts<T>::count; ++i) {
        if (!std::isfinite(static_cast<double>(x[i])) || !std::isfinite(static_cast<double>(y[i])))
            return std::numeric_limits<double>::infinity();
        top = std::max(top, std::fabs(static_cast<double>(x[i]) - static_cast<double>(y[i])));
        most = std::max(most, std::fabs(static_cast<double>(y[i])));
    }
    return most > 0.0 ? top / most : top;
}

const char* name_of(Op op) { return op == Op::N ? "N" : op == Op::T ? "T" : "C"; }

struct Worst {
    double error = -1.0;
    std::string where;
    void take(double e, const std::string& text) {
        if (e > error) {
            error = e;
            where = text;
        }
    }
};

// the error of one product: c (m, n) with ldc > m in places, the rows beyond m
// untouched; with beta zero c starts as NaN, which must not be read
template <class T>
double check_gemm(Blas& own, Blas& vendor, Op ta, Op tb, int m, int n, int k, bool beta_zero, unsigned seed) {
    const int ra = ta == Op::N ? m : k, ca = ta == Op::N ? k : m;
    const int rb = tb == Op::N ? k : n, cb = tb == Op::N ? n : k;
    const int lda = ra + 3, ldb = rb + 1, ldc = m + 2;
    Device<T> a(static_cast<std::size_t>(lda) * std::max(ca, 1)), b(static_cast<std::size_t>(ldb) * std::max(cb, 1));
    Device<T> c1(static_cast<std::size_t>(ldc) * n), c2(c1.n);
    a.put(noise<T>(a.n, seed));
    b.put(noise<T>(b.n, seed + 1));
    std::vector<T> start = noise<T>(c1.n, seed + 2);
    if (beta_zero) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) start[static_cast<std::size_t>(j) * ldc + i] = scalar<T>(nan, nan);
    }
    c1.put(start);
    c2.put(start);
    const T alpha = scalar<T>(0.7, -0.4), beta = beta_zero ? scalar<T>(0.0, 0.0) : scalar<T>(-0.3, 0.9);
    own.gemm(ta, tb, m, n, k, alpha, a.p, lda, b.p, ldb, beta, c1.p, ldc);
    vendor.gemm(ta, tb, m, n, k, alpha, a.p, lda, b.p, ldb, beta, c2.p, ldc);
    return error_of(c1.get(), c2.get());
}

template <class T>
double check_trsm(Blas& own, Blas& vendor, int m, int n, unsigned seed) {
    const int ldr = n + 2, ldb = m + 1;
    std::vector<T> r = noise<T>(static_cast<std::size_t>(ldr) * n, seed);
    // a well-conditioned triangle, as the Cholesky factor of an overlap is
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            using R = typename Parts<T>::Real;
            R* p = reinterpret_cast<R*>(&r[static_cast<std::size_t>(j) * ldr + i]);
            for (int q = 0; q < Parts<T>::count; ++q) p[q] *= static_cast<R>(0.5 / std::sqrt(static_cast<double>(n)));
        }
        r[static_cast<std::size_t>(j) * ldr + j] = scalar<T>(1.0 + 0.3 * std::sin(j), Parts<T>::count == 2 ? 0.1 : 0.0);
    }
    Device<T> rd(r.size()), b1(static_cast<std::size_t>(ldb) * n), b2(b1.n);
    rd.put(r);
    const std::vector<T> start = noise<T>(b1.n, seed + 1);
    b1.put(start);
    b2.put(start);
    own.trsm(m, n, rd.p, ldr, b1.p, ldb);
    vendor.trsm(m, n, rd.p, ldr, b2.p, ldb);
    return error_of(b1.get(), b2.get());
}

template <class T>
double check_copy(Blas& own, Blas& vendor, int n, int incx, int incy, unsigned seed) {
    Device<T> x(static_cast<std::size_t>(n) * incx), y1(static_cast<std::size_t>(n) * incy), y2(y1.n);
    x.put(noise<T>(x.n, seed));
    const std::vector<T> start = noise<T>(y1.n, seed + 1);
    y1.put(start);
    y2.put(start);
    own.copy(n, x.p, incx, y1.p, incy);
    vendor.copy(n, x.p, incx, y2.p, incy);
    return error_of(y1.get(), y2.get());
}

template <class T>
double check_real(Blas& own, Blas& vendor, int m, int n, unsigned seed) {
    const int incx = 5, incy = 3, lda = m + 2;
    Device<T> x(static_cast<std::size_t>(m) * incx), y(static_cast<std::size_t>(n) * incy);
    Device<T> a1(static_cast<std::size_t>(lda) * n), a2(a1.n);
    x.put(noise<T>(x.n, seed));
    y.put(noise<T>(y.n, seed + 1));
    const std::vector<T> start = noise<T>(a1.n, seed + 2);
    a1.put(start);
    a2.put(start);
    own.ger(m, n, static_cast<T>(0.6), x.p, incx, y.p, incy, a1.p, lda);
    vendor.ger(m, n, static_cast<T>(0.6), x.p, incx, y.p, incy, a2.p, lda);
    own.scal(m, static_cast<T>(-1.7), a1.p + lda);
    vendor.scal(m, static_cast<T>(-1.7), a2.p + lda);
    return error_of(a1.get(), a2.get());
}

double check_batched(Blas& own, Blas& vendor, Op tb, int m, int n, int k, int batch, unsigned seed) {
    const int lda = m, ldb = (tb == Op::N ? k : n) + 1, ldc = m;
    const long long stride_a = static_cast<long long>(lda) * k + 5, stride_c = static_cast<long long>(ldc) * n + 2;
    Device<double> a(static_cast<std::size_t>(stride_a) * batch), b(static_cast<std::size_t>(ldb) * (tb == Op::N ? n : k));
    Device<double> c1(static_cast<std::size_t>(stride_c) * batch), c2(c1.n);
    a.put(noise<double>(a.n, seed));
    b.put(noise<double>(b.n, seed + 1));
    const std::vector<double> start = noise<double>(c1.n, seed + 2);
    c1.put(start);
    c2.put(start);
    own.gemm_batched(tb, m, n, k, a.p, lda, stride_a, b.p, ldb, 0.5, c1.p, ldc, stride_c, batch);
    vendor.gemm_batched(tb, m, n, k, a.p, lda, stride_a, b.p, ldb, 0.5, c2.p, ldc, stride_c, batch);
    return error_of(c1.get(), c2.get());
}

template <class T>
int check_type(const char* type, double limit) {
    Blas own(Kernels::Mdapy), vendor(Kernels::Vendor);
    if (std::getenv("SMALL_TILES") != nullptr) own.set_small_tiles(std::atoi(std::getenv("SMALL_TILES")));
    Worst gemm, split, trsm, copy;
    const int shapes[][3] = {{1, 1, 1},      {3, 5, 7},       {32, 32, 16},   {33, 31, 17},  {64, 96, 48},
                             {100, 70, 130}, {257, 129, 65},  {1000, 3, 40},  {3, 1000, 40}, {40, 40, 0},
                             {2500, 60, 90}, {20000, 24, 50}, {45, 37, 5000}, {70, 70, 1024}, {8, 200, 3001}};
    unsigned seed = 11;
    for (const auto& s : shapes)
        for (const Op ta : {Op::N, Op::T, Op::C})
            for (const Op tb : {Op::N, Op::T, Op::C})
                for (const bool zero : {false, true}) {
                    char text[96];
                    std::snprintf(text, sizeof text, "%s%s %d x %d, sum of %d%s", name_of(ta), name_of(tb), s[0], s[1], s[2],
                                  zero ? ", beta 0" : "");
                    gemm.take(check_gemm<T>(own, vendor, ta, tb, s[0], s[1], s[2], zero, seed += 3), text);
                }
    // the long sums of the inner products, split along the sum; and whole, to
    // the same result within rounding
    const int longs[][3] = {{48, 48, 30000}, {130, 70, 20011}, {5, 3, 100000}, {300, 300, 4099}, {64, 1, 65536}};
    for (const auto& s : longs)
        for (const Op ta : {Op::T, Op::C})
            for (const bool whole : {false, true}) {
                char text[96];
                std::snprintf(text, sizeof text, "%sN %d x %d, sum of %d%s", name_of(ta), s[0], s[1], s[2],
                              whole ? ", not split" : "");
                own.set_segments(whole ? 1 : 0);
                split.take(check_gemm<T>(own, vendor, ta, Op::N, s[0], s[1], s[2], true, seed += 3), text);
            }
    own.set_segments(0);
    const int solves[][2] = {{1, 1}, {7, 5}, {100, 16}, {1000, 17}, {5000, 100}, {3000, 333}, {64, 48}, {20000, 40}};
    for (const auto& s : solves) {
        char text[64];
        std::snprintf(text, sizeof text, "%d rows, %d columns", s[0], s[1]);
        trsm.take(check_trsm<T>(own, vendor, s[0], s[1], seed += 2), text);
    }
    copy.take(check_copy<T>(own, vendor, 1, 1, 1, seed += 2), "1");
    copy.take(check_copy<T>(own, vendor, 300, 301, 1, seed += 2), "300 of a diagonal");
    copy.take(check_copy<T>(own, vendor, 1000, 1, 7, seed += 2), "1000 to a diagonal");
    int failed = 0;
    auto line = [&](const char* what, const Worst& w, double most) {
        const bool bad = !(w.error <= most);
        failed += bad;
        std::printf("%s %-22s largest error %.3e  (%s)%s\n", type, what, w.error, w.where.c_str(), bad ? "  FAILED" : "");
    };
    line("gemm", gemm, limit);
    line("gemm, long sums", split, limit);
    line("trsm", trsm, 20 * limit);
    line("copy", copy, 0.0);
    return failed;
}

template <class T>
int check_reals(const char* type, double limit) {
    Blas own(Kernels::Mdapy), vendor(Kernels::Vendor);
    if (std::getenv("SMALL_TILES") != nullptr) own.set_small_tiles(std::atoi(std::getenv("SMALL_TILES")));
    Worst w;
    w.take(check_real<T>(own, vendor, 1, 1, 5), "1 x 1");
    w.take(check_real<T>(own, vendor, 37, 61, 6), "37 x 61");
    w.take(check_real<T>(own, vendor, 400, 300, 7), "400 x 300");
    const bool bad = !(w.error <= limit);
    std::printf("%s %-22s largest error %.3e  (%s)%s\n", type, "ger and scal", w.error, w.where.c_str(), bad ? "  FAILED" : "");
    return bad;
}

int check() {
    int failed = check_type<Complex>("Z", 5e-14) + check_type<cuComplex>("C", 3e-5) + check_type<double>("D", 5e-14) +
                 check_type<float>("S", 3e-5);
    failed += check_reals<double>("D", 5e-15) + check_reals<float>("S", 5e-6);
    {
        Blas own(Kernels::Mdapy), vendor(Kernels::Vendor);
        if (std::getenv("SMALL_TILES") != nullptr) own.set_small_tiles(std::atoi(std::getenv("SMALL_TILES")));
        Worst w;
        w.take(check_batched(own, vendor, Op::N, 40, 50, 9, 7, 3), "N, 40 x 50, sum of 9, 7 products");
        w.take(check_batched(own, vendor, Op::T, 33, 16, 50, 5, 4), "T, 33 x 16, sum of 50, 5 products");
        w.take(check_batched(own, vendor, Op::N, 600, 110, 25, 64, 5), "N, 600 x 110, sum of 25, 64 products");
        const bool bad = !(w.error <= 5e-14);
        failed += bad;
        std::printf("D %-22s largest error %.3e  (%s)%s\n", "gemm, batched", w.error, w.where.c_str(), bad ? "  FAILED" : "");
    }
    std::printf(failed ? "FAILED: %d\n" : "all within the limits\n", failed);
    return failed ? 1 : 0;
}

// milliseconds of two products timed in turn, the least and the median
struct Time {
    double least, median;
};

template <class T>
void time_gemm(const char* type, Op ta, Op tb, int m, int n, int k, int repeats, const char* what = "") {
    Blas own(Kernels::Mdapy), vendor(Kernels::Vendor);
    if (std::getenv("SMALL_TILES") != nullptr) own.set_small_tiles(std::atoi(std::getenv("SMALL_TILES")));
    if (std::getenv("SEGMENTS") != nullptr) own.set_segments(std::atoi(std::getenv("SEGMENTS")));
    const int ra = ta == Op::N ? m : k, ca = ta == Op::N ? k : m;
    const int rb = tb == Op::N ? k : n, cb = tb == Op::N ? n : k;
    Device<T> a(static_cast<std::size_t>(ra) * ca), b(static_cast<std::size_t>(rb) * cb), c(static_cast<std::size_t>(m) * n);
    a.put(noise<T>(a.n, 1));
    b.put(noise<T>(b.n, 2));
    const T alpha = scalar<T>(1.0, 0.0), beta = scalar<T>(0.0, 0.0);
    cudaEvent_t start, stop;
    ok(cudaEventCreate(&start), "event");
    ok(cudaEventCreate(&stop), "event");
    std::vector<double> ms[2];
    for (int i = -2; i < repeats; ++i)
        for (int which = 0; which < 2; ++which) {
            Blas& blas = which == 0 ? vendor : own;
            ok(cudaEventRecord(start, cudaStreamPerThread), "record");
            blas.gemm(ta, tb, m, n, k, alpha, a.p, ra, b.p, rb, beta, c.p, m);
            ok(cudaEventRecord(stop, cudaStreamPerThread), "record");
            ok(cudaEventSynchronize(stop), "synchronize");
            float t = 0.0f;
            ok(cudaEventElapsedTime(&t, start, stop), "elapsed");
            if (i >= 0) ms[which].push_back(t);
        }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (auto& v : ms) std::sort(v.begin(), v.end());
    // floating-point operations: 2 a multiply-add of reals, 8 of complex numbers
    const double flops = 2.0 * Parts<T>::count * Parts<T>::count * m * n * static_cast<double>(k);
    std::printf("%sgemm %s%s %6d x %6d, sum of %6d   vendor %9.3f ms %7.0f GFLOP/s   ours %9.3f ms %7.0f GFLOP/s"
                "   ours/vendor %5.2f   %s\n", type, name_of(ta), name_of(tb), m, n, k, ms[0].front(),
                flops / ms[0].front() * 1e-6, ms[1].front(), flops / ms[1].front() * 1e-6, ms[1].front() / ms[0].front(),
                what);
    std::fflush(stdout);
}

// b r^-1 of b (m, n) and a well-conditioned upper triangle r, timed in turn
template <class T>
void time_trsm(const char* type, int m, int n, int repeats) {
    Blas own(Kernels::Mdapy), vendor(Kernels::Vendor);
    std::vector<T> r = noise<T>(static_cast<std::size_t>(n) * n, 3);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            using R = typename Parts<T>::Real;
            R* p = reinterpret_cast<R*>(&r[static_cast<std::size_t>(j) * n + i]);
            for (int q = 0; q < Parts<T>::count; ++q) p[q] *= static_cast<R>(0.5 / std::sqrt(static_cast<double>(n)));
        }
        r[static_cast<std::size_t>(j) * n + j] = scalar<T>(1.0 + 0.3 * std::sin(j), 0.0);
    }
    Device<T> rd(r.size()), start(static_cast<std::size_t>(m) * n), b(start.n);
    rd.put(r);
    start.put(noise<T>(start.n, 4));
    cudaEvent_t begin, end;
    ok(cudaEventCreate(&begin), "event");
    ok(cudaEventCreate(&end), "event");
    double least[2] = {1e30, 1e30};
    for (int i = -1; i < repeats; ++i)
        for (int which = 0; which < 2; ++which) {
            Blas& blas = which == 0 ? vendor : own;
            b.copy(start);
            ok(cudaEventRecord(begin, cudaStreamPerThread), "record");
            blas.trsm(m, n, rd.p, n, b.p, m);
            ok(cudaEventRecord(end, cudaStreamPerThread), "record");
            ok(cudaEventSynchronize(end), "synchronize");
            float t = 0.0f;
            ok(cudaEventElapsedTime(&t, begin, end), "elapsed");
            if (i >= 0) least[which] = std::min(least[which], static_cast<double>(t));
        }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    std::printf("%strsm %6d x %6d   vendor %9.3f ms   ours %9.3f ms   ours/vendor %5.2f\n", type, m, n, least[0], least[1],
                least[1] / least[0]);
    std::fflush(stdout);
}

Op op_of(const char* text) { return text[0] == 'N' ? Op::N : text[0] == 'T' ? Op::T : Op::C; }

void time_any(char type, Op ta, Op tb, int m, int n, int k, int repeats, const char* what = "") {
    if (type == 'z') time_gemm<Complex>("Z", ta, tb, m, n, k, repeats, what);
    if (type == 'c') time_gemm<cuComplex>("C", ta, tb, m, n, k, repeats, what);
    if (type == 'd') time_gemm<double>("D", ta, tb, m, n, k, repeats, what);
    if (type == 's') time_gemm<float>("S", ta, tb, m, n, k, repeats, what);
}

}  // namespace

#include "dft_blas_shapes.inc"

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "check";
    if (mode == "check") return check();
    if (mode == "bench") {
        const int repeats = argc > 2 ? std::atoi(argv[2]) : 10;
        for (const Shape& s : dft_shapes()) time_any(s.type, s.ta, s.tb, s.m, s.n, s.k, repeats, s.what);
        return 0;
    }
    if (mode == "large") {
        // larger products than those of the profile, as cells of hundreds of atoms take
        const int repeats = argc > 2 ? std::atoi(argv[2]) : 10;
        const Shape shapes[] = {
            {"projections", 'z', Op::C, Op::N, 200, 200, 20000},
            {"projections", 'z', Op::C, Op::N, 1000, 1500, 30000},
            {"projections, thin", 'z', Op::C, Op::N, 13, 64, 10000},
            {"rotation", 'z', Op::N, Op::N, 30000, 500, 500},
            {"rotation", 'z', Op::N, Op::N, 100000, 1500, 2000},
            {"overlaps", 'z', Op::C, Op::N, 500, 500, 30000},
            {"square", 'z', Op::N, Op::N, 2000, 2000, 2000},
            {"projections", 'd', Op::T, Op::N, 1000, 1500, 30000},
            {"rotation", 'd', Op::N, Op::N, 100000, 1500, 2000},
            {"overlaps", 'd', Op::T, Op::N, 500, 500, 30000},
            {"square", 'd', Op::N, Op::N, 4000, 4000, 4000},
        };
        for (const Shape& s : shapes) time_any(s.type, s.ta, s.tb, s.m, s.n, s.k, repeats, s.what);
        return 0;
    }
    if (mode == "time" && argc >= 8) {
        time_any(argv[2][0], op_of(argv[3]), op_of(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]),
                 argc > 8 ? std::atoi(argv[8]) : 10);
        return 0;
    }
    if (mode == "solve" && argc >= 5) {
        const int m = std::atoi(argv[3]), n = std::atoi(argv[4]), repeats = argc > 5 ? std::atoi(argv[5]) : 10;
        if (argv[2][0] == 'z') time_trsm<Complex>("Z", m, n, repeats);
        if (argv[2][0] == 'c') time_trsm<cuComplex>("C", m, n, repeats);
        if (argv[2][0] == 'd') time_trsm<double>("D", m, n, repeats);
        if (argv[2][0] == 's') time_trsm<float>("S", m, n, repeats);
        return 0;
    }
    std::fprintf(stderr, "usage: dft_blas_check check | bench [repeats] | time TYPE OPA OPB m n k [repeats] | solve TYPE m n [repeats]\n");
    return 2;
}
