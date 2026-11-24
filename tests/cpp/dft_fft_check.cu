// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// BoxFft against cuFFT, on its own (no Python): results and times.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 -Isrc -o dft_fft_check \
//        src/dft/fft/box_fft.cu tests/cpp/dft_fft_check.cu -lcufft
//   ./dft_fft_check check             every type on a set of sizes and batches
//   ./dft_fft_check bench [repeats]   times on the shapes of the DFT
//   ./dft_fft_check time n1 n2 n3 count [repeats]   one shape
//   ./dft_fft_check sweep n1 n2 n3 count precision kind   lines and threads of a block
//
// Errors are the largest |ours - cuFFT| over the largest |cuFFT| of a transform.

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "dft/fft/box_fft.h"

using mdapy::dft::BoxFft;

namespace {

void ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

void ok(cufftResult e, const char* what) {
    if (e != CUFFT_SUCCESS) {
        std::fprintf(stderr, "%s: cuFFT error %d\n", what, static_cast<int>(e));
        std::exit(2);
    }
}

template <class R>
struct Device {
    R* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t size) : n(size) { ok(cudaMalloc(&p, n * sizeof(R)), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<R>& host) { ok(cudaMemcpy(p, host.data(), n * sizeof(R), cudaMemcpyHostToDevice), "put"); }
    std::vector<R> get() const {
        std::vector<R> host(n);
        ok(cudaMemcpy(host.data(), p, n * sizeof(R), cudaMemcpyDeviceToHost), "get");
        return host;
    }
    void copy(const Device& other) { ok(cudaMemcpy(p, other.p, n * sizeof(R), cudaMemcpyDeviceToDevice), "copy"); }
};

template <class R>
std::vector<R> noise(std::size_t n, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<R> v(n);
    for (auto& x : v) x = static_cast<R>(u(gen));
    return v;
}

// the largest |a - b| over the largest |b|, of complex numbers when pairs
template <class R>
double error_of(const std::vector<R>& a, const std::vector<R>& b, bool pairs, double scale_a = 1.0) {
    double top = 0.0, most = 0.0;
    const std::size_t step = pairs ? 2 : 1;
    for (std::size_t i = 0; i + step <= a.size(); i += step) {
        double d = 0.0, m = 0.0;
        for (std::size_t k = 0; k < step; ++k) {
            const double x = scale_a * static_cast<double>(a[i + k]) - static_cast<double>(b[i + k]);
            d += x * x;
            m += static_cast<double>(b[i + k]) * static_cast<double>(b[i + k]);
        }
        top = std::max(top, d);
        most = std::max(most, m);
    }
    return std::sqrt(top / most);
}

template <class R>
struct Types {};
template <>
struct Types<double> {
    static constexpr cufftType complex = CUFFT_Z2Z, to_complex = CUFFT_D2Z, to_real = CUFFT_Z2D;
    static void exec(cufftHandle h, double* a, int dir) {
        ok(cufftExecZ2Z(h, reinterpret_cast<cufftDoubleComplex*>(a), reinterpret_cast<cufftDoubleComplex*>(a), dir), "Z2Z");
    }
    static void exec_forward(cufftHandle h, double* a, double* b) {
        ok(cufftExecD2Z(h, a, reinterpret_cast<cufftDoubleComplex*>(b)), "D2Z");
    }
    static void exec_inverse(cufftHandle h, double* a, double* b) {
        ok(cufftExecZ2D(h, reinterpret_cast<cufftDoubleComplex*>(a), b), "Z2D");
    }
    static constexpr const char* names[6] = {"Z2Z forward", "Z2Z inverse", "D2Z", "Z2D", "Z2Z round trip", "D2Z-Z2D round trip"};
};
template <>
struct Types<float> {
    static constexpr cufftType complex = CUFFT_C2C, to_complex = CUFFT_R2C, to_real = CUFFT_C2R;
    static void exec(cufftHandle h, float* a, int dir) {
        ok(cufftExecC2C(h, reinterpret_cast<cufftComplex*>(a), reinterpret_cast<cufftComplex*>(a), dir), "C2C");
    }
    static void exec_forward(cufftHandle h, float* a, float* b) {
        ok(cufftExecR2C(h, a, reinterpret_cast<cufftComplex*>(b)), "R2C");
    }
    static void exec_inverse(cufftHandle h, float* a, float* b) {
        ok(cufftExecC2R(h, reinterpret_cast<cufftComplex*>(a), b), "C2R");
    }
    static constexpr const char* names[6] = {"C2C forward", "C2C inverse", "R2C", "C2R", "C2C round trip", "R2C-C2R round trip"};
};

cufftHandle plan_of(const std::array<int, 3>& g, cufftType type, int count) {
    int n[3] = {g[0], g[1], g[2]};
    const int points = g[0] * g[1] * g[2], half = g[0] * g[1] * (g[2] / 2 + 1);
    int idist = points, odist = points;
    if (type == CUFFT_D2Z || type == CUFFT_R2C) odist = half;
    if (type == CUFFT_Z2D || type == CUFFT_C2R) idist = half;
    cufftHandle h{};
    ok(cufftPlanMany(&h, 3, n, nullptr, 1, idist, nullptr, 1, odist, type, count), "cufftPlanMany");
    return h;
}

struct Worst {
    double error = 0.0;
    std::string where;
    void take(double e, const std::array<int, 3>& g, int count) {
        if (e > error || where.empty()) {
            if (e >= error) {
                error = e;
                char text[96];
                std::snprintf(text, sizeof text, "%d x %d x %d, %d boxes", g[0], g[1], g[2], count);
                where = text;
            }
        }
    }
};

// errors of the six checks of a precision on one shape:
// forward, inverse, real forward, real inverse, round trip, real round trip
template <class R>
std::array<double, 6> check_shape(const std::array<int, 3>& g, int count, unsigned seed) {
    std::array<double, 6> e{};
    BoxFft fft(g);
    const std::size_t points = fft.points(), half = fft.spectrum();
    const std::size_t nc = 2 * points * count, nr = points * count, nh = 2 * half * count;
    {
        const std::vector<R> input = noise<R>(nc, seed);
        Device<R> a(nc), b(nc);
        const cufftHandle h = plan_of(g, Types<R>::complex, count);
        for (int dir = 0; dir < 2; ++dir) {
            a.put(input);
            b.put(input);
            Types<R>::exec(h, a.p, dir == 0 ? CUFFT_FORWARD : CUFFT_INVERSE);
            if (dir == 0)
                fft.forward(b.p, count);
            else
                fft.inverse(b.p, count);
            e[dir] = error_of(b.get(), a.get(), true);
        }
        // forward then inverse of ours gives the input times the points
        b.put(input);
        fft.forward(b.p, count);
        fft.inverse(b.p, count);
        e[4] = error_of(b.get(), input, true, 1.0 / static_cast<double>(points));
        cufftDestroy(h);
    }
    {
        const std::vector<R> input = noise<R>(nr, seed + 1);
        Device<R> values(nr), spectrum_a(nh), spectrum_b(nh), back_a(nr), back_b(nr);
        const cufftHandle hf = plan_of(g, Types<R>::to_complex, count), hi = plan_of(g, Types<R>::to_real, count);
        values.put(input);
        Types<R>::exec_forward(hf, values.p, spectrum_a.p);
        fft.forward(values.p, spectrum_b.p, count);
        const std::vector<R> reference = spectrum_a.get();
        e[2] = error_of(spectrum_b.get(), reference, true);  // the half-complex layout, element by element
        // the inverse of both from the spectrum of cuFFT
        spectrum_b.put(reference);
        Types<R>::exec_inverse(hi, spectrum_a.p, back_a.p);
        fft.inverse(spectrum_b.p, back_b.p, count);
        e[3] = error_of(back_b.get(), back_a.get(), false);
        // forward then inverse of ours
        fft.forward(values.p, spectrum_b.p, count);
        fft.inverse(spectrum_b.p, back_b.p, count);
        e[5] = error_of(back_b.get(), input, false, 1.0 / static_cast<double>(points));
        cufftDestroy(hf);
        cufftDestroy(hi);
    }
    return e;
}

// The inverse real transform of a spectrum that is not that of a real box
// (noise): cuFFT's result is then a matter of its order of the axes.
template <class R>
double check_unsymmetric(const std::array<int, 3>& g, int count, unsigned seed) {
    BoxFft fft(g);
    const std::size_t nr = fft.points() * count, nh = 2 * fft.spectrum() * count;
    const std::vector<R> input = noise<R>(nh, seed);
    Device<R> sa(nh), sb(nh), ba(nr), bb(nr);
    const cufftHandle hi = plan_of(g, Types<R>::to_real, count);
    sa.put(input);
    sb.put(input);
    Types<R>::exec_inverse(hi, sa.p, ba.p);
    fft.inverse(sb.p, bb.p, count);
    cufftDestroy(hi);
    return error_of(bb.get(), ba.get(), false);
}

template <class R>
int check_precision(const std::vector<std::pair<std::array<int, 3>, int>>& shapes, double limit) {
    Worst worst[6];
    Worst unsymmetric;
    unsigned seed = 7;
    for (const auto& [g, count] : shapes) {
        const std::array<double, 6> e = check_shape<R>(g, count, seed += 2);
        for (int i = 0; i < 6; ++i) worst[i].take(e[i], g, count);
        unsymmetric.take(check_unsymmetric<R>(g, count, seed + 100), g, count);
    }
    int failed = 0;
    for (int i = 0; i < 6; ++i) {
        const bool bad = !(worst[i].error < limit);
        failed += bad;
        std::printf("%-20s largest error %.3e  (%s)%s\n", Types<R>::names[i], worst[i].error, worst[i].where.c_str(),
                    bad ? "  FAILED" : "");
    }
    std::printf("%-20s largest error %.3e  (%s)  [spectrum of noise, not of a real box: for information]\n",
                Types<R>::names[3], unsymmetric.error, unsymmetric.where.c_str());
    return failed;
}

int check() {
    std::vector<std::pair<std::array<int, 3>, int>> shapes;
    // every size from 2 to 64 that is a product of 2, 3, 5, 7, along each axis in turn
    std::vector<int> sizes;
    for (int n = 2; n <= 64; ++n)
        if (BoxFft::supported({n, n, n})) sizes.push_back(n);
    int turn = 0;
    for (const int n : sizes) {
        std::array<int, 3> g = {6, 10, 7};
        g[turn % 3] = n;
        shapes.push_back({g, 1 + turn % 3});
        ++turn;
    }
    // lines longer than a block takes at once, more sizes of every radix, boxes of the DFT
    for (const int n : {72, 75, 81, 96, 98, 100, 105, 120, 125, 128, 135, 147, 160, 162, 175, 189, 200, 210, 243, 245,
                        250, 256, 270, 288, 300, 343, 384, 400, 432, 500, 512, 625, 729, 1024}) {
        shapes.push_back({{n, 4, 6}, 2});
        shapes.push_back({{3, n, 5}, 1});
        shapes.push_back({{5, 2, n}, 3});
    }
    for (const auto& g : std::vector<std::array<int, 3>>{{27, 27, 27}, {36, 36, 36}, {40, 40, 40}, {54, 54, 54}, {45, 48, 80},
                                                         {72, 72, 72}, {80, 80, 80}, {35, 49, 63}, {64, 64, 64}, {90, 90, 90},
                                                         {100, 100, 100}, {120, 120, 120}, {96, 60, 150}, {21, 25, 81}})
        for (const int count : {1, 2, 5}) shapes.push_back({g, count});
    shapes.push_back({{27, 27, 27}, 64});
    shapes.push_back({{48, 48, 48}, 33});
    shapes.push_back({{160, 160, 160}, 1});
    shapes.push_back({{200, 200, 200}, 1});
    shapes.push_back({{180, 192, 210}, 1});
    std::printf("%zu shapes\n", shapes.size());
    int failed = check_precision<double>(shapes, 2e-14);
    failed += check_precision<float>(shapes, 1e-5);
    std::printf(failed ? "FAILED: %d\n" : "all within the limits (2e-14 double, 1e-5 single)\n", failed);
    return failed ? 1 : 0;
}

// milliseconds of a transform: the least and the median of the repeats, each
// from a fresh copy of the input (not timed)
struct Time {
    double least, median;
};

template <class Run, class Reset>
Time timed(int repeats, Reset reset, Run run) {
    cudaEvent_t start, stop;
    ok(cudaEventCreate(&start), "event");
    ok(cudaEventCreate(&stop), "event");
    std::vector<double> ms;
    for (int i = -3; i < repeats; ++i) {
        reset();
        ok(cudaEventRecord(start), "record");
        run();
        ok(cudaEventRecord(stop), "record");
        ok(cudaEventSynchronize(stop), "synchronize");
        float t = 0.0f;
        ok(cudaEventElapsedTime(&t, start, stop), "elapsed");
        if (i >= 0) ms.push_back(t);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(ms.begin(), ms.end());
    return {ms.front(), ms[ms.size() / 2]};
}

// Two transforms timed in turn, a repeat of one after a repeat of the other,
// so that both see the same state of the GPU (its clocks move with the load).
template <class Reset, class First, class Second>
std::pair<Time, Time> timed_pair(int repeats, Reset reset, First first, Second second) {
    cudaEvent_t start, stop;
    ok(cudaEventCreate(&start), "event");
    ok(cudaEventCreate(&stop), "event");
    std::vector<double> ms[2];
    for (int i = -3; i < repeats; ++i)
        for (int which = 0; which < 2; ++which) {
            reset();
            ok(cudaEventRecord(start), "record");
            if (which == 0)
                first();
            else
                second();
            ok(cudaEventRecord(stop), "record");
            ok(cudaEventSynchronize(stop), "synchronize");
            float t = 0.0f;
            ok(cudaEventElapsedTime(&t, start, stop), "elapsed");
            if (i >= 0) ms[which].push_back(t);
        }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (auto& m : ms) std::sort(m.begin(), m.end());
    return {{ms[0].front(), ms[0][ms[0].size() / 2]}, {ms[1].front(), ms[1][ms[1].size() / 2]}};
}

void line(const char* name, const std::array<int, 3>& g, int count, Time cu, Time ours) {
    std::printf("%-12s %4d x %4d x %4d  x %3d   cuFFT %8.3f ms (median %8.3f)   ours %8.3f ms (median %8.3f)   ours/cuFFT %5.2f\n",
                name, g[0], g[1], g[2], count, cu.least, cu.median, ours.least, ours.median, ours.least / cu.least);
    std::fflush(stdout);
}

template <class R>
void time_shape(const std::array<int, 3>& g, int count, int repeats, BoxFft* tuned = nullptr, bool quiet = false,
                double* out = nullptr) {
    BoxFft own(g);
    BoxFft& fft = tuned ? *tuned : own;
    const std::size_t points = fft.points(), half = fft.spectrum();
    const std::size_t nc = 2 * points * count, nr = points * count, nh = 2 * half * count;
    {
        Device<R> input(nc), a(nc);
        input.put(noise<R>(nc, 3));
        const cufftHandle h = plan_of(g, Types<R>::complex, count);
        for (int dir = 0; dir < 2; ++dir) {
            const auto [cu, ours] = timed_pair(
                repeats, [&] { a.copy(input); },
                [&] { Types<R>::exec(h, a.p, dir == 0 ? CUFFT_FORWARD : CUFFT_INVERSE); }, [&] {
                    if (dir == 0)
                        fft.forward(a.p, count);
                    else
                        fft.inverse(a.p, count);
                });
            if (!quiet) line(Types<R>::names[dir], g, count, cu, ours);
            if (out) out[dir] = ours.least / cu.least;
        }
        cufftDestroy(h);
    }
    {
        Device<R> values(nr), spectrum(nh), keep(nh), back(nr);
        values.put(noise<R>(nr, 4));
        const cufftHandle hf = plan_of(g, Types<R>::to_complex, count), hi = plan_of(g, Types<R>::to_real, count);
        std::pair<Time, Time> t = timed_pair(
            repeats, [] {}, [&] { Types<R>::exec_forward(hf, values.p, spectrum.p); },
            [&] { fft.forward(values.p, spectrum.p, count); });
        if (!quiet) line(Types<R>::names[2], g, count, t.first, t.second);
        if (out) out[2] = t.second.least / t.first.least;
        Types<R>::exec_forward(hf, values.p, keep.p);
        t = timed_pair(
            repeats, [&] { spectrum.copy(keep); }, [&] { Types<R>::exec_inverse(hi, spectrum.p, back.p); },
            [&] { fft.inverse(spectrum.p, back.p, count); });
        const Time cu = t.first, ours = t.second;
        if (!quiet) line(Types<R>::names[3], g, count, cu, ours);
        if (out) out[3] = ours.least / cu.least;
        cufftDestroy(hf);
        cufftDestroy(hi);
    }
}

// The shapes of the DFT: grids of the bands (4 E_cut) of the cells of the
// verification and of the aluminium example, with the bands a batch holds
// (boxes of at most 256 MB, at most the bands of the cell), and one or two
// boxes for the density and the potentials.
struct Shape {
    const char* cell;
    std::array<int, 3> grid;
    int count;
};

const std::vector<Shape>& dft_shapes();

int bench(int repeats) {
    for (const Shape& s : dft_shapes()) {
        std::printf("# %s\n", s.cell);
        time_shape<double>(s.grid, s.count, repeats);
        time_shape<float>(s.grid, s.count, repeats);
    }
    return 0;
}

}  // namespace

#include "dft_fft_shapes.inc"

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "check";
    if (mode == "check") return check();
    if (mode == "bench") return bench(argc > 2 ? std::atoi(argv[2]) : 30);
    if ((mode == "time" || mode == "apart") && argc >= 6) {  // apart: every axis in its own kernel
        const std::array<int, 3> g = {std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
        const int count = std::atoi(argv[5]), repeats = argc > 6 ? std::atoi(argv[6]) : 30;
        BoxFft fft(g);
        if (mode == "apart") fft.only_axes(7 | 16);
        if (argc > 7) fft.tune(0, 2, 0, std::atoi(argv[7])), fft.tune(1, 2, 0, std::atoi(argv[7]));
        time_shape<double>(g, count, repeats, &fft);
        time_shape<float>(g, count, repeats, &fft);
        return 0;
    }
    if (mode == "axes" && argc >= 6) {  // the time of each axis of the complex transforms
        const std::array<int, 3> g = {std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
        const int count = std::atoi(argv[5]);
        for (const int mask : {4, 2, 1, 7, 12, 10, 9}) {  // 8: no passes, loads and stores alone
            BoxFft fft(g);
            fft.only_axes(mask);
            Device<double> d(2 * fft.points() * count);
            Device<float> f(2 * fft.points() * count);
            ok(cudaMemset(d.p, 0, d.n * 8), "memset");
            ok(cudaMemset(f.p, 0, f.n * 4), "memset");
            const Time td = timed(20, [] {}, [&] { fft.forward(d.p, count); });
            const Time tf = timed(20, [] {}, [&] { fft.forward(f.p, count); });
            std::printf("axes %d: double %.3f ms, single %.3f ms\n", mask, td.least, tf.least);
        }
        return 0;
    }
    if (mode == "sweep" && argc >= 8) {
        const std::array<int, 3> g = {std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
        const int count = std::atoi(argv[5]), precision = std::atoi(argv[6]), kind = std::atoi(argv[7]);
        for (int log_lines = 1; log_lines <= 6; ++log_lines)
            for (const int threads : {64, 128, 192, 256, 384, 512}) {
                BoxFft fft(g);
                fft.tune(precision, kind, log_lines, threads);
                double r[4] = {};
                if (precision == 0)
                    time_shape<double>(g, count, 10, &fft, true, r);
                else
                    time_shape<float>(g, count, 10, &fft, true, r);
                std::printf("lines 2^%d threads %3d  ours/cuFFT: forward %.2f inverse %.2f real forward %.2f real inverse %.2f\n",
                            log_lines, threads, r[0], r[1], r[2], r[3]);
                std::fflush(stdout);
            }
        return 0;
    }
    std::fprintf(stderr, "usage: dft_fft_check check | bench [repeats] | time n1 n2 n3 count [repeats] | sweep ...\n");
    return 2;
}
