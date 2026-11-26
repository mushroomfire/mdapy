// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The eigensolver and the Cholesky factorization of the DFT with MDAPY's
// kernels against the library of the vendor (cuSOLVER), on their own (no
// Python): results and times. Build and run with tests/cpp/dft_lapack_check.sh.
//
//   ./dft_lapack_check [repeats] [types] [sizes] [fused]   types of z d c s, sizes n,n,...,
//       fused the largest matrix tridiagonalized with one launch a step (-1: MDAPY's choice);
//       the environment variable FULL_DOUBLE (0 or 1) has the GPU taken as one of full double precision or not,
//       PANELS_FROM sets the smallest matrix tridiagonalized in panels, REFLECTORS the
//       reflectors of a block of the back-transformation, LEAF the rows of a leaf, BACK the
//       largest matrix back-transformed in one launch (0: none), HELD the blocks of the steps in one
//       launch with the trailing matrix in shared memory (0: those steps without it),
//       HOST_LIMIT the largest matrix MDAPY solves on the host, STEPS (0 or 1) whether the steps of
//       the small tridiagonalizations go in one launch,
//       WANTED the percent of the eigenvectors asked of MDAPY, the lowest (the vendor makes all)
//
// A row: the milliseconds of the vendor and of MDAPY (the least of the
// repeats), their ratio, and of MDAPY's result the largest |a v - v w| and
// |v^H v - 1| over the largest |a|, and the largest difference of the
// eigenvalues from the vendor's; then the same times of the factorization and
// the largest difference of the two factors. Built with -DMDAPY_DFT_PROFILE the spans of
// each size follow its row.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "dft/lapack/lapack.h"

#ifdef MDAPY_DFT_PROFILE
extern "C" void mdapy_dft_profile(const char* label);
#endif

using mdapy::dft::Complex;
using mdapy::dft::Kernels;
using mdapy::dft::Lapack;

namespace {

void ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

std::complex<double> wide(Complex x) { return {x.x, x.y}; }
std::complex<double> wide(cuComplex x) { return {x.x, x.y}; }
std::complex<double> wide(double x) { return x; }
std::complex<double> wide(float x) { return x; }
void narrow(std::complex<double> x, Complex& out) { out = make_cuDoubleComplex(x.real(), x.imag()); }
void narrow(std::complex<double> x, cuComplex& out) {
    out = make_cuComplex(static_cast<float>(x.real()), static_cast<float>(x.imag()));
}
void narrow(std::complex<double> x, double& out) { out = x.real(); }
void narrow(std::complex<double> x, float& out) { out = static_cast<float>(x.real()); }
bool complex_of(Complex) { return true; }
bool complex_of(cuComplex) { return true; }
bool complex_of(double) { return false; }
bool complex_of(float) { return false; }

double seconds() {
    ok(cudaStreamSynchronize(cudaStreamPerThread), "cudaStreamSynchronize");
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class T>
void run(const char* name, int n, int repeats, int fused) {
    const std::size_t size = static_cast<std::size_t>(n) * n;
    std::mt19937_64 gen(static_cast<unsigned>(n));
    std::normal_distribution<double> noise;
    // a Hermitian matrix of noise, and a positive definite one for the factorization
    std::vector<std::complex<double>> a(size), spd(size);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) {
            const std::complex<double> x(noise(gen), i == j || !complex_of(T{}) ? 0.0 : noise(gen));
            a[static_cast<std::size_t>(j) * n + i] = x;
            a[static_cast<std::size_t>(i) * n + j] = std::conj(x);
            spd[static_cast<std::size_t>(j) * n + i] = x / (2.0 * std::sqrt(static_cast<double>(n)));
            spd[static_cast<std::size_t>(i) * n + j] = std::conj(x) / (2.0 * std::sqrt(static_cast<double>(n)));
        }
    for (int i = 0; i < n; ++i) spd[static_cast<std::size_t>(i) * n + i] += 2.0;
    std::vector<T> host(size), factor_host(size);
    for (std::size_t i = 0; i < size; ++i) {
        narrow(a[i], host[i]);
        narrow(spd[i], factor_host[i]);
    }
    T *given = nullptr, *work = nullptr;
    ok(cudaMalloc(&given, 2 * size * sizeof(T)), "cudaMalloc");
    ok(cudaMalloc(&work, size * sizeof(T)), "cudaMalloc");
    ok(cudaMemcpy(given, host.data(), size * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    ok(cudaMemcpy(given + size, factor_host.data(), size * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    Lapack vendor(Kernels::Vendor), own(Kernels::Mdapy);
    own.set_fused_limit(fused);
    if (std::getenv("STEPS") != nullptr) own.set_together_steps(std::atoi(std::getenv("STEPS")));
    if (std::getenv("HOST_LIMIT") != nullptr) own.set_host_limit(std::atoi(std::getenv("HOST_LIMIT")));
    if (std::getenv("REFLECTORS") != nullptr) own.set_reflectors(std::atoi(std::getenv("REFLECTORS")));
    if (std::getenv("LEAF") != nullptr) own.set_leaf(std::atoi(std::getenv("LEAF")));
    if (std::getenv("BACK") != nullptr) own.set_back_limit(std::atoi(std::getenv("BACK")));
    if (std::getenv("HELD") != nullptr) own.set_held_blocks(std::atoi(std::getenv("HELD")));
    if (std::getenv("PANELS_FROM") != nullptr) own.set_panels_from(std::atoi(std::getenv("PANELS_FROM")));
    if (std::getenv("FULL_DOUBLE") != nullptr) own.set_full_double(std::atoi(std::getenv("FULL_DOUBLE")));
    const int wanted = std::getenv("WANTED") != nullptr ? std::max(1, n * std::atoi(std::getenv("WANTED")) / 100) : n;
    double times[2] = {1e30, 1e30}, factor_times[2] = {1e30, 1e30};
    std::vector<double> values[2];
    std::vector<T> vectors(size), factors[2] = {std::vector<T>(size), std::vector<T>(size)};
    for (int which = 0; which < 2; ++which) {
        Lapack& lapack = which == 0 ? vendor : own;
        for (int repeat = 0; repeat < repeats + 1; ++repeat) {
            ok(cudaMemcpy(work, given, size * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy");
            double start = seconds();
            lapack.eigh(work, n, values[which], which == 1 ? wanted : -1);
            double time = seconds() - start;
            if (repeat > 0) times[which] = std::min(times[which], time);  // the first warms up
            if (which == 1 && repeat == 0)
                ok(cudaMemcpy(vectors.data(), work, size * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
            ok(cudaMemcpy(work, given + size, size * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy");
            start = seconds();
            const int info = lapack.potrf(work, n);
            time = seconds() - start;
            if (info != 0) std::printf("potrf: info %d\n", info);
            if (repeat > 0) factor_times[which] = std::min(factor_times[which], time);
            if (repeat == 0) ok(cudaMemcpy(factors[which].data(), work, size * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
        }
    }
    // of MDAPY's eigenpairs
    double residual = 0.0, orthogonality = 0.0, scale = 0.0, difference = 0.0;
    std::vector<std::complex<double>> v(size);
    for (std::size_t i = 0; i < size; ++i) {
        v[i] = wide(vectors[i]);
        scale = std::max(scale, std::abs(a[i]));
    }
    for (int j = 0; j < n; ++j)
        difference = std::max(difference, std::abs(values[1][static_cast<std::size_t>(j)] - values[0][static_cast<std::size_t>(j)]));
    // of the large matrices, 64 of the eigenvectors spread over all
    for (int j = 0; j < wanted; j += std::max(1, n > 1024 ? n / 64 : 1)) {
        const std::complex<double>* vj = v.data() + static_cast<std::size_t>(j) * n;
        for (int i = 0; i < n; ++i) {
            std::complex<double> av = 0.0, vv = 0.0;
            const bool made = i < wanted;
            const std::complex<double>* vi = v.data() + static_cast<std::size_t>(i) * n;
            for (int k = 0; k < n; ++k) {
                av += a[static_cast<std::size_t>(k) * n + i] * vj[k];
                if (made) vv += std::conj(vi[k]) * vj[k];
            }
            residual = std::max(residual, std::abs(av - values[1][static_cast<std::size_t>(j)] * vj[i]));
            if (made) orthogonality = std::max(orthogonality, std::abs(vv - (i == j ? 1.0 : 0.0)));
        }
    }
    // of the factors, the upper triangles
    double apart = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) {
            const std::size_t at = static_cast<std::size_t>(j) * n + i;
            apart = std::max(apart, std::abs(wide(factors[1][at]) - wide(factors[0][at])));
        }
    std::printf("%s n %5d  eigh v %8.3f ms o %8.3f ms r %5.2f  res %.1e orth %.1e dw %.1e | potrf v %7.3f ms o %7.3f ms r %5.2f dr %.1e\n",
                name, n, 1e3 * times[0], 1e3 * times[1], times[1] / times[0], residual / scale, orthogonality,
                difference / scale, 1e3 * factor_times[0], 1e3 * factor_times[1], factor_times[1] / factor_times[0], apart);
    std::fflush(stdout);
#ifdef MDAPY_DFT_PROFILE
    std::fflush(stderr);
    mdapy_dft_profile((std::string(name) + std::to_string(n)).c_str());
#endif
    cudaFree(given);
    cudaFree(work);
}

}  // namespace

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 5;
    const std::string types = argc > 2 ? argv[2] : "zd";
    std::vector<int> sizes = {48, 64, 65, 96, 128, 154, 195, 230, 282, 308, 390, 512, 768, 1024, 1500, 2048};
    if (argc > 3) {
        sizes.clear();
        for (char* p = argv[3]; *p != 0;) {
            sizes.push_back(static_cast<int>(std::strtol(p, &p, 10)));
            if (*p == ',') ++p;
        }
    }
    const int fused = argc > 4 ? std::atoi(argv[4]) : -1;
    for (const char type : types)
        for (const int n : sizes) {
            if (type == 'z') run<Complex>("Z", n, repeats, fused);
            if (type == 'd') run<double>("D", n, repeats, fused);
            if (type == 'c') run<cuComplex>("C", n, repeats, fused);
            if (type == 's') run<float>("S", n, repeats, fused);
        }
    return 0;
}
