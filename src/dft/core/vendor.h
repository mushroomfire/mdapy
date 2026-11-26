// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// NVIDIA's cuBLAS, cuFFT and cuSOLVER, which MDAPY's kernels do without,
// loaded (dlopen) at their first use: the module names none of them and loads
// without them. Each library is found in the folder nvidia/<package>/lib that
// NVIDIA's package of it installs beside the package of the module (the
// extra dft of MDAPY), else where the loader of the system finds it
// (LD_LIBRARY_PATH, the folders of the system). The libraries cuSOLVER needs,
// cuBLAS, cuBLASLt, cuSPARSE and nvJitLink, are loaded before it the same way:
// NVIDIA's libraries look for one another in their own folder only, and the
// loader takes a library already loaded for one of the same soname. Each
// library is a table of the functions the DFT calls, loaded once by the first
// thread that asks, whose names src/dft/core/portable.h gives the calls. The
// sonames are those of the toolkit the module was built with, which the build
// gives (MDAPY_SONAME_*); those of CUDA 12 without it. With HIP the libraries
// of ROCm are linked, and none of this is compiled.

#pragma once

#if !defined(MDAPY_GPU_HIP)

#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <dlfcn.h>
#include <link.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MDAPY_SONAME_CUBLAS
#define MDAPY_SONAME_CUBLAS "libcublas.so.12"
#define MDAPY_SONAME_CUBLASLT "libcublasLt.so.12"
#define MDAPY_SONAME_CUFFT "libcufft.so.11"
#define MDAPY_SONAME_CUSOLVER "libcusolver.so.11"
#define MDAPY_SONAME_CUSPARSE "libcusparse.so.12"
#define MDAPY_SONAME_NVJITLINK "libnvJitLink.so.12"
#endif

namespace mdapy::dft::vendor {

struct Cublas {
    decltype(&cublasCreate) create;
    decltype(&cublasDestroy) destroy;
    decltype(&cublasSetStream) set_stream;
    decltype(&cublasScopy) scopy;
    decltype(&cublasDcopy) dcopy;
    decltype(&cublasCcopy) ccopy;
    decltype(&cublasZcopy) zcopy;
    decltype(&cublasSscal) sscal;
    decltype(&cublasDscal) dscal;
    decltype(&cublasSger) sger;
    decltype(&cublasDger) dger;
    decltype(&cublasSgemm) sgemm;
    decltype(&cublasDgemm) dgemm;
    decltype(&cublasCgemm) cgemm;
    decltype(&cublasZgemm) zgemm;
    decltype(&cublasDgemmStridedBatched) dgemm_strided_batched;
    decltype(&cublasStrsm) strsm;
    decltype(&cublasDtrsm) dtrsm;
    decltype(&cublasCtrsm) ctrsm;
    decltype(&cublasZtrsm) ztrsm;
};

struct Cufft {
    decltype(&cufftPlanMany) plan_many;
    decltype(&cufftSetStream) set_stream;
    decltype(&cufftDestroy) destroy;
    decltype(&cufftExecR2C) exec_r2c;
    decltype(&cufftExecC2R) exec_c2r;
    decltype(&cufftExecC2C) exec_c2c;
    decltype(&cufftExecD2Z) exec_d2z;
    decltype(&cufftExecZ2D) exec_z2d;
    decltype(&cufftExecZ2Z) exec_z2z;
};

struct Cusolver {
    decltype(&cusolverDnCreate) create;
    decltype(&cusolverDnDestroy) destroy;
    decltype(&cusolverDnSetStream) set_stream;
    decltype(&cusolverDnSsyevd_bufferSize) ssyevd_buffer_size;
    decltype(&cusolverDnDsyevd_bufferSize) dsyevd_buffer_size;
    decltype(&cusolverDnCheevd_bufferSize) cheevd_buffer_size;
    decltype(&cusolverDnZheevd_bufferSize) zheevd_buffer_size;
    decltype(&cusolverDnSsyevd) ssyevd;
    decltype(&cusolverDnDsyevd) dsyevd;
    decltype(&cusolverDnCheevd) cheevd;
    decltype(&cusolverDnZheevd) zheevd;
    decltype(&cusolverDnSpotrf_bufferSize) spotrf_buffer_size;
    decltype(&cusolverDnDpotrf_bufferSize) dpotrf_buffer_size;
    decltype(&cusolverDnCpotrf_bufferSize) cpotrf_buffer_size;
    decltype(&cusolverDnZpotrf_bufferSize) zpotrf_buffer_size;
    decltype(&cusolverDnSpotrf) spotrf;
    decltype(&cusolverDnDpotrf) dpotrf;
    decltype(&cusolverDnCpotrf) cpotrf;
    decltype(&cusolverDnZpotrf) zpotrf;
};

namespace detail {

// any address of the module, for the folder of its file
inline const int anchor = 0;

// The folder of the file of this module, from which NVIDIA's packages lie at
// ../nvidia; empty if the loader does not tell.
inline std::string folder() {
    Dl_info info{};
    if (dladdr(&anchor, &info) == 0 || info.dli_fname == nullptr) return {};
    const std::string file(info.dli_fname);
    const std::size_t slash = file.rfind('/');
    return slash == std::string::npos ? std::string(".") : file.substr(0, slash);
}

// The libraries loaded, by their files, and the messages of those that were not.
struct Opened {
    std::mutex mutex;
    std::vector<std::string> files;
};
inline Opened& opened() {
    static Opened state;
    return state;
}

// A library by its soname, from nvidia/<package>/lib beside the package of
// the module, else as the loader finds it; why not in why.
inline void* open(const char* package, const char* soname, std::string& why) {
    const std::string base = folder();
    void* handle = nullptr;
    if (!base.empty()) {
        const std::string file = base + "/../nvidia/" + package + "/lib/" + soname;
        handle = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
    }
    if (handle == nullptr) handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        why = error != nullptr ? error : soname;
        return nullptr;
    }
    link_map* map = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0 && map != nullptr && map->l_name != nullptr) {
        Opened& state = opened();
        const std::lock_guard<std::mutex> lock(state.mutex);
        state.files.emplace_back(map->l_name);
    }
    return handle;
}

// The function of that name in a library, whose type F is that of its
// declaration in the headers of the toolkit.
template <class F>
void find(void* handle, const char* library, const char* name, F& to) {
    static_assert(sizeof(F) == sizeof(void*), "a function is a pointer");
    void* symbol = dlsym(handle, name);
    if (symbol == nullptr) throw std::runtime_error(std::string(library) + " has no function " + name);
    std::memcpy(&to, &symbol, sizeof to);
}

// what to do when a library is missing
inline std::string missing(const std::string& why) {
    return "NVIDIA's cuBLAS, cuFFT and cuSOLVER, which DFT(kernels=\"vendor\") takes (and the FFT of a grid "
           "whose sizes are not products of 2, 3, 5 and 7), could not be loaded (" +
           why + "): pip install \"mdapy[cuda12,dft]\", or DFT(kernels=\"mdapy\")";
}

// A table loaded once, its error kept and thrown at each ask.
template <class Table, class Load>
const Table& loaded(Load load) {
    static Table table{};
    static std::string error;
    static std::once_flag once;
    std::call_once(once, [&] {
        try {
            load(table);
        } catch (const std::exception& e) {
            error = e.what();
        }
    });
    if (!error.empty()) throw std::runtime_error(error);
    return table;
}

// the name of the symbol of a function, after the macros of the headers (cublasZgemm is cublasZgemm_v2)
#define MDAPY_VENDOR_SYMBOL(name) MDAPY_VENDOR_SYMBOL_OF(name)
#define MDAPY_VENDOR_SYMBOL_OF(name) #name
#define MDAPY_VENDOR_FIND(field, name) find(handle, library, MDAPY_VENDOR_SYMBOL(name), table.field)

inline void* open_or_throw(const char* package, const char* soname) {
    std::string why;
    void* handle = open(package, soname, why);
    if (handle == nullptr) throw std::runtime_error(missing(why));
    return handle;
}

}  // namespace detail

inline const Cublas& cublas() {
    return detail::loaded<Cublas>([](Cublas& table) {
        using namespace detail;
        open_or_throw("cublas", MDAPY_SONAME_CUBLASLT);
        void* handle = open_or_throw("cublas", MDAPY_SONAME_CUBLAS);
        const char* library = MDAPY_SONAME_CUBLAS;
        MDAPY_VENDOR_FIND(create, cublasCreate);
        MDAPY_VENDOR_FIND(destroy, cublasDestroy);
        MDAPY_VENDOR_FIND(set_stream, cublasSetStream);
        MDAPY_VENDOR_FIND(scopy, cublasScopy);
        MDAPY_VENDOR_FIND(dcopy, cublasDcopy);
        MDAPY_VENDOR_FIND(ccopy, cublasCcopy);
        MDAPY_VENDOR_FIND(zcopy, cublasZcopy);
        MDAPY_VENDOR_FIND(sscal, cublasSscal);
        MDAPY_VENDOR_FIND(dscal, cublasDscal);
        MDAPY_VENDOR_FIND(sger, cublasSger);
        MDAPY_VENDOR_FIND(dger, cublasDger);
        MDAPY_VENDOR_FIND(sgemm, cublasSgemm);
        MDAPY_VENDOR_FIND(dgemm, cublasDgemm);
        MDAPY_VENDOR_FIND(cgemm, cublasCgemm);
        MDAPY_VENDOR_FIND(zgemm, cublasZgemm);
        MDAPY_VENDOR_FIND(dgemm_strided_batched, cublasDgemmStridedBatched);
        MDAPY_VENDOR_FIND(strsm, cublasStrsm);
        MDAPY_VENDOR_FIND(dtrsm, cublasDtrsm);
        MDAPY_VENDOR_FIND(ctrsm, cublasCtrsm);
        MDAPY_VENDOR_FIND(ztrsm, cublasZtrsm);
    });
}

inline const Cufft& cufft() {
    return detail::loaded<Cufft>([](Cufft& table) {
        using namespace detail;
        void* handle = open_or_throw("cufft", MDAPY_SONAME_CUFFT);
        const char* library = MDAPY_SONAME_CUFFT;
        MDAPY_VENDOR_FIND(plan_many, cufftPlanMany);
        MDAPY_VENDOR_FIND(set_stream, cufftSetStream);
        MDAPY_VENDOR_FIND(destroy, cufftDestroy);
        MDAPY_VENDOR_FIND(exec_r2c, cufftExecR2C);
        MDAPY_VENDOR_FIND(exec_c2r, cufftExecC2R);
        MDAPY_VENDOR_FIND(exec_c2c, cufftExecC2C);
        MDAPY_VENDOR_FIND(exec_d2z, cufftExecD2Z);
        MDAPY_VENDOR_FIND(exec_z2d, cufftExecZ2D);
        MDAPY_VENDOR_FIND(exec_z2z, cufftExecZ2Z);
    });
}

inline const Cusolver& cusolver() {
    return detail::loaded<Cusolver>([](Cusolver& table) {
        using namespace detail;
        // what cuSOLVER needs, first
        open_or_throw("nvjitlink", MDAPY_SONAME_NVJITLINK);
        open_or_throw("cusparse", MDAPY_SONAME_CUSPARSE);
        cublas();
        void* handle = open_or_throw("cusolver", MDAPY_SONAME_CUSOLVER);
        const char* library = MDAPY_SONAME_CUSOLVER;
        MDAPY_VENDOR_FIND(create, cusolverDnCreate);
        MDAPY_VENDOR_FIND(destroy, cusolverDnDestroy);
        MDAPY_VENDOR_FIND(set_stream, cusolverDnSetStream);
        MDAPY_VENDOR_FIND(ssyevd_buffer_size, cusolverDnSsyevd_bufferSize);
        MDAPY_VENDOR_FIND(dsyevd_buffer_size, cusolverDnDsyevd_bufferSize);
        MDAPY_VENDOR_FIND(cheevd_buffer_size, cusolverDnCheevd_bufferSize);
        MDAPY_VENDOR_FIND(zheevd_buffer_size, cusolverDnZheevd_bufferSize);
        MDAPY_VENDOR_FIND(ssyevd, cusolverDnSsyevd);
        MDAPY_VENDOR_FIND(dsyevd, cusolverDnDsyevd);
        MDAPY_VENDOR_FIND(cheevd, cusolverDnCheevd);
        MDAPY_VENDOR_FIND(zheevd, cusolverDnZheevd);
        MDAPY_VENDOR_FIND(spotrf_buffer_size, cusolverDnSpotrf_bufferSize);
        MDAPY_VENDOR_FIND(dpotrf_buffer_size, cusolverDnDpotrf_bufferSize);
        MDAPY_VENDOR_FIND(cpotrf_buffer_size, cusolverDnCpotrf_bufferSize);
        MDAPY_VENDOR_FIND(zpotrf_buffer_size, cusolverDnZpotrf_bufferSize);
        MDAPY_VENDOR_FIND(spotrf, cusolverDnSpotrf);
        MDAPY_VENDOR_FIND(dpotrf, cusolverDnDpotrf);
        MDAPY_VENDOR_FIND(cpotrf, cusolverDnCpotrf);
        MDAPY_VENDOR_FIND(zpotrf, cusolverDnZpotrf);
    });
}

#undef MDAPY_VENDOR_FIND

// The three libraries loaded, and the files of all that were (cuSOLVER's
// own libraries too); std::runtime_error saying what to install when one
// cannot be.
inline std::vector<std::string> load_all() {
    cublas();
    cufft();
    cusolver();
    detail::Opened& state = detail::opened();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.files;
}

}  // namespace mdapy::dft::vendor

#endif
