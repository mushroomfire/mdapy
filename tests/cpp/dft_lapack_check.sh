#!/bin/bash
# Builds and runs the times of MDAPY's eigensolver and Cholesky factorization of
# the DFT against cuSOLVER from the root of the sources, with the CUDA toolkit
# of the environment E:
#   E=/l/wuy33/envs/dft tests/cpp/dft_lapack_check.sh [repeats] [types] [sizes]
# PROFILE=1 builds with the spans of time.
set -eu
E=${E:-/l/wuy33/envs/dft}
export PATH=$E/bin:$PATH
B=build/lapack${PROFILE:+_profile}
mkdir -p $B
FLAGS="-O3 -std=c++17 -arch=sm_${ARCH:-86} -Isrc -Itests/cpp --default-stream per-thread -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 -Wno-deprecated-gpu-targets ${PROFILE:+-DMDAPY_DFT_PROFILE}"
for f in src/dft/blas/blas.cu src/dft/core/check.cpp src/dft/lapack/lapack.cu src/dft/lapack/host.cpp src/dft/lapack/divide.cpp tests/cpp/dft_lapack_check.cu; do
    o=$B/$(basename ${f%.*}).o
    if [ ! -e $o ] || [ -n "$(find src/dft tests/cpp/dft_lapack_check.cu -newer $o -print -quit)" ]; then
        nvcc $FLAGS -c -o $o $f &
    fi
done
wait
L=${LIB:-$E/lib}
nvcc -o $B/dft_lapack_check $B/*.o -L$L -lcusolver -lcublas -lcufft -ldl -Xlinker -rpath -Xlinker $L
./$B/dft_lapack_check "$@"
