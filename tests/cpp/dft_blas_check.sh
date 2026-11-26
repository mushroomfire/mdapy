#!/bin/bash
# Builds and runs the check of MDAPY's linear algebra of the DFT against cuBLAS
# from the root of the sources, with the CUDA toolkit of the environment E:
#   E=/l/wuy33/envs/dft tests/cpp/dft_blas_check.sh [check | bench | ...]
set -eu
E=${E:-/l/wuy33/envs/dft}
export PATH=$E/bin:$PATH
mkdir -p build/blas
FLAGS="-O3 -std=c++17 -arch=sm_${ARCH:-86} -Isrc -Itests/cpp --default-stream per-thread -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 -Wno-deprecated-gpu-targets"
for f in src/dft/blas/blas.cu src/dft/core/check.cpp tests/cpp/dft_blas_check.cu; do
    o=build/blas/$(basename ${f%.*}).o
    if [ ! -e $o ] || [ $f -nt $o ] || [ src/dft/blas/blas.h -nt $o ] || [ tests/cpp/dft_blas_shapes.inc -nt $o ]; then
        nvcc $FLAGS -c -o $o $f
    fi
done
nvcc -o build/blas/dft_blas_check build/blas/blas.o build/blas/check.o build/blas/dft_blas_check.o -L$E/lib -lcublas -lcufft -ldl \
    -Xlinker -rpath -Xlinker $E/lib
echo "blas.o $(stat -c %s build/blas/blas.o) bytes"
[ $# -gt 0 ] && ./build/blas/dft_blas_check "$@"
