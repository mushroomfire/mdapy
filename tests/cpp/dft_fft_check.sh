#!/bin/bash
# Builds and runs the check of BoxFft against cuFFT from the root of the
# sources, with the CUDA toolkit of the environment E (nvcc, cuFFT):
#   E=/l/wuy33/envs/dft tests/cpp/dft_fft_check.sh [check | bench | ...]
set -eu
E=${E:-/l/wuy33/envs/dft}
export PATH=$E/bin:$PATH
mkdir -p build/fft
for f in src/dft/fft/box_fft.cu tests/cpp/dft_fft_check.cu; do
    o=build/fft/$(basename $f .cu).o
    if [ ! -e $o ] || [ $f -nt $o ] || [ src/dft/fft/box_fft.h -nt $o ] || [ tests/cpp/dft_fft_shapes.inc -nt $o ]; then
        nvcc -O3 -std=c++17 -arch=sm_${ARCH:-86} -Isrc -Itests/cpp -c -o $o $f
    fi
done
nvcc -o build/fft/dft_fft_check build/fft/box_fft.o build/fft/dft_fft_check.o -L$E/lib -lcufft -Xlinker -rpath -Xlinker $E/lib
echo "box_fft.o $(stat -c %s build/fft/box_fft.o) bytes"
[ $# -gt 0 ] && ./build/fft/dft_fft_check "$@"
