// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Batched 3D FFTs of boxes of n1 x n2 x n3 points, the third index the fastest,
// written for the plane-wave DFT alone: sizes that are products of 2, 3, 5 and
// 7, boxes that follow one another, complex transforms in place and real ones
// out of place, in double and single precision. The conventions are those of
// cuFFT: unnormalized, exp(-i ...) forward and exp(+i ...) inverse, and a real
// box of n1 n2 n3 values has n1 n2 (n3 / 2 + 1) complex components.
//
// Complex numbers are pairs (real, imaginary) of the real type, the layout of
// std::complex, cuDoubleComplex and hipDoubleComplex.
//
// One kernel transforms one axis: a block loads whole lines into shared memory,
// runs every pass of radix 2, 3, 4, 5, 7 or 8 there and stores them, so that a
// box is read and written three times. The tables of a box (roots of unity and
// orders) do not depend on the number of boxes.

#pragma once

#include <array>
#include <cstddef>

namespace mdapy::dft {

class BoxFft {
public:
    // Whether every size is a product of 2, 3, 5 and 7 that fits a block.
    static bool supported(const std::array<int, 3>& grid) noexcept;

    explicit BoxFft(const std::array<int, 3>& grid);
    ~BoxFft();
    BoxFft(const BoxFft&) = delete;
    BoxFft& operator=(const BoxFft&) = delete;

    std::size_t points() const noexcept;    // n1 n2 n3
    std::size_t spectrum() const noexcept;  // n1 n2 (n3 / 2 + 1)

    // count complex boxes in place: 2 points() reals each. With window >= 0,
    // boxes of which only the planes of the first index within window of 0
    // matter (0 to window and n1 - window to n1 - 1), as those of the sphere
    // of the bands: inverse reads those planes alone, the others taken as
    // zeros and never read, and forward writes those alone, the others left
    // as the transform along the first axis made them; about a third less
    // work and memory traffic with the sphere of the bands in a box of the
    // density.
    void forward(double* boxes, int count, int window = -1);
    void inverse(double* boxes, int count, int window = -1);
    void forward(float* boxes, int count, int window = -1);
    void inverse(float* boxes, int count, int window = -1);

    // count real boxes of points() values to spectrum() complex components and
    // back; inverse overwrites its spectrum.
    void forward(const double* values, double* spectrum, int count);
    void inverse(double* spectrum, double* values, int count);
    void forward(const float* values, float* spectrum, int count);
    void inverse(float* spectrum, float* values, int count);

    // Lines per block (a power of two, as its logarithm) and threads per block
    // (a multiple of 64) of the kernels of a precision (0 double, 1 single) and
    // a kind of kernel (0 strided, 1 contiguous, 2 planes, of which the threads
    // alone), instead of the defaults; zeros give the defaults back. For
    // measurements.
    void tune(int precision, int kind, int log_lines, int threads);
    // Only the axes of a mask (1 the first, 2 the second, 4 the third) are
    // transformed; with 8 no pass is run (loads and stores alone), with 16 the
    // second and the third axis each in its own kernel. For measurements.
    void only_axes(int mask) { axes_ = mask; }
    // The planes that take up to this many bytes of shared memory go through
    // one kernel (never more than a block of the GPU can have). For
    // measurements.
    void set_plane_bytes(std::size_t bytes);

private:
    struct Tables;
    template <class R>
    void complex_boxes(R* boxes, int count, int sign);
    template <class R>
    void complex_window(R* boxes, int count, int sign, int window);
    template <class R>
    bool plane_fits() const;
    template <class R>
    void real_forward(const R* values, R* spectrum, int count);
    template <class R>
    void real_inverse(R* spectrum, R* values, int count);
    template <class R, int Sign>
    void strided(R* boxes, int axis, long long planes, long long columns, long long stride, int window = -1);
    template <class R, int Sign, int Mode>
    void contiguous(const R* in, R* out, long long lines);
    template <class R, int Sign>
    bool planes(R* boxes, long long count, int window = -1);
    template <class R, int Sign>
    bool real_planes(const R* in, R* out, long long count);

    std::array<int, 3> grid_;
    Tables* tables_[3] = {nullptr, nullptr, nullptr};  // of each axis; equal sizes share one
    Tables* half_ = nullptr;                           // of half the third size, when it is even
    int tuned_[2][3][2] = {};
    int axes_ = 7;
    std::size_t most_shared_ = 0;  // bytes of shared memory a block of this GPU can have
    std::size_t plane_bytes_ = 0;  // the largest plane that goes through one kernel
    bool narrow_ = false;          // a V100: smaller blocks of the lines of double precision
    int threads_[2][2][2] = {};    // of the blocks of the planes, by precision, direction and complex or real; 0 not asked yet
};

}  // namespace mdapy::dft
