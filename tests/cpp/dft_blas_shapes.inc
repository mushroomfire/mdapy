// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The products of the DFT that take its time, from whole calculations run with
// the profile of Blas (-DMDAPY_DFT_BLAS_PROFILE): c (m, n) = op(a) (m, k) op(b)
// (k, n) in double (z complex, d real, the bands of Gamma) and single (c, s).

namespace {

struct Shape {
    const char* what;
    char type;  // z c d s
    Op ta, tb;
    int m, n, k;
};

const std::vector<Shape>& dft_shapes() {
    static const std::vector<Shape> shapes = {
        {"Al32 15 Ha: projections of the bands", 'z', Op::C, Op::N, 576, 58, 9890},
        {"Al32 15 Ha: projections of the bands", 'c', Op::C, Op::N, 576, 58, 9890},
        {"Al32: projected back on the bands", 'z', Op::N, Op::N, 9890, 58, 576},
        {"Al32: projected back on the bands", 'c', Op::N, Op::N, 9890, 58, 576},
        {"Al32: rotation of the basis", 'z', Op::N, Op::N, 9890, 58, 116},
        {"Al32: rotation of the basis", 'c', Op::N, Op::N, 9890, 58, 116},
        {"Al32: overlaps of the bands", 'z', Op::C, Op::N, 58, 58, 9890},
        {"Al32: overlaps of the bands", 'c', Op::C, Op::N, 58, 58, 9890},
        {"Al32: subspace of two blocks", 'z', Op::C, Op::N, 116, 58, 9890},
        {"Al32: subspace of two blocks", 'c', Op::C, Op::N, 116, 58, 9890},
        {"Fe16 45 Ha: projections", 'z', Op::C, Op::N, 288, 154, 17640},
        {"Fe16 45 Ha: projections", 'c', Op::C, Op::N, 288, 154, 17640},
        {"Fe16: overlaps of the bands", 'z', Op::C, Op::N, 154, 154, 17640},
        {"Fe16: overlaps of the bands", 'c', Op::C, Op::N, 154, 154, 17640},
        {"Fe16: rotation of the basis", 'z', Op::N, Op::N, 17640, 154, 154},
        {"Fe16: rotation of the basis", 'c', Op::N, Op::N, 17640, 154, 154},
        {"Fe16: rotation of two blocks", 'z', Op::N, Op::N, 17640, 154, 308},
        {"Fe16: rotation of two blocks", 'c', Op::N, Op::N, 17640, 154, 308},
        {"Fe16: projected back", 'z', Op::N, Op::N, 17640, 154, 288},
        {"Fe16: projected back", 'c', Op::N, Op::N, 17640, 154, 288},
        {"Si64 Gamma (real): projections", 'd', Op::T, Op::N, 1152, 141, 31462},
        {"Si64 Gamma (real): projections", 's', Op::T, Op::N, 1152, 141, 31462},
        {"Si64 Gamma: projected back", 'd', Op::N, Op::N, 31464, 141, 1152},
        {"Si64 Gamma: projected back", 's', Op::N, Op::N, 31464, 141, 1152},
        {"Si64 Gamma: rotation of two blocks", 'd', Op::N, Op::N, 31464, 141, 282},
        {"Si64 Gamma: rotation of two blocks", 's', Op::N, Op::N, 31464, 141, 282},
        {"Si64 Gamma: overlaps of the bands", 'd', Op::T, Op::N, 141, 141, 31462},
        {"Si64 Gamma: overlaps of the bands", 's', Op::T, Op::N, 141, 141, 31462},
        {"Al108 Gamma (real): projections", 'd', Op::T, Op::N, 1944, 195, 33370},
        {"Al108 Gamma (real): projections", 's', Op::T, Op::N, 1944, 195, 33370},
        {"Al108 Gamma: projected back", 'd', Op::N, Op::N, 33372, 195, 1944},
        {"Al108 Gamma: projected back", 's', Op::N, Op::N, 33372, 195, 1944},
        {"Al108 Gamma: rotation of two blocks", 'd', Op::N, Op::N, 33372, 195, 390},
        {"Al108 Gamma: rotation of two blocks", 's', Op::N, Op::N, 33372, 195, 390},
        {"Al108 Gamma: overlaps of the bands", 'd', Op::T, Op::N, 195, 195, 33370},
        {"Al108 Gamma: overlaps of the bands", 's', Op::T, Op::N, 195, 195, 33370},
        {"Al108 Gamma: strengths on the projections", 'd', Op::N, Op::N, 1944, 195, 1944},
        {"Al108 Gamma: strengths on the projections", 's', Op::N, Op::N, 1944, 195, 1944},
        {"few bands: projections", 'z', Op::C, Op::N, 576, 13, 9890},
        {"few bands: projections", 'd', Op::C, Op::N, 576, 13, 9890},
        {"few bands: rotation", 'z', Op::N, Op::N, 9890, 13, 576},
        {"few bands: rotation", 'd', Op::N, Op::N, 9890, 13, 576},
    };
    return shapes;
}

}  // namespace
