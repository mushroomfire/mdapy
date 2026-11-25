// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The products of the DFT that take its time, from runs of whole calculations.

namespace {

struct Shape {
    const char* what;
    char type;  // z c d s
    Op ta, tb;
    int m, n, k;
};

const std::vector<Shape>& dft_shapes() {
    static const std::vector<Shape> shapes = {
        {"placeholder", 'z', Op::C, Op::N, 200, 200, 20000},
        {"placeholder", 'z', Op::N, Op::N, 20000, 100, 200},
    };
    return shapes;
}

}  // namespace
