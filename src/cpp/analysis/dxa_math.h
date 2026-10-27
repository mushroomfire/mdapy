// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Small vectors and matrices of the dislocation analysis.

#pragma once

#include <array>
#include <cmath>
#include <optional>

namespace mdapy::analysis::dxa {

using Vec = std::array<double, 3>;
using Mat = std::array<double, 9>;  // row-major

// matrices made of ideal vectors agree to this much when they are the same
inline constexpr double kTolerance = 1e-3;

inline Vec turn(const Mat& m, const Vec& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

inline Mat multiply(const Mat& a, const Mat& b) {
    Mat c{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                c[3 * i + j] += a[3 * i + k] * b[3 * k + j];
            }
        }
    }
    return c;
}

inline Mat transpose(const Mat& m) { return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]}; }

inline double determinant(const Mat& m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

inline std::optional<Mat> inverse(const Mat& m) {
    const double d = determinant(m);
    if (std::abs(d) < 1e-12) {
        return std::nullopt;
    }
    return Mat{(m[4] * m[8] - m[5] * m[7]) / d, (m[2] * m[7] - m[1] * m[8]) / d,
               (m[1] * m[5] - m[2] * m[4]) / d, (m[5] * m[6] - m[3] * m[8]) / d,
               (m[0] * m[8] - m[2] * m[6]) / d, (m[2] * m[3] - m[0] * m[5]) / d,
               (m[3] * m[7] - m[4] * m[6]) / d, (m[1] * m[6] - m[0] * m[7]) / d,
               (m[0] * m[4] - m[1] * m[3]) / d};
}

// the matrix with these columns
inline Mat columns(const Vec& a, const Vec& b, const Vec& c) {
    return {a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]};
}

inline bool close(const Mat& a, const Mat& b) {
    for (int k = 0; k < 9; ++k) {
        if (std::abs(a[k] - b[k]) > kTolerance) {
            return false;
        }
    }
    return true;
}

inline bool close(const Vec& a, const Vec& b) {
    return std::abs(a[0] - b[0]) < kTolerance && std::abs(a[1] - b[1]) < kTolerance &&
           std::abs(a[2] - b[2]) < kTolerance;
}

inline Vec minus(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

inline Vec add(const Vec& a, const Vec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }

inline Vec scaled(const Vec& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }

inline Vec cross(const Vec& a, const Vec& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline const Mat kIdentity = {1, 0, 0, 0, 1, 0, 0, 0, 1};

}  // namespace mdapy::analysis::dxa
