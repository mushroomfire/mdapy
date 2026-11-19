// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The cubic splines of tabulated potentials on the device, the coefficients
// of calculators::Spline in a precision, evaluated as it evaluates them.

#pragma once

#include <vector>

#include "calculators/eam.h"
#include "cuda/device.cuh"

namespace mdapy::cuda {

// The splines on the device: rows of (c3, c4, c5, c6) for the value and
// (c0, c1, c2, 0) for the slope, those of spline s from row[s].
template <class Real>
struct Splines {
    const Real4<Real>* value;
    const Real4<Real>* slope;
    const int* row;
    const int* size;
    const Real* inverse;
};

template <class Real>
__device__ inline Real spline_value(const Splines<Real>& t, int s, Real x) {
    Real p = x * t.inverse[s] + Real(1);
    int m = static_cast<int>(p);
    m = max(1, min(m, t.size[s] - 1));
    p -= static_cast<Real>(m);
    p = min(p, Real(1));
    const Real4<Real> c = t.value[t.row[s] + m];
    return ((c.x * p + c.y) * p + c.z) * p + c.w;
}

template <class Real>
__device__ inline void spline(const Splines<Real>& t, int s, Real x, Real& value, Real& slope) {
    Real p = x * t.inverse[s] + Real(1);
    int m = static_cast<int>(p);
    m = max(1, min(m, t.size[s] - 1));
    p -= static_cast<Real>(m);
    p = min(p, Real(1));
    const Real4<Real> c = t.value[t.row[s] + m];
    const Real4<Real> d = t.slope[t.row[s] + m];
    value = ((c.x * p + c.y) * p + c.z) * p + c.w;
    slope = (d.x * p + d.y) * p + d.z;
}

// The rows of splines of the host, in a precision.
template <class Real>
struct Rows {
    std::vector<Real4<Real>> value, slope;
    std::vector<int> row, size;
    std::vector<Real> inverse;

    void add(const calculators::Spline& s) {
        row.push_back(static_cast<int>(value.size()));
        size.push_back(s.size());
        inverse.push_back(static_cast<Real>(s.inverse_delta()));
        const auto* c = s.coefficients();
        for (int m = 0; m <= s.size(); ++m) {
            Real4<Real> v, d;
            v.x = static_cast<Real>(c[m][3]);
            v.y = static_cast<Real>(c[m][4]);
            v.z = static_cast<Real>(c[m][5]);
            v.w = static_cast<Real>(c[m][6]);
            d.x = static_cast<Real>(c[m][0]);
            d.y = static_cast<Real>(c[m][1]);
            d.z = static_cast<Real>(c[m][2]);
            d.w = Real(0);
            value.push_back(v);
            slope.push_back(d);
        }
    }
};

// The splines of rows on the device.
template <class Real>
struct DeviceRows {
    Buffer<Real4<Real>> value, slope;
    Buffer<int> row, size;
    Buffer<Real> inverse;

    void upload(const Rows<Real>& r) {
        value.upload(r.value.data(), r.value.size());
        slope.upload(r.slope.data(), r.slope.size());
        row.upload(r.row.data(), r.row.size());
        size.upload(r.size.data(), r.size.size());
        inverse.upload(r.inverse.data(), r.inverse.size());
    }
    Splines<Real> view() const {
        return {value.data(), slope.data(), row.data(), size.data(), inverse.data()};
    }
};

}  // namespace mdapy::cuda
