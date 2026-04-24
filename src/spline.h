// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
#pragma once
#include "type.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

namespace nb = nanobind;

// Cubic spline on an arbitrary strictly-increasing grid.
//
// Given data points (x_i, y_i), i = 0..n-1, constructs a piecewise-cubic
// s(x) that is C2 over the whole range, reproduces y_i at each x_i, and
// satisfies the chosen boundary condition at the two endpoints.
//
// Boundary conditions (matching scipy.interpolate.CubicSpline):
//   - NotAKnot (default): third derivative continuous at x_1 and x_{n-2}
//     (equivalently, the first two cubic pieces are the same polynomial,
//     and the last two also). Same default as scipy. Best when you have
//     no prior knowledge of the endpoint slopes.
//   - Natural:   s''(x_0) = s''(x_{n-1}) = 0. Produces the minimum-curvature
//     interpolant; straight lines at the ends.
//   - Clamped:   s'(x_0) = dy0, s'(x_{n-1}) = dyn. Derivatives are either
//     user-specified or estimated by a three-point formula at each end.
//
// Internally this stores y''_i at each knot (the "y2" array), then evaluates
// s(x) in O(log n) per point using the standard formula from Numerical
// Recipes §3.3.
class CubicSpline
{
public:
    enum class BCType { NotAKnot, Natural, Clamped };

    CubicSpline() = default;

    // Default: not-a-knot (matches scipy).
    CubicSpline(const ROneArrayD x_py,
                const ROneArrayD y_py,
                BCType bc_type = BCType::NotAKnot)
    {
        init_common(x_py.data(), y_py.data(), x_py.shape(0), y_py.shape(0));
        bc_type_ = bc_type;
        if (bc_type_ == BCType::Clamped) {
            estimate_endpoint_derivatives_3pt();
        }
        compute_second_derivatives();
    }

    // Clamped with user-supplied endpoint derivatives.
    CubicSpline(const ROneArrayD x_py,
                const ROneArrayD y_py,
                double dy0,
                double dyn)
    {
        init_common(x_py.data(), y_py.data(), x_py.shape(0), y_py.shape(0));
        bc_type_ = BCType::Clamped;
        dy0_ = dy0;
        dyn_ = dyn;
        compute_second_derivatives();
    }

    double operator()(double x) const { return evaluate(x); }

    // ========== Single-point evaluation (thread-safe) ==========
    double evaluate(double x) const
    {
        check_range(x);
        size_t i = find_interval(x);
        double h = x_[i + 1] - x_[i];
        double A = (x_[i + 1] - x) / h;
        double B = (x - x_[i]) / h;
        return A * y_[i] + B * y_[i + 1] +
               ((A * A * A - A) * y2_[i] + (B * B * B - B) * y2_[i + 1]) * (h * h) / 6.0;
    }

    double derivative(double x) const
    {
        check_range(x);
        size_t i = find_interval(x);
        double h = x_[i + 1] - x_[i];
        double A = (x_[i + 1] - x) / h;
        double B = (x - x_[i]) / h;
        return (y_[i + 1] - y_[i]) / h +
               ((3.0 * B * B - 1.0) * y2_[i + 1] - (3.0 * A * A - 1.0) * y2_[i]) * h / 6.0;
    }

    double second_derivative(double x) const
    {
        check_range(x);
        size_t i = find_interval(x);
        double h = x_[i + 1] - x_[i];
        double A = (x_[i + 1] - x) / h;
        double B = (x - x_[i]) / h;
        // s''(x) is piecewise-linear between the knots' y2 values.
        return A * y2_[i] + B * y2_[i + 1];
    }

    // ========== Batch evaluation — std::vector (C++ use, OpenMP) ==========
    std::vector<double> evaluate(const std::vector<double> &x_vals) const
    {
        size_t N = x_vals.size();
        std::vector<double> result(N);
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            double x = x_vals[i];
            if (x < x_[0] || x > x_[n_ - 1]) { result[i] = std::numeric_limits<double>::quiet_NaN(); continue; }
            result[i] = evaluate(x);
        }
        return result;
    }

    std::vector<double> derivative(const std::vector<double> &x_vals) const
    {
        size_t N = x_vals.size();
        std::vector<double> result(N);
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            double x = x_vals[i];
            if (x < x_[0] || x > x_[n_ - 1]) { result[i] = std::numeric_limits<double>::quiet_NaN(); continue; }
            result[i] = derivative(x);
        }
        return result;
    }

    std::vector<double> second_derivative(const std::vector<double> &x_vals) const
    {
        size_t N = x_vals.size();
        std::vector<double> result(N);
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            double x = x_vals[i];
            if (x < x_[0] || x > x_[n_ - 1]) { result[i] = std::numeric_limits<double>::quiet_NaN(); continue; }
            result[i] = second_derivative(x);
        }
        return result;
    }

    // ========== Batch evaluation — numpy (Python, OpenMP) ==========
    nb::ndarray<nb::numpy, double> evaluate(const ROneArrayD &x_vals) const
    {
        return batch_numpy(x_vals, [this](double x) { return evaluate(x); });
    }
    nb::ndarray<nb::numpy, double> derivative(const ROneArrayD &x_vals) const
    {
        return batch_numpy(x_vals, [this](double x) { return derivative(x); });
    }
    nb::ndarray<nb::numpy, double> second_derivative(const ROneArrayD &x_vals) const
    {
        return batch_numpy(x_vals, [this](double x) { return second_derivative(x); });
    }

private:
    void init_common(const double *x, const double *y, size_t nx, size_t ny)
    {
        if (nx != ny)
            throw std::invalid_argument("x and y must have the same length");
        if (nx < 2)
            throw std::invalid_argument("Need at least 2 points for interpolation");
        for (size_t i = 0; i + 1 < nx; ++i)
            if (x[i] >= x[i + 1])
                throw std::invalid_argument("x must be strictly increasing");
        n_ = nx;
        x_.assign(x, x + nx);
        y_.assign(y, y + nx);
    }

    inline void check_range(double x) const
    {
        if (x < x_[0] || x > x_[n_ - 1])
            throw std::out_of_range("x is outside interpolation range");
    }

    // Binary search for the interval [x_[i], x_[i+1]] that contains x.
    size_t find_interval(double x) const
    {
        auto it = std::lower_bound(x_.begin(), x_.end(), x);
        if (it == x_.end()) return n_ - 2;
        size_t idx = std::distance(x_.begin(), it);
        if (idx == 0) return 0;
        return idx - 1;
    }

    template <typename F>
    nb::ndarray<nb::numpy, double> batch_numpy(const ROneArrayD &x_vals, F evaluator) const
    {
        size_t N = x_vals.shape(0);
        const double *x_data = x_vals.data();
        double *res_data = new double[N];
        nb::capsule res_owner(res_data, [](void *p) noexcept { delete[] (double *)p; });
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            double x = x_data[i];
            if (x < x_[0] || x > x_[n_ - 1]) { res_data[i] = std::numeric_limits<double>::quiet_NaN(); continue; }
            res_data[i] = evaluator(x);
        }
        return nb::ndarray<nb::numpy, double>(res_data, {N}, res_owner);
    }

    // Three-point formula for the endpoint derivatives, used when BC is Clamped
    // and the user did not supply dy0/dyn. Fits a quadratic through the first
    // three (or last three) points and takes its analytic derivative at the
    // endpoint.
    void estimate_endpoint_derivatives_3pt()
    {
        if (n_ < 3) {
            // Only two points — fall back to the secant.
            double h0 = x_[1] - x_[0];
            dy0_ = (y_[1] - y_[0]) / h0;
            dyn_ = dy0_;
            return;
        }
        double h0 = x_[1] - x_[0];
        double h1 = x_[2] - x_[1];
        dy0_ = (-(2.0 * h0 + h1) * y_[0] + (h0 + h1) * (h0 + h1) / h1 * y_[1]
                - h0 * h0 / h1 * y_[2]) / (h0 * (h0 + h1));
        double hm2 = x_[n_ - 2] - x_[n_ - 3];
        double hm1 = x_[n_ - 1] - x_[n_ - 2];
        dyn_ = (hm1 * hm1 / hm2 * y_[n_ - 3]
                - (hm1 + hm2) * (hm1 + hm2) / hm2 * y_[n_ - 2]
                + (hm1 + 2.0 * hm2) * y_[n_ - 1]) / (hm1 * (hm1 + hm2));
    }

    // Standard tridiagonal Thomas solve. Overwrites c and d.
    static void thomas(std::vector<double> &a, std::vector<double> &b,
                       std::vector<double> &c, std::vector<double> &d,
                       std::vector<double> &x)
    {
        size_t N = b.size();
        c[0] /= b[0];
        d[0] /= b[0];
        for (size_t i = 1; i < N; ++i) {
            double m = b[i] - a[i] * c[i - 1];
            if (i < N - 1) c[i] /= m;
            d[i] = (d[i] - a[i] * d[i - 1]) / m;
        }
        x[N - 1] = d[N - 1];
        for (int i = static_cast<int>(N) - 2; i >= 0; --i)
            x[i] = d[i] - c[i] * x[i + 1];
    }

    void compute_second_derivatives()
    {
        y2_.assign(n_, 0.0);

        if (n_ == 2) {
            // Linear interpolation — y2 stays at 0.
            return;
        }

        std::vector<double> h(n_ - 1);
        for (size_t i = 0; i + 1 < n_; ++i) h[i] = x_[i + 1] - x_[i];

        if (bc_type_ == BCType::Natural || (bc_type_ == BCType::NotAKnot && n_ < 4))
        {
            // Natural: y2[0] = y2[n-1] = 0. For NotAKnot with n < 4 the BC at
            // x_1 and x_{n-2} become the same equation, leaving the spline
            // underdetermined; fall back to natural.
            solve_natural(h);
        }
        else if (bc_type_ == BCType::Clamped)
        {
            solve_clamped(h);
        }
        else   // NotAKnot, n >= 4
        {
            solve_not_a_knot(h);
        }
    }

    void solve_natural(const std::vector<double> &h)
    {
        std::vector<double> a(n_, 0.0), b(n_, 0.0), c(n_, 0.0), d(n_, 0.0);
        b[0] = 1.0;  // y2[0] = 0
        b[n_ - 1] = 1.0;  // y2[n-1] = 0
        for (size_t i = 1; i + 1 < n_; ++i) {
            a[i] = h[i - 1];
            b[i] = 2.0 * (h[i - 1] + h[i]);
            c[i] = h[i];
            d[i] = 6.0 * ((y_[i + 1] - y_[i]) / h[i] - (y_[i] - y_[i - 1]) / h[i - 1]);
        }
        thomas(a, b, c, d, y2_);
    }

    void solve_clamped(const std::vector<double> &h)
    {
        std::vector<double> a(n_, 0.0), b(n_, 0.0), c(n_, 0.0), d(n_, 0.0);
        b[0] = 2.0 * h[0];
        c[0] = h[0];
        d[0] = 6.0 * ((y_[1] - y_[0]) / h[0] - dy0_);
        for (size_t i = 1; i + 1 < n_; ++i) {
            a[i] = h[i - 1];
            b[i] = 2.0 * (h[i - 1] + h[i]);
            c[i] = h[i];
            d[i] = 6.0 * ((y_[i + 1] - y_[i]) / h[i] - (y_[i] - y_[i - 1]) / h[i - 1]);
        }
        a[n_ - 1] = h[n_ - 2];
        b[n_ - 1] = 2.0 * h[n_ - 2];
        d[n_ - 1] = 6.0 * (dyn_ - (y_[n_ - 1] - y_[n_ - 2]) / h[n_ - 2]);
        thomas(a, b, c, d, y2_);
    }

    // Not-a-knot: requires n >= 4. The "third derivative continuous at x_1"
    // BC involves y2[0], y2[1], y2[2] — three variables, not tridiagonal.
    // Use the BC to eliminate y2[0] (and symmetrically y2[n-1]) from the
    // interior equations at i=1 and i=n-2, giving a tridiagonal system of
    // size (n-2) for the interior values y2[1..n-2]. Then back out the
    // endpoints via the BC formulas.
    void solve_not_a_knot(const std::vector<double> &h)
    {
        const size_t M = n_ - 2;  // reduced system size
        std::vector<double> a(M, 0.0), b(M, 0.0), c(M, 0.0), d(M, 0.0);

        // First reduced row. After substituting
        //   y2[0] = ((h[0]+h[1])*y2[1] - h[0]*y2[2]) / h[1]
        // into the interior equation at i=1 and multiplying through by h[1]:
        //   (h[0]+h[1])*(h[0]+2*h[1]) * y2[1] + (h[1]^2 - h[0]^2) * y2[2]
        //     = 6 * h[1] * slope_diff[1]
        {
            double h0 = h[0], h1 = h[1];
            b[0] = (h0 + h1) * (h0 + 2.0 * h1);
            c[0] = h1 * h1 - h0 * h0;
            d[0] = 6.0 * h1 * ((y_[2] - y_[1]) / h1 - (y_[1] - y_[0]) / h0);
        }

        // Standard interior rows for i = 2..n-3 (reduced index j = 1..M-2).
        for (size_t j = 1; j + 1 < M; ++j) {
            size_t i = j + 1;
            a[j] = h[i - 1];
            b[j] = 2.0 * (h[i - 1] + h[i]);
            c[j] = h[i];
            d[j] = 6.0 * ((y_[i + 1] - y_[i]) / h[i] - (y_[i] - y_[i - 1]) / h[i - 1]);
        }

        // Last reduced row. Symmetric to the first: substitute
        //   y2[n-1] = ((h[n-3]+h[n-2])*y2[n-2] - h[n-2]*y2[n-3]) / h[n-3]
        // into the interior equation at i=n-2 and multiply by h[n-3]:
        //   (h[n-3]^2 - h[n-2]^2) * y2[n-3]
        //     + (h[n-3]+h[n-2])*(2*h[n-3]+h[n-2]) * y2[n-2]
        //     = 6 * h[n-3] * slope_diff[n-2]
        {
            size_t i = n_ - 2;
            double hm3 = h[i - 1], hm2 = h[i];  // h[n-3] and h[n-2]
            a[M - 1] = hm3 * hm3 - hm2 * hm2;
            b[M - 1] = (hm3 + hm2) * (2.0 * hm3 + hm2);
            d[M - 1] = 6.0 * hm3 * ((y_[i + 1] - y_[i]) / hm2 - (y_[i] - y_[i - 1]) / hm3);
        }

        std::vector<double> y2_inner(M);
        thomas(a, b, c, d, y2_inner);

        for (size_t j = 0; j < M; ++j) y2_[j + 1] = y2_inner[j];
        y2_[0] = ((h[0] + h[1]) * y2_[1] - h[0] * y2_[2]) / h[1];
        y2_[n_ - 1] = ((h[n_ - 3] + h[n_ - 2]) * y2_[n_ - 2]
                       - h[n_ - 2] * y2_[n_ - 3]) / h[n_ - 3];
    }

    size_t n_ = 0;
    std::vector<double> x_;  // strictly-increasing knot positions
    std::vector<double> y_;  // knot values
    std::vector<double> y2_; // second derivatives at knots
    BCType bc_type_ = BCType::NotAKnot;
    double dy0_ = 0.0;       // left endpoint first derivative (Clamped only)
    double dyn_ = 0.0;       // right endpoint first derivative (Clamped only)
};

// Uniform-grid cubic Hermite spline, LAMMPS/GPUMD convention.
//
// Grid: x_m = m * h for m = 0..n-1.
// Derivatives at grid points are estimated by centered finite differences
// in the normalized coordinate p = (x - x_m)/h (so dp per grid step = 1):
//   fp[0]   = y[1]-y[0]
//   fp[1]   = (y[2]-y[0])/2
//   fp[m]   = ((y[m-2]-y[m+2]) + 8*(y[m+1]-y[m-1]))/12   for 2 <= m <= n-3
//   fp[n-2] = (y[n-1]-y[n-3])/2
//   fp[n-1] = y[n-1]-y[n-2]
//
// On interval m in [0, n-2], the cubic is
//   f(x) = a[m] + b[m]*dx + c[m]*dx^2 + d[m]*dx^3,  dx = x - x_m
// with
//   a[m] = y[m]
//   b[m] = fp[m] / h
//   c[m] = (3*(y[m+1]-y[m]) - 2*fp[m] - fp[m+1]) / h^2
//   d[m] = (fp[m] + fp[m+1] - 2*(y[m+1]-y[m])) / h^3
//
// This matches LAMMPS' `interpolate` routine in `pair_eam.cpp` and GPUMD's
// `compute_lammps_spline`. Lookup is O(1).
class UniformCubicSpline
{
public:
    UniformCubicSpline() = default;

    UniformCubicSpline(double h, const std::vector<double> &y)
        : h_(h), n_(static_cast<int>(y.size()))
    {
        if (h <= 0.0)
        {
            throw std::invalid_argument("h must be positive");
        }
        if (n_ < 2)
        {
            throw std::invalid_argument("Need at least 2 points for interpolation");
        }

        std::vector<double> fp(n_, 0.0);
        fp[0] = y[1] - y[0];
        fp[n_ - 1] = y[n_ - 1] - y[n_ - 2];
        if (n_ >= 3)
        {
            fp[1] = 0.5 * (y[2] - y[0]);
            fp[n_ - 2] = 0.5 * (y[n_ - 1] - y[n_ - 3]);
        }
        for (int m = 2; m <= n_ - 3; ++m)
        {
            fp[m] = ((y[m - 2] - y[m + 2]) + 8.0 * (y[m + 1] - y[m - 1])) / 12.0;
        }

        a_.assign(n_, 0.0);
        b_.assign(n_, 0.0);
        c_.assign(n_, 0.0);
        d_.assign(n_, 0.0);

        const double inv_h = 1.0 / h_;
        const double inv_h2 = inv_h * inv_h;
        const double inv_h3 = inv_h2 * inv_h;

        for (int m = 0; m <= n_ - 2; ++m)
        {
            double dy = y[m + 1] - y[m];
            double B2 = 3.0 * dy - 2.0 * fp[m] - fp[m + 1];
            double B3 = fp[m] + fp[m + 1] - 2.0 * dy;
            a_[m] = y[m];
            b_[m] = fp[m] * inv_h;
            c_[m] = B2 * inv_h2;
            d_[m] = B3 * inv_h3;
        }
        // Last grid point: linear extrapolation slope (used only when the
        // query is clamped exactly to the right edge).
        a_[n_ - 1] = y[n_ - 1];
        b_[n_ - 1] = fp[n_ - 1] * inv_h;
        c_[n_ - 1] = 0.0;
        d_[n_ - 1] = 0.0;
    }

    // Locate interval and the in-interval offset dx.
    //
    // For x in [m*h, (m+1)*h], 0 <= m <= n-2: the normal Hermite cubic.
    // For x > (n-1)*h: use the m = n-1 slot, whose coefficients are set to
    //   (y[n-1], fp[n-1]/h, 0, 0) — a LINEAR extrapolation. This matches
    //   LAMMPS' behavior in ``pair_eam.cpp`` for values past the tabulated
    //   range (e.g. rho slightly above rho_max for a highly-compressed atom).
    //   Without this, a cubic polynomial extrapolated from the last interval
    //   would swing wildly (we saw F(1.08) flip from +19.8 to -1.3 for Cr).
    // For x < 0: clamp to y[0] (dx = 0 at the m=0 slot).
    inline int locate(double x, double &dx) const
    {
        int m = static_cast<int>(x / h_);
        if (m < 0) { m = 0; dx = 0.0; return m; }
        if (m > n_ - 1) m = n_ - 1;
        dx = x - m * h_;
        return m;
    }

    inline double evaluate(double x) const
    {
        double dx;
        int m = locate(x, dx);
        return a_[m] + (b_[m] + (c_[m] + d_[m] * dx) * dx) * dx;
    }

    inline double derivative(double x) const
    {
        double dx;
        int m = locate(x, dx);
        return b_[m] + (2.0 * c_[m] + 3.0 * d_[m] * dx) * dx;
    }

    // Combined evaluation (saves one locate() call at each neighbor lookup).
    inline void evaluate_and_derivative(double x, double &val, double &deriv) const
    {
        double dx;
        int m = locate(x, dx);
        val = a_[m] + (b_[m] + (c_[m] + d_[m] * dx) * dx) * dx;
        deriv = b_[m] + (2.0 * c_[m] + 3.0 * d_[m] * dx) * dx;
    }

    inline double h() const { return h_; }
    inline int n() const { return n_; }

private:
    double h_ = 0.0;
    int n_ = 0;
    std::vector<double> a_, b_, c_, d_;
};