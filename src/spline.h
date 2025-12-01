#pragma once
#include "type.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

namespace nb = nanobind;

class CubicSpline
{
public:
    CubicSpline() = default;

    // 构造函数，支持非等间距的 x（只要递增即可）- numpy array 版本
    CubicSpline(const ROneArrayD x_py,
                const ROneArrayD y_py)
    {
        if (x_py.shape(0) != y_py.shape(0))
        {
            throw std::invalid_argument("x and y must have the same length");
        }

        size_t n = x_py.shape(0);
        const double *x = x_py.data();
        const double *y = y_py.data();
        if (n < 2)
        {
            throw std::invalid_argument("Need at least 2 points for interpolation");
        }

        // 检查 x 是否严格递增
        for (size_t i = 0; i < n - 1; ++i)
        {
            if (x[i] >= x[i + 1])
            {
                throw std::invalid_argument("x must be strictly increasing");
            }
        }

        n_ = n;
        x_.assign(x, x + n);
        y_.assign(y, y + n);

        // 使用三点公式估计端点导数
        double h0 = x_[1] - x_[0];
        double h1 = x_[2] - x_[1];
        dy0_ = (-(2.0 * h0 + h1) * y_[0] + (h0 + h1) * (h0 + h1) / h1 * y_[1] - h0 * h0 / h1 * y_[2]) / (h0 * (h0 + h1));

        double hn_2 = x_[n_ - 2] - x_[n_ - 3];
        double hn_1 = x_[n_ - 1] - x_[n_ - 2];
        dyn_ = (hn_1 * hn_1 / hn_2 * y_[n_ - 3] - (hn_1 + hn_2) * (hn_1 + hn_2) / hn_2 * y_[n_ - 2] + (hn_1 + 2.0 * hn_2) * y_[n_ - 1]) / (hn_1 * (hn_1 + hn_2));

        compute_second_derivatives();
    }

    // 构造函数，支持非等间距的 x（只要递增即可）- std::vector 版本
    CubicSpline(const std::vector<double> &x_vec,
                const std::vector<double> &y_vec)
    {
        if (x_vec.size() != y_vec.size())
        {
            throw std::invalid_argument("x and y must have the same length");
        }

        size_t n = x_vec.size();
        if (n < 2)
        {
            throw std::invalid_argument("Need at least 2 points for interpolation");
        }

        // 检查 x 是否严格递增
        for (size_t i = 0; i < n - 1; ++i)
        {
            if (x_vec[i] >= x_vec[i + 1])
            {
                throw std::invalid_argument("x must be strictly increasing");
            }
        }

        n_ = n;
        x_ = x_vec;
        y_ = y_vec;

        // 使用三点公式估计端点导数
        double h0 = x_[1] - x_[0];
        double h1 = x_[2] - x_[1];
        dy0_ = (-(2.0 * h0 + h1) * y_[0] + (h0 + h1) * (h0 + h1) / h1 * y_[1] - h0 * h0 / h1 * y_[2]) / (h0 * (h0 + h1));

        double hn_2 = x_[n_ - 2] - x_[n_ - 3];
        double hn_1 = x_[n_ - 1] - x_[n_ - 2];
        dyn_ = (hn_1 * hn_1 / hn_2 * y_[n_ - 3] - (hn_1 + hn_2) * (hn_1 + hn_2) / hn_2 * y_[n_ - 2] + (hn_1 + 2.0 * hn_2) * y_[n_ - 1]) / (hn_1 * (hn_1 + hn_2));

        compute_second_derivatives();
    }

    double operator()(double x) const
    {
        return evaluate(x);
    }

    // ========== 单点计算（线程安全，可在 OpenMP 中调用）==========
    double evaluate(double x) const
    {
        if (x < x_[0] || x > x_[n_ - 1])
        {
            throw std::out_of_range("x is outside interpolation range");
        }

        // 二分查找定位区间
        size_t i = find_interval(x);

        double h = x_[i + 1] - x_[i];
        double A = (x_[i + 1] - x) / h;
        double B = (x - x_[i]) / h;

        return A * y_[i] + B * y_[i + 1] +
               ((A * A * A - A) * y2_[i] + (B * B * B - B) * y2_[i + 1]) * (h * h) / 6.0;
    }

    double derivative(double x) const
    {
        if (x < x_[0] || x > x_[n_ - 1])
        {
            throw std::out_of_range("x is outside interpolation range");
        }

        // 二分查找定位区间
        size_t i = find_interval(x);

        double h = x_[i + 1] - x_[i];
        double A = (x_[i + 1] - x) / h;
        double B = (x - x_[i]) / h;

        return (y_[i + 1] - y_[i]) / h +
               ((3.0 * B * B - 1.0) * y2_[i + 1] - (3.0 * A * A - 1.0) * y2_[i]) * h / 6.0;
    }

    // ========== 批量计算 - std::vector 输入输出（C++ 内部使用，OpenMP 加速）==========
    std::vector<double> evaluate(const std::vector<double> &x_vals) const
    {
        size_t N = x_vals.size();
        std::vector<double> result(N);

#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            double x = x_vals[i];

            // 边界检查
            if (x < x_[0] || x > x_[n_ - 1])
            {
                result[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            // 在并行区域内调用单点计算（线程安全）
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

            // 边界检查
            if (x < x_[0] || x > x_[n_ - 1])
            {
                result[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            // 在并行区域内调用单点计算（线程安全）
            result[i] = derivative(x);
        }

        return result;
    }

    // ========== 批量计算 - numpy array 输入输出（暴露给 Python，OpenMP 加速）==========
    auto evaluate(const ROneArrayD &x_vals) const
    {
        size_t N = x_vals.shape(0);
        const double *x_data = x_vals.data();

        double *res_data = new double[N];
        nb::capsule res_owner(res_data, [](void *p) noexcept
                              { delete[] (double *)p; });

#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            double x = x_data[i];

            // 边界检查
            if (x < x_[0] || x > x_[n_ - 1])
            {
                res_data[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            // 在并行区域内调用单点计算（线程安全）
            res_data[i] = evaluate(x);
        }

        return nb::ndarray<nb::numpy, double>(res_data, {N}, res_owner);
    }

    auto derivative(const ROneArrayD &x_vals) const
    {
        size_t N = x_vals.shape(0);
        const double *x_data = x_vals.data();

        double *res_data = new double[N];
        nb::capsule res_owner(res_data, [](void *p) noexcept
                              { delete[] (double *)p; });

#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            double x = x_data[i];

            // 边界检查
            if (x < x_[0] || x > x_[n_ - 1])
            {
                res_data[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            // 在并行区域内调用单点计算（线程安全）
            res_data[i] = derivative(x);
        }

        return nb::ndarray<nb::numpy, double>(res_data, {N}, res_owner);
    }

private:
    // 二分查找定位 x 所在的区间 [x_[i], x_[i+1]]
    // 返回左端点索引 i，线程安全
    size_t find_interval(double x) const
    {
        // 使用 std::lower_bound 进行二分查找
        auto it = std::lower_bound(x_.begin(), x_.end(), x);

        if (it == x_.end())
        {
            return n_ - 2; // x >= x_[n_-1]，返回最后一个区间
        }

        size_t idx = std::distance(x_.begin(), it);

        if (idx == 0)
        {
            return 0; // x <= x_[0]，返回第一个区间
        }

        // x 在 [x_[idx-1], x_[idx]] 之间
        return idx - 1;
    }

    void compute_second_derivatives()
    {
        y2_.resize(n_);

        if (n_ == 2)
        {
            y2_[0] = 0.0;
            y2_[1] = 0.0;
            return;
        }

        std::vector<double> a(n_);
        std::vector<double> b(n_);
        std::vector<double> c(n_);
        std::vector<double> d(n_);

        // 夹紧边界条件
        // 第一行
        double h0 = x_[1] - x_[0];
        b[0] = 2.0 * h0;
        c[0] = h0;
        d[0] = 6.0 * ((y_[1] - y_[0]) / h0 - dy0_);

        // 中间行（非等间距）
        for (size_t i = 1; i < n_ - 1; ++i)
        {
            double h_i = x_[i] - x_[i - 1];
            double h_ip1 = x_[i + 1] - x_[i];

            a[i] = h_i;
            b[i] = 2.0 * (h_i + h_ip1);
            c[i] = h_ip1;
            d[i] = 6.0 * ((y_[i + 1] - y_[i]) / h_ip1 - (y_[i] - y_[i - 1]) / h_i);
        }

        // 最后一行
        double hn_1 = x_[n_ - 1] - x_[n_ - 2];
        a[n_ - 1] = hn_1;
        b[n_ - 1] = 2.0 * hn_1;
        d[n_ - 1] = 6.0 * (dyn_ - (y_[n_ - 1] - y_[n_ - 2]) / hn_1);

        // Thomas 算法求解三对角系统
        c[0] /= b[0];
        d[0] /= b[0];

        for (size_t i = 1; i < n_; ++i)
        {
            double m = b[i] - a[i] * c[i - 1];
            if (i < n_ - 1)
            {
                c[i] /= m;
            }
            d[i] = (d[i] - a[i] * d[i - 1]) / m;
        }

        y2_[n_ - 1] = d[n_ - 1];
        for (int i = n_ - 2; i >= 0; --i)
        {
            y2_[i] = d[i] - c[i] * y2_[i + 1];
        }
    }

    size_t n_;
    std::vector<double> x_;  // 存储 x 坐标（可以非等间距）
    std::vector<double> y_;  // 存储 y 值
    std::vector<double> y2_; // 二阶导数
    double dy0_ = 0.0;       // 左端点导数
    double dyn_ = 0.0;       // 右端点导数
};