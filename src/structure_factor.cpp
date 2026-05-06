// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <complex>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>
#include "type.h"
#include "box.h"

namespace nb = nanobind;

// =============================================================================
// 常量定义
// =============================================================================
namespace constants
{
    constexpr double TWO_PI = 6.28318530717958647692;
    constexpr double PI = 3.14159265358979323846;
    constexpr double TOLERANCE = 1e-12; // 通用容差
}

// =============================================================================
// 工具函数
// =============================================================================

/// sinc(x) = sin(x)/x, 当 x=0 时为 1
inline double sinc(double x)
{
    if (std::abs(x) < constants::TOLERANCE)
    {
        return 1.0;
    }
    return std::sin(x) / x;
}

// =============================================================================
// Vec3 向量类
// =============================================================================
struct Vec3
{
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3 &other) const
    {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    Vec3 operator*(double scalar) const
    {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    double dot(const Vec3 &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    double norm() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    double norm_sq() const
    {
        return x * x + y * y + z * z;
    }
};

// =============================================================================
// StructureFactorDirect 类
// =============================================================================
class StructureFactorDirect
{
private:
    /// 获取倒易空间基矢
    Vec3 getReciprocalVector(const Box &box, int index) const
    {
        const double *a = box.data;
        const double *b = box.data + 3;
        const double *c = box.data + 6;
        const double volume = box.get_volume();

        if (std::abs(volume) < constants::TOLERANCE)
        {
            throw std::runtime_error("Box volume is too small or zero");
        }

        Vec3 recip;
        switch (index)
        {
        case 0:
            recip.x = (b[1] * c[2] - b[2] * c[1]) / volume;
            recip.y = (b[2] * c[0] - b[0] * c[2]) / volume;
            recip.z = (b[0] * c[1] - b[1] * c[0]) / volume;
            break;
        case 1:
            recip.x = (c[1] * a[2] - c[2] * a[1]) / volume;
            recip.y = (c[2] * a[0] - c[0] * a[2]) / volume;
            recip.z = (c[0] * a[1] - c[1] * a[0]) / volume;
            break;
        case 2:
            recip.x = (a[1] * b[2] - a[2] * b[1]) / volume;
            recip.y = (a[2] * b[0] - a[0] * b[2]) / volume;
            recip.z = (a[0] * b[1] - a[1] * b[0]) / volume;
            break;
        default:
            throw std::runtime_error("Reciprocal vector index must be 0, 1, or 2");
        }

        return recip * constants::TWO_PI;
    }

    /// 生成所有 k 点（完全确定性，无随机采样）
    std::vector<Vec3> generateKPoints(
        const Box &box,
        double k_max,
        double k_min) const
    {
        const Vec3 bx = getReciprocalVector(box, 0);
        const Vec3 by = getReciprocalVector(box, 1);
        const Vec3 bz = getReciprocalVector(box, 2);

        const double q_max = k_max / constants::TWO_PI;
        const double q_min = k_min / constants::TWO_PI;
        const double q_max_sq = q_max * q_max;
        const double q_min_sq = q_min * q_min;

        const int N_kx = static_cast<int>(std::ceil(q_max / (bx.norm() / constants::TWO_PI)));
        const int N_ky = static_cast<int>(std::ceil(q_max / (by.norm() / constants::TWO_PI)));
        const int N_kz = static_cast<int>(std::ceil(q_max / (bz.norm() / constants::TWO_PI)));

        // 每个线程维护自己的 k 点列表
        std::vector<std::vector<Vec3>> thread_k_points(omp_get_max_threads());

#pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic)
            for (int i = 0; i < N_kx; ++i)
            {
                const Vec3 k_vec_x = bx * static_cast<double>(i);

                for (int ky = 0; ky < N_ky; ++ky)
                {
                    const Vec3 k_vec_xy = k_vec_x + by * static_cast<double>(ky);

                    // 计算 kz 的有效范围
                    const double coef_a = bz.dot(bz);
                    const double coef_b = -2.0 * k_vec_xy.dot(bz);
                    const double coef_c_min = k_vec_xy.dot(k_vec_xy) - k_min * k_min;
                    const double coef_c_max = k_vec_xy.dot(k_vec_xy) - k_max * k_max;

                    const double b_over_2a = coef_b / (2.0 * coef_a);
                    const double discriminant_min = b_over_2a * b_over_2a - coef_c_min / coef_a;
                    const double discriminant_max = b_over_2a * b_over_2a - coef_c_max / coef_a;

                    if (discriminant_max < 0)
                    {
                        continue;
                    }

                    double kz_min_val;
                    if (discriminant_min < 0)
                    {
                        kz_min_val = 0.0;
                    }
                    else
                    {
                        kz_min_val = -b_over_2a + std::sqrt(discriminant_min);
                    }
                    const double kz_max_val = -b_over_2a + std::sqrt(discriminant_max);

                    int kz_min = static_cast<int>(std::floor(kz_min_val));
                    int kz_max = static_cast<int>(std::ceil(kz_max_val));

                    kz_min = std::max(kz_min, 0);
                    kz_max = std::min(kz_max, N_kz - 1);

                    for (int kz = kz_min; kz <= kz_max; ++kz)
                    {
                        const Vec3 k_vec = k_vec_xy + bz * static_cast<double>(kz);
                        const double k_sq = k_vec.norm_sq();
                        const double q_distance_sq = k_sq / (constants::TWO_PI * constants::TWO_PI);

                        if (q_distance_sq <= q_max_sq && q_distance_sq >= q_min_sq)
                        {
                            thread_k_points[thread_id].push_back(k_vec);
                        }
                    }
                }
            }
        }

        // 合并所有线程的结果
        std::vector<Vec3> k_points;
        size_t total_size = 0;
        for (const auto &thread_vec : thread_k_points)
        {
            total_size += thread_vec.size();
        }
        k_points.reserve(total_size);

        for (const auto &thread_vec : thread_k_points)
        {
            k_points.insert(k_points.end(), thread_vec.begin(), thread_vec.end());
        }

        return k_points;
    }

    /// 计算结构因子 F(k)
    std::vector<std::complex<double>> computeFK(
        const double *x,
        const double *y,
        const double *z,
        size_t n_points,
        unsigned int n_total,
        const std::vector<Vec3> &k_points) const
    {
        std::vector<std::complex<double>> F_k(k_points.size());
        const double normalization = 1.0 / std::sqrt(static_cast<double>(n_total));

#pragma omp parallel for schedule(dynamic)
        for (size_t k_idx = 0; k_idx < k_points.size(); ++k_idx)
        {
            std::complex<double> F_ki(0.0, 0.0);

            for (size_t i = 0; i < n_points; ++i)
            {
                const double alpha = k_points[k_idx].x * x[i] +
                                     k_points[k_idx].y * y[i] +
                                     k_points[k_idx].z * z[i];
                F_ki += std::exp(std::complex<double>(0.0, alpha));
            }

            F_k[k_idx] = F_ki * normalization;
        }

        return F_k;
    }

    /// 计算 S(k)
    std::vector<double> computeSK(
        const std::vector<std::complex<double>> &F_k_points,
        const std::vector<std::complex<double>> &F_k_query) const
    {
        if (F_k_points.size() != F_k_query.size())
        {
            throw std::runtime_error("F_k_points and F_k_query must have the same size");
        }

        std::vector<double> S_k(F_k_points.size());

#pragma omp parallel for
        for (size_t k_idx = 0; k_idx < F_k_points.size(); ++k_idx)
        {
            S_k[k_idx] = std::real(std::conj(F_k_points[k_idx]) * F_k_query[k_idx]);
        }

        return S_k;
    }

    /// 获取 bin 索引
    int getBin(double k_value, double k_min, double k_max, unsigned int bins) const
    {
        if (k_value < k_min || k_value >= k_max)
        {
            return -1;
        }
        const int bin = static_cast<int>((k_value - k_min) / (k_max - k_min) * bins);
        return std::min(bin, static_cast<int>(bins) - 1);
    }

public:
    /// 主计算函数
    void compute(
        ROneArrayD x_py, ROneArrayD y_py, ROneArrayD z_py,
        RTwoArrayD box_py, ROneArrayD origin, ROneArrayI boundary,
        OneArrayD structure_factor_py,
        unsigned int bins, double k_max, double k_min,
        nb::object query_x_py = nb::none(),
        nb::object query_y_py = nb::none(),
        nb::object query_z_py = nb::none(),
        unsigned int N_total = 0)
    {
        // 参数验证
        if (bins == 0)
        {
            throw std::invalid_argument("bins must be non-zero");
        }
        if (k_max <= 0)
        {
            throw std::invalid_argument("k_max must be positive");
        }
        if (k_min < 0)
        {
            throw std::invalid_argument("k_min must be non-negative");
        }
        if (k_max <= k_min)
        {
            throw std::invalid_argument("k_max must be greater than k_min");
        }

        const size_t n_points = x_py.shape(0);
        if (y_py.shape(0) != n_points || z_py.shape(0) != n_points)
        {
            throw std::invalid_argument("x, y, z must have the same length");
        }

        const double *x = x_py.data();
        const double *y = y_py.data();
        const double *z = z_py.data();

        const Box box = get_box(box_py, origin, boundary);

        // 生成所有 k 点（无随机采样）
        const auto k_points = generateKPoints(box, k_max, k_min);

        if (k_points.empty())
        {
            throw std::runtime_error("No k-points generated. Check k_min and k_max values.");
        }

        // 计算 F(k)
        std::vector<std::complex<double>> F_k_points;
        std::vector<std::complex<double>> F_k_query;

        unsigned int n_total_calc = N_total;

        const bool has_query = !query_x_py.is_none() &&
                               !query_y_py.is_none() &&
                               !query_z_py.is_none();

        if (!has_query)
        {
            // 没有查询点，使用自相关
            if (n_total_calc == 0)
            {
                n_total_calc = static_cast<unsigned int>(n_points);
            }
            F_k_points = computeFK(x, y, z, n_points, n_total_calc, k_points);
            F_k_query = F_k_points;
        }
        else
        {
            // 有查询点，使用互相关
            if (query_x_py.is_none() || query_y_py.is_none() || query_z_py.is_none())
            {
                throw std::invalid_argument(
                    "All query_x, query_y, query_z must be provided or none.");
            }
            if (n_total_calc == 0)
            {
                throw std::invalid_argument(
                    "N_total is required when query points are provided.");
            }

            const ROneArrayD query_x = nb::cast<ROneArrayD>(query_x_py);
            const ROneArrayD query_y = nb::cast<ROneArrayD>(query_y_py);
            const ROneArrayD query_z = nb::cast<ROneArrayD>(query_z_py);

            const size_t n_query_points = query_x.shape(0);
            if (query_y.shape(0) != n_query_points || query_z.shape(0) != n_query_points)
            {
                throw std::invalid_argument(
                    "query_x, query_y, query_z must have the same length");
            }

            const double *qx = query_x.data();
            const double *qy = query_y.data();
            const double *qz = query_z.data();

            F_k_points = computeFK(x, y, z, n_points, n_total_calc, k_points);
            F_k_query = computeFK(qx, qy, qz, n_query_points, n_total_calc, k_points);
        }

        // 计算 S(k)
        const auto S_k_all = computeSK(F_k_points, F_k_query);

        // 分 bin 并平均
        double *structure_factor = structure_factor_py.data();
        std::vector<unsigned int> bin_counts(bins, 0);

        // 初始化输出数组
        for (unsigned int i = 0; i < bins; ++i)
        {
            structure_factor[i] = 0.0;
        }

#pragma omp parallel
        {
            std::vector<double> local_sf(bins, 0.0);
            std::vector<unsigned int> local_counts(bins, 0);

#pragma omp for nowait
            for (size_t k_idx = 0; k_idx < k_points.size(); ++k_idx)
            {
                const double k_mag = k_points[k_idx].norm();
                const int bin = getBin(k_mag, k_min, k_max, bins);

                if (bin >= 0 && bin < static_cast<int>(bins))
                {
                    local_sf[bin] += S_k_all[k_idx];
                    local_counts[bin]++;
                }
            }

#pragma omp critical
            {
                for (unsigned int i = 0; i < bins; ++i)
                {
                    structure_factor[i] += local_sf[i];
                    bin_counts[i] += local_counts[i];
                }
            }
        }

        // 计算平均值
        for (unsigned int i = 0; i < bins; ++i)
        {
            if (bin_counts[i] > 0)
            {
                structure_factor[i] /= static_cast<double>(bin_counts[i]);
            }
            else
            {
                structure_factor[i] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
};

// =============================================================================
// 直接法（partial-friendly）：一次性计算所有 species 的 F_alpha(k)
//
// 输入位置 + 0..Ntype-1 的 type_list；返回 (Ntype, Ntype, nbins) 形状的
// Ashcroft-Langreth partial. 复杂度 O(Ntype * N * n_k) — 比对每个 (alpha,
// beta) 重新组装位置子集再各自累加 F 节省了 Ntype 倍。
// =============================================================================
class StructureFactorDirectPartial
{
public:
    void compute(
        ROneArrayD x_py, ROneArrayD y_py, ROneArrayD z_py,
        ROneArrayI type_list_py, int Ntype,
        RTwoArrayD box_py, ROneArrayD origin, ROneArrayI boundary,
        ThreeArrayD partial_out_py,
        unsigned int bins, double k_max, double k_min)
    {
        if (bins == 0)
            throw std::invalid_argument("bins must be non-zero");
        if (k_max <= k_min || k_min < 0)
            throw std::invalid_argument("require 0 <= k_min < k_max");

        const size_t n_points = x_py.shape(0);
        const double *x = x_py.data();
        const double *y = y_py.data();
        const double *z = z_py.data();
        const int *type_list = type_list_py.data();
        const Box box = get_box(box_py, origin, boundary);

        StructureFactorDirect helper;  // reuse the kpoint generator + binner
        const auto k_points = helper_kpoints(box, k_max, k_min);
        if (k_points.empty())
            throw std::runtime_error("No k-points generated; check k_min/k_max.");

        // Compute F_alpha(k) for each species in one parallel pass.
        // Layout: F[alpha * nk + k_idx]
        const size_t nk = k_points.size();
        std::vector<std::complex<double>> F(static_cast<size_t>(Ntype) * nk,
                                            std::complex<double>(0.0, 0.0));
        const double inv_sqrt_N = 1.0 / std::sqrt(static_cast<double>(n_points));

#pragma omp parallel for schedule(dynamic)
        for (size_t k_idx = 0; k_idx < nk; ++k_idx)
        {
            const Vec3 &kv = k_points[k_idx];
            // Per-thread accumulator across species, indexed by alpha.
            std::vector<std::complex<double>> acc(Ntype,
                                                  std::complex<double>(0.0, 0.0));
            for (size_t i = 0; i < n_points; ++i)
            {
                const double phase = kv.x * x[i] + kv.y * y[i] + kv.z * z[i];
                acc[type_list[i]] += std::exp(std::complex<double>(0.0, phase));
            }
            for (int a = 0; a < Ntype; ++a)
                F[static_cast<size_t>(a) * nk + k_idx] = acc[a] * inv_sqrt_N;
        }

        // Bin the cross-correlations into the (Ntype, Ntype, bins) array.
        auto out = partial_out_py.view();
        for (int a = 0; a < Ntype; ++a)
            for (int b = 0; b < Ntype; ++b)
                for (unsigned int k = 0; k < bins; ++k)
                    out(a, b, k) = 0.0;

        // Per-bin counts shared across partials (the k-point distribution
        // is the same for all (a, b)).
        std::vector<unsigned int> bin_counts(bins, 0);
        for (size_t k_idx = 0; k_idx < nk; ++k_idx)
        {
            const double k_mag = k_points[k_idx].norm();
            const int bin = bin_of(k_mag, k_min, k_max, bins);
            if (bin < 0)
                continue;
            bin_counts[bin]++;
        }

        // Flatten the upper-triangular pair loop so OpenMP can schedule
        // each (a, b) independently without colliding with the
        // non-rectangular ``b >= a`` constraint.
        std::vector<std::pair<int, int>> pairs;
        for (int a = 0; a < Ntype; ++a)
            for (int b = a; b < Ntype; ++b)
                pairs.emplace_back(a, b);

#pragma omp parallel for schedule(dynamic)
        for (size_t pi = 0; pi < pairs.size(); ++pi)
        {
            const int a = pairs[pi].first;
            const int b = pairs[pi].second;
            {
                std::vector<double> local(bins, 0.0);
                for (size_t k_idx = 0; k_idx < nk; ++k_idx)
                {
                    const double k_mag = k_points[k_idx].norm();
                    const int bin = bin_of(k_mag, k_min, k_max, bins);
                    if (bin < 0)
                        continue;
                    const auto &fa = F[static_cast<size_t>(a) * nk + k_idx];
                    const auto &fb = F[static_cast<size_t>(b) * nk + k_idx];
                    local[bin] += std::real(std::conj(fa) * fb);
                }
                for (unsigned int k = 0; k < bins; ++k)
                {
                    if (bin_counts[k] > 0)
                    {
                        const double val = local[k] / static_cast<double>(bin_counts[k]);
                        out(a, b, k) = val;
                        if (a != b)
                            out(b, a, k) = val;  // symmetric
                    }
                    else
                    {
                        out(a, b, k) = std::numeric_limits<double>::quiet_NaN();
                        if (a != b)
                            out(b, a, k) = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
        }
    }

private:
    // Re-implement the small kernel pieces to avoid friend access into
    // StructureFactorDirect. The k-point generator below mirrors the one
    // used for the total mode, so partial and total see identical k-grids.
    Vec3 reciprocal(const Box &box, int axis) const
    {
        const double *a = box.data;
        const double *bv = box.data + 3;
        const double *cv = box.data + 6;
        const double V = box.get_volume();
        Vec3 r;
        if (axis == 0)
        {
            r.x = (bv[1] * cv[2] - bv[2] * cv[1]) / V;
            r.y = (bv[2] * cv[0] - bv[0] * cv[2]) / V;
            r.z = (bv[0] * cv[1] - bv[1] * cv[0]) / V;
        }
        else if (axis == 1)
        {
            r.x = (cv[1] * a[2] - cv[2] * a[1]) / V;
            r.y = (cv[2] * a[0] - cv[0] * a[2]) / V;
            r.z = (cv[0] * a[1] - cv[1] * a[0]) / V;
        }
        else
        {
            r.x = (a[1] * bv[2] - a[2] * bv[1]) / V;
            r.y = (a[2] * bv[0] - a[0] * bv[2]) / V;
            r.z = (a[0] * bv[1] - a[1] * bv[0]) / V;
        }
        return r * constants::TWO_PI;
    }

    std::vector<Vec3> helper_kpoints(const Box &box, double k_max, double k_min) const
    {
        const Vec3 bx = reciprocal(box, 0);
        const Vec3 by = reciprocal(box, 1);
        const Vec3 bz = reciprocal(box, 2);
        const double q_max = k_max / constants::TWO_PI;
        const int N_kx = static_cast<int>(std::ceil(q_max / (bx.norm() / constants::TWO_PI)));
        const int N_ky = static_cast<int>(std::ceil(q_max / (by.norm() / constants::TWO_PI)));
        const int N_kz = static_cast<int>(std::ceil(q_max / (bz.norm() / constants::TWO_PI)));
        std::vector<std::vector<Vec3>> tk(omp_get_max_threads());
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
            for (int i = 0; i < N_kx; ++i)
            {
                const Vec3 kx = bx * static_cast<double>(i);
                for (int j = 0; j < N_ky; ++j)
                {
                    const Vec3 kxy = kx + by * static_cast<double>(j);
                    for (int k = 0; k < N_kz; ++k)
                    {
                        const Vec3 kvec = kxy + bz * static_cast<double>(k);
                        const double mag = kvec.norm();
                        if (mag > k_min && mag <= k_max)
                            tk[tid].push_back(kvec);
                    }
                }
            }
        }
        size_t total = 0;
        for (const auto &v : tk)
            total += v.size();
        std::vector<Vec3> out;
        out.reserve(total);
        for (const auto &v : tk)
            out.insert(out.end(), v.begin(), v.end());
        return out;
    }

    int bin_of(double k_mag, double k_min, double k_max, unsigned int bins) const
    {
        if (k_mag < k_min || k_mag >= k_max)
            return -1;
        const int b = static_cast<int>((k_mag - k_min) / (k_max - k_min) * bins);
        return std::min(b, static_cast<int>(bins) - 1);
    }
};

// =============================================================================
// Python 绑定
// =============================================================================
NB_MODULE(_sfc, m)
{
    m.doc() = "Structure factor calculation module (deterministic, no random sampling)";

    m.def(
        "compute_sfc_direct",
        [](ROneArrayD x, ROneArrayD y, ROneArrayD z,
           RTwoArrayD box, ROneArrayD origin, ROneArrayI boundary,
           OneArrayD structure_factor_py,
           unsigned int bins, double k_max, double k_min,
           nb::object query_x, nb::object query_y, nb::object query_z,
           unsigned int N_total)
        {
            StructureFactorDirect calc;
            calc.compute(
                x, y, z, box, origin, boundary, structure_factor_py,
                bins, k_max, k_min,
                query_x, query_y, query_z, N_total);
        },
        nb::arg("x"), nb::arg("y"), nb::arg("z"),
        nb::arg("box"), nb::arg("origin"), nb::arg("boundary"),
        nb::arg("structure_factor_py"),
        nb::arg("bins"), nb::arg("k_max"), nb::arg("k_min"),
        nb::arg("query_x") = nb::none(),
        nb::arg("query_y") = nb::none(),
        nb::arg("query_z") = nb::none(),
        nb::arg("N_total") = 0,
        "Compute the total / cross-pair direct structure factor on the "
        "deterministic reciprocal-lattice grid.");

    m.def(
        "compute_sfc_direct_partial",
        [](ROneArrayD x, ROneArrayD y, ROneArrayD z,
           ROneArrayI type_list, int Ntype,
           RTwoArrayD box, ROneArrayD origin, ROneArrayI boundary,
           ThreeArrayD partial_out,
           unsigned int bins, double k_max, double k_min)
        {
            StructureFactorDirectPartial calc;
            calc.compute(
                x, y, z, type_list, Ntype, box, origin, boundary,
                partial_out, bins, k_max, k_min);
        },
        nb::arg("x"), nb::arg("y"), nb::arg("z"),
        nb::arg("type_list"), nb::arg("Ntype"),
        nb::arg("box"), nb::arg("origin"), nb::arg("boundary"),
        nb::arg("partial_out"),
        nb::arg("bins"), nb::arg("k_max"), nb::arg("k_min"),
        "Compute every Ashcroft-Langreth partial S_alpha_beta(k) at once. "
        "F_alpha(k) is computed once per species; the cost scales as "
        "O(Ntype * N * n_k) instead of O(Ntype^2 * N * n_k) for the "
        "pair-by-pair workflow.");
}