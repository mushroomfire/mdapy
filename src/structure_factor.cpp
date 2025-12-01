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
// Debye 方法计算结构因子
// =============================================================================
void compute_sfc_debye(
    const ROneArrayD x_py, const ROneArrayD y_py, const ROneArrayD z_py,
    const RTwoArrayD box_py, const ROneArrayD origin, const ROneArrayI boundary,
    const ROneArrayD k_values_py, OneArrayD structure_factor_py,
    nb::object query_x_py = nb::none(),
    nb::object query_y_py = nb::none(),
    nb::object query_z_py = nb::none(),
    int N_total = 0)
{
    const int n_points = x_py.shape(0);
    const int n_total = (N_total > 0) ? N_total : n_points;

    const double *x = x_py.data();
    const double *y = y_py.data();
    const double *z = z_py.data();
    const double *k_values = k_values_py.data();
    double *structure_factor = structure_factor_py.data();

    const int num_k_values = k_values_py.shape(0);

    const Box box = get_box(box_py, origin, boundary);

    // 确定查询点
    const double *qx = nullptr, *qy = nullptr, *qz = nullptr;
    int n_query = 0;

    const bool has_query = !query_x_py.is_none() &&
                           !query_y_py.is_none() &&
                           !query_z_py.is_none();

    if (!has_query)
    {
        qx = x;
        qy = y;
        qz = z;
        n_query = n_points;
    }
    else
    {
        ROneArrayD qx_arr = nb::cast<ROneArrayD>(query_x_py);
        ROneArrayD qy_arr = nb::cast<ROneArrayD>(query_y_py);
        ROneArrayD qz_arr = nb::cast<ROneArrayD>(query_z_py);

        n_query = qx_arr.shape(0);
        qx = qx_arr.data();
        qy = qy_arr.data();
        qz = qz_arr.data();
    }

    // 计算所有距离
    const size_t total_pairs = static_cast<size_t>(n_points) * n_query;
    std::vector<double> distances(total_pairs);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n_points; ++i)
    {
        for (int j = 0; j < n_query; ++j)
        {
            const double xi = x[i];
            const double yi = y[i];
            const double zi = z[i];

            double xij = qx[j] - xi;
            double yij = qy[j] - yi;
            double zij = qz[j] - zi;

            box.pbc(xij, yij, zij);

            const double dis = std::sqrt(xij * xij + yij * yij + zij * zij);
            distances[i * n_query + j] = dis;
        }
    }

    // 初始化输出
    for (int b = 0; b < num_k_values; ++b)
    {
        structure_factor[b] = 0.0;
    }

    // 计算结构因子
#pragma omp parallel
    {
        std::vector<double> local_sf(num_k_values, 0.0);

#pragma omp for nowait
        for (size_t idx = 0; idx < distances.size(); ++idx)
        {
            const double r = distances[idx];
            for (int b = 0; b < num_k_values; ++b)
            {
                const double kr = k_values[b] * r;
                local_sf[b] += sinc(kr);
            }
        }

#pragma omp critical
        {
            for (int b = 0; b < num_k_values; ++b)
            {
                structure_factor[b] += local_sf[b];
            }
        }
    }

    // 归一化
    const double norm = 1.0 / n_total;
    for (int b = 0; b < num_k_values; ++b)
    {
        structure_factor[b] *= norm;
    }
}

// =============================================================================
// Python 绑定
// =============================================================================
NB_MODULE(_sfc, m)
{
    m.doc() = "Structure factor calculation module (deterministic, no random sampling)";

    m.def("compute_sfc_direct", [](ROneArrayD x, ROneArrayD y, ROneArrayD z, RTwoArrayD box, ROneArrayD origin, ROneArrayI boundary, OneArrayD structure_factor_py, unsigned int bins, double k_max, double k_min, nb::object query_x, nb::object query_y, nb::object query_z, unsigned int N_total)
          {
            StructureFactorDirect calc;
            calc.compute(x, y, z, box, origin, boundary, structure_factor_py,
                       bins, k_max, k_min,
                       query_x, query_y, query_z, N_total); }, nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("box"), nb::arg("origin"), nb::arg("boundary"), nb::arg("structure_factor_py"), nb::arg("bins"), nb::arg("k_max"), nb::arg("k_min"), nb::arg("query_x") = nb::none(), nb::arg("query_y") = nb::none(), nb::arg("query_z") = nb::none(), nb::arg("N_total") = 0, "Compute structure factor using direct method (uses all k-points, no sampling)");

    m.def("compute_sfc_debye",
          &compute_sfc_debye,
          nb::arg("x_py"), nb::arg("y_py"), nb::arg("z_py"),
          nb::arg("box_py"), nb::arg("origin"), nb::arg("boundary"),
          nb::arg("k_values_py"), nb::arg("structure_factor_py"),
          nb::arg("query_x_py") = nb::none(), nb::arg("query_y_py") = nb::none(),
          nb::arg("query_z_py") = nb::none(), nb::arg("N_total") = 0,
          "Compute structure factor using Debye method");
}