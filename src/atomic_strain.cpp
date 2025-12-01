#include "type.h"
#include "box.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <omp.h>

namespace nb = nanobind;

// 简单的3x3矩阵结构
struct Matrix3x3 {
    double data[9]; // 行主序: [0,1,2; 3,4,5; 6,7,8]
    
    Matrix3x3() {
        for (int i = 0; i < 9; ++i) data[i] = 0.0;
    }
    
    // 设置为单位矩阵
    void set_identity() {
        data[0] = data[4] = data[8] = 1.0;
        data[1] = data[2] = data[3] = data[5] = data[6] = data[7] = 0.0;
    }
    
    // 矩阵转置
    Matrix3x3 transpose() const {
        Matrix3x3 result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result.data[j * 3 + i] = data[i * 3 + j];
            }
        }
        return result;
    }
    
    // 矩阵乘法: this * other
    Matrix3x3 operator*(const Matrix3x3& other) const {
        Matrix3x3 result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += data[i * 3 + k] * other.data[k * 3 + j];
                }
                result.data[i * 3 + j] = sum;
            }
        }
        return result;
    }
    
    // 矩阵求逆 (3x3 直接公式)
    Matrix3x3 inverse() const {
        Matrix3x3 result;
        
        // 计算行列式
        double det = data[0] * (data[4] * data[8] - data[5] * data[7]) -
                     data[1] * (data[3] * data[8] - data[5] * data[6]) +
                     data[2] * (data[3] * data[7] - data[4] * data[6]);
        
        if (std::abs(det) < 1e-12) {
            // 矩阵不可逆，返回单位矩阵或抛出异常
            result.set_identity();
            return result;
        }
        
        double inv_det = 1.0 / det;
        
        // 伴随矩阵的转置
        result.data[0] = (data[4] * data[8] - data[5] * data[7]) * inv_det;
        result.data[1] = (data[2] * data[7] - data[1] * data[8]) * inv_det;
        result.data[2] = (data[1] * data[5] - data[2] * data[4]) * inv_det;
        
        result.data[3] = (data[5] * data[6] - data[3] * data[8]) * inv_det;
        result.data[4] = (data[0] * data[8] - data[2] * data[6]) * inv_det;
        result.data[5] = (data[2] * data[3] - data[0] * data[5]) * inv_det;
        
        result.data[6] = (data[3] * data[7] - data[4] * data[6]) * inv_det;
        result.data[7] = (data[1] * data[6] - data[0] * data[7]) * inv_det;
        result.data[8] = (data[0] * data[4] - data[1] * data[3]) * inv_det;
        
        return result;
    }
    
    // 矩阵减法
    Matrix3x3 operator-(const Matrix3x3& other) const {
        Matrix3x3 result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
    
    // 矩阵除以标量
    Matrix3x3 operator/(double scalar) const {
        Matrix3x3 result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }
    
    // 访问元素
    double& operator()(int i, int j) { return data[i * 3 + j]; }
    const double& operator()(int i, int j) const { return data[i * 3 + j]; }
};


// 计算原子应变的主函数
void cal_atomic_strain(
    const RTwoArrayI verlet_list_py,
    const ROneArrayI neighbor_number_py,
    const RTwoArrayD ref_box_py,
    const RTwoArrayD cur_box_py,
    const ROneArrayD ref_origin,
    const ROneArrayD cur_origin,
    const ROneArrayI boundary,
    const ROneArrayD ref_x_py,
    const ROneArrayD ref_y_py,
    const ROneArrayD ref_z_py,
    const ROneArrayD cur_x_py,
    const ROneArrayD cur_y_py,
    const ROneArrayD cur_z_py,
    OneArrayD shear_strain_py,
    OneArrayD volumetric_strain_py
)
{
    size_t N = verlet_list_py.shape(0);
    size_t max_neighbors = verlet_list_py.shape(1);
    
    const int* neighbor_number = neighbor_number_py.data();
    const int* verlet_list = verlet_list_py.data();

    Box ref_box = get_box(ref_box_py, ref_origin, boundary);
    Box cur_box = get_box(cur_box_py, cur_origin, boundary);

    const double* ref_x = ref_x_py.data();
    const double* ref_y = ref_y_py.data();
    const double* ref_z = ref_z_py.data();
    const double* cur_x = cur_x_py.data();
    const double* cur_y = cur_y_py.data();
    const double* cur_z = cur_z_py.data();
    double* shear_strain = shear_strain_py.data();
    double* volumetric_strain = volumetric_strain_py.data();
    
    // 单位矩阵
    Matrix3x3 identity;
    identity.set_identity();
    
    // 并行处理每个原子
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        Matrix3x3 V, W;
        
        const int num_neighbors = neighbor_number[i];
        const double refxi = ref_x[i];
        const double refyi = ref_y[i];
        const double refzi = ref_z[i];

        const double curxi = cur_x[i];
        const double curyi = cur_y[i];
        const double curzi = cur_z[i];
        
        for (int jj = 0; jj < num_neighbors; ++jj) {
            const int j = verlet_list[i * max_neighbors + jj]; 
            
            // 计算参考构型的距离矢量
            double delta_ref_x = ref_x[j] - refxi;
            double delta_ref_y = ref_y[j] - refyi;
            double delta_ref_z = ref_z[j] - refzi;
            ref_box.pbc(delta_ref_x, delta_ref_y, delta_ref_z);
            
            // 计算当前构型的距离矢量
            double delta_cur_x = cur_x[j] - curxi;
            double delta_cur_y = cur_y[j] - curyi;
            double delta_cur_z = cur_z[j] - curzi;
            cur_box.pbc(delta_cur_x, delta_cur_y, delta_cur_z);

            // 累加到V和W矩阵
            // V[m,n] += delta_ref[n] * delta_ref[m]
            // W[m,n] += delta_ref[n] * delta_cur[m]
            double ref_vec[3] = {delta_ref_x, delta_ref_y, delta_ref_z};
            double cur_vec[3] = {delta_cur_x, delta_cur_y, delta_cur_z};
            
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    V(m, n) += ref_vec[n] * ref_vec[m];
                    W(m, n) += ref_vec[n] * cur_vec[m];
                }
            }
        }
        
        // 计算变形梯度张量 F = (W * V^-1)^T
        Matrix3x3 V_inv = V.inverse();
        Matrix3x3 F = (W * V_inv).transpose();
        
        // 计算应变张量 s = (F^T * F - I) / 2
        Matrix3x3 FtF = F.transpose() * F;
        Matrix3x3 s = (FtF - identity) / 2.0;
        
        // 计算剪切应变
        double xydiff = s(0, 0) - s(1, 1);
        double yzdiff = s(1, 1) - s(2, 2);
        double xzdiff = s(0, 0) - s(2, 2);
        
        double shear_strain_val = std::sqrt(
            s(0, 1) * s(0, 1) +
            s(0, 2) * s(0, 2) +
            s(1, 2) * s(1, 2) +
            (xydiff * xydiff + xzdiff * xzdiff + yzdiff * yzdiff) / 6.0
        );
        
        shear_strain[i] = shear_strain_val;
        volumetric_strain[i] = (s(0, 0) + s(1, 1) + s(2, 2)) / 3.0;
    }
}

NB_MODULE(_strain, m) {
    m.def("cal_atomic_strain", &cal_atomic_strain);
}

