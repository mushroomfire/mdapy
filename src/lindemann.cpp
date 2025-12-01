#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <vector>

namespace nb = nanobind;


/**
 * 计算全局Lindemann指数
 * 
 * @param pos_list_py: (Nframes, Natoms, 3) 位置数组
 * @param pos_mean_py: (Natoms, Natoms) 距离均值数组
 * @param pos_variance_py: (Natoms, Natoms) 距离方差数组
 * @return 全局Lindemann指数
 */
double compute_global(
    const RThreeArrayD pos_list_py,
    TwoArrayD pos_mean_py,
    TwoArrayD pos_variance_py
) {
    auto pos_list = pos_list_py.view();
    auto pos_mean = pos_mean_py.view();
    auto pos_variance = pos_variance_py.view();
    
    size_t Nframes = pos_list.shape(0);
    size_t Natoms = pos_list.shape(1);
    
    // 计算所有原子对之间的距离均值和方差
    #pragma omp parallel for 
    for (size_t i = 0; i < Natoms; ++i) {
        for (size_t j = i + 1; j < Natoms; ++j) {
            double sum_rij = 0.0;
            double sum_rij_sq = 0.0;
            
            for (size_t frame = 0; frame < Nframes; ++frame) {
                // 计算距离 rij
                double dx = pos_list(frame, i, 0) - pos_list(frame, j, 0);
                double dy = pos_list(frame, i, 1) - pos_list(frame, j, 1);
                double dz = pos_list(frame, i, 2) - pos_list(frame, j, 2);
                double rijdis = std::sqrt(dx * dx + dy * dy + dz * dz);
                
                sum_rij += rijdis;
                sum_rij_sq += rijdis * rijdis;
            }
            
            pos_mean(i, j) = sum_rij;
            pos_variance(i, j) = sum_rij_sq;
        }
    }
    
    // 计算Lindemann指数
    double lin_index = 0.0;
    double factor = static_cast<double>(Natoms * (Natoms - 1)) / 2.0;
    
    #pragma omp parallel for reduction(+:lin_index)
    for (size_t i = 0; i < Natoms; ++i) {
        for (size_t j = i + 1; j < Natoms; ++j) {
            double rij_squared_mean = pos_variance(i, j) / Nframes;
            double rij_mean = pos_mean(i, j) / Nframes;
            double delta = rij_squared_mean - rij_mean * rij_mean;
            
            if (delta > 0) {
                lin_index += std::sqrt(delta) / rij_mean;
            }
        }
    }
    
    return lin_index / factor;
}

/**
 * 计算局部和全局Lindemann指数（使用Welford方法）
 * 
 * @param pos_list_py: (Nframes, Natoms, 3) 位置数组
 * @param pos_mean_py: (Natoms, Natoms) 距离均值数组
 * @param pos_variance_py: (Natoms, Natoms) 距离累积方差数组
 * @param lindemann_frame_py: (Nframes,) 每帧的Lindemann指数
 * @param lindemann_atom_py: (Nframes, Natoms) 每个原子的局部Lindemann指数
 */
void compute_all(
    const RThreeArrayD pos_list_py,
    TwoArrayD pos_mean_py,
    TwoArrayD pos_variance_py,
    OneArrayD lindemann_frame_py,
    TwoArrayD lindemann_atom_py
) {
    auto pos_list = pos_list_py.view();
    auto pos_mean = pos_mean_py.view();
    auto pos_variance = pos_variance_py.view();
    auto lindemann_frame = lindemann_frame_py.view();
    auto lindemann_atom = lindemann_atom_py.view();
    
    size_t Nframes = pos_list.shape(0);
    size_t Natoms = pos_list.shape(1);
    double factor = static_cast<double>(Natoms * (Natoms - 1));
    // 使用Welford方法逐帧更新均值和方差
    for (size_t frame = 0; frame < Nframes; ++frame) {
        // 更新所有原子对的距离均值和方差
        for (size_t i = 0; i < Natoms; ++i) {
            for (size_t j = i + 1; j < Natoms; ++j) {
                // 计算当前帧的rij距离
                double dx = pos_list(frame, i, 0) - pos_list(frame, j, 0);
                double dy = pos_list(frame, i, 1) - pos_list(frame, j, 1);
                double dz = pos_list(frame, i, 2) - pos_list(frame, j, 2);
                double rijdis = std::sqrt(dx * dx + dy * dy + dz * dz);
                
                // Welford方法更新均值和方差
                double mean = pos_mean(i, j);
                double var = pos_variance(i, j);
                double delta = rijdis - mean;
                
                pos_mean(i, j) = mean + delta / (frame + 1);
                pos_variance(i, j) = var + delta * (rijdis - pos_mean(i, j));
                
                // 对称存储
                pos_mean(j, i) = pos_mean(i, j);
                pos_variance(j, i) = pos_variance(i, j);
            }
        }
        
        // 计算当前帧的Lindemann指数
        double lindemann_index = 0.0;
        
        for (size_t i = 0; i < Natoms; ++i) {
            double atom_lindemann = 0.0;
            
            for (size_t j = 0; j < Natoms; ++j) {
                if (i != j && pos_variance(i, j) > 0) {
                    double ldm = std::sqrt(pos_variance(i, j) / (frame + 1)) / pos_mean(i, j);
                    lindemann_index += ldm;
                    atom_lindemann += ldm / (Natoms - 1);
                }
            }
            
            lindemann_atom(frame, i) = atom_lindemann;
        }
        
        lindemann_index /= factor;
        lindemann_frame(frame) = lindemann_index;
    }
}

// Nanobind模块定义
NB_MODULE(_lindemann, m) {
    
    m.def("compute_global", &compute_global);
    m.def("compute_all", &compute_all);
}