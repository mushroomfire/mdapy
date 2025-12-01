#include "type.h"
#include <omp.h>
#include <algorithm>

// 检查两个HCP原子是否堆叠在相邻层
inline bool are_stacked(
    const int a, const int b,
    const int* basal_neighbors,
    const int* hcp_neighbors,
    const int n_cols) {
    
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (hcp_neighbors[a * n_cols + basal_neighbors[i]] == 
                hcp_neighbors[b * n_cols + basal_neighbors[j]]) {
                return false;
            }
        }
    }
    return true;
}

// 二分查找
inline int binary_search(const int* arr, const int n, const int value) {
    int left = 0;
    int right = n - 1;
    int mid = 0;
    
    while (left <= right) {
        mid = (left + right) / 2;
        if (value < arr[mid]) {
            right = mid - 1;
        } else if (value > arr[mid]) {
            left = mid + 1;
        } else {
            break;
        }
    }
    return mid;
}

/**
 * 识别FCC结构中的层错和孪晶界（使用指针优化版本）
 * 
 * 参数:
 *   hcp_indices: HCP原子的索引数组
 *   hcp_neighbors: HCP邻居映射数组 (会被修改)
 *   ptm_indices: PTM邻居索引 (已处理为12列)
 *   structure_types: 结构类型数组
 *   fault_types: 层错类型数组 (会被修改，这是输出)
 */
void identify_sftb_fcc(
    const ROneArrayI hcp_indices,
    TwoArrayI hcp_neighbors,
    const RTwoArrayI ptm_indices,
    const ROneArrayI structure_types,
    OneArrayI fault_types,
    const bool identify_esf
) {
    
    // 获取数组维度
    size_t n_hcp = hcp_indices.shape(0);
    size_t n_atoms = structure_types.shape(0);
    
    // 获取指针（性能优化）
    const int* hcp_idx_ptr = hcp_indices.data();
    int* hcp_neigh_ptr = hcp_neighbors.data();
    const int* ptm_idx_ptr = ptm_indices.data();
    const int* struct_type_ptr = structure_types.data();
    int* fault_type_ptr = fault_types.data();
    
    // 定义邻居关系
    const int layer_dir[12] = {0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1};
    const int basal_neighbors[6] = {0, 1, 5, 6, 7, 8};
    const int outofplane_neighbors[6] = {2, 3, 4, 9, 10, 11};
    
    // ========== 第一步：建立HCP邻居映射 ==========
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_hcp; i++) {
        int aindex = hcp_idx_ptr[i];
        int* hcp_neigh_row = hcp_neigh_ptr + i * 12;
        const int* ptm_row = ptm_idx_ptr + aindex * 12;
        
        for (int j = 0; j < 12; j++) {
            int bindex = ptm_row[j];
            if (struct_type_ptr[bindex] == 2) {  // HCP
                hcp_neigh_row[j] = binary_search(hcp_idx_ptr, n_hcp, bindex);
            } else {
                hcp_neigh_row[j] = -bindex - 1;  // 编码非HCP
            }
        }
    }
    
    // ========== 第二步：初步分类每个HCP原子 ==========
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_hcp; i++) {
        int aindex = hcp_idx_ptr[i];
        const int* hcp_neigh_row = hcp_neigh_ptr + i * 12;
        
        int n_basal = 0;         // 同一基面的HCP邻居数
        int n_positive = 0;      // 上方的HCP邻居数
        int n_negative = 0;      // 下方的HCP邻居数
        int n_fcc_positive = 0;  // 上方的FCC邻居数
        int n_fcc_negative = 0;  // 下方的FCC邻居数
        
        for (int j = 0; j < 12; j++) {
            int neighbor_idx = hcp_neigh_row[j];
            
            if (neighbor_idx >= 0) {
                // 这是一个HCP邻居
                if (layer_dir[j] == 0) {
                    n_basal++;
                } else if (are_stacked(i, neighbor_idx, basal_neighbors, 
                                      hcp_neigh_ptr, 12)) {
                    if (layer_dir[j] == 1) {
                        n_positive++;
                    } else {
                        n_negative++;
                    }
                }
            } else if (layer_dir[j] != 0) {
                // 这是一个非HCP邻居
                int neighbor_type = struct_type_ptr[-neighbor_idx - 1];
                if (neighbor_type == 1) {  // FCC
                    if (layer_dir[j] > 0) {
                        n_fcc_positive++;
                    } else {
                        n_fcc_negative++;
                    }
                }
            }
        }
        
        // 根据邻居配置分类
        if ((n_positive != 0 && n_negative == 0) || 
            (n_positive == 0 && n_negative != 0)) {
            // 内禀层错
            fault_type_ptr[aindex] = 2;
        } else if (n_basal >= 1 && n_positive == 0 && n_negative == 0 &&
                   n_fcc_positive != 0 && n_fcc_negative != 0) {
            // 共格孪晶界
            fault_type_ptr[aindex] = 3;
        } else if (n_positive != 0 && n_negative != 0) {
            // 多层层错
            fault_type_ptr[aindex] = 4;
        } else {
            // 孤立的HCP原子
            fault_type_ptr[aindex] = 1;
        }
    }
    
    // ========== 第三步：基于邻居细化TB和孤立原子（必须串行）==========
    for (size_t i = 0; i < n_hcp; i++) {
        int aindex = hcp_idx_ptr[i];
        const int* hcp_neigh_row = hcp_neigh_ptr + i * 12;
        
        if (fault_type_ptr[aindex] == 3 || fault_type_ptr[aindex] == 1) {
            int n_isf_neighbors = 0;   // 基面的ISF邻居数
            int n_twin_neighbors = 0;  // 基面的TB邻居数
            
            // 检查6个基面邻居
            for (int jj = 0; jj < 6; jj++) {
                int j = basal_neighbors[jj];
                int neighbor_index = hcp_neigh_row[j];
                
                if (neighbor_index >= 0) {
                    int neighbor_atom = hcp_idx_ptr[neighbor_index];
                    int neighbor_fault = fault_type_ptr[neighbor_atom];
                    
                    if (neighbor_fault == 2) {
                        n_isf_neighbors++;
                    } else if (neighbor_fault == 3) {
                        n_twin_neighbors++;
                    }
                }
            }
            
            // 根据邻居重新分类
            if (n_isf_neighbors != 0 && n_twin_neighbors == 0) {
                fault_type_ptr[aindex] = 2;  // TB转换为ISF
            } else if (n_isf_neighbors == 0 && n_twin_neighbors != 0) {
                fault_type_ptr[aindex] = 3;  // 孤立原子转换为TB
            }
            
        } else if (fault_type_ptr[aindex] == 4) {
            // 传播多层分类
            for (int jj = 0; jj < 6; jj++) {
                int j = outofplane_neighbors[jj];
                int neighbor_index = hcp_neigh_row[j];
                
                if (neighbor_index >= 0) {
                    int neighbor_atom = hcp_idx_ptr[neighbor_index];
                    if (fault_type_ptr[neighbor_atom] == 2) {
                        fault_type_ptr[neighbor_atom] = 4;
                    }
                }
            }
        }
    }
    
    if (!identify_esf) {
        return;
    }
    // ========== 第四步：识别外禀层错 ==========
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_hcp; i++) {
        int aindex = hcp_idx_ptr[i];
        
        if (fault_type_ptr[aindex] == 3) {  // 孪晶界
            const int* ptm_row = ptm_idx_ptr + aindex * 12;
            bool found_esf = false;
            
            for (int j = 0; j < 12 && !found_esf; j++) {
                int jindex = ptm_row[j];
                
                if (struct_type_ptr[jindex] == 1) {  // FCC邻居
                    int fcc_count = 0, hcp_count = 0;
                    const int* ptm_row_j = ptm_idx_ptr + jindex * 12;
                    
                    // 统计二阶邻居
                    for (int k = 0; k < 12; k++) {
                        int kindex = ptm_row_j[k];
                        int k_type = struct_type_ptr[kindex];
                        
                        if (k_type == 1) {
                            fcc_count++;
                        } else if (k_type == 2) {
                            hcp_count++;
                        }
                    }
                    
                    // ESF特征：FCC原子有5-6个FCC和5-6个HCP邻居
                    if ((5 <= fcc_count && fcc_count <= 6) && 
                        (5 <= hcp_count && hcp_count <= 6)) {
                        fault_type_ptr[aindex] = 5;  // 外禀层错
                        found_esf = true;
                    }
                }
            }
        }
    }
}

// ========== Python绑定 ==========
NB_MODULE(_fccpft, m) {
    m.def("identify_sftb_fcc", &identify_sftb_fcc);
}