#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <omp.h>

void get_wcp(
    const RTwoArrayI verlet_list_py,
    const ROneArrayI neighbor_number_py,
    const ROneArrayI type_list_py,
    const int Ntype,
    TwoArrayD WCP_py
) {
    const int N = verlet_list_py.shape(0);
    auto verlet_list = verlet_list_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto type_list = type_list_py.view();
    auto WCP = WCP_py.view();

    std::vector<int> Zmn(Ntype * Ntype, 0);
    std::vector<int> Zm(Ntype, 0);
    std::vector<double> Alpha_n(Ntype, 0.0);

    // ✅ 使用 OpenMP 并行化
    #pragma omp parallel
    {
        std::vector<int> Zmn_local(Ntype * Ntype, 0);
        std::vector<int> Zm_local(Ntype, 0);
        std::vector<double> Alpha_n_local(Ntype, 0.0);

        #pragma omp for nowait
        for (int i = 0; i < N; ++i) {
            const int itype = type_list(i);
            const int i_neigh = neighbor_number(i);
            
            Alpha_n_local[itype] += 1.0;
            Zm_local[itype] += i_neigh;
            
            for (int jj = 0; jj < i_neigh; ++jj) {
                const int j = verlet_list(i, jj);
                const int jtype = type_list(j);
                Zmn_local[itype * Ntype + jtype]++;
            }
        }

        // 合并局部结果
        #pragma omp critical
        {
            for (int i = 0; i < Ntype * Ntype; ++i) {
                Zmn[i] += Zmn_local[i];
            }
            for (int i = 0; i < Ntype; ++i) {
                Zm[i] += Zm_local[i];
                Alpha_n[i] += Alpha_n_local[i];
            }
        }
    }

    // 归一化
    for (int i = 0; i < Ntype; ++i) {
        Alpha_n[i] /= N;
    }

    // 计算 WCP
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Ntype; ++i) {
        for (int j = 0; j < Ntype; ++j) {
            if (Alpha_n[j] > 0 && Zm[i] > 0) {
                WCP(i, j) = 1.0 - static_cast<double>(Zmn[i * Ntype + j]) / 
                                  (Alpha_n[j] * Zm[i]);
            } else {
                WCP(i, j) = 0.0;
            }
        }
    }
}

NB_MODULE(_wcp, m)
{
    m.def("get_wcp", &get_wcp);

}

