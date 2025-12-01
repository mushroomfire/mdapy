#include "box.h"
#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

void get_csp(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin,
    const ROneArrayI boundary,
    const RTwoArrayI verlet_list_py,
    const int N,
    OneArrayD csp_py)
{
    const int num_atoms = x_py.shape(0);
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    const Box box = get_box(box_py, origin, boundary);
    auto verlet_list = verlet_list_py.view();
    auto csp = csp_py.view();

    const int num_pairs = (N * (N - 1)) / 2;
    const int half_N = N / 2;
    std::vector<std::pair<int, int>> pair_indices;
    pair_indices.reserve(num_pairs);
    
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            pair_indices.emplace_back(i, j);
        }
    }

    #pragma omp parallel
    {

        std::vector<double> pair_distances(num_pairs);
        
        #pragma omp for schedule(static)
        for (int i = 0; i < num_atoms; ++i)
        {   
            const double xi = x(i);
            const double yi = y(i);
            const double zi = z(i);

            for (int idx = 0; idx < num_pairs; ++idx)
            {
                const int j_idx = pair_indices[idx].first;
                const int k_idx = pair_indices[idx].second;
                
                const int j = verlet_list(i, j_idx);
                const int k = verlet_list(i, k_idx);
                
                double xij = x(j) - xi;
                double yij = y(j) - yi;
                double zij = z(j) - zi;
                double xik = x(k) - xi;
                double yik = y(k) - yi;
                double zik = z(k) - zi;
                
                box.pbc(xij, yij, zij);
                box.pbc(xik, yik, zik);
                
                const double xijk = xij + xik;
                const double yijk = yij + yik;
                const double zijk = zij + zik;
                
                pair_distances[idx] = xijk * xijk + yijk * yijk + zijk * zijk;
            }

            std::partial_sort(pair_distances.begin(), 
                            pair_distances.begin() + half_N, 
                            pair_distances.end());

            double sum = 0.0;
            for (int k = 0; k < half_N; ++k)
            {
                sum += pair_distances[k];
            }
            csp(i) = sum;
        }
    }
}

NB_MODULE(_csp, m)
{
    m.def("get_csp", &get_csp);
}