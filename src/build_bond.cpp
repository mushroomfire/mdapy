// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
#include "type.h"
#include <nanobind/nanobind.h>
#include <vector>
#include <omp.h>

namespace nb = nanobind;

auto build_bond(
    const RTwoArrayI verlet_list,
    const RTwoArrayD distance_list,
    const ROneArrayI neighbor_number,
    const ROneArrayI type_list,
    const RTwoArrayD cutoff_matrix,
    const int num_t)
{
    const int *verlet = verlet_list.data();
    const double *distances = distance_list.data();
    const int *neigh_num = neighbor_number.data();
    const int *types = type_list.data();
    const double *cutoff = cutoff_matrix.data();

    const int N = static_cast<int>(verlet_list.shape(0));
    const int max_neigh = static_cast<int>(verlet_list.shape(1));
    const int ntype = static_cast<int>(cutoff_matrix.shape(1));

    const int nthread = num_t;
    std::vector<std::vector<int>> local_bonds(nthread);

#pragma omp parallel num_threads(num_t)
    {
        const int tid = omp_get_thread_num();
        auto &bonds = local_bonds[tid];

#pragma omp for schedule(dynamic)
        for (int i = 0; i < N; ++i)
        {
            const int itype = types[i];
            const int i_neigh = neigh_num[i];
            for (int jj = 0; jj < i_neigh; ++jj)
            {
                const int j = verlet[i * max_neigh + jj];
                if (j <= i)
                {
                    continue;
                }

                const int jtype = types[j];
                if (itype < 0 || itype >= ntype || jtype < 0 || jtype >= ntype)
                {
                    continue;
                }

                const double rc = cutoff[itype * ntype + jtype];
                if (distances[i * max_neigh + jj] <= rc)
                {
                    bonds.push_back(i);
                    bonds.push_back(j);
                }
            }
        }
    }

    size_t nbond = 0;
    for (const auto &bonds : local_bonds)
    {
        nbond += bonds.size() / 2;
    }

    int *bond_data = new int[nbond * 2];
    nb::capsule owner(bond_data, [](void *p) noexcept
                      { delete[] (int *) p; });

    size_t offset = 0;
    for (const auto &bonds : local_bonds)
    {
        for (size_t i = 0; i < bonds.size(); ++i)
        {
            bond_data[offset + i] = bonds[i];
        }
        offset += bonds.size();
    }

    return nb::ndarray<nb::numpy, int>(
        bond_data,
        {nbond, static_cast<size_t>(2)},
        owner);
}

NB_MODULE(_build_bond, m)
{
    m.def("build_bond", &build_bond, "Build bond pairs from neighbor information.");
}
