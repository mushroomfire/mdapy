#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <deque>
#include <omp.h>

int get_cluster(const RTwoArrayI verlet_list,
                const RTwoArrayD distance_list,
                const ROneArrayI neighbor_number,
                const double rc,
                OneArrayI particleClusters)
{

    auto c_verlet_list = verlet_list.view();
    auto c_distance_list = distance_list.view();
    auto c_neighbor_number = neighbor_number.view();
    auto c_particleClusters = particleClusters.view();
    int N = c_verlet_list.shape(0);
    int clusterid = 0;
    int currentParticle, neighborIndex, nl;

    for (int seedParticleIndex = 0; seedParticleIndex < N; seedParticleIndex++)
    {
        if (c_particleClusters(seedParticleIndex) != -1)
        {
            continue;
        }
        std::deque<int> toProcess;
        toProcess.push_back(seedParticleIndex);
        clusterid++;
        do
        {
            currentParticle = toProcess.front();
            toProcess.pop_front();
            nl = 0; // let single atom has cluster_id
            int max_neigh = c_neighbor_number(currentParticle);
            for (int j = 0; j < max_neigh; j++)
            {
                neighborIndex = c_verlet_list(currentParticle, j);

                if (c_distance_list(currentParticle, j) <= rc)
                {
                    nl++;
                    if (c_particleClusters(neighborIndex) == -1)
                    {
                        c_particleClusters(neighborIndex) = clusterid;
                        toProcess.push_back(neighborIndex);
                    }
                }
            }
            if (nl == 0)
            {
                c_particleClusters(currentParticle) = clusterid;
            }
        } while (toProcess.empty() == false); // toProcess is not empty, do the loop;
    }
    return clusterid;
}

int get_cluster_by_bond(
    const RTwoArrayI verlet_list,
    const ROneArrayI neighbor_number,
    OneArrayI particleClusters)
{

    auto c_verlet_list = verlet_list.view();
    auto c_neighbor_number = neighbor_number.view();
    auto c_particleClusters = particleClusters.view();
    int N = c_verlet_list.shape(0);
    int clusterid = 0;
    int currentParticle, neighborIndex, nl;

    for (int seedParticleIndex = 0; seedParticleIndex < N; seedParticleIndex++)
    {
        if (c_particleClusters(seedParticleIndex) != -1)
        {
            continue;
        }
        std::deque<int> toProcess;
        toProcess.push_back(seedParticleIndex);
        clusterid++;

        do
        {
            currentParticle = toProcess.front();
            toProcess.pop_front();
            nl = 0; // let single atom has cluster_id
            int max_neigh = c_neighbor_number(currentParticle);
            for (int j = 0; j < max_neigh; j++)
            {
                neighborIndex = c_verlet_list(currentParticle, j);
                if (neighborIndex > -1)
                {

                    nl++;
                    if (c_particleClusters(neighborIndex) == -1)
                    {
                        c_particleClusters(neighborIndex) = clusterid;
                        toProcess.push_back(neighborIndex);
                    }
                }
            }
            if (nl == 0)
            {
                c_particleClusters(currentParticle) = clusterid;
            }
        } while (toProcess.empty() == false); // toProcess is not empty, do the loop;
    }
    return clusterid;
}

void filter_by_type(
    TwoArrayI verlet_list_py,
    const RTwoArrayD distance_list_py,
    const ROneArrayI neighbor_number_py,
    const ROneArrayI type_list_py,
    const ROneArrayI type1_py,
    const ROneArrayI type2_py,
    const ROneArrayD r_py)
{
    auto verlet_list = verlet_list_py.view();
    auto distance_list = distance_list_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto type_list = type_list_py.view();
    auto type1 = type1_py.view();
    auto type2 = type2_py.view();
    auto r = r_py.view();

    const int N = verlet_list_py.shape(0);
    const int ntype = type1_py.shape(0);
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        const int n_neighbor = neighbor_number(i);
        for (int jj = 0; jj < n_neighbor; ++jj)
        {
            const int j = verlet_list(i, jj);
            for (int k = 0; k < ntype; ++k)
            {
                if ((type1(k) == type_list(i)) & (type2(k) == type_list(j)) & (distance_list(i, jj) > r(k)))
                {
                    verlet_list(i, jj) = -1;
                }
            }
        }
    }
}

NB_MODULE(_cluster, m)
{
    m.def("get_cluster", &get_cluster);
    m.def("get_cluster_by_bond", &get_cluster_by_bond);
    m.def("filter_by_type", &filter_by_type);
}