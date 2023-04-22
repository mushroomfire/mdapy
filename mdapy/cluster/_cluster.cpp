// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <deque>

namespace py = pybind11;
using namespace std;

int _get_cluster(py::array verlet_list, py::array distance_list, double rc, py::array particleClusters)
{

    auto c_verlet_list = verlet_list.mutable_unchecked<int, 2>();
    auto c_distance_list = distance_list.mutable_unchecked<double, 2>();
    auto c_particleClusters = particleClusters.mutable_unchecked<int, 1>();
    int N = c_verlet_list.shape(0);
    int max_neigh = c_verlet_list.shape(1);
    int clusterid = 0;
    int currentParticle, neighborIndex, nl;

    for (int seedParticleIndex = 0; seedParticleIndex < N; seedParticleIndex++)
    {
        if (c_particleClusters(seedParticleIndex) != -1)
        {
            continue;
        }
        deque<int> toProcess;
        toProcess.push_back(seedParticleIndex);
        clusterid++;
        do
        {
            currentParticle = toProcess.front();
            toProcess.pop_front();
            nl = 0; // let single atom has cluster_id
            for (int j = 0; j < max_neigh; j++)
            {
                neighborIndex = c_verlet_list(currentParticle, j);
                if (neighborIndex > -1)
                {
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
                else
                {
                    break;
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

PYBIND11_MODULE(_cluster_analysis, m)
{
    m.def("_get_cluster", &_get_cluster);
}
