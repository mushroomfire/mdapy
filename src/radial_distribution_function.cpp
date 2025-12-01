
#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

void _rdf(const RTwoArrayI verlet_list_py,
          const RTwoArrayD distance_list_py,
          const ROneArrayI neighbor_number_py,
          const ROneArrayI type_list_py,
          ThreeArrayD g_py,
          const double rc,
          const int nbin)
{
    auto verlet_list = verlet_list_py.view();
    auto distance_list = distance_list_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto type_list = type_list_py.view();
    auto g = g_py.view();
    const int N = verlet_list_py.shape(0);
    const double dr = rc / nbin;

    for (int i = 0; i < N; i++)
    {
        int i_type = type_list(i);
        int i_neigh = neighbor_number(i);
        for (int jindex = 0; jindex < i_neigh; jindex++)
        {
            int j = verlet_list(i, jindex);
            double dis = distance_list(i, jindex);
            if (dis < rc)
            {
                int j_type = type_list(j);
                int k = (int)(dis / dr);
                g(i_type, j_type, k) += 1.;
            }
        }
    }
}

void _rdf_single_species(const RTwoArrayI verlet_list_py,
                         const RTwoArrayD distance_list_py,
                         const ROneArrayI neighbor_number_py,
                         OneArrayD g_py,
                         const double rc,
                         const int nbin)
{

    auto verlet_list = verlet_list_py.view();
    auto distance_list = distance_list_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto g = g_py.view();
    const int N = verlet_list_py.shape(0);
    const double dr = rc / nbin;

    for (int i = 0; i < N; i++)
    {
        int i_neigh = neighbor_number(i);
        for (int jindex = 0; jindex < i_neigh; jindex++)
        {
            int j = verlet_list(i, jindex);
            double dis = distance_list(i, jindex);
            if (j > i && dis < rc)
            {
                int k = (int)(dis / dr);
                g(k) += 2.0;
            }
        }
    }
}

NB_MODULE(_rdf, m)
{
    m.def("_rdf", &_rdf);
    m.def("_rdf_single_species", &_rdf_single_species);
}