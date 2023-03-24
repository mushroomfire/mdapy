#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/stl.h>
#include <omp.h>

namespace py = pybind11;

void _rdf(py::array cverlet_list, py::array cdistance_list,
          py::array cneighbor_number, py::array ctype_list,
          py::array cg, py::array cconcentrates, double rc,
          int nbin)
{

    auto verlet_list = cverlet_list.unchecked<int, 2>();
    auto distance_list = cdistance_list.unchecked<double, 2>();
    auto neighbor_number = cneighbor_number.unchecked<int, 1>();
    auto type_list = ctype_list.unchecked<int, 1>();
    auto g = cg.mutable_unchecked<double, 3>();
    auto concentrates = cconcentrates.unchecked<double, 1>();
    int N = verlet_list.shape(0);

    double dr = rc / nbin;
    for (int i = 0; i < N; i++)
    {
        int i_type = type_list(i);
        int i_neigh = neighbor_number(i);
        for (int jindex = 0; jindex < i_neigh; jindex++)
        {
            int j = verlet_list(i, jindex);
            double dis = distance_list(i, jindex);
            if (j > i & dis < rc)
            {
                int j_type = type_list(j);
                if (j_type >= i_type)
                {
                    int k = (int)(dis / dr);
                    g(i_type, j_type, k) += 2.0 / concentrates(i_type) / concentrates(j_type);
                }
            }
        }
    }
}

void _rdf_single_species(py::array cverlet_list, py::array cdistance_list,
                         py::array cneighbor_number,
                         py::array cg, double rc,
                         int nbin)
{

    auto verlet_list = cverlet_list.unchecked<int, 2>();
    auto distance_list = cdistance_list.unchecked<double, 2>();
    auto neighbor_number = cneighbor_number.unchecked<int, 1>();
    auto g = cg.mutable_unchecked<double, 1>();
    int N = verlet_list.shape(0);

    double dr = rc / nbin;
    std::vector<std::vector<double>> g_local(omp_get_max_threads(), std::vector<double>(nbin, 0.0));
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int i = tid; i < N; i += omp_get_num_threads())
        {
            int i_neigh = neighbor_number(i);
            for (int jindex = 0; jindex < i_neigh; jindex++)
            {
                int j = verlet_list(i, jindex);
                double dis = distance_list(i, jindex);
                if (j > i & dis < rc)
                {
                    int k = (int)(dis / dr);
                    g_local[tid][k] += 2.0;
                }
            }
        }
    }
    for (int i = 0; i < omp_get_max_threads(); i++)
    {
        for (int j = 0; j < nbin; j++)
        {
#pragma omp atomic
            g(j) += g_local[i][j];
        }
    }
}

PYBIND11_PLUGIN(_rdf)
{
    pybind11::module m("_rdf", "auto-compiled c++ extension");
    m.def("_rdf", &_rdf);
    m.def("_rdf_single_species", &_rdf_single_species);
    return m.ptr();
}
