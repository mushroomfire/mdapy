#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <omp.h>

namespace nb = nanobind;

void calculate_structure_entropy(const double rc, const double sigma,
                                 const bool use_local_density,
                                 const double volume,
                                 const RTwoArrayD distance_list_py,
                                 const ROneArrayI neighbor_number_py,
                                 OneArrayD entropy_py)
{
    const double MY_PI{3.14159265358979323846};

    auto distance_list = distance_list_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto entropy = entropy_py.view();
    const int N{static_cast<int>(distance_list.shape(0))};
    const int nbins{static_cast<int>(std::floor(rc / sigma)) + 1};
    const double global_density{N / volume};
    // malloc memory
    double *g_m = new double[N * nbins]{};
    double *intergrad = new double[N * nbins]{};
    double *rlist = new double[nbins]{};
    double *rlist_sq = new double[nbins]{};
    double *prefactor = new double[nbins]{};

    const double step = rc / (nbins - 1);
    const double factor{(4. * MY_PI * global_density * std::sqrt(2. * MY_PI * sigma * sigma))};

    for (int i = 0; i < nbins; ++i)
    {
        rlist[i] = i * step;
        rlist_sq[i] = rlist[i] * rlist[i];
        prefactor[i] = rlist_sq[i] * factor;
    }
    prefactor[0] = prefactor[1];
    const double sigma_sq = sigma * sigma;
    const double local_vol = 4. / 3. * MY_PI * rc * rc * rc;

#pragma omp parallel for firstprivate(distance_list, neighbor_number, entropy)
    for (int i = 0; i < N; ++i)
    {
        int n_neigh = 0;
        for (int j = 0; j < nbins; ++j)
        {
            for (int k = 0; k < neighbor_number(i); ++k)
            {
                double dis = distance_list(i, k);
                if (dis <= rc)
                {
                    double delta = rlist[j] - dis;
                    g_m[i * nbins + j] += std::exp(
                                              -(delta * delta) / (2.0 * sigma_sq)) /
                                          prefactor[j];
                    if (j == 0)
                    {
                        ++n_neigh;
                    }
                }
            }
        }

        double density{0.0};
        if (use_local_density)
        {
            density = n_neigh / local_vol;
            double fac{global_density / density};
            for (int j = 0; j < nbins; ++j)
            {
                g_m[i * nbins + j] *= fac;
            }
        }
        else
        {
            density = global_density;
        }
        for (int j = 0; j < nbins; ++j)
        {
            if (g_m[i * nbins + j] >= 1e-10)
            {
                intergrad[i * nbins + j] = (g_m[i * nbins + j] * std::log(g_m[i * nbins + j]) - g_m[i * nbins + j] + 1.0) * rlist_sq[j];
            }
            else
            {
                intergrad[i * nbins + j] = rlist_sq[j];
            }
        }
        double sum_intergrad = 0.0;
        for (int j = 0; j < nbins - 1; ++j)
        {
            sum_intergrad += intergrad[i * nbins + j] + intergrad[i * nbins + j + 1];
        }
        entropy(i) = -MY_PI * density * sum_intergrad * sigma;
    }

    delete[] g_m;
    delete[] intergrad;
    delete[] rlist;
    delete[] rlist_sq;
    delete[] prefactor;
}

NB_MODULE(_structure_entropy, m)
{
    m.def("calculate_structure_entropy", &calculate_structure_entropy);
}
