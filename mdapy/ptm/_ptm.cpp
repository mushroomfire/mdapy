// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.
// We highly thanks to Dr. Peter M Larsen for the help on parallelism of this module.

#include "ptm_functions.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

namespace py = pybind11;

typedef struct
{
    double **pos;
    double *boxsize;
    int **verlet;
    int *boundary;

} ptmnbrdata_t;

static int get_neighbours(void *vdata, size_t central_index,
                          size_t atom_index, int num, size_t *nbr_indices,
                          int32_t *numbers, double (*nbr_pos)[3])
{
    ptmnbrdata_t *data = (ptmnbrdata_t *)vdata;
    double **pos = data->pos;
    int **verlet = data->verlet;
    double *boxsize = data->boxsize;
    int *boundary = data->boundary;

    int num_nbrs = std::min(num - 1, 18);
    nbr_pos[0][0] = nbr_pos[0][1] = nbr_pos[0][2] = 0;
    nbr_indices[0] = atom_index;
    numbers[0] = 0;
    for (int jj = 0; jj < num_nbrs; jj++)
    {

        int j = verlet[atom_index][jj];
        for (int k = 0; k < 3; k++)
        {
            double delta = pos[j][k] - pos[atom_index][k];
            if (boundary[k] == 1)
            {
                delta = delta - boxsize[k] * std::round(delta / boxsize[k]);
            }
            nbr_pos[jj + 1][k] = delta;
        }

        nbr_indices[jj + 1] = j;
        numbers[jj + 1] = 0;
    }
    return num_nbrs + 1;
}

typedef py::array_t<double, py::array::c_style | py::array::forcecast> double_py;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> int_py;
void get_ptm(char *structure, double_py pos, int_py verlet_list, py::array_t<double> cboxsize, py::array_t<int> cboundary, py::array_t<double> coutput, double rmsd_threshold, py::array cptm_indices)
{
    auto pos_buf = pos.request();
    auto verlet_buf = verlet_list.request();
    double *pos_ptr = (double *)pos_buf.ptr;
    int *verlet_ptr = (int *)verlet_buf.ptr;
    int pos_rows = (int)pos_buf.shape[0];
    int verlet_cols = (int)verlet_buf.shape[1];
    double **c_pos = new double *[pos_rows];
    int **c_verlet = new int *[pos_rows];
    auto output = coutput.mutable_unchecked<2>();
    auto ptm_indices = cptm_indices.mutable_unchecked<int, 2>();
    for (int i = 0; i < pos_rows; i++)
    {
        c_pos[i] = pos_ptr + i * 3;
        c_verlet[i] = verlet_ptr + i * verlet_cols;
    }

    double *boxsize = (double *)cboxsize.request().ptr;
    int *boundary = (int *)cboundary.request().ptr;

    char *structures = structure;
    char *ptr = structures;

    const char *strings[] = {"fcc", "hcp", "bcc", "ico", "sc",
                             "dcub", "dhex", "graphene", "all", "default"};
    int num_strings = sizeof(strings) / sizeof(const char *);

    int32_t flags[] = {
        PTM_CHECK_FCC,
        PTM_CHECK_HCP,
        PTM_CHECK_BCC,
        PTM_CHECK_ICO,
        PTM_CHECK_SC,
        PTM_CHECK_DCUB,
        PTM_CHECK_DHEX,
        PTM_CHECK_GRAPHENE,
        PTM_CHECK_ALL,
        PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_BCC | PTM_CHECK_ICO};

    int input_flags = 0;
    while (*ptr != '\0')
    {

        bool found = false;
        for (int i = 0; i < num_strings; i++)
        {
            int len = strlen(strings[i]);
            if (strncmp(ptr, strings[i], len) == 0)
            {
                input_flags |= flags[i];
                ptr += len;
                found = true;
                break;
            }
        }

        if (*ptr == '\0')
            break;

        ptr++;
    }

    ptmnbrdata_t nbrlist = {c_pos, boxsize, c_verlet, boundary};
    ptm_initialize_global();
    
#pragma omp parallel
    {
        ptm_local_handle_t local_handle = ptm_initialize_local();
#pragma omp for
        for (int i = 0; i < pos_rows; i++)
        {
            
            output(i, 0) = -1.0;
            int32_t type, alloy_type;
            double scale, rmsd, interatomic_distance;
            double q[4];
            bool standard_orientations = false;
            size_t p[19];

            ptm_index(local_handle, i, get_neighbours, (void *)&nbrlist,
                      input_flags, standard_orientations,
                      &type, &alloy_type, &scale, &rmsd, q,
                      nullptr, nullptr, nullptr, nullptr, &interatomic_distance, nullptr, p);


            if (rmsd > rmsd_threshold)
            {
                type = 0;
            }

            if (type == PTM_MATCH_NONE)
            {
                type = 0;
                rmsd = 10000.;
            }

            for (int k = 0; k < 18; k++)
            {
                ptm_indices(i, k) = p[k];
            }

            output(i, 0) = type;
            output(i, 1) = rmsd;
            output(i, 2) = interatomic_distance;
            output(i, 3) = q[0];
            output(i, 4) = q[1];
            output(i, 5) = q[2];
            output(i, 6) = q[3];
            
        }
        ptm_uninitialize_local(local_handle);
    }

    delete[] c_pos;
    delete[] c_verlet;
}

PYBIND11_MODULE(_ptm, m)
{
    m.def("get_ptm", &get_ptm);
}
