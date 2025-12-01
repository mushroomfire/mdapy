// repeat_cell.cpp
#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <omp.h>

namespace nb = nanobind;

/*
  repeat_cell:
    - new_pos: preallocated (nx*ny*nz*n_old*3) numpy array to be filled (cartesian)
    - old_box: (3,3) row-wise box vectors (cartesian)
    - old_pos: (n_old,3) basis positions in cartesian (should lie within cell defined by old_box)
    - nx,ny,nz: replication counts
*/

void repeat_cell(
    OneArrayD new_pos,
    RTwoArrayD old_box,
    RTwoArrayD old_pos,
    int nx, int ny, int nz)
{

    auto box = old_box.view();
    auto oldp = old_pos.view();
    auto newp = new_pos.view();

    const size_t n_old = (size_t)old_pos.shape(0);
    const size_t total_cells = (size_t)nx * (size_t)ny * (size_t)nz;

    // Extract old_box rows for speed
    double a1x = box(0, 0), a1y = box(0, 1), a1z = box(0, 2);
    double a2x = box(1, 0), a2y = box(1, 1), a2z = box(1, 2);
    double a3x = box(2, 0), a3y = box(2, 1), a3z = box(2, 2);

// Parallel fill with OpenMP
#pragma omp parallel for schedule(static)
    for (long long cell_idx = 0; cell_idx < (long long)total_cells; ++cell_idx)
    {
        long long tmp = cell_idx;
        int ix = (int)(tmp / (ny * nz));
        tmp = tmp % (ny * nz);
        int iy = (int)(tmp / nz);
        int iz = (int)(tmp % nz);

        double shiftx = ix * a1x + iy * a2x + iz * a3x;
        double shifty = ix * a1y + iy * a2y + iz * a3y;
        double shiftz = ix * a1z + iy * a2z + iz * a3z;

        for (size_t i = 0; i < n_old; ++i)
        {
            size_t out_idx = cell_idx * n_old + i;
            newp(out_idx * 3 + 0) = oldp(i, 0) + shiftx;
            newp(out_idx * 3 + 1) = oldp(i, 1) + shifty;
            newp(out_idx * 3 + 2) = oldp(i, 2) + shiftz;
        }
    }
}

NB_MODULE(_repeat_cell, m)
{
    m.def("repeat_cell", &repeat_cell);
}
