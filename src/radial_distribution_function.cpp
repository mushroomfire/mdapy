// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include "type.h"
#include "box.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>

namespace nb = nanobind;

// ============================================================================
// Verlet-list-based RDF kernels (legacy path).
//
// These take a pre-built neighbor list and bin distances by atom-type pair.
// ============================================================================

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

// ============================================================================
// Streaming RDF kernel.
//
// Bins pair distances directly into a (Ntype, Ntype, nbin) histogram without
// materialising a Verlet list. Memory scales with O(N + Ntype^2 * nbin) per
// thread, independent of the per-atom neighbour count — so rc up to L/2 is
// affordable on boxes where the verlet-list path would OOM.
//
// Two paths share the same per-pair binning code:
//   * cell_list:  ncell[i] >= 3 along every periodic axis. Standard 27-cell
//                 window with PBC wrap via mod(); no neighbour double-count.
//   * all_pairs:  fallback when the box is too small for a safe 3-cell grid.
//                 O(N^2) time; the cell-list path also degenerates to ~O(N^2)
//                 in this regime, so falling back avoids the bookkeeping cost
//                 of a partially-useful grid.
// ============================================================================

namespace
{

inline int safe_mod(int a, int b)
{
    int r = a % b;
    return r + (r >> 31 & b);
}

inline void cell_index_for(const Box &box, double xi, double yi, double zi,
                           const int *ncell, int &icel, int &jcel, int &kcel)
{
    // Maps a position (already wrapped if PBC) onto integer cell indices.
    // For triclinic boxes the inverse-box matrix gives fractional coords.
    if (box.triclinic)
    {
        double dx = xi - box.origin[0];
        double dy = yi - box.origin[1];
        double dz = zi - box.origin[2];
        double nx = dx * box.data[9] + dy * box.data[12] + dz * box.data[15];
        double ny = dx * box.data[10] + dy * box.data[13] + dz * box.data[16];
        double nz = dx * box.data[11] + dy * box.data[14] + dz * box.data[17];
        icel = static_cast<int>(std::floor(nx * ncell[0]));
        jcel = static_cast<int>(std::floor(ny * ncell[1]));
        kcel = static_cast<int>(std::floor(nz * ncell[2]));
    }
    else
    {
        icel = static_cast<int>(std::floor((xi - box.origin[0]) / box.data[0] * ncell[0]));
        jcel = static_cast<int>(std::floor((yi - box.origin[1]) / box.data[4] * ncell[1]));
        kcel = static_cast<int>(std::floor((zi - box.origin[2]) / box.data[8] * ncell[2]));
    }
    icel = std::max(0, std::min(icel, ncell[0] - 1));
    jcel = std::max(0, std::min(jcel, ncell[1] - 1));
    kcel = std::max(0, std::min(kcel, ncell[2] - 1));
}

} // namespace

void _rdf_streaming(const ROneArrayD x_py,
                    const ROneArrayD y_py,
                    const ROneArrayD z_py,
                    const ROneArrayI type_list_py,
                    const RTwoArrayD box_py,
                    const ROneArrayD origin_py,
                    const ROneArrayI boundary_py,
                    ThreeArrayD g_py,
                    const double rc,
                    const int nbin)
{
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto type_list = type_list_py.view();
    auto g = g_py.view();
    const int N = static_cast<int>(x_py.shape(0));
    const int Ntype = static_cast<int>(g_py.shape(0));
    const double dr = rc / nbin;
    const double rcsq = rc * rc;

    Box box = get_box(box_py, origin_py, boundary_py);

    // Decide path: cell-list requires >= 3 cells along each periodic axis so
    // the 27-cell window doesn't double-count via wrap-around.
    int ncell[3]{};
    bool can_use_cell_list = true;
    for (int d = 0; d < 3; ++d)
    {
        ncell[d] = std::max(1, static_cast<int>(std::floor(box.thickness[d] / rc)));
        if (box.boundary[d] && ncell[d] < 3)
            can_use_cell_list = false;
    }

    const int hist_size = Ntype * Ntype * nbin;
    const int n_threads = omp_get_max_threads();

    // Each thread writes into its own contiguous (Ntype, Ntype, nbin) buffer.
    // Final reduction merges them into the caller's view.
    std::vector<double> thread_hist(static_cast<size_t>(n_threads) * hist_size, 0.0);

    if (can_use_cell_list)
    {
        const int total_cell = ncell[0] * ncell[1] * ncell[2];
        std::vector<int> cell_head(total_cell, -1);
        std::vector<int> next_atom(N, -1);

        // Build the cell list by linking atoms into a per-cell singly-linked
        // chain (cell_head[c] points to the first atom in cell c; next_atom[i]
        // points to the next atom that shares i's cell, -1 to terminate).
        for (int i = 0; i < N; ++i)
        {
            double xi = x(i), yi = y(i), zi = z(i);
            if (box.boundary[0] || box.boundary[1] || box.boundary[2])
                box.wrap_into_box(xi, yi, zi);
            int ic, jc, kc;
            cell_index_for(box, xi, yi, zi, ncell, ic, jc, kc);
            int idx = ic * ncell[1] * ncell[2] + jc * ncell[2] + kc;
            next_atom[i] = cell_head[idx];
            cell_head[idx] = i;
        }

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double *local = thread_hist.data() + static_cast<size_t>(tid) * hist_size;

#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < N; ++i)
            {
                double xi = x(i), yi = y(i), zi = z(i);
                if (box.boundary[0] || box.boundary[1] || box.boundary[2])
                    box.wrap_into_box(xi, yi, zi);
                int ic, jc, kc;
                cell_index_for(box, xi, yi, zi, ncell, ic, jc, kc);
                const int it = type_list(i);

                for (int dx = -1; dx <= 1; ++dx)
                {
                    int icn = box.boundary[0] ? safe_mod(ic + dx, ncell[0]) : ic + dx;
                    if (icn < 0 || icn >= ncell[0])
                        continue;
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        int jcn = box.boundary[1] ? safe_mod(jc + dy, ncell[1]) : jc + dy;
                        if (jcn < 0 || jcn >= ncell[1])
                            continue;
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            int kcn = box.boundary[2] ? safe_mod(kc + dz, ncell[2]) : kc + dz;
                            if (kcn < 0 || kcn >= ncell[2])
                                continue;
                            int idx = icn * ncell[1] * ncell[2] + jcn * ncell[2] + kcn;
                            int j = cell_head[idx];
                            while (j != -1)
                            {
                                if (j != i)
                                {
                                    double rx = x(j) - xi;
                                    double ry = y(j) - yi;
                                    double rz = z(j) - zi;
                                    box.pbc(rx, ry, rz);
                                    double r2 = rx * rx + ry * ry + rz * rz;
                                    if (r2 < rcsq)
                                    {
                                        double r = std::sqrt(r2);
                                        int k = static_cast<int>(r / dr);
                                        if (k < nbin)
                                        {
                                            int jt = type_list(j);
                                            local[(it * Ntype + jt) * nbin + k] += 1.0;
                                        }
                                    }
                                }
                                j = next_atom[j];
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        // All-pairs streaming. rc is large relative to box thickness, so a
        // safe cell grid would have <3 cells per periodic axis and double-
        // count — go direct.
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double *local = thread_hist.data() + static_cast<size_t>(tid) * hist_size;

#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < N; ++i)
            {
                double xi = x(i), yi = y(i), zi = z(i);
                if (box.boundary[0] || box.boundary[1] || box.boundary[2])
                    box.wrap_into_box(xi, yi, zi);
                const int it = type_list(i);
                for (int j = 0; j < N; ++j)
                {
                    if (j == i)
                        continue;
                    double rx = x(j) - xi;
                    double ry = y(j) - yi;
                    double rz = z(j) - zi;
                    box.pbc(rx, ry, rz);
                    double r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 < rcsq)
                    {
                        double r = std::sqrt(r2);
                        int k = static_cast<int>(r / dr);
                        if (k < nbin)
                        {
                            int jt = type_list(j);
                            local[(it * Ntype + jt) * nbin + k] += 1.0;
                        }
                    }
                }
            }
        }
    }

    // Reduce thread-local histograms into the caller's nb::ndarray view.
    for (int it = 0; it < Ntype; ++it)
        for (int jt = 0; jt < Ntype; ++jt)
            for (int k = 0; k < nbin; ++k)
            {
                double s = 0.0;
                for (int t = 0; t < n_threads; ++t)
                    s += thread_hist[static_cast<size_t>(t) * hist_size + (it * Ntype + jt) * nbin + k];
                g(it, jt, k) += s;
            }
}

NB_MODULE(_rdf, m)
{
    m.def("_rdf", &_rdf);
    m.def("_rdf_single_species", &_rdf_single_species);
    m.def("_rdf_streaming", &_rdf_streaming,
          nb::arg("x"), nb::arg("y"), nb::arg("z"),
          nb::arg("type_list"),
          nb::arg("box"), nb::arg("origin"), nb::arg("boundary"),
          nb::arg("g"), nb::arg("rc"), nb::arg("nbin"),
          "Streaming RDF: bin pair distances into a (Ntype,Ntype,nbin) "
          "histogram with no Verlet list materialised.");
}
