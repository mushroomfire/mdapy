// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include "voro.hpp"
#include "voro++.hh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;
using namespace voro;

void get_voronoi_volume(py::array pos, py::array box, py::array boundary, py::array vol, py::array neighbor_number, py::array cavity_radius)
{
    voronoicell_neighbor c;
    vector<int> neigh;
    // double dx, dy, dz, boxx, boxy, boxz, rmax;
    int tnx, tny, tnz, ti;
    auto c_pos = pos.mutable_unchecked<double, 2>();
    auto c_box = box.mutable_unchecked<double, 2>();
    auto c_boundary = boundary.mutable_unchecked<bool, 1>();
    auto c_vol = vol.mutable_unchecked<double, 1>();
    auto c_neighbor_number = neighbor_number.mutable_unchecked<int, 1>();
    auto c_cavity_radius = cavity_radius.mutable_unchecked<double, 1>();
    int N = c_pos.shape(0);
    // auto result = py::array_t<double>(N);
    // auto c_result = result.request();
    // double* ptr_result = (double*)c_result.ptr;
    // boxx = c_box(0,1)-c_box(0,0);
    // boxy = c_box(1,1)-c_box(1,0);
    // boxz = c_box(2,1)-c_box(2,0);
    pre_container pcon(c_box(0, 0), c_box(0, 1), c_box(1, 0), c_box(1, 1), c_box(2, 0), c_box(2, 1), c_boundary(0), c_boundary(1), c_boundary(2));
    for (int i = 0; i < N; i++)
    {
        pcon.put(i, c_pos(i, 0), c_pos(i, 1), c_pos(i, 2));
    }
    pcon.guess_optimal(tnx, tny, tnz);
    container con(c_box(0, 0), c_box(0, 1), c_box(1, 0), c_box(1, 1), c_box(2, 0), c_box(2, 1), tnx, tny, tnz, c_boundary(0), c_boundary(1), c_boundary(2), 8);
    pcon.setup(con);
    c_loop_all cl(con);
    if (cl.start())
        do
            if (con.compute_cell(c, cl))
            {
                ti = cl.pid();
                c.neighbors(neigh);
                c_vol(ti) = c.volume();
                c_neighbor_number(ti) = neigh.size();
                c_cavity_radius(ti) = sqrt(c.max_radius_squared());
                // rmax=0.0;
                // for (int tj=0; tj < c_neighbor_number(ti); tj++){
                //    // dx, dy, dz = c_pos(ti) - c_pos(neigh[tj]);
                //    dx = c_pos(ti, 0) - c_pos(neigh[tj], 0);
                //    dy = c_pos(ti, 1) - c_pos(neigh[tj], 1);
                //    dz = c_pos(ti, 2) - c_pos(neigh[tj], 2);
                //    if (c_boundary(0)){
                //        dx -= boxx*round(dx/boxx);
                //    }
                //    if (c_boundary(1)){
                //        dy -= boxy*round(dy/boxy);
                //    }
                //    if (c_boundary(2)){
                //        dz -= boxz*round(dz/boxz);
                //    }
                //    double r = sqrt(dx*dx+dy*dy+dz*dz);
                //    if (r > rmax){
                //        rmax = r;
                //    }
                // }
            }
        while (cl.inc());
}
