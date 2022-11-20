#include "polygon.hpp"
#include "voro++.hh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/stl.h>
// #include <iostream>
#include <cmath>

namespace py = pybind11;
using namespace voro;

void get_cell_info(py::array pos, py::array box, py::array boundary, py::array volume_radius, py::array face_vertices, py::array vertices_pos, py::array face_areas)
{
    voronoicell_neighbor cell;

    int tnx, tny, tnz, ti;
    auto c_pos = pos.mutable_unchecked<double, 2>();
    auto c_box = box.mutable_unchecked<double, 2>();
    auto c_volume_radius = volume_radius.mutable_unchecked<double, 2>();
    auto c_face_areas = face_areas.mutable_unchecked<double, 2>();
    auto c_face_vertices = face_vertices.mutable_unchecked<int, 3>();
    auto c_vertices_pos = vertices_pos.mutable_unchecked<double, 3>();
    auto c_boundary = boundary.mutable_unchecked<bool, 1>();
    int N = c_pos.shape(0);

    pre_container pcon(c_box(0, 0), c_box(0, 1), c_box(1, 0), c_box(1, 1), c_box(2, 0), c_box(2, 1), c_boundary(0), c_boundary(1), c_boundary(2));
    // cout << "staring adding seeds..." << endl;
    for (int i = 0; i < N; i++)
    {
        pcon.put(i, c_pos(i, 0), c_pos(i, 1), c_pos(i, 2));
        // cout << i << ' ' << c_pos(i, 0) << ' ' << c_pos(i, 1) << ' ' << c_pos(i, 2) << endl;
    }
    pcon.guess_optimal(tnx, tny, tnz);
    container con(c_box(0, 0), c_box(0, 1), c_box(1, 0), c_box(1, 1), c_box(2, 0), c_box(2, 1), tnx, tny, tnz, c_boundary(0), c_boundary(1), c_boundary(2), 8);
    pcon.setup(con);
    c_loop_all cl(con);
    // cout << "added seed pos..." << endl;
    if (cl.start())
        do
            if (con.compute_cell(cell, cl))
            {
                ti = cl.pid();
                vector<double> query_point = {cl.x(), cl.y(), cl.z()};
                // cout << ti << ' ' << query_point[0] << ' ' << query_point[1] << ' ' << query_point[2] << endl;
                // get cell volume
                c_volume_radius(ti, 0) = cell.volume();
                // get maximum of cavity_radius
                c_volume_radius(ti, 1) = sqrt(cell.max_radius_squared());
                // get face_vertices
                vector<int> face_index_vector;
                cell.face_vertices(face_index_vector);
                // face_vertices.emplace_back(vector<vector<int>>());
                int j = 0;
                int n = 0;
                while (j < face_index_vector.size())
                {
                    vector<int> temp;
                    for (int i = 0; i < face_index_vector[j]; i++)
                    {
                        c_face_vertices(ti, n, i) = face_index_vector[j + 1 + i];
                    }
                    j += face_index_vector[j] + 1;
                    n++;
                }
                // get vertices_pos
                vector<double> vertices;
                cell.vertices(query_point[0], query_point[1], query_point[2], vertices);
                for (int i = 0; i < cell.p; i = i++)
                {
                    c_vertices_pos(ti, i, 0) = vertices[3 * i];
                    c_vertices_pos(ti, i, 1) = vertices[3 * i + 1];
                    c_vertices_pos(ti, i, 2) = vertices[3 * i + 2];
                }
                // get face_areas
                vector<double> face_areas_cell;
                cell.face_areas(face_areas_cell);
                for (int i = 0; i < face_areas_cell.size(); i++)
                {
                    c_face_areas(ti, i) = face_areas_cell[i];
                }
            }
        while (cl.inc());
}
