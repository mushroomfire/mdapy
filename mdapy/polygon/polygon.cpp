#include "voro++.hh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;
using namespace voro;

typedef std::vector<std::vector<std::vector<int>>> D3IntVector;
typedef std::vector<std::vector<std::vector<double>>> D3DouVector;

std::tuple<D3IntVector, D3DouVector, std::vector<double>, std::vector<double>, std::vector<std::vector<double>>> get_cell_info(py::array pos, py::array box, py::array boundary)
{
    voronoicell_neighbor cell;

    int tnx, tny, tnz, ti, i, j;
    auto c_pos = pos.mutable_unchecked<double, 2>();
    auto c_box = box.mutable_unchecked<double, 2>();
    auto c_boundary = boundary.mutable_unchecked<bool, 1>();
    int N = c_pos.shape(0);

    D3IntVector face_vertices_indices(N);
    D3DouVector face_vertices_positions(N);
    std::vector<std::vector<double>> face_areas(N);
    std::vector<double> volume(N), radius(N);

    pre_container pcon(c_box(0, 0), c_box(0, 1), c_box(1, 0), c_box(1, 1), c_box(2, 0), c_box(2, 1), c_boundary(0), c_boundary(1), c_boundary(2));
    for (i = 0; i < N; i++)
    {
        pcon.put(i, c_pos(i, 0), c_pos(i, 1), c_pos(i, 2));
    }
    pcon.guess_optimal(tnx, tny, tnz);
    container con(c_box(0, 0), c_box(0, 1), c_box(1, 0), c_box(1, 1), c_box(2, 0), c_box(2, 1), tnx, tny, tnz, c_boundary(0), c_boundary(1), c_boundary(2), 8);
    pcon.setup(con);
    c_loop_all cl(con);
    if (cl.start())
        do
            if (con.compute_cell(cell, cl))
            {
                ti = cl.pid();
                volume[ti] = cell.volume();
                radius[ti] = sqrt(cell.max_radius_squared());

                std::vector<int> face_index_vector;
                cell.face_vertices(face_index_vector);

                std::vector<std::vector<int>> cell_face_vector;
                j = 0;
                while (j < face_index_vector.size())
                {
                    std::vector<int> temp;
                    for (i = 0; i < face_index_vector[j]; i++)
                    {
                        temp.emplace_back(face_index_vector[j + 1 + i]);
                    }
                    j = j + face_index_vector[j] + 1;
                    cell_face_vector.emplace_back(temp);
                }
                face_vertices_indices[ti] = cell_face_vector;

                std::vector<double> vertices;
                cell.vertices(cl.x(), cl.y(), cl.z(), vertices);
                std::vector<std::vector<double>> cell_face_vector_pos(cell.p, std::vector<double>(3));
                for (i = 0; i < cell.p; i++)
                {
                    for (j = 0; j < 3; j++)
                    {
                        cell_face_vector_pos[i][j] = vertices[3 * i + j];
                    }
                }
                face_vertices_positions[ti] = cell_face_vector_pos;

                cell.face_areas(face_areas[ti]);
            }
        while (cl.inc());

    return std::make_tuple(face_vertices_indices, face_vertices_positions, volume, radius, face_areas);
}

PYBIND11_PLUGIN(_poly)
{
    pybind11::module m("_poly", "auto-compiled c++ extension");
    m.def("get_cell_info", &get_cell_info);
    return m.ptr();
}
