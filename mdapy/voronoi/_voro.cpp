// Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.
// We highly thanks to Dr. Jiayin Lu, Prof. Christipher Rycroft
// and Prof. Emanuel Lazar for the help on parallelism of this module.

#include "voro++.hh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/stl.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;
using namespace voro;

std::tuple<py::array_t<int>, py::array_t<double>, py::array_t<double>> get_voronoi_neighbor(py::array pos, py::array box, py::array boundary, py::array neighbor_number, int num_t)
{
    auto c_pos = pos.unchecked<double, 2>();
    auto c_box = box.mutable_unchecked<double, 2>();
    auto c_boundary = boundary.mutable_unchecked<bool, 1>();
    auto c_neighbor_number = neighbor_number.mutable_unchecked<int, 1>();
    size_t N = c_pos.shape(0);
    double ax = c_box(0, 0), bx = c_box(0, 1);
    double ay = c_box(1, 0), by = c_box(1, 1);
    double az = c_box(2, 0), bz = c_box(2, 1);
    double dx = bx - ax, dy = by - ay, dz = bz - az;
    double optimal_particles = 4.6;
    double ilscale = pow(N / (optimal_particles * dx * dy * dz), 1 / 3.0);
    int nx = int(dx * ilscale + 1);
    int ny = int(dy * ilscale + 1);
    int nz = int(dz * ilscale + 1);
    container_3d con(ax, bx, ay, by, az, bz, nx, ny, nz, c_boundary(0), c_boundary(1), c_boundary(2), optimal_particles, num_t);

    for (int i = 0; i < N; i++)
        con.put(i, c_pos(i, 0), c_pos(i, 1), c_pos(i, 2));

    std::vector<std::vector<int>> voro_verlet_list(N);
    std::vector<std::vector<double>> voro_face_areas(N);

#pragma omp parallel num_threads(num_t)
    {
        voronoicell_neighbor_3d c(con);
        int grid_num = con.nx * con.ny * con.nz;
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < grid_num; ijk++)
        {
            for (int q = 0; q < con.co[ijk]; q++)
            {
                if (con.compute_cell(c, ijk, q))
                {
                    int i = con.id[ijk][q];
                    c.neighbors(voro_verlet_list[i]);
                    c.face_areas(voro_face_areas[i]);
                    c_neighbor_number(i) = voro_verlet_list[i].size();
                }
            }
        }
    }

    size_t max_neighbor = c_neighbor_number(0);
    for (int i = 1; i < N; ++i)
    {
        if (c_neighbor_number(i) > max_neighbor)
        {
            max_neighbor = c_neighbor_number(i);
        }
    }

    size_t shape[2]{N, max_neighbor};
    auto verlet_list = py::array_t<int>(shape);
    auto c_verlet_list = verlet_list.mutable_unchecked<2>();
    auto face_areas = py::array_t<double>(shape);
    auto c_face_areas = face_areas.mutable_unchecked<2>();
    auto distance_list = py::array_t<double>(shape);
    auto c_distance_list = distance_list.mutable_unchecked<2>();

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        int i_neigh = c_neighbor_number(i);
        double i_x = c_pos(i, 0);
        double i_y = c_pos(i, 1);
        double i_z = c_pos(i, 2);
        for (int j = 0; j < max_neighbor; ++j)
        {
            if (j < i_neigh)
            {
                int j_index = voro_verlet_list[i][j];
                if (j_index >= 0)
                {
                    c_verlet_list(i, j) = j_index;
                    c_face_areas(i, j) = voro_face_areas[i][j];

                    double deltax = c_pos(j_index, 0) - i_x;
                    double deltay = c_pos(j_index, 1) - i_y;
                    double deltaz = c_pos(j_index, 2) - i_z;
                    if (c_boundary(0))
                        deltax = deltax - dx * std::round(deltax / dx);
                    if (c_boundary(1))
                        deltay = deltay - dy * std::round(deltay / dy);
                    if (c_boundary(2))
                        deltaz = deltaz - dz * std::round(deltaz / dz);
                    c_distance_list(i, j) = std::sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
                }
                else
                {
                    c_verlet_list(i, j) = -1;
                    c_face_areas(i, j) = 0.;
                    c_distance_list(i, j) = 10000.;
                }
            }
            else
            {
                c_verlet_list(i, j) = -1;
                c_face_areas(i, j) = 0.;
                c_distance_list(i, j) = 10000.;
            }
        }
    }

    return std::make_tuple(verlet_list, distance_list, face_areas);
}

void get_voronoi_volume(py::array pos, py::array box, py::array boundary, py::array vol, py::array neighbor_number, py::array cavity_radius, int num_t)
{

    // auto c_pos = pos.mutable_unchecked<double, 2>();
    auto c_pos = pos.unchecked<double, 2>();
    auto c_box = box.mutable_unchecked<double, 2>();
    auto c_boundary = boundary.mutable_unchecked<bool, 1>();
    auto c_vol = vol.mutable_unchecked<double, 1>();
    auto c_neighbor_number = neighbor_number.mutable_unchecked<int, 1>();
    auto c_cavity_radius = cavity_radius.mutable_unchecked<double, 1>();
    int N = c_pos.shape(0);
    double ax = c_box(0, 0), bx = c_box(0, 1);
    double ay = c_box(1, 0), by = c_box(1, 1);
    double az = c_box(2, 0), bz = c_box(2, 1);
    double dx = bx - ax, dy = by - ay, dz = bz - az;
    double optimal_particles = 4.6;
    double ilscale = pow(N / (optimal_particles * dx * dy * dz), 1 / 3.0);
    int nx = int(dx * ilscale + 1);
    int ny = int(dy * ilscale + 1);
    int nz = int(dz * ilscale + 1);
    container_3d con(ax, bx, ay, by, az, bz, nx, ny, nz, c_boundary(0), c_boundary(1), c_boundary(2), optimal_particles, num_t);

    for (int i = 0; i < N; i++)
        con.put(i, c_pos(i, 0), c_pos(i, 1), c_pos(i, 2));

#pragma omp parallel num_threads(num_t)
    {
        voronoicell_neighbor_3d c(con);
        int grid_num = con.nx * con.ny * con.nz;
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < grid_num; ijk++)
        {
            for (int q = 0; q < con.co[ijk]; q++)
            {
                if (con.compute_cell(c, ijk, q))
                {
                    int i = con.id[ijk][q];
                    std::vector<int> neigh;
                    c.neighbors(neigh);
                    c_vol(i) = c.volume();
                    c_neighbor_number(i) = neigh.size();
                    c_cavity_radius(i) = sqrt(c.max_radius_squared());
                }
            }
        }
    }
}

typedef std::vector<std::vector<std::vector<int>>> D3IntVector;
typedef std::vector<std::vector<std::vector<double>>> D3DouVector;

std::tuple<D3IntVector, D3DouVector, std::vector<double>, std::vector<double>, std::vector<std::vector<double>>> get_cell_info(py::array pos, py::array box, py::array boundary, int num_t)
{
    // auto c_pos = pos.mutable_unchecked<double, 2>();
    auto c_pos = pos.unchecked<double, 2>();
    auto c_box = box.mutable_unchecked<double, 2>();
    auto c_boundary = boundary.mutable_unchecked<bool, 1>();
    int N = c_pos.shape(0);

    D3IntVector face_vertices_indices(N);
    D3DouVector face_vertices_positions(N);
    std::vector<std::vector<double>> face_areas(N);
    std::vector<double> volume(N), radius(N);

    double ax = c_box(0, 0), bx = c_box(0, 1);
    double ay = c_box(1, 0), by = c_box(1, 1);
    double az = c_box(2, 0), bz = c_box(2, 1);
    double dx = bx - ax, dy = by - ay, dz = bz - az;
    double optimal_particles = 4.6;
    double ilscale = pow(N / (optimal_particles * dx * dy * dz), 1 / 3.0);
    int nx = int(dx * ilscale + 1);
    int ny = int(dy * ilscale + 1);
    int nz = int(dz * ilscale + 1);
    container_3d con(ax, bx, ay, by, az, bz, nx, ny, nz, c_boundary(0), c_boundary(1), c_boundary(2), optimal_particles, num_t);

    for (int i = 0; i < N; i++)
        con.put(i, c_pos(i, 0), c_pos(i, 1), c_pos(i, 2));

#pragma omp parallel num_threads(num_t)
    {
        voronoicell_neighbor_3d c(con);
        int grid_num = con.nx * con.ny * con.nz;
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < grid_num; ijk++)
        {
            for (int q = 0; q < con.co[ijk]; q++)
            {
                if (con.compute_cell(c, ijk, q))
                {
                    int ti = con.id[ijk][q];
                    volume[ti] = c.volume();
                    radius[ti] = sqrt(c.max_radius_squared());

                    std::vector<int> face_index_vector;
                    c.face_vertices(face_index_vector);

                    std::vector<std::vector<int>> cell_face_vector;
                    int j = 0;
                    while (j < face_index_vector.size())
                    {
                        std::vector<int> temp;
                        for (int i = 0; i < face_index_vector[j]; i++)
                        {
                            temp.emplace_back(face_index_vector[j + 1 + i]);
                        }
                        j = j + face_index_vector[j] + 1;
                        cell_face_vector.emplace_back(temp);
                    }
                    face_vertices_indices[ti] = cell_face_vector;

                    std::vector<double> vertices;
                    double x, y, z;
                    double *pp = con.p[ijk] + con.ps * q;
                    x = *(pp++);
                    y = *(pp++);
                    z = *pp;
                    c.vertices(x, y, z, vertices);
                    std::vector<std::vector<double>> cell_face_vector_pos(c.p, std::vector<double>(3));
                    for (int i = 0; i < c.p; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            cell_face_vector_pos[i][j] = vertices[3 * i + j];
                        }
                    }
                    face_vertices_positions[ti] = cell_face_vector_pos;

                    c.face_areas(face_areas[ti]);
                }
            }
        }
    }

    return std::make_tuple(face_vertices_indices, face_vertices_positions, volume, radius, face_areas);
}

PYBIND11_MODULE(_voronoi_analysis, m)
{
    m.def("get_voronoi_neighbor", &get_voronoi_neighbor);
    m.def("get_voronoi_volume", &get_voronoi_volume);
    m.def("get_cell_info", &get_cell_info);
}
