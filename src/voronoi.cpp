#include "box.h"
#include "type.h"
#include <voro++.hh>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <omp.h>
#include <numeric>

void get_voronoi_volume_number_radius(const ROneArrayD x_py,
                                      const ROneArrayD y_py,
                                      const ROneArrayD z_py,
                                      const RTwoArrayD box_py,
                                      const ROneArrayD origin,
                                      const ROneArrayI boundary,
                                      OneArrayD volume_py,
                                      OneArrayI neighbor_number_py,
                                      OneArrayD cavity_radius_py,
                                      const int num_t)
{
    Box box = get_box(box_py, origin, boundary);
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto volume = volume_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto cavity_radius = cavity_radius_py.view();

    const int N{static_cast<int>(x.shape(0))};
    const double ax = 0., bx = box.data[0];
    const double ay = 0., by = box.data[4];
    const double az = 0., bz = box.data[8];
    const double init_mem = 4.6;
    const double ilscale = std::pow(N / (init_mem * box.get_volume()), 1 / 3.0);
    const int nx = int(box.get_box_length(0) * ilscale + 1);
    const int ny = int(box.get_box_length(1) * ilscale + 1);
    const int nz = int(box.get_box_length(2) * ilscale + 1);

    voro::container_3d con(ax, bx, ay, by, az, bz, nx, ny, nz, bool(box.boundary[0]), bool(box.boundary[1]), bool(box.boundary[2]), init_mem, num_t);
#pragma omp parallel for num_threads(num_t) firstprivate(x, y, z)
    for (int i = 0; i < N; ++i)
    {
        con.put_parallel(i, x(i) - box.origin[0], y(i) - box.origin[1], z(i) - box.origin[2]);
    }
    con.put_reconcile_overflow();
#pragma omp parallel num_threads(num_t) firstprivate(volume, neighbor_number, cavity_radius)
    {
        voro::voronoicell_neighbor_3d cell(con);
        int grid_num = con.nx * con.ny * con.nz;
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < grid_num; ++ijk)
        {
            for (int q = 0; q < con.co[ijk]; q++)
            {
                if (con.compute_cell(cell, ijk, q))
                {
                    int i = con.id[ijk][q];
                    volume(i) = cell.volume();
                    neighbor_number(i) = cell.number_of_faces();
                    cavity_radius(i) = std::sqrt(cell.max_radius_squared());
                }
            }
        }
    }
}

void get_voronoi_volume_number_radius_tri(const ROneArrayD x_py,
                                          const ROneArrayD y_py,
                                          const ROneArrayD z_py,
                                          const RTwoArrayD box_py,
                                          const ROneArrayD origin,
                                          const ROneArrayI boundary,
                                          const RTwoArrayD rotation_py,
                                          OneArrayD volume_py,
                                          OneArrayI neighbor_number_py,
                                          OneArrayD cavity_radius_py,
                                          const bool need_rotation,
                                          const int num_t)
{
    Box box = get_box(box_py, origin, boundary);
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto rotation = rotation_py.view();
    auto volume = volume_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto cavity_radius = cavity_radius_py.view();

    const int N{static_cast<int>(x.shape(0))};
    const double bx = box.data[0];
    const double bxy = box.data[3], by = box.data[4];
    const double bxz = box.data[6], byz = box.data[7], bz = box.data[8];
    const double init_mem = 4.6;
    const double ilscale = std::pow(N / (init_mem * box.get_volume()), 1 / 3.0);
    const int nx = int(box.get_box_length(0) * ilscale + 1);
    const int ny = int(box.get_box_length(1) * ilscale + 1);
    const int nz = int(box.get_box_length(2) * ilscale + 1);

    voro::container_triclinic con(bx, bxy, by, bxz, byz, bz, nx, ny, nz, init_mem, num_t);
    // #pragma omp parallel for num_threads(num_t) firstprivate(x, y, z, rotation)
    for (int i = 0; i < N; ++i)
    {
        if (need_rotation)
        {
            double vec[3]{x(i) - box.origin[0], y(i) - box.origin[1], z(i) - box.origin[2]};
            double new_x = vec[0] * rotation(0, 0) + vec[1] * rotation(1, 0) + vec[2] * rotation(2, 0);
            double new_y = vec[0] * rotation(0, 1) + vec[1] * rotation(1, 1) + vec[2] * rotation(2, 1);
            double new_z = vec[0] * rotation(0, 2) + vec[1] * rotation(1, 2) + vec[2] * rotation(2, 2);
            con.put(i, new_x, new_y, new_z);
        }
        else
        {
            con.put(i, x(i) - box.origin[0], y(i) - box.origin[1], z(i) - box.origin[2]);
        }
    }
    std::vector<int> co;
    for (int i = 0; i < con.oxyz; ++i)
    {
        co.emplace_back(con.co[i]);
    }
#pragma omp parallel num_threads(num_t) firstprivate(volume, neighbor_number, cavity_radius)
    {
        voro::voronoicell_neighbor_3d cell(con);
        int grid_num = con.oxyz;
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < grid_num; ++ijk)
        {
            for (int q = 0; q < co[ijk]; q++)
            {
                if (con.compute_cell(cell, ijk, q))
                {
                    int i = con.id[ijk][q];

                    volume(i) = cell.volume();
                    neighbor_number(i) = cell.number_of_faces();
                    cavity_radius(i) = std::sqrt(cell.max_radius_squared());
                }
            }
        }
    }
}

auto get_voronoi_neighbor_tri(const ROneArrayD x_py,
                              const ROneArrayD y_py,
                              const ROneArrayD z_py,
                              const RTwoArrayD box_py,
                              const ROneArrayD origin,
                              const ROneArrayI boundary,
                              const RTwoArrayD rotation_py,
                              const bool need_rotation,
                              const double a_face_area_threshold,
                              const double r_face_area_threshold,
                              const int num_t)
{
    Box box = get_box(box_py, origin, boundary);
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto rotation = rotation_py.view();

    const int N{static_cast<int>(x.shape(0))};
    const double bx = box.data[0];
    const double bxy = box.data[3], by = box.data[4];
    const double bxz = box.data[6], byz = box.data[7], bz = box.data[8];
    const double init_mem = 4.6;
    const double ilscale = std::pow(N / (init_mem * box.get_volume()), 1 / 3.0);
    const int nx = int(box.get_box_length(0) * ilscale + 1);
    const int ny = int(box.get_box_length(1) * ilscale + 1);
    const int nz = int(box.get_box_length(2) * ilscale + 1);

    voro::container_triclinic con(bx, bxy, by, bxz, byz, bz, nx, ny, nz, init_mem, num_t);
    for (int i = 0; i < N; ++i)
    {
        if (need_rotation)
        {
            double vec[3]{x(i) - box.origin[0], y(i) - box.origin[1], z(i) - box.origin[2]};
            double new_x = vec[0] * rotation(0, 0) + vec[1] * rotation(1, 0) + vec[2] * rotation(2, 0);
            double new_y = vec[0] * rotation(0, 1) + vec[1] * rotation(1, 1) + vec[2] * rotation(2, 1);
            double new_z = vec[0] * rotation(0, 2) + vec[1] * rotation(1, 2) + vec[2] * rotation(2, 2);
            con.put(i, new_x, new_y, new_z);
        }
        else
        {
            con.put(i, x(i) - box.origin[0], y(i) - box.origin[1], z(i) - box.origin[2]);
        }
    }
    std::vector<int> co;
    for (int i = 0; i < con.oxyz; ++i)
    {
        co.emplace_back(con.co[i]);
    }

    std::vector<std::vector<int>> voro_verlet_list(N);
    std::vector<std::vector<double>> voro_face_areas(N);
    int *neighbor_number_data = new int[N];
    nb::capsule neighbor_owner(neighbor_number_data, [](void *p) noexcept
                               { delete[] (int *)p; });

#pragma omp parallel num_threads(num_t)
    {
        voro::voronoicell_neighbor_3d cell(con);
        int grid_num = con.oxyz;
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < grid_num; ++ijk)
        {
            for (int q = 0; q < co[ijk]; q++)
            {
                if (con.compute_cell(cell, ijk, q))
                {
                    int i = con.id[ijk][q];

                    cell.neighbors(voro_verlet_list[i]);
                    cell.face_areas(voro_face_areas[i]);
                    neighbor_number_data[i] = voro_verlet_list[i].size();
                }
            }
        }
    }

    size_t max_neighbor = neighbor_number_data[0];
    for (int i = 1; i < N; ++i)
    {
        if (neighbor_number_data[i] > max_neighbor)
        {
            max_neighbor = neighbor_number_data[i];
        }
    }

    int *verlet_list_data = new int[N * max_neighbor];
    nb::capsule verlet_owner(verlet_list_data, [](void *p) noexcept
                             { delete[] (int *)p; });
    double *distance_list_data = new double[N * max_neighbor];
    nb::capsule distance_owner(distance_list_data, [](void *p) noexcept
                               { delete[] (double *)p; });
    double *face_area_data = new double[N * max_neighbor];
    nb::capsule facearea_owner(face_area_data, [](void *p) noexcept
                               { delete[] (double *)p; });

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        int i_neigh = neighbor_number_data[i];
        double i_x = x(i);
        double i_y = y(i);
        double i_z = z(i);
        double area_min = 0.;
        if (a_face_area_threshold > 0)
            area_min = a_face_area_threshold;

        if (r_face_area_threshold > 0.)
        {
            auto vec = voro_face_areas[i];
            area_min = std::accumulate(vec.begin(), vec.end(), 0.0) * r_face_area_threshold;
        }

        // Use the maximum threshold.
        if (a_face_area_threshold > area_min)
            area_min = a_face_area_threshold;
        for (int j = 0; j < max_neighbor; ++j)
        {
            const int ijindex = i * max_neighbor + j;

            if (j < i_neigh)
            {
                const int j_index = voro_verlet_list[i][j];
                const double j_face_area = voro_face_areas[i][j];
                if ((j_index >= 0) && (j_face_area > area_min))
                {
                    verlet_list_data[ijindex] = j_index;
                    face_area_data[ijindex] = voro_face_areas[i][j];

                    double deltax = x(j_index) - i_x;
                    double deltay = y(j_index) - i_y;
                    double deltaz = z(j_index) - i_z;
                    box.pbc(deltax, deltay, deltaz);
                    distance_list_data[ijindex] = std::sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
                }
                else
                {
                    verlet_list_data[ijindex] = -1;
                    face_area_data[ijindex] = 0.;
                    distance_list_data[ijindex] = 10000.;
                }
            }
            else
            {
                verlet_list_data[ijindex] = -1;
                face_area_data[ijindex] = 0.;
                distance_list_data[ijindex] = 10000.;
            }
        }
    }

    return std::make_tuple(
        nb::ndarray<nb::numpy, int>(verlet_list_data, {static_cast<size_t>(N), static_cast<size_t>(max_neighbor)}, verlet_owner),
        nb::ndarray<nb::numpy, double>(distance_list_data, {static_cast<size_t>(N), static_cast<size_t>(max_neighbor)}, distance_owner),
        nb::ndarray<nb::numpy, double>(face_area_data, {static_cast<size_t>(N), static_cast<size_t>(max_neighbor)}, facearea_owner),
        nb::ndarray<nb::numpy, int>(neighbor_number_data, {static_cast<size_t>(N)}, neighbor_owner));
}

auto get_voronoi_neighbor(const ROneArrayD x_py,
                          const ROneArrayD y_py,
                          const ROneArrayD z_py,
                          const RTwoArrayD box_py,
                          const ROneArrayD origin,
                          const ROneArrayI boundary,
                          const double a_face_area_threshold,
                          const double r_face_area_threshold,
                          const int num_t)
{
    Box box = get_box(box_py, origin, boundary);
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();

    const int N{static_cast<int>(x.shape(0))};
    const double ax = 0., bx = box.data[0];
    const double ay = 0., by = box.data[4];
    const double az = 0., bz = box.data[8];
    const double init_mem = 4.6;
    const double ilscale = std::pow(N / (init_mem * box.get_volume()), 1 / 3.0);
    const int nx = int(box.get_box_length(0) * ilscale + 1);
    const int ny = int(box.get_box_length(1) * ilscale + 1);
    const int nz = int(box.get_box_length(2) * ilscale + 1);

    voro::container_3d con(ax, bx, ay, by, az, bz, nx, ny, nz, bool(box.boundary[0]), bool(box.boundary[1]), bool(box.boundary[2]), init_mem, num_t);
#pragma omp parallel for num_threads(num_t) firstprivate(x, y, z)
    for (int i = 0; i < N; ++i)
    {
        con.put_parallel(i, x(i) - box.origin[0], y(i) - box.origin[1], z(i) - box.origin[2]);
    }
    con.put_reconcile_overflow();

    std::vector<std::vector<int>> voro_verlet_list(N);
    std::vector<std::vector<double>> voro_face_areas(N);
    int *neighbor_number_data = new int[N];
    nb::capsule neighbor_owner(neighbor_number_data, [](void *p) noexcept
                               { delete[] (int *)p; });

#pragma omp parallel num_threads(num_t)
    {

        voro::voronoicell_neighbor_3d cell(con);
        int grid_num = con.nx * con.ny * con.nz;
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < grid_num; ++ijk)
        {
            for (int q = 0; q < con.co[ijk]; q++)
            {
                if (con.compute_cell(cell, ijk, q))
                {
                    int i = con.id[ijk][q];
                    cell.neighbors(voro_verlet_list[i]);
                    cell.face_areas(voro_face_areas[i]);
                    neighbor_number_data[i] = voro_verlet_list[i].size();
                }
            }
        }
    }

    size_t max_neighbor = neighbor_number_data[0];
    for (int i = 1; i < N; ++i)
    {
        if (neighbor_number_data[i] > max_neighbor)
        {
            max_neighbor = neighbor_number_data[i];
        }
    }

    int *verlet_list_data = new int[N * max_neighbor];
    nb::capsule verlet_owner(verlet_list_data, [](void *p) noexcept
                             { delete[] (int *)p; });
    double *distance_list_data = new double[N * max_neighbor];
    nb::capsule distance_owner(distance_list_data, [](void *p) noexcept
                               { delete[] (double *)p; });
    double *face_area_data = new double[N * max_neighbor];
    nb::capsule facearea_owner(face_area_data, [](void *p) noexcept
                               { delete[] (double *)p; });

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        int i_neigh = neighbor_number_data[i];
        double i_x = x(i);
        double i_y = y(i);
        double i_z = z(i);
        double area_min = 0.;
        if (a_face_area_threshold > 0)
            area_min = a_face_area_threshold;

        if (r_face_area_threshold > 0.)
        {
            auto vec = voro_face_areas[i];
            area_min = std::accumulate(vec.begin(), vec.end(), 0.0) * r_face_area_threshold;
        }

        // Use the maximum threshold.
        if (a_face_area_threshold > area_min)
            area_min = a_face_area_threshold;

        for (int j = 0; j < max_neighbor; ++j)
        {

            const int ijindex = i * max_neighbor + j;
            if (j < i_neigh)
            {
                const int j_index = voro_verlet_list[i][j];
                const double j_face_area = voro_face_areas[i][j];
                if ((j_index >= 0) && (j_face_area > area_min))
                {
                    verlet_list_data[ijindex] = j_index;
                    face_area_data[ijindex] = voro_face_areas[i][j];

                    double deltax = x(j_index) - i_x;
                    double deltay = y(j_index) - i_y;
                    double deltaz = z(j_index) - i_z;
                    box.pbc(deltax, deltay, deltaz);
                    distance_list_data[ijindex] = std::sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
                }
                else
                {
                    verlet_list_data[ijindex] = -1;
                    face_area_data[ijindex] = 0.;
                    distance_list_data[ijindex] = 10000.;
                }
            }
            else
            {
                verlet_list_data[ijindex] = -1;
                face_area_data[ijindex] = 0.;
                distance_list_data[ijindex] = 10000.;
            }
        }
    }

    return std::make_tuple(
        nb::ndarray<nb::numpy, int>(verlet_list_data, {static_cast<size_t>(N), static_cast<size_t>(max_neighbor)}, verlet_owner),
        nb::ndarray<nb::numpy, double>(distance_list_data, {static_cast<size_t>(N), static_cast<size_t>(max_neighbor)}, distance_owner),
        nb::ndarray<nb::numpy, double>(face_area_data, {static_cast<size_t>(N), static_cast<size_t>(max_neighbor)}, facearea_owner),
        nb::ndarray<nb::numpy, int>(neighbor_number_data, {static_cast<size_t>(N)}, neighbor_owner));
}

auto get_cell_info(const ROneArrayD x_py,
                   const ROneArrayD y_py,
                   const ROneArrayD z_py,
                   const RTwoArrayD box_py,
                   const ROneArrayD origin,
                   const ROneArrayI boundary,
                   const int num_t)
{
    Box box = get_box(box_py, origin, boundary);
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();

    const int N{static_cast<int>(x.shape(0))};
    const double ax = 0., bx = box.data[0];
    const double ay = 0., by = box.data[4];
    const double az = 0., bz = box.data[8];
    const double init_mem = 4.6;
    const double ilscale = std::pow(N / (init_mem * box.get_volume()), 1 / 3.0);
    const int nx = int(box.get_box_length(0) * ilscale + 1);
    const int ny = int(box.get_box_length(1) * ilscale + 1);
    const int nz = int(box.get_box_length(2) * ilscale + 1);

    voro::container_3d con(ax, bx, ay, by, az, bz, nx, ny, nz, bool(box.boundary[0]), bool(box.boundary[1]), bool(box.boundary[2]), init_mem, num_t);
#pragma omp parallel for num_threads(num_t) firstprivate(x, y, z)
    for (int i = 0; i < N; ++i)
    {
        con.put_parallel(i, x(i) - box.origin[0], y(i) - box.origin[1], z(i) - box.origin[2]);
    }
    con.put_reconcile_overflow();

    std::vector<std::vector<std::vector<int>>> face_vertices_indices(N);
    std::vector<std::vector<std::vector<double>>> face_vertices_positions(N);
    std::vector<std::vector<double>> face_areas(N);
    std::vector<double> volume(N), radius(N);
#pragma omp parallel num_threads(num_t)
    {
        voro::voronoicell_neighbor_3d cell(con);
        int grid_num = con.nx * con.ny * con.nz;
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < grid_num; ijk++)
        {
            for (int q = 0; q < con.co[ijk]; q++)
            {
                if (con.compute_cell(cell, ijk, q))
                {
                    int ti = con.id[ijk][q];
                    volume[ti] = cell.volume();
                    radius[ti] = sqrt(cell.max_radius_squared());

                    std::vector<int> face_index_vector;
                    cell.face_vertices(face_index_vector);

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
                    cell.vertices(x, y, z, vertices);
                    std::vector<std::vector<double>> cell_face_vector_pos(cell.p, std::vector<double>(3));
                    for (int i = 0; i < cell.p; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            cell_face_vector_pos[i][j] = vertices[3 * i + j];
                        }
                    }
                    face_vertices_positions[ti] = cell_face_vector_pos;

                    cell.face_areas(face_areas[ti]);
                }
            }
        }
    }

    return std::make_tuple(face_vertices_indices, face_vertices_positions, volume, radius, face_areas);
}

NB_MODULE(_voronoi, m)
{
    m.def("get_voronoi_volume_number_radius", &get_voronoi_volume_number_radius);
    m.def("get_voronoi_volume_number_radius_tri", &get_voronoi_volume_number_radius_tri);
    m.def("get_voronoi_neighbor", &get_voronoi_neighbor);
    m.def("get_voronoi_neighbor_tri", &get_voronoi_neighbor_tri);
    m.def("get_cell_info", &get_cell_info);
}