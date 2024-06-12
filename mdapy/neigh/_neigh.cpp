// Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

void build_cell_rec(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double bin_length)
{
    auto pos = cpos.unchecked<double, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto origin = corigin.unchecked<double, 1>();
    auto ncel = cncel.unchecked<int, 1>();
    int N = pos.shape(0);

    int im1 = ncel(0) - 1;
    int jm1 = ncel(1) - 1;
    int km1 = ncel(2) - 1;
    double xlo = origin(0);
    double ylo = origin(1);
    double zlo = origin(2);

    for (int i = 0; i < N; i++)
    {
        int icel = static_cast<int>(std::floor((pos(i, 0) - xlo) / bin_length));
        int jcel = static_cast<int>(std::floor((pos(i, 1) - ylo) / bin_length));
        int kcel = static_cast<int>(std::floor((pos(i, 2) - zlo) / bin_length));
        if (icel < 0)
            icel = 0;
        else if (icel > im1)
            icel = im1;

        if (jcel < 0)
            jcel = 0;
        else if (jcel > jm1)
            jcel = jm1;

        if (kcel < 0)
            kcel = 0;
        else if (kcel > km1)
            kcel = km1;

        atom_cell_list(i) = cell_id_list(icel, jcel, kcel);
        cell_id_list(icel, jcel, kcel) = i;
    }
}

void build_cell_rec_with_jishu(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double bin_length, py::array cmax_neigh_list)
{

    auto pos = cpos.unchecked<double, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto origin = corigin.unchecked<double, 1>();
    auto ncel = cncel.unchecked<int, 1>();
    auto max_neigh_list = cmax_neigh_list.mutable_unchecked<int, 3>();
    int N = pos.shape(0);

    int im1 = ncel(0) - 1;
    int jm1 = ncel(1) - 1;
    int km1 = ncel(2) - 1;
    double xlo = origin(0);
    double ylo = origin(1);
    double zlo = origin(2);

    for (int i = 0; i < N; i++)
    {

        int icel = static_cast<int>(std::floor((pos(i, 0) - xlo) / bin_length));
        int jcel = static_cast<int>(std::floor((pos(i, 1) - ylo) / bin_length));
        int kcel = static_cast<int>(std::floor((pos(i, 2) - zlo) / bin_length));
        if (icel < 0)
            icel = 0;
        else if (icel > im1)
            icel = im1;

        if (jcel < 0)
            jcel = 0;
        else if (jcel > jm1)
            jcel = jm1;

        if (kcel < 0)
            kcel = 0;
        else if (kcel > km1)
            kcel = km1;

        atom_cell_list(i) = cell_id_list(icel, jcel, kcel);
        cell_id_list(icel, jcel, kcel) = i;
        max_neigh_list(icel, jcel, kcel) += 1;
    }
}

void build_cell_tri(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array cbox, py::array cinverse_box, py::array cncel, double bin_length)
{

    auto pos = cpos.unchecked<double, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto ncel = cncel.unchecked<int, 1>();
    auto box = cbox.unchecked<double, 2>();
    auto inverse_box = cinverse_box.unchecked<double, 2>();
    int N = pos.shape(0);

    int im1 = ncel(0) - 1;
    int jm1 = ncel(1) - 1;
    int km1 = ncel(2) - 1;
    double xlo = box(3, 0);
    double ylo = box(3, 1);
    double zlo = box(3, 2);

    for (int i = 0; i < N; i++)
    {
        double dx = pos(i, 0) - xlo;
        double dy = pos(i, 1) - ylo;
        double dz = pos(i, 2) - zlo;

        double nx = dx * inverse_box(0, 0) + dy * inverse_box(1, 0) + dz * inverse_box(2, 0);
        double ny = dx * inverse_box(0, 1) + dy * inverse_box(1, 1) + dz * inverse_box(2, 1);
        double nz = dx * inverse_box(0, 2) + dy * inverse_box(1, 2) + dz * inverse_box(2, 2);

        int icel = static_cast<int>(std::floor((std::sqrt(std::pow(nx * box(0, 0), 2) + std::pow(nx * box(0, 1), 2) + std::pow(nx * box(0, 2), 2))) / bin_length));
        int jcel = static_cast<int>(std::floor((std::sqrt(std::pow(ny * box(1, 0), 2) + std::pow(ny * box(1, 1), 2) + std::pow(ny * box(1, 2), 2))) / bin_length));
        int kcel = static_cast<int>(std::floor((std::sqrt(std::pow(nz * box(2, 0), 2) + std::pow(nz * box(2, 1), 2) + std::pow(nz * box(2, 2), 2))) / bin_length));
        if (icel < 0)
            icel = 0;
        else if (icel > im1)
            icel = im1;

        if (jcel < 0)
            jcel = 0;
        else if (jcel > jm1)
            jcel = jm1;

        if (kcel < 0)
            kcel = 0;
        else if (kcel > km1)
            kcel = km1;

        atom_cell_list(i) = cell_id_list(icel, jcel, kcel);
        cell_id_list(icel, jcel, kcel) = i;
    }
}

void build_cell_tri_with_jishu(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array cbox, py::array cinverse_box, py::array cncel, double bin_length, py::array cmax_neigh_list)
{

    auto pos = cpos.unchecked<double, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto ncel = cncel.unchecked<int, 1>();
    auto max_neigh_list = cmax_neigh_list.mutable_unchecked<int, 3>();
    auto box = cbox.unchecked<double, 2>();
    auto inverse_box = cinverse_box.unchecked<double, 2>();
    int N = pos.shape(0);

    int im1 = ncel(0) - 1;
    int jm1 = ncel(1) - 1;
    int km1 = ncel(2) - 1;
    double xlo = box(3, 0);
    double ylo = box(3, 1);
    double zlo = box(3, 2);

    for (int i = 0; i < N; i++)
    {
        double dx = pos(i, 0) - xlo;
        double dy = pos(i, 1) - ylo;
        double dz = pos(i, 2) - zlo;
        double nx = dx * inverse_box(0, 0) + dy * inverse_box(1, 0) + dz * inverse_box(2, 0);
        double ny = dx * inverse_box(0, 1) + dy * inverse_box(1, 1) + dz * inverse_box(2, 1);
        double nz = dx * inverse_box(0, 2) + dy * inverse_box(1, 2) + dz * inverse_box(2, 2);
        int icel = static_cast<int>(std::floor((std::sqrt(std::pow(nx * box(0, 0), 2) + std::pow(nx * box(0, 1), 2) + std::pow(nx * box(0, 2), 2))) / bin_length));
        int jcel = static_cast<int>(std::floor((std::sqrt(std::pow(ny * box(1, 0), 2) + std::pow(ny * box(1, 1), 2) + std::pow(ny * box(1, 2), 2))) / bin_length));
        int kcel = static_cast<int>(std::floor((std::sqrt(std::pow(nz * box(2, 0), 2) + std::pow(nz * box(2, 1), 2) + std::pow(nz * box(2, 2), 2))) / bin_length));
        if (icel < 0)
            icel = 0;
        else if (icel > im1)
            icel = im1;

        if (jcel < 0)
            jcel = 0;
        else if (jcel > jm1)
            jcel = jm1;

        if (kcel < 0)
            kcel = 0;
        else if (kcel > km1)
            kcel = km1;

        atom_cell_list(i) = cell_id_list(icel, jcel, kcel);
        cell_id_list(icel, jcel, kcel) = i;
        max_neigh_list(icel, jcel, kcel) += 1;
    }
}


PYBIND11_MODULE(_neigh, m)
{
    m.def("build_cell_rec", &build_cell_rec);
    m.def("build_cell_rec_with_jishu", &build_cell_rec_with_jishu);
    m.def("build_cell_tri", &build_cell_tri);
    m.def("build_cell_tri_with_jishu", &build_cell_tri_with_jishu);
}
