// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

void _build_cell_rec_double(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double bin_length)
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

void _build_cell_rec_float(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double d_bin_length)
{
    auto pos = cpos.unchecked<float, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto origin = corigin.unchecked<float, 1>();
    auto ncel = cncel.unchecked<int, 1>();
    int N = pos.shape(0);

    int im1 = ncel(0) - 1;
    int jm1 = ncel(1) - 1;
    int km1 = ncel(2) - 1;
    float xlo = origin(0);
    float ylo = origin(1);
    float zlo = origin(2);

    float bin_length = d_bin_length;

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

void _build_cell_rec(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double bin_length, int flag)
{
    // 0 is double, 1 is single.

    if (flag==0){
        _build_cell_rec_double(cpos, catom_cell_list, ccell_id_list, corigin, cncel, bin_length);
    }
    else if (flag==1){
        _build_cell_rec_float(cpos, catom_cell_list, ccell_id_list, corigin, cncel, bin_length);
    }
}

void _build_cell_rec_with_jishu_double(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double bin_length, py::array cmax_neigh_list)
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

void _build_cell_rec_with_jishu_float(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double d_bin_length, py::array cmax_neigh_list)
{

    auto pos = cpos.unchecked<float, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto origin = corigin.unchecked<float, 1>();
    auto ncel = cncel.unchecked<int, 1>();
    auto max_neigh_list = cmax_neigh_list.mutable_unchecked<int, 3>();

    float bin_length = d_bin_length;
    int N = pos.shape(0);

    int im1 = ncel(0) - 1;
    int jm1 = ncel(1) - 1;
    int km1 = ncel(2) - 1;
    float xlo = origin(0);
    float ylo = origin(1);
    float zlo = origin(2);

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

void _build_cell_rec_with_jishu(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double bin_length, py::array cmax_neigh_list, int flag)
{
    // 0 is double, 1 is single.

    if (flag==0){
        _build_cell_rec_with_jishu_double(cpos, catom_cell_list, ccell_id_list, corigin, cncel, bin_length, cmax_neigh_list);
    }
    else if (flag==1){
        _build_cell_rec_with_jishu_float(cpos, catom_cell_list, ccell_id_list, corigin, cncel, bin_length, cmax_neigh_list);
    }
}

void _build_cell_tri_double(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array cbox, py::array cncel, double bin_length)
{

    auto pos = cpos.unchecked<double, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto ncel = cncel.unchecked<int, 1>();
    auto box = cbox.unchecked<double, 2>();
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
        double nz = dz / box(2, 2);
        double ny = (dy - nz * box(2, 1)) / box(1, 1);
        double nx = (dx - ny * box(1, 0) - nz * box(2, 0)) / box(0, 0);
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

void _build_cell_tri_float(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array cbox, py::array cncel, double d_bin_length)
{

    auto pos = cpos.unchecked<float, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto ncel = cncel.unchecked<int, 1>();
    auto box = cbox.unchecked<float, 2>();
    float bin_length = d_bin_length;
    int N = pos.shape(0);

    int im1 = ncel(0) - 1;
    int jm1 = ncel(1) - 1;
    int km1 = ncel(2) - 1;
    float xlo = box(3, 0);
    float ylo = box(3, 1);
    float zlo = box(3, 2);

    for (int i = 0; i < N; i++)
    {
        float dx = pos(i, 0) - xlo;
        float dy = pos(i, 1) - ylo;
        float dz = pos(i, 2) - zlo;
        float nz = dz / box(2, 2);
        float ny = (dy - nz * box(2, 1)) / box(1, 1);
        float nx = (dx - ny * box(1, 0) - nz * box(2, 0)) / box(0, 0);
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

void _build_cell_tri(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array cbox, py::array cncel, double bin_length, int flag)
{
    // 0 is double, 1 is single.

    if (flag==0){
        _build_cell_tri_double(cpos, catom_cell_list, ccell_id_list, cbox, cncel, bin_length);
    }
    else if (flag==1){
        _build_cell_tri_float(cpos, catom_cell_list, ccell_id_list, cbox, cncel, bin_length);
    }
}

void _build_cell_tri_with_jishu_double(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array cbox, py::array cncel, double bin_length, py::array cmax_neigh_list)
{

    auto pos = cpos.unchecked<double, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto ncel = cncel.unchecked<int, 1>();
    auto max_neigh_list = cmax_neigh_list.mutable_unchecked<int, 3>();
    auto box = cbox.unchecked<double, 2>();
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
        double nz = dz / box(2, 2);
        double ny = (dy - nz * box(2, 1)) / box(1, 1);
        double nx = (dx - ny * box(1, 0) - nz * box(2, 0)) / box(0, 0);
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

void _build_cell_tri_with_jishu_float(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array cbox, py::array cncel, double d_bin_length, py::array cmax_neigh_list)
{

    auto pos = cpos.unchecked<float, 2>();
    auto atom_cell_list = catom_cell_list.mutable_unchecked<int, 1>();
    auto cell_id_list = ccell_id_list.mutable_unchecked<int, 3>();
    auto ncel = cncel.unchecked<int, 1>();
    auto max_neigh_list = cmax_neigh_list.mutable_unchecked<int, 3>();
    auto box = cbox.unchecked<float, 2>();
    float bin_length = d_bin_length;
    int N = pos.shape(0);

    int im1 = ncel(0) - 1;
    int jm1 = ncel(1) - 1;
    int km1 = ncel(2) - 1;
    float xlo = box(3, 0);
    float ylo = box(3, 1);
    float zlo = box(3, 2);

    for (int i = 0; i < N; i++)
    {
        float dx = pos(i, 0) - xlo;
        float dy = pos(i, 1) - ylo;
        float dz = pos(i, 2) - zlo;
        float nz = dz / box(2, 2);
        float ny = (dy - nz * box(2, 1)) / box(1, 1);
        float nx = (dx - ny * box(1, 0) - nz * box(2, 0)) / box(0, 0);
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

void _build_cell_tri_with_jishu(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array cbox, py::array cncel, double bin_length, py::array cmax_neigh_list, int flag)
{
    // 0 is double, 1 is single.

    if (flag==0){
        _build_cell_tri_with_jishu_double(cpos, catom_cell_list, ccell_id_list, cbox, cncel, bin_length, cmax_neigh_list);
    }
    else if (flag==1){
        _build_cell_tri_with_jishu_float(cpos, catom_cell_list, ccell_id_list, cbox, cncel, bin_length, cmax_neigh_list);
    }
}

PYBIND11_MODULE(_neigh, m)
{
    m.def("_build_cell_rec", &_build_cell_rec);
    m.def("_build_cell_rec_with_jishu", &_build_cell_rec_with_jishu);
    m.def("_build_cell_tri", &_build_cell_tri);
    m.def("_build_cell_tri_with_jishu", &_build_cell_tri_with_jishu);
}
