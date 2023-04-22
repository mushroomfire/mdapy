#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

void _build_cell(py::array cpos, py::array catom_cell_list, py::array ccell_id_list, py::array corigin, py::array cncel, double bin_length)
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

PYBIND11_MODULE(_neigh, m)
{
    m.def("_build_cell", &_build_cell);
}
