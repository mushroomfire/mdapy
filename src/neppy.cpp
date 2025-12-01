// Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include "type.h"
#include <nep.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <vector>
#include <stdio.h>
#include <omp.h>

namespace nb = nanobind;

struct Atom
{
    int N;
    std::vector<int> type;
    std::vector<double> box, position, potential, force, virial, descriptor, latentspace;
};

class NEPCalculator
{
public:
    NEPCalculator(std::string);
    void setAtoms(const ROneArrayI type_py,
                  const ROneArrayD x_py,
                  const ROneArrayD y_py,
                  const ROneArrayD z_py,
                  const RTwoArrayD box_py);
    void calculate(const ROneArrayI type_py,
                   const ROneArrayD x_py,
                   const ROneArrayD y_py,
                   const ROneArrayD z_py,
                   const RTwoArrayD box_py,
                   OneArrayD potention_py,
                   TwoArrayD force_py,
                   TwoArrayD virial_py);
    nb::dict info;
    void get_descriptors(const ROneArrayI type_py,
                         const ROneArrayD x_py,
                         const ROneArrayD y_py,
                         const ROneArrayD z_py,
                         const RTwoArrayD box_py,
                         TwoArrayD descriptor_py);

    void get_latentspace(const ROneArrayI type_py,
                         const ROneArrayD x_py,
                         const ROneArrayD y_py,
                         const ROneArrayD z_py,
                         const RTwoArrayD box_py,
                         TwoArrayD latentspace_py);

private:
    Atom atom;
    NEP3 calc;
    std::string model_file;
};

NEPCalculator::NEPCalculator(std::string _model_file)
{
    model_file = _model_file;
    calc = NEP3(model_file);
    info["version"] = calc.paramb.version;
    info["zbl"] = calc.zbl.enabled;
    info["radial_cutoff"] = calc.paramb.rc_radial;
    info["angular_cutoff"] = calc.paramb.rc_angular;
    info["n_max_radial"] = calc.paramb.n_max_radial;
    info["n_max_angular"] = calc.paramb.n_max_angular;
    info["basis_size_radial"] = calc.paramb.basis_size_radial;
    info["basis_size_angular"] = calc.paramb.basis_size_angular;
    info["l_max_3body"] = calc.paramb.L_max;
    info["num_ndim"] = calc.annmb.dim;
    info["num_nlatent"] = calc.annmb.num_neurons1;
    info["num_para"] = calc.annmb.num_para;
    nb::list ele_list;
    for (auto v : calc.element_list)
        ele_list.append(v);
    info["element_list"] = ele_list;
}

void NEPCalculator::setAtoms(
    const ROneArrayI type_py,
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py)
{
    Atom _atom;
    auto type = type_py.view();
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto box = box_py.view();

    _atom.N = type.shape(0);
    // type[num_atoms] should be integers 0, 1, ..., mapping to the atom types in nep.txt in order
    // box[9] is ordered as ax, bx, cx, ay, by, cy, az, bz, cz
    // position[num_atoms * 3] is ordered as x[num_atoms], y[num_atoms], z[num_atoms]
    // potential[num_atoms]
    // force[num_atoms * 3] is ordered as fx[num_atoms], fy[num_atoms], fz[num_atoms]
    // virial[num_atoms * 9] is ordered as v_xx[num_atoms], v_xy[num_atoms], v_xz[num_atoms],
    // v_yx[num_atoms], v_yy[num_atoms], v_yz[num_atoms], v_zx[num_atoms], v_zy[num_atoms],
    // v_zz[num_atoms]
    // descriptor[num_atoms * dim] is ordered as d0[num_atoms], d1[num_atoms], ...

    _atom.box.resize(9);
    _atom.type.resize(_atom.N);
    _atom.position.resize(_atom.N * 3);
    _atom.potential.resize(_atom.N);
    _atom.force.resize(_atom.N * 3);
    _atom.virial.resize(_atom.N * 9);
    _atom.descriptor.resize(_atom.N * calc.annmb.dim);
    _atom.latentspace.resize(_atom.N * calc.annmb.num_neurons1);
    _atom.box[0] = box(0, 0); // ax
    _atom.box[1] = box(1, 0); // bx
    _atom.box[2] = box(2, 0); // cx
    _atom.box[3] = box(0, 1); // ay
    _atom.box[4] = box(1, 1); // by
    _atom.box[5] = box(2, 1); // cy
    _atom.box[6] = box(0, 2); // az
    _atom.box[7] = box(1, 2); // bz
    _atom.box[8] = box(2, 2); // cz

#pragma omp parallel for firstprivate(x, y, z, type)
    for (int i = 0; i < _atom.N; ++i)
    {
        _atom.type[i] = type(i);
        _atom.position[i] = x(i);
        _atom.position[i + _atom.N] = y(i);
        _atom.position[i + _atom.N * 2] = z(i);
    }

    atom = _atom;
}

void NEPCalculator::calculate(
    const ROneArrayI type_py,
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    OneArrayD potention_py,
    TwoArrayD force_py,
    TwoArrayD virial_py)
{
    setAtoms(type_py, x_py, y_py, z_py, box_py);
    calc.compute(atom.type, atom.box, atom.position, atom.potential, atom.force, atom.virial);
    auto potential = potention_py.view();
    auto force = force_py.view();
    auto virial = virial_py.view();
// potential[num_atoms]
// force[num_atoms * 3] is ordered as fx[num_atoms], fy[num_atoms], fz[num_atoms]
// virial[num_atoms * 9] is ordered as v_xx[num_atoms], v_xy[num_atoms], v_xz[num_atoms],
// v_yx[num_atoms], v_yy[num_atoms], v_yz[num_atoms], v_zx[num_atoms], v_zy[num_atoms],
// v_zz[num_atoms]
#pragma omp parallel for
    for (int i = 0; i < atom.N; ++i)
    {
        potential(i) = atom.potential[i];
        force(i, 0) = atom.force[i];
        force(i, 1) = atom.force[i + atom.N];
        force(i, 2) = atom.force[i + atom.N * 2];
        for (int j = 0; j < 9; ++j)
        {
            virial(i, j) = atom.virial[i + atom.N * j];
        }
    }
}

void NEPCalculator::get_descriptors(
    const ROneArrayI type_py,
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    TwoArrayD descriptor_py)
{
    setAtoms(type_py, x_py, y_py, z_py, box_py);
    calc.find_descriptor(atom.type, atom.box, atom.position, atom.descriptor);
    // descriptor[num_atoms * dim] is ordered as d0[num_atoms], d1[num_atoms], ...
    auto descriptor = descriptor_py.view();
#pragma omp parallel for
    for (int i = 0; i < atom.N; ++i)
    {

        for (int j = 0; j < calc.annmb.dim; ++j)
        {
            descriptor(i, j) = atom.descriptor[i + atom.N * j];
        }
    }
}

void NEPCalculator::get_latentspace(
    const ROneArrayI type_py,
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    TwoArrayD latentspace_py)
{
    setAtoms(type_py, x_py, y_py, z_py, box_py);
    calc.find_latent_space(atom.type, atom.box, atom.position, atom.latentspace);
    auto latentspace = latentspace_py.view();

#pragma omp parallel for
    for (int i = 0; i < atom.N; ++i)
    {

        for (int j = 0; j < calc.annmb.num_neurons1; ++j)
        {
            latentspace(i, j) = atom.latentspace[i + atom.N * j];
        }
    }
}

NB_MODULE(_nepcal, m)
{
    nb::class_<NEPCalculator>(m, "NEPCalculator")
        .def(nb::init<std::string>())
        .def_ro("info", &NEPCalculator::info)
        .def("calculate", &NEPCalculator::calculate)
        .def("get_descriptors", &NEPCalculator::get_descriptors)
        .def("get_latentspace", &NEPCalculator::get_latentspace);
}