// Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.
// Part of code is inspired by the PyNEP project (https://github.com/bigd4/PyNEP).
#include "nep.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <time.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

struct Atom {
  int N;
  std::vector<int> type;
  std::vector<double> box, position, potential, force, virial, descriptor;
};

class NEPCalculator
{
  public:
    NEPCalculator(std::string);
    void setAtoms(py::array, py::array, py::array);
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> calculate(py::array, py::array, py::array);
    py::dict info;
    std::vector<double> get_descriptors(py::array, py::array, py::array);

  private:
    Atom atom;
    NEP3 calc;
    std::string model_file;
    bool HAS_CALCULATED=false;
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
  info["num_node"] = calc.annmb.dim;
  info["num_para"] = calc.annmb.num_para;
  info["element_list"] = calc.element_list;
}

void NEPCalculator::setAtoms(
  py::array _type,
  py::array _box,
  py::array _position)
{
  Atom _atom;
  auto c_pos = _position.unchecked<double, 1>();
  auto c_box = _box.unchecked<double, 1>();
  auto c_type = _type.unchecked<int, 1>();

  _atom.N = c_type.shape(0);

  for (int i=0; i<c_box.shape(0);i++){
    _atom.box.emplace_back(c_box(i));
  }
  for (int i=0; i<_atom.N;i++){
    _atom.type.emplace_back(c_type(i));
  }
  for (int i=0; i<c_pos.shape(0);i++){
    _atom.position.emplace_back(c_pos(i));
  }
  _atom.potential.resize(_atom.N);
  _atom.force.resize(_atom.N * 3);
  _atom.virial.resize(_atom.N * 9);
  _atom.descriptor.resize(_atom.N * calc.annmb.dim);
  atom = _atom;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> NEPCalculator::calculate(
  py::array _type,
  py::array _box,
  py::array _position
)
{
  setAtoms(_type, _box, _position);
  calc.compute(atom.type, atom.box, atom.position, atom.potential, atom.force, atom.virial);
  return std::make_tuple(atom.potential, atom.force, atom.virial);
}

std::vector<double> NEPCalculator::get_descriptors(
  py::array _type,
  py::array _box,
  py::array _position
)
{
  setAtoms(_type, _box, _position);
  calc.find_descriptor(atom.type, atom.box, atom.position, atom.descriptor);
  return atom.descriptor;
}

PYBIND11_MODULE(_nep, m){
    m.doc() = "nep";
    py::class_<NEPCalculator>(m, "NEPCalculator")
		.def(py::init<std::string>())
    .def_readonly("info", &NEPCalculator::info)
		.def("calculate", &NEPCalculator::calculate)
    .def("get_descriptors", &NEPCalculator::get_descriptors)
		;
}