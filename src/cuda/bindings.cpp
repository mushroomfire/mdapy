// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// mdapy._cuda: the atoms, neighbor lists and potentials of MDAPY on an NVIDIA
// GPU. Built only with the CUDA toolkit (CMake option MDAPY_CUDA).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cuda/cuda.h"

namespace nb = nanobind;

namespace {

using mdapy::cuda::Atoms;
using mdapy::cuda::Box;
using mdapy::cuda::Neighbors;
using mdapy::cuda::Potential;
using mdapy::cuda::Precision;

using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Species = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Vector = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Table = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Flat = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Array = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using Masses = Flat;
using Movable = nb::ndarray<const bool, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::object new_array(std::initializer_list<std::size_t> dims, T*& data) {
    std::size_t size = 1;
    for (const std::size_t d : dims) size *= d;
    data = new T[size];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(data, dims.size(), dims.begin(), owner).cast();
}

Box make_box(Matrix h, Vector lo, std::array<bool, 3> pbc) {
    Box box{};
    for (int k = 0; k < 9; ++k) box.h[k] = h.data()[k];
    for (int k = 0; k < 3; ++k) {
        box.lo[k] = lo.data()[k];
        box.pbc[k] = pbc[static_cast<std::size_t>(k)];
    }
    return box;
}

Precision precision_of(const std::string& name) {
    if (name == "mixed") return Precision::Mixed;
    if (name == "double") return Precision::Double;
    throw std::invalid_argument("precision must be mixed or double, not " + name);
}

}  // namespace

NB_MODULE(_cuda, m) {
    nb::exception<mdapy::cuda::NotFinite>(m, "NotFinite");
    m.def("device_name", &mdapy::cuda::device_name);

    nb::class_<Atoms>(m, "Atoms")
        .def(nb::init<std::int64_t>(), nb::arg("n"))
        .def_prop_ro("size", &Atoms::size)
        .def(
            "set_species",
            [](Atoms& a, Species species) {
                if (static_cast<std::int64_t>(species.shape(0)) != a.size()) {
                    throw std::invalid_argument("one species per atom");
                }
                a.set_species(species.data());
            },
            nb::arg("species"))
        .def(
            "set_positions",
            [](Atoms& a, Positions positions) {
                if (static_cast<std::int64_t>(positions.shape(0)) != a.size()) {
                    throw std::invalid_argument("one position per atom");
                }
                a.set_positions(positions.data());
            },
            nb::arg("positions"))
        .def("positions",
             [](const Atoms& a) {
                 double* data;
                 nb::object out = new_array({static_cast<std::size_t>(a.size()), 3}, data);
                 a.get_positions(data);
                 return out;
             })
        .def("forces",
             [](const Atoms& a) {
                 double* data;
                 nb::object out = new_array({static_cast<std::size_t>(a.size()), 3}, data);
                 a.get_forces(data);
                 return out;
             })
        .def("energies",
             [](const Atoms& a) {
                 double* data;
                 nb::object out = new_array({static_cast<std::size_t>(a.size())}, data);
                 a.get_energies(data);
                 return out;
             })
        .def("virials",
             [](const Atoms& a) {
                 double* data;
                 nb::object out =
                     new_array({static_cast<std::size_t>(a.size()), 3, 3}, data);
                 a.get_virials(data);
                 return out;
             })
        .def(
            "set_masses",
            [](Atoms& a, Masses masses) {
                if (static_cast<std::int64_t>(masses.shape(0)) != a.size()) {
                    throw std::invalid_argument("one mass per atom");
                }
                a.set_masses(masses.data());
            },
            nb::arg("masses"))
        .def(
            "set_velocities",
            [](Atoms& a, Positions velocities) {
                if (static_cast<std::int64_t>(velocities.shape(0)) != a.size()) {
                    throw std::invalid_argument("one velocity per atom");
                }
                a.set_velocities(velocities.data());
            },
            nb::arg("velocities"))
        .def("velocities",
             [](const Atoms& a) {
                 double* data;
                 nb::object out = new_array({static_cast<std::size_t>(a.size()), 3}, data);
                 a.get_velocities(data);
                 return out;
             })
        .def("kick", &Atoms::kick, nb::arg("c"))
        .def("drift", &Atoms::drift, nb::arg("dt"))
        .def("kick_drift", &Atoms::kick_drift, nb::arg("c"), nb::arg("dt"))
        .def("scale", &Atoms::scale, nb::arg("fx"), nb::arg("fy"), nb::arg("fz"))
        .def("shear", &Atoms::shear, nb::arg("half"), nb::arg("yz"), nb::arg("xz"),
             nb::arg("xy"))
        .def(
            "remap",
            [](Atoms& a, Vector old_lo, Matrix old_inverse, Vector lo, Matrix h) {
                a.remap(old_lo.data(), old_inverse.data(), lo.data(), h.data());
            },
            nb::arg("old_lo"), nb::arg("old_inverse"), nb::arg("lo"), nb::arg("h"))
        .def("langevin", &Atoms::langevin, nb::arg("c1"), nb::arg("c2"), nb::arg("kt"),
             nb::arg("seed"), nb::arg("counter"), nb::arg("keep") = false)
        .def("stash_forces", &Atoms::stash_forces)
        .def("mix_forces", &Atoms::mix_forces, nb::arg("weight"), nb::arg("other"))
        .def(
            "set_springs",
            [](Atoms& a, Positions sites, Flat constants) {
                if (static_cast<std::int64_t>(sites.shape(0)) != a.size() ||
                    static_cast<std::int64_t>(constants.shape(0)) != a.size()) {
                    throw std::invalid_argument("one site and one constant per atom");
                }
                a.set_springs(sites.data(), constants.data());
            },
            nb::arg("sites"), nb::arg("constants"))
        .def("clear_springs", [](Atoms& a) { a.set_springs(nullptr, nullptr); })
        .def("springs", [](Atoms& a, Matrix h) { return a.springs(h.data()); }, nb::arg("h"))
        .def("drive", [](Atoms& a, Vector field) { a.drive(field.data()); }, nb::arg("field"))
        .def(
            "reflect",
            [](Atoms& a, Vector normal, double offset, double speed) {
                a.reflect(normal.data(), offset, speed);
            },
            nb::arg("normal"), nb::arg("offset"), nb::arg("speed"))
        .def("heat_flux",
             [](const Atoms& a) {
                 double* data;
                 nb::object out = new_array({3}, data);
                 a.heat_flux(data);
                 return out;
             })
        .def(
            "set_movable",
            [](Atoms& a, Movable movable, Matrix rotation) {
                if (static_cast<std::int64_t>(movable.shape(0)) != a.size()) {
                    throw std::invalid_argument("three directions per atom");
                }
                a.set_movable(movable.data(), rotation.data());
            },
            nb::arg("movable"), nb::arg("rotation"))
        .def("restrict_forces", &Atoms::restrict_forces)
        .def("restrict_velocities", &Atoms::restrict_velocities)
        .def("clear_movable", &Atoms::clear_movable)
        .def(
            "set_prescribed",
            [](Atoms& a, Positions velocities) {
                if (static_cast<std::int64_t>(velocities.shape(0)) != a.size()) {
                    throw std::invalid_argument("one velocity per atom");
                }
                a.set_prescribed(velocities.data());
            },
            nb::arg("velocities"))
        .def("clear_prescribed", [](Atoms& a) { a.set_prescribed(nullptr); })
        .def(
            "set_added",
            [](Atoms& a, Positions forces) {
                if (static_cast<std::int64_t>(forces.shape(0)) != a.size()) {
                    throw std::invalid_argument("one force per atom");
                }
                a.set_added(forces.data());
            },
            nb::arg("forces"))
        .def("clear_added", [](Atoms& a) { a.set_added(nullptr); })
        .def("add_forces", &Atoms::add_forces)
        .def(
            "set_groups",
            [](Atoms& a, Species bits) {
                if (static_cast<std::int64_t>(bits.shape(0)) != a.size()) {
                    throw std::invalid_argument("one group word per atom");
                }
                a.set_groups(bits.data());
            },
            nb::arg("bits"))
        .def("clear_groups", [](Atoms& a) { a.set_groups(nullptr); })
        .def(
            "group_forces",
            [](const Atoms& a, int count) {
                double* data;
                nb::object out = new_array({static_cast<std::size_t>(count), 3}, data);
                a.group_forces(count, data);
                return out;
            },
            nb::arg("count"))
        .def("kinetic_free", &Atoms::kinetic_free)
        .def("bath", &Atoms::bath, nb::arg("bit"), nb::arg("c1"), nb::arg("c2"), nb::arg("kt"),
             nb::arg("seed"), nb::arg("counter"))
        .def("kinetic",
             [](const Atoms& a) {
                 double* data;
                 nb::object out = new_array({3, 3}, data);
                 a.kinetic(data);
                 return out;
             })
        .def("images",
             [](const Atoms& a) {
                 std::int32_t* data;
                 nb::object out = new_array({static_cast<std::size_t>(a.size()), 3}, data);
                 a.get_images(data);
                 return out;
             })
        .def("total_energy", &Atoms::total_energy)
        .def("total_virial", [](const Atoms& a) {
            double* data;
            nb::object out = new_array({3, 3}, data);
            a.total_virial(data);
            return out;
        });

    nb::class_<Neighbors>(m, "Neighbors")
        .def(nb::init<double, double>(), nb::arg("cutoff"), nb::arg("skin"))
        .def(
            "update",
            [](Neighbors& l, Atoms& a, Matrix h, Vector lo, std::array<bool, 3> pbc) {
                const Box box = make_box(h, lo, pbc);
                nb::gil_scoped_release release;
                return l.update(a, box);
            },
            nb::arg("atoms"), nb::arg("h"), nb::arg("lo"), nb::arg("pbc"))
        .def_prop_ro("cutoff", &Neighbors::cutoff)
        .def_prop_ro("skin", &Neighbors::skin)
        .def_prop_ro("builds", &Neighbors::builds)
        .def("get", [](const Neighbors& l, const Atoms& a) {
            const std::int64_t n = a.size();
            std::int32_t* counts;
            std::int32_t* indices;
            std::int32_t* images;
            const std::size_t shape[2] = {static_cast<std::size_t>(n),
                                          static_cast<std::size_t>(l.most())};
            nb::object c = new_array({shape[0]}, counts);
            nb::object i = new_array({shape[0], shape[1]}, indices);
            nb::object g = new_array({shape[0], shape[1]}, images);
            l.get(a, counts, indices, images);
            return nb::make_tuple(c, i, g);
        });

    nb::class_<Potential>(m, "Potential")
        .def_prop_ro("cutoff", &Potential::cutoff)
        .def_prop_ro("species", &Potential::species)
        .def(
            "compute",
            [](Potential& p, Atoms& a, Neighbors& l, Matrix h, Vector lo,
               std::array<bool, 3> pbc, bool energy, bool virial, bool atomic) {
                const Box box = make_box(h, lo, pbc);
                nb::gil_scoped_release release;
                p.compute(a, l, box, energy, virial, atomic);
            },
            nb::arg("atoms"), nb::arg("neighbors"), nb::arg("h"), nb::arg("lo"), nb::arg("pbc"),
            nb::arg("energy"), nb::arg("virial"), nb::arg("atomic"));

    m.def(
        "make_eam",
        [](Table embedding, Table density, Table pair, double drho, double dr, double cutoff,
           double rho_max, const std::string& precision) {
            if (pair.shape(1) != density.shape(1)) {
                throw std::invalid_argument("inconsistent EAM tables");
            }
            const auto potential = mdapy::calculators::make_eam_potential(
                embedding.data(), static_cast<int>(embedding.shape(0)),
                static_cast<int>(embedding.shape(1)), density.data(),
                static_cast<int>(density.shape(0)), pair.data(),
                static_cast<int>(pair.shape(0)), static_cast<int>(density.shape(1)), drho, dr,
                cutoff, rho_max);
            return mdapy::cuda::make_eam(potential, precision_of(precision));
        },
        nb::arg("embedding"), nb::arg("density"), nb::arg("pair"), nb::arg("drho"),
        nb::arg("dr"), nb::arg("cutoff"), nb::arg("rho_max"), nb::arg("precision"));
    m.def(
        "make_adp",
        [](Table embedding, Table density, Table pair, Table dipole, Table quadrupole,
           double drho, double dr, double cutoff, const std::string& precision) {
            const auto elements = static_cast<std::size_t>(embedding.shape(0));
            const std::size_t pairs = elements * (elements + 1) / 2;
            if (density.shape(0) != elements || pair.shape(0) != pairs ||
                dipole.shape(0) != pairs || quadrupole.shape(0) != pairs ||
                pair.shape(1) != density.shape(1) || dipole.shape(1) != density.shape(1) ||
                quadrupole.shape(1) != density.shape(1)) {
                throw std::invalid_argument("inconsistent ADP tables");
            }
            const auto potential = mdapy::calculators::make_adp_potential(
                embedding.data(), static_cast<int>(elements), static_cast<int>(embedding.shape(1)),
                density.data(), pair.data(), dipole.data(), quadrupole.data(),
                static_cast<int>(density.shape(1)), drho, dr, cutoff);
            return mdapy::cuda::make_adp(potential, precision_of(precision));
        },
        nb::arg("embedding"), nb::arg("density"), nb::arg("pair"), nb::arg("dipole"),
        nb::arg("quadrupole"), nb::arg("drho"), nb::arg("dr"), nb::arg("cutoff"),
        nb::arg("precision"));
    m.def(
        "make_stillinger_weber",
        [](int elements, Flat rows, const std::string& precision) {
            const auto e = static_cast<std::size_t>(elements);
            if (elements < 1 || rows.shape(0) != 11 * e * e * e) {
                throw std::invalid_argument("inconsistent Stillinger-Weber tables");
            }
            const auto potential =
                mdapy::calculators::make_stillinger_weber(elements, rows.data());
            return mdapy::cuda::make_stillinger_weber(potential, precision_of(precision));
        },
        nb::arg("elements"), nb::arg("rows"), nb::arg("precision"));
    m.def(
        "make_tersoff",
        [](int elements, Flat rows, const std::string& precision) {
            const auto e = static_cast<std::size_t>(elements);
            if (elements < 1 || rows.shape(0) != 14 * e * e * e) {
                throw std::invalid_argument("inconsistent Tersoff tables");
            }
            const auto potential = mdapy::calculators::make_tersoff(elements, rows.data());
            return mdapy::cuda::make_tersoff(potential, precision_of(precision));
        },
        nb::arg("elements"), nb::arg("rows"), nb::arg("precision"));
    m.def(
        "make_meam",
        [](Table element, Table ebound, Array cmin, Array cmax, Array tables,
           const std::vector<int>& options, const std::vector<double>& numbers,
           const std::string& precision) {
            const auto e = static_cast<std::size_t>(element.shape(1));
            if (element.shape(0) != 17 || ebound.shape(0) != e || ebound.shape(1) != e ||
                cmin.size() != e * e * e || cmax.size() != e * e * e || tables.ndim() != 3 ||
                tables.shape(0) != e * (e + 1) / 2 || tables.shape(1) != 7 ||
                tables.shape(2) < 3 || options.size() != 4 || numbers.size() != 5) {
                throw std::invalid_argument("inconsistent MEAM tables");
            }
            const auto potential = mdapy::calculators::make_meam_potential(
                static_cast<int>(e), element.data(), ebound.data(), cmin.data(), cmax.data(),
                tables.data(), static_cast<int>(tables.shape(2)), options.data(), numbers.data());
            return mdapy::cuda::make_meam(potential, precision_of(precision));
        },
        nb::arg("element"), nb::arg("ebound"), nb::arg("cmin"), nb::arg("cmax"),
        nb::arg("tables"), nb::arg("options"), nb::arg("numbers"), nb::arg("precision"));
    m.def(
        "make_nep",
        [](const std::vector<int>& sizes, Flat rc_radial, Flat rc_angular, Flat w0, Flat b0,
           Flat w1, double b1, Flat c_radial, Flat c_angular, Flat q_scaler, Flat zbl,
           const std::string& precision) {
            if (sizes.size() != 9) {
                throw std::invalid_argument("make_nep needs 9 sizes");
            }
            const auto flat = [](Flat values) {
                return std::vector<double>(values.data(), values.data() + values.shape(0));
            };
            mdapy::calculators::NepModel model;
            model.types = sizes[0];
            model.n_radial = sizes[1];
            model.n_angular = sizes[2];
            model.k_radial = sizes[3];
            model.k_angular = sizes[4];
            model.l_max = sizes[5];
            model.four_body = sizes[6] != 0;
            model.five_body = sizes[7] != 0;
            model.neurons = sizes[8];
            model.rc_radial = flat(rc_radial);
            model.rc_angular = flat(rc_angular);
            model.w0 = flat(w0);
            model.b0 = flat(b0);
            model.w1 = flat(w1);
            model.b1 = b1;
            model.c_radial = flat(c_radial);
            model.c_angular = flat(c_angular);
            model.q_scaler = flat(q_scaler);
            model.zbl = flat(zbl);
            model.prepare();
            return mdapy::cuda::make_nep(model, precision_of(precision));
        },
        nb::arg("sizes"), nb::arg("rc_radial"), nb::arg("rc_angular"), nb::arg("w0"),
        nb::arg("b0"), nb::arg("w1"), nb::arg("b1"), nb::arg("c_radial"), nb::arg("c_angular"),
        nb::arg("q_scaler"), nb::arg("zbl"), nb::arg("precision"));
    m.def(
        "make_pair",
        [](const std::string& style, int species, Flat cutsq, Flat offset, Flat coefficients,
           const std::string& precision) {
            const auto t = static_cast<std::size_t>(species);
            if (species < 1 || cutsq.shape(0) != t * t || offset.shape(0) != t * t ||
                coefficients.shape(0) != 4 * t * t) {
                throw std::invalid_argument("inconsistent pair tables");
            }
            mdapy::calculators::PairStyle kind;
            if (style == "lj/cut") {
                kind = mdapy::calculators::PairStyle::LennardJones;
            } else if (style == "morse") {
                kind = mdapy::calculators::PairStyle::Morse;
            } else if (style == "ufm") {
                kind = mdapy::calculators::PairStyle::UhlenbeckFord;
            } else {
                throw std::invalid_argument("unknown pair style");
            }
            const auto potential = mdapy::calculators::make_pair_potential(
                kind, species, cutsq.data(), offset.data(), coefficients.data());
            return mdapy::cuda::make_pair(potential, precision_of(precision));
        },
        nb::arg("style"), nb::arg("species"), nb::arg("cutsq"), nb::arg("offset"),
        nb::arg("coefficients"), nb::arg("precision"));
}
