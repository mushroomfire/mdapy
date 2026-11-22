// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// mdapy_dft_cuda12._dft: the plane-wave DFT of MDAPY on the GPU. Built only with
// the CUDA toolkit (CMake option MDAPY_DFT_ONLY). For now its pieces alone,
// numpy arrays in and out, in hartree atomic units.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "dft/core/check.h"
#include "dft/core/device.h"
#include "dft/fft/fft.h"
#include "dft/hamiltonian/hamiltonian.h"
#include "dft/lattice/lattice.h"
#include "dft/scf/potentials.h"
#include "dft/xc/xc.h"

namespace nb = nanobind;
using mdapy::dft::Complex;
using mdapy::dft::DeviceArray;
using mdapy::dft::Int3;
using mdapy::dft::Lattice;
using mdapy::dft::Vec3;

namespace {

using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Index = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Coefficients =
    nb::ndarray<const std::complex<double>, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ComplexBoxes =
    nb::ndarray<const std::complex<double>, nb::ndim<4>, nb::c_contig, nb::device::cpu>;
using RealBoxes = nb::ndarray<const double, nb::ndim<4>, nb::c_contig, nb::device::cpu>;
using Rows = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Grid3 = nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using Kinds = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Spins = nb::ndarray<const double, nb::ndim<4>, nb::c_contig, nb::device::cpu>;

mdapy::dft::Functional functional_of(const std::string& name) {
    if (name == "lda_pw") return mdapy::dft::Functional::LdaPw;
    if (name == "lda_pw_mod") return mdapy::dft::Functional::LdaPwMod;
    if (name == "pbe") return mdapy::dft::Functional::Pbe;
    throw std::invalid_argument("no functional named " + name);
}

template <class T>
nb::ndarray<nb::numpy, T> new_array(std::initializer_list<std::size_t> dims, T*& data) {
    std::size_t size = 1;
    for (const std::size_t d : dims) size *= d;
    data = new T[size];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(data, dims.size(), dims.begin(), owner);
}

Lattice lattice_of(Matrix a) { return Lattice::from_rows(a.data()); }

Int3 grid_of(std::array<int, 3> grid) { return {grid[0], grid[1], grid[2]}; }

std::size_t points_of(const Int3& g) { return static_cast<std::size_t>(g[0]) * g[1] * g[2]; }

const Complex* as_device(const std::complex<double>* p) { return reinterpret_cast<const Complex*>(p); }

}  // namespace

NB_MODULE(_dft, m) {
    m.def("device_name", [] { return std::string(mdapy::dft::device_name()); });
    m.def("device_count", &mdapy::dft::device_count);

    m.def("reciprocal", [](Matrix a) {
        const Lattice lattice = lattice_of(a);
        double* b = nullptr;
        auto out = new_array<double>({3, 3}, b);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) b[3 * i + j] = lattice.b[i][j];
        return nb::make_tuple(out, lattice.volume);
    });
    m.def("good_size", &mdapy::dft::good_size);
    m.def("fft_grid", [](Matrix a, double cutoff) {
        return mdapy::dft::fft_grid(lattice_of(a), cutoff);
    });
    m.def(
        "sphere",
        [](Matrix a, std::array<double, 3> k, double cutoff, std::array<int, 3> grid, bool real) {
            const auto sphere =
                mdapy::dft::make_sphere(lattice_of(a), Vec3{k[0], k[1], k[2]}, cutoff, grid_of(grid), real);
            const std::size_t n = sphere.size();
            std::int32_t* miller = nullptr;
            double* kinetic = nullptr;
            std::int64_t *box = nullptr, *partner = nullptr;
            auto m_out = new_array<std::int32_t>({n, 3}, miller);
            auto k_out = new_array<double>({n}, kinetic);
            auto b_out = new_array<std::int64_t>({n}, box);
            auto p_out = new_array<std::int64_t>({n}, partner);
            for (std::size_t g = 0; g < n; ++g) {
                for (int i = 0; i < 3; ++i) miller[3 * g + i] = sphere.miller[g][i];
                kinetic[g] = sphere.kinetic[g];
                box[g] = sphere.box[g];
                partner[g] = sphere.partner[g];
            }
            return nb::make_tuple(m_out, k_out, b_out, p_out);
        },
        nb::arg("a"), nb::arg("k"), nb::arg("cutoff"), nb::arg("grid"), nb::arg("half") = false);
    m.def(
        "monkhorst_pack",
        [](std::array<int, 3> n, std::array<double, 3> offset, bool time_reversal) {
            const auto points = mdapy::dft::monkhorst_pack(grid_of(n), Vec3{offset[0], offset[1], offset[2]},
                                                           time_reversal);
            const std::size_t count = points.k.size();
            double *k = nullptr, *w = nullptr;
            auto k_out = new_array<double>({count, 3}, k);
            auto w_out = new_array<double>({count}, w);
            for (std::size_t i = 0; i < count; ++i) {
                for (int j = 0; j < 3; ++j) k[3 * i + j] = points.k[i][j];
                w[i] = points.weight[i];
            }
            return nb::make_tuple(k_out, w_out);
        },
        nb::arg("n"), nb::arg("offset"), nb::arg("time_reversal") = true);
    m.def("grid_of_spacing", [](Matrix a, double spacing) {
        return mdapy::dft::grid_of_spacing(lattice_of(a), spacing);
    });

    // the bands of coefficients (count, n) of a sphere to their values on the
    // grid (count, n1, n2, n3), on the GPU
    m.def("to_real_space", [](Coefficients c, Index box, std::array<int, 3> grid) {
        const Int3 g = grid_of(grid);
        const std::size_t points = points_of(g);
        const int count = static_cast<int>(c.shape(0));
        const std::size_t n = c.shape(1);
        if (box.shape(0) != n) throw std::invalid_argument("one place in the box per plane wave");
        mdapy::dft::Fft fft(g);
        DeviceArray<Complex> coefficients, boxes(count * points);
        coefficients.upload(as_device(c.data()), c.size());
        DeviceArray<std::int64_t> places;
        places.upload(box.data(), n);
        mdapy::dft::scatter(coefficients.data(), n, places.data(), nullptr, count, boxes.data(), points);
        fft.to_real(boxes.data(), count);
        std::complex<double>* out = nullptr;
        auto values = new_array<std::complex<double>>(
            {static_cast<std::size_t>(count), static_cast<std::size_t>(g[0]),
             static_cast<std::size_t>(g[1]), static_cast<std::size_t>(g[2])},
            out);
        boxes.download(reinterpret_cast<Complex*>(out), boxes.size());
        return values;
    });
    // the values (count, n1, n2, n3) on the grid to the coefficients of a sphere,
    // divided by the points of the grid: the inverse of to_real_space
    m.def("to_coefficients", [](ComplexBoxes v, Index box) {
        const Int3 g{static_cast<int>(v.shape(1)), static_cast<int>(v.shape(2)),
                     static_cast<int>(v.shape(3))};
        const std::size_t points = points_of(g);
        const int count = static_cast<int>(v.shape(0));
        const std::size_t n = box.shape(0);
        mdapy::dft::Fft fft(g);
        DeviceArray<Complex> boxes, coefficients(count * n);
        boxes.upload(as_device(v.data()), v.size());
        DeviceArray<std::int64_t> places;
        places.upload(box.data(), n);
        fft.to_reciprocal(boxes.data(), count);
        mdapy::dft::gather(boxes.data(), points, places.data(), n, count,
                           1.0 / static_cast<double>(points), coefficients.data());
        std::complex<double>* out = nullptr;
        auto c = new_array<std::complex<double>>({static_cast<std::size_t>(count), n}, out);
        coefficients.download(reinterpret_cast<Complex*>(out), coefficients.size());
        return c;
    });
    // real wavefunctions of Gamma: the coefficients of a half sphere to real
    // values on the grid, and back
    m.def("to_real_space_gamma",
          [](Coefficients c, Index box, Index partner, std::array<int, 3> grid) {
              const Int3 g = grid_of(grid);
              mdapy::dft::Fft fft(g);
              const std::size_t spectrum = fft.spectrum(), points = fft.points();
              const int count = static_cast<int>(c.shape(0));
              const std::size_t n = c.shape(1);
              DeviceArray<Complex> coefficients, spec(count * spectrum);
              coefficients.upload(as_device(c.data()), c.size());
              DeviceArray<std::int64_t> places, partners;
              places.upload(box.data(), n);
              partners.upload(partner.data(), n);
              mdapy::dft::scatter(coefficients.data(), n, places.data(), partners.data(), count,
                                  spec.data(), spectrum);
              DeviceArray<double> values(count * points);
              fft.to_real(spec.data(), values.data(), count);
              double* out = nullptr;
              auto result = new_array<double>(
                  {static_cast<std::size_t>(count), static_cast<std::size_t>(g[0]),
                   static_cast<std::size_t>(g[1]), static_cast<std::size_t>(g[2])},
                  out);
              values.download(out, values.size());
              return result;
          });
    m.def("to_coefficients_gamma", [](RealBoxes v, Index box) {
        const Int3 g{static_cast<int>(v.shape(1)), static_cast<int>(v.shape(2)),
                     static_cast<int>(v.shape(3))};
        mdapy::dft::Fft fft(g);
        const std::size_t spectrum = fft.spectrum(), points = fft.points();
        const int count = static_cast<int>(v.shape(0));
        const std::size_t n = box.shape(0);
        DeviceArray<double> values;
        values.upload(v.data(), v.size());
        DeviceArray<Complex> spec(count * spectrum), coefficients(count * n);
        fft.to_reciprocal(values.data(), spec.data(), count);
        DeviceArray<std::int64_t> places;
        places.upload(box.data(), n);
        mdapy::dft::gather(spec.data(), spectrum, places.data(), n, count,
                           1.0 / static_cast<double>(points), coefficients.data());
        std::complex<double>* out = nullptr;
        auto c = new_array<std::complex<double>>({static_cast<std::size_t>(count), n}, out);
        coefficients.download(reinterpret_cast<Complex*>(out), coefficients.size());
        return c;
    });
    // functions on a coarse grid carried to a fine one through their Fourier
    // components, (count, n1, n2, n3) complex values in and out
    m.def("to_fine_grid", [](ComplexBoxes v, std::array<int, 3> fine) {
        const Int3 coarse{static_cast<int>(v.shape(1)), static_cast<int>(v.shape(2)),
                          static_cast<int>(v.shape(3))};
        const Int3 f = grid_of(fine);
        const int count = static_cast<int>(v.shape(0));
        mdapy::dft::Fft small(coarse), large(f);
        mdapy::dft::Transfer transfer(coarse, f);
        DeviceArray<Complex> a, b(count * large.points());
        a.upload(as_device(v.data()), v.size());
        small.to_reciprocal(a.data(), count);
        transfer.to_fine(a.data(), b.data(), count);
        large.to_real(b.data(), count);
        // the round trip of the coarse transform multiplied by its points
        std::complex<double>* out = nullptr;
        auto result = new_array<std::complex<double>>(
            {static_cast<std::size_t>(count), static_cast<std::size_t>(f[0]),
             static_cast<std::size_t>(f[1]), static_cast<std::size_t>(f[2])},
            out);
        b.download(reinterpret_cast<Complex*>(out), b.size());
        const double scale = 1.0 / static_cast<double>(small.points());
        for (std::size_t i = 0; i < b.size(); ++i) out[i] *= scale;
        return result;
    });

    // the exchange-correlation functional at points: rho (n, 2) the spin
    // densities, sigma (n, 3) the products of their gradients; the energy per
    // volume, its derivatives by the densities (n, 2) and by sigma (n, 3)
    m.def(
        "xc",
        [](const std::string& name, Rows rho, Rows sigma, double threshold) {
            const mdapy::dft::Functional functional = functional_of(name);
            const std::size_t n = rho.shape(0);
            if (rho.shape(1) != 2 || sigma.shape(0) != n || sigma.shape(1) != 3)
                throw std::invalid_argument("rho is (n, 2) and sigma (n, 3)");
            DeviceArray<double> r, s, f(n), v(2 * n), vs(3 * n);
            r.upload(rho.data(), 2 * n);
            s.upload(sigma.data(), 3 * n);
            mdapy::dft::evaluate_xc(functional, n, r.data(), s.data(), threshold, f.data(),
                                    v.data(), vs.data());
            double *fo = nullptr, *vo = nullptr, *so = nullptr;
            auto f_out = new_array<double>({n}, fo);
            auto v_out = new_array<double>({n, 2}, vo);
            auto s_out = new_array<double>({n, 3}, so);
            f.download(fo, n);
            v.download(vo, 2 * n);
            vs.download(so, 3 * n);
            return nb::make_tuple(f_out, v_out, s_out);
        },
        nb::arg("name"), nb::arg("rho"), nb::arg("sigma"), nb::arg("threshold") = 1e-10);

    // the Hamiltonian of one k-point, for the tests: tables (list of (P, nq)
    // arrays), angular (list of lists), dij (list of (P, P)), the atoms by their
    // fractional positions (N, 3) and species (N,)
    nb::class_<mdapy::dft::Hamiltonian>(m, "Hamiltonian")
        .def("__init__",
             [](mdapy::dft::Hamiltonian* self, Matrix a, std::array<int, 3> grid, double cutoff,
                nb::list tables, nb::list angular, nb::list dij, double dq, Rows positions,
                Kinds species) {
                 std::vector<mdapy::dft::SpeciesProjectors> kinds;
                 for (std::size_t s = 0; s < nb::len(tables); ++s) {
                     auto t = nb::cast<Rows>(tables[s]);
                     auto d = nb::cast<Rows>(dij[s]);
                     mdapy::dft::SpeciesProjectors sp;
                     sp.dq = dq;
                     sp.points = static_cast<int>(t.shape(1));
                     sp.tables.assign(t.data(), t.data() + t.size());
                     sp.angular = nb::cast<std::vector<int>>(angular[s]);
                     sp.dij.assign(d.data(), d.data() + d.size());
                     kinds.push_back(std::move(sp));
                 }
                 if (positions.shape(1) != 3 || species.shape(0) != positions.shape(0))
                     throw std::invalid_argument("positions (N, 3) and species (N,)");
                 std::vector<mdapy::dft::Atom> atoms;
                 for (std::size_t i = 0; i < species.shape(0); ++i)
                     atoms.push_back({Vec3{positions.data()[3 * i], positions.data()[3 * i + 1],
                                           positions.data()[3 * i + 2]},
                                      species.data()[i]});
                 new (self) mdapy::dft::Hamiltonian(lattice_of(a), grid_of(grid), cutoff,
                                                    std::move(kinds), std::move(atoms));
             })
        .def("set_k", [](mdapy::dft::Hamiltonian& h, std::array<double, 3> k) {
            h.set_k(Vec3{k[0], k[1], k[2]});
        })
        .def("set_potential", [](mdapy::dft::Hamiltonian& h, Grid3 v) {
            DeviceArray<double> d;
            d.upload(v.data(), v.size());
            h.set_potential(d.data());
        })
        .def("apply", [](mdapy::dft::Hamiltonian& h, Coefficients psi) {
            const int count = static_cast<int>(psi.shape(0));
            const std::size_t n = psi.shape(1);
            if (n != h.plane_waves()) throw std::invalid_argument("psi is (bands, plane waves)");
            DeviceArray<Complex> a, b(static_cast<std::size_t>(count) * n);
            a.upload(as_device(psi.data()), psi.size());
            h.apply(a.data(), b.data(), count);
            std::complex<double>* out = nullptr;
            auto result = new_array<std::complex<double>>({static_cast<std::size_t>(count), n}, out);
            b.download(reinterpret_cast<Complex*>(out), b.size());
            return result;
        })
        .def_prop_ro("miller", [](const mdapy::dft::Hamiltonian& h) {
            const auto& sphere = h.sphere();
            std::int32_t* out = nullptr;
            auto result = new_array<std::int32_t>({sphere.size(), 3}, out);
            for (std::size_t g = 0; g < sphere.size(); ++g)
                for (int i = 0; i < 3; ++i) out[3 * g + i] = sphere.miller[g][static_cast<std::size_t>(i)];
            return nb::object(result.cast());
        })
        .def_prop_ro("beta", [](const mdapy::dft::Hamiltonian& h) {
            std::complex<double>* out = nullptr;
            auto result = new_array<std::complex<double>>(
                {static_cast<std::size_t>(h.projectors()), h.plane_waves()}, out);
            h.beta().download(reinterpret_cast<Complex*>(out), h.beta().size());
            return nb::object(result.cast());
        })
        .def_prop_ro("projectors", &mdapy::dft::Hamiltonian::projectors);

    // the Hartree and exchange-correlation potentials of densities rho (spins,
    // n1, n2, n3), with a core charge (n1, n2, n3) or None: V_H + V_xc of each
    // spin and the energies (E_H, E_xc, sum of int V rho)
    m.def(
        "potentials",
        [](Matrix a, double cutoff, const std::string& name, Spins rho, nb::object core,
           double threshold) {
            const Int3 g{static_cast<int>(rho.shape(1)), static_cast<int>(rho.shape(2)),
                         static_cast<int>(rho.shape(3))};
            const int spins = static_cast<int>(rho.shape(0));
            mdapy::dft::Potentials pot(lattice_of(a), g, cutoff, functional_of(name), spins, threshold);
            if (!core.is_none()) pot.set_core(nb::cast<Grid3>(core).data());
            DeviceArray<double> r, hxc(rho.size());
            r.upload(rho.data(), rho.size());
            const auto e = pot.compute(r.data(), hxc.data());
            double* out = nullptr;
            auto result = new_array<double>({static_cast<std::size_t>(spins), static_cast<std::size_t>(g[0]),
                                             static_cast<std::size_t>(g[1]), static_cast<std::size_t>(g[2])},
                                            out);
            hxc.download(out, hxc.size());
            return nb::make_tuple(result, nb::make_tuple(e.hartree, e.xc, e.hxc_density));
        },
        nb::arg("a"), nb::arg("cutoff"), nb::arg("functional"), nb::arg("rho"),
        nb::arg("core") = nb::none(), nb::arg("threshold") = 1e-10);
}
