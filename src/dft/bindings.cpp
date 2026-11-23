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

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#include "dft/core/check.h"
#include "dft/core/device.h"
#include "dft/fft/fft.h"
#include "dft/hamiltonian/hamiltonian.h"
#include "dft/lattice/lattice.h"
#include "dft/scf/augmentation.h"
#include "dft/scf/occupations.h"
#include "dft/scf/potentials.h"
#include "dft/scf/ewald.h"
#include "dft/scf/scf.h"
#include "dft/scf/structure.h"
#include "dft/solver/davidson.h"
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
    if (name == "pbe_pw") return mdapy::dft::Functional::PbePw;
    if (name == "pbesol") return mdapy::dft::Functional::PbeSol;
    if (name == "pbesol_pw") return mdapy::dft::Functional::PbeSolPw;
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

// orbitals, when given: (tables (orbitals, points), angular) of each species
std::vector<mdapy::dft::SpeciesProjectors> species_of(nb::list tables, nb::list angular, nb::list dij, double dq,
                                                     nb::object orbitals = nb::none()) {
    if (nb::len(angular) != nb::len(tables) || nb::len(dij) != nb::len(tables))
        throw std::invalid_argument("tables, angular and dij of every species");
    nb::list chi;
    if (!orbitals.is_none()) {
        chi = nb::cast<nb::list>(orbitals);
        if (nb::len(chi) != nb::len(tables)) throw std::invalid_argument("the orbitals of every species");
    }
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
        if (!orbitals.is_none()) {
            nb::tuple o = nb::cast<nb::tuple>(chi[s]);
            auto ot = nb::cast<Rows>(o[0]);
            if (ot.shape(0) > 0 && ot.shape(1) != t.shape(1))
                throw std::invalid_argument("the orbitals take the points of the projectors");
            sp.orbital_tables.assign(ot.data(), ot.data() + ot.size());
            sp.orbital_angular = nb::cast<std::vector<int>>(o[1]);
        }
        kinds.push_back(std::move(sp));
    }
    return kinds;
}

using Coefficients1 = nb::ndarray<const std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

mdapy::dft::SpeciesAugmentation augmentation_of(nb::handle item) {
    mdapy::dft::SpeciesAugmentation sp;
    if (item.is_none()) return sp;
    nb::dict d = nb::cast<nb::dict>(item);
    sp.functions = nb::cast<int>(d["functions"]);
    sp.pair_start = nb::cast<std::vector<int>>(d["pair_start"]);
    sp.term_lm = nb::cast<std::vector<int>>(d["term_lm"]);
    sp.term_l = nb::cast<std::vector<int>>(d["term_l"]);
    sp.term_table = nb::cast<std::vector<int>>(d["term_table"]);
    sp.term_coef = nb::cast<std::vector<double>>(d["term_coef"]);
    sp.lmax = nb::cast<int>(d["lmax"]);
    sp.points = nb::cast<int>(d["points"]);
    auto t = nb::cast<Rows>(d["tables"]);
    sp.tables.assign(t.data(), t.data() + t.size());
    return sp;
}

std::vector<double> doubles_of(nb::handle h) {
    auto a = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(h);
    return std::vector<double>(a.data(), a.data() + a.size());
}

// the one-center data of a PAW species (paw.onecenter), or none
mdapy::dft::SpeciesOneCenter paw_of(nb::handle item) {
    mdapy::dft::SpeciesOneCenter sp;
    if (item.is_none()) return sp;
    nb::dict d = nb::cast<nb::dict>(item);
    sp.mesh = nb::cast<int>(d["mesh"]);
    sp.kkbeta = nb::cast<int>(d["kkbeta"]);
    sp.lmax_rho = nb::cast<int>(d["lmax_rho"]);
    sp.l2 = nb::cast<int>(d["l2"]);
    sp.lh = nb::cast<int>(d["lh"]);
    sp.functions = nb::cast<int>(d["functions"]);
    sp.dx = nb::cast<double>(d["dx"]);
    sp.r = doubles_of(d["r"]);
    sp.whole = doubles_of(d["whole"]);
    sp.inner = doubles_of(d["inner"]);
    sp.ww = doubles_of(d["ww"]);
    sp.sin_theta = doubles_of(d["sin"]);
    sp.cos_theta = doubles_of(d["cos"]);
    sp.ylm = doubles_of(d["ylm"]);
    sp.dylmt = doubles_of(d["dylmt"]);
    sp.dylmp = doubles_of(d["dylmp"]);
    sp.ae = doubles_of(d["ae"]);
    sp.ps = doubles_of(d["ps"]);
    sp.coef = doubles_of(d["coef"]);
    sp.pair = nb::cast<std::vector<int>>(d["pair"]);
    sp.ae_core = doubles_of(d["ae_core"]);
    sp.ps_core = doubles_of(d["ps_core"]);
    sp.nx = static_cast<int>(sp.ww.size());
    sp.radial = sp.mesh > 0 ? static_cast<int>(sp.ae.size() / static_cast<std::size_t>(sp.mesh)) : 0;
    return sp;
}

std::vector<std::vector<double>> spins_of(Rows values) {
    std::vector<std::vector<double>> out;
    const std::size_t n = values.shape(1);
    for (std::size_t s = 0; s < values.shape(0); ++s) out.emplace_back(values.data() + s * n, values.data() + (s + 1) * n);
    return out;
}

std::vector<mdapy::dft::Atom> atoms_of(Rows positions, Kinds species) {
    if (positions.shape(1) != 3 || species.shape(0) != positions.shape(0))
        throw std::invalid_argument("positions (N, 3) and species (N,)");
    std::vector<mdapy::dft::Atom> atoms;
    for (std::size_t i = 0; i < species.shape(0); ++i)
        atoms.push_back({Vec3{positions.data()[3 * i], positions.data()[3 * i + 1], positions.data()[3 * i + 2]},
                         species.data()[i]});
    return atoms;
}

template <class T>
nb::object array_of(const std::vector<T>& values, std::initializer_list<std::size_t> dims) {
    T* out = nullptr;
    auto result = new_array<T>(dims, out);
    std::copy(values.begin(), values.end(), out);
    return nb::object(result.cast());
}

}  // namespace

NB_MODULE(_dft, m) {
    m.def("device_name", [] { return std::string(mdapy::dft::device_name()); });
    m.def("device_count", &mdapy::dft::device_count);
    m.def("set_device", [](int index) {
        if (index < 0 || index >= mdapy::dft::device_count())
            throw std::invalid_argument("no GPU " + std::to_string(index) + " of the " +
                                        std::to_string(mdapy::dft::device_count()) + " this process sees");
        mdapy::dft::check_cuda(cudaSetDevice(index), "cudaSetDevice");
    });

    m.def("reciprocal", [](Matrix a) {
        const Lattice lattice = lattice_of(a);
        double* b = nullptr;
        auto out = new_array<double>({3, 3}, b);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) b[3 * i + j] = lattice.b[i][j];
        return nb::make_tuple(out, lattice.volume);
    });
    m.def("good_size", &mdapy::dft::good_size);
    // the Ewald sum of point charges (atoms,) at Cartesian positions (atoms, 3)
    // in the cell of rows a, bohr: (energy, forces (atoms, 3), dE/d epsilon
    // (3, 3)), hartree
    m.def("ewald", [](Matrix a, Rows positions, nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> z) {
        if (positions.shape(1) != 3 || positions.shape(0) != z.shape(0))
            throw std::invalid_argument("positions (atoms, 3) and a charge for each");
        const auto r = mdapy::dft::ewald(a.data(), std::vector<double>(positions.data(), positions.data() + positions.size()),
                                         std::vector<double>(z.data(), z.data() + z.size()));
        const std::size_t n = z.shape(0);
        return nb::make_tuple(r.energy, array_of(r.forces, {n, 3}),
                              array_of(std::vector<double>(r.strain.begin(), r.strain.end()), {3, 3}));
    });
    // sum_i w_i f_{kind_i}(G) exp(-2 pi i m.tau_i) on the Miller indices m (n, 3)
    // of fractional positions tau (atoms, 3), factors (species, n): (n,) complex
    m.def("structure_sum", [](nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> miller,
                              Rows positions, Kinds kinds, nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> w,
                              Rows factors) {
        const std::size_t n = miller.shape(0), atoms = kinds.shape(0);
        if (miller.shape(1) != 3 || factors.shape(1) != n || positions.shape(0) != atoms || positions.shape(1) != 3 ||
            w.shape(0) != atoms)
            throw std::invalid_argument("miller (n, 3), factors (species, n), positions (atoms, 3), kinds and weights");
        for (std::size_t i = 0; i < atoms; ++i)
            if (kinds.data()[i] < 0 || static_cast<std::size_t>(kinds.data()[i]) >= factors.shape(0))
                throw std::invalid_argument("a kind without its factors");
        DeviceArray<int> places;
        places.upload(miller.data(), miller.size());
        DeviceArray<double> f;
        f.upload(factors.data(), factors.size());
        DeviceArray<Complex> out(n);
        mdapy::dft::structure_sum(places.data(), n, std::vector<double>(positions.data(), positions.data() + positions.size()),
                                  std::vector<int>(kinds.data(), kinds.data() + atoms),
                                  std::vector<double>(w.data(), w.data() + atoms), f.data(), out.data());
        std::complex<double>* host = nullptr;
        auto result = new_array<std::complex<double>>({n}, host);
        out.download(reinterpret_cast<Complex*>(host), n);
        return result;
    });
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
        [](const std::string& name, Rows rho, Rows sigma, double threshold, bool polarized, bool signed_gradient) {
            const mdapy::dft::Functional functional = functional_of(name);
            const std::size_t n = rho.shape(0);
            if (rho.shape(1) != 2 || sigma.shape(0) != n || sigma.shape(1) != 3)
                throw std::invalid_argument("rho is (n, 2) and sigma (n, 3)");
            DeviceArray<double> r, s, f(n), v(2 * n), vs(3 * n);
            r.upload(rho.data(), 2 * n);
            s.upload(sigma.data(), 3 * n);
            mdapy::dft::evaluate_xc(functional, n, r.data(), s.data(), threshold, f.data(),
                                    v.data(), vs.data(), polarized, signed_gradient);
            double *fo = nullptr, *vo = nullptr, *so = nullptr;
            auto f_out = new_array<double>({n}, fo);
            auto v_out = new_array<double>({n, 2}, vo);
            auto s_out = new_array<double>({n, 3}, so);
            f.download(fo, n);
            v.download(vo, 2 * n);
            vs.download(so, 3 * n);
            return nb::make_tuple(f_out, v_out, s_out);
        },
        nb::arg("name"), nb::arg("rho"), nb::arg("sigma"), nb::arg("threshold") = 1e-10,
        nb::arg("polarized") = true, nb::arg("signed_gradient") = true);

    // the Hamiltonian of one k-point, for the tests: tables (list of (P, nq)
    // arrays), angular (list of lists), dij (list of (P, P)), the atoms by their
    // fractional positions (N, 3) and species (N,)
    nb::class_<mdapy::dft::Hamiltonian>(m, "Hamiltonian")
        .def("__init__",
             [](mdapy::dft::Hamiltonian* self, Matrix a, std::array<int, 3> grid, double cutoff,
                nb::list tables, nb::list angular, nb::list dij, double dq, Rows positions,
                Kinds species, bool gamma) {
                 new (self) mdapy::dft::Hamiltonian(lattice_of(a), grid_of(grid), cutoff,
                                                    species_of(tables, angular, dij, dq),
                                                    atoms_of(positions, species));
                 self->set_gamma(gamma);
             },
             nb::arg("a"), nb::arg("grid"), nb::arg("cutoff"), nb::arg("tables"), nb::arg("angular"), nb::arg("dij"),
             nb::arg("dq"), nb::arg("positions"), nb::arg("species"), nb::arg("gamma") = false)
        .def_prop_ro("gamma", &mdapy::dft::Hamiltonian::gamma)
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
            h.beta().download(reinterpret_cast<Complex*>(out),
                              static_cast<std::size_t>(h.projectors()) * h.plane_waves());
            return nb::object(result.cast());
        })
        .def("set_strengths", [](mdapy::dft::Hamiltonian& h, nb::ndarray<const double, nb::c_contig, nb::device::cpu> d) {
            h.set_strengths(std::vector<double>(d.data(), d.data() + d.size()));
        })
        .def("set_overlaps", [](mdapy::dft::Hamiltonian& h, nb::ndarray<const double, nb::c_contig, nb::device::cpu> q) {
            h.set_overlaps(std::vector<double>(q.data(), q.data() + q.size()));
        })
        .def("apply_overlap", [](mdapy::dft::Hamiltonian& h, Coefficients psi) {
            const int count = static_cast<int>(psi.shape(0));
            const std::size_t n = psi.shape(1);
            if (n != h.plane_waves()) throw std::invalid_argument("psi is (bands, plane waves)");
            DeviceArray<Complex> a, b(static_cast<std::size_t>(count) * n);
            a.upload(as_device(psi.data()), psi.size());
            h.apply_overlap(a.data(), b.data(), count);
            std::complex<double>* out = nullptr;
            auto result = new_array<std::complex<double>>({static_cast<std::size_t>(count), n}, out);
            b.download(reinterpret_cast<Complex*>(out), b.size());
            return nb::object(result.cast());
        })
        .def("becsum", [](mdapy::dft::Hamiltonian& h, Coefficients psi,
                          nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> w, std::size_t size) {
            DeviceArray<Complex> a;
            a.upload(as_device(psi.data()), psi.size());
            std::vector<double> out(size, 0.0);
            h.add_becsum(a.data(), static_cast<int>(psi.shape(0)), w.data(), out);
            return array_of(out, {size});
        })
        .def_prop_ro("projectors", &mdapy::dft::Hamiltonian::projectors)
        .def_prop_ro("diagonal", [](const mdapy::dft::Hamiltonian& h) {
            DeviceArray<double> d(h.plane_waves());
            h.diagonal(d.data());
            double* out = nullptr;
            auto result = new_array<double>({h.plane_waves()}, out);
            d.download(out, d.size());
            return nb::object(result.cast());
        });

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

    // the lowest eigenpairs of a Hamiltonian by block Davidson from the guess
    // psi (bands, plane waves): eigenvalues, eigenvectors (bands, plane waves),
    // iterations, converged bands and the largest residual norm
    m.def(
        "davidson",
        [](mdapy::dft::Hamiltonian& h, Coefficients psi, double tolerance, int tight, double loose_tolerance,
           int max_iterations, int largest_basis) {
            const int count = static_cast<int>(psi.shape(0));
            const std::size_t n = psi.shape(1);
            if (n != h.plane_waves()) throw std::invalid_argument("psi is (bands, plane waves)");
            DeviceArray<Complex> a;
            a.upload(as_device(psi.data()), psi.size());
            mdapy::dft::Davidson solver;
            std::vector<double> e;
            const auto r = solver.solve(h, a.data(), count, e, tolerance, tight < 0 ? count : tight,
                                        loose_tolerance, max_iterations,
                                        largest_basis > 0 ? largest_basis : 4 * count);
            double* values = nullptr;
            auto eigenvalues = new_array<double>({static_cast<std::size_t>(count)}, values);
            std::copy(e.begin(), e.end(), values);
            std::complex<double>* out = nullptr;
            auto vectors = new_array<std::complex<double>>({static_cast<std::size_t>(count), n}, out);
            a.download(reinterpret_cast<Complex*>(out), psi.size());
            return nb::make_tuple(eigenvalues, vectors, r.iterations, r.converged, r.residual);
        },
        nb::arg("h"), nb::arg("psi"), nb::arg("tolerance") = 1e-8, nb::arg("tight") = -1,
        nb::arg("loose_tolerance") = 1e-4, nb::arg("max_iterations") = 100, nb::arg("largest_basis") = 0);

    // the occupations of eigenvalues (k-points, bands): f, the Fermi level,
    // -TS and the lowest empty level
    m.def(
        "occupy",
        [](Rows eigenvalues, nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> weights,
           double electrons, double f_max, const std::string& smearing, double sigma) {
            const std::size_t nk = eigenvalues.shape(0), nb_ = eigenvalues.shape(1);
            if (weights.shape(0) != nk) throw std::invalid_argument("a weight for every k-point");
            const auto occ = mdapy::dft::occupy(
                std::vector<double>(eigenvalues.data(), eigenvalues.data() + eigenvalues.size()),
                std::vector<double>(weights.data(), weights.data() + nk), static_cast<int>(nb_), electrons, f_max,
                mdapy::dft::smearing_of(smearing), sigma);
            return nb::make_tuple(array_of(occ.f, {nk, nb_}), occ.fermi, occ.entropy, occ.lowest_empty);
        },
        nb::arg("eigenvalues"), nb::arg("weights"), nb::arg("electrons"), nb::arg("f_max") = 2.0,
        nb::arg("smearing") = "gaussian", nb::arg("sigma") = 0.0);
    m.def("smeared", [](const std::string& smearing, double u) {
        const auto s = mdapy::dft::smearing_of(smearing);
        return nb::make_tuple(mdapy::dft::smeared_occupation(s, u), mdapy::dft::smeared_entropy(s, u));
    });

    // the self-consistent field: the cell, the grid and the cutoffs (hartree),
    // the projectors of the species, the atoms, the k-points (K, 3) and
    // weights, the form factors (species, n) of V_loc and rho_core (or None) on
    // the sphere of the density, the first density on the grid, the Ewald
    // energy, and the options; a dict of the results
    m.def(
        "scf",
        [](Matrix a, std::array<int, 3> grid, double cutoff, double density_cutoff, nb::list tables,
           nb::list angular, nb::list dij, double dq, Rows positions, Kinds species, Rows kpoints,
           nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> weights, Rows local,
           nb::object core, nb::object local_slopes, nb::object core_slopes, nb::object atom_factors,
           nb::ndarray<const double, nb::c_contig, nb::device::cpu> density, double ewald,
           int spins, int bands, double electrons, const std::string& functional, const std::string& smearing,
           double sigma, double beta, double kerker, double minimum, double magnetic_beta, int history,
           double metric_weight, double energy_tolerance, double density_tolerance, int max_iterations,
           int davidson_iterations, int davidson_basis, unsigned long long seed, nb::object basis,
           nb::object state, double magnetic_kerker, nb::object augmentation, nb::object smooth_grid,
           nb::object onecenter, nb::object becsum, nb::object paw, nb::object orbitals, nb::object symmetry,
           bool gamma, nb::object pools) {
            mdapy::dft::ScfInput in;
            in.gamma = gamma;
            if (!pools.is_none()) {
                // (rank, ranks, allreduce): allreduce(x) returns the sum over
                // the pools of the 1-D float64 array x, run with the GIL; its
                // errors come back as errors of the run
                nb::tuple t = nb::cast<nb::tuple>(pools);
                in.rank = nb::cast<int>(t[0]);
                in.ranks = nb::cast<int>(t[1]);
                nb::object f = t[2];
                in.allreduce = [f](double* x, std::size_t count) {
                    nb::gil_scoped_acquire gil;
                    try {
                        std::vector<double> part(x, x + count);
                        nb::object total = f(array_of(part, {count}));
                        auto values = nb::cast<nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(total);
                        if (values.shape(0) != count) throw std::invalid_argument("the sum over the pools has another size");
                        std::copy(values.data(), values.data() + count, x);
                    } catch (nb::python_error& error) {
                        throw std::runtime_error(std::string("the sum over the pools: ") + error.what());
                    }
                };
            }
            in.lattice = lattice_of(a);
            if (!basis.is_none()) {
                in.has_basis = true;
                in.basis = lattice_of(nb::cast<Matrix>(basis));
            }
            in.grid = grid_of(grid);
            in.cutoff = cutoff;
            in.density_cutoff = density_cutoff;
            in.species = species_of(tables, angular, dij, dq, orbitals);
            in.atoms = atoms_of(positions, species);
            if (kpoints.shape(1) != 3 || weights.shape(0) != kpoints.shape(0))
                throw std::invalid_argument("k-points (K, 3) and weights (K,)");
            for (std::size_t k = 0; k < kpoints.shape(0); ++k) {
                in.kpoints.push_back(
                    Vec3{kpoints.data()[3 * k], kpoints.data()[3 * k + 1], kpoints.data()[3 * k + 2]});
                in.weights.push_back(weights.data()[k]);
            }
            const std::size_t points = points_of(in.grid);
            if (density.size() != static_cast<std::size_t>(spins) * points)
                throw std::invalid_argument("the density of each spin on the grid");
            in.local_factors.assign(local.data(), local.data() + local.size());
            in.density.assign(density.data(), density.data() + density.size());
            if (!core.is_none()) {
                auto c = nb::cast<Rows>(core);
                in.core_factors.assign(c.data(), c.data() + c.size());
            }
            if (!local_slopes.is_none()) {
                auto c = nb::cast<Rows>(local_slopes);
                in.local_slopes.assign(c.data(), c.data() + c.size());
            }
            if (!core_slopes.is_none()) {
                auto c = nb::cast<Rows>(core_slopes);
                in.core_slopes.assign(c.data(), c.data() + c.size());
            }
            if (!atom_factors.is_none()) {
                auto c = nb::cast<Rows>(atom_factors);
                in.atom_factors.assign(c.data(), c.data() + c.size());
            }
            in.ewald = ewald;
            if (!smooth_grid.is_none()) in.smooth_grid = grid_of(nb::cast<std::array<int, 3>>(smooth_grid));
            if (!augmentation.is_none())
                for (nb::handle item : nb::cast<nb::list>(augmentation)) in.augmentation.push_back(augmentation_of(item));
            if (!paw.is_none())
                for (nb::handle item : nb::cast<nb::list>(paw)) in.paw.push_back(paw_of(item));
            if (!symmetry.is_none()) {
                // symmetry.payload: inverse_transposes (K, 9), translations (K, 3),
                // atoms (K, N), harmonics (K, sum (2l+1)^2), lmax
                nb::dict d = nb::cast<nb::dict>(symmetry);
                auto w = nb::cast<nb::ndarray<const int, nb::ndim<2>, nb::c_contig, nb::device::cpu>>(d["inverse_transposes"]);
                auto t = nb::cast<Rows>(d["translations"]);
                auto at = nb::cast<nb::ndarray<const int, nb::ndim<2>, nb::c_contig, nb::device::cpu>>(d["atoms"]);
                auto hm = nb::cast<Rows>(d["harmonics"]);
                const std::size_t ops = w.shape(0);
                if (w.shape(1) != 9 || t.shape(0) != ops || at.shape(0) != ops || hm.shape(0) != ops ||
                    at.shape(1) != in.atoms.size())
                    throw std::invalid_argument("the operations of the symmetry and their parts");
                in.symmetry.lmax = nb::cast<int>(d["lmax"]);
                for (std::size_t k = 0; k < ops; ++k) {
                    std::array<int, 9> matrix{};
                    for (int j = 0; j < 9; ++j)
                        matrix[static_cast<std::size_t>(j)] = w.data()[9 * k + static_cast<std::size_t>(j)];
                    in.symmetry.inverse_transposes.push_back(matrix);
                    in.symmetry.translations.push_back(Vec3{t.data()[3 * k], t.data()[3 * k + 1], t.data()[3 * k + 2]});
                    in.symmetry.atoms.emplace_back(at.data() + k * at.shape(1), at.data() + (k + 1) * at.shape(1));
                    in.symmetry.harmonics.emplace_back(hm.data() + k * hm.shape(1), hm.data() + (k + 1) * hm.shape(1));
                }
            }
            if (!onecenter.is_none() || !paw.is_none()) {
                if (becsum.is_none()) throw std::invalid_argument("PAW needs the first occupations becsum");
                in.becsum = spins_of(nb::cast<Rows>(becsum));
            }
            if (!onecenter.is_none()) {
                // a callable of becsum (spins, pairs) giving (E^1, D (spins, pairs)),
                // run with the GIL; its errors come back as errors of the run
                nb::object f = onecenter;
                in.onecenter = [f](const std::vector<std::vector<double>>& b, std::vector<std::vector<double>>& d) {
                    nb::gil_scoped_acquire gil;
                    try {
                        const std::size_t ns_ = b.size(), np_ = b.empty() ? 0 : b[0].size();
                        std::vector<double> flat;
                        flat.reserve(ns_ * np_);
                        for (const auto& x : b) flat.insert(flat.end(), x.begin(), x.end());
                        nb::tuple out = nb::cast<nb::tuple>(f(array_of(flat, {ns_, np_})));
                        const double e = nb::cast<double>(out[0]);
                        auto values = nb::cast<Rows>(out[1]);
                        if (values.shape(0) != ns_ || values.shape(1) != np_)
                            throw std::invalid_argument("the one-center strengths are (spins, pairs)");
                        d.assign(ns_, {});
                        for (std::size_t s = 0; s < ns_; ++s)
                            d[s].assign(values.data() + s * np_, values.data() + (s + 1) * np_);
                        return e;
                    } catch (nb::python_error& error) {
                        throw std::runtime_error(std::string("the one-center terms: ") + error.what());
                    }
                };
            }
            mdapy::dft::ScfOptions opt;
            opt.spins = spins;
            opt.bands = bands;
            opt.electrons = electrons;
            opt.functional = functional_of(functional);
            opt.smearing = mdapy::dft::smearing_of(smearing);
            opt.sigma = sigma;
            opt.mixing.beta = beta;
            opt.mixing.kerker = kerker;
            opt.mixing.minimum = minimum;
            opt.mixing.magnetic_beta = magnetic_beta;
            opt.mixing.magnetic_kerker = magnetic_kerker;
            opt.mixing.history = history;
            opt.mixing.metric_weight = metric_weight;
            opt.energy_tolerance = energy_tolerance;
            opt.density_tolerance = density_tolerance;
            opt.max_iterations = max_iterations;
            opt.davidson_iterations = davidson_iterations;
            opt.davidson_basis = davidson_basis;
            opt.seed = seed;
            mdapy::dft::ScfState* kept = state.is_none() ? nullptr : nb::cast<mdapy::dft::ScfState*>(state);
            mdapy::dft::ScfResult r;
            {
                nb::gil_scoped_release release;
                r = mdapy::dft::run_scf(in, opt, kept);
            }
            const std::size_t nk = in.kpoints.size(), nbands = static_cast<std::size_t>(bands),
                              ns_ = static_cast<std::size_t>(spins);
            std::vector<double> energies, accuracy, fermi, tolerance, moments;
            std::vector<int> davidson;
            for (const auto& step : r.steps) {
                energies.push_back(step.energy);
                accuracy.push_back(step.accuracy);
                fermi.push_back(step.fermi);
                tolerance.push_back(step.tolerance);
                moments.push_back(step.magnetization);
                davidson.push_back(step.davidson);
            }
            const std::size_t ns = r.steps.size();
            nb::dict out;
            out["converged"] = r.converged;
            out["restarted"] = r.restarted;
            out["gamma"] = in.gamma;
            nb::dict times;
            times["setup"] = r.times.setup;
            times["potentials"] = r.times.potentials;
            times["bands"] = r.times.bands;
            times["density"] = r.times.density;
            times["mixing"] = r.times.mixing;
            times["forces"] = r.times.forces;
            times["stress"] = r.times.stress;
            out["timings"] = times;
            out["iterations"] = r.iterations;
            out["energy"] = r.energy;
            out["band"] = r.band;
            out["one_electron"] = r.one_electron;
            out["hartree"] = r.hartree;
            out["xc"] = r.xc;
            out["ewald"] = r.ewald;
            out["entropy"] = r.entropy;
            out["fermi"] = r.fermi;
            out["lowest_empty"] = r.lowest_empty;
            out["magnetization"] = r.magnetization;
            out["absolute_magnetization"] = r.absolute_magnetization;
            out["eigenvalues"] = array_of(r.eigenvalues, {ns_, nk, nbands});
            out["occupations"] = array_of(r.occupations, {ns_, nk, nbands});
            out["density"] = array_of(r.density, {ns_, static_cast<std::size_t>(grid[0]),
                                                  static_cast<std::size_t>(grid[1]), static_cast<std::size_t>(grid[2])});
            out["energies"] = array_of(energies, {ns});
            out["accuracy"] = array_of(accuracy, {ns});
            out["fermi_levels"] = array_of(fermi, {ns});
            out["tolerances"] = array_of(tolerance, {ns});
            out["davidson"] = array_of(davidson, {ns});
            out["magnetizations"] = array_of(moments, {ns});
            out["forces"] = array_of(r.forces, {in.atoms.size(), 3});
            if (!r.becsum.empty()) {
                std::vector<double> flat;
                for (const auto& x : r.becsum) flat.insert(flat.end(), x.begin(), x.end());
                out["onecenter"] = r.onecenter;
                out["becsum"] = array_of(flat, {r.becsum.size(), r.becsum[0].size()});
            }
            if (r.has_strain) {
                const auto& e = r.strain;  // xx yy zz yz xz xy
                out["strain_derivative"] =
                    array_of(std::vector<double>{e[0], e[5], e[4], e[5], e[1], e[3], e[4], e[3], e[2]}, {3, 3});
            }
            return out;
        },
        nb::arg("a"), nb::arg("grid"), nb::arg("cutoff"), nb::arg("density_cutoff"), nb::arg("tables"),
        nb::arg("angular"), nb::arg("dij"), nb::arg("dq"), nb::arg("positions"), nb::arg("species"),
        nb::arg("kpoints"), nb::arg("weights"), nb::arg("local"), nb::arg("core").none(),
        nb::arg("local_slopes").none(), nb::arg("core_slopes").none(), nb::arg("atom_factors").none(),
        nb::arg("density"),
        nb::arg("ewald"), nb::arg("spins"), nb::arg("bands"), nb::arg("electrons"), nb::arg("functional"),
        nb::arg("smearing"), nb::arg("sigma"), nb::arg("beta"), nb::arg("kerker"), nb::arg("minimum"),
        nb::arg("magnetic_beta"), nb::arg("history"),
        nb::arg("metric_weight"), nb::arg("energy_tolerance"), nb::arg("density_tolerance"),
        nb::arg("max_iterations"), nb::arg("davidson_iterations"), nb::arg("davidson_basis"), nb::arg("seed"),
        nb::arg("basis").none() = nb::none(), nb::arg("state").none() = nb::none(),
        nb::arg("magnetic_kerker") = 0.0, nb::arg("augmentation").none() = nb::none(),
        nb::arg("smooth_grid").none() = nb::none(), nb::arg("onecenter").none() = nb::none(),
        nb::arg("becsum").none() = nb::none(), nb::arg("paw").none() = nb::none(),
        nb::arg("orbitals").none() = nb::none(), nb::arg("symmetry").none() = nb::none(),
        nb::arg("gamma") = false, nb::arg("pools").none() = nb::none());

    // the one-center terms of PAW, for the tests: the data of each species
    // (paw.onecenter or None), the species and projector functions of each
    // atom; compute(becsum (spins, pairs)) gives the energies and D (spins,
    // pairs)
    nb::class_<mdapy::dft::OneCenter>(m, "OneCenter")
        .def("__init__",
             [](mdapy::dft::OneCenter* self, nb::list species, std::vector<int> kinds, std::vector<int> widths,
                int spins, const std::string& functional, double threshold) {
                 std::vector<mdapy::dft::SpeciesOneCenter> all;
                 for (nb::handle item : species) all.push_back(paw_of(item));
                 new (self) mdapy::dft::OneCenter(std::move(all), std::move(kinds), std::move(widths), spins,
                                                  functional_of(functional), threshold);
             },
             nb::arg("species"), nb::arg("kinds"), nb::arg("widths"), nb::arg("spins"), nb::arg("functional"),
             nb::arg("threshold") = 1e-10)
        .def("compute", [](mdapy::dft::OneCenter& x, Rows becsum) {
            std::vector<std::vector<double>> d;
            mdapy::dft::OneCenterEnergies parts;
            double total = 0.0;
            {
                const auto b = spins_of(becsum);
                nb::gil_scoped_release release;
                total = x.compute(b, d, &parts);
            }
            nb::dict energies;
            energies["hartree_ae"] = parts.hartree_ae;
            energies["xc_ae"] = parts.xc_ae;
            energies["hartree_ps"] = parts.hartree_ps;
            energies["xc_ps"] = parts.xc_ps;
            energies["total"] = total;
            std::vector<double> flat;
            for (const auto& v : d) flat.insert(flat.end(), v.begin(), v.end());
            return nb::make_tuple(energies, array_of(flat, {d.size(), d.empty() ? 0 : d[0].size()}));
        });

    // the augmentation charges of ultrasoft species on the sphere of the
    // density, for the tests: the cell, the cutoff and grid of the density, the
    // atoms, and a dict of each species (augmentation.terms) or None
    nb::class_<mdapy::dft::Augmentation>(m, "Augmentation")
        .def("__init__",
             [](mdapy::dft::Augmentation* self, Matrix a, double cutoff, std::array<int, 3> grid, Rows positions,
                Kinds species, nb::list kinds) {
                 std::vector<mdapy::dft::SpeciesAugmentation> all;
                 for (std::size_t s = 0; s < nb::len(kinds); ++s) all.push_back(augmentation_of(kinds[s]));
                 const auto sphere = mdapy::dft::make_sphere(lattice_of(a), Vec3{0.0, 0.0, 0.0}, cutoff, grid_of(grid), false);
                 new (self) mdapy::dft::Augmentation(lattice_of(a), sphere, atoms_of(positions, species), std::move(all));
             })
        .def("overlaps", [](const mdapy::dft::Augmentation& x, int s) {
            const auto q = x.overlaps(s);
            return array_of(q, {q.size()});
        })
        .def("components", [](mdapy::dft::Augmentation& x, int s) {
            const auto q = x.components(s);
            const std::size_t pairs = static_cast<std::size_t>(x.pairs(s));
            std::vector<std::complex<double>> out(q.size());
            for (std::size_t i = 0; i < q.size(); ++i) out[i] = {q[i].x, q[i].y};
            return array_of(out, {pairs == 0 ? 0 : q.size() / pairs, pairs});
        })
        .def("add_density",
             [](mdapy::dft::Augmentation& x, nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> becsum,
                std::size_t n) {
                 DeviceArray<Complex> rho(n);
                 rho.zero();
                 x.add_density(std::vector<double>(becsum.data(), becsum.data() + becsum.size()), rho.data());
                 std::vector<Complex> host(n);
                 rho.download(host.data(), n);
                 std::vector<std::complex<double>> out(n);
                 for (std::size_t i = 0; i < n; ++i) out[i] = {host[i].x, host[i].y};
                 return array_of(out, {n});
             })
        .def("strengths", [](mdapy::dft::Augmentation& x, Coefficients1 v) {
            DeviceArray<Complex> d;
            d.upload(as_device(v.data()), v.size());
            const auto out = x.strengths(d.data());
            return array_of(out, {out.size()});
        });

    // what a run of the self-consistent field leaves for the next one
    nb::class_<mdapy::dft::ScfState>(m, "State")
        .def(nb::init<>())
        .def_prop_ro("valid", [](const mdapy::dft::ScfState& s) { return s.valid; })
        .def("clear", [](mdapy::dft::ScfState& s) { s = mdapy::dft::ScfState(); });
}
