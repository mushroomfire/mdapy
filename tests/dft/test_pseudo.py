# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The parts of the plane-wave DFT on the processor: UPF files read, in
hartree; Simpson's rule and the spherical Bessel functions; the transforms of
the projectors and the local potential against analytic ones, and the
interpolation of their tables; the Ewald sum against the Madelung constants
of rock salt and CsCl, and its forces against its differences. Pseudopotentials
of PseudoDojo are read from MDAPY_PSEUDO_DIR when it is set."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from mdapy.calculators._dft import radial
from mdapy.calculators._dft.ewald import ewald
from mdapy.calculators._dft.upf import read_upf
from mdapy.errors import InvalidParameter

from dft import gpu

BOHR = 0.529177210903  # Å


def write_upf(path, kind="NC", version="2.0.1", spin_orbit="F", core="T"):
    """A small UPF file: a mesh of 401 points every 0.01 bohr, a local potential,
    two projectors of l = 0 and 1, D, a core charge and the atom."""
    r = 0.01 * np.arange(401)

    def block(tag, values, extra=""):
        rows = "\n".join(" ".join(f"{v:.12e}" for v in values[i : i + 4])
                         for i in range(0, len(values), 4))  # fmt: skip
        return f'<{tag} type="real" size="{len(values)}" columns="4"{extra}>\n{rows}\n</{tag}>'

    beta0 = r * np.exp(-r * r)
    beta1 = r * r * np.exp(-r * r)
    text = f"""<UPF version="{version}">
<PP_INFO>input with a < sign & an ampersand</PP_INFO>
<PP_HEADER element="X" pseudo_type="{kind}" is_ultrasoft="F" is_paw="F" has_so="{spin_orbit}"
 core_correction="{core}" functional="PBE" z_valence="  3.00" mesh_size="401"
 number_of_proj="2"/>
<PP_MESH>
{block("PP_R", r)}
{block("PP_RAB", np.full(401, 0.01))}
</PP_MESH>
{block("PP_LOCAL", -6.0 * np.exp(-r))}
<PP_NONLOCAL>
{block("PP_BETA.2", beta1, ' index="2" angular_momentum="1"')}
{block("PP_BETA.1", beta0, ' index="1" angular_momentum="0"')}
{block("PP_DIJ", np.array([4.0, 0.0, 0.0, -2.0]))}
</PP_NONLOCAL>
<PP_PSWFC>
{block("PP_CHI.1", r * np.exp(-r), ' label="3S" l="0" occupation="2.0"')}
</PP_PSWFC>
{block("PP_NLCC", np.exp(-r))}
{block("PP_RHOATOM", 3 * 4 * r * r * np.exp(-2 * r) * 4 / (4 * math.pi) * 4 * math.pi / 4)}
</UPF>
"""
    path.write_text(text)
    return path


def test_a_upf_file_in_hartree(tmp_path):
    pp = read_upf(write_upf(tmp_path / "X.upf"))
    assert pp.element == "X" and pp.valence == 3.0 and pp.functional == "PBE"
    assert len(pp.r) == 401 and pp.r[-1] == pytest.approx(4.0)
    np.testing.assert_allclose(pp.local, -3.0 * np.exp(-pp.r), rtol=1e-11)  # Ry to Ha
    assert [p.angular for p in pp.projectors] == [0, 1]  # by their index
    np.testing.assert_allclose(pp.dij, [[2.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(pp.core, np.exp(-pp.r), rtol=1e-11)
    assert pp.orbitals[0].label == "3S" and pp.orbitals[0].occupation == 2.0
    assert pp.lmax == 1 and len(pp.digest) == 64
    plain = read_upf(write_upf(tmp_path / "Y.upf", core="F"))
    assert plain.core is None


@pytest.mark.parametrize(
    ("options", "message"),
    [({"version": "1.0"}, "version 1 or 2"), ({"kind": "US"}, "no PP_AUGMENTATION"),
     ({"kind": "PAW"}, "no PP_AUGMENTATION"),
     ({"kind": "XX"}, "norm-conserving, ultrasoft and PAW"), ({"spin_orbit": "T"}, "spin-orbit")],
)  # fmt: skip
def test_what_is_refused(tmp_path, options, message):
    with pytest.raises(InvalidParameter, match=message):
        read_upf(write_upf(tmp_path / "X.upf", **options))


def test_the_file_of_an_element_in_a_folder(tmp_path):
    """{element}.upf, or the one {element}_* or {element}.* of it, the element
    and the extension in either case, as the files of SSSP are named."""
    from mdapy.calculators._dft.calculator import _find_pseudopotential

    for name in ["ni_pbe_v1.4.uspp.F.UPF", "s_pbe_v1.4.uspp.F.UPF", "sb_pbe_v1.4.uspp.F.UPF",
                 "Sb_other.upf", "Hg.us.z_12.uspp.gbrv.v1.upf", "Al.upf", "Al_extra.upf", "notes.txt"]:  # fmt: skip
        (tmp_path / name).write_text("")
    assert _find_pseudopotential(tmp_path, "Ni").name == "ni_pbe_v1.4.uspp.F.UPF"
    assert _find_pseudopotential(tmp_path, "S").name == "s_pbe_v1.4.uspp.F.UPF"
    assert _find_pseudopotential(tmp_path, "Hg").name == "Hg.us.z_12.uspp.gbrv.v1.upf"
    assert (
        _find_pseudopotential(tmp_path, "Al").name == "Al.upf"
    )  # the plain name first
    with pytest.raises(InvalidParameter, match="Sb_other.upf, sb_pbe_v1.4.uspp.F.UPF"):
        _find_pseudopotential(tmp_path, "Sb")
    with pytest.raises(InvalidParameter, match="found none"):
        _find_pseudopotential(tmp_path, "Xe")


def test_simpson_and_bessel():
    r = 0.01 * np.arange(1001)
    rab = np.full_like(r, 0.01)
    assert radial.simpson(r**3, rab) == pytest.approx(10.0**4 / 4, rel=1e-13)
    # an even number of points closes on the last interval, as QE 7.6 does
    assert radial.simpson(r**2, rab, 1000) == pytest.approx(9.99**3 / 3, rel=1e-13)
    assert radial.simpson(r**3, rab, 1000) == pytest.approx(9.99**4 / 4, rel=1e-9)
    assert radial.simpson(r, rab, 2) == 0.0
    assert radial.reach(r) == 1001  # the first beyond 10 bohr, odd
    x = np.linspace(0.0, 20.0, 2001)
    safe = np.where(x > 0, x, 1.0)
    exact = {
        0: np.sinc(x / math.pi),
        1: np.where(x > 0, np.sin(safe) / safe**2 - np.cos(safe) / safe, 0.0),
    }
    for ell, values in exact.items():
        np.testing.assert_allclose(radial.bessel(ell, x), values, atol=1e-12)
    # the series and the closed forms agree where they meet
    for ell in range(4):
        below = radial.bessel(ell, np.array([1.0 - 1e-9]))
        above = radial.bessel(ell, np.array([1.0]))
        assert below[0] == pytest.approx(above[0], rel=1e-8)
    # l up to 6 against j_l = (-i)^l / 2 int exp(ixt) P_l(t) dt by Gauss-Legendre
    t, w = np.polynomial.legendre.leggauss(200)
    for ell in range(7):
        legendre = np.polynomial.legendre.Legendre.basis(ell)(t)
        exact = np.real(
            0.5 * (-1j) ** ell * (np.exp(1j * np.outer(x, t)) * legendre * w).sum(1)
        )
        np.testing.assert_allclose(radial.bessel(ell, x), exact, atol=1e-13)
    with pytest.raises(ValueError, match="up to l = 6"):
        radial.bessel(7, x)


def gaussian_pseudopotential(ell):
    """r beta(r) = r^(l+1) exp(-r^2 / 2): its transform is (4 pi / sqrt(Omega))
    sqrt(pi / 2) q^l exp(-q^2 / 2)."""

    class P:
        r = 0.01 * np.arange(1601)
        rab = np.full(1601, 0.01)
        kkbeta = 1601
        projectors = (
            type(
                "B",
                (),
                {
                    "angular": ell,
                    "r_beta": r ** (ell + 1) * np.exp(-r * r / 2),
                    "cutoff_index": len(r),
                },
            ),
        )

    return P


@pytest.mark.parametrize("ell", [0, 1, 2, 3])
def test_projector_tables_and_their_interpolation(ell):
    volume = 300.0
    table = radial.projector_table(gaussian_pseudopotential(ell), 6.0, volume)
    q = radial.DQ * np.arange(table.shape[1])
    factor = 4 * math.pi / math.sqrt(volume) * math.sqrt(math.pi / 2)
    # Simpson's rule every 0.01 bohr, to a few 1e-9
    np.testing.assert_allclose(
        table[0], factor * q**ell * np.exp(-q * q / 2), atol=1e-8
    )
    between = np.linspace(0.0, 5.9, 97)
    np.testing.assert_allclose(
        radial.interpolate(table, between)[0],
        factor * between**ell * np.exp(-(between**2) / 2),
        atol=1e-8,
    )


def test_the_local_potential_of_a_bare_erf():
    """V(r) = -Z erf(r) / r: what is left in real space is nothing; V(q) =
    -(4 pi / Omega) Z exp(-q^2/4) / q^2, and at q = 0 pi Z / Omega."""

    class P:
        r = 0.01 * np.arange(1201)
        rab = np.full(1201, 0.01)
        valence = 3.0
        local = -3.0 * np.array([math.erf(x) / x if x > 0 else 2 / math.sqrt(math.pi)
                                 for x in r])  # fmt: skip

    volume = 250.0
    q = np.array([0.0, 0.3, 1.0, 2.5])
    v = radial.local_potential(P, q, volume)
    assert v[0] == pytest.approx(math.pi * 3.0 / volume, rel=1e-6)
    np.testing.assert_allclose(
        v[1:],
        -4 * math.pi / volume * 3.0 * np.exp(-(q[1:] ** 2) / 4) / q[1:] ** 2,
        rtol=1e-9,
    )


def test_the_slopes_of_the_tables():
    """The slopes in q are the derivatives of the interpolated values, which
    stay continuous across the points of the tables."""

    class P:
        r = 0.01 * np.arange(1201)
        rab = np.full(1201, 0.01)
        valence = 3.0
        local = np.array([-3.0 * math.erf(x) / x if x > 0 else -6 / math.sqrt(math.pi)
                          for x in r]) + 2.0 * np.exp(-r * r)  # fmt: skip
        core = np.exp(-2 * r * r)

    volume = 250.0
    q = np.array([0.137, 0.731, 1.2345, 3.3321])
    h = 1e-6
    for value, slope in [
        (lambda x: radial.local_potential(P, x, volume),
         radial.local_potential_slope(P, q, volume)),
        (lambda x: radial.radial_density(P.core, P, x, volume, True),
         radial.radial_density_slope(P.core, P, q, volume, True)),
    ]:  # fmt: skip
        np.testing.assert_allclose(
            (value(q + h) - value(q - h)) / (2 * h), slope, rtol=1e-6, atol=1e-10
        )
        knots = radial.DQ * np.array([17.0, 123.0])
        np.testing.assert_allclose(
            value(knots - 1e-12), value(knots + 1e-12), rtol=1e-10
        )
    assert radial.local_potential_slope(P, np.array([0.0]), volume)[0] == 0.0
    assert (
        radial.radial_density_slope(P.core, P, np.array([0.0]), volume, True)[0] == 0.0
    )


def test_ewald_against_madelung():
    a = 10.0
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    rocksalt = ewald(
        np.eye(3) * a, np.vstack([fcc, fcc + [0.5, 0, 0]]) * a, [1] * 4 + [-1] * 4
    )
    assert -rocksalt / 4 * (a / 2) == pytest.approx(1.747564594633, rel=1e-10)
    cscl = ewald(np.eye(3) * a, np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * a, [1, -1])
    assert -cscl * a * math.sqrt(3) / 2 == pytest.approx(1.762674773070, rel=1e-10)
    # silicon with an atom moved: a translation and a swap of the atoms leave
    # it (QE run live compares it in test_scf.py)
    half = 5.431 / 2 / BOHR
    cell = np.array([[0, half, half], [half, 0, half], [half, half, 0]])
    positions = np.array([[0, 0, 0], [0.27, 0.25, 0.25]]) @ cell
    assert ewald(cell, positions[::-1] + 1.3, [4.0, 4.0]) == pytest.approx(
        ewald(cell, positions, [4.0, 4.0]), abs=1e-11
    )


def test_ewald_forces():
    rng = np.random.default_rng(7)
    cell = np.array([[6.0, 0.3, 0.0], [0.5, 7.0, 0.2], [0.0, -0.4, 6.5]])
    positions = rng.uniform(0, 1, (4, 3)) @ cell
    charges = np.array([3.0, 1.0, 2.0, 4.0])
    energy, forces = ewald(cell, positions, charges, forces=True)
    assert energy == ewald(cell, positions, charges)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-10)
    h = 1e-4
    for atom, d in [(0, 0), (2, 1), (3, 2)]:
        up, down = positions.copy(), positions.copy()
        up[atom, d] += h
        down[atom, d] -= h
        slope = (ewald(cell, up, charges) - ewald(cell, down, charges)) / (2 * h)
        assert forces[atom, d] == pytest.approx(-slope, abs=1e-8)
    # the atoms of the rock salt feel nothing
    a = 10.0
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    _, f = ewald(np.eye(3) * a, np.vstack([fcc, fcc + [0.5, 0, 0]]) * a,
                 [1] * 4 + [-1] * 4, forces=True)  # fmt: skip
    np.testing.assert_allclose(f, 0.0, atol=1e-12)


PSEUDO = os.environ.get("MDAPY_PSEUDO_DIR")


@pytest.mark.skipif(not PSEUDO, reason="MDAPY_PSEUDO_DIR names no pseudopotentials")
@pytest.mark.parametrize("element", ["Si", "Fe", "Cu"])
def test_pseudodojo(element):
    pp = read_upf(Path(PSEUDO) / f"{element}_std.upf")
    assert pp.element == element and pp.core is not None
    # the free atom holds the valence
    assert radial.simpson(pp.atom, pp.rab) == pytest.approx(pp.valence, rel=1e-5)
    np.testing.assert_allclose(pp.dij, pp.dij.T)


@pytest.mark.skipif(not PSEUDO, reason="MDAPY_PSEUDO_DIR names no pseudopotentials")
@pytest.mark.parametrize("name", ["ni_pbe_v1.4.uspp.F.UPF", "W_pbe_v1.2.uspp.F.UPF"])
def test_gbrv_of_version_1(name):
    """Ultrasoft pseudopotentials of GBRV in files of version 1, from SSSP: the
    header, the projectors, the orbitals; inside rinner the augmentation
    charges are r^(L+2) times the polynomial of their coefficients, which
    keeps their integrals of L = 0, q_ij (test_ultrasoft_qe runs them against
    Quantum ESPRESSO)."""
    path = Path(PSEUDO) / "sssp" / name
    if not path.is_file():
        pytest.skip(f"no {path.name}")
    pp = read_upf(path)
    assert pp.ultrasoft and pp.paw is None and pp.core is not None
    from mdapy.calculators._dft.scf import functional_of

    assert functional_of([pp]) == "pbe_pw"
    assert radial.simpson(pp.atom, pp.rab) < pp.valence  # without the augmentation
    angular = [p.angular for p in pp.projectors]
    assert angular == sorted(angular) and all(p.cutoff_index > 0 for p in pp.projectors)
    assert [o.label[-1] for o in pp.orbitals][:3] == ["S", "P", "D"]
    np.testing.assert_allclose(pp.q, pp.q.T)
    for (i, j, ell), charge in pp.augmentation.items():
        assert abs(angular[i] - angular[j]) <= ell <= angular[i] + angular[j]
        if ell == 0:
            assert radial.simpson(charge, pp.rab) == pytest.approx(pp.q[i, j], abs=1e-5)


@pytest.mark.skipif(not PSEUDO, reason="MDAPY_PSEUDO_DIR names no pseudopotentials")
def test_fortran_numbers_of_atompaw():
    """The PAW datasets of the lanthanides of SSSP, made by atompaw, write
    1.d-12 in their attributes."""
    path = Path(PSEUDO) / "sssp" / "La.paw.z_11.atompaw.wentzcovitch.v1.2.upf"
    if not path.is_file():
        pytest.skip(f"no {path.name}")
    pp = read_upf(path)
    assert pp.paw is not None and pp.valence == 11.0 and pp.lmax == 3


@pytest.mark.skipif(not PSEUDO, reason="MDAPY_PSEUDO_DIR names no pseudopotentials")
@pytest.mark.parametrize("element", ["Al", "Fe"])
def test_pslibrary_ultrasoft(element):
    """The ultrasoft pseudopotentials of pslibrary 1.0.0 made by ld1.x: the
    integrals of the augmentation charges of L = 0 are q_ij of the file, to the
    1e-6 at which its two parts agree (the DFT takes the integrals, as QE)."""
    name = {
        "Al": "pslibrary/Al.pbe-n-rrkjus_psl.1.0.0.UPF",
        "Fe": "pslibrary/Fe.pbe-spn-rrkjus_psl.1.0.0.UPF",
    }
    path = Path(PSEUDO) / name[element]
    if not path.is_file():
        pytest.skip(f"no {path.name}")
    pp = read_upf(path)
    assert pp.ultrasoft and pp.element == element
    np.testing.assert_allclose(pp.q, pp.q.T)
    angular = [p.angular for p in pp.projectors]
    for i in range(len(angular)):
        for j in range(i, len(angular)):
            if angular[i] == angular[j]:
                integral = radial.simpson(pp.augmentation[(i, j, 0)], pp.rab)
                assert integral == pytest.approx(pp.q[i, j], abs=2e-6)
            else:
                assert (i, j, 0) not in pp.augmentation
    # L runs from |l_i - l_j| to l_i + l_j by twos
    for i, j, ell in pp.augmentation:
        assert abs(angular[i] - angular[j]) <= ell <= angular[i] + angular[j]
        assert (ell + angular[i] + angular[j]) % 2 == 0


def test_functional_of_headers():
    from types import SimpleNamespace

    from mdapy.calculators._dft.scf import functional_of

    def of(*names):
        return functional_of([SimpleNamespace(functional=n) for n in names])

    # pslibrary writes QE's parts with several spaces between them
    # as Quantum ESPRESSO takes them, PW92 with the constants of its paper
    assert of(" SLA  PW   PBX  PBC") == "pbe_pw"
    assert of("PBE", "sla pw pbx pbc") == "pbe_pw"
    assert of("PW", "LDA") == "lda_pw"
    assert of("PBESOL") == "pbesol_pw" and of(" SLA  PW   PSX  PSC") == "pbesol_pw"
    with pytest.raises(ValueError, match="different functionals"):
        of("PBE", "PW")
    with pytest.raises(ValueError, match="no functional"):
        of("B3LYP")


def test_ewald_of_the_extension():
    """The Ewald sum of the extension, whose split grows with the atoms,
    against the NumPy one: a small triclinic cell and 64 atoms."""
    _dft = gpu.module()
    rng = np.random.default_rng(11)
    for count, cell in [
        (4, np.array([[6.0, 0.3, 0.0], [0.5, 7.0, 0.2], [0.0, -0.4, 6.5]])),
        (64, np.eye(3) * 20.5 + rng.normal(0, 0.3, (3, 3))),
    ]:
        positions = rng.uniform(0, 1, (count, 3)) @ cell
        charges = rng.uniform(1, 4, count)
        energy, forces, strain = ewald(cell, positions, charges, True, True)
        e, f, s = _dft.ewald(cell, np.ascontiguousarray(positions), charges)
        assert e == pytest.approx(energy, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(f, forces, rtol=0, atol=1e-10)
        np.testing.assert_allclose(s, strain, rtol=0, atol=1e-9)


def test_first_density_on_the_gpu(tmp_path):
    """The density and magnetization of the free atoms on the sphere, summed
    over the atoms on the GPU, against NumPy."""
    _dft = gpu.module()
    if _dft.device_count() == 0:
        pytest.skip("no GPU")
    from mdapy.calculators._dft import scf

    cell = np.array([[7.0, 0.2, 0.0], [0.0, 6.5, 0.3], [0.4, 0.0, 8.0]])
    rng = np.random.default_rng(5)
    positions = rng.uniform(0, 1, (5, 3))
    kinds = [0, 1, 1, 0, 1]
    pseudos = [
        read_upf(write_upf(tmp_path / "X.upf")),
        read_upf(write_upf(tmp_path / "Y.upf", core="F")),
    ]
    magmoms = [0.5, -0.2, 0.1, 0.3, 0.0]
    miller, _, _, _ = _dft.sphere(
        cell, [0.0, 0.0, 0.0], 20.0, _dft.fft_grid(cell, 20.0)
    )
    _, _, atom, moment = scf.radial_parts(
        cell, positions, kinds, pseudos, miller, magmoms
    )
    _, _, factors = scf.form_factors(cell, pseudos, miller)
    gpu_atom, gpu_moment = scf.first_density(
        factors, positions, kinds, pseudos, miller, magmoms
    )
    np.testing.assert_allclose(gpu_atom, atom, rtol=0, atol=1e-12 * np.abs(atom).max())
    np.testing.assert_allclose(
        gpu_moment, moment, rtol=0, atol=1e-12 * np.abs(atom).max()
    )
