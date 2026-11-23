# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The symmetry of crystals in the plane-wave DFT: the operations of spglib
and the atoms they take each atom to, the irreducible k-points of a mesh, the
rotations of the real harmonics; runs with symmetry against runs on the
whole mesh without, and against pw.x 7.6 using its own symmetry on the same
irreducible k-points."""

import numpy as np
import pytest

from mdapy.calculators._dft import angular, paw, scf, symmetry
from mdapy.calculators._dft.upf import read_upf

from dft import espresso, paw_reference
from dft.test_paw import dataset

pytest.importorskip("spglib")
_dft = pytest.importorskip("mdapy_dft_cuda12._dft")

BOHR = 0.529177210903
KBAR = 294210.1


def fcc(a):
    h = a / 2 / BOHR
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


def bcc(a):
    h = a / 2 / BOHR
    return np.array([[-h, h, h], [h, -h, h], [h, h, -h]])


DIAMOND = [[0, 0, 0], [0.25, 0.25, 0.25]]
SHIFTED = [
    [0.01, 0.01, 0.01],
    [0.25, 0.25, 0.25],
]  # D3d: inversion through their middle


def test_operations():
    ops = symmetry.operations(fcc(4.05), [[0, 0, 0]], [0])
    assert len(ops) == 48 and not ops.translations.any()
    ops = symmetry.operations(fcc(5.431), DIAMOND, [0, 0])
    assert len(ops) == 48
    assert (np.abs(ops.translations).sum(axis=1) > 1e-8).sum() == 24
    # the FFT grids of diamond fit translations of a quarter, as QE's
    assert symmetry.fft_factors(ops) == (4, 4, 4)
    assert symmetry.fft_factors(symmetry.operations(fcc(4.05), [[0, 0, 0]], [0])) == (
        1,
        1,
        1,
    )
    # each operation takes the atoms onto atoms
    cell = fcc(5.431)
    for r, t, atoms in zip(ops.rotations, ops.translations, ops.atoms):
        images = np.asarray(DIAMOND) @ r.T + t
        apart = images - np.asarray(DIAMOND)[atoms]
        np.testing.assert_allclose(apart - np.round(apart), 0.0, atol=1e-12)
    # C acts on Cartesian vectors as R on fractions, and is orthogonal
    for r, c in zip(ops.rotations, ops.cartesian):
        np.testing.assert_allclose(c @ c.T, np.eye(3), atol=1e-12)
        x = np.array([0.3, -0.1, 0.2])
        np.testing.assert_allclose(c @ (cell.T @ x), cell.T @ (r @ x), atol=1e-12)
    assert len(symmetry.operations(fcc(5.431), SHIFTED, [0, 0])) == 12
    # opposite moments on the two atoms of CsCl keep the operations of one
    # cube, not those that swap them
    ops = symmetry.operations(
        np.eye(3) * 5.4, [[0, 0, 0], [0.5, 0.5, 0.5]], [0, 0], [2.0, -2.0]
    )
    assert len(ops) == 48 and (ops.atoms == np.arange(2)).all()
    assert (
        len(symmetry.operations(np.eye(3) * 5.4, [[0, 0, 0], [0.5, 0.5, 0.5]], [0, 0]))
        == 96
    )


def test_irreducible_kpoints():
    ops = symmetry.operations(fcc(4.05), [[0, 0, 0]], [0])
    full, _ = _dft.monkhorst_pack([4, 4, 4], [0.0, 0.0, 0.0], False)
    k, w = symmetry.irreducible(full, ops.rotations)
    assert len(k) == 8 and w.sum() == pytest.approx(1.0)

    # the stars of the irreducible points cover the mesh, as their weights say
    def key(x):
        return tuple(np.round((x - np.floor(x + 1e-9)) * 1e6).astype(int))

    mesh = {key(x) for x in full}
    for point, weight in zip(k, w):
        star = {key(s * (r.T @ point)) for r in ops.rotations for s in (1, -1)}
        assert star <= mesh and len(star) == round(weight * len(full))
    # without symmetry, time reversal alone
    k1, w1 = symmetry.irreducible(full, np.eye(3, dtype=int)[None])
    assert len(k1) == 36 and np.all(w1 > 0)


def test_rotations_of_the_harmonics():
    ops = symmetry.operations(fcc(5.431), DIAMOND, [0, 0])
    blocks = symmetry.harmonics(ops.cartesian)
    rng = np.random.default_rng(2)
    v = rng.normal(size=(20, 3))
    for c, op in zip(ops.cartesian, blocks):
        y = angular.real_harmonics(3, v).T
        yr = angular.real_harmonics(3, v @ np.linalg.inv(c).T).T
        for ell, m in enumerate(op):
            part = slice(ell * ell, (ell + 1) * (ell + 1))
            np.testing.assert_allclose(m @ m.T, np.eye(2 * ell + 1), atol=1e-12)
            np.testing.assert_allclose(yr[:, part], y[:, part] @ m, atol=1e-12)
    # a representation: M(C1 C2) = M(C1) M(C2)
    c = ops.cartesian[5] @ ops.cartesian[9]
    k = int(np.argmin([np.abs(x - c).max() for x in ops.cartesian]))
    for ell in range(4):
        np.testing.assert_allclose(
            blocks[k][ell], blocks[5][ell] @ blocks[9][ell], atol=1e-12
        )


def ultrasoft_or_nc(name):
    return (
        dataset(name.split()[0])
        if "PAW" in name
        else read_upf(espresso.pseudo(f"{name.split()[0]}_std.upf"))
    )


CASES = {
    "Al fcc": (
        fcc(4.05),
        [[0, 0, 0]],
        dict(kgrid=(4, 4, 4), smearing="fermi-dirac", sigma=0.00225),
    ),
    "Si diamond": (fcc(5.431), DIAMOND, dict(kgrid=(3, 3, 3), smearing="fixed")),
    "Si shifted": (fcc(5.431), SHIFTED, dict(kgrid=(3, 3, 3), smearing="fixed")),
    "Si PAW diamond": (fcc(5.431), DIAMOND, dict(kgrid=(3, 3, 3), smearing="fixed")),
    "Fe PAW bcc": (
        bcc(2.87),
        [[0, 0, 0]],
        dict(kgrid=(4, 4, 4), smearing="gaussian", sigma=0.01, magmoms=[2.5]),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_with_and_without_symmetry(name):
    """The same on the irreducible k-points as on the whole mesh. With PAW
    the quadrature of the one-center terms is not cubic, and a run without
    symmetry keeps a trace of that, as in QE: 1e-8 hartree in iron."""
    if _dft.device_count() == 0:
        pytest.skip("no GPU")
    cell, positions, options = CASES[name]
    pp = ultrasoft_or_nc(name)
    cutoff = 15.0 if "Fe" in name else 12.0
    runs = [
        scf.run(cell, positions, [0] * len(positions), [pp], cutoff, functional="pbe_pw",
                energy_tolerance=1e-12, density_tolerance=1e-14, symmetry=s, **options)
        for s in (True, False)
    ]  # fmt: skip
    a, b = runs
    assert (
        a["operations"] > 1
        and b["operations"] == 1
        and len(a["kpoints"]) < len(b["kpoints"])
    )
    tolerance = 3e-8 if "PAW" in name else 1e-12
    assert a["energy"] == pytest.approx(b["energy"], abs=tolerance)
    np.testing.assert_allclose(a["forces"], b["forces"], atol=1e-8)
    np.testing.assert_allclose(
        a["strain_derivative"],
        b["strain_derivative"],
        atol=3e-7 if "PAW" in name else 1e-7,
    )
    if "magmoms" in options:
        assert a["magnetization"] == pytest.approx(b["magnetization"], abs=1e-6)
    if name == "Si shifted":
        assert np.abs(b["forces"]).max() > 1e-2


@pytest.mark.parametrize("name", ["Si PAW diamond", "Fe PAW bcc"])
def test_against_pw_with_symmetry(name):
    if _dft.device_count() == 0:
        pytest.skip("no GPU")
    cell, positions, options = CASES[name]
    element = name.split()[0]
    path = espresso.pseudo(
        {
            "Si": "pslibrary/Si.pbe-n-kjpaw_psl.1.0.0.UPF",
            "Fe": "pslibrary/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
        }[element]
    )
    pp = read_upf(path)
    cutoff, dual = (20.0 if element == "Fe" else 15.0), 8.0
    first = scf.run(cell, positions, [0] * len(positions), [pp], cutoff, functional="pbe_pw", density_cutoff=dual * cutoff,
                    energy_tolerance=1e-10, stress=False, **options)  # fmt: skip
    magmoms = options.get("magmoms")
    qe = espresso.run(
        cell, positions, [element] * len(positions), {element: path}, cutoff, first["kpoints"], first["weights"],
        options["smearing"], options.get("sigma", 0.0), conv_thr=1e-13, magmoms=magmoms, beta=0.3, maxstep=300,
        extra=f", ecutrho = {2 * dual * cutoff:.10f}", symmetric=True,
    )  # fmt: skip
    assert "48 Sym. Ops." in qe["text"]
    # the grids chosen with the symmetry are those of pw.x with it
    assert tuple(first["grid"]) == qe["grid"]
    assert tuple(first["smooth_grid"]) == qe["smooth_grid"]
    r = scf.run(cell, positions, [0] * len(positions), [pp], cutoff, functional="pbe_pw", density_cutoff=dual * cutoff,
                grid=qe["grid"], smooth_grid=qe["smooth_grid"], energy_tolerance=1e-12, density_tolerance=1e-14,
                **options)  # fmt: skip
    energy = r["energy"]
    if magmoms is not None:
        energy += espresso.rounded_gamma(r, cell, positions, [0], [pp], dual * cutoff)
        energy += paw_reference.rounded_gamma(paw.onecenter(pp, True), r["becsum"])
        assert r["magnetization"] == pytest.approx(qe["magnetization"], abs=1e-5)
    assert energy == pytest.approx(qe["energy"], abs=1e-8 * len(positions))
    np.testing.assert_allclose(r["forces"], qe["forces"], atol=1e-7)
    volume = abs(np.linalg.det(cell))
    np.testing.assert_allclose(
        -r["strain_derivative"] / volume * KBAR, qe["stress"] * KBAR, atol=0.02
    )
