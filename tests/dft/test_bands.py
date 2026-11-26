# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The bands at any k-points in the potential of a self-consistent density
(one step of the field at that density, every band converged) against pw.x
7.6 run live, calculation = "bands" after its own field with the same
pseudopotentials, cutoffs, grids and mesh: silicon, aluminium and iron with
spin (norm-conserving), nickel (ultrasoft, GBRV), silicon and iron with spin
(PAW), occupied and empty bands, eigenvalues from QE's XML data file. PAW
needs the occupations of the projectors of the field with its density. The
invariances: the mesh of the field again, k and -k, k and k + G, k and its
image under a rotation of the crystal."""

import numpy as np
import pytest

from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf
from mdapy.properties import _paths

from dft import espresso, gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
SIGMA = 0.01


def fcc(a):
    h = a / 2 / BOHR
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


def bcc(a):
    h = a / 2 / BOHR
    return np.array([[-h, h, h], [h, -h, h], [h, h, -h]])


# special points in units of the reciprocal vectors of these primitive cells
FCC = [("Γ", (0, 0, 0)), ("X", (0.5, 0, 0.5)), ("W", (0.5, 0.25, 0.75)),
       ("L", (0.5, 0.5, 0.5)), ("Γ", (0, 0, 0))]  # fmt: skip
BCC = [("Γ", (0, 0, 0)), ("H", (0.5, -0.5, 0.5)), ("N", (0, 0, 0.5)),
       ("P", (0.25, 0.25, 0.25)), ("Γ", (0, 0, 0))]  # fmt: skip
SHIFTED = [[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]]
# cell, positions, file, smearing, mesh, moments, cutoff, dual, bands, path,
# the tolerance of the eigenvalues (hartree) and options of pw.x. QE's
# spin-polarized PBE takes a rounded gamma (test_spin.py): its eigenvalues
# then differ by some 1e-6
CASES = {
    "Si": (fcc(5.431), SHIFTED, "Si_std.upf", "fixed", (3, 3, 3), None, 12.0, 4.0, 12,
           FCC, 1e-7, {}),
    "Al": (fcc(4.05), [[0.0, 0.0, 0.0]], "Al_std.upf", "marzari-vanderbilt", (4, 4, 4),
           None, 12.0, 4.0, 8, FCC, 1e-7, {}),
    "Fe": (bcc(2.87), [[0.0, 0.0, 0.0]], "Fe_std.upf", "gaussian", (3, 3, 3), [2.5], 25.0,
           4.0, 14, BCC, 3e-6, dict(beta=0.3)),
    "Ni-ultrasoft": (fcc(3.52), [[0.0, 0.0, 0.0]], "gbrv/ni_pbe_v1.4.uspp.F.UPF",
                     "gaussian", (3, 3, 3), None, 15.0, 8.0, 14, FCC, 1e-7, dict(beta=0.3)),
    "Si-paw": (fcc(5.431), SHIFTED, "pslibrary/Si.pbe-n-kjpaw_psl.1.0.0.UPF", "fixed",
               (2, 2, 2), None, 15.0, 8.0, 12, FCC, 1e-7, {}),
    "Fe-paw": (bcc(2.87), [[0.0, 0.0, 0.0]], "pslibrary/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
               "gaussian", (3, 3, 3), [2.5], 20.0, 8.0, 14, BCC, 3e-6,
               dict(beta=0.3, maxstep=300)),
}  # fmt: skip


def path_of(cell, sections):
    """4 k-points on each line of the path, and one without any symmetry."""
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    k, *_ = _paths.sample(_paths.given_path([sections]), 4, reciprocal)
    return np.vstack([k, [[0.13, 0.27, -0.41]]])


def field(case):
    """Our field and pw.x's, with its bands on the path of the case."""
    cell, positions, name, smearing, kgrid, magmoms, cutoff, dual, count, sections, _, options = CASES[case]  # fmt: skip
    element = case.split("-")[0]
    files = {element: espresso.pseudo(name)}
    pp = read_upf(files[element])
    kpoints, weights = _dft.monkhorst_pack(list(kgrid), [0.0, 0.0, 0.0], True)
    path = path_of(cell, sections)
    qe = espresso.run(
        cell, positions, [element] * len(positions), files, cutoff, kpoints, weights,
        smearing, SIGMA, conv_thr=1e-14, magmoms=magmoms,
        extra=f", ecutrho = {2 * dual * cutoff:.10f}", band_kpoints=path,
        band_count=count, **options,
    )  # fmt: skip
    settings = dict(
        smearing=smearing, sigma=SIGMA, functional="pbe_pw", grid=qe["grid"],
        smooth_grid=qe["smooth_grid"], density_cutoff=dual * cutoff, magmoms=magmoms,
        symmetry=False,
    )  # fmt: skip
    arguments = (cell, positions, [0] * len(positions), [pp], cutoff)
    r = scf.run(*arguments, kgrid=kgrid, density_tolerance=1e-14,
                energy_tolerance=1e-12, max_iterations=200, **settings)  # fmt: skip
    assert r["converged"]
    return arguments, settings, r, qe, path, count


_FIELDS = {}


def cached(case):
    if case not in _FIELDS:
        _FIELDS[case] = field(case)
    return _FIELDS[case]


def bands_at(arguments, settings, r, kpoints, count, becsum=True, **more):
    """One step at the density of r: count bands of the k-points, each to a
    residual norm of 1e-7, four more above them loosely."""
    options = dict(
        kpoints=kpoints, bands=count + 4, max_iterations=1, davidson_iterations=300,
        stress=False, gamma=False, start_density=r["density"],
        start_becsum=r.get("becsum") if becsum else None, first_tolerance=1e-7,
        first_held=count,
    )  # fmt: skip
    options.update(more)
    p = scf.run(*arguments, **settings, **options)
    assert p["unconverged"][0] == 0
    return p["eigenvalues"][..., :count]


@pytest.mark.parametrize("case", list(CASES))
def test_against_pw(case):
    arguments, settings, r, qe, path, count = cached(case)
    ours = bands_at(arguments, settings, r, path, count)
    reference = qe["band_eigenvalues"]
    assert ours.shape == reference.shape
    worst = np.abs(ours - reference).max()
    print(f"\n{case}: max |e - e_QE| = {worst:.2e} Ha = {worst * 27.211386245988:.2e} eV "
          f"over {reference.size} eigenvalues, {count} bands")  # fmt: skip
    assert worst < CASES[case][10]


@pytest.mark.parametrize("case", ["Si-paw", "Fe-paw"])
def test_paw_takes_the_occupations_of_the_field(case):
    # without them the one-center strengths are those of the free atom, and
    # the bands are not those of the density
    arguments, settings, r, qe, path, count = cached(case)
    ours = bands_at(arguments, settings, r, path, count, becsum=False)
    worst = np.abs(ours - qe["band_eigenvalues"]).max()
    print(
        f"\n{case} with the occupations of the free atom: max |e - e_QE| = {worst:.2e} Ha"
    )
    assert worst > 1e-4


@pytest.mark.parametrize("case", ["Si", "Al", "Fe", "Ni-ultrasoft", "Si-paw"])
def test_the_mesh_of_the_field_again(case):
    arguments, settings, r, _, _, _ = cached(case)
    count = r["bands"] - 2
    again = bands_at(arguments, settings, r, r["kpoints"], count)
    held = r["occupations"][..., :count] > 1e-6
    # to the residuals that the last step of the field left
    np.testing.assert_allclose(
        again[held], r["eigenvalues"][..., :count][held], atol=1e-7
    )


def test_equivalent_k_points():
    cell, _, name, _, _, _, cutoff, _, _, _, _, _ = CASES["Si"]
    pp = read_upf(espresso.pseudo(name))
    arguments = (cell, [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], [0, 0], [pp], cutoff)
    settings = dict(smearing="fixed", functional="pbe_pw", symmetry=False)
    r = scf.run(*arguments, kgrid=(2, 2, 2), density_tolerance=1e-13,
                energy_tolerance=1e-11, **settings)  # fmt: skip
    settings.update(grid=r["grid"])
    k = np.array([0.13, 0.27, -0.41])
    # time reversal; a reciprocal lattice vector; the threefold axis [111],
    # which turns the vectors of the cell into one another
    others = [k, -k, k + [1, -2, 0], np.roll(k, 1), np.roll(k, 2)]
    e = bands_at(arguments, settings, r, np.array(others), 10)
    for other in e[1:]:
        np.testing.assert_allclose(other, e[0], atol=1e-9)
    # more bands leave the lower ones as they are
    more = bands_at(arguments, settings, r, k[None], 14)
    np.testing.assert_allclose(more[0, :10], e[0], atol=1e-9)
    # the weights of the k-points do not enter the bands
    weighted = bands_at(arguments, settings, r, np.array(others), 10,
                        weights=[0.6, 0.1, 0.1, 0.1, 0.1])  # fmt: skip
    np.testing.assert_allclose(weighted, e, atol=1e-9)


def test_bands_left_open_are_counted():
    arguments, settings, r, _, path, count = cached("Si")
    p = scf.run(
        *arguments, **settings, kpoints=path, bands=count, max_iterations=1,
        davidson_iterations=1, stress=False, gamma=False, start_density=r["density"],
        first_tolerance=1e-12,
    )  # fmt: skip
    assert 0 < p["unconverged"][0] <= count * len(path)
    assert r["unconverged"].shape == r["davidson"].shape
    assert r["unconverged"][-1] == 0


def test_errors():
    arguments, settings, r, _, path, _ = cached("Si")
    with pytest.raises(ValueError, match="kpoints are"):
        scf.run(*arguments, **settings, kpoints=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="adding up to one"):
        scf.run(*arguments, **settings, kpoints=path, weights=np.ones(len(path)))
    with pytest.raises(ValueError, match="weights come with kpoints"):
        scf.run(*arguments, **settings, weights=[1.0])
    with pytest.raises(ValueError, match="start_becsum takes the occupations"):
        scf.run(*arguments, **settings, start_becsum=np.zeros((1, 3)))
    arguments, settings, r, _, _, _ = cached("Si-paw")
    with pytest.raises(ValueError, match="start_becsum takes the occupations"):
        scf.run(*arguments, **settings, start_becsum=r["becsum"][:, :-1])
