# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""PAW against pw.x 7.6 run live on the same pslibrary datasets, cutoffs and
grids, the one-center terms on the GPU: the energy, its plane-wave parts, the eigenvalues, the forces and the
stress of aluminium, silicon of two displaced atoms and iron with spin. QE's
spin-polarized PBE rounds gamma; the energy of iron is compared with the
first-order change that makes, of the plane-wave and the one-center
densities."""

import numpy as np
import pytest

from mdapy.calculators._dft import paw, scf

from dft import espresso, paw_reference
from dft.test_paw import dataset, xc_of

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
KBAR = 294210.1  # hartree / bohr^3 in kbar
NAME = "pbe_pw"


def onecenter(data, becsum):
    energies, d = paw_reference.onecenter(data, becsum, NAME, xc_of(NAME))
    return energies["total"], d


FILES = {
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
}
CASES = {
    "Al": (4.05, "fcc", [[0.01, 0.0, 0.0]], "gaussian", (3, 3, 3), None, 15.0),
    "Si": (
        5.431,
        "fcc",
        [[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]],
        "fixed",
        (2, 2, 2),
        None,
        15.0,
    ),
    "Fe": (2.87, "bcc", [[0.0, 0.0, 0.0]], "gaussian", (3, 3, 3), [2.5], 20.0),
}


@pytest.mark.parametrize(
    ("element", "engine"),
    [("Al", "gpu"), ("Al", "reference"), ("Si", "gpu"), ("Fe", "gpu")],
)
def test_against_pw(element, engine):
    """The one-center terms of the GPU, or of the NumPy reference through the
    callable of the SCF."""
    a, lattice, positions, smearing, kgrid, magmoms, cutoff = CASES[element]
    h = a / 2 / BOHR
    if lattice == "fcc":
        cell = np.array([[0, h, h], [h, 0, h], [h, h, 0]])
    else:
        cell = np.array([[-h, h, h], [h, -h, h], [h, h, -h]])
    pp = dataset(element)
    dual = 8.0
    kpoints, weights = _dft.monkhorst_pack(list(kgrid), [0.0, 0.0, 0.0], True)
    symbols = [element] * len(positions)
    files = {element: espresso.pseudo("pslibrary/" + FILES[element])}
    qe = espresso.run(
        cell, positions, symbols, files, cutoff, kpoints, weights, smearing, 0.01,
        conv_thr=1e-13, magmoms=magmoms, beta=0.3, maxstep=300,
        extra=f", ecutrho = {2 * dual * cutoff:.10f}",
    )  # fmt: skip
    kinds = [0] * len(positions)
    r = scf.run(
        cell, positions, kinds, [pp], cutoff, kgrid=kgrid, smearing=smearing,
        sigma=0.01, functional=NAME, grid=qe["grid"], smooth_grid=qe["smooth_grid"],
        density_cutoff=dual * cutoff, density_tolerance=1e-13, energy_tolerance=1e-11,
        magmoms=magmoms, onecenter=onecenter if engine == "reference" else None,
        max_iterations=150,
    )  # fmt: skip
    assert r["converged"]
    energy = r["energy"]
    if magmoms is not None:
        energy += espresso.rounded_gamma(r, cell, positions, kinds, [pp], dual * cutoff)
        energy += paw_reference.rounded_gamma(paw.onecenter(pp, True), r["becsum"])
        assert r["magnetization"] == pytest.approx(qe["magnetization"], abs=1e-6)
    assert energy == pytest.approx(qe["energy"], abs=1e-8 * len(positions))
    for key in ("hartree", "xc", "ewald", "entropy"):
        assert r[key] == pytest.approx(qe[key] or 0.0, abs=2e-6)
    # E^1 of the input occupations, first order in their residual
    assert r["onecenter"] == pytest.approx(qe["onecenter"], abs=1e-5)
    held = r["occupations"][..., : qe["eigenvalues"].shape[-1]] > 1e-3
    gap = r["eigenvalues"][..., : qe["eigenvalues"].shape[-1]] - qe["eigenvalues"]
    assert np.abs(gap[held]).max() < 1e-7
    np.testing.assert_allclose(r["forces"], qe["forces"], atol=1e-7)
    volume = abs(np.linalg.det(cell))
    np.testing.assert_allclose(
        -r["strain_derivative"] / volume * KBAR, qe["stress"] * KBAR, atol=0.01
    )
