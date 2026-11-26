# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The sums of the electronic density of states against dos.x 7.6 of Quantum
ESPRESSO run live, on the energies of its own calculation = "nscf" on a mesh
through Γ (every k-point, without symmetry): the linear tetrahedron method
(bz_sum = "tetrahedra_lin") and Gaussians (QE's degauss is sqrt(2) sigma),
silicon without spin and iron with spin. dos.x writes 4 significant digits.
No GPU: the energies are QE's; ours at the same k-points are in
test_bands.py."""

import numpy as np
import pytest

from mdapy.properties import _electronic
from mdapy.properties._phonons import _mesh

from dft import espresso

BOHR = 0.529177210903
HARTREE = 27.211386245988
RY = HARTREE / 2
DEGAUSS = 0.02  # Ry


def fcc(a):
    h = a / 2 / BOHR
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


def bcc(a):
    h = a / 2 / BOHR
    return np.array([[-h, h, h], [h, -h, h], [h, h, -h]])


# cell, positions, file, smearing (hartree), mesh of the field, moments,
# cutoff (hartree), mesh of the density of states, bands, options of pw.x
CASES = {
    "Si": (fcc(5.431), [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], "Si_std.upf",
           ("fixed", 0.0), 3, None, 12.0, 6, 8, {}),
    "Fe": (bcc(2.87), [[0.0, 0.0, 0.0]], "Fe_std.upf", ("gaussian", 0.01), 3, [2.5],
           25.0, 5, 12, dict(beta=0.3)),
}  # fmt: skip


@pytest.mark.parametrize("case", list(CASES))
def test_against_dos_x(case):
    cell, positions, name, (smearing, sigma), field, magmoms, cutoff, mesh, count, options = CASES[case]  # fmt: skip
    element = case
    files = {element: espresso.pseudo(name)}
    kpoints = _mesh(field)
    weights = np.full(len(kpoints), 1.0 / len(kpoints))
    # every band inside; the energies of dos.x are Emin + n DeltaE, written
    # with 3 decimals, and both sums take the same bands
    window = (-120.0, 60.0)
    settings = (
        f"bz_sum = 'tetrahedra_lin', Emin = {window[0]}, Emax = {window[1]}, DeltaE = 0.01",
        f"bz_sum = 'smearing', degauss = {DEGAUSS}, ngauss = 0, "
        f"Emin = {window[0]}, Emax = {window[1]}, DeltaE = 0.01",
    )
    qe = espresso.run(
        cell, positions, [element] * len(positions), files, cutoff, kpoints, weights,
        smearing, sigma, conv_thr=1e-12, magmoms=magmoms, dos_mesh=(mesh,) * 3,
        band_count=count, dos_inputs=settings, **options,
    )  # fmt: skip
    e = qe["mesh_eigenvalues"] * HARTREE
    if e.ndim == 2:
        e = e[None]
    spins = len(e)
    assert e.shape == (spins, mesh**3, count)
    assert window[0] < e.min() and e.max() < window[1]
    counts = np.array([mesh] * 3)
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T  # 1/bohr
    tetrahedra, gaussians = qe["dos"]
    for table, ours in (
        (tetrahedra, _electronic._tetrahedron_sum(e, counts, reciprocal, tetrahedra[:, 0])),
        (gaussians, _electronic._gaussian_sum(
            e, np.full(mesh**3, 1.0 / mesh**3), gaussians[:, 0], DEGAUSS * RY / np.sqrt(2))),
    ):  # fmt: skip
        ours = ours * (2 / spins)
        reference = table[:, 1 : 1 + spins].T
        assert ours.shape == reference.shape
        worst = np.abs(ours - reference).max()
        print(f"\n{case}: max |dos - dos.x| = {worst:.2e} states/eV of "
              f"{reference.max():.2f}")  # fmt: skip
        np.testing.assert_allclose(
            ours, reference, rtol=1e-3, atol=2e-4 * reference.max()
        )
