# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The forces on the ions: central differences of the energy (the free energy
with a smearing) of displaced atoms, without and with spin, and Quantum
ESPRESSO run live; their mean is taken out."""

import numpy as np
import pytest

from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf

from dft import espresso

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
TIGHT = dict(density_tolerance=1e-15, energy_tolerance=1e-13, functional="pbe_pw")


def silicon():
    h = 5.431 / 2 / BOHR
    cell = np.array([[0, h, h], [h, 0, h], [h, h, 0]])
    return cell, np.array([[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]])


def central(cell, positions, atom, run, step=2e-3):
    """-dE/dx of an atom along x, y and z by central differences, bohr."""
    out = []
    for d in range(3):
        energies = []
        for sign in (1, -1):
            moved = positions.copy()
            moved[atom] += sign * step * np.linalg.inv(cell)[d]
            energies.append(run(moved)["energy"])
        out.append(-(energies[0] - energies[1]) / (2 * step))
    return np.array(out)


def test_silicon():
    cell, positions = silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))

    def run(x):
        return scf.run(cell, x, [0, 0], [pp], 12.0, kgrid=(2, 2, 2), **TIGHT)

    r = run(positions)
    raw = r["forces"] + r["net_force"]
    np.testing.assert_allclose(raw[0], central(cell, positions, 0, run), atol=1e-7)
    np.testing.assert_allclose(r["forces"].sum(axis=0), 0.0, atol=1e-14)
    assert np.abs(r["net_force"]).max() < 1e-6
    kpoints, weights = _dft.monkhorst_pack([2, 2, 2], [0.0, 0.0, 0.0], True)
    qe = espresso.run(cell, positions, ["Si", "Si"], {"Si": espresso.pseudo("Si_std.upf")},
                      12.0, kpoints, weights, conv_thr=1e-14)  # fmt: skip
    np.testing.assert_allclose(r["forces"], qe["forces"], atol=1e-7)


def test_aluminium_with_a_smearing():
    a = 4.05 / BOHR
    cell = np.diag([a, a, a])
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.47, 0.02], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    )
    pp = read_upf(espresso.pseudo("Al_std.upf"))

    def run(x):
        return scf.run(cell, x, [0] * 4, [pp], 10.0, kgrid=(2, 2, 2),
                       smearing="marzari-vanderbilt", sigma=0.01, **TIGHT)  # fmt: skip

    r = run(positions)
    raw = r["forces"] + r["net_force"]
    np.testing.assert_allclose(raw[1], central(cell, positions, 1, run), atol=2e-7)


def test_iron_with_spin():
    a = 2.87 / BOHR
    cell = np.diag([a, a, a])
    positions = np.array([[0.0, 0.0, 0.0], [0.52, 0.49, 0.5]])
    pp = read_upf(espresso.pseudo("Fe_std.upf"))

    def run(x):
        return scf.run(cell, x, [0, 0], [pp], 20.0, kgrid=(2, 2, 2), smearing="gaussian",
                       sigma=0.01, magmoms=[2.5, 2.5], **TIGHT)  # fmt: skip

    r = run(positions)
    assert r["magnetization"] > 3.0
    raw = r["forces"] + r["net_force"]
    np.testing.assert_allclose(raw[1], central(cell, positions, 1, run), atol=1e-6)


def test_the_correction_of_a_loose_field():
    """QE's correction for the change of the input density with the atoms
    brings the forces of a loosely converged field nearer to those of a tight
    one."""
    a = 5.431 / BOHR
    base = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    positions = np.vstack([base, base + 0.25]) + np.random.default_rng(1).normal(
        0, 0.01, (8, 3)
    )
    pp = read_upf(espresso.pseudo("Si_std.upf"))

    def run(tolerance, correct=True):
        return scf.run(np.diag([a] * 3), positions, [0] * 8, [pp], 15.0, kgrid=(2, 2, 2),
                       density_tolerance=tolerance, energy_tolerance=tolerance,
                       correct_forces=correct)["forces"]  # fmt: skip

    tight = run(1e-15)
    corrected = np.abs(run(1e-5 / 27.2) - tight).max()
    plain = np.abs(run(1e-5 / 27.2, False) - tight).max()
    assert corrected < 0.2 * plain
