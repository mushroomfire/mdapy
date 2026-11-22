# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""dE/d epsilon of a homogeneous strain at fixed Miller indices: against
central differences of the energy of strained cells whose plane waves are
chosen by the cell at rest (the same Miller indices and FFT grid), without and
with spin, and against Quantum ESPRESSO run live for silicon, aluminium and
iron in sheared cells."""

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
SHEAR = np.array([[1.01, 0.02, 0.0], [0.02, 0.98, 0.01], [0.0, 0.01, 1.0]])
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


def fcc(a):
    h = a / 2 / BOHR
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


def differences(cell, positions, pp, cutoff, grid, step=1e-4, **options):
    """dE/d epsilon by central differences at the basis of cell."""

    def energy(strain):
        return scf.run(cell @ (np.eye(3) + strain), positions, [0] * len(positions), [pp],
                       cutoff, grid=grid, basis_cell=cell, stress=False, **options,
                       **TIGHT)["energy"]  # fmt: skip

    out = np.zeros((3, 3))
    for a, b in PAIRS:
        e = np.zeros((3, 3))
        e[a, b] += 0.5 * step
        e[b, a] += 0.5 * step
        out[a, b] = out[b, a] = (energy(e) - energy(-e)) / (2 * step)
    return out


def test_silicon_against_its_differences():
    cell = fcc(5.431) @ SHEAR
    positions = np.array([[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]])
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    r = scf.run(cell, positions, [0, 0], [pp], 12.0, kgrid=(2, 2, 2), **TIGHT)
    np.testing.assert_allclose(
        r["strain_derivative"], r["strain_derivative"].T, atol=1e-14
    )
    fd = differences(cell, positions, pp, 12.0, r["grid"], kgrid=(2, 2, 2))
    np.testing.assert_allclose(r["strain_derivative"], fd, atol=5e-8)


def test_iron_with_spin_against_its_differences():
    a = 2.87 / BOHR
    cell = np.diag([a, a, a]) @ SHEAR
    positions = np.array([[0.0, 0.0, 0.0], [0.52, 0.49, 0.5]])
    pp = read_upf(espresso.pseudo("Fe_std.upf"))
    options = dict(kgrid=(2, 2, 2), smearing="gaussian", sigma=0.01, magmoms=[2.5, 2.5])
    r = scf.run(cell, positions, [0, 0], [pp], 16.0, **options, **TIGHT)
    fd = differences(cell, positions, pp, 16.0, r["grid"], **options)
    np.testing.assert_allclose(r["strain_derivative"], fd, atol=2e-7)


@pytest.mark.parametrize("case", ["Si", "Al", "Fe", "Fe with spin"])
def test_against_qe(case):
    element, spin = case.split()[0], case.endswith("spin")
    magmoms, smearing = ([2.5] if spin else None), "gaussian"
    if element == "Si":
        cell, cutoff, kgrid, smearing = fcc(5.431) @ SHEAR, 15.0, 3, "fixed"
        positions = [[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]]
    elif element == "Al":
        cell, positions, cutoff, kgrid = fcc(4.05) @ SHEAR, [[0.0, 0.0, 0.0]], 12.0, 4
    else:
        h = 2.87 / 2 / BOHR
        cell = np.array([[-h, h, h], [h, -h, h], [h, h, -h]]) @ SHEAR
        positions, cutoff, kgrid = [[0.0, 0.0, 0.0]], 20.0, 3
    symbols = [element] * len(positions)
    files = {element: espresso.pseudo(f"{element}_std.upf")}
    kpoints, weights = _dft.monkhorst_pack([kgrid] * 3, [0.0, 0.0, 0.0], True)
    qe = espresso.run(cell, positions, symbols, files, cutoff, kpoints, weights, smearing,
                      0.01, conv_thr=1e-14, magmoms=magmoms, beta=0.3)  # fmt: skip
    r = scf.run(cell, positions, [0] * len(positions), [read_upf(files[element])], cutoff,
                kgrid=(kgrid,) * 3, smearing=smearing, sigma=0.01, grid=qe["grid"],
                magmoms=magmoms, **TIGHT)  # fmt: skip
    # QE's stress is -(1/Omega) dE/d epsilon. With spin, QE's PBE correlation
    # takes a rounded gamma, which moves its gradient term by about 1e-5 (0.02
    # kbar here); test_iron_with_spin_against_its_differences checks ours
    stress = -r["strain_derivative"] / abs(np.linalg.det(cell))
    np.testing.assert_allclose(stress, qe["stress"], atol=1e-7 if spin else 1e-8)
