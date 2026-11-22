# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ultrasoft pseudopotentials (pslibrary 1.0.0, made by ld1.x) against Quantum
ESPRESSO run live, on one grid (ecutrho = 4 ecutwfc) and on two (8 ecutwfc):
silicon, aluminium with a smearing, iron with spin. The energy and its terms,
the occupied eigenvalues, the Fermi level, the magnetization and the forces;
the forces also against differences of the energy."""

import numpy as np
import pytest

from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf

from dft import espresso

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
TIGHT = dict(functional="pbe_pw", density_tolerance=1e-14, energy_tolerance=1e-12)


def fcc(a):
    h = a / 2 / BOHR
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


CASES = {
    "Si": (fcc(5.431), [[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]], "pslibrary/Si.pbe-n-rrkjus_psl.1.0.0.UPF",
           "fixed", (3, 3, 3), None),
    "Al": (fcc(4.05), [[0.0, 0.0, 0.0]], "pslibrary/Al.pbe-n-rrkjus_psl.1.0.0.UPF", "gaussian", (4, 4, 4), None),
    "Fe": (np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * 2.87 / 2 / BOHR, [[0.0, 0.0, 0.0]],
           "pslibrary/Fe.pbe-spn-rrkjus_psl.1.0.0.UPF", "gaussian", (3, 3, 3), [2.5]),
}  # fmt: skip


@pytest.mark.parametrize(
    ("case", "dual"), [("Si", 4.0), ("Si", 8.0), ("Al", 8.0), ("Fe", 8.0)]
)
def test_against_qe(case, dual):
    cell, positions, name, smearing, kgrid, magmoms = CASES[case]
    cutoff = 15.0
    files = {case: espresso.pseudo(name)}
    pp = read_upf(files[case])
    kpoints, weights = _dft.monkhorst_pack(list(kgrid), [0.0, 0.0, 0.0], True)
    qe = espresso.run(
        cell, positions, [case] * len(positions), files, cutoff, kpoints, weights, smearing,
        0.01, conv_thr=1e-14, magmoms=magmoms, beta=0.3, maxstep=400,
        extra=f", ecutrho = {2 * dual * cutoff:.10f}",
    )  # fmt: skip
    r = scf.run(
        cell, positions, [0] * len(positions), [pp], cutoff, kgrid=kgrid, smearing=smearing,
        sigma=0.01, grid=qe["grid"], smooth_grid=qe["smooth_grid"],
        density_cutoff=dual * cutoff, magmoms=magmoms, **TIGHT,
    )  # fmt: skip
    assert r["converged"] and tuple(r["smooth_grid"]) == qe["smooth_grid"]
    n = len(positions)
    if magmoms is None:
        for key in ["energy", "band", "hartree", "xc", "ewald"]:
            assert r[key] == pytest.approx(qe[key], abs=1e-7 * n), key
        eigen = 1e-7
    else:
        # QE's spin-polarized PBE takes a rounded gamma (test_spin.py)
        gamma = espresso.rounded_gamma(r, cell, positions, [0], [pp], dual * cutoff)
        assert r["energy"] + gamma == pytest.approx(qe["energy"], abs=1e-7 * n)
        assert r["magnetization"] == pytest.approx(qe["magnetization"], abs=1e-6)
        eigen = 1e-6
    width = qe["eigenvalues"].shape[-1]
    held = r["occupations"][..., :width] > 1e-6
    np.testing.assert_allclose(
        r["eigenvalues"][..., :width][held], qe["eigenvalues"][held], atol=eigen
    )
    assert r["fermi"] == pytest.approx(qe["fermi"], abs=eigen)
    np.testing.assert_allclose(r["forces"], qe["forces"], atol=1e-7)


def test_forces_against_differences():
    cell, positions, name, _, _, _ = CASES["Si"]
    positions = np.array(positions)
    pp = read_upf(espresso.pseudo(name))

    def run(x):
        return scf.run(cell, x, [0, 0], [pp], 12.0, kgrid=(2, 2, 2), smearing="fixed",
                       density_cutoff=96.0, **TIGHT)  # fmt: skip

    r = run(positions)
    raw = r["forces"] + r["net_force"]
    step = 2e-3
    for d in range(3):
        energies = []
        for sign in (1, -1):
            moved = positions.copy()
            moved[0] += sign * step * np.linalg.inv(cell)[d]
            energies.append(run(moved)["energy"])
        assert raw[0, d] == pytest.approx(
            -(energies[0] - energies[1]) / (2 * step), abs=2e-7
        )
