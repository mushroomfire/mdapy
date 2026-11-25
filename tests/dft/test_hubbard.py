# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""DFT+U of Dudarev on the pseudo-orbitals (Quantum ESPRESSO's atomic
projectors) against pw.x run live: antiferromagnetic NiO of the second kind
with U = 6 eV on the 3d of nickel, and without spin two atoms of copper with
U = 4 eV on its 3d, their atoms displaced: the energy, the Hubbard energy, the
traces of the occupations, the moments, the forces, the stress and the
occupied eigenvalues. U = 0 is the plain run; the symmetry and the real bands
at Gamma change nothing."""

import os

import numpy as np
import pytest

import mdapy
from mdapy.calculators import DFT
from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf
from mdapy.errors import InvalidParameter

from dft import espresso, gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
EV = 1 / 27.211386245988  # hartree
U = 6.0 * EV
CUTOFF = 25.0
# the rhombohedral cell of NiO doubled along [111], a = 4.17 Å, the two
# nickel atoms of opposite moments
CELL = 4.17 / BOHR * np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
SITES = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
DISPLACED = SITES + np.random.default_rng(3).uniform(-0.01, 0.01, (4, 3))
SYMBOLS = ["Ni", "Ni", "O", "O"]
KINDS = [0, 0, 1, 1]
MOMENTS = [2.0, -2.0, 0.0, 0.0]
# fcc copper, a = 3.61 Å, its primitive cell doubled along the first vector
COPPER = 3.61 / 2 / BOHR * np.array([[0, 2, 2], [1, 0, 1], [1, 1, 0]])
COPPER_SITES = np.array([[0, 0, 0], [0.5, 0, 0]]) + np.random.default_rng(4).uniform(
    -0.01, 0.01, (2, 3)
)


def pseudos():
    files = {s: espresso.pseudo(f"{s}_std.upf") for s in ("Ni", "O")}
    return files, [read_upf(files["Ni"]), read_upf(files["O"])]


def ours(positions, grid=None, magmoms=MOMENTS, u=U, kgrid=2, **options):
    _, pps = pseudos()
    return scf.run(
        CELL, positions, KINDS, pps, CUTOFF, kgrid=(kgrid,) * 3, smearing="gaussian",
        sigma=0.005, functional="pbe_pw", grid=grid, magmoms=magmoms,
        density_tolerance=1e-13, energy_tolerance=1e-11, max_iterations=300,
        hubbard=None if u is None else {0: ("3D", u)}, **options,
    )  # fmt: skip


def copper(u=4.0 * EV, kgrid=2, **options):
    pp = read_upf(espresso.pseudo("Cu_std.upf"))
    return scf.run(
        COPPER, COPPER_SITES, [0, 0], [pp], CUTOFF, kgrid=(kgrid,) * 3, smearing="gaussian",
        sigma=0.005, functional="pbe_pw", density_tolerance=1e-13, energy_tolerance=1e-11,
        max_iterations=200, hubbard=None if u is None else {0: ("3D", u)}, **options,
    )  # fmt: skip


@pytest.mark.parametrize("spin", [True, False])
def test_against_qe(spin):
    """NiO with spin, copper without."""
    if spin:
        files, pps = pseudos()
        cell, positions, symbols, kinds, u = CELL, DISPLACED, SYMBOLS, KINDS, U
        hubbard = {"Ni": ("3d", U)}
    else:
        files = {"Cu": espresso.pseudo("Cu_std.upf")}
        pps = [read_upf(files["Cu"])]
        cell, positions, symbols, kinds, u = (
            COPPER,
            COPPER_SITES,
            ["Cu", "Cu"],
            [0, 0],
            4.0 * EV,
        )
        hubbard = {"Cu": ("3d", u)}
    magmoms = MOMENTS if spin else None
    kpoints, weights = _dft.monkhorst_pack([2, 2, 2], [0.0, 0.0, 0.0], True)
    qe = espresso.run(
        cell, positions, symbols, files, CUTOFF, kpoints, weights, "gaussian", 0.005,
        conv_thr=1e-11, magmoms=magmoms, beta=0.3, maxstep=400, hubbard=hubbard,
    )  # fmt: skip
    if spin:
        r = ours(DISPLACED, grid=qe["grid"], symmetry=False)
    else:
        r = copper(u, grid=qe["grid"], symmetry=False)
    assert r["converged"]
    n = len(symbols)
    # QE's spin-polarized PBE takes a rounded gamma (test_spin.py)
    gamma = (
        espresso.rounded_gamma(r, cell, positions, kinds, pps, 4 * CUTOFF)
        if spin
        else 0.0
    )
    assert r["energy"] + gamma == pytest.approx(qe["energy"], abs=1e-7 * n)
    # QE prints the Hubbard energy and the traces to 8 and 5 digits
    assert r["hubbard_energy"] == pytest.approx(qe["hubbard_energy"], abs=1e-7)
    traces = np.array([[np.trace(x) for x in m] for m in r["hubbard_occupations"]])
    # with spin up and down, without the total of both
    expected = qe["hubbard_traces"][:, :2] if spin else qe["hubbard_traces"][:, :1] / 2
    np.testing.assert_allclose(traces, expected, atol=1e-5)
    if spin:
        assert r["magnetization"] == pytest.approx(qe["magnetization"], abs=1e-6)
        assert r["absolute_magnetization"] == pytest.approx(
            qe["absolute_magnetization"], abs=1e-6
        )
    np.testing.assert_allclose(r["forces"], qe["forces"], atol=1e-6)
    stress = -r["strain_derivative"] / abs(np.linalg.det(cell))
    np.testing.assert_allclose(stress, qe["stress"], atol=1e-7)
    width = qe["eigenvalues"].shape[-1]
    held = r["occupations"][..., :width] > 1e-6
    np.testing.assert_allclose(
        r["eigenvalues"][..., :width][held], qe["eigenvalues"][held], atol=2e-6
    )


def test_no_u_is_the_plain_run():
    with_zero = copper(0.0, symmetry=False)
    plain = copper(None, symmetry=False)
    assert with_zero["energy"] == pytest.approx(plain["energy"], abs=1e-10)
    assert with_zero["hubbard_energy"] == 0.0
    np.testing.assert_allclose(with_zero["forces"], plain["forces"], atol=1e-9)
    np.testing.assert_allclose(
        with_zero["strain_derivative"], plain["strain_derivative"], atol=1e-9
    )


def test_symmetry_changes_nothing():
    """The crystal of NiO, whose operations keep the moments, with and without
    them."""
    on, off = (ours(SITES, symmetry=s) for s in (True, False))
    assert on["operations"] > 1
    assert on["energy"] == pytest.approx(off["energy"], abs=1e-9)
    # E_U, not stationary, is good to the first order of the density: 1e-8
    assert on["hubbard_energy"] == pytest.approx(off["hubbard_energy"], abs=1e-7)
    for a, b in zip(on["hubbard_occupations"], off["hubbard_occupations"]):
        np.testing.assert_allclose(
            np.linalg.eigvalsh(a), np.linalg.eigvalsh(b), atol=1e-6
        )


def test_real_bands_at_gamma():
    real, complex_ = (copper(kgrid=1, gamma=g, symmetry=False) for g in (True, False))
    assert real["gamma"] and not complex_["gamma"]
    assert real["energy"] == pytest.approx(complex_["energy"], abs=1e-9)
    np.testing.assert_allclose(real["forces"], complex_["forces"], atol=1e-8)
    np.testing.assert_allclose(
        real["strain_derivative"], complex_["strain_derivative"], atol=1e-8
    )


def test_errors():
    _, pps = pseudos()
    with pytest.raises(ValueError, match="no single pseudo-orbital 4F"):
        scf.run(CELL, SITES, KINDS, pps, 10.0, hubbard={0: ("4F", U)})
    ultrasoft = read_upf(espresso.pseudo("pslibrary/Fe.pbe-spn-rrkjus_psl.1.0.0.UPF"))
    with pytest.raises(ValueError, match="norm-conserving pseudopotentials only"):
        scf.run(
            np.eye(3) * 5.4, [[0, 0, 0]], [0], [ultrasoft], 10.0, hubbard={0: ("3D", U)}
        )


def test_the_calculator():
    """DFT with U in eV: the energy of scf.run, the Hubbard energy and the
    occupations in last; a restart from the last run gives the same."""
    pseudos()
    folder = os.environ["MDAPY_PSEUDO_DIR"]
    nio = mdapy.System(SITES @ CELL * BOHR, mdapy.Box(CELL * BOHR), elements=SYMBOLS)
    dft = DFT(CUTOFF / EV, pseudopotentials=folder, kpoints=(2, 2, 2), sigma=0.005 / EV,
              magmoms=MOMENTS, hubbard={"Ni": ("3d", 6.0)}, functional="pbe_pw",
              tolerance=1e-9, restart=True)  # fmt: skip
    first = dft.compute(nio).energy
    reference = ours(SITES)
    assert first == pytest.approx(reference["energy"] / EV, abs=1e-6)
    # E_U to the first order of the density of a tolerance of 1e-9 eV
    assert dft.last["hubbard_energy"] == pytest.approx(
        reference["hubbard_energy"] / EV, abs=1e-4
    )
    assert len(dft.last["hubbard_occupations"]) == 2
    assert dft.last["provenance"]["hubbard"] == {"Ni": ("3d", 6.0)}
    again = dft.compute(nio).energy
    assert dft.last["restarted"] and again == pytest.approx(first, abs=1e-6)
    for wrong in ({"Ni": "3d"}, {"Ni": ("3d", -1.0)}, {"Ni": (3, 6.0)}):
        with pytest.raises(InvalidParameter, match="hubbard maps an element"):
            DFT(300.0, pseudopotentials=folder, hubbard=wrong)
