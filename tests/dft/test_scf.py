# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The self-consistent field on the GPU against Quantum ESPRESSO run live with
the same PseudoDojo pseudopotentials, cutoff and k-points: silicon and diamond
with fixed occupations, aluminium with each smearing. The total energy, its
terms, the eigenvalues and the Fermi level."""

import math

import numpy as np
import pytest

from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf

from dft import espresso

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903


def fcc(a):
    h = a / 2 / BOHR
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


def compare(cell, positions, symbols, cutoff, kgrid, smearing="fixed", sigma=0.0,
            functional="pbe_pw", extra=""):  # fmt: skip
    """Our SCF and QE's: the terms of the energy to 1e-7 hartree per atom (QE's
    Ewald sum is good to about 1e-7 rydberg), the eigenvalues of the bands that
    hold electrons and the Fermi level to 1e-7 hartree, the same plane waves
    and grid. extra goes into QE's &system."""
    files = {s: espresso.pseudo(f"{s}_std.upf") for s in dict.fromkeys(symbols)}
    kinds = [list(files).index(s) for s in symbols]
    pseudos = [read_upf(f) for f in files.values()]
    kpoints, weights = _dft.monkhorst_pack(list(kgrid), [0.0, 0.0, 0.0], True)
    qe = espresso.run(
        cell, positions, symbols, files, cutoff, kpoints, weights, smearing, sigma,
        conv_thr=1e-14, extra=extra,
    )  # fmt: skip
    # on QE's grid, which ours equals but is given here to be sure
    ours = scf.run(
        cell, positions, kinds, pseudos, cutoff, kgrid=kgrid, smearing=smearing,
        sigma=sigma, functional=functional, grid=qe["grid"], density_tolerance=1e-14,
        energy_tolerance=1e-12, symmetry=False
    )  # fmt: skip
    assert ours["converged"]
    assert [len(_dft.sphere(cell, list(k), cutoff, ours["grid"])[0])
            for k in kpoints] == qe["plane_waves"]  # fmt: skip
    n = len(symbols)
    for key in ["energy", "band", "one_electron", "hartree", "xc", "ewald"]:
        assert ours[key] == pytest.approx(qe[key], abs=1e-7 * n), key
    if smearing != "fixed":
        assert ours["entropy"] == pytest.approx(qe["entropy"], abs=1e-7 * n)
    # the bands that hold electrons; the empty ones are converged loosely
    width = qe["eigenvalues"].shape[1]
    held = ours["occupations"][:, :width] > 1e-6
    np.testing.assert_allclose(
        ours["eigenvalues"][:, :width][held], qe["eigenvalues"][held], atol=1e-7
    )
    assert ours["fermi"] == pytest.approx(qe["fermi"], abs=1e-7)
    return ours


def test_silicon():
    r = compare(
        fcc(5.431), [[0, 0, 0], [0.27, 0.25, 0.25]], ["Si", "Si"], 20.0, (4, 4, 4)
    )
    assert r["iterations"] <= 15


def test_silicon_pbesol():
    """PBEsol on the PBE pseudopotential, as QE's input_dft = 'PBESOL'."""
    compare(fcc(5.431), [[0, 0, 0], [0.27, 0.25, 0.25]], ["Si", "Si"], 20.0, (4, 4, 4),
            functional="pbesol_pw", extra=", input_dft = 'PBESOL'")  # fmt: skip


def test_diamond():
    compare(fcc(3.567), [[0, 0, 0], [0.25, 0.25, 0.25]], ["C", "C"], 25.0, (3, 3, 3))


@pytest.mark.parametrize(
    "smearing", ["gaussian", "methfessel-paxton", "marzari-vanderbilt", "fermi-dirac"]
)
def test_aluminium(smearing):
    compare(fcc(4.05), [[0, 0, 0]], ["Al"], 15.0, (6, 6, 6), smearing, 0.01)


def test_the_bands_by_default():
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    r = scf.run(fcc(5.431), [[0, 0, 0], [0.25, 0.25, 0.25]], [0, 0], [pp], 8.0,
                density_tolerance=1e-8, symmetry=False)  # fmt: skip
    assert r["bands"] == 8 and r["converged"]
    assert math.isclose(r["occupations"][:, :4].sum(), 4.0)
    r = scf.run(fcc(5.431), [[0, 0, 0], [0.25, 0.25, 0.25]], [0, 0], [pp], 8.0,
                smearing="gaussian", sigma=0.01, density_tolerance=1e-8, symmetry=False)  # fmt: skip
    assert r["bands"] == 8


def test_the_conditioning_of_the_mixing():
    """A tighter bound on the condition number of the history of Pulay's
    mixing reaches the same ground state; a bound below 1 is refused."""
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    cell = fcc(5.431) @ np.diag([1.01, 1.0, 0.99])
    args = (cell, [[0, 0, 0], [0.27, 0.25, 0.23]], [0, 0], [pp], 12.0)
    common = dict(kgrid=(3, 3, 3), energy_tolerance=1e-12, density_tolerance=1e-14,
                  symmetry=False)  # fmt: skip
    usual = scf.run(*args, **common)
    tight = scf.run(*args, conditioning=1e2, **common)
    assert usual["converged"] and tight["converged"]
    assert tight["energy"] == pytest.approx(usual["energy"], abs=1e-10)
    with pytest.raises(ValueError, match="conditioning"):
        scf.run(*args, conditioning=0.5, **common)


@pytest.mark.parametrize("case", ["silicon", "aluminium-us", "iron-paw"])
def test_the_energies_of_the_outputs(case):
    """The Kohn-Sham free energy of the output of each step is above that of
    the ground state, variational (within the residuals of the bands), and
    meets the Harris-Foulkes energy of the inputs at convergence."""
    from dft.test_threads import CASES

    base, fractions, name, cutoff, options = CASES[case]
    pp = read_upf(espresso.pseudo(name))
    r = scf.run(base, fractions, [0] * len(fractions), [pp], cutoff, kgrid=(3, 3, 3),
                functional="pbe_pw", energy_tolerance=1e-12, density_tolerance=1e-14,
                symmetry=False, output_energies=True, **options)  # fmt: skip
    assert r["converged"]
    out = np.asarray(r["output_energies"])
    assert out.shape == np.asarray(r["energies"]).shape
    assert out[-1] == pytest.approx(r["energy"], abs=1e-9)
    assert (out >= r["energy"] - 1e-7).all()
    assert "output_energies" not in scf.run(
        base, fractions, [0] * len(fractions), [pp], cutoff, kgrid=(2, 2, 2),
        functional="pbe_pw", symmetry=False, **options,
    )  # fmt: skip


@pytest.mark.parametrize("case", ["silicon", "aluminium-us", "iron-paw"])
def test_adaptive_damping_reaches_the_ground_state(case):
    """Adaptive damping ends where the fixed damping does, every step taken
    lower in energy or in residual than the one it came from."""
    from dft.test_threads import CASES

    base, fractions, name, cutoff, options = CASES[case]
    pp = read_upf(espresso.pseudo(name))
    common = dict(kgrid=(3, 3, 3), functional="pbe_pw", energy_tolerance=1e-12,
                  density_tolerance=1e-14, symmetry=False, **options)  # fmt: skip
    args = (base, fractions, [0] * len(fractions), [pp], cutoff)
    fixed = scf.run(*args, **common)
    adaptive = scf.run(*args, adaptive_damping=True, **common)
    assert fixed["converged"] and adaptive["converged"]
    assert adaptive["energy"] == pytest.approx(fixed["energy"], abs=1e-9)
    taken = np.flatnonzero(adaptive["accepted"])
    out, acc = np.asarray(adaptive["output_energies"]), np.asarray(adaptive["accuracy"])
    for before, after in zip(taken[:-1], taken[1:]):
        assert out[after] < out[before] or acc[after] < acc[before]


def test_adaptive_damping_against_charge_sloshing():
    """An elongated cell of aluminium without Kerker's preconditioner, where
    the residual of long wavelengths sloshes: steps are taken back and tried
    closer, and the run reaches the state of the preconditioned one."""
    pp = read_upf(espresso.pseudo("Al_std.upf"))
    a = 4.04 / BOHR
    cell = np.diag([a, a, 4 * a])
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    fractions = np.concatenate([(base + [0, 0, k]) * [1, 1, 0.25] for k in range(4)])
    fractions[0] += 0.01
    args = (cell, fractions, [0] * 16, [pp], 10.0)
    common = dict(kgrid=(3, 3, 1), smearing="gaussian", sigma=0.01, functional="pbe_pw",
                  energy_tolerance=1e-10, density_tolerance=1e-12, symmetry=False,
                  max_iterations=60)  # fmt: skip
    reference = scf.run(*args, **common)
    sloshing = scf.run(*args, kerker=0.0, beta=0.8, adaptive_damping=True, **common)
    assert reference["converged"] and sloshing["converged"]
    assert sloshing["energy"] == pytest.approx(reference["energy"], abs=1e-8)
    assert min(sloshing["dampings"]) < 1.0 and not all(sloshing["accepted"])
