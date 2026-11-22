# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The real bands of Gamma (half a sphere of plane waves, real transforms and
real products) against the complex bands at Gamma: H, S and the occupations
of the projectors on real functions, the lowest eigenpairs, and whole runs of
norm-conserving, ultrasoft and PAW pseudopotentials, with spin: the energy,
the eigenvalues, the forces, the stress and the occupations of PAW; a run
against Quantum ESPRESSO; restarts."""

import math

import numpy as np
import pytest

from mdapy.calculators._dft import radial, scf
from mdapy.calculators._dft.upf import read_upf

from dft import espresso
from dft.hamiltonians import Gaussians, setup

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
GAMMA = [0.0, 0.0, 0.0]


def pair(cutoff=3.0, **options):
    """The Hamiltonian of Gamma of complex bands and that of real ones, of the
    same cell, projectors, D and potential (the options of setup)."""
    full, a, grid, table, dij, positions, v = setup(GAMMA, cutoff=cutoff, **options)
    half = _dft.Hamiltonian(a, grid, cutoff, [table], [Gaussians.angular], [dij], radial.DQ, positions,
                            np.array([0, 0], dtype=np.int32), gamma=True)  # fmt: skip
    half.set_k(GAMMA)
    half.set_potential(v)
    assert half.gamma and not full.gamma
    return full, half


def expand(full, half, c):
    """Coefficients c (bands, half sphere) of real functions on the whole
    sphere: c(-G) = conj c(G)."""
    where = {tuple(m): i for i, m in enumerate(full.miller)}
    out = np.zeros((len(c), len(full.miller)), dtype=complex)
    for j, m in enumerate(half.miller):
        out[:, where[tuple(m)]] = c[:, j]
        out[:, where[tuple(-m)]] = np.conj(c[:, j])
    return out


def restrict(full, half, c):
    """The values of c (bands, whole sphere) on the half sphere."""
    where = {tuple(m): i for i, m in enumerate(full.miller)}
    return c[:, [where[tuple(m)] for m in half.miller]]


def real_bands(half, count, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(count, len(half.miller))) + 1j * rng.normal(
        size=(count, len(half.miller))
    )
    c[:, 0] = c[:, 0].real
    return c


def test_the_half_sphere():
    full, half = pair()
    # G = 0 first, and every other G of the sphere either kept or its opposite
    assert not np.any(half.miller[0])
    assert 2 * len(half.miller) - 1 == len(full.miller)


def test_h_and_s_on_real_functions():
    full, half = pair()
    rng = np.random.default_rng(1)
    widths = sum(2 * ell + 1 for ell in (0, 1, 1, 2, 3))
    q = rng.normal(size=(2, widths, widths))
    q = 0.05 * (q + q.transpose(0, 2, 1))
    for h in (full, half):
        h.set_overlaps(np.ascontiguousarray(q.reshape(-1)))
    c = real_bands(half, 7)
    whole = expand(full, half, c)
    for name in ("apply", "apply_overlap"):
        ours = getattr(half, name)(c)
        reference = restrict(full, half, getattr(full, name)(whole))
        np.testing.assert_allclose(
            ours, reference, rtol=0, atol=1e-11 * np.abs(reference).max()
        )
        # the results are real functions too
        assert np.all(ours[:, 0].imag == 0.0)
    # rubbish in Im c_0 changes nothing
    dirty = c.copy()
    dirty[:, 0] += 0.3j
    np.testing.assert_allclose(half.apply(dirty), half.apply(c), rtol=0, atol=1e-12)
    # the occupations of the projectors
    w = np.linspace(2.0, 0.5, 7)
    size = 2 * widths * (widths + 1) // 2
    np.testing.assert_allclose(
        half.becsum(c, w, size), full.becsum(whole, w, size), rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        half.diagonal, restrict(full, half, full.diagonal[None])[0], rtol=1e-13
    )


def test_the_lowest_eigenpairs():
    # a spectrum led by the kinetic energy, as test_davidson's
    full, half = pair(cutoff=8.0, potential=1.0, nonlocal_=1e-3)
    guess = real_bands(half, 6, seed=2)
    e_half, v_half, _, converged, _ = _dft.davidson(half, guess, tolerance=1e-9)
    assert converged == 6
    e_full, _, _, converged, _ = _dft.davidson(
        full, expand(full, half, guess), tolerance=1e-9
    )
    assert converged == 6
    np.testing.assert_allclose(e_half, e_full, rtol=0, atol=1e-10)
    # the eigenvectors are real functions of the eigenvalues
    hv = half.apply(v_half)
    np.testing.assert_allclose(hv, e_half[:, None] * v_half, rtol=0, atol=1e-8)


def cubic_silicon():
    a = 5.431 / BOHR
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                     [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])  # fmt: skip
    rng = np.random.default_rng(3)
    return np.eye(3) * a, base + rng.normal(0.0, 0.004, base.shape)


def both(cell, positions, kinds, pseudos, cutoff, **options):
    """Runs of complex and of real bands at Gamma, converged tightly."""
    runs = [
        scf.run(
            cell,
            positions,
            kinds,
            pseudos,
            cutoff,
            energy_tolerance=1e-12,
            density_tolerance=1e-14,
            symmetry=False,
            gamma=gamma,
            **options,
        )  # fmt: skip
        for gamma in (False, True)
    ]
    assert [r["gamma"] for r in runs] == [False, True]
    assert all(r["converged"] for r in runs)
    return runs


def agree(complex_, real, atoms, forces=1e-8):
    assert real["energy"] == pytest.approx(complex_["energy"], abs=1e-10 * atoms)
    held = complex_["occupations"] > 1e-8
    np.testing.assert_allclose(
        real["eigenvalues"][held], complex_["eigenvalues"][held], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(real["forces"], complex_["forces"], rtol=0, atol=forces)
    np.testing.assert_allclose(
        real["strain_derivative"],
        complex_["strain_derivative"],
        rtol=0,
        atol=1e-8 * atoms,
    )
    assert real["magnetization"] == pytest.approx(complex_["magnetization"], abs=1e-8)


def test_norm_conserving_silicon():
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    c, r = both(cell, positions, [0] * 8, [pp], 12.0, functional="pbe_pw")
    agree(c, r, 8)


def test_ultrasoft_aluminium_with_smearing():
    a = 4.05 / BOHR
    positions = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) + [[0.01, 0, 0], [0, 0, 0], [0, 0, 0],
                                                                                 [0, -0.01, 0.005]]  # fmt: skip
    pp = read_upf(espresso.pseudo("pslibrary/Al.pbe-n-rrkjus_psl.1.0.0.UPF"))
    c, r = both(np.eye(3) * a, positions, [0] * 4, [pp], 12.0, density_cutoff=96.0, smearing="gaussian",
                sigma=0.02)  # fmt: skip
    agree(c, r, 4)


def test_paw_iron_with_spin():
    h = 2.87 / BOHR
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.49, 0.5]])
    pp = read_upf(espresso.pseudo("pslibrary/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"))
    c, r = both(np.eye(3) * h, positions, [0, 0], [pp], 15.0, density_cutoff=120.0, smearing="gaussian",
                sigma=0.02, magmoms=[2.5, 2.5])  # fmt: skip
    agree(c, r, 2)
    np.testing.assert_allclose(r["becsum"], c["becsum"], rtol=0, atol=1e-9)
    assert r["onecenter"] == pytest.approx(c["onecenter"], abs=1e-9)


def test_against_quantum_espresso():
    """Real bands of Gamma against pw.x at Gamma, as test_scf."""
    cell, positions = cubic_silicon()
    files = {"Si": espresso.pseudo("Si_std.upf")}
    qe = espresso.run(
        cell, positions, ["Si"] * 8, files, 12.0, [GAMMA], [1.0], conv_thr=1e-14
    )
    ours = scf.run(cell, positions, [0] * 8, [read_upf(files["Si"])], 12.0, functional="pbe_pw", grid=qe["grid"],
                   energy_tolerance=1e-12, density_tolerance=1e-14, symmetry=False)  # fmt: skip
    assert ours["gamma"]
    for key in ["energy", "band", "one_electron", "hartree", "xc", "ewald"]:
        assert ours[key] == pytest.approx(qe[key], abs=1e-7 * 8), key
    width = qe["eigenvalues"].shape[1]
    np.testing.assert_allclose(
        ours["eigenvalues"][:, :width], qe["eigenvalues"], rtol=0, atol=1e-7
    )


def test_restart_of_real_bands():
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    state = _dft.State()
    options = dict(
        functional="pbe_pw",
        energy_tolerance=1e-10,
        density_tolerance=1e-12,
        symmetry=False,
    )
    first = scf.run(cell, positions, [0] * 8, [pp], 12.0, state=state, **options)
    assert first["gamma"] and not first["restarted"]
    moved = positions.copy()
    moved[0, 0] += 0.003
    warm = scf.run(cell, moved, [0] * 8, [pp], 12.0, state=state, **options)
    cold = scf.run(cell, moved, [0] * 8, [pp], 12.0, **options)
    assert warm["restarted"] and warm["iterations"] <= cold["iterations"]
    assert warm["energy"] == pytest.approx(cold["energy"], abs=1e-8)
    # complex bands are not taken for real ones
    again = scf.run(
        cell, moved, [0] * 8, [pp], 12.0, state=state, gamma=False, **options
    )
    assert not again["restarted"]


def test_real_bands_take_gamma_only():
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    with pytest.raises(ValueError, match="Gamma"):
        scf.run(cell, positions, [0] * 8, [pp], 12.0, kgrid=(2, 1, 1), gamma=True)
    r = scf.run(cell, positions, [0] * 8, [pp], 12.0, kgrid=(2, 1, 1), max_iterations=1)
    assert not r["gamma"]
    assert math.isfinite(r["energy"])
