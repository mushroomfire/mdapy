# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
"""The two kernels of the DFT: MDAPY's own kernels (the FFTs) against the GPU
libraries of the vendor. The transforms agree to rounding on the grids of the
DFT, and so do whole calculations: norm-conserving, ultrasoft and PAW
pseudopotentials, k-points and the real bands of Gamma, spin, mixed precision.
"""

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

BOHR, HARTREE = 0.529177210903, 27.211386245988
KERNELS = ("mdapy", "vendor")


def sphere_of(cell, cutoff, grid, half=False):
    miller, _, box, partner = _dft.sphere(
        cell, [0.0, 0.0, 0.0], cutoff, list(grid), half
    )
    return miller, box, partner


# sizes of every radix, odd and even, a plane that fits a block and one that
# does not; 22 = 2 x 11 is not MDAPY's to transform and goes to the vendor's
@pytest.mark.parametrize(
    "grid",
    [
        (27, 27, 27),
        (36, 30, 25),
        (20, 21, 35),
        (45, 48, 80),
        (64, 64, 64),
        (90, 90, 96),
        (22, 24, 20),
    ],
)
def test_the_transforms_agree(grid):
    rng = np.random.default_rng(1)
    cell = np.diag([9.0, 9.5, 10.0])
    _, box, _ = sphere_of(cell, 3.0, grid)
    c = rng.normal(size=(3, len(box))) + 1j * rng.normal(size=(3, len(box)))
    values = [_dft.to_real_space(c, box, list(grid), kernels=b) for b in KERNELS]
    scale = np.abs(values[1]).max()
    assert np.abs(values[0] - values[1]).max() < 1e-13 * scale
    # two programs differ in the last bits; one program gives the same bits
    assert np.array_equal(values[0], values[1]) == (grid == (22, 24, 20))
    noise = rng.normal(size=(2, *grid)) + 1j * rng.normal(size=(2, *grid))
    back = [_dft.to_coefficients(noise, box, kernels=b) for b in KERNELS]
    assert np.abs(back[0] - back[1]).max() < 1e-13 * np.abs(back[1]).max()
    # the round trip of MDAPY's own
    again = _dft.to_coefficients(values[0], box, kernels="mdapy")
    assert np.abs(again - c).max() < 1e-13 * np.abs(c).max()


@pytest.mark.parametrize(
    "grid",
    [
        (27, 27, 27),
        (36, 30, 24),
        (20, 21, 35),
        (45, 48, 80),
        (48, 45, 75),
        (90, 90, 96),
    ],
)
def test_the_real_transforms_agree(grid):
    rng = np.random.default_rng(2)
    cell = np.diag([9.0, 9.5, 10.0])
    _, box, partner = sphere_of(cell, 3.0, grid, half=True)
    c = rng.normal(size=(3, len(box))) + 1j * rng.normal(size=(3, len(box)))
    c[:, 0] = c[:, 0].real  # G = 0, the first of the half sphere, is real
    values = [
        _dft.to_real_space_gamma(c, box, partner, list(grid), kernels=b)
        for b in KERNELS
    ]
    assert np.abs(values[0] - values[1]).max() < 1e-13 * np.abs(values[1]).max()
    noise = rng.normal(size=(2, *grid))
    back = [_dft.to_coefficients_gamma(noise, box, kernels=b) for b in KERNELS]
    assert np.abs(back[0] - back[1]).max() < 1e-13 * np.abs(back[1]).max()


def test_a_backend_unknown_is_refused():
    cell = np.diag([9.0, 9.5, 10.0])
    _, box, _ = sphere_of(cell, 3.0, (20, 20, 20))
    with pytest.raises(ValueError, match="kernels"):
        _dft.to_real_space(
            np.zeros((1, len(box)), dtype=complex), box, [20, 20, 20], kernels="other"
        )
    with pytest.raises(ValueError, match="kernels"):
        scf.run(np.eye(3) * 8.0, np.zeros((1, 3)), [0], [None], 10.0, kernels="other")
    with pytest.raises(InvalidParameter, match="kernels"):
        DFT(300.0, kernels="other")


def both(cell, positions, kinds, pseudos, cutoff, **options):
    """The same calculation with each kernels, converged tightly."""
    runs = [
        scf.run(cell, positions, kinds, pseudos, cutoff, energy_tolerance=1e-12, density_tolerance=1e-14,
                symmetry=False, kernels=kernels, **options)
        for kernels in KERNELS
    ]  # fmt: skip
    assert all(r["converged"] for r in runs)
    return runs


def agree(own, vendor, atoms, energy=1e-10, forces=1e-9, strain=1e-9, levels=1e-9):
    """Energies (hartree), forces (hartree/bohr) and strain derivatives of the
    two kernels within the tolerances, the first and the last per atom."""
    assert own["grid"] == vendor["grid"] and own["bands"] == vendor["bands"]
    assert own["energy"] == pytest.approx(vendor["energy"], abs=energy * atoms)
    np.testing.assert_allclose(own["forces"], vendor["forces"], rtol=0, atol=forces)
    np.testing.assert_allclose(
        own["strain_derivative"],
        vendor["strain_derivative"],
        rtol=0,
        atol=strain * atoms,
    )
    held = vendor["occupations"] > 1e-8
    np.testing.assert_allclose(
        own["eigenvalues"][held], vendor["eigenvalues"][held], rtol=0, atol=levels
    )


def cubic_silicon():
    a = 5.431 / BOHR
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                     [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])  # fmt: skip
    rng = np.random.default_rng(3)
    return np.eye(3) * a, base + rng.normal(0.0, 0.004, base.shape)


@pytest.mark.parametrize("gamma", [False, True])
def test_norm_conserving_silicon_at_gamma(gamma):
    """Complex bands, and the real bands of Gamma through the real transforms."""
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    own, vendor = both(
        cell, positions, [0] * 8, [pp], 12.0, functional="pbe_pw", gamma=gamma
    )
    assert own["gamma"] == gamma
    agree(own, vendor, 8)


def test_norm_conserving_silicon_with_k_points():
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    own, vendor = both(
        cell, positions, [0] * 8, [pp], 12.0, functional="pbe_pw", kgrid=(2, 2, 2)
    )
    agree(own, vendor, 8)


def test_ultrasoft_aluminium_with_smearing():
    """Two grids: the bands on the smooth one, the density on the dense one."""
    a = 4.05 / BOHR
    positions = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) + [[0.01, 0, 0], [0, 0, 0], [0, 0, 0],
                                                                                 [0, -0.01, 0.005]]  # fmt: skip
    pp = read_upf(espresso.pseudo("pslibrary/Al.pbe-n-rrkjus_psl.1.0.0.UPF"))
    own, vendor = both(np.eye(3) * a, positions, [0] * 4, [pp], 12.0, density_cutoff=96.0, smearing="gaussian",
                       sigma=0.02, kgrid=(2, 2, 2))  # fmt: skip
    agree(own, vendor, 4)


@pytest.mark.parametrize("gamma", [False, True])
def test_paw_iron_with_spin(gamma):
    h = 2.87 / BOHR
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.49, 0.5]])
    pp = read_upf(espresso.pseudo("pslibrary/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"))
    own, vendor = both(np.eye(3) * h, positions, [0, 0], [pp], 15.0, density_cutoff=120.0, smearing="gaussian",
                       sigma=0.02, magmoms=[2.5, 2.5], gamma=gamma)  # fmt: skip
    agree(own, vendor, 2)
    assert own["magnetization"] == pytest.approx(vendor["magnetization"], abs=1e-8)


@pytest.mark.parametrize("gamma", [False, True])
def test_mixed_precision(gamma):
    """The steps in single precision take the single transforms of each
    kernels; the run ends in double precision, at the same result."""
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    own, vendor = both(
        cell,
        positions,
        [0] * 8,
        [pp],
        12.0,
        functional="pbe_pw",
        gamma=gamma,
        precision="mixed",
    )
    assert np.sum(own["single"]) > 0 and np.sum(vendor["single"]) > 0
    agree(own, vendor, 8, energy=1e-9, forces=1e-7, strain=1e-7, levels=1e-7)


def test_the_calculator_keeps_its_backend():
    """DFT(kernels=...): "mdapy" by default, kept in the provenance; the
    energy, the forces and the virial of the two agree."""
    a = 4.05
    cell = np.diag([a, a, a])
    fractions = np.array([[0.01, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    system = mdapy.System(fractions @ cell, mdapy.Box(cell), elements=["Al"] * 4)
    common = dict(pseudopotentials={"Al": espresso.pseudo("Al_std.upf")}, kpoints=(2, 2, 2), sigma=0.2,
                  functional="pbe_pw", tolerance=1e-10, symmetry=False)  # fmt: skip
    own, vendor = DFT(300.0, **common), DFT(300.0, kernels="vendor", **common)
    assert own.kernels == "mdapy" and vendor.kernels == "vendor"
    first = own.compute(system, ("energy", "forces", "virial"))
    second = vendor.compute(system, ("energy", "forces", "virial"))
    assert own.last["provenance"]["kernels"] == "mdapy"
    assert vendor.last["provenance"]["kernels"] == "vendor"
    assert first.energy == pytest.approx(second.energy, abs=1e-8)
    np.testing.assert_allclose(first.forces, second.forces, atol=1e-7)
    np.testing.assert_allclose(first.virial, second.virial, atol=1e-6)
