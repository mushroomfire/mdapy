# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.calculators import Calculator, CalculatorResult
from mdapy.calculators.testing import check_calculator
from mdapy.interop import from_ase, to_ase, to_ase_calculator
from mdapy.neighbor import NeighborList
from mdapy.simulation import minimize

ase = pytest.importorskip("ase")
from ase.calculators.calculator import PropertyNotImplementedError  # noqa: E402
from ase.calculators.fd import calculate_numerical_stress  # noqa: E402
from ase.filters import FrechetCellFilter  # noqa: E402
from ase.optimize import BFGS  # noqa: E402


class Morse(Calculator):
    """A Morse pair potential shifted to zero at the cutoff, in NumPy."""

    def __init__(self, depth=0.4, alpha=1.4, r0=2.6, cutoff=6.0, implemented=None):
        self.depth, self.alpha, self.r0, self.cutoff = depth, alpha, r0, cutoff
        if implemented is not None:
            self.implemented = frozenset(implemented)

    def _pair(self, r):
        e = np.exp(-self.alpha * (r - self.r0))
        ec = np.exp(-self.alpha * (self.cutoff - self.r0))
        energy = self.depth * (e * e - 2 * e) - self.depth * (ec * ec - 2 * ec)
        slope = self.depth * self.alpha * (2 * e - 2 * e * e)
        return energy, slope

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        self._check_properties(properties)
        nl = (
            NeighborList.build(system, self.cutoff)
            if neighbors is None
            else neighbors.within(self.cutoff)
        )
        n = len(system)
        r, v, i = nl.distances, nl.vectors, nl.i
        energy, slope = self._pair(r)
        # each pair appears from both of its atoms
        atomic = 0.5 * np.bincount(i, weights=energy, minlength=n)
        along = (slope / r)[:, None] * v
        forces = np.stack(
            [np.bincount(i, weights=along[:, a], minlength=n) for a in range(3)], axis=1
        )
        pair_virials = -0.5 * along[:, :, None] * v[:, None, :]
        virials = np.zeros((n, 3, 3))
        np.add.at(virials, i, pair_virials)
        values = {
            "energy": float(atomic.sum()),
            "atomic_energies": atomic,
            "forces": forces,
            "virial": virials.sum(axis=0),
            "atomic_virials": virials,
        }
        return CalculatorResult(**{name: values[name] for name in properties})


def crystal(repeat=3, a=3.7, jitter=0.0, seed=0):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    grid = np.stack(np.meshgrid(*[np.arange(repeat)] * 3, indexing="ij"), -1)
    positions = ((grid.reshape(-1, 1, 3) + basis) * a).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    positions = positions + rng.normal(scale=jitter, size=positions.shape)
    return mdapy.System(positions, mdapy.Box(repeat * a), elements="Cu")


def test_the_reference_potential_is_consistent():
    check_calculator(Morse(), crystal(repeat=2, jitter=0.1, seed=1))


def test_results_are_those_of_the_mdapy_calculator():
    system = crystal(jitter=0.1, seed=2)
    morse = Morse()
    direct = morse.compute(system, sorted(morse.implemented))
    atoms = to_ase(system)
    atoms.calc = to_ase_calculator(morse)
    assert atoms.get_potential_energy() == pytest.approx(direct.energy, rel=1e-14)
    assert atoms.get_potential_energy(force_consistent=True) == pytest.approx(
        direct.energy, rel=1e-14
    )
    np.testing.assert_allclose(atoms.get_forces(), direct.forces, atol=1e-13)
    np.testing.assert_allclose(
        atoms.get_potential_energies(), direct.atomic_energies, atol=1e-13
    )
    # ASE's stress is -virial / volume in eV/Å³, in the order xx yy zz yz xz xy
    stress = direct.stress(system) * units.GPA_TO_EV_PER_A3
    expected = [
        stress[0, 0],
        stress[1, 1],
        stress[2, 2],
        stress[1, 2],
        stress[0, 2],
        stress[0, 1],
    ]
    np.testing.assert_allclose(atoms.get_stress(), expected, atol=1e-14)
    np.testing.assert_allclose(
        atoms.get_stresses().sum(axis=0), atoms.get_stress(), atol=1e-13
    )
    assert repr(atoms.calc).startswith("to_ase_calculator(")


def test_stress_agrees_with_finite_differences_of_the_energy():
    atoms = to_ase(crystal(jitter=0.1, seed=3))
    atoms.calc = to_ase_calculator(Morse())
    numerical = calculate_numerical_stress(atoms, eps=1e-6)
    np.testing.assert_allclose(atoms.get_stress(), numerical, atol=1e-8)


def test_ase_relaxes_the_atoms_as_mdapy_does():
    system = crystal(repeat=2, jitter=0.08, seed=4)
    morse = Morse()
    atoms = to_ase(system)
    atoms.calc = to_ase_calculator(morse)
    BFGS(atoms, logfile=None).run(fmax=1e-6, steps=500)
    relaxed = minimize(system, morse, max_force=1e-6, max_steps=5000)
    assert relaxed.converged
    assert atoms.get_potential_energy() == pytest.approx(relaxed.energy, abs=1e-8)


def test_ase_relaxes_the_cell_by_the_stress():
    # the lattice constant where the shifted Morse crystal has no stress
    atoms = to_ase(crystal(repeat=2, a=3.9))
    atoms.calc = to_ase_calculator(Morse())
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-6, steps=500)
    assert np.abs(atoms.get_stress()).max() < 1e-6
    relaxed = from_ase(atoms)
    result = Morse().compute(relaxed, ("energy", "virial"))
    assert np.abs(result.stress(relaxed)).max() < 1e-3  # GPa


def test_properties_follow_the_mdapy_calculator():
    atoms = to_ase(crystal(repeat=2))
    atoms.calc = to_ase_calculator(Morse(implemented={"energy", "forces"}))
    assert set(atoms.calc.implemented_properties) == {"energy", "free_energy", "forces"}
    atoms.get_forces()
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()
    # energies alone, or forces alone
    atoms.calc = to_ase_calculator(Morse(implemented={"energy"}))
    assert atoms.get_potential_energy() < 0
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_forces()
    atoms.calc = to_ase_calculator(Morse(implemented={"forces"}))
    np.testing.assert_allclose(atoms.get_forces(), 0.0, atol=1e-12)  # perfect crystal
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_potential_energy()
