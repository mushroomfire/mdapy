# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.calculators import Calculator, CalculatorResult
from mdapy.calculators.testing import check_calculator
from mdapy.errors import InvalidParameter
from mdapy.interop import from_ase, from_ase_calculator, to_ase, to_ase_calculator
from mdapy.neighbor import NeighborList
from mdapy.simulation import minimize

ase = pytest.importorskip("ase")
from ase.calculators.calculator import Calculator as AseCalculator  # noqa: E402
from ase.calculators.calculator import PropertyNotImplementedError  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402
from ase.calculators.fd import calculate_numerical_stress  # noqa: E402
from ase.calculators.lj import LennardJones  # noqa: E402
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


# ------------------------------------------------- ASE calculators in mdapy


def test_an_ase_calculator_passes_the_checks_of_mdapy():
    # a smaller step than the default: the cutoff function of EMT bends the
    # energy enough that the default step misses the virial by 2e-6
    calculator = from_ase_calculator(EMT())
    assert calculator.implemented == {"energy", "atomic_energies", "forces", "virial"}
    assert calculator.cutoff == 0.0
    check_calculator(
        calculator, crystal(repeat=2, a=3.6, jitter=0.05), displacement=1e-5
    )
    assert repr(calculator).startswith("from_ase_calculator(")


def test_both_ways_give_the_mdapy_calculator_back():
    system = crystal(jitter=0.1, seed=5)
    morse = Morse()
    properties = sorted(morse.implemented)
    direct = morse.compute(system, properties)
    back = from_ase_calculator(to_ase_calculator(morse)).compute(system, properties)
    assert back.energy == pytest.approx(direct.energy, rel=1e-14)
    for name in ("atomic_energies", "forces", "virial", "atomic_virials"):
        np.testing.assert_allclose(
            getattr(back, name), getattr(direct, name), atol=1e-12, err_msg=name
        )


def test_the_free_energy_is_the_energy_where_there_is_one():
    class Smeared(AseCalculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def __init__(self, free=True):
            super().__init__()
            if not free:
                self.implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": -1.0,
                "free_energy": -1.5,
                "forces": np.zeros((len(self.atoms), 3)),
            }

    system = crystal(repeat=1)
    assert from_ase_calculator(Smeared()).compute(system).energy == -1.5
    assert from_ase_calculator(Smeared(free=False)).compute(system).energy == -1.0


def test_mdapy_relaxes_with_an_ase_calculator_as_ase_does():
    system = crystal(repeat=2, a=3.6, jitter=0.08, seed=6)
    relaxed = minimize(
        system, from_ase_calculator(EMT()), max_force=1e-6, max_steps=5000
    )
    assert relaxed.converged
    atoms = to_ase(system)
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-6, steps=500)
    assert relaxed.energy == pytest.approx(atoms.get_potential_energy(), abs=1e-8)


def test_properties_the_ase_calculator_lacks():
    calculator = from_ase_calculator(EMT())
    with pytest.raises(InvalidParameter, match="atomic_virials"):
        calculator.compute(crystal(repeat=1, a=3.6), ["atomic_virials"])


def test_atomic_stresses_of_an_ase_calculator():
    # Lennard-Jones of ASE gives the stress of each atom too
    lennard_jones = LennardJones(sigma=2.3, epsilon=0.1, rc=6.0, smooth=True)
    calculator = from_ase_calculator(lennard_jones)
    assert calculator.implemented == {
        "energy",
        "atomic_energies",
        "forces",
        "virial",
        "atomic_virials",
    }
    system = crystal(repeat=2, a=3.6, jitter=0.05, seed=8)
    check_calculator(calculator, system, displacement=1e-5)
    result = calculator.compute(system, ["atomic_virials"])
    atoms = to_ase(system)
    atoms.calc = lennard_jones
    stresses = atoms.get_stresses(voigt=False)
    np.testing.assert_allclose(
        result.atomic_virials, -stresses * system.box.volume, atol=1e-12
    )
