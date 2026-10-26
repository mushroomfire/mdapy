# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import dataclasses

import numpy as np
import pytest

import mdapy
from mdapy.calculators import Calculator, CalculatorResult
from mdapy.calculators.testing import check_calculator
from mdapy.errors import InvalidParameter

from .eam_files import alloy


class Harmonic(Calculator):
    """E = sum over pairs of k (r - r0)^2 / 2, in NumPy."""

    cutoff = 3.2
    reach = 3.2  # radius of the neighbor search

    def __init__(self, k=1.0, r0=2.5):
        self.k, self.r0 = k, r0

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        self._check_properties(properties)
        nl = neighbors or mdapy.NeighborList.build(system, self.reach)
        keep = nl.distances < self.reach
        i, r, d = nl.i[keep], nl.vectors[keep], nl.distances[keep]
        n = len(system)
        energies = np.zeros(n)
        np.add.at(energies, i, 0.25 * self.k * (d - self.r0) ** 2)
        # force on i from the pair: -dE/dx_i, with r = x_j - x_i
        f = (self.k * (d - self.r0) / d)[:, None] * r
        forces = np.zeros((n, 3))
        np.add.at(forces, i, f)
        virials = np.zeros((n, 3, 3))
        np.add.at(virials, i, -0.5 * r[:, :, None] * f[:, None, :])
        return CalculatorResult(
            energy=float(energies.sum()),
            atomic_energies=energies,
            forces=forces,
            virial=virials.sum(axis=0),
            atomic_virials=virials,
        )


class Broken(Harmonic):
    """Harmonic with one deliberate error. The errors of the energy are made
    so that only the check they are named after can see them."""

    def __init__(self, error, centroid=None):
        super().__init__()
        self.error = error
        self.centroid = centroid

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        result = super().compute(system, properties, neighbors=neighbors)
        x = system["positions"]

        def energy(delta):
            """The energy changed by delta, on the first atom too."""
            atomic = result.atomic_energies.copy()
            atomic[0] += delta
            return {"energy": result.energy + delta, "atomic_energies": atomic}

        changes = {
            "forces": lambda: {"forces": -result.forces},
            "atomic_energies": lambda: {
                "atomic_energies": result.atomic_energies + 0.1
            },
            "atomic_virials": lambda: {"atomic_virials": result.atomic_virials * 2},
            "virial": lambda: {
                "virial": result.virial * 0.5,
                "atomic_virials": result.atomic_virials * 0.5,
            },
            "missing": lambda: {"forces": None},
            "shape": lambda: {"forces": result.forces[:-1]},
            # jumps when the atoms move by more than 1 Å together
            "translation": lambda: energy(
                0.0 if np.abs(x.mean(axis=0) - self.centroid).max() < 1.0 else 1.0
            ),
            # quadratic in an off-diagonal element of the box: zero slope under strain
            "rotation": lambda: energy(system.box.matrix[0, 1] ** 2),
            # depends on which element each row has
            "order": lambda: energy(
                0.01
                * float(
                    np.arange(len(x)) @ system.table["element"].to_physical().to_numpy()
                )
            ),
            "doubling": lambda: energy(1.0),
            "neighbors": lambda: energy(1e-6 if neighbors is not None else 0.0),
        }
        return dataclasses.replace(result, **changes[self.error]())


class EnergyOnly(Harmonic):
    implemented = frozenset({"energy"})
    cutoff = 0.0  # no neighbor list check

    def compute(self, system, properties=("energy",), *, neighbors=None):
        self._check_properties(properties)
        result = Harmonic.compute(self, system, ("energy",))
        return CalculatorResult(energy=result.energy)


def system(pbc=True):
    return alloy(seed=20, pbc=pbc, jitter=0.1)


@pytest.mark.parametrize("pbc", [True, False])
def test_a_correct_calculator_passes(pbc):
    check_calculator(Harmonic(), system(pbc), rtol=1e-6, atol=1e-6)


def test_an_energy_only_calculator_passes():
    calculator = EnergyOnly()  # declares no cutoff: no neighbor list check
    check_calculator(calculator, system())
    with pytest.raises(InvalidParameter, match=r"does not compute \['forces'\]"):
        calculator.compute(system(), ("forces",))


@pytest.mark.parametrize(
    "error, message",
    [
        ("forces", "force on atom"),
        ("atomic_energies", "sum of atomic energies"),
        ("atomic_virials", "sum of atomic virials"),
        ("virial", "virial against finite differences"),
        ("missing", "forces is implemented but was not returned"),
        ("shape", r"forces has shape \(31, 3\)"),
        ("translation", "energy after a translation"),
        ("rotation", "energy after a rotation"),
        ("order", "energy after reordering"),
        ("doubling", "energy of a doubled cell"),
        ("neighbors", "energy with a given neighbor list"),
    ],
)
def test_each_check_catches_its_error(error, message):
    test_system = system()
    calculator = Broken(error, centroid=test_system["positions"].mean(axis=0))
    with pytest.raises(AssertionError, match=message):
        check_calculator(calculator, test_system)


def test_non_periodic_systems_skip_the_doubled_cell():
    check_calculator(Broken("doubling"), system(pbc=False))
