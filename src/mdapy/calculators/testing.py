# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Checks that any calculator can run on itself."""

import numpy as np
import polars as pl

from mdapy.calculators._base import PROPERTIES, Calculator
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.neighbor import NeighborList


def _replicate(system: System, counts: tuple[int, int, int]) -> System:
    """The system repeated along the lattice vectors (atoms copy by copy)."""
    h = system.box.matrix
    shifts = np.array(
        [
            [a, b, c]
            for a in range(counts[0])
            for b in range(counts[1])
            for c in range(counts[2])
        ]
    )
    positions = (system.positions[None, :, :] + (shifts @ h)[:, None, :]).reshape(-1, 3)
    box = Box(
        h * np.array(counts)[:, None], origin=system.box.origin, pbc=system.box.pbc
    )
    tiled = pl.concat([system.table] * len(shifts)).with_columns(
        pl.Series("positions", positions)
    )
    return System(table=tiled, box=box, info=dict(system.info))


def _deformed(system: System, gradient: np.ndarray) -> System:
    """The system with box and positions multiplied by gradient on the right."""
    box = system.box
    return System(
        table=system.table.with_columns(
            pl.Series("positions", system.positions @ gradient)
        ),
        box=Box(box.matrix @ gradient, origin=box.origin @ gradient, pbc=box.pbc),
        info=dict(system.info),
    )


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about axis (Rodrigues' formula)."""
    k = axis / np.linalg.norm(axis)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _close(name: str, actual, expected, rtol: float, atol: float) -> None:
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    error = np.abs(actual - expected)
    bound = atol + rtol * np.abs(expected)
    if not (error <= bound).all():
        worst = (
            np.unravel_index(np.argmax(error - bound), error.shape)
            if error.shape
            else ()
        )
        raise AssertionError(
            f"{name}: largest deviation {error[worst]:.3e} (allowed {bound[worst]:.3e}) "
            f"at {tuple(int(k) for k in worst)}: {actual[worst]!r} instead of {expected[worst]!r}"
        )


def check_calculator(
    calculator: Calculator,
    system: System,
    *,
    displacement: float = 1e-4,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    atoms: int = 3,
    seed: int = 0,
) -> None:
    """Checks that a calculator is consistent with itself and with physics.

    The checks are: atomic energies and atomic virials sum to the energy and
    the virial; forces and the virial agree with finite differences of the
    energy; translation, rotation and reordering of the atoms change nothing
    beyond rotating vectors; a doubled cell has twice the energy (for systems
    periodic along a); and a neighbor list given to the calculator gives the
    same result.

    Parameters
    ----------
    calculator : Calculator
        The calculator.
    system : System
        A small system, a few dozen atoms, that the calculator handles.
    displacement : float, optional
        Step in Å (and strain step) of the finite differences.
    rtol, atol : float, optional
        Relative and absolute tolerances of the comparisons, in the units of
        each quantity.
    atoms : int, optional
        Number of atoms whose forces are checked by finite differences.
    seed : int, optional
        Seed of the random rotation, translation and atom choice.

    Raises
    ------
    AssertionError
        If a check fails; the message names it and the largest deviation.
    """
    rng = np.random.default_rng(seed)
    implemented = calculator.implemented
    result = calculator.compute(system, implemented)
    n = len(system)
    shapes = {
        "atomic_energies": (n,),
        "forces": (n, 3),
        "virial": (3, 3),
        "atomic_virials": (n, 3, 3),
    }
    for name in implemented & PROPERTIES:
        value = getattr(result, name)
        if value is None:
            raise AssertionError(f"{name} is implemented but was not returned")
        if name in shapes and np.shape(value) != shapes[name]:
            raise AssertionError(
                f"{name} has shape {np.shape(value)}, not {shapes[name]}"
            )

    def energy(other: System) -> float:
        return calculator.compute(other, ("energy",)).energy

    if result.atomic_energies is not None:
        _close(
            "sum of atomic energies",
            result.atomic_energies.sum(),
            result.energy,
            1e-10,
            1e-10,
        )
    if result.atomic_virials is not None and result.virial is not None:
        _close(
            "sum of atomic virials",
            result.atomic_virials.sum(axis=0),
            result.virial,
            1e-10,
            1e-10,
        )

    if result.forces is not None:
        for i in rng.choice(n, size=min(atoms, n), replace=False):
            for a in range(3):
                step = np.zeros((n, 3))
                step[i, a] = displacement
                plus = energy(system.with_positions(system.positions + step))
                minus = energy(system.with_positions(system.positions - step))
                _close(
                    f"force on atom {i} along {'xyz'[a]} against finite differences",
                    result.forces[i, a],
                    -(plus - minus) / (2 * displacement),
                    rtol,
                    atol,
                )

    if result.virial is not None:
        finite = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                strain = np.eye(3)
                strain[a, b] += displacement
                plus = energy(_deformed(system, strain))
                strain[a, b] -= 2 * displacement
                minus = energy(_deformed(system, strain))
                finite[a, b] = -(plus - minus) / (2 * displacement)
        _close(
            "virial against finite differences of strain",
            result.virial,
            finite,
            rtol,
            atol,
        )

    shifted = system.with_positions(system.positions + rng.normal(size=3) * 3.0)
    moved = calculator.compute(shifted, implemented)
    _close("energy after a translation", moved.energy, result.energy, 1e-10, 1e-9)
    if result.forces is not None:
        _close("forces after a translation", moved.forces, result.forces, 1e-8, 1e-8)

    rotation = _rotation(rng.normal(size=3), rng.uniform(0.3, 3.0))
    rotated = calculator.compute(_deformed(system, rotation), implemented)
    _close("energy after a rotation", rotated.energy, result.energy, 1e-10, 1e-9)
    if result.forces is not None:
        _close(
            "forces after a rotation",
            rotated.forces,
            result.forces @ rotation,
            1e-8,
            1e-8,
        )
    if result.virial is not None:
        _close(
            "virial after a rotation",
            rotated.virial,
            rotation.T @ result.virial @ rotation,
            1e-8,
            1e-8,
        )

    order = rng.permutation(n)
    permuted = calculator.compute(system.gather(order), implemented)
    _close("energy after reordering atoms", permuted.energy, result.energy, 1e-10, 1e-9)
    if result.forces is not None:
        _close(
            "forces after reordering atoms",
            permuted.forces,
            result.forces[order],
            1e-10,
            1e-10,
        )

    if system.box.pbc[0]:
        doubled = energy(_replicate(system, (2, 1, 1)))
        _close("energy of a doubled cell", doubled, 2 * result.energy, 1e-10, 1e-9)

    cutoff = getattr(calculator, "cutoff", None)
    if cutoff:
        neighbors = NeighborList.build(system, 1.25 * cutoff)
        given = calculator.compute(system, implemented, neighbors=neighbors)
        _close(
            "energy with a given neighbor list",
            given.energy,
            result.energy,
            1e-12,
            1e-12,
        )


__all__ = ["check_calculator"]
