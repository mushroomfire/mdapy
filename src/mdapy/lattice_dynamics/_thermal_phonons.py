# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Phonons at finite temperature: the force constants that best give the forces
of atoms displaced by molecular dynamics."""

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike

from mdapy.analysis._symmetry import _keeping, _Operations, _symmetry
from mdapy.calculators._lammps import LAMMPS
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.threads import CancelToken
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.lattice_dynamics._phonons import (
    Phonons,
    _default_supercell,
    _lattice_points,
    _masses,
    _supercell,
    _supercell_matrix,
    _symmetrized,
)
from mdapy.simulation._dynamics import MolecularDynamics

# damping time in ps of the thermostat, which samples vibrations as a Nosé-Hoover
# thermostat does not
_DAMPING = 0.1
# added to the diagonal of the normal equations, relative to their largest value
_RIDGE = 1e-10


def _cell_table(matrix: np.ndarray) -> np.ndarray:
    """(S, S) the cell of the supercell at lattice point L + lattice point M."""
    repeats = _lattice_points(matrix)
    found = {tuple(r): k for k, r in enumerate(repeats.tolist())}
    inverse = np.linalg.inv(matrix)
    table = np.zeros((len(repeats), len(repeats)), dtype=np.int64)
    for k, shift in enumerate(repeats):
        fractional = (repeats + shift) @ inverse
        inside = np.round((fractional - np.floor(fractional + 1e-9)) @ matrix)
        table[k] = [found[tuple(r)] for r in inside.astype(int).tolist()]
    return table


def _displacements(reference: System, frame: System) -> np.ndarray:
    """(N, 3) how far each atom is from its place, with no total displacement."""
    fractional = reference.box.to_fractional(
        frame["positions"]
    ) - reference.box.to_fractional(reference["positions"])
    shifts = (fractional - np.round(fractional)) @ reference.box.matrix
    return shifts - shifts.mean(axis=0)


def _fit(
    cell: System,
    matrix: np.ndarray,
    moves: list[np.ndarray],
    forces: list[np.ndarray],
) -> np.ndarray:
    """(n, n S, 3, 3) force constants by least squares, the same at each cell
    of the supercell; a pair depends on the difference of the displacements of
    its atoms, which leaves the crystal free to translate."""
    n, table = len(cell), _cell_table(matrix)
    cells = len(table)
    size = n * cells
    constants = np.zeros((n, size, 3, 3))
    atoms = np.arange(size)
    basis, inner = atoms % n, atoms // n
    for a in range(n):
        others = np.delete(atoms, a)  # all pairs of the atom of the first cell
        columns = table[:, inner[others]] * n + basis[others]  # (S, M) in each cell
        own = np.arange(cells) * n + a  # (S,) the atom itself in each cell
        normal = np.zeros((3 * (size - 1), 3 * (size - 1)))
        right = np.zeros((3 * (size - 1), 3))
        for move, force in zip(moves, forces):
            rows = (move[columns] - move[own][:, None, :]).reshape(cells, -1)
            normal += rows.T @ rows
            right += rows.T @ force[own]
        normal[np.diag_indices_from(normal)] += _RIDGE * np.abs(normal).max()
        solved = np.linalg.solve(normal, right)  # (3 M, 3), beta of j, alpha
        blocks = -solved.reshape(size - 1, 3, 3).transpose(0, 2, 1)  # alpha, beta
        constants[a, others] = blocks
        constants[a, a] = -blocks.sum(axis=0)
    return constants


def _symmetric_average(
    constants: np.ndarray,
    matrix: np.ndarray,
    operations: _Operations,
) -> np.ndarray:
    """The force constants averaged over the symmetry operations of the crystal
    that map the supercell onto itself."""
    keep = _keeping(operations.rotations, matrix)
    rotations, cartesian = operations.rotations[keep], operations.cartesian[keep]
    atoms, shifts = operations.atoms[keep], operations.shifts[keep]
    n = constants.shape[0]
    repeats = _lattice_points(matrix)
    found = {tuple(r): k for k, r in enumerate(repeats.tolist())}
    inverse = np.linalg.inv(matrix)
    total = np.zeros_like(constants)
    for k in range(len(rotations)):
        for a in range(n):
            # where the operation takes each atom, the image of a moved back
            # into the first cell
            lattice = (repeats @ rotations[k].T)[:, None, :] + shifts[k] - shifts[k, a]
            fractional = lattice.reshape(-1, 3) @ inverse
            inside = np.round((fractional - np.floor(fractional + 1e-9)) @ matrix)
            cell = np.array([found[tuple(r)] for r in inside.astype(int).tolist()])
            image = cell * n + np.tile(atoms[k], len(repeats))
            total[atoms[k, a], image] += cartesian[k] @ constants[a] @ cartesian[k].T
    return total / len(rotations)


def phonons_from_frames(
    cell: System,
    frames: Iterable[System],
    *,
    supercell: int | ArrayLike,
    symmetry: bool | None = None,
) -> Phonons:
    """Phonons from atoms displaced in a supercell and the forces on them, from
    molecular dynamics of any kind: the force constants that best give the
    forces, which hold the temperature of the frames.

    Parameters
    ----------
    cell : System
        The cell of the crystal, periodic in every direction, with the atoms at
        their places at the volume of the frames, each with an element or a
        mass; best its primitive cell.
    frames : iterable of System
        The atoms of the supercell, in its order, with a ``forces`` column:
        cells in the order of ``Phonons.force_constants``, as
        ``phonons`` makes them; far enough apart in time to differ.
    supercell : int32 or array_like
        Repeats of the cell along its vectors, one or three, or a (3, 3)
        integer matrix whose rows times the vectors of the cell are the
        vectors of the supercell.
    symmetry : bool, optional
        Average the force constants over the symmetry operations of the cell,
        found by spglib: True, False, or by default whenever spglib is
        installed.

    Returns
    -------
    Phonons
        The force constants, which give frequencies, polarization vectors,
        band structures, densities of states and thermal properties.

    Raises
    ------
    InvalidParameter
        If the box is not periodic, an atom has no mass, the supercell is not
        valid, a frame has another number of atoms, or there are too few
        frames for the force constants.
    MissingProperty
        If a frame has no ``forces`` column.
    """
    if not cell.box.pbc.all() or len(cell) == 0:
        raise InvalidParameter(
            "phonons need a box periodic in every direction with atoms"
        )
    masses = _masses(cell)
    matrix = _supercell_matrix(supercell)
    big = _supercell(cell, matrix)
    moves, forces = [], []
    for frame in frames:
        if len(frame) != len(big):
            raise InvalidParameter(
                f"a frame has {len(frame)} atoms, the supercell {len(big)}"
            )
        if "forces" not in frame.table.columns:
            raise MissingProperty("the frames need the forces column")
        moves.append(_displacements(big, frame))
        forces.append(frame["forces"])
    cells = round(np.linalg.det(matrix))
    needed = int(np.ceil(3 * (len(big) - 1) / cells))
    if len(moves) < needed:
        raise InvalidParameter(
            f"{len(moves)} frames are too few: the force constants need {needed} "
            "or more"
        )
    constants = _fit(cell, matrix, moves, forces)
    operations = _symmetry(cell, symmetry)
    if operations is not None:
        constants = _symmetric_average(constants, matrix, operations)
    return Phonons(cell, matrix, _symmetrized(constants, matrix), masses)


def thermal_phonons(
    cell: System,
    potential: LAMMPS,
    temperature: float,
    *,
    supercell: int | ArrayLike | None = None,
    pressure: float | None = 0.0,
    timestep: float = 0.001,
    equilibration: float = 10.0,
    production: float = 100.0,
    frames: int = 200,
    symmetry: bool | None = None,
    seed: int = 1,
    progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
) -> Phonons:
    """Phonons of a crystal at a temperature, from molecular dynamics in LAMMPS:
    the force constants that best give the forces on the moving atoms, which
    change with the temperature as the vibrations feel the crystal further from
    its minimum.

    Parameters
    ----------
    cell : System
        The cell of the crystal, relaxed, with an ``element`` column, periodic
        in every direction; best its primitive cell.
    potential : LAMMPS
        The interaction.
    temperature : float64
        Temperature in K, positive.
    supercell : int32 or array_like, optional
        Repeats of the cell along its vectors, one or three, or a (3, 3)
        integer matrix. Default the supercell nearest a cube that is at least
        twice the cutoff of the potential thick in every direction.
    pressure : float64 or None, optional
        Pressure in GPa at which the box takes its mean size first; None to
        keep the box of cell. Default 0.
    timestep : float64, optional
        Time step in ps. Default 0.001.
    equilibration : float64, optional
        Time in ps of each equilibration. Default 10.
    production : float64, optional
        Time in ps over which the frames are taken. Default 100.
    frames : int32, optional
        Number of frames. Default 200.
    symmetry : bool, optional
        Average the force constants over the symmetry operations of the cell,
        found by spglib: True, False, or by default whenever spglib is
        installed.
    seed : int32, optional
        Seed of the random numbers, positive. Default 1.
    progress : callable, optional
        Called as ``progress(done, total)`` with the steps run.
    cancel : CancelToken, optional
        Stops the calculation, which raises ``mdapy.errors.Cancelled``.

    Returns
    -------
    Phonons
        The phonons, whose ``cell`` has the mean box at this temperature.

    Raises
    ------
    InvalidParameter
        For invalid options, a box not periodic in every direction, too few
        frames, or if LAMMPS fails.
    MissingProperty
        If the system has no ``element`` column.
    Cancelled
        If cancelled.
    """
    if not isinstance(potential, LAMMPS):
        raise InvalidParameter("potential must be a mdapy.calculators.LAMMPS")
    if "element" not in cell.table.columns:
        raise MissingProperty("molecular dynamics needs the element column")
    if not cell.box.pbc.all() or len(cell) == 0:
        raise InvalidParameter(
            "phonons need a box periodic in every direction with atoms"
        )
    if not temperature > 0:
        raise InvalidParameter(f"temperature must be positive, not {temperature!r}")
    if not min(timestep, equilibration, production) > 0:
        raise InvalidParameter(
            "timestep, equilibration and production must be positive"
        )
    if int(frames) < 1 or int(seed) < 1:
        raise InvalidParameter("frames and seed must be positive")
    if supercell is None:
        matrix = _default_supercell(cell.box.matrix, 2 * potential.cutoff)
    else:
        matrix = _supercell_matrix(supercell)
    settle = max(int(round(equilibration / timestep)), 1)
    every = max(int(round(production / timestep)) // int(frames), 1)
    total = 2 * settle + every * int(frames)

    def check(done: int) -> None:
        if progress is not None:
            progress(done, total)
        if cancel is not None and cancel.cancelled:
            raise Cancelled("the phonons were cancelled")

    big = _supercell(cell, matrix)
    if pressure is not None:
        with MolecularDynamics(big, potential, timestep=timestep) as md:
            md.velocities(2 * temperature, seed=int(seed))
            md.run(
                settle,
                ensemble="npt",
                temperature=temperature,
                pressure=pressure,
                barostat="aniso",
                thermo=settle,
                frames=max(settle // 50, 1),
                cancel=cancel,
            )
            boxes = [frame.box.matrix for frame in md.frames]
        mean = np.mean(boxes[len(boxes) // 2 :], axis=0)
        # the box of the cell by the same strain as the mean box of the supercell,
        # with the symmetry of the crystal, which the average only comes close to
        scale = np.linalg.inv(big.box.matrix) @ mean
        operations = _symmetry(cell, symmetry)
        if operations is not None:
            turns = operations.cartesian
            scale = (turns @ scale @ turns.transpose(0, 2, 1)).mean(axis=0)
        box = Box(cell.box.matrix @ scale, origin=cell.box.origin)
        cell = cell.with_box(box, scale_positions=True)
        big = _supercell(cell, matrix)
    check(settle)
    samples = []
    with MolecularDynamics(big, potential, timestep=timestep) as md:
        md.velocities(2 * temperature, seed=int(seed) + 1)
        md.command(
            f"fix mdapy_thermostat all langevin {float(temperature)!r} "
            f"{float(temperature)!r} {_DAMPING} {int(seed) + 2} zero yes"
        )
        md.run(settle, ensemble="nve", thermo=settle, cancel=cancel)
        check(2 * settle)
        for k in range(int(frames)):
            md.run(every, ensemble="nve", thermo=every, cancel=cancel)
            check(2 * settle + (k + 1) * every)
            samples.append(md.system)
    with_forces = (
        frame.with_columns(forces=potential.compute(frame, ("forces",)).forces)
        for frame in samples
    )
    return phonons_from_frames(cell, with_forces, supercell=matrix, symmetry=symmetry)
