# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The box split among processes, as LAMMPS splits it: a grid of bricks in
fractional coordinates, one for each process, which owns the atoms inside it
and holds copies, ghosts, of those within a reach of it, periodic images
among them. Atoms move to the process of their brick, and ghosts are found
again, when the neighbor lists are built; in between, the ghosts follow their
atoms. Communication goes brick to brick, along x, then y, then z, each step
carrying what the steps before brought, so that the corners come too; along a
box vector of one brick there are no ghosts, the box being periodic there.

DomainAtoms holds the atoms of one process for MDAPY's dynamics on the
processor, with the interface of _HostAtoms: the calculator computes the atoms
of the brick and the ghosts as a system periodic only along the box vectors of
one brick, the
ghosts as far as the force on an atom reaches (twice the cutoff of a many-body
potential), so that the calculators need no change."""

import math
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from mdapy.calculators._base import Calculator
from mdapy.calculators._springs import Springs
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.simulation._integrator import _HostAtoms


def grid(processes: int, h: np.ndarray, pbc: np.ndarray) -> tuple[int, int, int]:
    """The bricks along each box vector: the factors of the number of
    processes that make the bricks of least surface, as LAMMPS's
    processors * * *."""
    area = np.array(
        [np.linalg.norm(np.cross(h[(d + 1) % 3], h[(d + 2) % 3])) for d in range(3)]
    )
    best, chosen = np.inf, (processes, 1, 1)
    for a in range(1, processes + 1):
        if processes % a:
            continue
        for b in range(1, processes // a + 1):
            if (processes // a) % b:
                continue
            c = processes // a // b
            counts = np.array([a, b, c])
            # the faces between bricks
            surface = float(np.sum(area * (counts - 1)))
            if surface < best - 1e-12:
                best, chosen = surface, (a, b, c)
    return chosen


class Bricks:
    """The bricks of a box among the processes of a communicator, and the
    atoms of this process: their fields, local ones first, then ghosts.

    Parameters
    ----------
    comm
        An mpi4py communicator.
    h : (3, 3) float64
        The box vectors in rows.
    lo : (3,) float64
        The origin of the box.
    pbc : (3,) bool
        The periodic directions.
    reach : float64
        How far from its brick a process holds ghosts, in Å.
    """

    def __init__(
        self, comm: Any, h: np.ndarray, lo: np.ndarray, pbc: Any, reach: float
    ):
        self.comm = comm
        self.pbc = np.asarray(pbc, dtype=bool)
        self.counts = np.array(grid(comm.Get_size(), h, self.pbc))
        rank = comm.Get_rank()
        # the coordinates of this brick, x fastest
        self.at = np.array(
            [
                rank % self.counts[0],
                (rank // self.counts[0]) % self.counts[1],
                rank // (self.counts[0] * self.counts[1]),
            ]
        )
        self.reach = float(reach)
        self.set_box(h, lo)
        self.local = 0  # the atoms this process owns, at the start of the arrays
        self.__sends: list = []

    def rank_of(self, at: np.ndarray) -> int:
        at = np.asarray(at) % self.counts
        return int(at[0] + self.counts[0] * (at[1] + self.counts[1] * at[2]))

    def set_box(self, h: np.ndarray, lo: np.ndarray) -> None:
        self.h = np.array(h, dtype=np.float64)
        self.lo = np.array(lo, dtype=np.float64)
        self.inverse = np.linalg.inv(self.h)
        thick = 1.0 / np.linalg.norm(self.inverse, axis=0)
        # the reach in fractional coordinates along each box vector
        self.fraction = self.reach / thick
        width = 1.0 / self.counts
        too_thin = (self.fraction > width + 1e-12) & (self.counts > 1)
        if too_thin.any():
            d = int(np.flatnonzero(too_thin)[0])
            raise ValueError(
                f"the bricks of {self.counts.prod()} processes are "
                f"{thick[d] * width[d]:.3f} Å thick along box vector {d + 1}, less "
                f"than the reach of the ghosts, {self.reach:.3f} Å: use fewer processes"
            )

    def fractional(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) @ self.inverse

    def migrate(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The local atoms after each went to the process of its brick:
        wrapped into the box along periodic directions (``image`` counts
        the boxes), then passed brick to brick along x, y and z. fields holds
        (n, ...) arrays of the local atoms: ``positions`` and ``image``
        among them."""
        fields = {k: np.ascontiguousarray(v) for k, v in fields.items()}
        s = self.fractional(fields["positions"])
        whole = np.where(self.pbc, np.floor(s), 0.0)
        if whole.any():
            fields["positions"] = fields["positions"] - whole @ self.h
            fields["image"] = fields["image"] + whole.astype(fields["image"].dtype)
            s = s - whole
        for d in range(3):
            if self.counts[d] == 1:
                continue
            s = self.fractional(fields["positions"])
            brick = np.floor(s[:, d] * self.counts[d]).astype(np.int64)
            brick = np.clip(brick, 0, self.counts[d] - 1)
            step = brick - self.at[d]
            if self.pbc[d]:
                # the nearer way round, the next brick across the boundary
                half = self.counts[d] // 2
                step = (step + half) % self.counts[d] - half
            if self.maximum(int(np.abs(step).max(initial=0))) > 1:
                raise InvalidParameter(
                    "an atom moved farther than a brick between two builds of "
                    "the neighbor lists: the time step may be too long"
                )
            keep = step == 0
            received = []
            for side in (-1, 1):
                destination, source = self.__neighbors(d, side)
                sent = {k: v[step == side] for k, v in fields.items()}
                got = _exchange_fields(self.comm, sent, destination, source)
                received.append(got)
            fields = {
                k: np.concatenate([v[keep]] + [r[k] for r in received])
                for k, v in fields.items()
            }
        self.local = len(fields["positions"])
        return fields

    def __neighbors(self, d: int, side: int) -> tuple[int | None, int | None]:
        """The process this one sends to on a side along a box vector, and
        the one it receives from on the other; None at a boundary that is
        not periodic."""
        last = self.counts[d] - 1
        to, back = self.at.copy(), self.at.copy()
        to[d] += side
        back[d] -= side
        destination = (
            None if not self.pbc[d] and not 0 <= to[d] <= last else self.rank_of(to)
        )
        source = (
            None if not self.pbc[d] and not 0 <= back[d] <= last else self.rank_of(back)
        )
        return destination, source

    def borders(
        self, positions: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The positions and ids of the local atoms followed by those of the
        ghosts within the reach of this brick, periodic images among them;
        the lists of what went where are kept for ``follow``."""
        comm = self.comm
        x, idx = np.ascontiguousarray(positions), np.ascontiguousarray(ids)
        self.__sends = []
        lower = self.at / self.counts
        upper = (self.at + 1) / self.counts
        for d in range(3):
            # along a box vector of one brick the box itself is periodic
            if self.counts[d] == 1:
                continue
            # the atoms before this direction: local ones and the ghosts of
            # the directions before, which carry the corners
            before = len(x)
            s = self.fractional(x)[:, d]
            for side in (-1, 1):
                # those in the slab of the reach along the face on this side
                # go to the brick there, shifted by a box vector across the
                # boundary
                if side < 0:
                    near = (s >= lower[d]) & (s < lower[d] + self.fraction[d])
                    edge = self.at[d] == 0
                else:
                    near = (s < upper[d]) & (s >= upper[d] - self.fraction[d])
                    edge = self.at[d] == self.counts[d] - 1
                chosen = np.flatnonzero(near[:before])
                if edge and not self.pbc[d]:
                    chosen = chosen[:0]
                shift = np.zeros(3)
                if edge:
                    shift[d] = -side  # in box vectors, added to what is sent
                destination, source = self.__neighbors(d, side)
                sent = (x[chosen] + shift @ self.h, idx[chosen])
                got = _exchange(comm, sent, destination, source)
                self.__sends.append((chosen, shift, destination, source))
                x = np.concatenate([x, got[0]])
                idx = np.concatenate([idx, got[1]])
        return x, idx

    def follow(self, positions: np.ndarray) -> np.ndarray:
        """The positions of the ghosts again, from those of the local atoms,
        along the lists of the last ``borders``."""
        x = np.array(positions[: self.local])
        for chosen, shift, destination, source in self.__sends:
            got = _exchange(
                self.comm, (x[chosen] + shift @ self.h,), destination, source
            )[0]
            x = np.concatenate([x, got])
        return x

    def set_reach(self, reach: float) -> None:
        self.reach = float(reach)
        self.set_box(self.h, self.lo)

    def total(self, value: Any) -> Any:
        """The sum over the processes."""
        return self.comm.allreduce(value)

    def maximum(self, value: Any) -> Any:
        """The largest over the processes."""
        return max(self.comm.allgather(value))


def _exchange(
    comm: Any, sent: tuple, destination: int | None, source: int | None
) -> tuple:
    """What a process sends one way and receives from the other; empty
    arrays where there is no process, at a boundary that is not periodic
    (never on both sides: a direction of one brick that is not periodic
    sends nothing)."""
    if destination is None:
        return comm.recv(source=source)
    if source is None:
        comm.send(sent, dest=destination)
        return tuple(a[:0] for a in sent)
    return comm.sendrecv(sent, dest=destination, source=source)


def _exchange_fields(
    comm: Any, sent: dict[str, np.ndarray], destination: int | None, source: int | None
) -> dict[str, np.ndarray]:
    """The fields of the atoms a process sends one way and receives from the
    other, as _exchange."""
    if destination is None:
        return comm.recv(source=source)
    if source is None:
        comm.send(sent, dest=destination)
        return {k: v[:0] for k, v in sent.items()}
    return comm.sendrecv(sent, dest=destination, source=source)


def world() -> Any:
    """The communicator of all the processes when this one is one of several
    under MPI, as mpirun, mpiexec or srun start them, or after mpi4py was
    imported; else None."""
    launched = False
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"):
        try:
            launched = launched or int(os.environ.get(name, "1")) > 1
        except ValueError:
            continue
    if not launched and "mpi4py.MPI" not in sys.modules:
        return None
    try:
        from mpi4py import MPI
    except ImportError:
        raise InvalidParameter(
            "MDAPY's dynamics on several processes needs mpi4py"
        ) from None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


def _reach(calculator: Any, target: bool = False) -> tuple[float, float]:
    """How far from an atom the atoms that its force depends on are, and
    those whose neighbors it needs: the cutoff and none for a pair potential;
    twice the cutoff and the cutoff for MDAPY's many-body potentials, whose
    forces depend on what the neighbors have of their neighbors, as the
    density of EAM. The springs of a Switch tie each atom to its site alone."""
    from mdapy.calculators import ADP, EAM, MEAM, NEP, StillingerWeber, Tersoff
    from mdapy.calculators._pair import _PairCalculator

    if isinstance(calculator, _PairCalculator) or (
        target and isinstance(calculator, Springs)
    ):
        return float(calculator.cutoff), 0.0
    if isinstance(calculator, (EAM, ADP, MEAM, NEP, StillingerWeber, Tersoff)):
        return 2.0 * float(calculator.cutoff), float(calculator.cutoff)
    raise InvalidParameter(
        "MDAPY's dynamics on several processes splits MDAPY's own potentials, EAM, "
        "ADP, MEAM, NEP, Tersoff, Stillinger-Weber and the pair potentials, not "
        f"{type(calculator).__name__}: run it on one process"
    )


def _centered(neighbors: NeighborList, centers: int, symmetric: bool) -> NeighborList:
    """The pairs of the first atoms, the centers, alone: the others have no
    neighbors, and the calculator spends nothing on them. symmetric: the
    pairs of a center and another kept both ways, for the potentials that
    give each pair's terms to its other atom through the pair back; no term
    of another reaches the atoms of the process, whose neighbors are all
    centers."""
    offsets = np.asarray(neighbors.offsets)
    n = len(offsets) - 1
    if symmetric:
        first = np.repeat(np.arange(n), np.diff(offsets))
        kept = (first < centers) | (np.asarray(neighbors.j) < centers)
        counts = np.bincount(first[kept], minlength=n)
    else:
        kept = slice(0, int(offsets[centers]))
        counts = np.diff(offsets)
        counts[centers:] = 0
    return NeighborList(
        neighbors.system,
        neighbors.cutoff,
        np.concatenate([[0], np.cumsum(counts)]).astype(offsets.dtype),
        np.array(neighbors.j[kept]),
        np.array(neighbors.shifts[kept]),
        np.array(neighbors.distances[kept]),
    )


class DomainAtoms(_HostAtoms):
    """The atoms of one MPI process in MDAPY's dynamics on the processor:
    those in its brick of the box, wrapped into the box along the periodic
    directions with the boxes they crossed counted, and the ghosts within the
    reach of the potential and the skin. The calculator computes the atoms of
    the brick and the ghosts as a system without periodic directions; the
    forces of the atoms of the brick are those of the whole system, and their
    energies and virials sum to its energy and virial. The arrays of all the
    atoms that the dynamics reads, positions, velocities and forces, are
    gathered on every process, in the order of the atoms."""

    def __init__(
        self,
        comm: Any,
        calculator: Calculator,
        x: np.ndarray,
        v: np.ndarray,
        masses: np.ndarray,
        restrict: Any,
        box: Box,
        skin: float,
    ):
        self.__skin = float(skin)
        try:
            self.__bricks = Bricks(
                comm,
                box.matrix,
                box.origin,
                box.pbc,
                _reach(calculator)[0] + self.__skin,
            )
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        self.__masses_of_all = masses
        super().__init__(calculator, x, v, masses, restrict)
        self.__distribute(x, v, None)

    def __distribute(
        self, x: np.ndarray, v: np.ndarray, forces: np.ndarray | None
    ) -> None:
        """This process's atoms of arrays of all the atoms."""
        bricks = self.__bricks
        self._count = len(x)
        s = bricks.fractional(x)
        whole = np.where(bricks.pbc, np.floor(s), 0.0)
        brick = np.clip(
            np.floor((s - whole) * bricks.counts), 0, bricks.counts - 1
        ).astype(np.int64)
        mine = np.flatnonzero((brick == bricks.at).all(axis=1))
        self._x = x[mine] - whole[mine] @ bricks.h
        self._v = np.array(v[mine], dtype=float)
        self._forces = None if forces is None else np.array(forces[mine], dtype=float)
        self.__image = whole[mine].astype(np.int64)
        self.__mine(mine.astype(np.int64))
        bricks.local = len(mine)
        self.__built: np.ndarray | None = None  # the positions at the last build
        self.__ghosts: np.ndarray = self._x  # this process's atoms and the ghosts
        # the order of the ghosts, nearest the brick first, and how far each is
        self.__order: np.ndarray = np.arange(len(mine))
        self.__far: np.ndarray = np.zeros(0)
        self.__part: Any = None  # their table
        self.__box: Box | None = None  # their box, not periodic

    def __mine(self, ids: np.ndarray) -> None:
        self._ids = ids
        self._rows = ids
        self._masses = self.__masses_of_all[ids]

    def __gather(self, values: np.ndarray) -> np.ndarray:
        """Values of this process's atoms as an array of all, on every
        process."""
        parts = self.__bricks.comm.allgather((self._ids, values))
        out = np.empty((self._count,) + values.shape[1:], dtype=values.dtype)
        for ids, part in parts:
            out[ids] = part
        return out

    def _total(self, value: Any) -> Any:
        return self.__bricks.total(value)

    def _agreed(self, value: Any) -> Any:
        return self.__bricks.comm.bcast(value, root=0)

    def _least(self, value: Any) -> Any:
        return np.min(np.stack(self.__bricks.comm.allgather(np.asarray(value))), axis=0)

    @property
    def positions(self) -> np.ndarray:
        """Unwrapped, as on one process."""
        return self.__gather(self._x + self.__image @ self.__bricks.h)

    @property
    def velocities(self) -> np.ndarray:
        return self.__gather(self._v)

    @velocities.setter
    def velocities(self, v: np.ndarray) -> None:
        self._v = np.array(v[self._ids], dtype=float)

    @property
    def forces(self) -> np.ndarray:
        return self.__gather(self._forces)

    def compute(
        self, table: pl.DataFrame, box: Box, energy: bool, virial: bool
    ) -> tuple[float, np.ndarray]:
        self.__update(table, box)
        return super().compute(table, box, energy, virial)

    def remap(
        self, old_lo: np.ndarray, old: np.ndarray, lo: np.ndarray, new: np.ndarray
    ) -> None:
        """The atoms carried with the box, and the bricks with them: the
        boxes the atoms crossed count in the new box at once, and the
        ghosts are found again."""
        super().remap(old_lo, old, lo, new)
        self.__set_box(new, lo)

    def __set_box(self, h: np.ndarray, lo: np.ndarray) -> None:
        try:
            self.__bricks.set_box(h, lo)
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        self.__built = None

    def __update(self, table: pl.DataFrame, box: Box) -> None:
        """The atoms moved to the processes of their bricks and the ghosts
        found again when the box changed or an atom moved half the skin since
        the last time; else the ghosts moved with their atoms."""
        bricks = self.__bricks
        if not (
            np.array_equal(box.matrix, bricks.h)
            and np.array_equal(box.origin, bricks.lo)
        ):
            self.__set_box(box.matrix, box.origin)
        if self.__built is not None:
            step = self._x - self.__built
            moved = math.sqrt(float((step * step).sum(axis=1).max(initial=0.0)))
            if not 2.0 * bricks.maximum(moved) < self.__skin:
                self.__built = None
        if self.__built is None:
            self.__build(table, box)
        else:
            self.__ghosts = bricks.follow(self._x)[self.__order]

    def __build(self, table: pl.DataFrame, box: Box) -> None:
        """The atoms moved to their bricks and the ghosts found; the rows of
        the table of all the atoms, fixed in a run, taken for both."""
        bricks = self.__bricks
        fields = bricks.migrate(
            {"positions": self._x, "velocities": self._v, "image": self.__image,
             "id": self._ids}
        )  # fmt: skip
        self._x, self._v = fields["positions"], fields["velocities"]
        self.__image = fields["image"]
        self.__mine(fields["id"])
        ghosts, ids = bricks.borders(self._x, self._ids)
        # the ghosts nearest the brick first, so that those whose neighbors a
        # potential needs come before the others
        n = bricks.local
        far = self.__beyond(ghosts[n:])
        order = np.argsort(far, kind="stable")
        self.__order = np.concatenate([np.arange(n), n + order])
        self.__far = far[order]
        self.__ghosts, ids = ghosts[self.__order], ids[self.__order]
        self.__part = table[ids]
        self.__box = Box(
            box.matrix, origin=box.origin, pbc=bricks.pbc & (bricks.counts == 1)
        )
        self.__built = self._x.copy()
        self._lists = {}

    def __beyond(self, x: np.ndarray) -> np.ndarray:
        """How far atoms are from the brick, in Å across its faces; the end
        bricks along a direction that is not periodic reach out of the box."""
        bricks = self.__bricks
        s = bricks.fractional(x)
        closed = bricks.pbc
        lower = np.where(~closed & (bricks.at == 0), -np.inf, bricks.at / bricks.counts)
        upper = np.where(
            ~closed & (bricks.at == bricks.counts - 1),
            np.inf,
            (bricks.at + 1) / bricks.counts,
        )
        out = np.maximum(np.maximum(lower - s, s - upper), 0.0)
        thick = 1.0 / np.linalg.norm(bricks.inverse, axis=0)
        return (out * thick).max(axis=1, initial=0.0)

    def __neighbors(self, slot: str, current: System, calculator: Any) -> Any:
        """The pairs of the atoms whose neighbors the calculator needs: this
        process's atoms, and for a many-body potential the ghosts within its
        cutoff and the skin of the brick."""
        cutoff = float(calculator.cutoff)
        wide = self._lists.get(slot)
        neighbors = wide._follow(current, cutoff) if wide is not None else None
        if neighbors is None:
            centers = self.__bricks.local
            inner = _reach(calculator, target=True)[1]
            if inner > 0.0:
                centers += int(
                    np.searchsorted(self.__far, inner + self.__skin, "right")
                )
            from mdapy.calculators import MEAM, NEP, StillingerWeber, Tersoff

            full = NeighborList.build(current, cutoff + self.__skin)
            symmetric = isinstance(calculator, (MEAM, NEP, StillingerWeber, Tersoff))
            self._lists[slot] = wide = _centered(full, centers, symmetric)
            neighbors = wide._follow(current, cutoff)
        return neighbors

    def _evaluate(
        self, slot: str, calculator: Any, table: pl.DataFrame, box: Box, wanted: tuple
    ) -> tuple[float, np.ndarray, np.ndarray, Any]:
        n = self.__bricks.local
        virial = "virial" in wanted
        if isinstance(calculator, Springs):
            # each atom tied to its own site: this process's atoms alone, in
            # the periodic box, at their unwrapped places
            springs = Springs(
                calculator.sites[self._ids], calculator.constants[self._ids]
            )
            current = System(self._x + self.__image @ self.__bricks.h, box)
            result = springs.compute(current, ("energy", "atomic_energies", "forces"))
            energies = np.asarray(result.atomic_energies, dtype=float)
            return (
                float(self._total(energies.sum())),
                np.asarray(result.forces, dtype=float),
                np.full((3, 3), np.nan),
                SimpleNamespace(atomic_energies=energies, atomic_virials=None),
            )
        extra = ("atomic_energies",) + (("atomic_virials",) if virial else ())
        wanted = tuple(wanted) + tuple(p for p in extra if p not in wanted)
        current = System(table=self.__part.with_columns(
            pl.Series("positions", self.__ghosts.copy(), dtype=pl.Array(pl.Float64, 3))
        ), box=self.__box)  # fmt: skip
        neighbors = self.__neighbors(slot, current, calculator)
        result = calculator.compute(current, wanted, neighbors=neighbors)
        energies = np.asarray(result.atomic_energies, dtype=float)[:n]
        virials = (
            np.asarray(result.atomic_virials, dtype=float)[:n]
            if result.atomic_virials is not None
            else None
        )
        work = self._total(virials.sum(axis=0)) if virial else np.full((3, 3), np.nan)
        return (
            float(self._total(energies.sum())),
            np.asarray(result.forces, dtype=float)[:n],
            work,
            SimpleNamespace(atomic_energies=energies, atomic_virials=virials),
        )

    def set_switch(self, target: Any) -> None:
        super().set_switch(target)
        if target:
            reach = _reach(target, target=True)[0] + self.__skin
            if reach > self.__bricks.reach:
                try:
                    self.__bricks.set_reach(reach)
                except ValueError as error:
                    raise InvalidParameter(str(error)) from None
                self.__built = None

    # this process's part of the atoms: found again from a restart
    _TRANSIENT = _HostAtoms._TRANSIENT + (
        "_rows",
        "_DomainAtoms__bricks",
        "_DomainAtoms__masses_of_all",
        "_DomainAtoms__image",
        "_DomainAtoms__built",
        "_DomainAtoms__ghosts",
        "_DomainAtoms__order",
        "_DomainAtoms__far",
        "_DomainAtoms__part",
        "_DomainAtoms__box",
        "_DomainAtoms__skin",
    )

    def state(self) -> dict[str, Any]:
        """As on one process: arrays of all the atoms, in their order, so
        that a restart does not depend on the number of processes."""
        kept = super().state()
        kept["_x"] = self.positions
        kept["_v"] = self.velocities
        kept["_forces"] = None if self._forces is None else self.forces
        kept["_masses"] = self.__masses_of_all
        kept["_ids"] = np.arange(self._count, dtype=np.int64)
        for name in ("_HostAtoms__virials", "_HostAtoms__energies"):
            if kept.get(name) is not None:
                kept[name] = self.__gather(kept[name])
        return kept

    def restore(self, state: dict[str, Any]) -> None:
        super().restore(state)
        self.__distribute(state["_x"], state["_v"], state["_forces"])
        for name in ("_HostAtoms__virials", "_HostAtoms__energies"):
            if state.get(name) is not None:
                vars(self)[name] = state[name][self._ids]
