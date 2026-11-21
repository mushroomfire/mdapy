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

import hashlib
import math
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from mdapy._core import calculators as _native_calculators
from mdapy.calculators._base import WITHOUT_VIRIAL, Calculator
from mdapy.calculators._springs import Springs
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.threads import get_num_threads
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.simulation._integrator import _counted, _HostAtoms


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
        # the process that is not, of mpi4py's MPI, or None for others
        mpi = sys.modules.get("mpi4py.MPI")
        real = mpi is not None and isinstance(comm, mpi.Comm)
        self.__null = mpi.PROC_NULL if real else None
        # the operations of the sums, of mpi4py's MPI, or named for others
        self.__sum = mpi.SUM if real else "sum"
        self.__max = mpi.MAX if real else "max"

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
        # bricks thinner than the reach pass the ghosts on, brick to brick,
        # as many times as it takes; a reach beyond the box would need images
        # of images along a vector cut into bricks
        too_far = (self.fraction >= 1.0) & (self.counts > 1)
        if too_far.any():
            d = int(np.flatnonzero(too_far)[0])
            raise ValueError(
                f"the box is {thick[d]:.3f} Å thick along box vector {d + 1}, less "
                f"than the reach of the ghosts, {self.reach:.3f} Å: do not cut it "
                f"into bricks along that vector, with fewer processes"
            )
        self.hops = np.maximum(np.ceil(self.fraction * self.counts - 1e-12), 1).astype(
            int
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
            # first the atoms before this direction: local ones and the ghosts
            # of the directions before, which carry the corners; then, when
            # the bricks are thinner than the reach, those that came the hop
            # before from the other side, passed on
            before = len(x)
            came = {-1: np.arange(before), 1: np.arange(before)}
            for _ in range(int(self.hops[d])):
                s = self.fractional(x)[:, d]
                for side in (-1, 1):
                    # those in the slab of the reach along the face on this
                    # side go to the brick there, shifted by a box vector
                    # across the boundary
                    candidates = came[side]
                    t = s[candidates]
                    if side < 0:
                        near = (t >= lower[d]) & (t < lower[d] + self.fraction[d])
                        edge = self.at[d] == 0
                    else:
                        near = (t < upper[d]) & (t >= upper[d] - self.fraction[d])
                        edge = self.at[d] == self.counts[d] - 1
                    chosen = candidates[near]
                    if edge and not self.pbc[d]:
                        chosen = chosen[:0]
                    shift = np.zeros(3)
                    if edge:
                        shift[d] = -side  # in box vectors, added to what is sent
                    destination, source = self.__neighbors(d, side)
                    sent = (x[chosen] + shift @ self.h, idx[chosen])
                    got = _exchange(comm, sent, destination, source)
                    # what went where, and where what came is among the atoms
                    count = len(got[0])
                    self.__sends.append(
                        (chosen, shift, destination, source, len(x), count)
                    )
                    came[side] = np.arange(len(x), len(x) + count)
                    x = np.concatenate([x, got[0]])
                    idx = np.concatenate([idx, got[1]])
        return x, idx

    def follow(self, positions: np.ndarray) -> np.ndarray:
        """The positions of the ghosts again, from those of the local atoms,
        along the lists of the last ``borders``."""
        return self.forward(positions[: self.local], shifted=True)

    def forward(self, values: np.ndarray, shifted: bool = False) -> np.ndarray:
        """Values of the local atoms, (local, ...), and those of their ghosts
        after them, along the lists of the last ``borders``; positions,
        shifted, by the box vectors the ghosts crossed. The ghosts of each
        message come straight into their rows, the buffers of MPI."""
        values = np.asarray(values)
        last = self.__sends[-1] if self.__sends else None
        size = last[4] + last[5] if last is not None else len(values)
        x = np.empty((size,) + values.shape[1:], dtype=values.dtype)
        x[: self.local] = values
        for chosen, shift, destination, source, start, count in self.__sends:
            sent = x[chosen] + shift @ self.h if shifted else x[chosen]
            self.__swap(sent, destination, x[start : start + count], source)
        return x

    def reverse(self, values: np.ndarray) -> np.ndarray:
        """Values of the local atoms and the ghosts, (all, ...), the values of
        each ghost added to those of its atom: the values of the local atoms.
        The lists of the last ``borders`` are gone through backwards, so that
        the ghosts of ghosts come back through them."""
        x = np.array(values)
        for chosen, _, destination, source, start, count in reversed(self.__sends):
            back = np.empty((len(chosen),) + x.shape[1:], dtype=x.dtype)
            self.__swap(x[start : start + count], source, back, destination)
            x[chosen] += back
        return x[: self.local]

    def __swap(
        self,
        sent: np.ndarray,
        destination: int | None,
        received: np.ndarray,
        source: int | None,
    ) -> None:
        """Rows sent one way, the rows of received filled from the other, of
        known numbers, without pickles; nothing where there is no process."""
        null = self.__null
        self.comm.Sendrecv(
            np.ascontiguousarray(sent),
            dest=null if destination is None else destination,
            recvbuf=received,
            source=null if source is None else source,
        )

    def set_reach(self, reach: float) -> None:
        self.reach = float(reach)
        self.set_box(self.h, self.lo)

    def total(self, value: Any) -> Any:
        """The sum over the processes: of an array of float64 through the
        buffers of MPI, as each step sums one."""
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            out = np.empty_like(value)
            self.comm.Allreduce(np.ascontiguousarray(value), out, op=self.__sum)
            return out
        return self.comm.allreduce(value)

    def maximum(self, value: float) -> float:
        """The largest over the processes."""
        out = np.empty(1)
        self.comm.Allreduce(np.array([float(value)]), out, op=self.__max)
        return float(out[0])


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

    def more_than_one(name: str) -> bool:
        try:
            return int(os.environ.get(name, "1")) > 1
        except ValueError:
            return False

    # mpirun of Open MPI, PMI of MPICH, Intel MPI or srun --mpi=pmi2, MVAPICH;
    # srun --mpi=pmix names its tasks alone
    launched = any(
        more_than_one(name)
        for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE")
    ) or ("PMIX_RANK" in os.environ and more_than_one("SLURM_STEP_NUM_TASKS"))
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


def resolved(comm: Any) -> Any:
    """The communicator of the processes that share one simulation: those
    MPI started when comm is None (world), or comm, an mpi4py communicator;
    None for one process alone."""
    if comm is None:
        return world()
    if not all(
        callable(getattr(comm, name, None)) for name in ("Get_size", "Get_rank")
    ):
        raise InvalidParameter(f"comm must be a communicator of mpi4py, not {comm!r}")
    return comm if comm.Get_size() > 1 else None


def digest(*values: Any) -> bytes:
    """A digest of numbers, strings and arrays, in tuples and objects: an
    object given by its type and its attributes, whatever its repr; one
    within it by its repr when it has one of its own."""
    h = hashlib.blake2b(digest_size=16)

    def add(value: Any, depth: int) -> None:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            h.update(repr((array.dtype.str, array.shape)).encode())
            h.update(array.tobytes())
        elif value is None or isinstance(
            value, (bool, int, float, complex, str, bytes, np.generic)
        ):
            h.update(repr(value).encode())
        elif isinstance(value, (list, tuple)):
            h.update(b"(")
            for item in value:
                add(item, depth)
            h.update(b")")
        elif isinstance(value, dict):
            h.update(b"{")
            for key in sorted(value, key=str):
                h.update(str(key).encode())
                add(value[key], depth)
            h.update(b"}")
        elif depth > 0 and type(value).__repr__ is not object.__repr__:
            # an object within, as a calculator, whose caches may differ from
            # process to process: by what it says it is
            h.update(repr(value).encode())
        else:
            h.update(type(value).__name__.encode())
            fields = getattr(value, "__dict__", None)
            if fields is not None and depth < 3:
                add(fields, depth + 1)
        h.update(b"|")

    for value in values:
        add(value, 0)
    return h.digest()


def of_system(system: System) -> tuple:
    """What the processes of one simulation must hold alike of its atoms."""
    table = system.table
    values: list[Any] = [
        len(system),
        system.box.matrix,
        system.box.origin,
        system.box.pbc,
    ]
    for name in ("positions", "velocities", "mass", "movable"):
        if name in table.columns:
            values += [name, np.asarray(table[name].to_numpy())]
    if "element" in table.columns:
        values += ["element", table["element"].to_physical().to_numpy()]
    return tuple(values)


def agree(comm: Any, what: str, *values: Any) -> None:
    """Raises on every process of comm unless each gives the same values:
    the processes of one simulation run the same script."""
    if comm is None:
        return
    digests = comm.allgather(digest(*values))
    other = next((r for r, d in enumerate(digests) if d != digests[0]), None)
    if other is not None:
        raise InvalidParameter(
            f"the processes of one simulation differ in {what}: process {other} "
            "from process 0. They run one simulation together, split among them; "
            "for a simulation on each process alone, give comm=MPI.COMM_SELF, or "
            "a communicator of each group of processes"
        )


def _reach(calculator: Any, target: bool = False) -> tuple[float, float]:
    """How far from the brick a process needs ghosts, and those whose
    neighbors it needs too: the cutoff, and none, for the potentials that
    compute the atoms of a process alone (``_split``), sending what each atom
    has of its neighbors to the ghosts or the forces on the ghosts back, and
    the pair potentials; for MEAM, whose screening reaches the neighbors of
    the neighbors, twice the cutoff and the cutoff. The springs of a Switch
    tie each atom to its site alone."""
    from mdapy.calculators import ADP, EAM, MEAM, NEP, StillingerWeber, Tersoff
    from mdapy.calculators._overlay import Overlay
    from mdapy.calculators._pair import _PairCalculator

    pair = isinstance(calculator, _PairCalculator)
    if isinstance(calculator, Overlay) and calculator._split is not None:
        # split as its calculators, each with its own reach
        return (
            max(_reach(p, target)[0] for p in calculator.calculators),
            max(_reach(p, target)[1] for p in calculator.calculators),
        )
    if isinstance(calculator, Overlay) and not _on_gpu(calculator):
        # computed whole, the atoms and the ghosts: a pair potential needs
        # ghosts within its cutoff, a many-body one within twice its cutoff,
        # with the neighbors of those within its cutoff
        reaches = []
        for part in calculator.calculators:
            _reach(part)  # each one that splits
            c = float(part.cutoff)
            reaches.append(
                (c, 0.0) if isinstance(part, _PairCalculator) else (2.0 * c, c)
            )
        return max(r for r, _ in reaches), max(c for _, c in reaches)
    if isinstance(calculator, MEAM) or (
        _on_gpu(calculator) and not pair
    ):  # fmt: skip
        # MEAM, and any many-body potential on the GPU, which computes the
        # atoms and the ghosts whole
        return 2.0 * float(calculator.cutoff), float(calculator.cutoff)
    if (
        isinstance(calculator, (_PairCalculator, EAM, ADP, NEP, StillingerWeber, Tersoff))
        or (target and isinstance(calculator, Springs))
    ):  # fmt: skip
        return float(calculator.cutoff), 0.0
    raise InvalidParameter(
        "MDAPY's dynamics on several processes splits MDAPY's own potentials, EAM, "
        "ADP, MEAM, NEP, Tersoff, Stillinger-Weber and the pair potentials, not "
        f"{type(calculator).__name__}: run it on one process"
    )


def _on_gpu(calculator: Any) -> bool:
    return getattr(calculator, "device", "cpu") == "cuda"


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
        # of the last build: its system, and the neighbor lists and species of
        # the calculators, by slot
        self.__system: System | None = None
        self.__kept: dict = {}
        self.__matrix = np.ascontiguousarray(bricks.h)
        self.__sub_ids: np.ndarray = self._ids  # the ids of them
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
        self.__sub_ids = np.ascontiguousarray(ids, dtype=np.int64)
        self.__part = table[ids]
        self.__box = Box(
            box.matrix, origin=box.origin, pbc=bricks.pbc & (bricks.counts == 1)
        )
        self.__matrix = np.ascontiguousarray(self.__box.matrix)
        self.__system = None
        self.__kept = {}
        self.__built = self._x.copy()
        self._lists = {}

    def __built_system(self) -> System:
        """The atoms and the ghosts at the last build, as a system."""
        if self.__system is None:
            self.__system = System(table=self.__part.with_columns(
                pl.Series("positions", self.__ghosts.copy(),
                          dtype=pl.Array(pl.Float64, 3))
            ), box=self.__box)  # fmt: skip
        return self.__system

    def __forward(self, values: np.ndarray) -> np.ndarray:
        """Values of this process's atoms, and of the ghosts after them."""
        return self.__bricks.forward(values)[self.__order]

    def __reverse(self, values: np.ndarray) -> np.ndarray:
        """Values of the atoms and the ghosts: those of this process's atoms,
        the ghosts' added."""
        ordered = np.empty_like(values)
        ordered[self.__order] = values
        return self.__bricks.reverse(ordered)

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

    def _wide(self, current: System, calculator: Any) -> NeighborList:
        """The pairs of the atoms whose neighbors the calculator needs: this
        process's atoms, and for a many-body potential the ghosts within its
        cutoff and the skin of the brick."""
        from mdapy.calculators import MEAM, NEP, StillingerWeber, Tersoff

        centers = self.__bricks.local
        inner = _reach(calculator, target=True)[1]
        if inner > 0.0:
            centers += int(np.searchsorted(self.__far, inner + self.__skin, "right"))
        full = NeighborList.build(current, float(calculator.cutoff) + self.__skin)
        parts = getattr(calculator, "calculators", (calculator,))
        symmetric = any(
            isinstance(p, (MEAM, NEP, StillingerWeber, Tersoff)) for p in parts
        )
        return _centered(full, centers, symmetric)

    def _evaluate(
        self, slot: str, calculator: Any, table: pl.DataFrame, box: Box, wanted: tuple
    ) -> tuple[float, np.ndarray, np.ndarray, Any]:
        n = self.__bricks.local
        virial = "virial" in wanted and WITHOUT_VIRIAL not in wanted
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
                float(energies.sum()),
                np.asarray(result.forces, dtype=float),
                np.full((3, 3), np.nan),
                SimpleNamespace(atomic_energies=energies, atomic_virials=None),
            )
        # a potential on the GPU computes the atoms and the ghosts whole,
        # with a neighbor list of its own
        split = None if _on_gpu(calculator) else getattr(calculator, "_split", None)
        if split is not None:
            # the calculator computes this process's atoms alone, and sends
            # what it needs to the ghosts or back from them; its neighbor list
            # and species, of the system of the last build, serve until the
            # next, which comes before an atom has moved half the skin
            kept = self.__kept.get(slot)
            if kept is None:
                built = self.__built_system()
                wide = self._wide(built, calculator)
                # on one thread, the pairs this process computes once among
                # the processes, as LAMMPS's newton on
                once = None
                if getattr(calculator, "_pairs_once", False) and get_num_threads() == 1:
                    once = _native_calculators.once_pairs(
                        wide.offsets, wide.j, wide.shifts, n, self.__sub_ids,
                        np.ascontiguousarray(self.__ghosts), self.__matrix,
                    )  # fmt: skip
                kept = self.__kept[slot] = (
                    wide,
                    np.ascontiguousarray(calculator._species(built)),
                    once,
                )
            neighbors, species, once = kept
            atoms = (species, np.ascontiguousarray(self.__ghosts), self.__matrix)
            part = split(atoms, tuple(wanted), neighbors, n, self.__forward,
                         self.__reverse, self.__sub_ids, once)  # fmt: skip
            return (
                float(part.energy),
                np.asarray(part.forces, dtype=float),
                part.virial if virial else np.full((3, 3), np.nan),
                SimpleNamespace(atomic_energies=part.atomic_energies,
                                atomic_virials=part.atomic_virials),
            )  # fmt: skip
        current = System(table=self.__part.with_columns(
            pl.Series("positions", self.__ghosts.copy(), dtype=pl.Array(pl.Float64, 3))
        ), box=self.__box)  # fmt: skip
        neighbors = (
            None if _on_gpu(calculator) else self._neighbors(slot, current, calculator)
        )
        extra = ("atomic_energies",) + (("atomic_virials",) if virial else ())
        wanted = tuple(wanted) + tuple(p for p in extra if p not in wanted)
        result = calculator.compute(current, wanted, neighbors=neighbors)
        energies = np.asarray(result.atomic_energies, dtype=float)[:n]
        virials = (
            np.asarray(result.atomic_virials, dtype=float)[:n]
            if result.atomic_virials is not None
            else None
        )
        work = virials.sum(axis=0) if virial else np.full((3, 3), np.nan)
        return (
            float(energies.sum()),
            np.asarray(result.forces, dtype=float)[:n],
            work,
            SimpleNamespace(atomic_energies=energies, atomic_virials=virials),
        )

    def _neighbor_counts(self, which: np.ndarray, cutoff: float) -> np.ndarray:
        """From the list of the potential over this process's atoms and the
        ghosts, or for a potential on the GPU a list of the processor's own,
        kept until the next build."""
        kept = self.__kept.get("potential")
        wide = kept[0] if kept is not None else self._lists.get("potential")
        if wide is None:
            wide = self.__kept.get("stopping")
            if wide is None:
                wide = self.__kept["stopping"] = NeighborList.build(
                    self.__built_system(), cutoff + self.__skin
                )
        return _counted(wide, self.__ghosts, self.__matrix, which, cutoff)

    def set_switch(self, target: Any) -> None:
        super().set_switch(target)
        self.__kept.pop("target", None)
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
        "_DomainAtoms__system",
        "_DomainAtoms__kept",
        "_DomainAtoms__matrix",
        "_DomainAtoms__sub_ids",
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


class SharedForces:
    """The forces of a calculator on many processes for a computation that
    every process takes whole, as a minimization: every process has every
    atom, and computes the atoms of its brick with the ghosts within the
    reach of the calculator and the skin, found from the atoms it has; what
    a many-body potential sends to the ghosts or back goes through all the
    atoms at once. The results, of all the atoms, are the same on every
    process, so that the steps that use them are too."""

    def __init__(self, comm: Any, calculator: Calculator, skin: float):
        self.comm = comm
        self.calculator = calculator
        self.__skin = float(skin)
        reach, inner = _reach(calculator)
        self.__reach, self.__inner = reach + self.__skin, inner
        # a potential on the GPU computes the atoms and the ghosts whole there
        self.__split = (
            None if _on_gpu(calculator) else getattr(calculator, "_split", None)
        )
        self.__kept: Any = None  # of the last build, while it serves

    def __build(self, system: System) -> None:
        """This process's atoms and ghosts: the atom and the shift of each,
        local ones first, their neighbor list and species."""
        comm, box = self.comm, system.box
        h, lo = np.asarray(box.matrix, dtype=float), np.asarray(box.origin, dtype=float)
        pbc = np.asarray(box.pbc, dtype=bool)
        counts = np.array(grid(comm.Get_size(), h, pbc))
        rank = comm.Get_rank()
        at = np.array([rank // (counts[1] * counts[2]), (rank // counts[2]) % counts[1],
                       rank % counts[2]])  # fmt: skip
        inverse = np.linalg.inv(h)
        thick = 1.0 / np.linalg.norm(inverse, axis=0)
        split = counts > 1
        if (split & (thick <= self.__reach)).any():
            raise InvalidParameter(
                "the box is thinner than the reach of the ghosts along a vector it "
                "is cut along: fewer processes"
            )
        x = np.asarray(system["positions"], dtype=float)
        s = (x - lo) @ inverse
        whole = np.where(pbc, np.floor(s), 0.0)
        s -= whole
        brick = np.clip(np.floor(s * counts), 0, counts - 1).astype(np.int64)
        lower = np.where(~pbc & (at == 0), -np.inf, at / counts)
        upper = np.where(~pbc & (at == counts - 1), np.inf, (at + 1) / counts)
        mine = np.flatnonzero((brick == at).all(axis=1))
        # the atoms and their images across the periodic vectors cut into
        # bricks, within the reach of the brick
        steps = [(-1, 0, 1) if (pbc[d] and split[d]) else (0,) for d in range(3)]
        ghosts, shifts = [], []
        for a in steps[0]:
            for b in steps[1]:
                for c in steps[2]:
                    f = s + np.array([a, b, c])
                    out = np.maximum(np.maximum(lower - f, f - upper), 0.0)
                    far = (out * thick).max(axis=1)
                    near = np.flatnonzero(far <= self.__reach)
                    if (a, b, c) == (0, 0, 0):
                        near = np.setdiff1d(near, mine, assume_unique=True)
                    ghosts.append((near, far[near]))
                    shifts.append(np.repeat([[a, b, c]], len(near), axis=0))
        far = np.concatenate([g[1] for g in ghosts])
        order = np.argsort(far, kind="stable")  # the nearest first
        ids = np.concatenate([mine, np.concatenate([g[0] for g in ghosts])[order]])
        shift = np.concatenate(
            [np.zeros((len(mine), 3)), np.concatenate(shifts)[order].astype(float)]
        )
        # the places of the atoms are those they have, wrapped, and shifted
        offsets = (shift - whole[ids]) @ h
        sub_box = Box(h, origin=lo, pbc=pbc & ~split)
        positions = x[ids] + offsets
        sub = System(table=system.table[ids].with_columns(
            pl.Series("positions", positions, dtype=pl.Array(pl.Float64, 3))
        ), box=sub_box)  # fmt: skip
        calculator = self.calculator
        centers = len(mine)
        if self.__inner > 0.0:
            centers += int(
                np.searchsorted(far[order], self.__inner + self.__skin, "right")
            )
        from mdapy.calculators import MEAM, NEP, StillingerWeber, Tersoff

        symmetric = isinstance(calculator, (MEAM, NEP, StillingerWeber, Tersoff))
        full = (
            None
            if _on_gpu(calculator)
            else NeighborList.build(sub, float(calculator.cutoff) + self.__skin)
        )
        self.__kept = SimpleNamespace(
            x=x.copy(), h=h.copy(), lo=lo.copy(), pbc=pbc, ids=ids,
            mine=mine, offsets=offsets, sub=sub,
            neighbors=None if full is None else _centered(full, centers, symmetric),
            species=np.ascontiguousarray(calculator._species(sub))
            if self.__split is not None else None,
        )  # fmt: skip

    def __fresh(self, system: System) -> None:
        kept, box = self.__kept, system.box
        if kept is not None and (
            np.array_equal(kept.h, box.matrix)
            and np.array_equal(kept.lo, box.origin)
            and np.array_equal(kept.pbc, box.pbc)
        ):
            step = np.asarray(system["positions"], dtype=float) - kept.x
            moved = math.sqrt(float((step * step).sum(axis=1).max(initial=0.0)))
            if 2.0 * moved < self.__skin:
                return
        self.__build(system)

    def __gathered(self, values: np.ndarray, n: int) -> np.ndarray:
        """The values of this process's atoms as an array of all."""
        out = np.zeros((n,) + values.shape[1:], dtype=values.dtype)
        for ids, part in self.comm.allgather((self.__kept.mine, values)):
            out[ids] = part
        return out

    def compute(self, system: System, properties: tuple) -> Any:
        """What calculator.compute gives, of all the atoms, on every
        process."""
        from mdapy.calculators._base import CalculatorResult

        self.__fresh(system)
        kept, comm, n = self.__kept, self.comm, len(system)
        local = len(kept.mine)
        positions = np.ascontiguousarray(
            np.asarray(system["positions"], dtype=float)[kept.ids] + kept.offsets
        )
        if self.__split is not None:

            def forward(values: np.ndarray) -> np.ndarray:
                return self.__gathered(values, n)[kept.ids]

            def reverse(values: np.ndarray) -> np.ndarray:
                every = np.zeros((n,) + values.shape[1:])
                np.add.at(every, kept.ids, values)
                total = np.empty_like(every)
                comm.Allreduce(every, total, op=_sum_of(comm))
                return total[kept.mine]

            part = self.__split(
                (kept.species, positions, np.ascontiguousarray(kept.h)),
                tuple(properties), kept.neighbors, local, forward, reverse,
                np.ascontiguousarray(kept.ids, dtype=np.int64),
            )  # fmt: skip
            energy, forces, virial = part.energy, part.forces, part.virial
            energies, virials = part.atomic_energies, part.atomic_virials
        else:  # whole, the atoms and the ghosts
            calculator = self.calculator
            current = kept.sub.with_columns(positions=positions)
            # the list of the build with the atoms where they are now, which
            # MDAPY's potentials take with its skin; on the GPU one of its own
            neighbors = (
                None
                if kept.neighbors is None
                else kept.neighbors._reuse(current, float(calculator.cutoff))
            )
            extra = ("atomic_energies", "atomic_virials")
            wanted = tuple(properties) + tuple(p for p in extra if p not in properties)
            result = calculator.compute(current, wanted, neighbors=neighbors)
            energies = np.asarray(result.atomic_energies, dtype=float)[:local]
            virials = np.asarray(result.atomic_virials, dtype=float)[:local]
            energy, forces = float(energies.sum()), np.asarray(result.forces)[:local]
            virial = virials.sum(axis=0)
        sums = comm.allreduce(
            np.concatenate([[energy], np.ravel(np.asarray(virial, dtype=float))])
        )
        return CalculatorResult(
            energy=float(sums[0]),
            forces=self.__gathered(np.asarray(forces, dtype=float), n),
            virial=sums[1:].reshape(3, 3) if "virial" in properties else None,
            atomic_energies=self.__gathered(energies, n)
            if "atomic_energies" in properties else None,
            atomic_virials=self.__gathered(virials, n)
            if "atomic_virials" in properties else None,
            volume=system.box.volume,
        )  # fmt: skip


def _sum_of(comm: Any) -> Any:
    """The sum of mpi4py's MPI, or named for others, as Bricks takes it."""
    mpi = sys.modules.get("mpi4py.MPI")
    return mpi.SUM if mpi is not None and isinstance(comm, mpi.Comm) else "sum"
