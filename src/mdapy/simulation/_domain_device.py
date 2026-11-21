# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The atoms of one process of MPI on its GPU, as LAMMPS's KOKKOS holds
them: those of its brick of the box and, after them, the ghosts within the
reach of the potential and the skin, all on the device, where the steps of
the integrator go through the atoms of the brick. Each step sends the
positions of the atoms other processes keep as ghosts, gathered and
scattered on the device, and sums the few numbers the thermostats and the
barostat need over the processes; between two builds of the neighbor list
nothing else leaves the device. A build, when an atom has moved half the
skin or the box has changed, is done on the device too, as the bricks of
simulation/_domain.py do it on the processor: the ghosts dropped, the atoms
wrapped into the box, those that left the brick sent with all they carry to
the process of the next brick, then the ghosts found again, each set of
atoms chosen on the device. What goes from process to process is handed to
MPI in the memory of the GPU when MPI moves that memory itself, as KOKKOS
does, else through pinned memory of the host."""

import ctypes
import os
from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.simulation._device import _NOT_FINITE, DeviceAtoms
from mdapy.simulation._domain import Bricks, _reach


def _gpu_aware(comm: Any) -> bool:
    """Whether the MPI of comm moves the memory of a GPU itself, the same on
    every process: as MDAPY_GPU_AWARE_MPI says, 1 or 0, when it is set; else
    as Open MPI says (MPIX_Query_cuda_support); else not."""
    value = os.environ.get("MDAPY_GPU_AWARE_MPI")
    if value is not None and value.strip() not in ("0", "1"):
        raise InvalidParameter(f"MDAPY_GPU_AWARE_MPI must be 1 or 0, not {value!r}")
    mine = value.strip() == "1" if value is not None else _open_mpi_aware()
    return all(comm.allgather(bool(mine)))


def _open_mpi_aware(maps: str = "/proc/self/maps") -> bool:
    """Whether the Open MPI loaded in this process moves the memory of the
    GPU, as its MPIX_Query_cuda_support tells; False for another MPI."""
    try:
        with open(maps, encoding="utf-8") as lines:
            paths = sorted({
                line.split()[-1] for line in lines
                if os.path.basename(line.split()[-1]).startswith("libmpi.so")
            })  # fmt: skip
    except OSError:
        return False
    for path in paths:
        try:
            query = getattr(ctypes.CDLL(path), "MPIX_Query_cuda_support", None)
        except OSError:
            continue
        if query is not None:
            return bool(query())
    return False


class DomainDeviceAtoms(DeviceAtoms):
    """The atoms of one process of MPI on its GPU, with the interface of
    DeviceAtoms: the arrays of all the atoms that the dynamics reads,
    positions, velocities and forces, gathered on every process in the
    order of the atoms."""

    def __init__(
        self,
        comm: Any,
        calculator: Calculator,
        system: System,
        x: np.ndarray,
        v: np.ndarray,
        masses: np.ndarray,
        skin: float,
        movable: np.ndarray | None,
        rotation: np.ndarray | None,
        box: Box,
    ):
        try:
            self._bricks = Bricks(
                comm,
                box.matrix,
                box.origin,
                box.pbc,
                _reach(calculator, staged=True)[0] + skin,
            )
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        self._comm = comm
        x_local, v_local = self.__distribute(x, v)
        self._exchange: Any = None
        self._built = False
        self._sub_box: tuple | None = None  # (h, lo, pbc) of the atoms and the ghosts
        # the messages of the last build: (box vectors added to what each
        # sends, destination, source)
        self._messages: list = []
        super().__init__(calculator, system, x_local, v_local, masses, skin, movable,
                         rotation)  # fmt: skip
        self._atoms.set_ids(self._sub_ids)
        self._h = np.array(box.matrix, dtype=np.float64)
        self._exchange = self._module.Exchange()
        self._exchange.set_device(_gpu_aware(comm))

    def __distribute(
        self, x: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """This process's atoms of arrays of all the atoms, their positions as
        given: the device wraps them at the first build and counts the boxes
        it moves them by."""
        bricks = self._bricks
        s = bricks.fractional(x)
        whole = np.where(bricks.pbc, np.floor(s), 0.0)
        brick = np.clip(
            np.floor((s - whole) * bricks.counts), 0, bricks.counts - 1
        ).astype(np.int64)
        mine = np.flatnonzero((brick == bricks.at).all(axis=1))
        bricks.local = len(mine)
        self._local = len(mine)
        self._ids = mine.astype(np.int64)  # this process's atoms among all
        self._sub_ids = self._ids  # those of the rows on the device
        return np.array(x[mine], dtype=float), np.array(v[mine], dtype=float)

    # -- the rows on the device, and the sums over the processes ------------

    def _rows_of(self, values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(values)[self._sub_ids])

    def _total(self, value: Any) -> Any:
        array = np.asarray(value, dtype=np.float64)
        out = self._bricks.total(np.ascontiguousarray(np.atleast_1d(array)))
        return float(out[0]) if array.ndim == 0 else out.reshape(array.shape)

    def _agreed(self, value: Any) -> Any:
        return self._comm.bcast(value, root=0)

    def __gather(self, values: np.ndarray) -> np.ndarray:
        """Values of this process's atoms as an array of all, on every
        process."""
        parts = self._comm.allgather((self._ids, values))
        out = np.empty((self._count,) + values.shape[1:], dtype=values.dtype)
        for ids, part in parts:
            out[ids] = part
        return out

    def __unwrapped(self) -> np.ndarray:
        """This process's atoms from the device, unwrapped by the boxes the
        device moved them by."""
        m, atoms = self._local, self._atoms
        return atoms.positions()[:m] + atoms.images()[:m] @ self._h

    @property
    def positions(self) -> np.ndarray:
        return self.__gather(self.__unwrapped())

    @property
    def velocities(self) -> np.ndarray:
        return self.__gather(self._atoms.velocities()[: self._local])

    @velocities.setter
    def velocities(self, v: np.ndarray) -> None:
        rows = np.zeros((len(self._sub_ids), 3))
        rows[: self._local] = np.asarray(v, dtype=float)[self._ids]
        self._atoms.set_velocities(rows)

    @property
    def forces(self) -> np.ndarray:
        return self.__gather(self._atoms.forces()[: self._local])

    def step_limit(self, distance: float) -> float:
        from mdapy.simulation._integrator import _step_limit

        m = self._local
        mine = _step_limit(self._atoms.velocities()[:m], self._atoms.forces()[:m],
                           self._masses[self._ids], distance)  # fmt: skip
        return float(np.min(self._comm.allgather(float(mine))))

    def extent(self, lo: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = (self.__unwrapped() - lo) @ np.linalg.inv(h)
        low = s.min(axis=0) if len(s) else np.full(3, np.inf)
        high = s.max(axis=0) if len(s) else np.full(3, -np.inf)
        parts = self._comm.allgather((low, high))
        return (
            np.min([p[0] for p in parts], axis=0),
            np.max([p[1] for p in parts], axis=0),
        )

    # -- the builds and the ghosts -----------------------------------------

    def _update(self, box: Box, h: np.ndarray, lo: np.ndarray, pbc: Any) -> tuple:
        bricks = self._bricks
        if not (
            np.array_equal(box.matrix, bricks.h)
            and np.array_equal(box.origin, bricks.lo)
        ):
            self.__set_box(box.matrix, box.origin)
        rebuild = not self._built
        if not rebuild:
            try:
                stale = self._neighbors.stale(self._atoms, *self._sub_box)
            except self._module.NotFinite:
                stale = True  # found again, and raised, at the build
            rebuild = bricks.maximum(float(stale)) > 0.0
        if rebuild:
            self.__build()
        else:
            self.__forward()
        return self._sub_box

    def _evaluate(
        self, potential: Any, sub: tuple, energy: bool, virial: bool, heat: bool
    ) -> None:
        """A potential that computes the atoms of the process alone, with
        ghosts within its cutoff: in stages, the values of each atom of one,
        such as the slopes of the embedding functions of EAM, sent to the
        ghosts, message by message, before the next; for the three-body
        potentials, whose terms are those of the atoms of the process, the
        forces on the ghosts sent back after, with their virials for the heat
        flux."""
        if potential.stages == 1 and not potential.reverse:
            super()._evaluate(potential, sub, energy, virial, heat)
            return
        atoms, neighbors, exchange, bricks = (
            self._atoms,
            self._neighbors,
            self._exchange,
            self._bricks,
        )
        for stage in range(potential.stages):
            if stage > 0:
                for k, (_, destination, source) in enumerate(self._messages):
                    bricks.swap(exchange.pack_forwarded(potential, k), destination,
                                exchange.receive_forwarded(potential, k), source)  # fmt: skip
                    exchange.unpack_forwarded(potential, k)
            potential.compute_stage(stage, atoms, neighbors, *sub, energy, virial, heat)
        if potential.reverse:
            # the last message first: the ghosts it sent may have come in one before
            for k in reversed(range(len(self._messages))):
                _, destination, source = self._messages[k]
                bricks.swap(exchange.pack_forces(atoms, k), source,
                            exchange.receive_forces(k), destination)  # fmt: skip
                exchange.add_forces(atoms, k)
                if heat:
                    bricks.swap(exchange.pack_virials(atoms, k), source,
                                exchange.receive_virials(k), destination)  # fmt: skip
                    exchange.add_virials(atoms, k)

    def __set_box(self, h: np.ndarray, lo: np.ndarray) -> None:
        try:
            self._bricks.set_box(h, lo)
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        self._built = False

    def __build(self) -> None:
        """On the device: the ghosts dropped, the atoms wrapped into the box
        along its periodic directions, moved brick to brick along x, y and z
        with all they carry, then the ghosts of the bricks around found and
        added; what the controls of the run set given again, and the neighbor
        list built."""
        bricks, atoms, exchange = self._bricks, self._atoms, self._exchange
        h, lo = np.ascontiguousarray(bricks.h), np.ascontiguousarray(bricks.lo)
        atoms.settle()
        atoms.truncate(self._local)  # the atoms of the process come first
        atoms.wrap(h, lo, tuple(bool(p) for p in bricks.pbc))
        exchange.clear()
        for d in range(3):
            if bricks.counts[d] == 1:
                continue
            down, up, step = exchange.leaving(atoms, d, h, lo, int(bricks.counts[d]),
                                              int(bricks.at[d]), bool(bricks.pbc[d]))  # fmt: skip
            # every process raises together, or the others would wait for ever:
            # 3 for a position gone wrong, else how many bricks an atom moved
            worst = bricks.maximum(3.0 if step == 127 else float(step))
            if worst >= 3.0:
                raise CalculationFailed(_NOT_FINITE)
            if worst > 1.0:
                raise InvalidParameter(
                    "an atom moved farther than a brick between two builds of the "
                    "neighbor lists: the time step may be too long"
                )
            for side, count in ((0, down), (1, up)):
                destination, source = bricks.neighbors_along(d, 2 * side - 1)
                came = self.__count(count, destination, source)
                bricks.swap(exchange.pack_leaving(atoms, side), destination,
                            exchange.receive_arriving(side, came), source)  # fmt: skip
            exchange.migrate(atoms)
        owned = atoms.size
        self.__borders(h, lo)
        self._sub_ids = np.asarray(atoms.ids(), dtype=np.int64)
        self._local = owned
        self._ids = self._sub_ids[:owned]
        atoms.set_owned(owned)
        self._set_free(self._free)
        self._set_run_arrays()
        if self._sites is not None:
            self._set_springs()
        self._sub_box = self.__sub_box()
        try:
            self._neighbors.build(atoms, *self._sub_box)
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        except self._module.NotFinite:
            raise CalculationFailed(_NOT_FINITE) from None
        exchange.place(atoms)
        self._built = True

    def __count(self, sent: int, destination: Any, source: Any) -> int:
        """How many rows come from source while sent go to destination."""
        received = np.zeros(1, dtype=np.int64)
        self._bricks.swap(
            np.array([sent], dtype=np.int64), destination, received, source
        )
        return int(received[0])

    def __borders(self, h: np.ndarray, lo: np.ndarray) -> None:
        """The ghosts of the bricks around added after the atoms, along x, y
        and z in turn, as Bricks.borders finds them: first the atoms and the
        ghosts of the directions before, which carry the corners, then, when
        the bricks are thinner than the reach, those that came the hop before
        from the other side, passed on."""
        bricks, atoms, exchange = self._bricks, self._atoms, self._exchange
        lower = bricks.at / bricks.counts
        upper = (bricks.at + 1) / bricks.counts
        self._messages = []
        for d in range(3):
            if bricks.counts[d] == 1:
                continue
            before = atoms.size
            came = {-1: (0, before), 1: (0, before)}
            for _ in range(int(bricks.hops[d])):
                for side in (-1, 1):
                    begin, end = came[side]
                    if side < 0:
                        low, high = lower[d], lower[d] + bricks.fraction[d]
                        edge = bricks.at[d] == 0
                    else:
                        low, high = upper[d] - bricks.fraction[d], upper[d]
                        edge = bricks.at[d] == bricks.counts[d] - 1
                    shift = np.zeros(3)
                    if edge:
                        shift[d] = -side  # in box vectors, added to what is sent
                    k = exchange.messages
                    count = exchange.border(atoms, d, h, lo, float(low), float(high), begin,
                                            end, bool(edge and not bricks.pbc[d]),
                                            np.ascontiguousarray(shift @ h))  # fmt: skip
                    destination, source = bricks.neighbors_along(d, side)
                    got = self.__count(count, destination, source)
                    bricks.swap(exchange.border_out(k), destination,
                                exchange.border_in(k, got), source)  # fmt: skip
                    start = atoms.size
                    exchange.append_border(atoms, k)
                    self._messages.append((shift, destination, source))
                    came[side] = (start, start + got)

    def __sub_box(self) -> tuple:
        """The box of the atoms and the ghosts: along a vector cut into
        bricks, the brick and the reach of its ghosts on each side, not
        periodic; along the others the box, periodic where it is."""
        bricks = self._bricks
        h, lo = np.array(bricks.h), np.array(bricks.lo)
        cut = bricks.counts > 1
        low = np.where(cut, bricks.at / bricks.counts - bricks.fraction - 1e-9, 0.0)
        high = np.where(
            cut, (bricks.at + 1) / bricks.counts + bricks.fraction + 1e-9, 1.0
        )
        matrix = np.ascontiguousarray(h * (high - low)[:, None])
        origin = np.ascontiguousarray(lo + low @ h)
        return matrix, origin, tuple(bool(p) for p in bricks.pbc & ~cut)

    def __forward(self) -> None:
        """The positions of the ghosts from those of their atoms, message by
        message, as the borders found them."""
        atoms, exchange, bricks = self._atoms, self._exchange, self._bricks
        for k, (shift, destination, source) in enumerate(self._messages):
            sent = exchange.pack_positions(
                atoms, k, np.ascontiguousarray(shift @ bricks.h)
            )
            bricks.swap(sent, destination, exchange.receive_positions(k), source)
            exchange.unpack_positions(atoms, k)

    def remap(
        self, old_lo: np.ndarray, old: np.ndarray, lo: np.ndarray, new: np.ndarray
    ) -> None:
        """The atoms carried with the box, and the bricks with it: the
        ghosts are found again at the next computation."""
        super().remap(old_lo, old, lo, new)
        self.__set_box(new, lo)

    def _set_neighbors(self, cutoff: float) -> None:
        super()._set_neighbors(cutoff)
        self._built = False

    def set_switch(self, target: Any) -> None:
        """As on one process; a target of a longer reach brings ghosts from
        farther."""
        super().set_switch(target)
        if target:
            reach = _reach(target, target=True, staged=True)[0] + self._skin
            if reach > self._bricks.reach:
                try:
                    self._bricks.set_reach(reach)
                except ValueError as error:
                    raise InvalidParameter(str(error)) from None
                self._built = False

    # -- the means over all the atoms --------------------------------------

    def _drive(self, field: np.ndarray) -> None:
        total = self._total(np.asarray(self._atoms.push(field), dtype=float))
        self._atoms.pushed(np.ascontiguousarray(total * (1.0 / self._count)))

    def _convect(self, field: np.ndarray, kick: float) -> None:
        from mdapy.simulation._device import _MVV2E

        total = self._total(self._atoms.carried(kick, _MVV2E))
        self._atoms.convected(field, total / self._count)

    def langevin(self, c1: float, c2: float, kt: float, keep: bool = False) -> None:
        if not keep:
            super().langevin(c1, c2, kt, False)
            return
        noise = self._atoms.draw_noise(c2, kt, self._seed, self._counter)
        total = self._total(np.asarray(noise, dtype=float))
        self._atoms.keep_noise(c1, np.ascontiguousarray(total * (1.0 / self._count)))
        self._counter += 1
        if self._free is not None:
            self._atoms.restrict_velocities()

    # -- a restart, whatever the number of processes -------------------------

    _TRANSIENT = DeviceAtoms._TRANSIENT + (
        "_bricks",
        "_comm",
        "_exchange",
        "_built",
        "_sub_box",
        "_local",
        "_ids",
        "_sub_ids",
        "_messages",
    )

    def restore(self, state: dict[str, Any]) -> None:
        """As on one process, the unwrapped positions and the forces of all
        the atoms taken for those of this process's; the device wraps them
        at the next build, which counts the boxes they crossed."""
        super().restore(state)
        self._built = False
