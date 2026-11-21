# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The atoms of one MPI process of MPI on its GPU, as LAMMPS's KOKKOS holds
them: those of its brick of the box and, after them, the ghosts within the
reach of the potential and the skin, all on the device, where the steps of
the integrator go through the atoms of the brick. Each step sends the
positions of the atoms other processes keep as ghosts, gathered and
scattered on the device, and sums the few numbers the thermostats and the
barostat need over the processes; between two builds of the neighbor list
nothing else leaves the device. A build, when an atom has moved half the
skin or the box has changed, brings the atoms of the brick back, moves them
to the processes of their bricks and finds the ghosts again, as
simulation/_domain.py does on the processor."""

from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.simulation._device import _NOT_FINITE, DeviceAtoms
from mdapy.simulation._domain import Bricks, _reach


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
                comm, box.matrix, box.origin, box.pbc, _reach(calculator)[0] + skin
            )
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        self._comm = comm
        x_local, v_local = self.__distribute(x, v)
        self._exchange: Any = None
        self._built = False
        self._sub_box: tuple | None = None  # (h, lo, pbc) of the atoms and the ghosts
        super().__init__(calculator, system, x_local, v_local, masses, skin, movable,
                         rotation)  # fmt: skip
        self._h = np.array(box.matrix, dtype=np.float64)
        self._exchange = self._module.Exchange()

    def __distribute(
        self, x: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """This process's atoms of arrays of all the atoms: their rows, wrapped
        positions and images, and velocities."""
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
        self._image = whole[mine].astype(np.int64)
        return x[mine] - whole[mine] @ bricks.h, np.array(v[mine], dtype=float)

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

    def __mine(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This process's atoms from the device: wrapped positions, their
        images with those of the device, velocities and forces."""
        m, atoms = self._local, self._atoms
        x = atoms.positions()[:m]
        image = self._image + atoms.images()[:m].astype(np.int64)
        return x, image, atoms.velocities()[:m], atoms.forces()[:m]

    @property
    def positions(self) -> np.ndarray:
        x, image, _, _ = self.__mine()
        return self.__gather(x + image @ self._h)

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
        x, image, _, _ = self.__mine()
        s = (x + image @ self._h - lo) @ np.linalg.inv(h)
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

    def __set_box(self, h: np.ndarray, lo: np.ndarray) -> None:
        try:
            self._bricks.set_box(h, lo)
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        self._built = False

    def __build(self) -> None:
        """The atoms moved to the processes of their bricks, the ghosts found,
        and the atoms and the ghosts given to the device, with what the
        controls of the run set on them; the neighbor list built."""
        bricks = self._bricks
        x, image, v, f = self.__mine()
        # every process raises together, or the others would wait for ever
        bad = not (np.isfinite(x).all() and np.isfinite(v).all())
        if bricks.maximum(float(bad)) > 0.0:
            raise CalculationFailed(_NOT_FINITE)
        fields = bricks.migrate(
            {"positions": x, "velocities": v, "image": image, "id": self._ids,
             "forces": f}
        )  # fmt: skip
        self._local = bricks.local
        self._ids = fields["id"]
        self._image = fields["image"]
        ghosts, ids = bricks.borders(fields["positions"], self._ids)
        self._sub_ids = np.ascontiguousarray(ids, dtype=np.int64)
        n, m = len(ghosts), self._local
        velocities = np.zeros((n, 3))
        velocities[:m] = fields["velocities"]
        forces = np.zeros((n, 3))
        forces[:m] = fields["forces"]
        self._new_atoms(ghosts, velocities)
        atoms = self._atoms
        atoms.set_owned(m)
        atoms.set_ids(self._sub_ids)
        atoms.set_forces(forces)
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
        exchange = self._exchange
        exchange.clear()
        for k, (chosen, _, _, _, start, count) in enumerate(bricks.messages):
            exchange.set(atoms, k, np.ascontiguousarray(chosen, dtype=np.int32),
                         int(start), int(count))  # fmt: skip
        self._built = True

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
        for k, (_, shift, destination, source, _, count) in enumerate(bricks.messages):
            sent = exchange.pack_positions(
                atoms, k, np.ascontiguousarray(shift @ bricks.h)
            )
            received = np.empty((count, 3))
            bricks.swap(sent, destination, received, source)
            exchange.unpack_positions(atoms, k, received)

    def _set_springs(self) -> None:
        """Springs to sites of the unwrapped positions: the device unwraps
        by its own images alone, so the sites of each atom take back the
        boxes it was moved by on the processor."""
        sites, constants = self._sites
        rows = np.array(sites[self._sub_ids], dtype=np.float64)
        rows[: self._local] -= self._image @ self._h
        self._atoms.set_springs(
            np.ascontiguousarray(rows), np.ascontiguousarray(constants[self._sub_ids])
        )

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
            reach = _reach(target, target=True)[0] + self._skin
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
        "_image",
        "_carried",
    )

    def restore(self, state: dict[str, Any]) -> None:
        """As on one process, the forces of all the atoms taken for those of
        this process's; the atoms, given again with the system, are moved to
        the device's rows at the next computation."""
        super().restore(state)
        self._built = False
