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
carrying what the steps before brought, so that the corners come too."""

from typing import Any

import numpy as np


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
        too_thin = (self.fraction > width + 1e-12) & ((self.counts > 1) | self.pbc)
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
            if np.abs(step).max(initial=0) > 1:
                raise RuntimeError(
                    "an atom moved farther than a brick between two builds"
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
            if not self.pbc[d] and self.counts[d] == 1:
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
                self.__sends.append((chosen, shift, destination, source, len(got[0])))
                x = np.concatenate([x, got[0]])
                idx = np.concatenate([idx, got[1]])
        return x, idx

    def follow(self, positions: np.ndarray) -> np.ndarray:
        """The positions of the ghosts again, from those of the local atoms,
        along the lists of the last ``borders``."""
        x = np.array(positions[: self.local])
        for chosen, shift, destination, source, count in self.__sends:
            got = _exchange(
                self.comm, (x[chosen] + shift @ self.h,), destination, source
            )[0]
            if len(got) != count:
                raise RuntimeError("the ghosts changed between two builds")
            x = np.concatenate([x, got])
        return x

    def total(self, value: Any) -> Any:
        """The sum over the processes."""
        from mpi4py import MPI

        return self.comm.allreduce(value, op=MPI.SUM)


def _exchange(
    comm: Any, sent: tuple, destination: int | None, source: int | None
) -> tuple:
    """What a process sends one way and receives from the other; empty
    arrays where there is no process."""
    if destination is None and source is None:
        return tuple(a[:0] for a in sent)
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
    other."""
    if destination is None and source is None:
        return {k: v[:0] for k, v in sent.items()}
    if destination is None:
        return comm.recv(source=source)
    if source is None:
        comm.send(sent, dest=destination)
        return {k: v[:0] for k, v in sent.items()}
    return comm.sendrecv(sent, dest=destination, source=source)
