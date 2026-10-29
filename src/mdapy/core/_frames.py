# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Many systems one after another, such as the frames of a training set, as
the arrays of the compiled kernels that compute them a frame on a thread."""

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from mdapy.core.system import System
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter, MissingProperty


class Frames:
    """The atoms of systems one after another and the box of each.

    Parameters
    ----------
    systems : sequence of System
        The frames.
    """

    def __init__(self, systems: Sequence[System]):
        self.count = len(systems)
        sizes = np.fromiter((len(s) for s in systems), dtype=np.int64, count=self.count)
        #: (count + 1,) the first atom of each frame, and the number of atoms
        self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.positions = np.ascontiguousarray(
            np.concatenate(
                [s.table["positions"].to_numpy() for s in systems] + [np.zeros((0, 3))]
            ),
            dtype=np.float64,
        )
        self.matrices = np.array(
            [s.box.matrix for s in systems], dtype=np.float64
        ).reshape(-1, 3, 3)
        self.origins = np.array(
            [s.box.origin for s in systems], dtype=np.float64
        ).reshape(-1, 3)
        self.pbc = np.array([s.box.pbc for s in systems], dtype=bool).reshape(-1, 3)
        #: the atomic number of each atom, None when a frame has no elements
        self.numbers = None
        if all("element" in s.table.columns for s in systems):
            self.numbers = np.concatenate(
                [s.table["element"].to_physical().to_numpy() for s in systems]
                + [np.zeros(0, dtype=np.uint8)]
            ).astype(np.int64)

    def arrays(self) -> tuple[np.ndarray, ...]:
        """starts, positions, matrices, origins and pbc, for a kernel."""
        return self.starts, self.positions, self.matrices, self.origins, self.pbc

    def species(self, elements: Sequence[str], name: str) -> np.ndarray:
        """The index in elements of the element of each atom, as int32."""
        if self.numbers is None:
            raise MissingProperty(f"{name} needs the element column")
        lookup = np.full(len(SYMBOLS), -1, dtype=np.int32)
        for index, element in enumerate(elements):
            lookup[SYMBOLS.index(element)] = index
        indices = lookup[self.numbers]
        if (indices < 0).any():
            missing = sorted({SYMBOLS[z] for z in self.numbers[indices < 0]})
            raise InvalidParameter(
                f"the potential has no parameters for {missing}; it has {list(elements)}"
            )
        return indices


def batches(systems: Iterable[System], size: int = 10000) -> Iterator[list[System]]:
    """The systems in lists of up to size, read once, such as the frames of
    a trajectory read as they are needed."""
    batch = []
    for system in systems:
        batch.append(system)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
