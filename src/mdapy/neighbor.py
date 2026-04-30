# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _neighbor
from mdapy.box import Box
from typing import Optional
import numpy as np
import polars as pl
import mdapy.tool_function as tool


class Neighbor:
    """
    Construct the neighbor list for all atoms within a cutoff distance ``rc``.

    The :class:`Neighbor` class builds the Verlet neighbor list and corresponding
    distance list for each atom in the system.

    Parameters
    ----------
    rc : float
        Cutoff radius for neighbor searching. Must be positive.
    box : Box
        Simulation box represented by a :class:`mdapy.box.Box` instance. Supports
        both orthogonal and triclinic boxes.
    data : pl.DataFrame
        Atomic data containing at least the columns ``"x"``, ``"y"``, and ``"z"``,
        representing atomic coordinates.
    max_neigh : int, optional
        Pre-allocated per-atom slot count. The fast path (used when
        ``max_neigh`` is given) writes directly into a ``(N, max_neigh)``
        buffer and is significantly faster than the dynamic-sizer
        fallback. The C++ kernel guards each write against the slot
        bound, so an over-tight ``max_neigh`` cannot corrupt memory; if
        the true coordination exceeds ``max_neigh`` for any atom,
        :meth:`compute` raises ``ValueError`` reporting the required
        size. When omitted, the dynamic-sizer kernel runs (slower, no
        size hint required).

    Attributes
    ----------
    rc : float
        Neighbor search cutoff radius.
    box : Box
        Simulation box used for the neighbor search.
    data : pl.DataFrame
        Input atomic data.
    max_neigh : Optional[int]
        Maximum number of neighbors allowed per atom (if specified).
    N : int
        Number of atoms in the input data.
    verlet_list : np.ndarray
        Integer array of shape ``(N, max_neigh)`` or dynamically sized,
        storing the neighbor indices for each atom.
    distance_list : np.ndarray
        Float array of the same shape as ``verlet_list``,
        storing the corresponding neighbor distances.
    neighbor_number : np.ndarray
        Integer array of length ``N``, storing the number of neighbors for each atom.
    _enlarge_data : pl.DataFrame, optional
        Internal replicated atomic data when periodic extension is required.
    _enlarge_box : Box, optional
        The enlarged simulation box corresponding to replicated atoms.
    """

    def __init__(
        self, rc: float, box: Box, data: pl.DataFrame, max_neigh: Optional[int] = None
    ):
        rc = float(rc)
        assert rc > 0, f"rc must be positive, got {rc}."
        if max_neigh is not None:
            max_neigh = int(max_neigh)
            assert max_neigh > 0, f"max_neigh must be positive, got {max_neigh}."
        for col in ("x", "y", "z"):
            assert col in data.columns, f"data must contain column {col!r}."
        self.rc = rc
        self.box = box
        self.data = data
        self.max_neigh = max_neigh
        self.N = self.data.shape[0]
        assert self.N > 0, "data must contain at least one atom."

    def compute(self):
        """
        Build the neighbor list and compute interatomic distances.

        Modifies ``self`` in place; ``self.verlet_list``,
        ``self.distance_list`` and ``self.neighbor_number`` are populated.
        For systems whose box is too small for ``rc`` along periodic
        directions, ``self._enlarge_data`` and ``self._enlarge_box`` are
        also populated and the neighbor arrays index into them.
        """
        repeat = self.box.check_small_box(self.rc)

        if sum(repeat) != 3:
            self._enlarge_data, self._enlarge_box = tool.replicate(
                self.data, self.box, *repeat
            )
            data, box = self._enlarge_data, self._enlarge_box
        else:
            data, box = self.data, self.box

        x = data["x"].to_numpy(allow_copy=False)
        y = data["y"].to_numpy(allow_copy=False)
        z = data["z"].to_numpy(allow_copy=False)
        N = data.shape[0]

        if self.max_neigh is None:
            # Dynamic sizer — slower (counts first, then fills) but
            # always returns the exact size.
            self.verlet_list, self.distance_list, self.neighbor_number = (
                _neighbor.build_neighbor_without_max_neigh(
                    x, y, z, box.box, box.origin, box.boundary, self.rc,
                )
            )
            return

        # Fast path: pre-allocated buffer, single kernel pass. The C++
        # kernel guards every write against shape(1) so an over-tight
        # max_neigh cannot corrupt memory; it instead leaves
        # `neighbor_number[i]` recording the true count, which we check
        # below and surface as a clean ValueError.
        self.verlet_list = np.full((N, self.max_neigh), -1, np.int32)
        self.distance_list = np.full(
            (N, self.max_neigh), self.rc + 1.0, np.float64
        )
        self.neighbor_number = np.zeros(N, np.int32)
        _neighbor.build_neighbor(
            x, y, z, box.box, box.origin, box.boundary, self.rc,
            self.verlet_list, self.distance_list, self.neighbor_number,
        )
        real_max = int(self.neighbor_number.max(initial=0))
        if real_max > self.max_neigh:
            raise ValueError(
                f"max_neigh={self.max_neigh} is too small: at least one "
                f"atom has {real_max} neighbors within rc={self.rc}. "
                f"Re-run with max_neigh>={real_max} (or omit max_neigh "
                "to let mdapy size the buffer automatically)."
            )
