# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
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
        Cutoff radius for neighbor searching.
    box : Box
        Simulation box represented by a :class:`mdapy.box.Box` instance. Supports
        both orthogonal and triclinic boxes.
    data : pl.DataFrame
        Atomic data containing at least the columns ``"x"``, ``"y"``, and ``"z"``,
        representing atomic coordinates.
    max_neigh : int, optional
        Maximum number of neighbors allowed for each atom. If not provided, the
        neighbor list will be estimated as a largest safe value, and it will raise some performance overhead, expecially for memory.

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
        self.rc = rc
        self.box = box
        self.data = data
        self.max_neigh = max_neigh
        self.N = self.data.shape[0]

    def compute(self):
        """
        Build the neighbor list and compute interatomic distances.

        This method determines whether the box is large enough to directly perform
        neighbor searching or needs to be replicated.

        Raises
        ------
        AssertionError
            If ``max_neigh`` is provided but smaller than the actual maximum neighbor count.
        """
        repeat = self.box.check_small_box(self.rc)

        if sum(repeat) != 3:
            # Small box: replicate system
            self._enlarge_data, self._enlarge_box = tool.replicate(
                self.data, self.box, *repeat
            )
            # tool._replicate_pos(self.data, self.box, *repeat)

            N = self._enlarge_data.shape[0]
            if self.max_neigh is None:
                self.verlet_list, self.distance_list, self.neighbor_number = (
                    _neighbor.build_neighbor_without_max_neigh(
                        self._enlarge_data["x"].to_numpy(allow_copy=False),
                        self._enlarge_data["y"].to_numpy(allow_copy=False),
                        self._enlarge_data["z"].to_numpy(allow_copy=False),
                        self._enlarge_box.box,
                        self._enlarge_box.origin,
                        self._enlarge_box.boundary,
                        self.rc,
                    )
                )
            else:
                assert self.max_neigh > 0, "max_neigh should be larger than 0."
                self.verlet_list = np.full((N, self.max_neigh), -1, np.int32)
                self.distance_list = np.full(
                    (N, self.max_neigh), self.rc + 1.0, np.float64
                )
                self.neighbor_number = np.zeros(N, np.int32)
                _neighbor.build_neighbor(
                    self._enlarge_data["x"].to_numpy(allow_copy=False),
                    self._enlarge_data["y"].to_numpy(allow_copy=False),
                    self._enlarge_data["z"].to_numpy(allow_copy=False),
                    self._enlarge_box.box,
                    self._enlarge_box.origin,
                    self._enlarge_box.boundary,
                    self.rc,
                    self.verlet_list,
                    self.distance_list,
                    self.neighbor_number,
                )
        else:
            # Large box: direct computation
            if self.max_neigh is None:
                self.verlet_list, self.distance_list, self.neighbor_number = (
                    _neighbor.build_neighbor_without_max_neigh(
                        self.data["x"].to_numpy(allow_copy=False),
                        self.data["y"].to_numpy(allow_copy=False),
                        self.data["z"].to_numpy(allow_copy=False),
                        self.box.box,
                        self.box.origin,
                        self.box.boundary,
                        self.rc,
                    )
                )
            else:
                assert self.max_neigh > 0, "max_neigh should be larger than 0."
                self.verlet_list = np.full((self.N, self.max_neigh), -1, np.int32)
                self.distance_list = np.full(
                    (self.N, self.max_neigh), self.rc + 1.0, np.float64
                )
                self.neighbor_number = np.zeros(self.N, np.int32)
                _neighbor.build_neighbor(
                    self.data["x"].to_numpy(allow_copy=False),
                    self.data["y"].to_numpy(allow_copy=False),
                    self.data["z"].to_numpy(allow_copy=False),
                    self.box.box,
                    self.box.origin,
                    self.box.boundary,
                    self.rc,
                    self.verlet_list,
                    self.distance_list,
                    self.neighbor_number,
                )

        if self.max_neigh is not None:
            real_max_neigh = self.neighbor_number.max()
            assert real_max_neigh <= self.max_neigh, (
                f"Increase max_neigh to {real_max_neigh}!"
            )
