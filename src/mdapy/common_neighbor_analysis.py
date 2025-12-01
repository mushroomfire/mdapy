# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
from mdapy import _cna
import numpy as np
import polars as pl
from mdapy.box import Box
from mdapy.knn import NearestNeighbor
from mdapy.neighbor import Neighbor
from typing import Optional
import mdapy.tool_function as tool


class CommonNeighborAnalysis:
    """
    Perform Common Neighbor Analysis (CNA) on atomic structures.

    This class classifies each atom according to its local atomic environment
    using either an adaptive or fixed cutoff approach.

    Two modes are supported:

    1. Adaptive cutoff (default): determines neighbors based on the 14 nearest atoms.
    2. Fixed cutoff: uses a user-specified cutoff distance `rc` to define neighbors.

    After computation, each atom is assigned a coordination pattern stored in `pattern`:

    - 0 = Other (unknown coordination)
    - 1 = FCC (face-centered cubic)
    - 2 = HCP (hexagonal close-packed)
    - 3 = BCC (body-centered cubic)
    - 4 = ICO (icosahedral coordination)

    Parameters
    ----------
    data : pl.DataFrame
        Atomic coordinates stored as a Polars DataFrame with columns ['x', 'y', 'z'].
    box : Box
        Simulation box object (supports periodic boundaries).
    verlet_list : Optional[np.ndarray], default=None
        Precomputed neighbor indices (optional). If not provided, neighbors
        will be determined automatically.
    neighbor_number : Optional[np.ndarray], default=None
        Number of neighbors for each atom (required if `verlet_list` is provided).
    rc : Optional[float], default=None
        Cutoff radius for fixed-cutoff CNA. If None, adaptive cutoff is used.

    Attributes
    ----------
    pattern : np.ndarray
        Array of integer codes indicating the coordination type of each atom.

    Notes
    -----
    - Adaptive cutoff automatically finds neighbors based on the 14 nearest atoms.
    - Fixed cutoff requires the `rc` parameter and uses all neighbors within `rc`.

    References
    ----------
    1. Faken D, Jónsson H. Systematic analysis of local atomic structure combined with 3D computer graphics[J]. Computational Materials Science, 1994, 2(2): 279-286.
    2. Stukowski A. Structure identification methods for atomistic simulations of crystalline materials[J]. Modelling and Simulation in Materials Science and Engineering, 2012, 20(4): 045021.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        verlet_list: Optional[np.ndarray] = None,
        neighbor_number: Optional[np.ndarray] = None,
        rc: Optional[float] = None,
    ):
        self.data = data
        self.box = box
        self.verlet_list = verlet_list
        self.neighbor_number = neighbor_number
        if rc is not None:
            assert rc > 0
        self.rc = rc
        self.pattern = None

    def compute(self):
        """
        This method fills `self.pattern` with integer codes representing
        the local coordination environment of each atom.

        The method automatically handles small simulation boxes by replicating
        atoms as needed to ensure sufficient neighbors.
        """
        N = self.data.shape[0]
        if sum(self.box.boundary) == 0 and N <= 14:
            self.pattern = np.zeros(N, dtype=np.int32)
            return

        box = self.box
        data = self.data
        verlet_list = self.verlet_list
        neighbor_number = self.neighbor_number

        rNum = 500  # safe atom number

        if verlet_list is None:
            repeat = [1, 1, 1]
            if N < rNum:
                if sum(self.box.boundary) > 0:
                    while np.prod(repeat) * N < rNum:
                        for i in range(3):
                            if self.box.boundary[i] == 1:
                                repeat[i] += 1

            if sum(repeat) != 3:
                # Small box: replicate atoms to find enough neighbors
                data, box = tool._replicate_pos(data, box, *repeat)

            if self.rc is None:
                knn = NearestNeighbor(data, box, 14)
                knn.compute()
                verlet_list = knn.indices_py
            else:
                repeat = box.check_small_box(self.rc)
                if sum(repeat) != 3:
                    data, box = tool._replicate_pos(data, box, *repeat)
                neigh = Neighbor(self.rc, box, data)
                neigh.compute()
                verlet_list = neigh.verlet_list
                neighbor_number = neigh.neighbor_number
        else:
            assert neighbor_number is not None

        N = data.shape[0]
        self.pattern = np.zeros(N, dtype=np.int32)

        if self.rc is None:
            _cna.acna(
                data["x"].to_numpy(allow_copy=False),
                data["y"].to_numpy(allow_copy=False),
                data["z"].to_numpy(allow_copy=False),
                box.box,
                box.origin,
                box.boundary,
                verlet_list,
                self.pattern,
            )
        else:
            _cna.fcna(
                data["x"].to_numpy(allow_copy=False),
                data["y"].to_numpy(allow_copy=False),
                data["z"].to_numpy(allow_copy=False),
                box.box,
                box.origin,
                box.boundary,
                verlet_list,
                neighbor_number,
                self.pattern,
                self.rc,
            )
