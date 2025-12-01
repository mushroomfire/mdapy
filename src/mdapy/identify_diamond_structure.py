# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _cna
import numpy as np
from numpy.typing import NDArray
import polars as pl
from mdapy.box import Box
from mdapy.knn import NearestNeighbor
from typing import Optional
import mdapy.tool_function as tool


class IdentifyDiamondStructure:
    """

    This modifier analyzes the coordination structure of particles to identify diamond
    lattice structures, including both cubic and hexagonal diamond polymorphs. The
    identification is performed using an extended Common Neighbor Analysis method that
    considers both first and second nearest neighbor shells.

    The `pattern` array assigns one of the following structure type identifiers to each particle:

    - 0 Other
    - 1 Cubic diamond
    - 2 Cubic diamond (1st neighbor)
    - 3 Cubic diamond (2nd neighbor)
    - 4 Hexagonal diamond
    - 5 Hexagonal diamond (1st neighbor)
    - 6 Hexagonal diamond (2nd neighbor)

    Parameters
    ----------
    data : pl.DataFrame
        Particle properties data frame containing columns: 'x', 'y', 'z' for positions.
    box : Box
        Simulation cell object containing boundary conditions and dimensions.
    verlet_list : NDArray[np.int32], optional
        Pre-computed neighbor list for each particle with at least 4 neighbors.
        If None, k-nearest neighbor search (k=4) is performed automatically.
        Shape: (N, k) where N is the number of particles and k ≥ 4.

    Attributes
    ----------
    data : pl.DataFrame
        Input particle configuration data.
    box : Box
        Simulation cell information.
    verlet_list : NDArray[np.int32] or None
        Neighbor list used for structure identification.
    pattern : NDArray[np.int32]
        Structure type classification for each particle. Available after calling `compute()`.
        Shape: (N,) where N is the number of particles.

    References
    ----------
    [1] Maras E, Trushin O, Stukowski A, et al. Global transition path search for dislocation formation in Ge on Si (001)[J]. Computer Physics Communications, 2016, 205: 13-21.

    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        verlet_list: Optional[NDArray[np.int32]] = None,
    ) -> None:
        self.data: pl.DataFrame = data
        self.box: Box = box
        self.verlet_list: Optional[NDArray[np.int32]] = verlet_list
        self.pattern: NDArray[np.int32] = np.array([], dtype=np.int32)

    def compute(self) -> None:
        """
        Perform diamond structure identification on all particles.
        The method automatically handles different system configurations:

        - For systems with ≤4 particles and no periodic boundaries, all particles
          are classified as type 0 (Other)
        - For small systems (N < 500) with periodic boundaries, the simulation cell
          is automatically replicated to ensure accurate neighbor identification
        - If no neighbor list is provided, a k-nearest neighbor search with k=4
          is performed automatically
        """
        N = self.data.shape[0]
        if sum(self.box.boundary) == 0 and N <= 4:
            self.pattern = np.zeros(N, dtype=np.int32)
            return

        box = self.box
        data = self.data
        verlet_list = self.verlet_list
        rNum = 500  # Safe atom number threshold for replication

        if self.verlet_list is None:
            repeat = [1, 1, 1]
            if N < rNum:
                if sum(self.box.boundary) > 0:
                    # Replicate box until we have enough atoms
                    while np.prod(repeat) * N < rNum:
                        for i in range(3):
                            if self.box.boundary[i] == 1:
                                repeat[i] += 1

            if sum(repeat) != 3:
                # Small box: replicate atoms to find enough neighbors
                data, box = tool._replicate_pos(data, box, *repeat)

            # Perform k-nearest neighbor search with k=4 for diamond structures
            knn = NearestNeighbor(data, box, 4)
            knn.compute()
            verlet_list = knn.indices_py

        N = data.shape[0]
        self.pattern = np.zeros(N, dtype=np.int32)
        new_verlet_list = np.zeros((N, 12), dtype=np.int32)

        _cna.ids(
            data["x"].to_numpy(allow_copy=False),
            data["y"].to_numpy(allow_copy=False),
            data["z"].to_numpy(allow_copy=False),
            box.box,
            box.origin,
            box.boundary,
            verlet_list,
            new_verlet_list,
            self.pattern,
        )


if __name__ == "__main__":
    pass
