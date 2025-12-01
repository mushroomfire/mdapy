# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _aabbtree
from mdapy.box import Box
import mdapy.tool_function as tool
import polars as pl
import numpy as np


class NearestNeighbor:
    """
    Perform a nearest-neighbor search for atoms within a periodic or non-periodic box.

    This class computes the indices and distances of the `k` nearest neighbors
    for each atom, considering periodic boundary conditions (PBC) if applicable.

    For small systems, where the number of atoms is less than `k`,
    the simulation box is automatically replicated according to its
    boundary conditions to ensure enough neighbors are available.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing atomic coordinates with columns
        ``"x"``, ``"y"``, and ``"z"``.
    box : Box
        The simulation box, defined as an instance of :class:`mdapy.box.Box`.
    k : int
        The number of nearest neighbors to search for.
        Must be less than 25.

    Attributes
    ----------
    indices_py : np.ndarray
        A 2D integer array of shape ``(N, k)``, storing the indices
        of the nearest neighbors for each atom.
    distances_py : np.ndarray
        A 2D float array of shape ``(N, k)``, storing the distances
        to the corresponding neighbors.
    _enlarge_data : pl.DataFrame, optional
        Internal replicated atomic data when periodic extension is required.
    _enlarge_box : Box, optional
        The enlarged simulation box corresponding to replicated atoms.
    """

    def __init__(self, data: pl.DataFrame, box: Box, k: int):
        self.data = data
        self.box = box
        assert k < 25, "k cannot be larger than 25."
        self.k = k

    def compute(self):
        """
        Compute the nearest neighbors for all atoms in the system.

        If the number of atoms is smaller than `k` and periodic boundaries
        are enabled, the system will be automatically replicated along the
        periodic directions to ensure sufficient neighbors are found.

        Returns
        -------
        None
            Results are stored in the following attributes:

            - ``indices_py``: nearest neighbor indices.
            - ``distances_py``: corresponding neighbor distances.
        """
        data = self.data
        box = self.box
        repeat = self._check_repeat_nearest()
        if sum(repeat) != 3:
            # Small box: replicate atoms to find enough neighbors
            self._enlarge_data, self._enlarge_box = tool.replicate(data, box, *repeat)
            box = self._enlarge_box
            data = self._enlarge_data
        N = data.shape[0]

        self.indices_py = np.zeros((N, self.k), np.int32)
        self.distances_py = np.zeros((N, self.k), np.float64)

        _aabbtree.knn(
            data["x"].to_numpy(allow_copy=False),
            data["y"].to_numpy(allow_copy=False),
            data["z"].to_numpy(allow_copy=False),
            box.box,
            box.origin,
            box.boundary,
            self.k,
            self.indices_py,
            self.distances_py,
        )

    def _check_repeat_nearest(self):
        """
        Check and determine how many box replications are needed for KNN.

        If `k` is greater than the number of atoms in the original system,
        the box will be replicated along periodic directions until the
        replicated system contains at least `k` atoms.

        Returns
        -------
        repeat : list of int
            The replication count along x, y, and z directions.
        """
        repeat = [1, 1, 1]
        N = self.data.shape[0]
        if self.k > N:
            assert sum(self.box.boundary) > 0, (
                f"Need periodic boundary if you want to query {self.k} neighbors "
                f"in {N}-atom system."
            )
            while np.prod(repeat) * N < self.k:
                for i in range(3):
                    if self.box.boundary[i] == 1:
                        repeat[i] += 3  # a safe number
        return repeat


if __name__ == "__main__":
    from ovito.io import import_file
    from ovito.data import NearestNeighborFinder
    from mdapy import System
    from time import time

    filename = "test1.xyz"
    k = 12
    atom = import_file(filename).compute()
    system = System(ovito_atom=atom)
    print("atom number: ", system.N)
    start = time()
    finder = NearestNeighborFinder(k, atom)
    ind, vec = finder.find_all()
    end = time()
    print("ovito time:", end - start)
    start = time()
    system.build_nearest_neighbor(k)
    end = time()
    print("mdapy time:", end - start)
    res = np.linalg.norm(vec, axis=-1)
    assert np.allclose(res, system.distance_list)
    import freud

    aq = freud.locality.AABBQuery.from_system(atom)
    start = time()
    query_result = aq.query(
        aq.points, dict(mode="nearest", num_neighbors=k, exclude_ii=True)
    )
    nlist = query_result.toNeighborList()
    end = time()
    print("freud time:", end - start)
