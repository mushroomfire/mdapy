# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _chill_plus
import numpy as np
from numpy.typing import NDArray
import polars as pl
from mdapy.box import Box
from mdapy.neighbor import Neighbor
from mdapy.parallel import get_num_threads
from typing import Optional


class ChillPlus:
    """
    CHILL+ structure identification for water phases.

    Classifies each particle (oxygen / coarse-grained water bead) as one of
    six phases by examining the bond-orientation correlation
    :math:`c_{ij}` between neighbouring molecules within ``cutoff``:

    * 0 -- Other
    * 1 -- Hexagonal ice
    * 2 -- Cubic ice
    * 3 -- Interfacial ice
    * 4 -- Gas hydrate
    * 5 -- Interfacial gas hydrate

    The cutoff defines the O-O distance below which two molecules are
    considered bonded (default 3.5 Å). The algorithm operates only on the
    particles supplied -- pass oxygen / water-bead positions, not hydrogens.

    Parameters
    ----------
    data : pl.DataFrame
        Particle data with at least 'x', 'y', 'z' columns.
    box : Box
        Simulation box.
    cutoff : float, optional
        O-O cutoff radius in Å. Default 3.5.
    verlet_list, distance_list, neighbor_number : np.ndarray, optional
        Pre-computed cutoff neighbour list. When omitted, an internal
        :class:`~mdapy.neighbor.Neighbor` build is performed at ``cutoff``.

    Attributes
    ----------
    pattern : np.ndarray of int32, shape (N,)
        Structure code per particle (filled by :meth:`compute`).

    References
    ----------
    Nguyen, A. H. & Molinero, V. *Identification of Clathrate Hydrates,
    Hexagonal Ice, Cubic Ice, and Liquid Water in Simulations: the
    CHILL+ Algorithm.* J. Phys. Chem. B 119 (2015) 9369-9376.
    DOI: 10.1021/jp510289t
    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        cutoff: float = 3.5,
        verlet_list: Optional[NDArray[np.int32]] = None,
        distance_list: Optional[NDArray[np.float64]] = None,
        neighbor_number: Optional[NDArray[np.int32]] = None,
    ) -> None:
        self.data = data
        self.box = box
        self.cutoff = float(cutoff)
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        self.pattern: NDArray[np.int32] = np.array([], dtype=np.int32)

    def compute(self) -> None:
        """Run the CHILL+ classification, populating ``self.pattern``."""
        if (
            self.verlet_list is None
            or self.distance_list is None
            or self.neighbor_number is None
        ):
            neigh = Neighbor(self.cutoff, self.box, self.data)
            neigh.compute()
            self.verlet_list = neigh.verlet_list
            self.distance_list = neigh.distance_list
            self.neighbor_number = neigh.neighbor_number

        N = self.data.shape[0]
        self.pattern = np.zeros(N, dtype=np.int32)
        _chill_plus.compute_chill_plus(
            self.data["x"].to_numpy(allow_copy=False),
            self.data["y"].to_numpy(allow_copy=False),
            self.data["z"].to_numpy(allow_copy=False),
            self.box.box,
            self.box.origin,
            self.box.boundary,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.cutoff,
            self.pattern,
            get_num_threads(),
        )


if __name__ == "__main__":
    pass
