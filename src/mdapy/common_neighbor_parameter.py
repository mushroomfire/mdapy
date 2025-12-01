# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _cnp
import numpy as np
import polars as pl
from mdapy.box import Box


class CommonNeighborParameter:
    """
    Calculate the Common Neighbor Parameter (CNP) for structural analysis.

    The Common Neighbor Parameter quantifies the local structural order by analyzing
    the bonding topology between neighbors. It is effective for distinguishing between
    crystalline and disordered regions.

    Parameters
    ----------
    data : pl.DataFrame
        Atomic data containing at least 'x', 'y', 'z' position columns.
    box : Box
        Simulation box object.
    rc : float
        Cutoff radius for neighbor bonds. Must be positive.
    verlet_list : np.ndarray
        Neighbor list array of shape (N, max_neigh).
    distance_list : np.ndarray
        Distance list array of shape (N, max_neigh).
    neighbor_number : np.ndarray
        Number of neighbors for each atom, shape (N,).

    Attributes
    ----------
    data : pl.DataFrame
        Input atomic data.
    box : Box
        Simulation box.
    rc : float
        Cutoff radius.
    verlet_list : np.ndarray
        Neighbor indices.
    distance_list : np.ndarray
        Neighbor distances.
    neighbor_number : np.ndarray
        Neighbor counts.
    cnp : np.ndarray
        Common Neighbor Parameters after calling compute(). Shape (N,).
        Values closer to 0 indicate more ordered/crystalline structures,
        while larger values indicate disordered regions.

    Notes
    -----
    The CNP is calculated by analyzing the connectivity graph of nearest neighbors.
    For each atom, it examines which of its neighbors are also neighbors of each other,
    forming a local bonding topology signature.

    Typical CNP interpretations:
    - Crystalline structures (FCC, BCC, HCP): Low CNP values (< 1)
    - Grain boundaries: Intermediate values (1-3)
    - Liquid/amorphous: Higher values (> 3)

    References
    ----------
    .. [1] Tsuzuki, H., Branicio, P. S., & Rino, J. P. (2007). Structural
           characterization of deformed crystals by analysis of common atomic
           neighborhood. Computer Physics Communications, 177(6), 518-523.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        rc: float,
        verlet_list: np.ndarray,
        distance_list: np.ndarray,
        neighbor_number: np.ndarray,
    ) -> None:
        self.data = data
        self.box = box
        self.rc = rc
        assert rc > 0
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number

    def compute(self) -> None:
        """
        Compute the Common Neighbor Parameter for all atoms.

        This method calculates the CNP value for each atom based on its
        neighbor topology and stores the result in the ``cnp`` attribute.

        Notes
        -----
        After calling this method, the ``cnp`` attribute will contain a float
        array with CNP values for each atom.
        """
        self.cnp = np.zeros(self.data.shape[0], float)

        _cnp.compute_cnp(
            self.data["x"].to_numpy(allow_copy=False),
            self.data["y"].to_numpy(allow_copy=False),
            self.data["z"].to_numpy(allow_copy=False),
            self.box.box,
            self.box.origin,
            self.box.boundary,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.cnp,
            self.rc,
        )
