# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _aja
import numpy as np
import polars as pl
from mdapy.box import Box


class AcklandJonesAnalysis:
    """
    Perform Ackland-Jones Analysis (AJA) to identify crystal structures.

    The Ackland-Jones Analysis is a structure identification method that classifies
    local atomic environments based on coordination patterns and bond angles. It can
    distinguish between FCC, BCC, HCP, and other common crystal structures.

    Parameters
    ----------
    data : pl.DataFrame
        Atomic data containing at least the columns 'x', 'y', 'z' for positions.
    box : Box
        Simulation box object.
    verlet_list : np.ndarray
        Neighbor list array of shape (N, max_neigh) containing neighbor indices.
    distance_list : np.ndarray
        Distance list array of shape (N, max_neigh) containing neighbor distances.

    Attributes
    ----------
    data : pl.DataFrame
        Input atomic data.
    box : Box
        Simulation box.
    verlet_list : np.ndarray
        Neighbor indices.
    distance_list : np.ndarray
        Neighbor distances.
    aja : np.ndarray
        Structure type array of shape (N,) assigned after calling compute().
        Possible values:

        - 0: Unknown/other structure
        - 1: FCC
        - 2: HCP
        - 3: BCC
        - 4: ICO (icosahedral)

    Notes
    -----
    This analysis requires a pre-computed neighbor list. Typically, at least
    14 nearest neighbors are needed for accurate classification.

    References
    ----------
    .. [1] Ackland, G. J., & Jones, A. P. (2006). Applications of local crystal
           structure measures in experiment and simulation. Physical Review B,
           73(5), 054104.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        verlet_list: np.ndarray,
        distance_list: np.ndarray,
    ) -> None:
        self.data = data
        self.box = box
        self.verlet_list = verlet_list
        self.distance_list = distance_list

    def compute(self) -> None:
        """
        Perform the Ackland-Jones structure analysis.

        This method computes the structure type for each atom based on the neighbor
        list and assigns the result to the ``aja`` attribute.

        Notes
        -----
        After calling this method, the ``aja`` attribute will contain integer
        structure identifiers for each atom.
        """
        self.aja = np.zeros(self.data.shape[0], dtype=np.int32)
        _aja.compute_aja(
            self.data["x"].to_numpy(allow_copy=False),
            self.data["y"].to_numpy(allow_copy=False),
            self.data["z"].to_numpy(allow_copy=False),
            self.box.box,
            self.box.origin,
            self.box.boundary,
            self.verlet_list,
            self.distance_list,
            self.aja,
        )


if __name__ == "__main__":
    import mdapy as mp

    system = mp.System("tests/input_files/rec_box_small.xyz")
    aja = AcklandJonesAnalysis(system.data, system.box)
    aja.compute()
    print(aja.aja)
