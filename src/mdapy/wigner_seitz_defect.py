# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
from typing import Union, Dict
import numpy as np
import polars as pl
from mdapy import _aabbtree
from mdapy import System


class WignerSeitzAnalysis:
    """Wigner-Seitz defect analysis.

    This class performs point defect analysis using the Wigner-Seitz cell method,
    similar to the implementation in OVITO. It identifies vacancies and interstitials
    by comparing a reference configuration (ideal lattice sites) with a current
    configuration (displaced atoms). Each atom in the current configuration is
    assigned to the nearest reference site. Site occupancies are then computed to
    detect defects: sites with occupancy 0 are vacancies, and sites with occupancy
    >1 indicate interstitials.

    Parameters
    ----------
    ref : System
        Reference system defining the ideal lattice sites.
    affine : bool, optional
        If True, applies an affine transformation to map current positions to the
        reference cell before assignment (handles cell deformation). Default is False.

    """

    def __init__(self, ref: System, affine: bool = False):
        self.ref = ref
        if "element" in self.ref.data.columns:
            type_list = self.ref.data["element"]
        elif "type" in self.ref.data.columns:
            type_list = self.ref.data["type"]
        else:
            type_list = pl.Series("type", [1] * self.ref.N)
        self.type_list = type_list
        self.affine = affine
        self._build_tree_from_ref()

    def _build_tree_from_ref(self):
        """Build AABB tree from reference system positions.

        This internal method constructs an AABB tree for fast nearest-neighbor
        queries on the reference positions.

        Notes
        -----
        The tree is built using the reference positions, box, origin, and boundary
        conditions for handling periodic systems if applicable.
        """
        self._tree = _aabbtree.AABBTree()
        self._tree.build_with_coords(
            self.ref.data["x"].to_numpy(allow_copy=False),
            self.ref.data["y"].to_numpy(allow_copy=False),
            self.ref.data["z"].to_numpy(allow_copy=False),
            self.ref.box.box,
            self.ref.box.origin,
            self.ref.box.boundary,
        )

    def compute(self, current: System) -> Dict[str, Union[int, np.ndarray]]:
        """Perform Wigner-Seitz analysis.

        This method assigns each atom in the current configuration to the nearest
        reference site and computes site occupancies to identify defects.

        Parameters
        ----------
        current : System
            The system to analyze (the "current" or "displaced" configuration).

        Returns
        -------
        dict
            Dictionary containing:
                'site_occupancy' : np.ndarray (ref.N,)
                    Occupancy counts per reference site.
                'atom_site_index' : np.ndarray (current.N,)
                    For each atom, the index of the assigned reference site.
                'atom_site_type' : np.ndarray (current.N,)
                    For each atom, the type of the assigned reference site.
                'atom_occupancy' : np.ndarray (current.N,)
                    For each atom, the occupancy of its assigned site.
                'vacancy_count' : int
                    Number of vacant sites (occupancy == 0).
                'interstitial_count' : int
                    Number of sites with interstitials (occupancy > 1).
        """
        data = current.data
        if self.affine:
            map_matrix = np.linalg.solve(current.box.box, self.ref.box.box)
            # pos @ map_matrix
            data = data.select(
                x=pl.col("x") * map_matrix[0, 0]
                + pl.col("y") * map_matrix[1, 0]
                + pl.col("z") * map_matrix[2, 0],
                y=pl.col("x") * map_matrix[0, 1]
                + pl.col("y") * map_matrix[1, 1]
                + pl.col("z") * map_matrix[2, 1],
                z=pl.col("x") * map_matrix[0, 2]
                + pl.col("y") * map_matrix[1, 2]
                + pl.col("z") * map_matrix[2, 2],
            )
        indice = np.zeros(current.N, np.int32)
        self._tree.query_nearest_batch(
            data["x"].to_numpy(allow_copy=False),
            data["y"].to_numpy(allow_copy=False),
            data["z"].to_numpy(allow_copy=False),
            indice,
        )
        site_occ = np.zeros(self.ref.N, np.int32)
        np.add.at(site_occ, indice, 1)
        # vacancy and interstitial counts
        vacancy_count = int(np.sum(site_occ == 0))
        interstitial_count = int((site_occ[site_occ > 1] - 1).sum())

        result: Dict[str, Union[int, np.ndarray]] = {
            "site_occupancy": site_occ,
            "atom_site_index": indice,
            "atom_site_type": self.type_list[indice],
            "atom_occupancy": site_occ[indice],
            "vacancy_count": vacancy_count,
            "interstitial_count": interstitial_count,
        }
        return result


if __name__ == "__main__":
    pass
