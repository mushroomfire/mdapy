# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
from mdapy.box import Box
from mdapy.knn import NearestNeighbor
from typing import Optional
import mdapy.tool_function as tool
from mdapy import _ptm


class PolyhedralTemplateMatching:
    """
    Polyhedral Template Matching (PTM) classifier for identifying local atomic structures.

    This class implements the Polyhedral Template Matching algorithm to classify the local
    structural environment of atoms in a simulation dataset. It matches the neighborhood of
    each atom against predefined polyhedral templates for common crystal structures, providing
    robustness against thermal fluctuations and elastic strains compared to methods like Common
    Neighbor Analysis (CNA).

    The supported structure types include:

    - FCC (Face-Centered Cubic)
    - HCP (Hexagonal Close-Packed)
    - BCC (Body-Centered Cubic)
    - ICO (Icosahedral)
    - SC (Simple Cubic)
    - DCUB (Cubic Diamond)
    - DHEX (Hexagonal Diamond)
    - Graphene

    The algorithm computes a Root-Mean-Square Deviation (RMSD) for the best-matching template
    and assigns a structure type if below the specified threshold. Additional outputs include
    RMSD values, scale factors, orientations (as quaternions), and neighbor indices.

    References
    ----------
    - [1] Larsen P M, Schmidt S, Schiøtz J. Robust structural identification via polyhedral template matching[J]. Modelling and Simulation in Materials Science and Engineering, 2016, 24(5): 055007.

    Parameters
    ----------
    structure : str
        String specifying the structure types to consider, separated by hyphens (e.g., "fcc-hcp-bcc").
        Supported values: "fcc", "hcp", "bcc", "ico", "sc", "dcub", "dhex", "graphene".
        Special values: "all" (all types), "default" (fcc, hcp, bcc).
    data : pl.DataFrame
        DataFrame containing atomic positions with columns 'x', 'y', 'z'. Optionally includes
        'type' or 'element' for atomic types.
    box : Box
        Simulation box object defining cell dimensions, origin, and boundary conditions.
    rmsd_threshold : float, optional
        Maximum RMSD for a valid structure match. Particles exceeding this are classified as "Other".
        Default is 0.1.
    verlet_list : np.ndarray, optional
        Precomputed Verlet neighbor list (shape (N, 18)). If None, computed internally.

    Attributes
    ----------
    output : np.ndarray
        Array of shape (N, 8) containing per-atom results:

        - Column 0: Structure type (integer, 0=Other, 1=FCC, 2=HCP, 3=BCC, 4=ICO, 5=SC, 6=DCUB, 7=DHEX, 8=Graphene)
        - Column 1: Ordering type (interger, 0=Other, 1=L10, 2=L12 (A-site), 3=L12 (B-site), 4=B2, 5=zincblende / wurtzite)
        - Column 2: RMSD value
        - Column 3: Interatomic distance
        - Columns 4-7: Orientation quaternion (x, y, z, w)

    ptm_indices : np.ndarray
        Array of shape (N, 18) containing indices of neighboring atoms used in the template matching.
    """

    def __init__(
        self,
        structure: str,
        data: pl.DataFrame,
        box: Box,
        rmsd_threshold: float = 0.1,
        verlet_list: Optional[np.ndarray] = None,
    ):
        """
        Initialize the PolyhedralTemplateMatching classifier.


        """
        self.structure = structure
        self.data = data
        self.box = box
        self.rmsd_threshold = rmsd_threshold
        self.verlet_list = verlet_list
        structure_list = [
            "fcc",
            "hcp",
            "bcc",
            "ico",
            "sc",
            "dcub",
            "dhex",
            "graphene",
            "all",
            "default",
        ]
        for i in self.structure.split("-"):
            assert i in structure_list, (
                'Structure should in ["fcc", "hcp", "bcc", "ico", "sc","dcub", "dhex", "graphene", "all", "default"].'
            )

    def compute(self) -> None:
        """
        Perform the PTM computation and store results in `self.output` and `self.ptm_indices`.

        Raises
        ------
        AssertionError
            If invalid structure types are specified.
        """
        N = self.data.shape[0]
        if sum(self.box.boundary) == 0 and N <= 18:
            self.output = np.zeros((N, 7))
            self.ptm_indices = np.zeros((N, 18), np.int32)
            return
        box = self.box
        data = self.data
        verlet_list = self.verlet_list
        rNum = 250  # safe atom number

        if self.verlet_list is None:
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
            knn = NearestNeighbor(data, box, 18)
            knn.compute()
            verlet_list = knn.indices_py
        N = data.shape[0]
        self.output = np.zeros((N, 8))
        self.ptm_indices = np.zeros((N, 18), np.int32)
        if "type" in data.columns:
            type_list = data["type"].to_numpy(allow_copy=False)
        elif "element" in data.columns:
            ele2type = {j: i + 1 for i, j in enumerate(data["element"].unique().sort())}
            type_list = data.with_columns(
                pl.col("element").replace_strict(ele2type).alias("type")
            )["type"].to_numpy(allow_copy=False)
        else:
            type_list = np.ones(data.shape[0], np.int32)
        _ptm.get_ptm(
            self.structure,
            data["x"].to_numpy(allow_copy=False),
            data["y"].to_numpy(allow_copy=False),
            data["z"].to_numpy(allow_copy=False),
            box.box,
            box.origin,
            box.boundary,
            verlet_list,
            type_list,
            self.rmsd_threshold,
            self.output,
            self.ptm_indices,
        )
