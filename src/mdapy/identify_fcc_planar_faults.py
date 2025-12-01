# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _fccpft
import numpy as np


class IdentifyFccPlanarFaults:
    """
    Identify planar faults in FCC structures including stacking faults, twin boundaries, and ESF.

    This class identifies intrinsic and extrinsic stacking faults as well as coherent twin
    boundaries in FCC crystal structures. The algorithm is based on the method described in
    the OVITO documentation [1]_, with an additional capability to identify Extrinsic
    Stacking Faults (ESF).

    Parameters
    ----------
    structure_types : np.ndarray
        Array of structure type identifiers for each atom, shape (N,).
        Typically from PTM analysis where 1=FCC, 2=HCP, etc.
    ptm_indices : np.ndarray
        PTM nearest neighbor indices for each atom, shape (N, 12).
        Contains the indices of the 12 nearest neighbors in FCC/HCP ordering.
    cal_esf : bool, default=True
        Whether to identify Extrinsic Stacking Faults (ESF) in addition to
        intrinsic stacking faults and twin boundaries.

    Attributes
    ----------
    fault_types : np.ndarray
        Array of fault type identifiers for each atom, shape (N,).
        Available after calling :meth:`compute`.

        Fault type codes:

        - 0: Non-hcp atoms (e.g. perfect fcc or disordered)
        - 1: Indeterminate hcp-like (isolated hcp-like atoms, not forming a planar defect)
        - 2: Intrinsic stacking fault (ISF, two adjacent hcp-like layers)
        - 3: Coherent twin boundary (TB, one hcp-like layer)
        - 4: Multi-layer stacking fault (three or more adjacent hcp-like layers)
        - 5: Extrinsic Stacking Fault (ESF, if cal_esf=True)

    structure_types : np.ndarray
        Input structure types array.
    ptm_indices : np.ndarray
        Input PTM neighbor indices array.
    cal_esf : bool
        Flag for ESF identification.

    References
    ----------
    .. [1] OVITO Identify FCC Planar Faults Modifier.
           https://www.ovito.org/manual/reference/pipelines/modifiers/identify_fcc_planar_faults.html
    """

    def __init__(
        self, structure_types: np.ndarray, ptm_indices: np.ndarray, cal_esf: bool = True
    ):
        self.structure_types = structure_types
        self.ptm_indices = ptm_indices
        self.cal_esf = cal_esf

    def compute(self):
        """
        Compute fault type for each atom.

        This method identifies planar faults by analyzing HCP-classified atoms
        and their neighbors. Results are stored in :attr:`fault_types`.

        """
        hcp_indices = np.where(self.structure_types == 2)[0].astype(np.int32)
        hcp_neighbors = np.zeros((hcp_indices.shape[0], 12), dtype=np.int32)
        self.fault_types = np.zeros_like(self.structure_types)
        _fccpft.identify_sftb_fcc(
            hcp_indices,
            hcp_neighbors,
            self.ptm_indices,
            self.structure_types,
            self.fault_types,
            self.cal_esf,
        )
