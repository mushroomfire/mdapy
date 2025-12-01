# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _csp
import polars as pl
from mdapy.box import Box
import numpy as np


class CentroSymmetryParameter:
    """
    Calculate the Centro-Symmetry Parameter (CSP) for detecting crystal defects.

    The Centro-Symmetry Parameter is a structural metric that measures the local
    symmetry of an atom's environment. It is particularly useful for identifying
    defects in centro-symmetric structures like FCC and BCC crystals.

    Parameters
    ----------
    data : pl.DataFrame
        Atomic data containing at least 'x', 'y', 'z' position columns.
    box : Box
        Simulation box object.
    N : int
        Number of nearest neighbors to consider. Must be a positive even number.
        Typical values are 12 for FCC and 8 for BCC structures.
    verlet_list : np.ndarray
        Neighbor list array of shape (n_atoms, max_neigh) containing neighbor indices.

    Attributes
    ----------
    data : pl.DataFrame
        Input atomic data.
    box : Box
        Simulation box.
    N : int
        Number of neighbors used in calculation.
    verlet_list : np.ndarray
        Neighbor indices.
    csp : np.ndarray
        Centro-symmetry parameters after calling compute(). Shape (n_atoms,).
        Lower values indicate higher symmetry (perfect crystal), while higher
        values indicate defects, surfaces, or disordered regions.

    Notes
    -----
    The CSP is calculated as:

    .. math::

        CSP = \\sum_{i=1}^{N/2} |\\mathbf{r}_i + \\mathbf{r}_{i+N/2}|^2

    where :math:`\\mathbf{r}_i` and :math:`\\mathbf{r}_{i+N/2}` are vectors to
    opposite nearest neighbors.

    For perfect centro-symmetric crystals, CSP ≈ 0. Typical CSP values:
    - Perfect FCC/BCC bulk: 0-0.1
    - Surfaces/interfaces: 1-10
    - Dislocations: 10-50
    - Amorphous regions: > 50

    References
    ----------
    .. [1] Kelchner, C. L., Plimpton, S. J., & Hamilton, J. C. (1998).
           Dislocation nucleation and defect structure during surface indentation.
           Physical Review B, 58(17), 11085.
    """

    def __init__(
        self, data: pl.DataFrame, box: Box, N: int, verlet_list: np.ndarray
    ) -> None:
        self.data = data
        self.box = box
        assert N % 2 == 0 and N > 0, f"N must be a positive even number: {N}."
        self.N = int(N)
        self.verlet_list = verlet_list

    def compute(self) -> None:
        """
        Compute the Centro-Symmetry Parameter for all atoms.

        This method calculates the CSP value for each atom and stores the result
        in the ``csp`` attribute.

        Notes
        -----
        After calling this method, the ``csp`` attribute will contain a float array
        of CSP values, one for each atom.
        """
        self.csp = np.zeros(self.data.shape[0])
        _csp.get_csp(
            self.data["x"].to_numpy(allow_copy=False),
            self.data["y"].to_numpy(allow_copy=False),
            self.data["z"].to_numpy(allow_copy=False),
            self.box.box,
            self.box.origin,
            self.box.boundary,
            self.verlet_list,
            self.N,
            self.csp,
        )


if __name__ == "__main__":
    pass
