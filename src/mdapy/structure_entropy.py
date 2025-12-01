# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
This module implements the calculation of local structural entropy
based on the method proposed by Piaggi and Parrinello
(`J. Chem. Phys. 147, 114112 (2017)`), which quantifies the degree
of local ordering in atomic configurations.

The local structural entropy measures how similar the local
radial distribution around each atom is to the average distribution
of its surroundings. It is a useful descriptor for identifying
crystalline and disordered regions, phase transitions, or defects.

The implementation supports both global and density-weighted
(normalized) entropy evaluation, and an optional neighbor-based
averaging of the entropy field.

References
----------
.. [1] P.M. Piaggi and M. Parrinello,Entropy based fingerprint for local crystalline order,
        J. Chem. Phys. 147, 114112 (2017)
        https://doi.org/10.1063/1.4998408
"""

from mdapy import _structure_entropy
from mdapy import _neighbor
from mdapy.box import Box
import numpy as np


class StructureEntropy:
    """
    Calculate the local structural entropy for each atom based on
    its neighbor environment within a specified cutoff radius.

    The local structural entropy quantifies the degree of
    local disorder by comparing the local pair distribution
    to a reference distribution. Lower entropy values correspond
    to more ordered (crystalline) environments, while higher values
    indicate disordered or amorphous regions.

    Parameters
    ----------
    box : Box
        Simulation box object defining the system boundaries.
    verlet_list : np.ndarray
        Neighbor list of atomic indices. Each row corresponds to
        one atom and contains indices of its neighbors.
    distance_list : np.ndarray
        Distance list corresponding to the `verlet_list`. Each
        element stores the pairwise distances between atoms.
    neighbor_number : np.ndarray
        Number of valid neighbors for each atom.
    rc : float
        Cutoff radius (in Å) used for calculating the local
        pair distribution function.
    sigma : float
        Gaussian smoothing width used in constructing the local
        radial distribution function.
    use_local_density : bool
        If True, the local RDF is normalized by the local atomic
        density. If False, a global normalization is used.
    average_rc : float, optional
        Cutoff radius for averaging entropy values over neighboring
        atoms. If zero or not specified, no averaging is performed.

    Attributes
    ----------
    entropy : np.ndarray
        Array of local structural entropy values for all atoms.
    entropy_ave : np.ndarray, optional
        Averaged entropy values over neighbors within `average_rc`.
        Only computed if `average_rc > 0`.

    Notes
    -----
    - The algorithm internally uses a precomputed neighbor list
      (`verlet_list`, `distance_list`, and `neighbor_number`).
    - The entropy field can be spatially averaged to reduce
      statistical noise or highlight larger-scale structural
      features.
    """

    def __init__(
        self,
        box: Box,
        verlet_list: np.ndarray,
        distance_list: np.ndarray,
        neighbor_number: np.ndarray,
        rc: float,
        sigma: float,
        use_local_density: bool,
        average_rc: float = 0.0,
    ):
        self.box = box
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        self.rc = rc
        self.sigma = sigma
        self.use_local_density = use_local_density
        self.average_rc = average_rc

    def compute(self):
        """
        Perform the local structural entropy calculation.

        This method computes the entropy value for each atom based on
        its local pair distribution function within `rc`. Optionally,
        if `average_rc > 0`, the resulting entropy field is averaged
        over neighbors within the given radius.

        Returns
        -------
        None
            Results are stored in the instance attributes:
            ``entropy`` and, if applicable, ``entropy_ave``.
        """
        self.entropy = np.zeros(self.verlet_list.shape[0])
        _structure_entropy.calculate_structure_entropy(
            self.rc,
            self.sigma,
            self.use_local_density,
            self.box.volume,
            self.distance_list,
            self.neighbor_number,
            self.entropy,
        )

        if self.average_rc > 0:
            assert self.average_rc <= self.rc, "average_rc should be smaller than rc."
            self.entropy_ave = np.zeros_like(self.entropy)
            _neighbor.average_by_neighbor(
                self.average_rc,
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
                self.entropy,
                self.entropy_ave,
                True,
            )
