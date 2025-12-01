# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
from mdapy import _cluster
from typing import Optional, Union, Dict


class ClusterAnalysis:
    """
    Perform cluster analysis based on atomic connectivity within a cutoff distance.

    This class identifies connected groups of atoms ("clusters") by examining
    neighbor relationships. Two atoms belong to the same cluster if their
    interatomic distance is smaller than a given cutoff value.

    The cutoff can be either:
      - a single scalar value (applied to all atom pairs), or
      - a dictionary defining different cutoffs between atom types,
        e.g. ``{"1-1": 1.2, "1-2": 1.5, "2-2": 1.8}``.

    Parameters
    ----------
    rc : float, int, or dict of str to float
        The cutoff distance for bonding. If a dict is provided, the keys should
        be in the form ``"type1-type2"`` (e.g. ``"1-2"``), and ``type_list``
        must be given.
    verlet_list : np.ndarray
        The neighbor list array, where each row contains neighbor indices for
        each atom.
    distance_list : np.ndarray
        The corresponding neighbor distances for each atom.
    neighbor_number : np.ndarray
        The number of neighbors for each atom.
    type_list : np.ndarray, optional
        Atom type array. Required if `rc` is a dict specifying per-type cutoffs.

    Attributes
    ----------
    max_rc : float
        The maximum cutoff distance among all pair types.
    particleClusters : np.ndarray
        The cluster ID assigned to each atom after computation.
    cluster_number : int
        The total number of clusters identified.
    """

    def __init__(
        self,
        rc: Union[float, int, Dict[str, float]],
        verlet_list: np.ndarray,
        distance_list: np.ndarray,
        neighbor_number: np.ndarray,
        type_list: Optional[np.ndarray] = None,
    ):
        self.rc = rc
        if isinstance(rc, float) or isinstance(rc, int):
            self.max_rc = self.rc
        elif isinstance(rc, dict):
            assert type_list is not None, "Need type_list for multi cutoff mode."
            self.max_rc = max([i for i in self.rc.values()])
        else:
            raise TypeError(
                "rc should be a positive number, or a dict like {'1-1':1.5, '1-2':1.3}"
            )
        if isinstance(rc, Dict):
            self.verlet_list = verlet_list.copy()
        else:
            self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        self.type_list = type_list

    def _filter_verlet(self):
        """Filter the neighbor list according to type-dependent cutoff distances."""
        type1, type2, r = [], [], []
        for key, value in self.rc.items():
            left, right = key.split("-")
            type1.append(left)
            type2.append(right)
            r.append(value)
            if left != right:
                type1.append(right)
                type2.append(left)
                r.append(value)
        type1 = np.array(type1, np.int32)
        type2 = np.array(type2, np.int32)
        r = np.array(r, float)
        _cluster.filter_by_type(
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.type_list,
            type1,
            type2,
            r,
        )

    def compute(self):
        """
        Perform the actual cluster analysis.

        This function assigns a unique cluster ID to each atom
        based on connectivity within the given cutoff distance.

        Returns
        -------
        None
            The results are stored in the attributes:

            - ``particleClusters``: array of cluster IDs.
            - ``cluster_number``: total number of clusters found.
        """
        if isinstance(self.rc, dict):
            self._filter_verlet()

        N = self.verlet_list.shape[0]
        self.particleClusters = np.full(N, -1, dtype=np.int32)
        if isinstance(self.rc, dict):
            self.cluster_number = _cluster.get_cluster_by_bond(
                self.verlet_list, self.neighbor_number, self.particleClusters
            )
        else:
            self.cluster_number = _cluster.get_cluster(
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
                self.max_rc,
                self.particleClusters,
            )
