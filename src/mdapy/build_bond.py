# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from mdapy import _build_bond
from mdapy.parallel import get_num_threads
import numpy as np


def build_bond(
    verlet_list: np.ndarray,
    distance_list: np.ndarray,
    neighbor_number: np.ndarray,
    type_list: np.ndarray,
    cutoff_matrix: np.ndarray,
) -> np.ndarray:
    """Build bond pairs from precomputed neighbor information."""
    return _build_bond.build_bond(
        verlet_list,
        distance_list,
        neighbor_number,
        type_list,
        cutoff_matrix,
        get_num_threads(),
    )
