# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""How much of descriptor space a set of atomic environments covers: the
descriptor of QUESTS, the entropy of a set of descriptors and the information
that new ones add to it."""

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdapy._core import analysis as _native
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built

# the bandwidth of QUESTS for its descriptor, in 1/Å
_BANDWIDTH = 0.015


class QUESTS:
    """The descriptor of QUESTS of each atom: the distances to its nearest
    neighbors and between them, each weighted to vanish at the cutoff,
    whatever their elements.

    Parameters
    ----------
    cutoff : float64, optional
        The cutoff in Å, above zero. Default 5.
    num_neighbors : int32, optional
        The number of nearest neighbors, at least 1. Default 32.
    """

    def __init__(self, cutoff: float = 5.0, num_neighbors: int = 32):
        if isinstance(cutoff, bool) or not (
            isinstance(cutoff, int | float | np.number) and 0 < cutoff < math.inf
        ):
            raise InvalidParameter(f"cutoff must be above 0 Å, not {cutoff!r}")
        if (
            isinstance(num_neighbors, bool)
            or not isinstance(num_neighbors, int | np.integer)
            or num_neighbors < 1
        ):
            raise InvalidParameter(
                f"num_neighbors must be an integer of at least 1, not {num_neighbors!r}"
            )
        self.cutoff = float(cutoff)
        self.num_neighbors = int(num_neighbors)

    @property
    def dimension(self) -> int:
        """The number of values per atom."""
        return 2 * self.num_neighbors - 1

    def descriptors(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> np.ndarray:
        """The descriptor of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        neighbors : NeighborList, optional
            A neighbor list of this system reaching at least ``cutoff``.

        Returns
        -------
        numpy.ndarray
            (N, dimension) float64: the weights of the nearest neighbors
            over their distances, nearest first, then the mean of the
            sorted cross terms of the neighbors, largest first; zeros for
            the neighbors an atom lacks within the cutoff.
        """
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            keep = neighbors.distances < self.cutoff
            owner = np.repeat(np.arange(len(system)), np.diff(neighbors.offsets))
            counts = np.bincount(owner[keep], minlength=len(system))
            offsets = np.concatenate([[0], np.cumsum(counts)])
            return _native.quests(
                self.num_neighbors,
                self.cutoff,
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix),
                offsets.astype(np.int64),
                np.ascontiguousarray(neighbors.j[keep]),
                np.ascontiguousarray(neighbors.shifts[keep]),
                progress,
                _native_token(cancel),
            )

    def __repr__(self) -> str:
        return f"QUESTS(cutoff={self.cutoff!r}, num_neighbors={self.num_neighbors})"


def _points(values: ArrayLike, name: str) -> np.ndarray:
    x = np.ascontiguousarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise InvalidParameter(f"{name} must be (M, D) with at least one point")
    if not np.isfinite(x).all():
        raise InvalidParameter(f"{name} must be finite")
    return x


def _bandwidth(bandwidth: float) -> float:
    if isinstance(bandwidth, bool) or not (
        isinstance(bandwidth, int | float | np.number) and 0 < bandwidth < math.inf
    ):
        raise InvalidParameter(f"bandwidth must be above zero, not {bandwidth!r}")
    return float(bandwidth)


def entropy(
    points: ArrayLike,
    bandwidth: float = _BANDWIDTH,
    *,
    progress: Any = None,
    cancel: Any = None,
) -> float:
    """The entropy of a set of points, such as the descriptors of the atoms of
    a training set, by a Gaussian density around each point.

    Parameters
    ----------
    points : array_like
        (N, D) float64, the points.
    bandwidth : float64, optional
        The width of the Gaussians, in the units of the points. Default 0.015,
        that of QUESTS for its descriptor.

    Returns
    -------
    float64
        The entropy in nats: zero when the points are all the same, the
        logarithm of their number when no two are close.
    """
    x = _points(points, "points")
    sums = _native.log_gaussian_sums(
        x, x, _bandwidth(bandwidth), progress, _native_token(cancel)
    )
    return float(math.log(len(x)) - sums.mean())


def information_gain(
    points: ArrayLike,
    reference: ArrayLike,
    bandwidth: float = _BANDWIDTH,
    *,
    progress: Any = None,
    cancel: Any = None,
) -> np.ndarray:
    """What each point adds to a set of reference points, such as the
    descriptors of the atoms of new structures to those of a training set:
    the differential entropy of QUESTS.

    Parameters
    ----------
    points : array_like
        (M, D) float64, the new points.
    reference : array_like
        (K, D) float64, the reference points.
    bandwidth : float64, optional
        The width of the Gaussians, in the units of the points. Default 0.015,
        that of QUESTS for its descriptor.

    Returns
    -------
    numpy.ndarray
        (M,) float64, in nats: negative for a point among reference points,
        positive for one away from them, growing with the square of its
        distance from the nearest.
    """
    x = _points(points, "points")
    y = _points(reference, "reference")
    if x.shape[1] != y.shape[1]:
        raise InvalidParameter(
            f"points have {x.shape[1]} values each and reference {y.shape[1]}"
        )
    return -_native.log_gaussian_sums(
        x, y, _bandwidth(bandwidth), progress, _native_token(cancel)
    )
