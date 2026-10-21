# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The pair entropy fingerprint of local order."""

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


@dataclass(frozen=True)
class EntropyResult(Result):
    """The pair entropy fingerprint of each atom.

    Attributes
    ----------
    entropy : numpy.ndarray
        (N,) entropy in units of the Boltzmann constant; more negative is
        more ordered.
    average : bool
        Whether it is averaged over neighbors.
    """

    entropy: np.ndarray
    average: bool

    def columns(self) -> dict[str, np.ndarray]:
        return {"entropy_average" if self.average else "entropy": self.entropy}


class EntropyFingerprint:
    """The pair entropy fingerprint of Piaggi and Parrinello.

    Parameters
    ----------
    cutoff : float
        Largest distance of the local radial distribution, in Å.
    sigma : float, optional
        Width of the Gaussian that smears each neighbor, in Å, and of the
        bins. Default 0.2.
    local_density : bool, optional
        Use the density of the neighbors within cutoff of each atom instead
        of that of the system. Default False.
    average_cutoff : float, optional
        Average each atom's value with those of its neighbors closer than
        this, in Å.
    """

    def __init__(
        self,
        cutoff: float,
        sigma: float = 0.2,
        *,
        local_density: bool = False,
        average_cutoff: float | None = None,
    ):
        if not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        if not 0 < sigma < cutoff:
            raise InvalidParameter(
                f"sigma must be positive and below cutoff, not {sigma!r}"
            )
        if average_cutoff is not None and not average_cutoff > 0:
            raise InvalidParameter(
                f"average_cutoff must be positive, not {average_cutoff!r}"
            )
        self.cutoff = float(cutoff)
        self.sigma = float(sigma)
        self.local_density = bool(local_density)
        self.average_cutoff = None if average_cutoff is None else float(average_cutoff)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> EntropyResult:
        """The fingerprint of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        neighbors : NeighborList, optional
            A neighbor list of this system reaching at least ``cutoff`` and
            ``average_cutoff``.

        Returns
        -------
        EntropyResult
            The fingerprint of each atom.
        """
        reach = max(self.cutoff, self.average_cutoff or 0.0)
        nl = _given_or_built(system, reach, neighbors)
        local = nl.within(self.cutoff) if nl.cutoff > self.cutoff else nl
        n = len(system)
        if self.local_density:
            density = local.counts / (4.0 / 3.0 * pi * self.cutoff**3)
        else:
            density = np.full(n, n / system.box.volume)
        entropy = _native.pair_entropy(
            local.offsets,
            np.ascontiguousarray(local.distances),
            self.cutoff,
            self.sigma,
            np.ascontiguousarray(density, dtype=np.float64),
            progress,
            _native_token(cancel),
        )
        if self.average_cutoff is not None:
            near = (
                nl.within(self.average_cutoff)
                if nl.cutoff > self.average_cutoff
                else nl
            )
            total = entropy + np.bincount(near.i, weights=entropy[near.j], minlength=n)
            entropy = total / (1 + near.counts)
        return EntropyResult(entropy, self.average_cutoff is not None)

    def __repr__(self) -> str:
        return (
            f"EntropyFingerprint(cutoff={self.cutoff!r}, sigma={self.sigma!r}, "
            f"local_density={self.local_density}, average_cutoff={self.average_cutoff!r})"
        )
