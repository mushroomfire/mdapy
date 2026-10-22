# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Special quasirandom structures."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mdapy._core import build as _native
from mdapy.build._alloy import random_alloy
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList


def _shells(system: System, shells: int) -> NeighborList:
    """The pairs of the first shells of neighbors of a perfect lattice, with
    the shell of each pair in the order of distance."""
    nearest = NeighborList.build(system, num_neighbors=1).distances.min()
    tolerance = 1e-3 * nearest
    cutoff = 1.2 * nearest
    while True:
        nl = NeighborList.build(system, cutoff)
        distinct = np.unique(np.round(nl.distances / tolerance)) * tolerance
        gaps = np.flatnonzero(np.diff(distinct) > tolerance)
        if len(gaps) >= shells:
            break
        cutoff *= 1.3
    radii = np.concatenate([[distinct[0]], distinct[gaps + 1]])[:shells]
    edge = 0.5 * (radii[-1] + distinct[distinct > radii[-1] + tolerance][0])
    return nl.within(edge)


def _shell_index(nl: NeighborList, tolerance: float) -> np.ndarray:
    distinct = np.sort(nl.distances)
    starts = np.concatenate(
        [[distinct[0]], distinct[1:][np.diff(distinct) > tolerance]]
    )
    return (np.searchsorted(starts, nl.distances + tolerance) - 1).astype(np.int8)


def sqs_alloy(
    system: System,
    fractions: Mapping[str, float],
    *,
    shells: int = 3,
    weights: Sequence[float] | None = None,
    steps: int = 100_000,
    temperature: float = 1.0,
    seed: int | np.random.Generator | None = None,
    progress: Any = None,
    cancel: Any = None,
) -> System:
    """A special quasirandom structure: elements in given fractions on the
    sites of a perfect lattice, arranged so that the pairs of the nearest
    shells of neighbors are those of a random alloy.

    Parameters
    ----------
    system : System
        The sites, a perfect lattice such as from :func:`build_crystal`.
    fractions : mapping of str to float
        Fraction of each element, positive; normalized to sum to 1, with the
        counts of :func:`random_alloy`.
    shells : int, optional
        The number of shells of neighbors made random, 1 to 10. Default 3.
    weights : sequence of float, optional
        The weight of each shell in the deviation minimized. Default all 1.
    steps : int, optional
        Number of swaps of two atoms tried. Default 100000.
    temperature : float, optional
        The starting temperature of the annealing, in units of the typical
        change of a swap; 0 accepts only improvements. Default 1.
    seed : int or numpy.random.Generator, optional
        Seed of the starting alloy and of the swaps.

    Returns
    -------
    System
        The system with a new ``element`` column.
    """
    if not 1 <= shells <= 10:
        raise InvalidParameter(f"shells must be from 1 to 10, not {shells!r}")
    w = np.ones(shells) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (shells,) or (w < 0).any():
        raise InvalidParameter(f"weights must be {shells} non-negative numbers")
    if isinstance(steps, bool) or not isinstance(steps, int | np.integer) or steps < 0:
        raise InvalidParameter(f"steps must be a non-negative integer, not {steps!r}")
    if not temperature >= 0:
        raise InvalidParameter(f"temperature must not be negative, not {temperature!r}")
    rng = np.random.default_rng(seed)
    start = random_alloy(system, fractions, seed=rng)
    names = list(fractions)
    kinds = start.table["element"].to_list()
    type_ = np.array([names.index(e) for e in kinds], dtype=np.int32)
    nl = _shells(system, shells)
    shell = _shell_index(nl, 1e-3 * nl.distances.min())
    best, _, _, _ = _native.sqs(
        nl.offsets,
        nl.j,
        shell,
        shells,
        type_,
        len(names),
        np.ascontiguousarray(w),
        int(steps),
        float(temperature),
        int(rng.integers(2**63)),
        progress,
        _native_token(cancel),
    )
    return system.with_columns(element=[names[t] for t in best])
