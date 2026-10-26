# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Random alloys."""

from collections.abc import Mapping

import numpy as np

from mdapy.core.system import System
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter


def random_alloy(
    system: System,
    fractions: Mapping[str, float],
    *,
    seed: int | np.random.Generator | None = None,
) -> System:
    """The system with its elements drawn at random in given fractions.

    Parameters
    ----------
    system : System
        The sites, for example a crystal from :func:`build_crystal`.
    fractions : mapping of str to float
        Fraction of each element, positive; they are normalized to sum to 1.
    seed : int or numpy.random.Generator, optional
        Seed of the random order; the same seed gives the same alloy.

    Returns
    -------
    System
        The system with a new ``element`` column. The number of atoms of each
        element is its fraction of the atoms, rounded so that the counts add
        up, the largest remainders rounded up.
    """
    if not fractions:
        raise InvalidParameter("fractions must name at least one element")
    for element, fraction in fractions.items():
        if element not in SYMBOLS[1:]:
            raise InvalidParameter(f"unknown element {element!r} in fractions")
        if not fraction > 0:
            raise InvalidParameter(f"the fraction of {element} must be positive")
    names = list(fractions)
    share = np.array([fractions[e] for e in names], dtype=float)
    share /= share.sum()
    n = len(system)
    exact = share * n
    counts = np.floor(exact).astype(int)
    # the largest remainders get the atoms left over, ties to the first named
    left = n - counts.sum()
    counts[np.argsort(-(exact - counts), kind="stable")[:left]] += 1
    elements = np.repeat(np.array(names, dtype=object), counts)
    np.random.default_rng(seed).shuffle(elements)
    return system.with_columns(element=elements.tolist())
